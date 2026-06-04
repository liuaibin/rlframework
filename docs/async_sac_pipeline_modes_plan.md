# Async SAC Pipeline Modes Plan

## 1. 背景和目标

当前 `rlframework/algorithms/async_sac.py` 的异步链路主要面向 Ray 2.54：

- EnvRunner 采样通过 `foreach_env_runner_async_fetch_ready()` 做非阻塞 fetch/reissue；
- Learner 训练通过 `learner_group.update(async_update=True)` 做非阻塞 update；
- Ray 2.49 没有 Ray 2.54 的 async EnvRunner API，因此当前会退回父类同步训练链路。

现在需要把“Env 采样异步”和“Learner 训练异步”拆成两个可独立配置的维度。这样测试时可以先验证：

1. Env 异步采样 + Learner 同步训练；
2. Env 异步采样 + Learner 异步训练。

第一阶段不要引入 ReplayBuffer actor 化、PER 异步 priority 回写、WeightPublisher 等更大架构改造。先保持 driver-local replay buffer，降低改动范围和验证成本。

## 2. Ray 2.54 参考点

参考 Ray 2.54 相关实现：

- `EnvRunner.sample_get_state_and_metrics()`：一次远程调用里执行 `sample()`、读取 connector state、读取 metrics，并将 episodes 通过 `ray.put()` 以 ObjectRef 返回。
- `EnvRunnerGroup.foreach_env_runner_async_fetch_ready()`：先 fetch ready 的旧异步结果，再发起新的 async request。
- `LearnerGroup.update(async_update=True)`：提交当前异步 learner update，同时返回之前 ready 的 learner results；本地 learner，即 `num_learners=0`，不支持 `async_update=True`。
- Ray 2.54 DQN/SAC new API 默认 off-policy 路径仍是同步链路：`synchronous_parallel_sample -> replay add/sample -> learner_group.update -> priority update -> weight sync`。

源码参考：

- <https://github.com/ray-project/ray/blob/ray-2.54.0/rllib/env/env_runner.py>
- <https://github.com/ray-project/ray/blob/ray-2.54.0/rllib/env/env_runner_group.py>
- <https://github.com/ray-project/ray/blob/ray-2.54.0/rllib/core/learner/learner_group.py>
- <https://github.com/ray-project/ray/blob/ray-2.54.0/rllib/algorithms/dqn/dqn.py>

## 3. 总体方案

将当前“Ray 2.54 下同时启用 async env sampling + async learner training”的隐式行为拆成两个显式配置：

```python
config.algorithm_options(
    {
        "env_sampling": "async",
        "learner_training": "sync",
    }
)
```

建议配置取值使用字符串，而不是单纯 bool：

- `env_sampling`: `"auto" | "sync" | "async"`
- `learner_training`: `"auto" | "sync" | "async"`

原因：

- `auto` 可以保持现有默认行为；
- 显式 `"async"` 可以在 Ray 版本/API 不支持时 fail fast；
- Ray 2.49 和 Ray 2.54 的兼容语义更清楚；
- 后续如果加 `"threaded"`、`"actor"`、`"remote"` 等模式，也更容易扩展。

默认兼容策略：

- Ray 2.49：`auto` 退回同步 `CustomSAC`/RLlib 父类路径；
- Ray 2.54：`auto` 默认等价于当前行为，即 Env 异步采样 + Learner 异步训练；
- 用户显式指定 `"async"` 但运行时不支持时，直接报清晰错误，不静默 fallback。

## 4. 模式矩阵

| Env 采样 | Learner 训练 | Ray 2.49 | Ray 2.54 | 用途 |
|---|---:|---:|---:|---|
| sync | sync | 支持 | 支持 | 原始 SAC baseline |
| async | sync | 不支持，显式报错 | 支持 | 第一阶段：只验证异步采样收益 |
| async | async | 不支持，显式报错 | 支持 | 第二阶段：当前 Async SAC 目标形态 |
| sync | async | 暂不支持 | 暂不支持/后续再看 | 价值较小，先不作为主路径 |

第一版建议只正式支持三条路径：

1. `sync env + sync learner`：完全走父类同步链路；
2. `async env + sync learner`：新加的阶段一链路；
3. `async env + async learner`：当前 `async_sac.py` 的主链路，拆成可配置。

## 5. 路径一：Env 异步采样 + Learner 同步训练

目标：先消除 `synchronous_parallel_sample()` 的 straggler barrier，但保持 replay 和 learner 语义简单。

流程：

```text
training_step()
  -> fetch ready EnvRunner samples
  -> immediately reissue next async sample request
  -> add ready episodes to local replay buffer
  -> aggregate EnvRunner metrics
  -> update train credit
  -> if warmup passed and train credit >= 1:
       replay_buffer.sample()
       learner_group.update(async_update=False, return_state=True)
       process learner results
       sync weights + connector states to EnvRunners
       train credit -= 1
```

关键设计：

- 先 reissue sample，再执行同步 learner update。
  - learner 同步训练期间，remote EnvRunner 已经开始下一轮采样；
  - 这样能实现采样和训练的部分重叠。
- Learner 使用 `async_update=False`。
  - 这条路径不要求 `num_learners > 0`；
  - 本地 learner 和 remote learner 都可以同步 update。
- 每个 `training_step` 默认最多执行一次 learner update。
  - 防止 driver 长时间卡在连续同步训练中；
  - 保持 driver 能高频 fetch ready sample。
- 继续使用 train credit。
  - 只有新样本进入 replay 后才增加训练额度；
  - credit 不足时不训练，避免在采样停滞时反复训练旧 replay。

适合测试的问题：

- 异步 EnvRunner 是否能绕开慢 EnvRunner；
- learner 同步训练是否还能和 EnvRunner 采样产生重叠；
- replay add/sample 是否成为新的 driver-local 瓶颈；
- 对 reward 曲线和 throughput 的影响是否稳定。

## 6. 路径二：Env 异步采样 + Learner 异步训练

目标：在路径一基础上，把 learner update 也做成有界异步，让 learner/GPU 计算和 driver 调度进一步重叠。

流程：

```text
training_step()
  -> fetch ready EnvRunner samples
  -> immediately reissue next async sample request
  -> add ready episodes to local replay buffer
  -> aggregate EnvRunner metrics

  -> drain ready learner results
       process learner metrics
       sync latest weights + connector states

  -> update train credit
  -> if warmup passed and train credit >= 1 and learner in-flight not full:
       replay_buffer.sample()
       learner_group.update(async_update=True, return_state=True)
       process old ready learner results returned by this call
       train credit -= 1
```

关键设计：

- `async_update=True` 必须要求 `num_learners > 0`。
  - Ray 的本地 learner 不支持 async update；
  - 显式配置 learner async 且 `num_learners=0` 时应在 validate 阶段报错。
- 必须用 learner in-flight 限流。
  - 参考 `max_requests_in_flight_per_learner`；
  - outstanding 已满时只 drain，不提交新 update。
- 要明确异步 learner result 语义。
  - `learner_group.update(async_update=True)` 返回的是之前 ready 的结果，不是当前 batch 的结果；
  - 权重同步天然会滞后一轮或多轮。
- 每个 `training_step` 默认最多提交一个 learner update。
  - 降低 object store 和 actor manager 队列压力；
  - 便于和同步 learner 路径做 A/B 对比。

## 7. Runtime 模式解析与校验

建议在 `AsyncCustomSACConfig` 中保存用户配置，在 `validate()` 和 `AsyncCustomSAC.__init__()` 中分别处理静态校验和运行时能力解析。

运行时能力：

```text
supports_async_env_sampling =
    hasattr(env_runner_group, "foreach_env_runner_async_fetch_ready")

supports_async_learner_training =
    learner_group supports update(async_update=True)
    and config.num_learners > 0
```

最终模式解析规则：

```text
if env_sampling == "auto":
    use_async_env_sampling = supports_async_env_sampling
elif env_sampling == "async":
    require supports_async_env_sampling
else:
    use_async_env_sampling = False

if learner_training == "auto":
    use_async_learner_training = (
        use_async_env_sampling
        and supports_async_learner_training
    )
elif learner_training == "async":
    require use_async_env_sampling
    require supports_async_learner_training
else:
    use_async_learner_training = False
```

校验建议：

- `env_sampling="async"` 但没有 Ray 2.54 async EnvRunner API：报错；
- `learner_training="async"` 但 `num_learners <= 0`：报错；
- `learner_training="async"` 但 `env_sampling="sync"`：第一版报错；
- async SAC 仍只支持非 prioritized episode replay buffer；
- `env_sampling="sync"` 时不要强制 `num_learners > 0`。

## 8. `async_sac.py` 结构调整建议

当前 `AsyncCustomSAC` 可以拆成更清楚的几个子步骤。

建议新增/调整字段：

```text
self._use_async_env_sampling
self._use_async_learner_training
self._train_credit
self._latest_connector_states_by_actor
```

建议主入口：

```text
training_step()
  -> if not self._use_async_env_sampling:
       return super().training_step()
  -> self.on_before_training_step()
  -> self._training_step_async_env()
  -> self.on_after_training_step(...)
```

建议拆分 helper：

```text
_training_step_async_env()
_fetch_ready_samples_and_reissue()
_maybe_train_sync_learner()
_maybe_issue_async_learner_update()
_drain_learner_results()
_sample_from_replay_buffer()
_process_learner_results()
_calc_credit_increment()
_current_sampled_timesteps()
```

`_training_step_async_env()` 内部根据 learner 模式分支：

```text
new_env_steps = _fetch_ready_samples_and_reissue()

if self._use_async_learner_training:
    _drain_learner_results()
    update train credit
    _maybe_issue_async_learner_update()
else:
    update train credit
    _maybe_train_sync_learner()
```

## 9. 权重同步和 Connector State

旧实现使用 `_pending_connector_states` 保存本轮采样产生的 connector states。当前实现改为按 actor 保存最近一次 connector state：

```text
self._latest_connector_states_by_actor[actor_id] = connector_state
```

原因：

- async learner result ready 的时刻不一定刚好有新的 EnvRunner sample ready；
- 如果本轮没有 ready sample，也应该能使用最近一次 connector state 进行 state sync；
- 对 straggler 场景更稳。

同步策略：

- sync learner path：每次同步 learner update 返回新权重后，立即 sync EnvRunner states；
- async learner path：drain 到 learner results 且带 `_rl_module_state_after_update` 时，sync EnvRunner states；
- connector states 使用 latest snapshot；
- `rl_module_state` 仍通过 Ray/RLlib 的 env runner state sync 机制下发。

## 10. Metrics 建议

为了验证模式是否生效，建议补充以下指标。

Env 采样：

- `async_sac_env_ready_results`
- `async_sac_env_steps_added`
- `async_sac_new_env_steps`
- `async_sac_env_sample_inflight`
- `async_sac_env_sample_inflight_after_reissue`
- `async_sac_remote_env_runners_healthy`

Learner：

- `async_sac_learner_inflight`
- `async_sac_learner_inflight_before_issue`
- `async_sac_learner_inflight_after_issue`
- `async_sac_learner_results_drained`
- `async_sac_learner_results_processed`
- `async_sac_async_learner_updates_issued`
- `async_sac_sync_learner_updates`
- `async_sac_learner_blocked_by_inflight`

训练节奏：

- `async_sac_pipeline_step`
- `async_sac_train_credit`
- `async_sac_train_credit_added`
- `async_sac_train_credit_spent`
- `async_sac_train_credit_blocked`
- `async_sac_warmup_blocked`
- `async_sac_current_sampled_timesteps`
- `async_sac_replay_sampled_episodes`
- `async_sac_weight_syncs`
- `async_sac_connector_states_synced`

可选日志：

- `algorithm_options({"pipeline_log_interval": N})`：每 N 个 async training step 打一条 driver INFO 日志；
- 默认 `0`，即关闭周期日志，避免训练时刷屏。

整体观察：

- env steps/sec；
- learner updates/sec；
- replay add/sample time；
- training step wall time；
- weight sync count；
- policy lag 或权重版本滞后。

## 11. 测试计划

单元测试：

- Ray 2.49 下 `auto` fallback 不破坏现有 `AsyncCustomSACConfig` 测试；
- 显式 `env_sampling="async"` 但没有 async EnvRunner API 时，报清晰错误；
- 显式 `learner_training="async"` 且 `num_learners=0` 时，报清晰错误；
- `env_sampling="async", learner_training="sync"` 不要求 `num_learners > 0`；
- `env_sampling="sync", learner_training="sync"` 走父类同步路径；
- `_dont_auto_sync_env_runner_states` 只在 async env path 开启。

小环境功能测试：

- Pendulum：`async env + sync learner`；
- Pendulum：`async env + async learner`；
- 检查 reward、metrics、checkpoint、evaluation 是否正常；
- 检查 learner async outstanding requests 是否能正常 drain。

Straggler 测试：

- 两个 EnvRunner，其中一个人为 sleep；
- `async env + sync learner` 应该能先消费快 EnvRunner ready samples；
- 对比同步 baseline 的 sample barrier 时间。

Benchmark 对比：

1. `sync env + sync learner` baseline；
2. `async env + sync learner`；
3. `async env + async learner`。

对比指标：

- training iteration wall time；
- env steps/sec；
- replay add/sample 占比；
- learner update 占比；
- driver CPU；
- learner/GPU 利用率；
- reward 曲线是否有明显回退。

## 12. 风险与边界

主要风险：

- replay buffer 仍在 driver 本地，EnvRunner 数量上来后 replay add/sample 可能成为新瓶颈；
- sync learner path 如果 learner update 很慢，driver 仍会在 update 期间阻塞，只是 EnvRunner 已经提前 reissue sample；
- async learner path 需要严格控制 in-flight，否则 batch stale 和 object store 压力会上升；
- 权重同步滞后会带来 policy lag，需要 metrics 观察；
- prioritized replay 暂不支持，因为 async priority update 需要 sample handle、generation、eviction 一致性设计。

第一版边界：

- 不做 ReplayBuffer actor 化；
- 不做 PER async priority update；
- 不做 Learner 自驱动 train loop；
- 不做 WeightPublisher；
- 不改变 SAC loss、module、optimizer 语义；
- 只拆 pipeline 调度模式。

## 13. 落地顺序

建议按以下顺序推进：

1. 新增 `algorithm_options({"env_sampling": ..., "learner_training": ...})` 配置，但默认行为保持不变；
2. 拆出 runtime 模式解析和 validate 规则；
3. 实现 `async env + sync learner` 路径；
4. 将当前 async learner 逻辑挂到 `async env + async learner` 分支；
5. 把 connector state 从单轮 pending list 改成 latest-by-actor；
6. 补充关键 metrics；
7. 增加配置校验测试和 Pendulum smoke test；
8. 做三组 benchmark：sync baseline、async env + sync learner、async env + async learner；
9. 根据 metrics 再判断是否进入 ReplayBuffer actor 化和 Learner 自驱动训练循环。

## 14. 推荐第一版用户配置

第一阶段：只验证 Env 异步采样。

```python
config = (
    AsyncCustomSACConfig()
    .environment("Pendulum-v1")
    .training(
        replay_buffer_config={"type": "EpisodeReplayBuffer", "capacity": 100_000},
        train_batch_size_per_learner=256,
        num_steps_sampled_before_learning_starts=1500,
    )
    .env_runners(
        num_env_runners=2,
        num_envs_per_env_runner=1,
        rollout_fragment_length=16,
        max_requests_in_flight_per_env_runner=1,
    )
    .learners(
        num_learners=0,
        num_gpus_per_learner=0,
    )
    .algorithm_options(
        {
            "env_sampling": "async",
            "learner_training": "sync",
            "pipeline_log_interval": 100,
        }
    )
)
```

第二阶段：验证 Env 异步采样 + Learner 异步训练。

```python
config = (
    AsyncCustomSACConfig()
    .environment("Pendulum-v1")
    .training(
        replay_buffer_config={"type": "EpisodeReplayBuffer", "capacity": 100_000},
        train_batch_size_per_learner=256,
        num_steps_sampled_before_learning_starts=1500,
    )
    .env_runners(
        num_env_runners=2,
        num_envs_per_env_runner=1,
        rollout_fragment_length=16,
        max_requests_in_flight_per_env_runner=1,
    )
    .learners(
        num_learners=1,
        num_gpus_per_learner=0,
        max_requests_in_flight_per_learner=1,
    )
    .algorithm_options(
        {
            "env_sampling": "async",
            "learner_training": "async",
            "pipeline_log_interval": 100,
        }
    )
)
```
