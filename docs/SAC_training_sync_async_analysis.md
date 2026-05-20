# SAC 训练流程同步/异步性能分析

本文基于当前仓库中的 `examples/02_sac_pendulum.py` 和 `rlframework/algorithms/sac.py`，分析 SAC 底层复用 RLlib DQN 训练循环时的同步性能影响，以及是否适合改成异步流程。

## 1. 当前调用链

`examples/02_sac_pendulum.py` 使用的是 `CustomSACConfig`：

```python
config = (
    CustomSACConfig()
    .framework_run("pendulum", root_dir="./runs")
    .environment("Pendulum-v1")
    .training(
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        train_batch_size=256,
        replay_buffer_config={
            "type": "EpisodeReplayBuffer",
            "capacity": 100_000,
        },
        target_entropy="auto",
        tau=0.005,
    )
    .env_runners(num_env_runners=1, rollout_fragment_length=1)
    .evaluation(evaluation_interval=20)
    .checkpointing(freq=20)
    .metrics(reporters=metric_reporters, reporter_configs=reporter_configs)
    .storage(upload_async=True, best_upload_freq=10)
)
```

仓库中的 `CustomSAC.training_step()` 只是包了一层框架 hook：

```python
def training_step(self) -> None:
    self.on_before_training_step()
    super().training_step()
    metrics = self.metrics
    if metrics is None:
        return
    result = metrics.peek()
    result = self.on_after_training_step(result)
    ...
```

也就是说，真正的训练逻辑仍然来自 RLlib 的 `SAC`。本地依赖版本中，`SAC` 的继承链是：

```text
SAC -> DQN -> Algorithm
```

并且 `SAC.training_step` 实际使用的是 `DQN.training_step()`。新 API stack 下会进入：

```python
return self._training_step_new_api_stack()
```

核心流程可以概括为：

```text
采样 EnvRunner
  -> 写入 local replay buffer
  -> 从 replay buffer sample 训练 batch
  -> learner_group.update()
  -> 根据 TD-error 更新 replay priority
  -> 同步 learner 权重到 EnvRunner
```

## 2. 当前同步点

RLlib 新 API stack 的 DQN/SAC 训练流程主要有以下同步点。

### 2.1 同步采样

训练循环中调用：

```python
episodes, env_runner_results = synchronous_parallel_sample(
    worker_set=self.env_runner_group,
    concat=True,
    sample_timeout_s=self.config.sample_timeout_s,
    _uses_new_env_runners=True,
    _return_metrics=True,
)
```

`synchronous_parallel_sample()` 的语义是同步 rollout：

```text
Runs parallel and synchronous rollouts on all remote workers.
Waits for all workers to return from the remote calls.
```

也就是说：

- 多个 EnvRunner 并行采样，但 driver 会等待一批采样结果返回；
- 如果某个 worker 慢，会拖住这一轮训练；
- `sample_timeout_s` 可以避免无限等待，但不能消除同步 barrier。

### 2.2 同步 learner update

训练 batch 从 replay buffer 取出后，调用：

```python
learner_results = self.learner_group.update(
    episodes=episodes,
    timesteps={...},
)
```

默认是同步更新。更新期间，driver 等待 learner 计算完成后才继续处理 TD-error、更新优先级、同步权重。

RLlib 的 `LearnerGroup.update()` 支持 `async_update=True`，但有一个重要限制：

```text
Can't call update(async_update=True) when running with num_learners=0.
Set config.num_learners > 0 to allow async updates.
```

因此如果当前是本地 learner，不能直接打开异步 learner update。

### 2.3 权重同步

训练完成后会执行：

```python
self.env_runner_group.sync_weights(
    from_worker_or_learner_group=self.learner_group,
    policies=modules_to_update,
    global_vars=None,
    inference_only=True,
)
```

新 API stack 下，`sync_weights()` 会从 learner 取 RLModule state，然后通过 Ray object store 同步给 EnvRunner。其远端设置权重的默认 `timeout_seconds=0.0`，偏 fire-and-forget，不一定会等待所有 worker set_state 完成。

但它仍然有成本：

- 从 learner 读取权重；
- 将权重放入 Ray object store；
- 向所有远端 EnvRunner 发起 `set_state`；
- 如果模型较大或 EnvRunner 很多，这部分会成为明显开销。

## 3. 对 `02_sac_pendulum.py` 的影响

当前示例配置是：

```python
.env_runners(num_env_runners=1, rollout_fragment_length=1)
```

这个配置对 toy 环境来说非常细粒度：

- `Pendulum-v1` 单步环境很轻；
- `rollout_fragment_length=1` 意味着每次只采 1 个 fragment；
- Ray 调用、metrics 聚合、replay add、训练循环调度的固定开销会被放大；
- 性能瓶颈可能不是 SAC 计算本身，而是频繁同步调用和框架调度开销。

因此，当前示例中“同步影响性能”是成立的，但更准确地说：

> 主要问题不是同步采样这个概念本身，而是同步点发生得太频繁，且每次处理的数据太小。

一个训练 iteration 的耗时可以粗略拆成：

```text
T_iter ≈ T_sample_barrier
       + T_replay_add
       + N * (T_replay_sample + T_learner_update + T_priority_update)
       + T_weight_sync
       + callback/checkpoint/evaluation
```

其中 `rollout_fragment_length=1` 会让 `T_sample_barrier` 和各种固定开销在整体耗时中占比变大。

## 4. 改异步是否会更好

结论：异步流程可能提升吞吐，但不建议直接把当前 SAC 改成完全异步。

### 4.1 异步的潜在收益

SAC 是 off-policy 算法，天然比 PPO 这类 on-policy 算法更适合异步采样。异步化可能带来：

- EnvRunner 在 learner 更新时继续采样，提高 CPU/GPU 重叠；
- 多 EnvRunner 时不用被最慢 worker 拖住；
- 对慢仿真环境、远端环境、多 worker 场景，吞吐提升可能明显；
- learner 较慢时，可以用 replay buffer 吸收采样和训练速率差。

### 4.2 异步的主要风险

异步化不是简单替换一个函数，至少有以下风险。

#### Policy lag

EnvRunner 使用的 actor 权重可能落后于 learner 最新权重。

SAC 可以接受一定程度的 stale data，但如果权重同步太慢或 in-flight 请求太多，数据分布会明显滞后，可能影响收敛速度和稳定性。

#### Prioritized replay 的 TD-error 对齐

当前训练流程中：

```text
replay_buffer.sample()
  -> learner update 返回 TD-error
  -> update_priorities_in_episode_replay_buffer()
```

这个流程默认 TD-error 和最近一次 sample 的数据是顺序对应的。

如果 learner update 也异步化，多个 replay batch 同时 in-flight，TD-error 乱序返回时就可能出现：

- TD-error 对不上对应 sample；
- priority 更新到错误样本；
- replay buffer 内部 `_last_sampled_indices` 被后续 sample 覆盖。

因此，全异步 learner update 需要额外设计 sample id / priority id 映射，不能直接复用当前优先级更新逻辑。

#### Metrics 和 checkpoint 顺序

异步采样、异步学习会让结果乱序返回，需要重新设计：

- EnvRunner metrics 聚合；
- Learner metrics 聚合；
- 训练步数统计；
- checkpoint/evaluation 触发时机；
- 权重同步频率。

否则指标会难以解释，checkpoint 保存的模型和已记录 metrics 也可能不严格对应。

## 5. 推荐优化顺序

建议不要第一步就改成完全异步 SAC，而是按风险从低到高逐步优化。

### 5.1 先通过 timers 定位瓶颈

优先查看这些 RLlib timer：

- `timers.env_runner_sampling_timer`
- `timers.replay_buffer_add_data_timer`
- `timers.replay_buffer_sampling_timer`
- `timers.learner_update_timer`
- `timers.replay_buffer_update_prios_timer`
- `timers.synch_weights`

判断方式：

- 如果 `env_runner_sampling_timer` 占比高，说明采样/环境是瓶颈，异步采样或更多 EnvRunner 有价值；
- 如果 `learner_update_timer` 占比高，说明 learner 是瓶颈，应优先优化 batch、模型、GPU、learner 数量；
- 如果 `synch_weights` 占比高，说明模型权重同步成本明显，可以降低同步频率或减少 EnvRunner 数量；
- 如果每个 timer 都不大但 iteration 很慢，可能是 callback、checkpoint、evaluation 或日志开销。

### 5.2 先调大 fragment，降低同步频率

当前配置：

```python
.env_runners(num_env_runners=1, rollout_fragment_length=1)
```

可以先尝试：

```python
.env_runners(num_env_runners=1, rollout_fragment_length=16)
```

或：

```python
.env_runners(num_env_runners=1, rollout_fragment_length=32)
```

这样每次 Ray 调用携带更多样本，固定开销被摊薄。对 `Pendulum-v1` 这种轻环境，这通常比改异步更直接。

### 5.3 Toy 环境可考虑不用远端 EnvRunner

如果只是本地 demo 或调试，远端 Ray actor 的开销可能大于收益。可以试：

```python
.env_runners(num_env_runners=0, rollout_fragment_length=1)
```

这样用 local EnvRunner，减少远端 actor 通信成本。

### 5.4 真实慢环境再增加并行度

如果业务环境单步比较慢，再考虑：

```python
.env_runners(
    num_env_runners=4,
    num_envs_per_env_runner=2,
    rollout_fragment_length=16,
)
```

这类配置通常比直接重写异步算法更稳。

### 5.5 如需异步，先做“异步采样 + 同步学习”

如果确实要改代码，建议第一版只做 bounded async sampling：

```text
启动 EnvRunner 异步 sample 请求
  -> driver 拉取已完成采样结果
  -> 写 replay buffer
  -> replay buffer 同步 sample
  -> learner 同步 update
  -> priority 同步 update
  -> 每 N 次 update 同步一次权重
```

这个方案的优点：

- 能隐藏部分环境采样耗时；
- 保留 learner update 和 priority update 的顺序语义；
- 比全异步 learner 风险低；
- 便于通过参数控制 policy lag。

可以考虑新增一个独立算法类，例如：

```text
AsyncCustomSAC
```

而不是直接改现有 `CustomSAC`，避免破坏稳定路径。

### 5.6 全异步 learner 放到后续阶段

全异步 learner update 需要额外解决：

- `num_learners > 0` 的资源配置；
- replay sample 和 TD-error 的 id 对齐；
- 多个 in-flight batch 的优先级更新；
- learner result 的乱序聚合；
- 权重同步节流；
- checkpoint/evaluation 与 learner step 的一致性。

这更像一个新的 off-policy async training executor，不建议作为第一版优化。

## 6. 建议的工程方案

如果要在框架中长期支持异步 SAC，建议分三阶段实现。

### 阶段一：配置调优和指标验证

目标是不改算法逻辑，只通过配置和指标确认瓶颈。

建议：

- 在示例中尝试 `rollout_fragment_length=16/32`；
- 对 toy 环境尝试 `num_env_runners=0`；
- 对慢环境尝试增加 `num_env_runners` 和 `num_envs_per_env_runner`；
- 输出并记录关键 timer；
- 比较 samples/sec、training iteration time、episode return 收敛曲线。

### 阶段二：新增异步采样 SAC

新增一个可选算法类，不替换当前 `CustomSAC`。

设计要点：

- 使用 `env_runner_group.foreach_env_runner_async()` 发采样请求；
- 使用 `env_runner_group.fetch_ready_async_reqs()` 拉取完成结果；
- 限制每个 EnvRunner 的 in-flight 请求数量；
- replay buffer 写入仍在 driver 侧串行执行；
- learner update 保持同步；
- 每隔固定 learner step 或 env step 同步一次权重；
- 对外暴露配置项，例如：
  - `async_sampling=True`
  - `max_sample_requests_in_flight_per_env_runner`
  - `weight_sync_interval`
  - `min_replay_size_before_learning`

### 阶段三：评估全异步 learner

只有在阶段二仍不能满足吞吐时，再考虑：

- `learner_group.update(async_update=True)`；
- `num_learners > 0`；
- replay sample id 跟踪；
- TD-error 和 sample priority 的精确对齐；
- result reorder buffer；
- 对 stale data 的上限控制。

## 7. 总结

当前 SAC 训练流程确实存在同步 barrier：

- 采样是同步 barrier；
- learner update 默认同步；
- 权重同步也有固定成本；
- `rollout_fragment_length=1` 会显著放大这些成本。

但对当前 `02_sac_pendulum.py` 来说，第一优先级不应是立刻改全异步，而是：

1. 先看 timer，确认瓶颈；
2. 调大 `rollout_fragment_length`，降低 Ray 调用频率；
3. toy 环境尝试 local EnvRunner；
4. 慢环境增加并行 EnvRunner；
5. 若仍有瓶颈，再做“异步采样 + 同步学习”；
6. 最后才考虑全异步 learner。

简短判断：

> SAC 是 off-policy，适合异步采样；但当前代码直接改成全异步风险较高。更稳的路线是先做配置调优，再新增可选的 bounded async sampling 版本。
