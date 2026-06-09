# Async Env Sampling Metrics

本文档记录 `AsyncCustomSAC` 中测试 `async_env_sampling` 时最需要关注的打点，以及新增 train credit 消费逻辑下每个打点的判断方式。

## 建议测试配置

只看 EnvRunner 异步采样时，建议关闭 Learner 异步训练，避免 learner 侧 async in-flight/update 指标干扰判断：

```python
config = (
    AsyncCustomSACConfig()
    .env_runners(
        num_env_runners=1,  # 必须 > 0，才会使用 remote EnvRunner async sampling
        max_requests_in_flight_per_env_runner=1,
    )
    .algorithm_options(
        {
            "env_sampling": "async",
            "learner_training": "sync",
            "pipeline_log_interval": 10,
        }
    )
)
```

如果想更纯粹地观察采样链路，可以临时把 `num_steps_sampled_before_learning_starts` 设大，让算法先只做 fetch/reissue sample，不进入 learner update。此时 `async_sac_warmup_blocked = 1`，`async_sac_train_credit_added = 0`，`async_sac_train_credit_spent = 0` 都是预期现象。

如果想同时验证新增 credit 消费逻辑，保持 warmup 较小或为 `0`，并重点观察：

```text
async_sac_train_credit_added
async_sac_train_credit_spent
async_sac_train_credit
async_sac_sync_learner_updates
```

sync learner 默认会在一个 async training step 内按 credit 连续执行多次 update。如果担心单个 step 被 learner update 卡太久，可以设置可选上限：

```python
.algorithm_options(
    {
        "env_sampling": "async",
        "learner_training": "sync",
        "max_sync_learner_updates_per_step": 2,
    }
)
```

## EnvRunner 采样打点

| 打点 | 主要含义 | 正常/预期现象 |
| --- | --- | --- |
| `async_sac_env_sampling_async` | 当前 runtime 是否真的启用了 async EnvRunner sampling。 | 应为 `1`。如果是 `0`，说明没有进入 async env sampling path。 |
| `async_sac_learner_training_async` | 当前 learner training 是否异步。 | 只测试 env sampling 时应为 `0`；如果是 `1`，learner async 的 in-flight/update 会混入测试结果。 |
| `async_sac_remote_env_runners_healthy` | 当前健康的 remote EnvRunner 数量。 | 应 `> 0`。如果是 `0`，会走 local synchronous sampling fallback。 |
| `async_sac_env_sample_inflight_after_reissue` | 每轮 fetch ready samples 后，重新发起下一轮 sample request 后的在途请求数。 | 关键指标。稳定状态通常 `>= 1`；`num_env_runners=1` 且 `max_requests_in_flight_per_env_runner=1` 时通常为 `1`。 |
| `async_sac_env_sample_inflight` | training step 末尾仍在途的 async sample request 数。 | 正常 async pipeline 下通常保持非零，说明 EnvRunner 正在后台采样。 |
| `async_sac_env_ready_results` | 本次 step 从 EnvRunner 取回多少个已经完成的 sample 结果。 | 第一轮可能为 `0`；后续应周期性 `> 0`。这个数按 EnvRunner actor result 计，不是 episode 数。 |
| `async_sac_env_steps_added` | 本次取回的 sample 中，有多少 env steps 被加入 replay buffer。 | 有 ready result 时应 `> 0`；如果长期为 `0`，说明没有拿到有效 sample。 |
| `async_sac_new_env_steps` | pipeline step 维度记录的本轮新增 env steps。 | 通常与 `async_sac_env_steps_added` 一致，可作为 step-level 汇总查看。 |
| `async_sac_current_sampled_timesteps` | 当前累计 sampled timesteps。 | 应持续增长，用于确认采样结果确实进入 RLlib metrics。 |
| `async_sac_cached_connector_states` | 当前缓存的 EnvRunner connector state 数量，用于后续权重/connector state 同步。 | 有 remote ready result 后应增长或保持有值；一直为 `0` 通常说明没拿到 remote sample result。 |
| `timers/env_runner_sampling_timer` | EnvRunner sampling 相关计时。async path 下主要计时 fetch-ready + reissue request，而不是完整环境采样耗时。 | 应较小，不应像同步采样一样被环境 step 时间长期阻塞。 |

## Train Credit 打点

`async_sac_train_credit` 是 backlog/gauge，不是累计计数器。一个 `1.0` credit 表示：新增采样量按训练强度换算后，允许执行一次 learner update。

| 打点 | 主要含义 | 正常/预期现象 |
| --- | --- | --- |
| `async_sac_warmup_blocked` | 当前是否仍处于 `num_steps_sampled_before_learning_starts` warmup 阶段。 | warmup 前为 `1`，不会新增或消费 train credit；warmup 后为 `0`。 |
| `async_sac_train_credit_added` | 本轮新增 env steps 换算出的 credit。 | warmup 后，有新 sample 时通常 `> 0`；无 ready sample 时为 `0`。 |
| `async_sac_train_credit_spent` | 本轮实际消费的 credit 数，也就是本轮实际执行/发起的 learner update 数。 | 新逻辑下可以大于 `1.0`；sync learner 会尽量追平 backlog，async learner 受 in-flight 限制。 |
| `async_sac_train_credit` | 本轮结束时尚未消费的 credit backlog。 | 短期波动正常；如果长期增长，说明 credit 生产速度大于 learner 消费能力，或 sync update limit / async in-flight 正在限流。 |
| `async_sac_train_credit_blocked` | 本轮是否因为 credit 不足而没有训练。 | `1` 表示 credit `< 1.0` 且本轮未训练；warmup 后、采样较慢时可能出现。 |
| `async_sac_sync_learner_updates` | sync learner 模式下，本轮执行了多少次阻塞 learner update。 | 新逻辑下可以大于 `1`，这是对齐 RLlib `sample_and_train_weight` 多 update 语义的预期行为。 |
| `async_sac_sync_learner_update_limit_reached` | sync learner 模式下，本轮是否因为 `max_sync_learner_updates_per_step` 达到上限而停止继续消费 credit。 | 默认无上限时通常为 `0`；配置上限后为 `1` 表示剩余 credit 被有意保留到后续 step。 |
| `async_sac_async_learner_updates_issued` | async learner 模式下，本轮发起了多少个 async learner update。 | 只测 env sampling 且 `learner_training="sync"` 时不用看；async learner 模式下可以大于 `1`，但不会超过 in-flight 容量。 |
| `async_sac_learner_blocked_by_inflight` | async learner 模式下，本轮是否因为 learner in-flight 已满而停止发 update。 | 只测 env sampling 时通常不用看；为 `1` 表示 credit 保留为 backlog 是合理限流结果。 |
| `async_sac_learner_inflight` | training step 末尾 learner update 在途请求数。 | sync learner 模式应为 `0`；async learner 模式下可能 `> 0`。 |

## 最小判断组合

只判断 `async_env_sampling` 是否跑起来，可以先看下面这组：

```text
async_sac_env_sampling_async = 1
async_sac_learner_training_async = 0
async_sac_remote_env_runners_healthy > 0
async_sac_env_sample_inflight_after_reissue > 0
async_sac_env_ready_results 周期性 > 0
async_sac_env_steps_added 周期性 > 0
```

看到这组现象，基本可以认为：

```text
主线程没有同步等待 EnvRunner 完整 sample；
EnvRunner sample request 保持后台 in-flight；
下一轮 training_step 再 fetch 已完成结果，并立即 reissue 新请求。
```

如果同时判断 credit 消费是否健康，可以再看：

```text
warmup 后：
  async_sac_warmup_blocked = 0
  async_sac_train_credit_added 随 async_sac_new_env_steps 增加
  async_sac_train_credit_spent 约等于本轮 learner update 数
  async_sac_sync_learner_updates == async_sac_train_credit_spent  # learner_training="sync"
```

长期观察时重点看：

```text
async_sac_train_credit_added 长期 > async_sac_train_credit_spent
  => async_sac_train_credit 会增长，说明 learner 消费跟不上或被上限限制。

async_sac_train_credit_added 长期约等于 async_sac_train_credit_spent
  => credit backlog 稳定，采样/训练节奏基本匹配。
```

## 典型时序

Ray 2.54 的 `EnvRunnerGroup.foreach_env_runner_async_fetch_ready()` 行为是：先取回之前已经 ready 的异步结果，再发起新的异步请求。

因此常见时序是：

```text
step 1:
  async_sac_env_sampling_async = 1
  async_sac_learner_training_async = 0
  async_sac_env_ready_results = 0  # 可能为 0，因为第一轮主要是发起 async request
  async_sac_env_sample_inflight_after_reissue = 1
  async_sac_train_credit_added = 0
  async_sac_train_credit_spent = 0

step 2+ warmup 前:
  async_sac_env_ready_results 周期性 > 0
  async_sac_env_steps_added 周期性 > 0
  async_sac_current_sampled_timesteps 持续增长
  async_sac_warmup_blocked = 1
  async_sac_train_credit_added = 0
  async_sac_train_credit_spent = 0

step 2+ warmup 后:
  async_sac_env_ready_results 周期性 > 0
  async_sac_env_steps_added 周期性 > 0
  async_sac_env_sample_inflight_after_reissue 继续 >= 1
  async_sac_train_credit_added 根据新 env steps 增加
  async_sac_sync_learner_updates 可能为 0、1 或 >1
  async_sac_train_credit_spent = async_sac_sync_learner_updates
```

## 容易误判的点

- `async_sac_env_ready_results` 第一轮为 `0` 是正常的，因为第一轮通常只是发起 async sample request。
- `async_sac_env_sample_inflight_after_reissue` 比 `async_sac_env_ready_results` 更能说明 pipeline 是否保持“后台采样中”。
- `async_sac_remote_env_runners_healthy = 0` 时，即使配置了 `"env_sampling": "async"`，也不是 remote async sampling。
- 只测 env sampling 时，应使用 `"learner_training": "sync"`，否则 learner async 的 in-flight/update 耗时会混在一起。
- `async_sac_train_credit` 不是累计训练次数；它是未消费 backlog。训练正常时它可以上下波动，也可以长期接近 `0`。
- 新 credit 逻辑下，`async_sac_train_credit_spent > 1` 和 `async_sac_sync_learner_updates > 1` 是正常的，表示单个 async training step 内消费了多个 train credit。
- 如果配置了 `max_sync_learner_updates_per_step`，看到 `async_sac_sync_learner_update_limit_reached = 1` 且 credit 剩余，不一定是 bug，而是主动限制单步 learner update 时长。
- 如果只想看采样性能，不要把 sync learner 多 update 导致的单步耗时增加误判为 EnvRunner async fetch/reissue 变慢；可以临时提高 warmup 或设置较小的 `max_sync_learner_updates_per_step` 做对照。

## 代码位置

主要实现位置：

- `rlframework/algorithms/async_sac.py:498`：`training_step()` 根据 `_use_async_env_sampling` 选择 async path。
- `rlframework/algorithms/async_sac.py:516`：`_training_step_async()` 的整体 pipeline，包含 fetch/reissue、warmup、credit added 和 learner 调度。
- `rlframework/algorithms/async_sac.py:567`：`_maybe_run_sync_learner_update()` 按 credit 连续执行 sync learner update，并支持 `max_sync_learner_updates_per_step`。
- `rlframework/algorithms/async_sac.py:620`：`_maybe_issue_async_learner_update()` 按 credit 连续发起 async learner update，直到 learner in-flight 满。
- `rlframework/algorithms/async_sac.py:723`：`_record_async_pipeline_metrics()` 记录 step-level pipeline 和 train credit metrics。
- `rlframework/algorithms/async_sac.py:853`：`_fetch_ready_samples()` fetch ready samples 并 reissue request。
- `rlframework/algorithms/async_sac.py:987`：`_process_learner_results()` 聚合 learner results，并用最新 RLModule state 同步 EnvRunner 权重一次。
- `rlframework/algorithms/async_sac.py:1052`：`_calc_credit_increment()` 将新增 env steps 换算为 train credit。
