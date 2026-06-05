# Async Env Sampling Metrics

本文档记录 `AsyncCustomSAC` 中只测试 `async_env_sampling` 时最需要关注的打点，以及每个打点的主要含义。

## 建议测试配置

只看 EnvRunner 异步采样时，建议关闭 Learner 异步训练，避免 learner 侧 in-flight/update 指标干扰判断：

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

如果想更纯粹地观察采样链路，可以临时把 `num_steps_sampled_before_learning_starts` 设大，让算法先只做 fetch/reissue sample，不进入 learner update。

## 核心打点

| 打点 | 主要含义 | 正常/预期现象 |
| --- | --- | --- |
| `async_sac_env_sampling_async` | 当前 runtime 是否真的启用了 async EnvRunner sampling。 | 应为 `1`。如果是 `0`，说明没有进入 async env sampling path。 |
| `async_sac_learner_training_async` | 当前 learner training 是否异步。 | 只测试 env sampling 时应为 `0`；如果是 `1`，learner 异步会混入测试结果。 |
| `async_sac_remote_env_runners_healthy` | 当前健康的 remote EnvRunner 数量。 | 应 `> 0`。如果是 `0`，会走 local synchronous sampling fallback。 |
| `async_sac_env_sample_inflight_after_reissue` | 每轮 fetch ready samples 后，重新发起下一轮 sample request 后的在途请求数。 | 关键指标。稳定状态通常 `>= 1`；`num_env_runners=1` 且 `max_requests_in_flight_per_env_runner=1` 时通常为 `1`。 |
| `async_sac_env_sample_inflight` | training step 末尾仍在途的 async sample request 数。 | 正常 async pipeline 下通常保持非零，说明 EnvRunner 正在后台采样。 |
| `async_sac_env_ready_results` | 本次 step 从 EnvRunner 取回多少个已经完成的 sample 结果。 | 第一轮可能为 `0`；后续应周期性 `> 0`。这个数按 EnvRunner actor result 计，不是 episode 数。 |
| `async_sac_env_steps_added` | 本次取回的 sample 中，有多少 env steps 被加入 replay buffer。 | 有 ready result 时应 `> 0`；如果长期为 `0`，说明没有拿到有效 sample。 |
| `async_sac_new_env_steps` | pipeline step 维度记录的本轮新增 env steps。 | 通常与 `async_sac_env_steps_added` 一致，可作为 step-level 汇总查看。 |
| `async_sac_current_sampled_timesteps` | 当前累计 sampled timesteps。 | 应持续增长，用于确认采样结果确实进入 RLlib metrics。 |
| `async_sac_cached_connector_states` | 当前缓存的 EnvRunner connector state 数量，用于后续权重/connector state 同步。 | 有 remote ready result 后应增长或保持有值；一直为 `0` 通常说明没拿到 remote sample result。 |
| `timers/env_runner_sampling_timer` | EnvRunner sampling 相关计时。async path 下主要计时 fetch-ready + reissue request，而不是完整环境采样耗时。 | 应较小，不应像同步采样一样被环境 step 时间长期阻塞。 |

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

## 典型时序

Ray 2.54 的 `EnvRunnerGroup.foreach_env_runner_async_fetch_ready()` 行为是：先取回之前已经 ready 的异步结果，再发起新的异步请求。

因此常见时序是：

```text
step 1:
  async_sac_env_sampling_async = 1
  async_sac_learner_training_async = 0
  async_sac_env_ready_results = 0  # 可能为 0，因为第一轮主要是发起 async request
  async_sac_env_sample_inflight_after_reissue = 1

step 2+:
  async_sac_env_ready_results 周期性 > 0
  async_sac_env_steps_added 周期性 > 0
  async_sac_env_sample_inflight_after_reissue 继续 >= 1
  async_sac_current_sampled_timesteps 持续增长
```

## 容易误判的点

- `async_sac_env_ready_results` 第一轮为 `0` 是正常的，因为第一轮通常只是发起 async sample request。
- `async_sac_env_sample_inflight_after_reissue` 比 `async_sac_env_ready_results` 更能说明 pipeline 是否保持“后台采样中”。
- `async_sac_remote_env_runners_healthy = 0` 时，即使配置了 `"env_sampling": "async"`，也不是 remote async sampling。
- 只测 env sampling 时，应使用 `"learner_training": "sync"`，否则 learner async 的 in-flight/update 耗时会混在一起。

## 代码位置

主要实现位置：

- `rlframework/algorithms/async_sac.py:473`：`training_step()` 根据 `_use_async_env_sampling` 选择 async path。
- `rlframework/algorithms/async_sac.py:491`：`_training_step_async()` 的整体 pipeline。
- `rlframework/algorithms/async_sac.py:665`：`_record_async_pipeline_metrics()` 记录 step-level pipeline metrics。
- `rlframework/algorithms/async_sac.py:795`：`_fetch_ready_samples()` fetch ready samples 并 reissue request。
- `rlframework/algorithms/async_sac.py:831`：调用 `foreach_env_runner_async_fetch_ready()`。
