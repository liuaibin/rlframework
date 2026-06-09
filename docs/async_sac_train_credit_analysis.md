# Async SAC Train Credit Analysis

本文档记录 `AsyncCustomSAC` 中 `async_sac_train_credit` 持续增长的问题分析，以及它和 RLlib 原版 SAC/DQN `calculate_rr_weights()` 训练强度语义之间的关系。

## 实施状态

截至 2026-06-09，本文中的主要建议已经落地到 `rlframework/algorithms/async_sac.py`：

- `learner_training="sync"` 时会按 train credit 连续执行多次 learner update，并通过 `max_sync_learner_updates_per_step` 支持可选上限；
- sync learner 多次 update 的 learner results 会聚合处理，EnvRunner 权重只使用最新 RLModule state 同步一次；
- `learner_training="async"` 时会按 train credit 连续发起 learner update，直到 Learner in-flight 达到 `max_requests_in_flight_per_learner`；
- `async_sac_sync_learner_updates`、`async_sac_async_learner_updates_issued` 和 `async_sac_train_credit_spent` 现在可以大于 1，表示单个 async training step 内消费了多个 train credit。

后文“当前 AsyncCustomSAC”的描述保留为问题分析背景，其中“每次最多一次 update”的说法指优化前实现。

## 背景

RLlib SAC 继承自 DQN，new API stack 的核心训练逻辑来自：

- `/Users/lab/rl/ray-ray-2.54.0/rllib/algorithms/dqn/dqn.py:648`
- `/Users/lab/rl/ray-ray-2.54.0/rllib/algorithms/dqn/dqn.py:650`
- `/Users/lab/rl/ray-ray-2.54.0/rllib/algorithms/dqn/dqn.py:684`

原版同步 SAC/DQN 每个 `training_step()` 先计算 round-robin 权重：

```python
store_weight, sample_and_train_weight = calculate_rr_weights(self.config)
```

然后执行：

```python
for _ in range(store_weight):
    sample_from_env_runners()
    add_to_replay_buffer()

if current_ts >= num_steps_sampled_before_learning_starts:
    for _ in range(sample_and_train_weight):
        sample_from_replay_buffer()
        learner_group.update(...)

    sync_weights_to_env_runners()
```

因此原版逻辑中，一轮采样之后可能会做多轮 learner update。`sample_and_train_weight` 是训练强度的一部分，不是固定一次 update。

## 原版 `calculate_rr_weights()` 语义

`calculate_rr_weights()` 在：

- `/Users/lab/rl/ray-ray-2.54.0/rllib/algorithms/dqn/dqn.py:568`

核心逻辑：

```python
if not config.training_intensity:
    return [1, 1]

native_ratio = config.total_train_batch_size / (
    config.get_rollout_fragment_length()
    * config.num_envs_per_env_runner
    * max(config.num_env_runners + 1, 1)
)

sample_and_train_weight = config.training_intensity / native_ratio
if sample_and_train_weight < 1:
    return [int(np.round(1 / sample_and_train_weight)), 1]
else:
    return [1, int(np.round(sample_and_train_weight))]
```

含义：

- `store_weight`：每个 training step 采样并写入 replay buffer 的轮数。
- `sample_and_train_weight`：每个 training step 从 replay buffer 采样并更新 learner 的轮数。
- `training_intensity` 本质表达的是 `steps_replayed / steps_sampled`。

如果 `training_intensity` 较高，原版同步 SAC 会在一次 `training_step()` 中做多次 learner update。

## 当前 AsyncCustomSAC 的 train credit 逻辑

当前 async sampling path 位于：

- `rlframework/algorithms/async_sac.py:491`：`_training_step_async()`
- `rlframework/algorithms/async_sac.py:522`：新增 train credit
- `rlframework/algorithms/async_sac.py:540`：sync learner update
- `rlframework/algorithms/async_sac.py:580`：async learner update
- `rlframework/algorithms/async_sac.py:994`：`_calc_credit_increment()`

简化逻辑：

```python
new_env_steps = self._fetch_ready_samples_and_reissue()

if not warmup_blocked:
    credit_added = self._calc_credit_increment(new_env_steps)
    self._train_credit += credit_added

    if self._use_async_learner_training:
        self._maybe_issue_async_learner_update()
    else:
        self._maybe_run_sync_learner_update()
```

当前 `_maybe_run_sync_learner_update()` 每次最多消费 `1.0` 个 credit：

```python
if self._train_credit < 1.0:
    return

learner_group.update(...)
self._train_credit -= 1.0
```

当前 `_maybe_issue_async_learner_update()` 也是每次最多发起一次 async learner update，并消费 `1.0` 个 credit。

## `async_sac_train_credit` 的含义

`async_sac_train_credit` 是一个 backlog/gauge，不是累计计数器。

一个 `1.0` credit 表示：当前 replay buffer 中新增采样量按训练强度换算后，允许执行一次 learner update。

如果配置了 `training_intensity`，当前 credit 计算公式是：

```python
credit_added = new_env_steps * training_intensity / total_train_batch_size
```

对应代码：

- `rlframework/algorithms/async_sac.py:1013`

因此：

```text
credit_added > credit_spent
```

长期成立时，`async_sac_train_credit` 会持续增长。

## 为什么会持续增长

当前 async path 每个 `training_step()` 最多消费一次 learner update：

```text
每轮最多 credit_spent = 1.0
```

但采样端可能每轮增加超过 `1.0` credit：

```text
credit_added = new_env_steps * training_intensity / total_train_batch_size
```

例如：

```text
new_env_steps = 64
total_train_batch_size = 32
training_intensity = 1.0
credit_added = 64 * 1.0 / 32 = 2.0
```

如果每轮只消费 `1.0`，则每轮剩余 `+1.0`，`async_sac_train_credit` 就会单调上涨。

这说明当前 async path 的 learner 消费节奏没有完全对齐原版 SAC 的 `sample_and_train_weight` 多 update 语义。

## 和原版 SAC 的差异

原版 SAC：

```text
一轮 training_step 内：
  sample/store 可能执行 store_weight 次
  learner update 可能执行 sample_and_train_weight 次
  最后 sync weights 一次
```

当前 AsyncCustomSAC：

```text
一轮 training_step 内：
  fetch ready samples + reissue async sample 一次
  learner update 最多一次
  sync learner result/weights 可能每次 update 后发生
```

所以当 `sample_and_train_weight > 1` 或按 credit 换算后 `credit_added > 1` 时，当前实现会积累 train credit。

## 建议的语义调整

如果目标是对齐 RLlib SAC 的训练强度，建议区分 sync learner 和 async learner 两种模式。

### 1. `learner_training="sync"`

sync learner update 是阻塞完成的，所以可以在一个 `training_step()` 内连续消费 credit：

```python
while self._train_credit >= 1.0:
    run_one_sync_learner_update()
    self._train_credit -= 1.0
```

这样更接近原版：

```python
for _ in range(sample_and_train_weight):
    learner_group.update(...)
```

为了避免单个 `training_step()` 被 learner update 卡太久，可以加一个可配置上限，例如：

```text
max_sync_learner_updates_per_step
```

推荐默认可以是 `None` 或一个保守值，具体取决于我们更想优先对齐 RLlib 语义，还是优先保证 async sampler 高频 polling。

### 2. `learner_training="async"`

async learner update 不能无限发起，否则会超过 Ray learner actor 的 in-flight 限制。

更合理的逻辑是：

```python
while self._train_credit >= 1.0 and learner_inflight < max_requests_in_flight_per_learner:
    issue_one_async_learner_update()
    self._train_credit -= 1.0
```

如果 learner in-flight 已满，剩余 credit 保留为 backlog。

这时 `async_sac_train_credit` 增长表示 learner 消费能力不足，而不是一定代表 bug。

## 需要注意的权重同步问题

不能简单地在 sync learner 模式下直接循环调用当前 `_maybe_run_sync_learner_update()`。

原因：当前 `_maybe_run_sync_learner_update()` 每次 update 后会调用 `_process_learner_results()`，而 `_process_learner_results()` 可能同步 EnvRunner 权重：

- `rlframework/algorithms/async_sac.py:566`
- `rlframework/algorithms/async_sac.py:929`
- `rlframework/algorithms/async_sac.py:982`

原版 RLlib SAC 是多轮 learner update 完成后，最后只 sync weights 一次。

因此更合理的重构是：

```python
all_learner_results = []
latest_rl_module_state = None

while self._train_credit >= 1.0:
    learner_results = run_one_sync_learner_update_without_weight_sync()
    all_learner_results.extend(learner_results)
    self._train_credit -= 1.0

aggregate_learner_metrics(all_learner_results)
sync_latest_weights_once_to_env_runners()
```

这样可以：

- 保持多轮 learner update 的训练强度语义。
- 避免每个 learner update 都同步一次 EnvRunner 权重。
- 更接近原版 SAC 的执行结构。

## 建议观察的指标

分析 `async_sac_train_credit` 是否合理时，应同时观察：

| 打点 | 含义 |
| --- | --- |
| `async_sac_train_credit` | 当前尚未消费的训练 credit/backlog。 |
| `async_sac_train_credit_added` | 本轮由新增 env steps 换算得到的 credit。 |
| `async_sac_train_credit_spent` | 本轮实际消费的 credit。 |
| `async_sac_train_credit_blocked` | 有无因为 credit 不足而跳过 learner update。 |
| `async_sac_new_env_steps` | 本轮新增 env steps。 |
| `async_sac_sync_learner_updates` | 本轮执行的同步 learner update 次数。当前实现最多为 1。 |
| `async_sac_async_learner_updates_issued` | 本轮发起的异步 learner update 次数。当前实现最多为 1。 |
| `async_sac_learner_blocked_by_inflight` | async learner 是否因为 in-flight 满而无法继续发 update。 |
| `async_sac_learner_inflight` | 当前 learner 侧在途 update 请求数。 |
| `async_sac_warmup_blocked` | 是否仍在 warmup 阶段，warmup 阶段不会训练。 |

判断重点：

```text
如果 async_sac_train_credit_added 长期大于 async_sac_train_credit_spent，
async_sac_train_credit 就会持续增长。
```

## 结论

`async_sac_train_credit` 持续增长的主要原因，是当前 `AsyncCustomSAC` 每个 `training_step()` 最多执行或发起一次 learner update，而 RLlib 原版 SAC 会根据 `sample_and_train_weight` 在一个 `training_step()` 内执行多轮 learner update。

如果目标是对齐 RLlib SAC 的 `training_intensity` 语义，建议：

1. `learner_training="sync"` 时，根据 train credit 连续执行 learner update，直到 `train_credit < 1.0` 或达到可配置上限。
2. `learner_training="async"` 时，根据 train credit 连续发起 learner update，但最多发到 learner in-flight 上限。
3. 重构 learner result 处理逻辑，避免 sync learner 多轮 update 时每轮都同步 EnvRunner 权重；应尽量聚合 learner results 后同步最新权重一次。
