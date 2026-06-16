# EpisodeReplayBuffer sample() 性能优化分析

本文聚焦 RLlib `EpisodeReplayBuffer.sample()` 的性能，而不是 `add()`、actor 调度或 inqueue/outqueue。结论先行：当前 async SAC 主路径使用 `sample_episodes=True`，其主要开销来自 Python 对象切片和重建，不是随机索引本身。针对 stateless SAC，可以加一个保守的 direct-transition fast path，预计比默认 `_sample_episodes()` 明显更快。

## 当前调用路径

本仓库 `AsyncCustomSAC` 的 replay sample 参数在 `rlframework/algorithms/async_sac.py` 中生成：

```python
return {
    "num_items": num_items or runtime_config.total_train_batch_size,
    "n_step": runtime_config.n_step,
    "batch_length_T": (
        self._module_is_stateful * runtime_config.model_config.get("max_seq_len", 0)
    ),
    "lookback": int(self._module_is_stateful),
    "min_batch_length_T": (
        runtime_config.burn_in_len if hasattr(runtime_config, "burn_in_len") else 0
    ),
    "gamma": runtime_config.gamma,
    "beta": runtime_config.replay_buffer_config.get("beta"),
    "sample_episodes": True,
}
```

这意味着：

- `sample_episodes=True`，所以 RLlib 会走 `EpisodeReplayBuffer._sample_episodes()`。
- stateless 模型下，`self._module_is_stateful == False`，所以 `batch_length_T == 0`，`lookback == 0`。
- 这个场景本质上是采样一批 transition episodes，常见于 SAC/DQN 等 off-policy learner 输入。

相关 RLlib 源码位于：

- `.venv/lib/python3.12/site-packages/ray/rllib/utils/replay_buffers/episode_replay_buffer.py`
- `.venv/lib/python3.12/site-packages/ray/rllib/env/single_agent_episode.py`
- `.venv/lib/python3.12/site-packages/ray/rllib/env/utils/infinite_lookback_buffer.py`

本地环境中的 Ray 版本为 `2.49.2`。

## sample() 的主要流程

`EpisodeReplayBuffer.sample()` 先按 `sample_episodes` 分流：

```python
if sample_episodes:
    return self._sample_episodes(...)
else:
    return self._sample_batch(...)
```

当前 async SAC 走的是 `_sample_episodes()`。简化后的逻辑如下：

```python
while B < batch_size_B:
    # 1. 从 timestep 级 _indices 里随机取一个位置
    index_tuple = self._indices[self.rng.integers(len(self._indices))]
    episode_idx = index_tuple[0] - self._num_episodes_evicted
    episode_ts = index_tuple[1]
    episode = self.episodes[episode_idx]

    # 2. 如果 n_step 是区间，随机采一个实际 n_step
    if random_n_step:
        actual_n_step = int(self.rng.integers(n_step[0], n_step[1]))

    # 3. 对原 episode 做 slice
    sampled_episode = episode.slice(slice(episode_ts, episode_ts + actual_n_step))

    # 4. 从 sliced episode 里取 reward，并计算 discounted n-step return
    raw_rewards = sampled_episode.get_rewards()
    rewards = scipy.signal.lfilter([1], [1, -gamma], raw_rewards[::-1], axis=0)[-1]

    # 5. 再构造一个新的 SingleAgentEpisode，只包含 obs_t、obs_t+n、action_t、reward
    sampled_episode = SingleAgentEpisode(...)

    # 6. 写入 n_step 和 weights extra_model_outputs
    sampled_episode.extra_model_outputs["n_step"] = InfiniteLookbackBuffer(...)
    sampled_episode.extra_model_outputs["weights"] = InfiniteLookbackBuffer(...)

    # 7. 更新 unique episode/timestep 统计
    sampled_env_step_idxs.add(hashlib.sha256(...).hexdigest())
```

这个流程的关键点是：每个 sampled item 都会先创建一个临时 sliced episode，再创建最终返回给 learner 的 sampled episode。

## 主要性能瓶颈

### 1. 每个 transition 都调用 episode.slice()

`SingleAgentEpisode.slice()` 会分别切：

- observations
- infos
- actions
- rewards
- extra_model_outputs

每一种数据都会经过 `InfiniteLookbackBuffer.get()` / `_get_slice()` / `_interpret_slice()`。对于 `num_items=256`，一次 sample call 就会重复执行 256 次 episode slice。

在 stateless SAC 的 transition sampling 中，实际只需要：

```text
obs_t
action_t
reward_t:t+n
obs_t+n
terminated/truncated
```

因此完整 `episode.slice()` 是偏重的。

### 2. 每个 transition 构造两次 SingleAgentEpisode 相关对象

默认路径大致是：

```text
原 episode
  -> episode.slice(...) 生成临时 sampled_episode
  -> 读取临时 sampled_episode 的 obs/action/reward/info
  -> 再构造最终 SingleAgentEpisode
```

`SingleAgentEpisode.__init__()` 内部会创建多个 `InfiniteLookbackBuffer`，并执行 `validate()`。这在 batch 维度上被放大。

### 3. n_step=1 仍然走 scipy.signal.lfilter()

当前代码即使 `n_step=1`，也会执行：

```python
rewards = scipy.signal.lfilter([1], [1, -gamma], raw_rewards[::-1], axis=0)[-1]
```

对 `n_step=1` 来说，这等价于直接取当前 reward：

```python
reward = episode.rewards[episode_ts]
```

为单个标量调用 scipy 会有明显固定开销。

### 4. metrics 中使用 sha256，并且存在重复计算

`_sample_episodes()` 为了统计 unique env step，使用：

```python
sampled_env_step_idxs.add(
    hashlib.sha256(f"{episode.id_}-{episode_ts}".encode()).hexdigest()
)
```

而且本地源码中相同的 `sha256` add 出现了两次。对统计 unique count 来说，加密 hash 没有必要；用 `(episode.id_, episode_ts)` 或内部整数 key 就足够。

### 5. 每个 item 单独调用 rng.integers()

默认实现每采一个 item 调一次：

```python
self.rng.integers(len(self._indices))
```

这不是最大瓶颈，但当对象构造和 slice 被优化后，随机数调用会变得更明显。可以一次性生成：

```python
sample_positions = self.rng.integers(len(self._indices), size=batch_size_B)
```

## 本地 profiling 观察

测试环境：

- Ray `2.49.2`
- synthetic replay buffer
- `800` 个 episode
- 每个 episode `64` step
- observation shape `(8,)`
- action shape `(2,)`
- `num_items=256`
- `n_step=1`
- `sample_episodes=True`
- stateless：`batch_length_T=0`，`lookback=0`

粗略 benchmark 结果：

| 采样方式 | 单次 sample call 耗时 |
| --- | ---: |
| RLlib `EpisodeReplayBuffer.sample(sample_episodes=True)` | 约 `71 ms` |
| prototype direct-transition fast path | 约 `16 ms` |
| RLlib `_sample_batch()`，即 `sample_episodes=False` | 约 `7.7 ms` |

说明：

- 这个 benchmark 是 synthetic workload，绝对值不能直接等同真实训练。
- 相对关系有参考意义：`sample_episodes=True` 的对象构造开销明显高于直接 batch 路径。
- direct-transition fast path 仍然返回 `SingleAgentEpisode` list，所以保留了与当前 learner 输入更接近的语义。
- `_sample_batch()` 更快，但返回格式不同，接入 RLlib new API learner 的改动更大。

cProfile 也显示默认路径的主要累计耗时集中在：

- `EpisodeReplayBuffer._sample_episodes()`
- `SingleAgentEpisode.__init__()`
- `SingleAgentEpisode.slice()`
- `InfiniteLookbackBuffer.__init__()` / `get()` / `_get_slice()`
- `scipy.signal.lfilter()`
- metrics logging

## 优化方向一：stateless direct-transition fast path

这是最推荐先做的优化。目标不是改变 replay sampling 语义，而是在最常见的 stateless transition sampling 场景下跳过昂贵的通用 episode slicing。

### 命中条件

建议第一版只在非常保守的条件下启用：

```text
sample_episodes == True
batch_length_T is None or batch_length_T == 0
lookback == 0
include_extra_model_outputs == False
to_numpy == False
非 prioritized replay buffer
```

可以先支持：

```text
n_step 是 int
min_batch_length_T == 0
```

后续再支持：

```text
n_step 是 tuple
min_batch_length_T > 0
include_infos
```

未命中条件时直接 fallback 到 RLlib 原实现：

```python
return super()._sample_episodes(...)
```

### 核心思路

默认路径：

```text
sample index
  -> episode.slice(...)
  -> sampled_episode.get_observations/get_actions/get_rewards
  -> scipy.signal.lfilter(...)
  -> SingleAgentEpisode(...)
```

fast path：

```text
sample index
  -> 直接访问 episode.observations/actions/rewards/infos 的底层 data
  -> 直接计算 n-step reward
  -> 只构造最终 SingleAgentEpisode
```

伪代码：

```python
def _sample_episodes_fast_transition(...):
    positions = self.rng.integers(len(self._indices), size=batch_size_B)
    sampled_episodes = []

    for pos in positions:
        eps_abs_idx, episode_ts = self._indices[pos]
        episode_idx = eps_abs_idx - self._num_episodes_evicted
        episode = self.episodes[episode_idx]

        end_ts = min(episode_ts + actual_n_step, len(episode))

        obs_t = get_observation_direct(episode, episode_ts)
        obs_tp_n = get_observation_direct(episode, end_ts)
        action_t = get_action_direct(episode, episode_ts)
        reward = compute_n_step_reward_direct(
            episode,
            episode_ts,
            end_ts,
            gamma,
        )

        sampled_episode = SingleAgentEpisode(
            id_=episode.id_,
            agent_id=episode.agent_id,
            module_id=episode.module_id,
            observation_space=episode.observation_space,
            action_space=episode.action_space,
            observations=[obs_t, obs_tp_n],
            actions=[action_t],
            rewards=[reward],
            infos=[info_t, info_tp_n],
            terminated=episode.is_terminated if end_ts == len(episode) else False,
            truncated=episode.is_truncated if end_ts == len(episode) else False,
            t_started=episode_ts,
            len_lookback_buffer=0,
        )

        sampled_episode.extra_model_outputs["n_step"] = ...
        sampled_episode.extra_model_outputs["weights"] = ...
        sampled_episodes.append(sampled_episode)

    update_sample_metrics(...)
    return sampled_episodes
```

### 需要注意的语义细节

#### observation/action/reward 的索引关系

`SingleAgentEpisode` 中：

```text
observations: o0 ... oT
actions:      a1 ... aT
rewards:      r1 ... rT
len(episode) == len(actions) == len(rewards)
len(observations) == len(episode) + 1
```

当采样 `episode_ts` 时：

```text
obs_t      = observations[episode_ts]
action_t   = actions[episode_ts]
reward_t   = rewards[episode_ts]
obs_t+n    = observations[episode_ts + n]
```

如果靠近 episode 末尾，`episode_ts + n` 可能超过 episode 长度，需要截断到 `len(episode)`，并保持与 RLlib 原逻辑一致。

#### infos 是否保留

RLlib 当前 `_sample_episodes()` 构造最终 `SingleAgentEpisode` 时会填入：

```python
infos=[
    sampled_episode.get_infos(0),
    sampled_episode.get_infos(-1),
]
```

即使调用参数里 `include_infos=False`，源码仍然会构造 infos。为了保持兼容，fast path 第一版建议也直接复制 `info_t` 和 `info_t+n`，而不是用空 dict 替代。

#### n_step 和 weights

默认实现会给 sampled episode 写入：

```python
sampled_episode.extra_model_outputs["n_step"]
sampled_episode.extra_model_outputs["weights"]
```

fast path 也必须保留，否则 learner/loss 侧可能缺字段。

#### terminated/truncated

默认逻辑只有 slice 到原 episode 末尾时才保留 done 状态。fast path 应该保持：

```python
done_at_end = end_ts == len(episode)
terminated = episode.is_terminated if done_at_end else False
truncated = episode.is_truncated if done_at_end else False
```

#### t_started

默认最终 sampled episode 使用：

```python
t_started=episode_ts
```

fast path 也应保持一致。

## 优化方向二：n-step reward O(1) 或低开销计算

### n_step=1

第一优先级是特判：

```python
if actual_n_step == 1:
    reward = rewards[episode_ts]
```

这可以完全跳过 `scipy.signal.lfilter()`。

### 小 n_step

如果 `n_step` 通常较小，可以用 Python loop：

```python
reward = 0.0
discount = 1.0
for r in rewards[episode_ts:end_ts]:
    reward += discount * r
    discount *= gamma
```

这对短 n-step 往往比每个 transition 调 scipy 更划算。

### 大 n_step 或固定 n_step

如果 `n_step` 较大，且 `gamma` 固定，可以在 episode 入库时或首次采样时构建 discounted prefix：

```text
G[t] = r[t] + gamma * r[t+1] + gamma^2 * r[t+2] + ...
```

然后：

```text
return(t, n) = G[t] - gamma^n * G[t+n]
```

这样 n-step reward 可以从 `O(n)` 降到 `O(1)`。

代价：

- 每个 episode 需要额外缓存一个数组。
- 如果 `gamma` 可变，缓存需要按 gamma 区分，复杂度上升。
- 如果 episode 会 concat，需要更新缓存或延迟重建。

因此建议第一版先做 `n_step=1` 特判和小 n-step loop。

## 优化方向三：减少 metrics 开销

默认 sample metrics 会统计：

- sampled timesteps
- unique episodes per sample
- unique env steps per sample
- sampled n-step
- utilization
- per-agent metrics
- per-module metrics

这些指标有价值，但不一定需要在每个高频 sample call 中完整计算。

可选优化：

1. 用 `(episode_internal_idx, episode_ts)` 替代 `sha256`。
2. 去掉重复的 `sampled_env_step_idxs.add(...)`。
3. 增加 metrics 采样频率，例如每 N 次 sample 完整统计一次，其余只更新 lifetime counter。
4. 在 replay actor 外层统计 latency，把 replay buffer 内部 metrics 降低频率。

风险：

- RLlib 现有结果字典可能依赖这些 metrics。
- 单元测试可能比较 metrics 内容。
- 如果要做 drop-in replacement，第一版可以只替换 hash key，不降低 metrics 频率。

## 优化方向四：批量随机索引

默认每个 item 调一次 RNG：

```python
index_tuple = self._indices[self.rng.integers(len(self._indices))]
```

可以改成：

```python
positions = self.rng.integers(len(self._indices), size=batch_size_B)
for pos in positions:
    index_tuple = self._indices[pos]
```

收益不会像去掉 `episode.slice()` 那么大，但实现简单，且不会改变采样分布。

需要注意：

- 如果有 `min_batch_length_T` 导致 resample，预生成的 positions 可能不够，需要补采。
- 如果 `n_step` 是 tuple，也可以同时批量生成 n-step 候选。

## 优化方向五：返回 batch/tensor，而不是 episode list

从 benchmark 看，RLlib `_sample_batch()` 明显更快。原因是它直接构造 batch dict，而不是为每个 transition 构造 `SingleAgentEpisode`。

理论上，最优路径应该是：

```text
ReplayBuffer.sample()
  -> 返回 obs/actions/rewards/next_obs/dones/weights/n_step 的 tensor batch
  -> Learner 直接消费 tensor batch
```

但这需要确认 RLlib new API SAC learner 当前是否能接受非 episode 输入，或者需要改 learner connector / update 输入格式。这个方向收益更大，但改动范围也更大。

建议作为第二阶段，不要和第一版 fast path 混在一起。

## 建议的落地方案

### 新增一个自定义 buffer 类

建议在 `rlframework/utils/replay_buffers.py` 新增：

```python
class FastSampleEpisodeReplayBuffer(BatchEvictEpisodeReplayBuffer):
    """Episode replay buffer with a stateless transition-sampling fast path."""

    def _sample_episodes(...):
        if self._can_use_fast_transition_sample(...):
            return self._sample_episodes_fast_transition(...)
        return super()._sample_episodes(...)
```

这样可以复用已有 `BatchEvictEpisodeReplayBuffer` 的 add/evict 优化，同时单独优化 sample。

### AsyncCustomSAC 需要允许该类型

当前 async SAC 对 replay buffer type 有白名单。新增类后需要把它加入：

```python
_SUPPORTED_ASYNC_REPLAY_BUFFER_TYPES = (
    EpisodeReplayBuffer,
    BatchEvictEpisodeReplayBuffer,
    FastSampleEpisodeReplayBuffer,
)
```

以及对应的字符串名集合。

### 配置方式

训练配置可以写成：

```python
from rlframework.utils.replay_buffers import FastSampleEpisodeReplayBuffer

config = (
    AsyncCustomSACConfig()
    .training(
        replay_buffer_config={
            "type": FastSampleEpisodeReplayBuffer,
            "capacity": 100_000,
        }
    )
)
```

或者字符串路径：

```python
replay_buffer_config={
    "type": "rlframework.utils.replay_buffers.FastSampleEpisodeReplayBuffer",
    "capacity": 100_000,
}
```

## 验证计划

### 1. 采样语义 parity 测试

构造固定 episode：

```text
observations = [0, 1, 2, 3, 4]
actions      = [10, 11, 12, 13]
rewards      = [1.0, 2.0, 3.0, 4.0]
```

测试：

- `n_step=1`
- `n_step=3`
- 靠近 episode 末尾的截断 n-step
- terminated episode
- truncated episode
- 多 episode buffer
- 固定 RNG seed 后，对比 sampled episode 的 obs/action/reward/done/t_started

### 2. fallback 测试

以下场景必须回退到 RLlib 原实现：

- `batch_length_T > 0`
- `lookback > 0`
- `include_extra_model_outputs=True`
- `to_numpy=True`
- stateful module / RNN
- prioritized replay

测试重点不是性能，而是确保不会误走 fast path。

### 3. metrics 测试

至少验证：

- `sampled_timesteps` 增加正确。
- `get_metrics()` 中 sampled steps 相关字段仍存在。
- unique episode / env step count 不出现明显错误。
- `n_step` 平均值正确。

### 4. benchmark

建议增加一个本地 benchmark 脚本或 example：

```text
buffer type:
  EpisodeReplayBuffer
  BatchEvictEpisodeReplayBuffer
  FastSampleEpisodeReplayBuffer

parameters:
  capacity
  num_episodes
  episode_length
  train_batch_size
  n_step
  obs shape
  action shape
```

输出：

```text
add ms/call
sample ms/call
sampled transitions/sec
fast_path_hit_rate
```

### 5. 训练 smoke test

用 Pendulum-v1 或已有 SAC example 跑短训练：

- 确认没有 learner 输入字段缺失。
- 确认 loss 能正常计算。
- 确认 episode sampling metrics 正常。
- 对比默认 buffer 的短期 reward 曲线不要出现明显异常。

## 风险和边界

### 风险一：直接访问 InfiniteLookbackBuffer.data 是内部实现细节

fast path 需要绕开 `get_observations()` / `get_actions()` 等通用 API，直接读底层 data。这样更快，但依赖 RLlib 内部结构。

缓解方式：

- 只在自定义 buffer 内部使用。
- 保留 fallback。
- 对 list-backed 和 numpy-backed `InfiniteLookbackBuffer` 都加测试。
- 记录支持的 Ray 版本范围。

### 风险二：infos / extra_model_outputs 语义

如果 learner 或 connector 依赖 infos/extra_model_outputs，fast path 可能漏字段。

缓解方式：

- 第一版默认复制 `info_t` 和 `info_t+n`。
- `include_extra_model_outputs=True` 时 fallback。
- 始终保留 `n_step` 和 `weights`。

### 风险三：episode 末尾 n-step 截断

靠近 episode 末尾时，RLlib 原实现允许实际 slice 短于 requested n-step。fast path 必须保持相同行为。

缓解方式：

- 明确测试 `episode_ts + n_step > len(episode)`。
- 对比 reward、next obs、terminated/truncated。

### 风险四：future Ray 版本变更

RLlib 的 `SingleAgentEpisode` 和 `InfiniteLookbackBuffer` 属于 RLlib 内部对象，未来版本可能改结构。

缓解方式：

- class 名称和 docstring 中说明这是针对当前 RLlib episode replay 实现的优化。
- fallback 到 `super()`。
- 在 CI 中覆盖当前支持的 Ray 版本。

## 推荐实施顺序

1. 增加 `FastSampleEpisodeReplayBuffer`，只支持 stateless transition fast path。
2. fast path 首版只支持 `n_step` 为 int，先特判 `n_step=1`。
3. 保持 `infos`、`n_step`、`weights`、`t_started`、done 语义。
4. 其他复杂场景全部 fallback。
5. 加 parity tests 和 micro-benchmark。
6. 把新 buffer 加入 `AsyncCustomSAC` 白名单。
7. 训练 smoke test 通过后，再考虑支持 tuple n-step、prefix return cache、metrics 降频。

## 总结

`EpisodeReplayBuffer.sample()` 的主要问题不是 IO，也不是 actor 调度，而是每个 transition 上的通用 episode slicing 和对象重建。对当前 async SAC 的 stateless 主路径，最有效的优化是：

```text
跳过 episode.slice()
直接从原 episode 中取 obs/action/reward/next_obs
只构造最终 learner 需要的 SingleAgentEpisode
```

这个方向不会减少 replay buffer 的总样本量，也不会改变采样分布，但能显著降低单个 replay actor 的 CPU 时间。后续如果 sample 仍然是瓶颈，再考虑返回 tensor batch 或 replay sharding。
