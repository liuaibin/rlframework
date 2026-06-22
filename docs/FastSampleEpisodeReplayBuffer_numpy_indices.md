# FastSampleEpisodeReplayBuffer 使用 NumPy 固定索引内存的方案

> 落地实现：`rlframework.utils.replay_buffers.NumpyIndexedFastSampleEpisodeReplayBuffer`
> 继承自 `FastSampleEpisodeReplayBuffer`，使用本文第一版方案
> “NumPy fixed arrays + eviction compaction”。

## 背景

当前 `FastSampleEpisodeReplayBuffer` 继承自：

```text
FastSampleEpisodeReplayBuffer
  -> BatchEvictEpisodeReplayBuffer
  -> EpisodeReplayBuffer
```

底层索引仍然沿用 RLlib 的 `_indices`：

```python
self._indices = [
    (episode_abs_idx, timestep_in_episode),
    ...
]
```

每个元素是一个 Python tuple，例如：

```python
[
    (0, 0), (0, 1), (0, 2),
    (1, 0), (1, 1),
    (2, 0), (2, 1), (2, 2),
]
```

采样时从 `_indices` 中随机取一个 timestep 索引：

```python
episode_abs_idx, episode_ts = self._indices[
    self.rng.integers(len(self._indices))
]
```

capacity 满了以后，eviction 会把旧 episode 的 timestep 索引删掉。当前优化后的
`BatchEvictEpisodeReplayBuffer` 虽然已经做到“一次 add 最多 rebuild 一次”，但仍然会创建新的
list：

```python
self._indices = [
    idx_tuple
    for idx_tuple in self._indices
    if idx_tuple[0] not in evicted_episode_indices
]
```

这会带来几个问题：

- 每次 rebuild 都分配一个新的大 list。
- 旧 list 被释放，触发大量 Python 对象引用计数更新。
- 每个 timestep 都是 Python tuple，新增数据时也会不断创建 tuple。
- capacity 越大，`_indices` 越长，内存分配和 GC 压力越明显。

所以，“把 `_indices` 放到一块固定申请的内存上”是合理方向。

## 核心思路

不要使用 `np.array(list_of_tuples)`，而是把原来的 tuple 拆成两列连续数组：

```python
self._index_episode = np.empty(index_capacity, dtype=np.int64)
self._index_timestep = np.empty(index_capacity, dtype=np.int32)
self._num_indices = 0
```

原来一个元素是：

```python
(episode_abs_idx, timestep_in_episode)
```

NumPy 版改成：

```python
self._index_episode[slot] = episode_abs_idx
self._index_timestep[slot] = timestep_in_episode
```

例如：

```text
slot:              0   1   2   3   4
_index_episode:   10  10  10  11  11
_index_timestep:   0   1   2   0   1
_num_indices = 5
```

这等价于：

```python
_indices = [
    (10, 0), (10, 1), (10, 2),
    (11, 0), (11, 1),
]
```

采样时：

```python
slot = int(self.rng.integers(self._num_indices))

episode_abs_idx = int(self._index_episode[slot])
episode_ts = int(self._index_timestep[slot])
```

## 推荐第一版：NumPy 固定数组 + eviction 时压缩

第一版建议先做“固定数组 + NumPy compaction”。它不是最极致的版本，但收益大、风险低。

### 1. 初始化固定数组

在 `FastSampleEpisodeReplayBuffer.__init__()` 中分配数组：

```python
def __init__(
    self,
    *args,
    index_capacity: int | None = None,
    **kwargs,
) -> None:
    super().__init__(*args, **kwargs)

    self._index_capacity = index_capacity or self.capacity
    self._index_episode = np.empty(self._index_capacity, dtype=np.int64)
    self._index_timestep = np.empty(self._index_capacity, dtype=np.int32)
    self._num_indices = 0
```

这里 `index_capacity` 不一定必须等于 `capacity`，原因见后文“容量问题”。

### 2. add 时批量写入 NumPy 数组

原来的写法：

```python
self._indices.extend((eps_idx, i) for i in range(eps_len))
```

NumPy 版：

```python
def _append_indices_np(self, eps_idx: int, start_ts: int, length: int) -> None:
    end = self._num_indices + length

    if end > self._index_capacity:
        raise BufferError(
            f"Replay index capacity exceeded: need {end}, "
            f"capacity={self._index_capacity}"
        )

    write_slice = slice(self._num_indices, end)

    self._index_episode[write_slice] = eps_idx
    self._index_timestep[write_slice] = np.arange(
        start_ts,
        start_ts + length,
        dtype=self._index_timestep.dtype,
    )

    self._num_indices = end
```

新 episode：

```python
self._append_indices_np(eps_idx, 0, eps_len)
```

已有 episode 追加 fragment：

```python
old_len = len(existing_eps)
self._append_indices_np(eps_idx, old_len, eps_len)
```

这样新增 timestep 时不再为每个 timestep 创建 Python tuple。

### 3. fast sample 改成读 NumPy 数组

当前 `FastSampleEpisodeReplayBuffer._sample_episodes_fast_transition()` 中有：

```python
episode_abs_idx, episode_ts = self._indices[
    self.rng.integers(len(self._indices))
]
```

改为：

```python
slot = int(self.rng.integers(self._num_indices))

episode_abs_idx = int(self._index_episode[slot])
episode_ts = int(self._index_timestep[slot])
```

后续逻辑基本不变：

```python
episode_idx = episode_abs_idx - self._num_episodes_evicted
episode = self.episodes[episode_idx]
```

### 4. eviction 后压缩 NumPy 数组

原来的 rebuild 是创建新 list。

NumPy 版可以做压缩：

```python
def _rebuild_indices_batch_np(self, evicted_episode_indices: set[int]) -> None:
    live = self._num_indices

    if not evicted_episode_indices or live == 0:
        return

    episode_col = self._index_episode[:live]

    if len(evicted_episode_indices) == 1:
        evicted_idx = next(iter(evicted_episode_indices))
        keep = episode_col != evicted_idx
    else:
        evicted = np.fromiter(
            evicted_episode_indices,
            dtype=self._index_episode.dtype,
        )
        keep = ~np.isin(episode_col, evicted)

    new_size = int(np.count_nonzero(keep))

    self._index_episode[:new_size] = self._index_episode[:live][keep]
    self._index_timestep[:new_size] = self._index_timestep[:live][keep]

    self._num_indices = new_size
```

示例：

```text
_index_episode:  [0, 0, 1, 1, 2, 2]
_index_timestep: [0, 1, 0, 1, 0, 1]
```

evict episode `1` 后：

```text
keep:            [T, T, F, F, T, T]

_index_episode:  [0, 0, 2, 2, ...]
_index_timestep: [0, 1, 0, 1, ...]
_num_indices = 4
```

这个方案仍然需要扫描 `_num_indices` 个元素，但扫描主要在 NumPy/C 层完成，并且不会创建大量 Python tuple/list。

注意：它并不是“完全零临时内存”，因为 `keep` 是临时 boolean 数组，fancy indexing 也可能产生临时数组。但相比 Python list-of-tuples，GC 压力会低很多。

## 容量问题

不能简单认为：

```python
index_capacity = capacity
```

原因是 RLlib 当前 add 顺序大致是：

```text
先 append 新 episode indices
再 while 超过 capacity 时 evict 老 episode
```

假设：

```text
capacity = 1_000_000
当前 _num_indices = 1_000_000
新 episode 长度 = 1000
```

append 的瞬间需要写到 `1_001_000`，但 eviction 还没执行。如果 NumPy 数组长度正好是
`1_000_000`，会直接越界。

有三个处理方式。

### 选择 A：index_capacity = capacity + max_episode_len

如果训练中能估计单次 add 的最大长度，例如 rollout fragment 最多 1000：

```python
index_capacity = capacity + 1000
```

这是最接近当前 RLlib add 语义的方式。

### 选择 B：add 前先 evict 腾空间

在 append 前先判断是否需要 eviction：

```python
while (
    self._num_indices + eps_len > self._index_capacity
    and self.get_num_episodes() > 1
):
    evict_oldest_episode()
```

优点是更省内存；缺点是和 RLlib 原始 add 顺序略有差异。

### 选择 C：允许偶尔扩容

严格来说这不是固定内存，但工程上比较稳：

```python
def _ensure_index_capacity(self, required: int) -> None:
    if required <= self._index_capacity:
        return

    new_capacity = max(required, int(self._index_capacity * 1.5))

    new_episode = np.empty(new_capacity, dtype=self._index_episode.dtype)
    new_timestep = np.empty(new_capacity, dtype=self._index_timestep.dtype)

    new_episode[:self._num_indices] = self._index_episode[:self._num_indices]
    new_timestep[:self._num_indices] = self._index_timestep[:self._num_indices]

    self._index_episode = new_episode
    self._index_timestep = new_timestep
    self._index_capacity = new_capacity
```

推荐第一版使用 A 或 C：

- 如果环境 episode 长度可控，用 A。
- 如果 episode 长度不稳定，用 C。

## 和父类 `_indices` 的兼容问题

不能直接把父类的：

```python
self._indices = []
```

替换成 NumPy 数组，因为 RLlib 的父类方法假设 `_indices` 是 list-of-tuples：

- `_sample_episodes()` 会用 `self._indices[random_pos]`。
- `get_num_timesteps()` 返回 `len(self._indices)`。
- `get_state()` / `set_state()` 会保存和恢复 `_indices`。

因此第一版建议：

1. `FastSampleEpisodeReplayBuffer` 内部维护新的 NumPy index：

   ```python
   self._index_episode
   self._index_timestep
   self._num_indices
   ```

2. fast transition sample 主路径只读 NumPy index。
3. `_indices` 不再作为主路径数据结构。
4. 对必须依赖父类的 fallback sample，在 fallback 前临时 materialize `_indices`。

## fallback sample 处理

当前 fast path 只支持：

```text
n_step = 1
lookback = 0
batch_length_T = None
include_extra_model_outputs = False
to_numpy = False
min_batch_length_T = 0
```

其他模式会 fallback 到父类：

```python
return super()._sample_episodes(...)
```

父类需要 `_indices` 是 list-of-tuples，所以可以在 fallback 前临时生成：

```python
def _materialize_indices_for_fallback(self) -> None:
    self._indices = list(
        zip(
            self._index_episode[:self._num_indices].tolist(),
            self._index_timestep[:self._num_indices].tolist(),
        )
    )
```

然后：

```python
if not self._can_use_fast_transition_sample(...):
    self._materialize_indices_for_fallback()
    return super()._sample_episodes(...)
```

这意味着：

- 正常 SAC fast path 不再维护大 Python `_indices`。
- 如果经常 fallback，仍然会创建大 list，优化效果会变差。
- 如果训练路径固定走 fast transition sample，这个方案足够。

## checkpoint / state 处理

父类 `get_state()` 现在会保存：

```python
"_indices": self._indices
```

NumPy 版应该保存有效区间：

```python
def get_state(self) -> dict:
    return {
        "episodes": [eps.get_state() for eps in self.episodes],
        "episode_id_to_index": list(self.episode_id_to_index.items()),
        "_num_episodes_evicted": self._num_episodes_evicted,
        "_num_timesteps": self._num_timesteps,
        "_num_timesteps_added": self._num_timesteps_added,
        "sampled_timesteps": self.sampled_timesteps,
        "_index_episode": self._index_episode[:self._num_indices].copy(),
        "_index_timestep": self._index_timestep[:self._num_indices].copy(),
        "_num_indices": self._num_indices,
        "_index_capacity": self._index_capacity,
    }
```

恢复时：

```python
def set_state(self, state) -> None:
    self._set_episodes(state)
    self.episode_id_to_index = dict(state["episode_id_to_index"])
    self._num_episodes_evicted = state["_num_episodes_evicted"]
    self._num_timesteps = state["_num_timesteps"]
    self._num_timesteps_added = state["_num_timesteps_added"]
    self.sampled_timesteps = state["sampled_timesteps"]

    saved_num_indices = state["_num_indices"]
    saved_capacity = state.get("_index_capacity", saved_num_indices)

    self._index_capacity = max(saved_capacity, saved_num_indices, self.capacity)
    self._index_episode = np.empty(self._index_capacity, dtype=np.int64)
    self._index_timestep = np.empty(self._index_capacity, dtype=np.int32)

    self._index_episode[:saved_num_indices] = state["_index_episode"]
    self._index_timestep[:saved_num_indices] = state["_index_timestep"]
    self._num_indices = saved_num_indices
```

为了兼容旧 checkpoint，可以额外处理旧的 `_indices`：

```python
if "_indices" in state and "_index_episode" not in state:
    old_indices = state["_indices"]
    self._num_indices = len(old_indices)
    self._index_capacity = max(self.capacity, self._num_indices)
    self._index_episode = np.empty(self._index_capacity, dtype=np.int64)
    self._index_timestep = np.empty(self._index_capacity, dtype=np.int32)

    for i, (episode_abs_idx, episode_ts) in enumerate(old_indices):
        self._index_episode[i] = episode_abs_idx
        self._index_timestep[i] = episode_ts
```

## 更进一步：避免 eviction 全量扫描

上面的第一版主要解决 Python tuple/list 分配和 GC 压力，但 eviction 时仍然需要扫描整个 NumPy index。

如果 profiling 后发现瓶颈仍然在 eviction rebuild，可以进一步做 “live slots + swap-delete”。

核心结构：

```python
self._index_episode      # physical_slot -> episode_abs_idx
self._index_timestep     # physical_slot -> timestep
self._live_slots         # live_pos -> physical_slot
self._slot_to_live_pos   # physical_slot -> live_pos
self._episode_to_slots   # episode_abs_idx -> slots owned by this episode
self._free_slots         # reusable physical slots
self._num_live_slots
```

采样时先抽 live position：

```python
live_pos = int(self.rng.integers(self._num_live_slots))
slot = int(self._live_slots[live_pos])

episode_abs_idx = int(self._index_episode[slot])
episode_ts = int(self._index_timestep[slot])
```

删除一个 slot 时使用 swap-delete：

```python
def _remove_slot(self, slot: int) -> None:
    live_pos = self._slot_to_live_pos[slot]
    last_live_pos = self._num_live_slots - 1
    last_slot = self._live_slots[last_live_pos]

    self._live_slots[live_pos] = last_slot
    self._slot_to_live_pos[last_slot] = live_pos

    self._num_live_slots -= 1
    self._free_slots.append(slot)
```

evict 一个 episode 时：

```python
for slot in self._episode_to_slots[evicted_episode_idx]:
    self._remove_slot(slot)
```

优点：

- sample 仍然 O(1)。
- evict 成本接近 O(被驱逐 episode 的 timestep 数)。
- 不需要每次全量 rebuild/compact。

缺点：

- 实现复杂很多。
- 删除会改变 sample slot 顺序，随机种子下的采样 parity 可能和原实现不同。
- `episode_to_slots` 如果用 Python list，也会带来一些对象开销。
- checkpoint 要保存更多结构。

所以不建议第一版直接做这个。

## 建议落地顺序

1. 先实现 `NumPy fixed arrays + compaction`。
2. `FastSampleEpisodeReplayBuffer` fast path 改成读 NumPy index。
3. fallback sample 发生时临时 materialize `_indices`。
4. override `get_num_timesteps()` / `get_state()` / `set_state()`。
5. 加测试验证 add、evict、sample、state restore。
6. 跑 profiling，看瓶颈是否还在 NumPy compaction。
7. 如果 eviction 扫描仍是瓶颈，再考虑 live slots + swap-delete。

## 第一版需要覆盖的测试

建议至少加这些测试：

1. add 后 `_num_indices == get_num_timesteps()`。
2. fast sample 和原 `BatchEvictEpisodeReplayBuffer` 在同 seed 下语义一致。
3. eviction 后 NumPy index 不包含被驱逐 episode。
4. ongoing episode fragment 追加后，timestep index 从旧长度继续。
5. fallback sample 会 materialize `_indices` 并正常工作。
6. `get_state()` / `set_state()` 后可以继续 sample。
7. 旧 `_indices` checkpoint 可以迁移到 NumPy index。

## 总结

推荐方案是：

```text
主路径：
Python list-of-tuples _indices
    -> NumPy int64/int32 固定数组

新增：
    _index_episode
    _index_timestep
    _num_indices

fast sample：
    从 NumPy 数组随机读 slot

eviction：
    第一版用 NumPy boolean mask 压缩

fallback：
    临时 materialize 原始 _indices
```

第一版能显著降低 Python tuple/list 分配和 GC 压力，同时保持实现风险可控。等 profiling 证明 eviction 全量扫描仍然是主要瓶颈后，再升级到 live slots + swap-delete。
