# ReplayBuffer Snapshot Async Design

本文记录一个 ReplayBuffer 内部异步化方案：用快照（snapshot / RCU / MVCC 思路）把 `add/evict` 和 `sample` 解耦，让写入路径和采样路径可以并行推进，同时避免采样读到正在被修改的索引结构。

## 背景

当前 ReplayBuffer 的关键瓶颈在于：

- `add()` 会追加 episode、维护 timestep 级采样索引，并在容量超限时 evict 旧 episode。
- `sample()` 会从索引结构中随机抽取 timestep，再 materialize 成训练 batch。
- 如果 `add/evict` 和 `sample` 直接并发访问同一套可变结构，很容易出现索引错位、采样到已删除 slot、slot 复用后 handle 指向错误样本等问题。

一种保守做法是 ReplayBuffer actor 化：对外 `add.remote()` / `sample.remote()` 异步，actor 内部串行执行。这能把 ReplayBuffer 从 driver 中移出去，但单个 ReplayBuffer 内部仍然是串行的。

快照方案的目标是更进一步：让同一个 ReplayBuffer 内部的写入和采样也能重叠执行。

## 核心思想

将 ReplayBuffer 拆成两层：

```text
Writer State：唯一写入者，负责 add / evict / slot 复用 / generation 更新
Read Snapshot：只读快照，负责 sample
```

语义变化是：

```text
sample 读的是某个已发布快照，不一定看到最新 add。
add 写的是 mutable state，提交后再发布新快照。
```

对于 SAC 这类 off-policy replay，sample 稍微滞后通常可以接受。

## Writer State

Writer 持有完整可变状态，例如：

```text
episodes
episode_id_to_index
_index_episode
_index_timestep
_live_slots
_slot_to_live_pos
_free_slots
_episode_to_slots
_slot_generation
_num_live_slots
_num_timesteps
epoch
```

其中 `live_slots + slot_to_live_pos + free_slots` 可以使用 swap-delete 方式维护有效 sample slots，避免 eviction 时全量 rebuild/compact。

## Read Snapshot

Sampler 不直接读 writer state，而是只读已发布 snapshot：

```text
ReplaySnapshot {
  epoch
  num_live_slots
  live_slots 或 compact sample index
  index_episode
  index_timestep
  slot_generation
  episode_refs / episode table
}
```

采样时只访问 snapshot 内部数据：

```text
snapshot = current_snapshot
positions = rng.integers(0, snapshot.num_live_slots, size=batch_size)
slots = snapshot.live_slots[positions]
episode_idx = snapshot.index_episode[slots]
timestep = snapshot.index_timestep[slots]
materialize batch
```

只要 snapshot 不变，多个 sampler 就可以并发读取。

## Add / Evict 流程

Writer 的 `add_batch` 流程：

```text
add_batch(episodes)
  -> 写入 writer state
  -> append slots
  -> evict old episodes if capacity exceeded
  -> swap-delete remove evicted slots
  -> update slot_generation
  -> epoch += 1
  -> publish_snapshot_if_needed()
```

Evict 一个 episode 时：

```text
for slot in episode_to_slots[evicted_episode_idx]:
    remove_slot_by_swap_delete(slot)
    slot_generation[slot] += 1
    free_slots.append(slot)
```

这样 eviction 成本接近 `O(被驱逐 episode 的 timestep 数)`，而不是 `O(整个 replay index 大小)`。

## Sample 流程

Sampler 的 `sample` 流程：

```text
snapshot = current_snapshot
positions = rng.integers(0, snapshot.num_live_slots, size=batch_size)
slots = snapshot.live_slots[positions]
episode_idx = snapshot.index_episode[slots]
timestep = snapshot.index_timestep[slots]
materialize batch from snapshot episode table
return batch, optional sample_handle
```

如果未来支持 prioritized replay，sample 需要返回 handle：

```text
SampleHandle {
  epoch
  slots
  generations
  episode_indices
  timesteps
  replay_version
}
```

Priority update 时检查 generation，避免 stale update 写到复用后的 slot：

```text
if current_slot_generation[slot] == sampled_generation:
    update_priority(slot, td_error)
else:
    drop_stale_priority_update()
```

Uniform replay 第一版可以不使用 handle，但建议预留 generation 设计。

## 快照安全性

不能只快照 `live_slots`，然后继续读 writer 的 `_index_episode` / `_index_timestep` / `episodes`。

错误场景：

```text
snapshot 中包含 slot=10
writer evict slot=10
writer 将 slot=10 放入 free list
writer 复用 slot=10 写入新 episode
sampler 读旧 snapshot 的 slot=10，却拿到新 episode 数据
```

因此需要下面两类安全方案之一。

## 方案 A：Compact Snapshot

发布快照时，把采样需要的数据压成独立只读数组：

```text
snapshot_episode_idx = _index_episode[live_slots[:num_live_slots]].copy()
snapshot_timestep    = _index_timestep[live_slots[:num_live_slots]].copy()
snapshot_generation  = _slot_generation[live_slots[:num_live_slots]].copy()
snapshot_episode_refs = current episode table / immutable episode references
```

sample 随机的是 snapshot 内部位置：

```text
pos -> snapshot_episode_idx[pos], snapshot_timestep[pos]
```

优点：

- 实现简单，安全边界清晰。
- sample 完全不碰 writer 的 slot arrays。
- slot 被复用也不影响旧 snapshot。

缺点：

- publish snapshot 时需要复制 live index。
- replay 很大时不能每次 add 都 publish，否则 copy 成本太高。

这是推荐的第一版。

## 方案 B：RCU No-Reuse

Snapshot 可以只持有 `live_slots`，但 writer 必须保证：只要某个 snapshot 可能还在被 reader 使用，就不能复用该 snapshot 中可能引用的 slot。

需要维护：

```text
reader_epoch / snapshot refcount
retired_slots[epoch]
free_slots
```

流程：

```text
evict slot -> retired_slots[current_epoch]
等所有 reader 都离开旧 epoch
retired_slots -> free_slots
```

优点：

- snapshot publish 成本低。
- 避免频繁复制大数组。

缺点：

- 实现复杂。
- free slot 不能立刻复用。
- 需要 reader epoch、引用计数、retired list 回收。
- checkpoint 和恢复逻辑更复杂。

该方案适合 compact snapshot 被证明成本过高之后再考虑。

## Episode 对象的可变性

只快照索引还不够，episode 对象本身也可能是 mutable 的。

例如 writer 可能执行：

```text
existing_eps.concat_episode(eps)
```

如果 snapshot 持有 `existing_eps` 引用，而 writer 同时修改它，sampler 就可能读到中间态。

可选解决方案：

1. **sample 时 materialize/copy**  
   sampler 从 snapshot 立即构造训练 batch，不把可变 episode 引用长期交给外部。

2. **episode copy-on-write**  
   如果要 concat 一个已发布到 snapshot 的 episode，不原地修改；创建新 episode version，writer state 指向新版本，旧 snapshot 继续引用旧版本。

3. **chunked episode**  
   episode 由 immutable chunks 组成，新 chunk append 后发布新 episode view。长期最干净，但实现复杂。

第一版建议采用 compact snapshot + sample materialization，必要时再引入 copy-on-write。

## Snapshot 发布频率

不要每个 episode add 都发布快照。可以按条件发布：

```text
每新增 K 个 env steps
或者每 M ms
或者每次 evict 后
或者 sample queue 快空时
```

示例策略：

```text
snapshot_publish_min_steps = 1024
snapshot_publish_interval_ms = 20
```

需要监控：

```text
writer_epoch
snapshot_epoch
snapshot_lag_steps
snapshot_lag_ms
sample_from_snapshot_size
snapshot_publish_time
snapshot_copy_bytes
```

## Ray Actor 实现选项

Ray actor 默认是串行执行的。如果要内部异步，有两种路线。

### 单 Actor + Concurrency Groups

```text
ReplayBufferActor
  writer group: max_concurrency=1
  reader group: max_concurrency=N
```

- `add_batch` 走 writer group。
- `sample` 走 reader group。
- reader 只读 immutable snapshot。
- writer 只修改 writer state，并原子替换 `current_snapshot`。

优点是部署简单；缺点是 Python 进程/GIL 和 actor 内部调度仍可能限制 CPU 并行。

### Writer Actor + Sampler Actor(s)

```text
ReplayWriterActor
  -> publish SnapshotRef

ReplaySamplerActor[0..N]
  -> consume latest SnapshotRef
  -> sample batches
```

Writer 定期发布 snapshot object ref，Sampler actors 更新本地 snapshot 后并行 sample。

优点：

- writer 和 sampler 是不同进程，CPU 并行更真实。
- sampler 可以水平扩展。
- driver 可以退出 replay data plane。

缺点：

- snapshot 分发、版本管理和 metrics 更复杂。
- snapshot 太大时 object store 压力需要监控。

长期更推荐该路线。

## Sample Prefetch

可以让 sampler 维护 bounded ready queue：

```text
background_sampler_loop:
  while ready_queue not full:
      snapshot = current_snapshot
      batch = sample_from(snapshot)
      ready_queue.put(batch)
```

Learner 获取 batch 时优先从 ready queue 中取，降低 sample 延迟。

需要限制：

```text
sample_prefetch_batches
max_sample_age_ms
max_sample_epoch_lag
```

避免 learner 消费过旧 batch。

## 一致性语义

建议明确采用 snapshot consistency：

1. `add_batch` 返回前，数据已进入 writer state。
2. `sample` 只从已发布 snapshot 中采样。
3. `add_batch` 完成后，新数据不保证立刻被下一次 `sample` 看到。
4. 新数据在下一次 snapshot publish 后对 sampler 可见。
5. 如果需要 read-after-add，需要等待对应 snapshot epoch 发布。

对于 off-policy replay，这种滞后一般可接受。

## Checkpoint

Checkpoint 需要保存：

```text
writer state:
  episodes
  episode_id_to_index
  _index_episode
  _index_timestep
  _live_slots
  _slot_to_live_pos
  _free_slots
  _episode_to_slots
  _slot_generation
  _num_live_slots
  _num_timesteps
  epoch

snapshot metadata:
  latest_published_epoch
  publish policy counters
```

恢复后可以从 writer state 重建最新 snapshot，不一定要把 snapshot 本体写入 checkpoint。

## Metrics

需要新增指标：

```text
replay_writer_add_time
replay_writer_evict_time
replay_snapshot_publish_time
replay_snapshot_copy_bytes
replay_snapshot_lag_steps
replay_snapshot_lag_ms
replay_sample_time
replay_sample_queue_depth
replay_stale_priority_updates
replay_dropped_stale_priority_updates
replay_num_live_slots
replay_num_free_slots
replay_num_retired_slots
```

这些指标用于判断瓶颈到底在 writer、snapshot publish、sample materialization，还是 object store / queue。

## 演进路线

推荐分阶段实现：

1. **ReplayBuffer actor 化，对外异步，内部串行**  
   风险最低，先把 driver 从 replay add/sample 高频路径中解放出来。

2. **单 writer + compact snapshot + sample prefetch**  
   Writer 修改 mutable state，Sampler 从 immutable snapshot 采样。`add/evict` 和 `sample` 可以重叠。

3. **多 Sampler actor**  
   Writer 发布 snapshot，多个 sampler 并行 materialize batch。

4. **多 ReplayShard**  
   每个 shard 一个 writer，自有 snapshot 和 sampler。Learner 从多个 shard 拉 batch，真正横向扩展。

5. **Prioritized replay / priority update**  
   引入 sample handle、slot generation、stale update drop metrics。必要时再设计跨 shard priority mass。

## 结论

可以用快照把 ReplayBuffer 内部从纯串行改成读写异步。关键原则是：

```text
add/evict 只修改 writer state；sample 只读 immutable snapshot。
```

第一版推荐 compact snapshot：虽然发布快照需要复制索引，但实现简单、安全边界清楚。只有当 snapshot copy 成本被 profiling 证明是瓶颈后，再考虑 RCU no-reuse 或更复杂的分片方案。
