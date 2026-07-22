# Replay Buffer Eviction Release Diagnostics Design

## Goal

Keep `copy_episodes_on_add=False` so replay insertion remains zero-copy, while proving whether evicted episodes and their Ray-backed NumPy storage are released after `deque.popleft()`.

The change must not copy sampled transitions, call `gc.collect()` on the training hot path, or depend on Ray's deprecated internal object-freeing APIs.

## Background

Remote EnvRunners return episode batches through `ray.put()`. The algorithm calls `ray.get()` and receives NumPy arrays whose base chain may end in a Ray `SubBuffer`. With `copy_episodes_on_add=False`, the replay buffer owns those deserialized episode objects directly.

Eviction already removes the replay buffer's strong reference with `popleft()`. Memory can nevertheless remain pinned when another reference survives, for example a sampled NumPy view held by an in-flight Learner update. Plasma memory may also remain reserved for reuse after an object becomes reclaimable, so process RSS alone is not evidence of a leak.

## Design

### Eviction behavior

Both `BatchEvictEpisodeReplayBuffer` and `NumpyIndexedFastSampleEpisodeReplayBuffer` will keep their current capacity and eviction semantics. After all metrics and index bookkeeping for an evicted episode are complete, the local `evicted_eps` reference will be deleted explicitly.

The implementation will not clear episode payloads destructively. Callers that violate the ownership contract by retaining the input episode will therefore not observe an emptied object.

The implementation will not call `gc.collect()`. `SingleAgentEpisode` and NumPy arrays normally use CPython reference counting, and a full collection per eviction would add unpredictable training pauses without releasing live views.

### Optional reference diagnostics

The replay-buffer constructor will accept an opt-in diagnostic flag:

```python
track_evicted_episode_refs: bool = False
```

When disabled, the hot path will only perform the explicit local-reference deletion and will not create weak references or walk episode payloads.

When enabled, the buffer will record weak references immediately before dropping an evicted episode. `SingleAgentEpisode` uses `__slots__` without `__weakref__` in Ray 2.49.2, so the episode object itself cannot be weak-referenced. The diagnostics will instead track:

- the episode's `InfiniteLookbackBuffer` containers for observations, actions, rewards, infos, and extra model outputs as proxies for episode-container lifetime;
- NumPy arrays in observations, actions, rewards, infos, extra model outputs, and custom data;
- NumPy arrays encountered through each tracked array's `.base` chain, so a sampled view retaining a root array is visible.

Diagnostics will keep separate bounded windows containing the most recent 10,000 container weak references and 10,000 array weak references. Reaching a bound drops the oldest diagnostic handle only; it never changes episode lifetime.

The buffer will expose:

```python
get_evicted_episode_release_stats() -> dict[str, int]
```

The result will contain the current diagnostic window counts:

```python
{
    "tracked_container_refs": int,
    "pending_container_refs": int,
    "tracked_array_refs": int,
    "pending_array_refs": int,
}
```

The tracked counts are the number of weak-reference handles currently present in each bounded window. The pending counts are the handles in that window whose referents remain alive. Dead handles remain in the window until displaced by newer evictions, allowing one call to show both tracked and released objects without retaining those objects. Pending containers mean some Python code still owns the episode or its lookback buffers. Pending arrays with no pending containers mean downstream NumPy views still own the episode's backing storage.

This API reports Python and NumPy ownership only. Ray Dashboard or `ray memory` remains the source of truth for Plasma used/pinned bytes.

### Manual freeing policy

The replay buffer will not call `ray._private.internal_api.free()` or other internal deletion APIs. The buffer does not retain the original batch `ObjectRef`, those APIs are deprecated, and Ray cannot safely free an object while a zero-copy NumPy view is live.

If diagnostics show pending arrays, the correct remediation is to remove the owning sample, Learner in-flight request, callback, or metrics reference. If no weak references remain but RSS stays high, the memory is considered allocator or Plasma reservation until Ray reports that used/pinned object-store bytes remain high.

## Tests

Tests will cover both `FastSampleEpisodeReplayBuffer` and `NumpyIndexedFastSampleEpisodeReplayBuffer` with `copy_episodes_on_add=False`.

1. Evicting an episode with no downstream sample releases its lookback containers and NumPy backing arrays after caller references are removed.
2. Retaining a sampled transition after eviction leaves the source backing array pending.
3. Deleting the sampled transition releases the pending backing array.
4. Diagnostics disabled by default create no tracked references and preserve existing buffer behavior.
5. Existing state, sampling parity, and eviction tests continue to pass.

The tests will use weak references and `gc.collect()` only in test code to make assertions deterministic. Production eviction will never trigger a collection.

## Success Criteria

- `copy_episodes_on_add=False` remains a true zero-copy add path.
- Eviction removes the replay buffer's last local episode reference as soon as its bookkeeping completes.
- Tests distinguish normal release from downstream-view retention.
- Runtime diagnostics are opt-in and bounded.
- No deprecated Ray free API, destructive payload clearing, or production hot-path garbage collection is introduced.
