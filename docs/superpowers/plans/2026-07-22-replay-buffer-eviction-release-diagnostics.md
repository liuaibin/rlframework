# Replay Buffer Eviction Release Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep `copy_episodes_on_add=False` zero-copy while proving whether evicted replay episodes release their containers and NumPy backing arrays.

**Architecture:** Add an opt-in, bounded weak-reference tracker to `BatchEvictEpisodeReplayBuffer`, inherited by both fast buffer variants. Record lookback containers, NumPy payload arrays, and ndarray base chains immediately before eviction drops its local reference; expose a small stats API without invoking GC or Ray internal free APIs.

**Tech Stack:** Python 3.10+, NumPy, Ray RLlib `SingleAgentEpisode`, pytest, weak references.

---

### Task 1: Add failing release-diagnostic tests

**Files:**
- Modify: `tests/test_utils.py`

- [ ] **Step 1: Add a NumPy-backed episode helper and failing tests**

Add this helper to `TestReplayBuffers` after `_make_episode`:

```python
    @staticmethod
    def _make_numpy_episode(episode_id: str, length: int = 2):
        import numpy as np
        from ray.rllib.env.single_agent_episode import SingleAgentEpisode

        return SingleAgentEpisode(
            id_=episode_id,
            observations=np.arange((length + 1) * 2, dtype=np.float32).reshape(
                length + 1, 2
            ),
            actions=np.arange(length * 2, dtype=np.float32).reshape(length, 2),
            rewards=np.ones(length, dtype=np.float32),
            terminated=True,
            len_lookback_buffer=0,
        )
```

Add these tests after `test_replay_buffer_can_skip_copying_owned_episodes`:

```python
    @pytest.mark.parametrize(
        "buffer_name",
        ["FastSampleEpisodeReplayBuffer", "NumpyIndexedFastSampleEpisodeReplayBuffer"],
    )
    def test_eviction_diagnostics_report_released_backing(self, buffer_name):
        import gc

        from rlframework.utils import replay_buffers

        buffer_type = getattr(replay_buffers, buffer_name)
        buffer = buffer_type(
            capacity=2,
            copy_episodes_on_add=False,
            track_evicted_episode_refs=True,
        )
        episode = self._make_numpy_episode("A")
        observation_backing = episode.observations.data
        buffer.add(episode)

        del observation_backing
        del episode
        buffer.add(self._make_numpy_episode("B"))
        gc.collect()

        stats = buffer.get_evicted_episode_release_stats()
        assert stats["tracked_container_refs"] >= 4
        assert stats["pending_container_refs"] == 0
        assert stats["tracked_array_refs"] >= 3
        assert stats["pending_array_refs"] == 0

    @pytest.mark.parametrize(
        "buffer_name",
        ["FastSampleEpisodeReplayBuffer", "NumpyIndexedFastSampleEpisodeReplayBuffer"],
    )
    def test_eviction_diagnostics_detect_sampled_view_retention(self, buffer_name):
        import gc

        import numpy as np

        from rlframework.utils import replay_buffers

        buffer_type = getattr(replay_buffers, buffer_name)
        buffer = buffer_type(
            capacity=2,
            copy_episodes_on_add=False,
            track_evicted_episode_refs=True,
        )
        episode = self._make_numpy_episode("A")
        observation_backing = episode.observations.data
        buffer.add(episode)
        buffer.rng = np.random.default_rng(123)
        sample = buffer.sample(
            sample_episodes=True,
            batch_size_B=1,
            batch_length_T=None,
            n_step=1,
            lookback=0,
        )
        assert np.shares_memory(
            sample[0].get_observations(0),
            observation_backing,
        )

        del observation_backing
        del episode
        buffer.add(self._make_numpy_episode("B"))
        gc.collect()

        retained_stats = buffer.get_evicted_episode_release_stats()
        assert retained_stats["pending_container_refs"] == 0
        assert retained_stats["pending_array_refs"] > 0

        del sample
        gc.collect()

        released_stats = buffer.get_evicted_episode_release_stats()
        assert released_stats["pending_array_refs"] == 0

    @pytest.mark.parametrize(
        "buffer_name",
        ["FastSampleEpisodeReplayBuffer", "NumpyIndexedFastSampleEpisodeReplayBuffer"],
    )
    def test_eviction_diagnostics_are_disabled_by_default(self, buffer_name):
        from rlframework.utils import replay_buffers

        buffer_type = getattr(replay_buffers, buffer_name)
        buffer = buffer_type(capacity=2, copy_episodes_on_add=False)
        buffer.add(self._make_numpy_episode("A"))
        buffer.add(self._make_numpy_episode("B"))

        assert buffer.get_evicted_episode_release_stats() == {
            "tracked_container_refs": 0,
            "pending_container_refs": 0,
            "tracked_array_refs": 0,
            "pending_array_refs": 0,
        }
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
.venv/bin/pytest tests/test_utils.py -k "eviction_diagnostics" -q
```

Expected: six failures because `track_evicted_episode_refs` is forwarded to RLlib's `EpisodeReplayBuffer.__init__`, which does not accept it, or because `get_evicted_episode_release_stats` does not exist.

### Task 2: Implement bounded weak-reference diagnostics

**Files:**
- Modify: `rlframework/utils/replay_buffers.py`
- Test: `tests/test_utils.py`

- [ ] **Step 1: Add imports and the diagnostic window constant**

Update the module imports to include:

```python
import weakref
from collections import deque
from collections.abc import Iterator, Mapping
```

Add below the replay-buffer imports:

```python
_EVICTED_REF_DIAGNOSTIC_LIMIT = 10_000
```

- [ ] **Step 2: Initialize opt-in tracker state**

Change `BatchEvictEpisodeReplayBuffer.__init__` to consume the diagnostic flag before calling RLlib:

```python
    def __init__(
        self,
        *args: Any,
        copy_episodes_on_add: bool = True,
        track_evicted_episode_refs: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.copy_episodes_on_add = copy_episodes_on_add
        self.track_evicted_episode_refs = track_evicted_episode_refs
        self._evicted_container_refs: deque[weakref.ReferenceType[Any]] | None = None
        self._evicted_array_refs: deque[weakref.ReferenceType[np.ndarray]] | None = None
        if track_evicted_episode_refs:
            self._evicted_container_refs = deque(maxlen=_EVICTED_REF_DIAGNOSTIC_LIMIT)
            self._evicted_array_refs = deque(maxlen=_EVICTED_REF_DIAGNOSTIC_LIMIT)
```

- [ ] **Step 3: Add payload traversal, tracking, and stats methods**

Add these methods to `BatchEvictEpisodeReplayBuffer` before `_copy_episode_for_add`:

```python
    @staticmethod
    def _iter_numpy_arrays(value: Any) -> Iterator[np.ndarray]:
        """Yield NumPy arrays from supported nested episode payloads."""
        stack = [value]
        seen: set[int] = set()
        while stack:
            current = stack.pop()
            current_id = id(current)
            if current_id in seen:
                continue
            seen.add(current_id)

            if isinstance(current, np.ndarray):
                yield current
            elif isinstance(current, InfiniteLookbackBuffer):
                stack.append(current.data)
            elif isinstance(current, Mapping):
                stack.extend(current.values())
            elif isinstance(current, (list, tuple, set, frozenset)):
                stack.extend(current)

    def _track_evicted_episode_references(self, eps: SingleAgentEpisode) -> None:
        """Record non-owning handles for an episode about to be evicted."""
        if self._evicted_container_refs is None or self._evicted_array_refs is None:
            return

        containers = [
            eps.observations,
            eps.actions,
            eps.rewards,
            eps.infos,
            *eps.extra_model_outputs.values(),
        ]
        seen_containers: set[int] = set()
        for container in containers:
            container_id = id(container)
            if container_id in seen_containers:
                continue
            seen_containers.add(container_id)
            try:
                self._evicted_container_refs.append(weakref.ref(container))
            except TypeError:
                continue

        seen_arrays: set[int] = set()
        for array in self._iter_numpy_arrays([containers, eps.custom_data]):
            current: Any = array
            while isinstance(current, np.ndarray):
                current_id = id(current)
                if current_id not in seen_arrays:
                    seen_arrays.add(current_id)
                    self._evicted_array_refs.append(weakref.ref(current))
                current = current.base

    def get_evicted_episode_release_stats(self) -> dict[str, int]:
        """Return live-reference counts from the bounded diagnostic windows."""
        if self._evicted_container_refs is None or self._evicted_array_refs is None:
            return {
                "tracked_container_refs": 0,
                "pending_container_refs": 0,
                "tracked_array_refs": 0,
                "pending_array_refs": 0,
            }
        return {
            "tracked_container_refs": len(self._evicted_container_refs),
            "pending_container_refs": sum(
                ref() is not None for ref in self._evicted_container_refs
            ),
            "tracked_array_refs": len(self._evicted_array_refs),
            "pending_array_refs": sum(ref() is not None for ref in self._evicted_array_refs),
        }
```

- [ ] **Step 4: Track and explicitly drop each evicted episode**

In both eviction loops, immediately after incrementing `_num_episodes_evicted`, add:

```python
                self._track_evicted_episode_references(evicted_eps)
                del evicted_eps
```

This must be applied to `BatchEvictEpisodeReplayBuffer.add` and `NumpyIndexedFastSampleEpisodeReplayBuffer.add` after their existing metrics and index bookkeeping have finished using the object.

- [ ] **Step 5: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/pytest tests/test_utils.py -k "eviction_diagnostics" -q
```

Expected: six passed tests. The retained-view tests must show zero pending containers and at least one pending backing array before deleting the sample.

- [ ] **Step 6: Run the complete replay-buffer test class**

Run:

```bash
.venv/bin/pytest tests/test_utils.py::TestReplayBuffers -q
```

Expected: all replay-buffer tests pass.

- [ ] **Step 7: Commit the tested implementation**

```bash
git add rlframework/utils/replay_buffers.py tests/test_utils.py
git commit -m "feat: diagnose replay episode release after eviction"
```

### Task 3: Document and verify the runtime diagnostic

**Files:**
- Modify: `docs/ReplayBuffer_memory_profiling.md`
- Verify: `rlframework/utils/replay_buffers.py`
- Verify: `tests/test_utils.py`

- [ ] **Step 1: Document configuration and interpretation**

Append this section to `docs/ReplayBuffer_memory_profiling.md`:

````markdown
## 检查 eviction 后的引用释放

使用零拷贝写入时，可临时启用弱引用诊断：

```python
replay_buffer_config = {
    "type": NumpyIndexedFastSampleEpisodeReplayBuffer,
    "capacity": 1_000_000,
    "copy_episodes_on_add": False,
    "track_evicted_episode_refs": True,
}
```

读取统计：

```python
stats = algo.local_replay_buffer.get_evicted_episode_release_stats()
print(stats)
```

- `pending_container_refs > 0`：episode 或其 lookback container 仍被外部代码持有。
- `pending_container_refs == 0` 且 `pending_array_refs > 0`：通常是 sampled NumPy view 或 Learner in-flight batch 仍持有底层数组。
- 两个 pending 都为 0：Python/NumPy 引用已释放；若 RSS 不下降，应继续检查 Ray Plasma 的 used/pinned bytes，而不是调用 `gc.collect()`。

该诊断最多保留最近 10,000 个 container 和 array 弱引用，默认关闭。它不会复制 episode，不会强制释放 Ray 对象，也不会在训练热路径调用垃圾回收。
````

- [ ] **Step 2: Run formatting, lint, and type checks**

Run:

```bash
uv run ruff format rlframework/utils/replay_buffers.py tests/test_utils.py
uv run ruff check rlframework/utils/replay_buffers.py tests/test_utils.py
uv run mypy rlframework/utils/replay_buffers.py
```

Expected: all commands exit zero without diagnostics.

- [ ] **Step 3: Run the broader utility test file**

Run:

```bash
.venv/bin/pytest tests/test_utils.py -q
```

Expected: all tests in `tests/test_utils.py` pass.

- [ ] **Step 4: Review the final diff and commit documentation**

Run:

```bash
git diff --check
git status --short
```

Confirm only the planned implementation, tests, documentation, and plan are changed; preserve the pre-existing untracked `docs/codex-skills-guide.md`.

Commit:

```bash
git add docs/ReplayBuffer_memory_profiling.md docs/superpowers/plans/2026-07-22-replay-buffer-eviction-release-diagnostics.md
git commit -m "docs: explain replay eviction release diagnostics"
```
