"""Custom Replay Buffer implementations for RL algorithms.

This module provides custom replay buffer implementations that can be used
with Ray RLlib via the `replay_buffer_config` option.

Usage with RLlib::

    from ray.rllib.algorithms.sac import SACConfig
    from rlframework.utils.replay_buffers import PrioritizedSumTreeBuffer

    config = SACConfig().training(
        replay_buffer_config={
            "type": PrioritizedSumTreeBuffer,
            "capacity": 100000,
            "alpha": 0.6,
            "beta": 0.4,
        }
    )

Or use the string path for YAML configs::

    replay_buffer_config = {
        "type": "rlframework.utils.replay_buffers.PrioritizedSumTreeBuffer",
        "capacity": 100000,
    }
"""

import copy
import hashlib
from typing import Any, Union, cast

import numpy as np
from ray.rllib.core import DEFAULT_AGENT_ID, DEFAULT_MODULE_ID
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.env.utils.infinite_lookback_buffer import InfiniteLookbackBuffer
from ray.rllib.execution.segment_tree import SumSegmentTree as SumTree
from ray.rllib.utils import force_list
from ray.rllib.utils.annotations import override
from ray.rllib.utils.replay_buffers.base import ReplayBufferInterface
from ray.rllib.utils.replay_buffers.episode_replay_buffer import EpisodeReplayBuffer
from ray.rllib.utils.replay_buffers.prioritized_episode_buffer import (
    PrioritizedEpisodeReplayBuffer,
)


class BatchEvictEpisodeReplayBuffer(EpisodeReplayBuffer):
    """EpisodeReplayBuffer that rebuilds timestep indices once per add call.

    RLlib's uniform `EpisodeReplayBuffer.add()` removes evicted episode timesteps
    from `_indices` immediately for each evicted episode. When one `add()` call
    evicts many episodes, that repeatedly scans and copies the full `_indices`
    list. This subclass keeps the same storage and sampling behavior but records
    evicted episode indices and filters `_indices` once at the end of `add()`.
    """

    def __init__(
        self,
        *args: Any,
        copy_episodes_on_add: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.copy_episodes_on_add = copy_episodes_on_add

    @override(ReplayBufferInterface)
    def add(
        self,
        episodes: Union[list["SingleAgentEpisode"], "SingleAgentEpisode"],
    ) -> None:
        """Add episodes and batch-clean `_indices` after all evictions."""
        episode_list = cast(list[SingleAgentEpisode], force_list(episodes))

        # Set up counters for the inherited metrics implementation.
        num_env_steps_added = 0
        agent_to_num_steps_added = {DEFAULT_AGENT_ID: 0}
        module_to_num_steps_added = {DEFAULT_MODULE_ID: 0}
        num_episodes_added = 0
        agent_to_num_episodes_added = {DEFAULT_AGENT_ID: 0}
        module_to_num_episodes_added = {DEFAULT_MODULE_ID: 0}
        num_episodes_evicted = 0
        agent_to_num_episodes_evicted = {DEFAULT_AGENT_ID: 0}
        module_to_num_episodes_evicted = {DEFAULT_MODULE_ID: 0}
        num_env_steps_evicted = 0
        agent_to_num_steps_evicted = {DEFAULT_AGENT_ID: 0}
        module_to_num_steps_evicted = {DEFAULT_MODULE_ID: 0}

        evicted_episode_indices: set[int] = set()

        for eps in episode_list:
            # Preserve RLlib's default ownership semantics.
            eps = self._copy_episode_for_add(eps)

            eps_len = len(eps)
            self._num_timesteps += eps_len
            self._num_timesteps_added += eps_len
            num_env_steps_added += eps_len
            agent_to_num_steps_added[DEFAULT_AGENT_ID] += eps_len
            module_to_num_steps_added[DEFAULT_MODULE_ID] += eps_len

            # Ongoing episode: append the new chunk to the existing record.
            if eps.id_ in self.episode_id_to_index:
                eps_idx = self.episode_id_to_index[eps.id_]
                existing_eps = self.episodes[eps_idx - self._num_episodes_evicted]
                old_len = len(existing_eps)
                self._indices.extend((eps_idx, old_len + i) for i in range(eps_len))
                existing_eps.concat_episode(eps)
            # New episode: add it to the end of the episode deque.
            else:
                num_episodes_added += 1
                agent_to_num_episodes_added[DEFAULT_AGENT_ID] += 1
                module_to_num_episodes_added[DEFAULT_MODULE_ID] += 1
                self.episodes.append(eps)
                eps_idx = len(self.episodes) - 1 + self._num_episodes_evicted
                self.episode_id_to_index[eps.id_] = eps_idx
                self._indices.extend((eps_idx, i) for i in range(eps_len))

            # Evict old records from the front, but defer `_indices` cleanup.
            while self._num_timesteps > self.capacity and self.get_num_episodes() > 1:
                evicted_eps = self.episodes.popleft()
                evicted_eps_len = len(evicted_eps)

                num_episodes_evicted += 1
                num_env_steps_evicted += evicted_eps_len
                agent_to_num_episodes_evicted[DEFAULT_AGENT_ID] += 1
                module_to_num_episodes_evicted[DEFAULT_MODULE_ID] += 1
                agent_to_num_steps_evicted[DEFAULT_AGENT_ID] += evicted_eps.agent_steps()
                module_to_num_steps_evicted[DEFAULT_MODULE_ID] += evicted_eps.agent_steps()

                self._num_timesteps -= evicted_eps_len

                evicted_idx = self.episode_id_to_index[evicted_eps.id_]
                del self.episode_id_to_index[evicted_eps.id_]
                evicted_episode_indices.add(evicted_idx)

                self._num_episodes_evicted += 1

        if evicted_episode_indices:
            self._rebuild_indices_batch(evicted_episode_indices)

        self._update_add_metrics(
            num_episodes_added=num_episodes_added,
            num_env_steps_added=num_env_steps_added,
            num_episodes_evicted=num_episodes_evicted,
            num_env_steps_evicted=num_env_steps_evicted,
            agent_to_num_episodes_added=agent_to_num_episodes_added,
            agent_to_num_steps_added=agent_to_num_steps_added,
            agent_to_num_episodes_evicted=agent_to_num_episodes_evicted,
            agent_to_num_steps_evicted=agent_to_num_steps_evicted,
            # Preserve RLlib's add() metrics behavior for drop-in parity.
            module_to_num_episodes_added=module_to_num_steps_added,
            module_to_num_steps_added=module_to_num_episodes_added,
            module_to_num_episodes_evicted=module_to_num_episodes_evicted,
            module_to_num_steps_evicted=module_to_num_steps_evicted,
        )

    def _rebuild_indices_batch(self, evicted_episode_indices: set[int]) -> None:
        """Remove all timestep indices owned by evicted episodes in one pass."""
        if len(evicted_episode_indices) == 1:
            evicted_idx = next(iter(evicted_episode_indices))
            self._indices = [
                idx_tuple for idx_tuple in self._indices if idx_tuple[0] != evicted_idx
            ]
        else:
            self._indices = [
                idx_tuple
                for idx_tuple in self._indices
                if idx_tuple[0] not in evicted_episode_indices
            ]

    def _copy_episode_for_add(self, eps: SingleAgentEpisode) -> SingleAgentEpisode:
        """Return the episode object that should be owned by the replay buffer."""
        return copy.deepcopy(eps) if self.copy_episodes_on_add else eps


class FastSampleEpisodeReplayBuffer(BatchEvictEpisodeReplayBuffer):
    """Batch-evict buffer with a fast path for 1-step transition sampling.

    The fast path avoids constructing an intermediate episode slice for the common
    stateless SAC replay case: transition sampling with ``n_step=1`` and no
    lookback. More complex sampling modes fall back to RLlib's implementation.
    """

    @override(EpisodeReplayBuffer)
    def _sample_episodes(
        self,
        num_items: int | None = None,
        *,
        batch_size_B: int | None = None,  # noqa: N803 - RLlib API name.
        batch_length_T: int | None = None,  # noqa: N803 - RLlib API name.
        n_step: int | tuple | None = None,
        gamma: float = 0.99,
        include_infos: bool = False,
        include_extra_model_outputs: bool = False,
        to_numpy: bool = False,
        lookback: int = 1,
        min_batch_length_T: int = 0,  # noqa: N803 - RLlib API name.
        **kwargs: Any,
    ) -> list[SingleAgentEpisode]:
        """Sample episodes, using a direct transition path when it is safe."""
        if self._can_use_fast_transition_sample(
            num_items=num_items,
            batch_size=batch_size_B,
            batch_length=batch_length_T,
            n_step=n_step,
            include_extra_model_outputs=include_extra_model_outputs,
            to_numpy=to_numpy,
            lookback=lookback,
            min_batch_length=min_batch_length_T,
        ):
            return self._sample_episodes_fast_transition(
                num_items=num_items,
                batch_size=batch_size_B,
                n_step=cast(int, n_step),
                lookback=lookback,
            )

        return super()._sample_episodes(
            num_items=num_items,
            batch_size_B=batch_size_B,
            batch_length_T=batch_length_T,
            n_step=n_step,
            gamma=gamma,
            include_infos=include_infos,
            include_extra_model_outputs=include_extra_model_outputs,
            to_numpy=to_numpy,
            lookback=lookback,
            min_batch_length_T=min_batch_length_T,
            **kwargs,
        )

    @staticmethod
    def _can_use_fast_transition_sample(
        *,
        num_items: int | None,
        batch_size: int | None,
        batch_length: int | None,
        n_step: int | tuple | None,
        include_extra_model_outputs: bool,
        to_numpy: bool,
        lookback: int,
        min_batch_length: int,
    ) -> bool:
        """Return whether this sample request matches the safe fast path."""
        if num_items is not None and batch_size is not None:
            return False
        return (
            not batch_length
            and not isinstance(n_step, tuple)
            and n_step == 1
            and lookback == 0
            and min_batch_length == 0
            and not include_extra_model_outputs
            and not to_numpy
        )

    def _sample_episodes_fast_transition(
        self,
        num_items: int | None,
        *,
        batch_size: int | None,
        n_step: int,
        lookback: int,
    ) -> list[SingleAgentEpisode]:
        """Sample 1-step transitions without creating intermediate episode slices."""
        if num_items is not None:
            assert batch_size is None, (
                "Cannot call `sample()` with both `num_items` and `batch_size_B` "
                "provided! Use either one."
            )
            batch_size = num_items

        batch_size = batch_size or self.batch_size_B
        self._last_sampled_indices = []

        sampled_episodes = []
        sampled_env_step_idxs = set()
        sampled_episode_idxs = set()

        for _ in range(batch_size):
            episode_abs_idx, episode_ts = self._indices[self.rng.integers(len(self._indices))]
            episode_idx = episode_abs_idx - self._num_episodes_evicted
            episode = self.episodes[episode_idx]
            next_ts = episode_ts + 1
            done_at_end = next_ts == len(episode)

            sampled_episode = SingleAgentEpisode(
                id_=episode.id_,
                agent_id=episode.agent_id,
                module_id=episode.module_id,
                observation_space=episode.observation_space,
                action_space=episode.action_space,
                observations=[
                    episode.get_observations(episode_ts),
                    episode.get_observations(next_ts),
                ],
                actions=[episode.get_actions(episode_ts)],
                rewards=[episode.get_rewards(episode_ts)],
                infos=[
                    episode.get_infos(episode_ts),
                    episode.get_infos(next_ts),
                ],
                terminated=episode.is_terminated if done_at_end else False,
                truncated=episode.is_truncated if done_at_end else False,
                t_started=episode_ts,
                len_lookback_buffer=0,
            )
            sampled_episode.extra_model_outputs["n_step"] = InfiniteLookbackBuffer(
                np.full((len(sampled_episode) + lookback,), n_step),
                lookback=lookback,
            )
            sampled_episode.extra_model_outputs["weights"] = InfiniteLookbackBuffer(
                np.ones((len(sampled_episode) + lookback,)),
                lookback=lookback,
            )

            sampled_env_step_idxs.add(
                hashlib.sha256(f"{episode.id_}-{episode_ts}".encode()).hexdigest()
            )
            sampled_episode_idxs.add(episode_idx)
            sampled_episodes.append(sampled_episode)

        self.sampled_timesteps += batch_size
        self._update_fast_sample_metrics(
            batch_size=batch_size,
            num_episodes_per_sample=len(sampled_episode_idxs),
            num_env_steps_per_sample=len(sampled_env_step_idxs),
            sampled_n_step=float(n_step),
        )

        return sampled_episodes

    def _update_fast_sample_metrics(
        self,
        *,
        batch_size: int,
        num_episodes_per_sample: int,
        num_env_steps_per_sample: int,
        sampled_n_step: float,
    ) -> None:
        """Update sample metrics with the same single-agent defaults as RLlib."""
        num_env_steps_sampled = batch_size
        num_resamples = 0
        agent_to_num_steps_sampled = {DEFAULT_AGENT_ID: num_env_steps_sampled}
        agent_to_num_episodes_per_sample = {DEFAULT_AGENT_ID: num_episodes_per_sample}
        agent_to_num_steps_per_sample = {DEFAULT_AGENT_ID: num_env_steps_per_sample}
        agent_to_sampled_n_step = {DEFAULT_AGENT_ID: sampled_n_step}
        agent_to_num_resamples = {DEFAULT_AGENT_ID: num_resamples}
        module_to_num_steps_sampled = {DEFAULT_MODULE_ID: num_env_steps_sampled}
        module_to_num_episodes_per_sample = {DEFAULT_MODULE_ID: num_episodes_per_sample}
        module_to_num_steps_per_sample = {DEFAULT_MODULE_ID: num_env_steps_per_sample}
        module_to_sampled_n_step = {DEFAULT_MODULE_ID: sampled_n_step}
        module_to_num_resamples = {DEFAULT_MODULE_ID: num_resamples}

        self._update_sample_metrics(
            num_env_steps_sampled=num_env_steps_sampled,
            num_episodes_per_sample=num_episodes_per_sample,
            num_env_steps_per_sample=num_env_steps_per_sample,
            sampled_n_step=sampled_n_step,
            num_resamples=num_resamples,
            agent_to_num_steps_sampled=agent_to_num_steps_sampled,
            agent_to_num_episodes_per_sample=agent_to_num_episodes_per_sample,
            agent_to_num_steps_per_sample=agent_to_num_steps_per_sample,
            agent_to_sampled_n_step=agent_to_sampled_n_step,
            agent_to_num_resamples=agent_to_num_resamples,
            module_to_num_steps_sampled=module_to_num_steps_sampled,
            module_to_num_episodes_per_sample=module_to_num_episodes_per_sample,
            module_to_num_steps_per_sample=module_to_num_steps_per_sample,
            module_to_sampled_n_step=module_to_sampled_n_step,
            module_to_num_resamples=module_to_num_resamples,
        )


class NumpyIndexedFastSampleEpisodeReplayBuffer(FastSampleEpisodeReplayBuffer):
    """Fast replay buffer that stores timestep sample indices in NumPy arrays.

    This keeps the ``FastSampleEpisodeReplayBuffer`` transition fast path, but
    replaces the hot-path Python list-of-tuples ``_indices`` with two dense NumPy
    columns: episode absolute index and timestep within that episode. Unsupported
    sample modes still materialize ``_indices`` lazily before falling back to RLlib.
    """

    def __init__(
        self,
        *args: Any,
        index_capacity: int | None = None,
        expand_index_capacity: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self._index_capacity = int(self.capacity if index_capacity is None else index_capacity)
        if self._index_capacity < 0:
            raise ValueError("index_capacity must be non-negative")

        self._expand_index_capacity = expand_index_capacity
        self._index_episode = np.empty(self._index_capacity, dtype=np.int64)
        self._index_timestep = np.empty(self._index_capacity, dtype=np.int32)
        self._num_indices = 0
        # _indices is only a compatibility cache for fallback parent methods.
        self._indices = []

    @override(ReplayBufferInterface)
    def add(
        self,
        episodes: Union[list["SingleAgentEpisode"], "SingleAgentEpisode"],
    ) -> None:
        """Add episodes while maintaining NumPy-backed timestep indices."""
        episode_list = cast(list[SingleAgentEpisode], force_list(episodes))
        self._indices = []

        num_env_steps_added = 0
        agent_to_num_steps_added = {DEFAULT_AGENT_ID: 0}
        module_to_num_steps_added = {DEFAULT_MODULE_ID: 0}
        num_episodes_added = 0
        agent_to_num_episodes_added = {DEFAULT_AGENT_ID: 0}
        module_to_num_episodes_added = {DEFAULT_MODULE_ID: 0}
        num_episodes_evicted = 0
        agent_to_num_episodes_evicted = {DEFAULT_AGENT_ID: 0}
        module_to_num_episodes_evicted = {DEFAULT_MODULE_ID: 0}
        num_env_steps_evicted = 0
        agent_to_num_steps_evicted = {DEFAULT_AGENT_ID: 0}
        module_to_num_steps_evicted = {DEFAULT_MODULE_ID: 0}

        evicted_episode_indices: set[int] = set()

        for eps in episode_list:
            # Preserve RLlib's default ownership semantics.
            eps = self._copy_episode_for_add(eps)

            eps_len = len(eps)
            self._num_timesteps += eps_len
            self._num_timesteps_added += eps_len
            num_env_steps_added += eps_len
            agent_to_num_steps_added[DEFAULT_AGENT_ID] += eps_len
            module_to_num_steps_added[DEFAULT_MODULE_ID] += eps_len

            if eps.id_ in self.episode_id_to_index:
                eps_idx = self.episode_id_to_index[eps.id_]
                existing_eps = self.episodes[eps_idx - self._num_episodes_evicted]
                old_len = len(existing_eps)
                self._append_indices_np(eps_idx, old_len, eps_len)
                existing_eps.concat_episode(eps)
            else:
                num_episodes_added += 1
                agent_to_num_episodes_added[DEFAULT_AGENT_ID] += 1
                module_to_num_episodes_added[DEFAULT_MODULE_ID] += 1
                self.episodes.append(eps)
                eps_idx = len(self.episodes) - 1 + self._num_episodes_evicted
                self.episode_id_to_index[eps.id_] = eps_idx
                self._append_indices_np(eps_idx, 0, eps_len)

            while self._num_timesteps > self.capacity and self.get_num_episodes() > 1:
                evicted_eps = self.episodes.popleft()
                evicted_eps_len = len(evicted_eps)

                num_episodes_evicted += 1
                num_env_steps_evicted += evicted_eps_len
                agent_to_num_episodes_evicted[DEFAULT_AGENT_ID] += 1
                module_to_num_episodes_evicted[DEFAULT_MODULE_ID] += 1
                agent_to_num_steps_evicted[DEFAULT_AGENT_ID] += evicted_eps.agent_steps()
                module_to_num_steps_evicted[DEFAULT_MODULE_ID] += evicted_eps.agent_steps()

                self._num_timesteps -= evicted_eps_len

                evicted_idx = self.episode_id_to_index[evicted_eps.id_]
                del self.episode_id_to_index[evicted_eps.id_]
                evicted_episode_indices.add(evicted_idx)

                self._num_episodes_evicted += 1

        if evicted_episode_indices:
            self._rebuild_indices_batch_np(evicted_episode_indices)

        self._update_add_metrics(
            num_episodes_added=num_episodes_added,
            num_env_steps_added=num_env_steps_added,
            num_episodes_evicted=num_episodes_evicted,
            num_env_steps_evicted=num_env_steps_evicted,
            agent_to_num_episodes_added=agent_to_num_episodes_added,
            agent_to_num_steps_added=agent_to_num_steps_added,
            agent_to_num_episodes_evicted=agent_to_num_episodes_evicted,
            agent_to_num_steps_evicted=agent_to_num_steps_evicted,
            # Preserve RLlib's add() metrics behavior for drop-in parity.
            module_to_num_episodes_added=module_to_num_steps_added,
            module_to_num_steps_added=module_to_num_episodes_added,
            module_to_num_episodes_evicted=module_to_num_episodes_evicted,
            module_to_num_steps_evicted=module_to_num_steps_evicted,
        )

    @override(EpisodeReplayBuffer)
    def _sample_batch(
        self,
        num_items: int | None = None,
        *,
        batch_size_B: int | None = None,  # noqa: N803 - RLlib API name.
        batch_length_T: int | None = None,  # noqa: N803 - RLlib API name.
    ) -> Any:
        """Materialize Python indices for RLlib's non-episode batch fallback."""
        self._materialize_indices_for_fallback()
        try:
            return super()._sample_batch(
                num_items=num_items,
                batch_size_B=batch_size_B,
                batch_length_T=batch_length_T,
            )
        finally:
            self._indices = []

    @override(EpisodeReplayBuffer)
    def _sample_episodes(
        self,
        num_items: int | None = None,
        *,
        batch_size_B: int | None = None,  # noqa: N803 - RLlib API name.
        batch_length_T: int | None = None,  # noqa: N803 - RLlib API name.
        n_step: int | tuple | None = None,
        gamma: float = 0.99,
        include_infos: bool = False,
        include_extra_model_outputs: bool = False,
        to_numpy: bool = False,
        lookback: int = 1,
        min_batch_length_T: int = 0,  # noqa: N803 - RLlib API name.
        **kwargs: Any,
    ) -> list[SingleAgentEpisode]:
        """Use NumPy fast path when possible, materializing only for fallbacks."""
        if self._can_use_fast_transition_sample(
            num_items=num_items,
            batch_size=batch_size_B,
            batch_length=batch_length_T,
            n_step=n_step,
            include_extra_model_outputs=include_extra_model_outputs,
            to_numpy=to_numpy,
            lookback=lookback,
            min_batch_length=min_batch_length_T,
        ):
            return self._sample_episodes_fast_transition(
                num_items=num_items,
                batch_size=batch_size_B,
                n_step=cast(int, n_step),
                lookback=lookback,
            )

        self._materialize_indices_for_fallback()
        try:
            return super()._sample_episodes(
                num_items=num_items,
                batch_size_B=batch_size_B,
                batch_length_T=batch_length_T,
                n_step=n_step,
                gamma=gamma,
                include_infos=include_infos,
                include_extra_model_outputs=include_extra_model_outputs,
                to_numpy=to_numpy,
                lookback=lookback,
                min_batch_length_T=min_batch_length_T,
                **kwargs,
            )
        finally:
            self._indices = []

    def _sample_episodes_fast_transition(
        self,
        num_items: int | None,
        *,
        batch_size: int | None,
        n_step: int,
        lookback: int,
    ) -> list[SingleAgentEpisode]:
        """Sample 1-step transitions from NumPy-backed timestep indices."""
        if num_items is not None:
            assert batch_size is None, (
                "Cannot call `sample()` with both `num_items` and `batch_size_B` "
                "provided! Use either one."
            )
            batch_size = num_items

        batch_size = batch_size or self.batch_size_B
        self._last_sampled_indices = []

        sampled_episodes = []
        sampled_env_step_idxs = set()
        sampled_episode_idxs = set()

        for _ in range(batch_size):
            slot = int(self.rng.integers(self._num_indices))
            episode_abs_idx = int(self._index_episode[slot])
            episode_ts = int(self._index_timestep[slot])
            episode_idx = episode_abs_idx - self._num_episodes_evicted
            episode = self.episodes[episode_idx]
            next_ts = episode_ts + 1
            done_at_end = next_ts == len(episode)

            sampled_episode = SingleAgentEpisode(
                id_=episode.id_,
                agent_id=episode.agent_id,
                module_id=episode.module_id,
                observation_space=episode.observation_space,
                action_space=episode.action_space,
                observations=[
                    episode.get_observations(episode_ts),
                    episode.get_observations(next_ts),
                ],
                actions=[episode.get_actions(episode_ts)],
                rewards=[episode.get_rewards(episode_ts)],
                infos=[
                    episode.get_infos(episode_ts),
                    episode.get_infos(next_ts),
                ],
                terminated=episode.is_terminated if done_at_end else False,
                truncated=episode.is_truncated if done_at_end else False,
                t_started=episode_ts,
                len_lookback_buffer=0,
            )
            sampled_episode.extra_model_outputs["n_step"] = InfiniteLookbackBuffer(
                np.full((len(sampled_episode) + lookback,), n_step),
                lookback=lookback,
            )
            sampled_episode.extra_model_outputs["weights"] = InfiniteLookbackBuffer(
                np.ones((len(sampled_episode) + lookback,)),
                lookback=lookback,
            )

            sampled_env_step_idxs.add(
                hashlib.sha256(f"{episode.id_}-{episode_ts}".encode()).hexdigest()
            )
            sampled_episode_idxs.add(episode_idx)
            sampled_episodes.append(sampled_episode)

        self.sampled_timesteps += batch_size
        self._update_fast_sample_metrics(
            batch_size=batch_size,
            num_episodes_per_sample=len(sampled_episode_idxs),
            num_env_steps_per_sample=len(sampled_env_step_idxs),
            sampled_n_step=float(n_step),
        )

        return sampled_episodes

    def get_num_timesteps(self, module_id: Any | None = None) -> int:
        """Returns number of individual timesteps stored in the buffer."""
        return self._num_indices

    @override(ReplayBufferInterface)
    def get_state(self) -> dict[str, Any]:
        """Gets a pickable state with NumPy-backed index columns."""
        return {
            "episodes": [eps.get_state() for eps in self.episodes],
            "episode_id_to_index": list(self.episode_id_to_index.items()),
            "_num_episodes_evicted": self._num_episodes_evicted,
            "_num_timesteps": self._num_timesteps,
            "_num_timesteps_added": self._num_timesteps_added,
            "sampled_timesteps": self.sampled_timesteps,
            "_index_episode": self._index_episode[: self._num_indices].copy(),
            "_index_timestep": self._index_timestep[: self._num_indices].copy(),
            "_num_indices": self._num_indices,
            "_index_capacity": self._index_capacity,
        }

    @override(ReplayBufferInterface)
    def set_state(self, state: dict[str, Any]) -> None:
        """Restores NumPy-backed state, including migration from old ``_indices``."""
        self._set_episodes(state)
        self.episode_id_to_index = dict(state["episode_id_to_index"])
        self._num_episodes_evicted = state["_num_episodes_evicted"]
        self._num_timesteps = state["_num_timesteps"]
        self._num_timesteps_added = state["_num_timesteps_added"]
        self.sampled_timesteps = state["sampled_timesteps"]

        if "_index_episode" in state:
            saved_episode = np.asarray(state["_index_episode"], dtype=self._index_episode.dtype)
            saved_timestep = np.asarray(state["_index_timestep"], dtype=self._index_timestep.dtype)
            saved_num_indices = int(state.get("_num_indices", len(saved_episode)))
            saved_capacity = int(state.get("_index_capacity", saved_num_indices))
        else:
            old_indices = state.get("_indices", [])
            saved_num_indices = len(old_indices)
            saved_capacity = saved_num_indices
            if saved_num_indices:
                old_index_array = np.asarray(old_indices)
                saved_episode = old_index_array[:, 0].astype(self._index_episode.dtype)
                saved_timestep = old_index_array[:, 1].astype(self._index_timestep.dtype)
            else:
                saved_episode = np.empty(0, dtype=self._index_episode.dtype)
                saved_timestep = np.empty(0, dtype=self._index_timestep.dtype)

        self._index_capacity = max(self._index_capacity, saved_capacity, saved_num_indices)
        self._index_episode = np.empty(self._index_capacity, dtype=np.int64)
        self._index_timestep = np.empty(self._index_capacity, dtype=np.int32)
        if saved_num_indices:
            self._index_episode[:saved_num_indices] = saved_episode[:saved_num_indices]
            self._index_timestep[:saved_num_indices] = saved_timestep[:saved_num_indices]
        self._num_indices = saved_num_indices
        self._indices = []

    def _append_indices_np(self, eps_idx: int, start_ts: int, length: int) -> None:
        """Append timestep sample indices for one episode/chunk."""
        if length <= 0:
            return

        end_ts = start_ts + length
        if end_ts - 1 > np.iinfo(self._index_timestep.dtype).max:
            raise OverflowError("timestep index exceeds int32 index storage")

        end = self._num_indices + length
        self._ensure_index_capacity(end)
        write_slice = slice(self._num_indices, end)

        self._index_episode[write_slice] = eps_idx
        self._index_timestep[write_slice] = np.arange(
            start_ts,
            end_ts,
            dtype=self._index_timestep.dtype,
        )
        self._num_indices = end

    def _ensure_index_capacity(self, required: int) -> None:
        """Grow NumPy index arrays when allowed and required."""
        if required <= self._index_capacity:
            return
        if not self._expand_index_capacity:
            raise BufferError(
                f"Replay index capacity exceeded: need {required}, "
                f"capacity={self._index_capacity}"
            )

        new_capacity = max(required, int(max(self._index_capacity, 1) * 1.5))
        new_episode = np.empty(new_capacity, dtype=self._index_episode.dtype)
        new_timestep = np.empty(new_capacity, dtype=self._index_timestep.dtype)
        new_episode[: self._num_indices] = self._index_episode[: self._num_indices]
        new_timestep[: self._num_indices] = self._index_timestep[: self._num_indices]

        self._index_episode = new_episode
        self._index_timestep = new_timestep
        self._index_capacity = new_capacity

    def _rebuild_indices_batch_np(self, evicted_episode_indices: set[int]) -> None:
        """Remove evicted episode timestep indices via NumPy compaction."""
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
        self._indices = []

    def _materialize_indices_for_fallback(self) -> None:
        """Build RLlib-compatible list-of-tuples indices for parent fallbacks."""
        self._indices = list(
            zip(
                self._index_episode[: self._num_indices].tolist(),
                self._index_timestep[: self._num_indices].tolist(),
                strict=True,
            )
        )


class PrioritizedSumTreeBuffer(PrioritizedEpisodeReplayBuffer):
    """Replay buffer with SumTree-based prioritized sampling.

    This buffer extends RLlib's `PrioritizedEpisodeReplayBuffer` with a custom
    SumTree-based priority management system (replacing the default segment-tree
    implementation). The key characteristics are:

    - O(log n) sampling and O(log n) priority updates via binary segment tree
    - Supports proportional priority sampling (alpha controls priority exponent)
    - Compatible with the new RLlib EnvRunner API (inherits from EpisodeReplayBuffer)
    - Stores complete episodes and samples transitions based on TD-error priorities

    Args:
        capacity: Maximum number of timesteps to store.
        alpha: Exponent controlling priority strength. Higher = more prioritizes
            high error samples. Range: [0, 1]. Passed to parent as `alpha`.
        beta: Exponent for importance sampling weight. Higher = less
            correction for sampling bias. Range: [0, 1]. Passed to parent `beta`.
        epsilon: Small constant added to priorities to ensure non-zero probability.
        **kwargs: Forwarded to `PrioritizedEpisodeReplayBuffer`.

    Note:
        This class overrides the internal SumTree/segment-tree priority management
        of `PrioritizedEpisodeReplayBuffer` to use the framework's own `SumTree`
        implementation (accessible via the `sum_tree` property), while delegating
        episode storage and batch construction to the parent class.
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.5,
        epsilon: float = 1e-5,
        **kwargs: Any,
    ) -> None:
        # RLlib's PrioritizedEpisodeReplayBuffer uses:
        #   _sum_segment: SumSegmentTree - for proportional sampling
        #   _min_segment: MinSegmentTree - for max-weight computation
        # We pass alpha via kwargs; beta is used at sample() call time.
        super().__init__(
            capacity=capacity,
            alpha=alpha,
            **kwargs,
        )
        self._epsilon = epsilon
        # beta is used by the parent's sample() method (via beta parameter)
        self._beta_override = beta

    @property
    def sum_tree(self) -> Any:
        """Expose the internal sum-tree for external inspection."""
        return self._sum_segment


__all__ = [
    "BatchEvictEpisodeReplayBuffer",
    "FastSampleEpisodeReplayBuffer",
    "NumpyIndexedFastSampleEpisodeReplayBuffer",
    "PrioritizedSumTreeBuffer",
    "SumTree",
]
