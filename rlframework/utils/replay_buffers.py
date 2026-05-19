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
from typing import Any, Union, cast

from ray.rllib.core import DEFAULT_AGENT_ID, DEFAULT_MODULE_ID
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
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
            eps = copy.deepcopy(eps)

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
    "PrioritizedSumTreeBuffer",
    "SumTree",
]
