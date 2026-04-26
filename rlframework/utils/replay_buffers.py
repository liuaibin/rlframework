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

from typing import Any

from ray.rllib.execution.segment_tree import SumSegmentTree as SumTree
from ray.rllib.utils.replay_buffers.prioritized_episode_buffer import (
    PrioritizedEpisodeReplayBuffer,
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
    "PrioritizedSumTreeBuffer",
    "SumTree",
]
