"""Common gymnasium wrappers."""

from typing import Any

import numpy as np
from gymnasium import Env, ObservationWrapper, Wrapper, spaces


class NormalizeObsWrapper(ObservationWrapper):
    """Normalize observations to zero mean / unit variance using a running estimate.

    Args:
        env: Base environment to wrap.
        epsilon: Small value to avoid division by zero.
    """

    def __init__(self, env: Env[Any, Any], epsilon: float = 1e-8) -> None:
        super().__init__(env)
        self._epsilon = epsilon
        self._mean: np.ndarray | None = None
        self._var: np.ndarray | None = None
        self._count = 0
        if isinstance(env.observation_space, spaces.Box):
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=env.observation_space.shape,
                dtype=np.float32,
            )

    def _update_stats(self, obs: np.ndarray) -> None:
        obs = obs.astype(np.float64, copy=False)
        if self._mean is None:
            self._mean = obs.copy()
            self._var = np.zeros_like(obs, dtype=np.float64)
            self._count = 1
            return

        self._count += 1
        delta = obs - self._mean
        self._mean += delta / self._count
        self._var += delta * (obs - self._mean)

    def observation(self, obs: np.ndarray) -> np.ndarray:
        self._update_stats(obs)
        assert self._mean is not None and self._var is not None
        std = np.sqrt(self._var / max(self._count, 1) + self._epsilon)
        return ((obs - self._mean) / std).astype(np.float32)


class RecordEpisodeStatsWrapper(Wrapper):
    """Attach episode-level statistics to the ``info`` dict.

    After each episode, the info dict returned from ``step()`` will contain:
    ``episode_return``, ``episode_length``.
    """

    def __init__(self, env: Env[Any, Any]) -> None:
        super().__init__(env)
        self._episode_return = 0.0
        self._episode_length = 0

    def reset(self, **kwargs: Any) -> Any:
        self._episode_return = 0.0
        self._episode_length = 0
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> Any:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_return += float(reward)
        self._episode_length += 1
        if terminated or truncated:
            info["episode_return"] = self._episode_return
            info["episode_length"] = self._episode_length
        return obs, reward, terminated, truncated, info
