"""Common gymnasium wrappers."""

from gymnasium import ObservationWrapper, Wrapper


class NormalizeObsWrapper(ObservationWrapper):
    """Normalize observations to zero mean / unit variance using a running estimate.

    Args:
        env: Base environment to wrap.
        epsilon: Small value to avoid division by zero.
    """

    def __init__(self, env, epsilon: float = 1e-8):
        super().__init__(env)
        self._epsilon = epsilon
        self._mean = None
        self._var = None
        self._count = 0

    def _update_stats(self, obs):
        import numpy as np

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

    def observation(self, obs):
        import numpy as np
        self._update_stats(obs)
        std = np.sqrt(self._var / max(self._count, 1) + self._epsilon)
        return (obs - self._mean) / std


class RecordEpisodeStatsWrapper(Wrapper):
    """Attach episode-level statistics to the ``info`` dict.

    After each episode, the info dict returned from ``step()`` will contain:
    ``episode_return``, ``episode_length``.
    """

    def __init__(self, env):
        super().__init__(env)
        self._episode_return = 0.0
        self._episode_length = 0

    def reset(self, **kwargs):
        self._episode_return = 0.0
        self._episode_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_return += reward
        self._episode_length += 1
        if terminated or truncated:
            info["episode_return"] = self._episode_return
            info["episode_length"] = self._episode_length
        return obs, reward, terminated, truncated, info
