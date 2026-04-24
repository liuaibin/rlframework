"""Tests for rlframework.envs.wrappers."""

from typing import ClassVar

import gymnasium as gym
import numpy as np


class _ToyEnv(gym.Env):
    metadata: ClassVar[dict[str, list[str]]] = {"render_modes": []}

    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=-10.0, high=10.0, shape=(2,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(1)
        self._step = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        return np.array([1.0, 3.0], dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        obs = np.array([2.0, 4.0], dtype=np.float32)
        reward = 1.5
        terminated = self._step >= 1
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


class TestEnvWrappers:
    def test_normalize_obs_wrapper_applies_observation_hook(self):
        from rlframework.envs.wrappers import NormalizeObsWrapper

        env = NormalizeObsWrapper(_ToyEnv())

        obs0, _ = env.reset()
        # First observation normalizes to ~0 (mean initialized from obs itself).
        assert np.allclose(obs0, np.zeros_like(obs0))

        obs1, *_ = env.step(0)
        assert env._count == 2
        assert np.all(np.isfinite(obs1))
        assert not np.allclose(obs1, np.array([2.0, 4.0], dtype=np.float32))

    def test_record_episode_stats_wrapper_adds_episode_info(self):
        from rlframework.envs.wrappers import RecordEpisodeStatsWrapper

        env = RecordEpisodeStatsWrapper(_ToyEnv())
        env.reset()
        _, reward, terminated, truncated, info = env.step(0)

        assert terminated is True
        assert truncated is False
        assert info["episode_return"] == reward
        assert info["episode_length"] == 1
