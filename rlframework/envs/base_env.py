"""Base environment class built on Gymnasium.

Subclass this to implement a custom RL environment without boilerplate::

    from rlframework.envs import BaseEnv

    class MyEnv(BaseEnv):
        def __init__(self):
            super().__init__(
                observation_space=gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),
                action_space=gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                max_episode_steps=200,
            )

        def _reset(self, seed, options):
            # ... return obs, info
            return obs, info

        def _step(self, action):
            # ... return obs, reward, terminated, truncated, info
            return obs, reward, terminated, truncated, info

Then register with RLlib::

    tune.register_env("MyEnv-v0", lambda cfg: MyEnv())
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class BaseEnv(gym.Env):
    """Gymnasium environment base class with common boilerplate pre-wired.

    Subclasses only need to override ``_reset()`` and ``_step()``.
    All other Gymnasium methods have safe defaults.

    Args:
        observation_space: Observation space.
        action_space: Action space.
        max_episode_steps: Truncation horizon (passed to gymnasium.wrappers.TimeLimit).
        kwargs: Forwarded to the Gymnasium Env init.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        max_episode_steps: int | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self._max_episode_steps = max_episode_steps if max_episode_steps is not None else 999999999

        self._elapsed_steps = 0
        self.render_mode = None
        self._window = None

    # ------------------------------------------------------------------
    # Public Gymnasium API — delegates to abstract methods
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._elapsed_steps = 0
        obs, info = self._reset(seed=seed, options=options)
        return np.asarray(obs, dtype=self.observation_space.dtype), info

    def step(
        self, action
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._elapsed_steps += 1
        obs, reward, terminated, truncated, info = self._step(action)

        # TimeLimit truncation if subclass didn't already set it
        if not truncated and self._elapsed_steps >= self._max_episode_steps:
            truncated = True

        return (
            np.asarray(obs, dtype=self.observation_space.dtype),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def seed(self, seed: int | None = None) -> list[int]:
        return [seed]

    def close(self) -> None:
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass
        self._window = None
        super().close()

    def render(self) -> np.ndarray | None:
        if self.render_mode == "rgb_array":
            return self._render_rgb_array()
        if self.render_mode == "human":
            self._render_human()
        return None

    # ------------------------------------------------------------------
    # Abstract methods — subclasses must implement
    # ------------------------------------------------------------------

    def _reset(
        self,
        seed: int | None,
        options: dict | None,
    ) -> tuple[np.ndarray | dict, dict]:
        """Reset the environment.

        Args:
            seed: Optional seed for reproducibility.
            options: Optional dict to configure reset (e.g. set a specific goal).

        Returns:
            (observation, info) — same as Gymnasium's ``reset()``.
        """
        raise NotImplementedError

    def _step(self, action) -> tuple[np.ndarray | dict, float, bool, bool, dict]:
        """Apply one step.

        Args:
            action: Action from the agent.

        Returns:
            (observation, reward, terminated, truncated, info) — same as Gymnasium's ``step()``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Override-able render helpers
    # ------------------------------------------------------------------

    def _render_rgb_array(self) -> np.ndarray | None:
        """Override this to provide an RGB image of the current state."""
        return None

    def _render_human(self) -> None:
        """Override this to render to a display (e.g. pygame)."""
        pass
