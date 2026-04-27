"""RemoteEnv — abstract base class for remote environment clients.

Subclass this to implement a custom remote env (HTTP, gRPC, Unix socket, etc.)::

    from rlframework.envs import RemoteEnv

    class HTTPEnv(RemoteEnv):
        def __init__(self, url):
            super().__init__()
            self._url = url
            # ... set observation_space / action_space ...

        def _send_reset(self, seed, options):
            resp = requests.post(f"{self._url}/reset", json={...})
            return resp.json()["obs"], resp.json().get("info", {})

        def _send_step(self, action):
            resp = requests.post(f"{self._url}/step", json={"action": action})
            d = resp.json()
            return d["obs"], d["reward"], d["terminated"], d["truncated"], d.get("info", {})

        def _send_close(self):
            requests.post(f"{self._url}/close")

Then configure with RLlib::

    config = CustomPPOConfig().environment(env=HTTPEnv, env_config={"url": "http://localhost:8000"})
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


class RemoteEnv(gym.Env):
    """Abstract base class for remote environment clients.

    Bridges any external env service (HTTP, gRPC, Unix socket, shared memory, ...)
    into the Gymnasium / RLlib interface.  Subclasses only need to implement
    the three ``_send_*`` hooks; all Gymnasium boilerplate is pre-wired here.

    Subclasses **must** set ``self.observation_space`` and ``self.action_space``
    in ``__init__`` (or before the first ``reset`` call).
    """

    metadata: dict[str, Any] = {  # noqa: RUF012 - Gymnasium expects class-level metadata.
        "render_modes": [],
        "render_fps": 0,
    }

    def __init__(self) -> None:
        super().__init__()
        self._elapsed_steps = 0
        self.render_mode = None

    # ------------------------------------------------------------------
    # Public Gymnasium API — delegates to abstract _send_* hooks
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._elapsed_steps = 0
        obs, info = self._send_reset(seed=seed, options=options)
        return np.asarray(obs, dtype=self.observation_space.dtype), info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._elapsed_steps += 1
        obs, reward, terminated, truncated, info = self._send_step(action)
        return (
            np.asarray(obs, dtype=self.observation_space.dtype),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def close(self) -> None:
        self._send_close()
        super().close()

    # ------------------------------------------------------------------
    # Abstract hooks — subclasses must implement
    # ------------------------------------------------------------------

    def _send_reset(
        self,
        seed: int | None,
        options: dict[str, Any] | None,
    ) -> tuple[np.ndarray | dict[str, Any], dict[str, Any]]:
        """Send a reset request to the remote service.

        Args:
            seed: Optional seed forwarded to the remote service.
            options: Optional config dict forwarded to the remote service.

        Returns:
            (observation, info) — same as Gymnasium's ``reset()``.
        """
        raise NotImplementedError

    def _send_step(
        self, action: Any
    ) -> tuple[np.ndarray | dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Send a step request to the remote service.

        Args:
            action: Action from the agent.

        Returns:
            (observation, reward, terminated, truncated, info) — same as Gymnasium's ``step()``.
        """
        raise NotImplementedError

    def _send_close(self) -> None:
        """Send a close request to the remote service (optional)."""
        pass
