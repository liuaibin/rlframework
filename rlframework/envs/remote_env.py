"""RemoteEnv - wraps a remote environment service using RLlib's ExternalEnv.

Architecture
------------
The remote environment runs as a separate service (e.g. a Python process or
Docker container).  This class acts as the client that connects to that
service and bridges it into the RLlib sampling pipeline.

The remote service must implement a minimal HTTP/gRPC API:
    POST /reset              → {"obs": ..., "info": ...}
    POST /step  body={action} → {"obs":..., "reward":..., "terminated":...,
                                  "truncated":..., "info":...}
    POST /close              → {}

Usage::

    from rlframework.envs import RemoteEnv, RemoteEnvConfig

    cfg = RemoteEnvConfig(
        url="http://env-service:8000",
        env_id="Pendulum-v1",
        timeout=10.0,
    )

    # Register with gym so RLlib can create it by name
    import gymnasium as gym
    gym.register("remote-pendulum-v0",
                 entry_point=lambda: RemoteEnv(cfg))

    config = CustomSACConfig().environment("remote-pendulum-v0")
"""

from dataclasses import dataclass, field
from typing import Any

try:
    import requests
    _HAS_REQUESTS = True
except ImportError:
    _HAS_REQUESTS = False

import gymnasium as gym
import numpy as np


@dataclass
class RemoteEnvConfig:
    """Connection parameters for a remote environment service."""

    url: str                          # Base URL of the env service
    env_id: str = "unknown"           # Logical name (for logging)
    timeout: float = 30.0             # HTTP request timeout in seconds
    headers: dict[str, str] = field(default_factory=dict)
    extra_params: dict[str, Any] = field(default_factory=dict)


class RemoteEnv(gym.Env):
    """Gymnasium-compatible wrapper around a remote env service.

    The observation / action spaces are fetched from the service at
    ``__init__`` time via ``GET /spaces``.  If the service does not expose
    that endpoint you can pass *observation_space* and *action_space*
    directly.

    Args:
        config: Connection configuration.
        observation_space: Override the observation space (optional).
        action_space: Override the action space (optional).
    """

    def __init__(
        self,
        config: RemoteEnvConfig,
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
    ):
        if not _HAS_REQUESTS:
            raise ImportError(
                "RemoteEnv requires the 'requests' package. "
                "Install with: pip install requests"
            )
        super().__init__()
        self._cfg = config
        self._session = requests.Session()
        self._session.headers.update(config.headers)

        if observation_space is not None and action_space is not None:
            self.observation_space = observation_space
            self.action_space = action_space
        else:
            self._fetch_spaces()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self._cfg.url.rstrip('/')}/{path.lstrip('/')}"

    def _fetch_spaces(self) -> None:
        """Retrieve spaces from the remote service's /spaces endpoint."""
        try:
            resp = self._session.get(
                self._url("/spaces"), timeout=self._cfg.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            # Expect data = {"observation_space": <gym Space JSON>,
            #                "action_space": <gym Space JSON>}
            # Simple box support for now; extend as needed.
            obs = data["observation_space"]
            act = data["action_space"]
            self.observation_space = gym.spaces.Box(
                low=np.float32(obs["low"]),
                high=np.float32(obs["high"]),
                shape=obs["shape"],
                dtype=np.float32,
            )
            self.action_space = gym.spaces.Box(
                low=np.float32(act["low"]),
                high=np.float32(act["high"]),
                shape=act["shape"],
                dtype=np.float32,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fetch spaces from {self._cfg.url}. "
                f"Provide observation_space and action_space manually. "
                f"Error: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        payload: dict[str, Any] = {}
        if seed is not None:
            payload["seed"] = seed
        resp = self._session.post(
            self._url("/reset"),
            json=payload,
            timeout=self._cfg.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return np.array(data["obs"], dtype=np.float32), data.get("info", {})

    def step(
        self, action
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        if hasattr(action, "tolist"):
            action = action.tolist()
        resp = self._session.post(
            self._url("/step"),
            json={"action": action},
            timeout=self._cfg.timeout,
        )
        resp.raise_for_status()
        d = resp.json()
        return (
            np.array(d["obs"], dtype=np.float32),
            float(d["reward"]),
            bool(d["terminated"]),
            bool(d["truncated"]),
            d.get("info", {}),
        )

    def close(self) -> None:
        try:
            self._session.post(
                self._url("/close"), timeout=self._cfg.timeout
            )
        except Exception:
            pass
        finally:
            self._session.close()
