"""ModelManager - lightweight model version tracking.

Stores metadata (path, version, timestamp, metrics) in a local JSON catalogue.
Can be extended to use a proper database for team-wide tracking.

Usage::

    from rlframework.storage import ModelManager

    mgr = ModelManager(catalogue_path="./models.json")
    mgr.register(
        name="ppo_cartpole",
        version="v1",
        path="s3://rl-checkpoints/ppo_cartpole/iter_500",
        metrics={"episode_return_mean": 450.0},
    )

    best = mgr.best(name="ppo_cartpole", metric="episode_return_mean")
    print(best["path"])
"""

import json
import os
import time
from typing import Any


class ModelManager:
    """JSON-backed model version catalogue.

    Args:
        catalogue_path: Path to the JSON file used as the catalogue.
            Created automatically if it does not exist.
    """

    def __init__(self, catalogue_path: str = "./model_catalogue.json"):
        self._path = catalogue_path
        self._catalogue: dict[str, list[dict]] = self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(
        self,
        name: str,
        version: str,
        path: str,
        metrics: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict:
        """Register a new model version.

        Args:
            name: Logical model name (e.g. ``"ppo_cartpole"``).
            version: Version string (e.g. ``"v1"``).
            path: Checkpoint path (local or remote).
            metrics: Evaluation metrics for this version.
            metadata: Free-form metadata dict.

        Returns:
            The newly created version record.
        """
        record = {
            "version": version,
            "path": path,
            "registered_at": time.time(),
            "metrics": metrics or {},
            "metadata": metadata or {},
        }
        self._catalogue.setdefault(name, []).append(record)
        self._save()
        return record

    def latest(self, name: str) -> dict | None:
        """Return the most recently registered version for *name*."""
        versions = self._catalogue.get(name, [])
        return versions[-1] if versions else None

    def get(self, name: str, version: str) -> dict | None:
        """Return a specific version for *name*."""
        versions = self._catalogue.get(name, [])
        for v in versions:
            if v["version"] == version:
                return v
        return None

    def best(self, name: str, metric: str, mode: str = "max") -> dict | None:
        """Return the version with the best *metric* value.

        Args:
            name: Model name.
            metric: Metric key to rank by.
            mode: ``"max"`` (higher is better) or ``"min"`` (lower is better).
        """
        versions = [
            v for v in self._catalogue.get(name, [])
            if metric in v.get("metrics", {})
        ]
        if not versions:
            return None
        def key(v):
            return v["metrics"][metric]
        return max(versions, key=key) if mode == "max" else min(versions, key=key)

    def list_versions(self, name: str) -> list[dict]:
        """Return all registered versions for *name*."""
        return list(self._catalogue.get(name, []))

    def list_models(self) -> list[str]:
        """Return all registered model names."""
        return list(self._catalogue.keys())

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if os.path.exists(self._path):
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self._path)), exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._catalogue, f, indent=2)
