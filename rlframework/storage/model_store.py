"""ModelStore - unified model storage combining checkpoint + metadata management.

Usage::

    from rlframework.storage import ModelStore

    store = ModelStore(
        backend="minio",
        backend_config=dict(
            endpoint="minio.example.com:9000",
            access_key="minio",
            secret_key="minioadmin",
            bucket="rl-checkpoints",
        ),
        catalogue_path="./model_catalogue.json",
    )

    # Save model (upload + register)
    store.save(
        name="ppo_cartpole",
        version="iter_500",
        local_path="./checkpoints/iter_500",
        metrics={"episode_return_mean": 450.0},
    )

    # Get best model info
    best = store.get_best("ppo_cartpole", metric="episode_return_mean")
    print(f"Best: {best['version']} at {best['path']}")

    # Load best model
    store.load("ppo_cartpole", "best", local_dir="./loaded_models")
"""

from concurrent.futures import Future
from typing import Any

from rlframework.storage.checkpoint_manager import CheckpointManager
from rlframework.storage.model_manager import ModelManager


class ModelStore:
    """Unified model storage: combines checkpoint files + metadata tracking.

    This class wraps :class:`~rlframework.storage.CheckpointManager` and
    :class:`~rlframework.storage.ModelManager` to provide a single interface
    for saving and loading models with their associated metadata.

    Args:
        backend: Storage backend name (``"local"``, ``"minio"``, ``"s3"``) or
            a pre-configured backend instance.
        backend_config: Keyword arguments passed to the backend constructor.
        catalogue_path: Path to the JSON file for model metadata.
        upload_async: Whether to upload checkpoints asynchronously.
        upload_workers: Number of workers for async uploads.
        upload_retries: Number of retries on upload failure.
    """

    def __init__(
        self,
        backend: str | Any = "local",
        backend_config: dict | None = None,
        catalogue_path: str = "./model_catalogue.json",
        upload_async: bool = True,
        upload_workers: int = 2,
        upload_retries: int = 3,
    ):
        self._ckpt_mgr = CheckpointManager(
            backend=backend,
            backend_config=backend_config,
            upload_async=upload_async,
            upload_workers=upload_workers,
            upload_retries=upload_retries,
        )
        self._model_mgr = ModelManager(catalogue_path=catalogue_path)

    # ------------------------------------------------------------------
    # Public API - Save
    # ------------------------------------------------------------------

    def save(
        self,
        name: str,
        version: str,
        local_path: str,
        metrics: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        upload_async: bool = True,
    ) -> str | Future:
        """Save a model: upload checkpoint + register metadata.

        Args:
            name: Logical model name (e.g. ``"ppo_cartpole"``).
            version: Version string (e.g. ``"iter_500"``, ``"best"``).
            local_path: Path to the local checkpoint file or directory.
            metrics: Evaluation metrics for this version.
            metadata: Free-form metadata dict.
            upload_async: If True, upload asynchronously and return a Future.
                If False, block until upload completes and return the remote path.

        Returns:
            Remote path string (sync mode) or a Future that resolves to the
            remote path (async mode).
        """
        remote_path = f"{name}/{version}.tar"

        if upload_async:
            upload_future = self._ckpt_mgr.upload(local_path, remote_path)
            # Register metadata immediately (may complete before upload)
            self._model_mgr.register(
                name=name,
                version=version,
                path=remote_path,
                metrics=metrics,
                metadata=metadata,
            )
            return upload_future
        else:
            result = self._ckpt_mgr.upload(local_path, remote_path)
            self._model_mgr.register(
                name=name,
                version=version,
                path=remote_path,
                metrics=metrics,
                metadata=metadata,
            )
            return result

    def save_best(
        self,
        name: str,
        local_path: str,
        metrics: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        upload_async: bool = True,
    ) -> str | Future:
        """Convenience method to save as the "best" version.

        This is equivalent to calling ``save(..., version="best", ...)``.
        """
        return self.save(
            name=name,
            version="best",
            local_path=local_path,
            metrics=metrics,
            metadata=metadata,
            upload_async=upload_async,
        )

    # ------------------------------------------------------------------
    # Public API - Load
    # ------------------------------------------------------------------

    def load(
        self,
        name: str,
        version: str,
        local_dir: str,
    ) -> str:
        """Download a model from remote storage to local directory.

        Args:
            name: Model name.
            version: Version to load (e.g. ``"best"``, ``"iter_500"``).
            local_dir: Local directory to download to.

        Returns:
            Absolute path to the downloaded checkpoint.

        Raises:
            KeyError: If the model/version is not found in the catalogue.
        """
        info = self._model_mgr.get(name, version)
        if info is None:
            raise KeyError(f"Model {name} version {version} not found")
        return self._ckpt_mgr.download(info["path"], local_dir)

    def load_best(
        self,
        name: str,
        metric: str = "episode_return_mean",
        mode: str = "max",
        local_dir: str = ".",
    ) -> str:
        """Download the best model by a specific metric.

        Args:
            name: Model name.
            metric: Metric key to rank by.
            mode: ``"max"`` or ``"min"``.
            local_dir: Local directory to download to.

        Returns:
            Absolute path to the downloaded checkpoint.
        """
        best_info = self._model_mgr.best(name, metric, mode)
        if best_info is None:
            raise KeyError(f"No versions found for model {name}")
        return self._ckpt_mgr.download(best_info["path"], local_dir)

    # ------------------------------------------------------------------
    # Public API - Query
    # ------------------------------------------------------------------

    def get(self, name: str, version: str) -> dict | None:
        """Get metadata for a specific model version."""
        return self._model_mgr.get(name, version)

    def latest(self, name: str) -> dict | None:
        """Get the most recently saved version."""
        return self._model_mgr.latest(name)

    def get_best(
        self,
        name: str,
        metric: str = "episode_return_mean",
        mode: str = "max",
    ) -> dict | None:
        """Get the best version by a specific metric (alias for :meth:`best`)."""
        return self.best(name, metric, mode)

    def best(self, name: str, metric: str, mode: str = "max") -> dict | None:
        """Get the best version by a metric."""
        return self._model_mgr.best(name, metric, mode)

    def list_versions(self, name: str) -> list[dict]:
        """List all versions of a model."""
        return self._model_mgr.list_versions(name)

    def list_models(self) -> list[str]:
        """List all registered model names."""
        return self._model_mgr.list_models()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the async upload thread pool."""
        self._ckpt_mgr.shutdown(wait=wait)
