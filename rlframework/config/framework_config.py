"""FrameworkConfigMixin - provides storage, metrics, and checkpointing features.

This mixin can be combined with any Ray RLlib AlgorithmConfig (PPOConfig,
SACConfig, etc.) to add rlframework's storage and metrics features.

Usage::

    from ray.rllib.algorithms.ppo import PPOConfig
    from rlframework.config import FrameworkConfigMixin

    class MyPPOConfig(PPOConfig, FrameworkConfigMixin):
        pass

    config = (
        MyPPOConfig()
        .environment("CartPole-v1")
        .training(lr=1e-4)
        .storage(backend="minio", endpoint="minio:9000",
                 access_key="admin", secret_key="admin",
                 bucket="rl-models")
        .metrics(reporters=["influxdb", "file"])
        .checkpointing(freq=10, upload_async=True)
    )

    algo = config.build()

Alternatively, use the algorithm-specific configs (:class:`CustomPPOConfig`,
:class:`CustomSACConfig`) which already include this mixin.
"""

from typing import Any


class FrameworkConfigMixin:
    """Mixin that adds rlframework features to any AlgorithmConfig.

    Provides storage, metrics, and checkpointing configuration.
    Note: This is a mixin, not a standalone config class.
    Subclasses must call ``_init_framework_mixin()`` in their ``__init__``.
    """

    def _init_framework_mixin(self):
        """Initialize framework-specific attributes.

        Call this in the subclass's __init__ after super().__init__().
        """
        # --- storage section ---
        self._storage_backend: str = "local"
        self._storage_backend_config: dict[str, Any] = {}

        # --- metrics section ---
        self._metrics_reporters: list[str] = []
        self._metrics_reporter_configs: dict[str, dict] = {}

        # --- checkpointing section ---
        self._checkpoint_freq: int = 0           # 0 = manual only
        self._checkpoint_upload_async: bool = True
        self._checkpoint_local_dir: str = "./checkpoints"

    # ------------------------------------------------------------------
    # Fluent setters
    # ------------------------------------------------------------------

    def storage(
        self,
        backend: str = "local",
        **backend_kwargs,
    ) -> "FrameworkConfigMixin":
        """Configure the storage backend (chain-friendly).

        Args:
            backend: ``"local"``, ``"minio"``, or ``"s3"``.
            **backend_kwargs: Forwarded to the backend constructor.
                For MinIO: ``endpoint``, ``access_key``, ``secret_key``, ``bucket``.
                For S3: ``bucket``, optionally ``prefix``, ``region_name``.

        Returns:
            This config (for method chaining).
        """
        self._storage_backend = backend
        self._storage_backend_config = backend_kwargs
        return self

    def metrics(
        self,
        reporters: list[str] | None = None,
        reporter_configs: dict[str, dict] | None = None,
    ) -> "FrameworkConfigMixin":
        """Configure metric reporters.

        Args:
            reporters: Names of reporters to enable.
                Supported: ``"file"``, ``"influxdb"``, ``"prometheus"``.
            reporter_configs: Per-reporter configuration dicts
                (keyed by reporter name).

        Returns:
            This config (for method chaining).
        """
        self._metrics_reporters = reporters or []
        self._metrics_reporter_configs = reporter_configs or {}
        return self

    def checkpointing(
        self,
        freq: int = 0,
        local_dir: str = "./checkpoints",
        upload_async: bool = True,
    ) -> "FrameworkConfigMixin":
        """Configure automatic checkpoint saving.

        Args:
            freq: Save a checkpoint every *freq* training iterations.
                ``0`` disables automatic checkpointing.
            local_dir: Local directory to save checkpoints into.
            upload_async: When ``True``, upload to the remote storage backend
                in a background thread so training is not blocked.

        Returns:
            This config (for method chaining).
        """
        self._checkpoint_freq = freq
        self._checkpoint_local_dir = local_dir
        self._checkpoint_upload_async = upload_async
        return self

    # ------------------------------------------------------------------
    # Helper: build reporters from config
    # ------------------------------------------------------------------

    def build_reporters(self) -> list:
        """Instantiate and return the configured reporter objects."""
        from rlframework.logging.reporters import (
            FileReporter,
            InfluxDBReporter,
            PrometheusReporter,
        )

        built = []
        for name in self._metrics_reporters:
            cfg = self._metrics_reporter_configs.get(name, {})
            if name == "file":
                built.append(FileReporter(**cfg))
            elif name == "influxdb":
                built.append(InfluxDBReporter(**cfg))
            elif name == "prometheus":
                built.append(PrometheusReporter(**cfg))
            else:
                raise ValueError(f"Unknown reporter: '{name}'")
        return built

    # ------------------------------------------------------------------
    # Helper: build storage backend from config
    # ------------------------------------------------------------------

    def build_checkpoint_manager(self):
        """Instantiate and return a :class:`~rlframework.storage.CheckpointManager`."""
        from rlframework.storage import CheckpointManager
        return CheckpointManager(
            backend=self._storage_backend,
            backend_config=self._storage_backend_config,
        )
