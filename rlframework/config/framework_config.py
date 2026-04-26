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
                 bucket="rl-models", upload_async=True, best_upload_freq=5)
        .metrics(reporters=["influxdb", "file"])
        .checkpointing(freq=10)
    )

    algo = config.build()

Alternatively, use the algorithm-specific configs (:class:`CustomPPOConfig`,
:class:`CustomSACConfig`) which already include this mixin.
"""

from functools import partial
from typing import Any, cast

from ray.rllib.callbacks.callbacks import RLlibCallback

from rlframework.callbacks import FrameworkCallback
from rlframework.config import validators
from rlframework.observability.reporters import BaseReporter
from rlframework.storage import CheckpointManager


class FrameworkConfigMixin:
    """Mixin that adds rlframework features to any AlgorithmConfig.

    Provides storage, metrics, and checkpointing configuration.
    Note: This is a mixin, not a standalone config class.
    Subclasses must call ``_init_framework_mixin()`` in their ``__init__``.
    """

    def _init_framework_mixin(self) -> None:
        """Initialize framework-specific attributes.

        Call this in the subclass's __init__ after super().__init__().
        """
        # --- storage section ---
        self._storage_backend: str = "local"
        self._storage_backend_config: dict[str, Any] = {}
        self._storage_configured: bool = False  # True only if user called .storage()

        # --- metrics section ---
        self._metrics_reporters: list[str] = []
        self._metrics_reporter_configs: dict[str, dict[str, Any]] = {}

        # --- checkpointing section ---
        self._checkpoint_freq: int = 0  # 0 = manual only
        self._checkpoint_local_dir: str = "./checkpoints"

        # --- storage section ---
        self._storage_upload_async: bool = True
        self._best_upload_freq: int = 1  # upload best model every N improvements

    def _validate_framework_config(self) -> None:
        """Validate framework and training config parameters.

        Called automatically before training starts (in ``_apply_framework_runtime_config``).
        Raises ``ValidationError`` on any invalid value.
        """
        # lr and gamma live in the training section of the config dict.
        train_cfg = getattr(self, "training", {})
        if isinstance(train_cfg, dict):
            if "lr" in train_cfg:
                validators.validate_lr(train_cfg["lr"], field="training.lr")
            if "gamma" in train_cfg:
                validators.validate_gamma(train_cfg["gamma"], field="training.gamma")

    # ------------------------------------------------------------------
    # Fluent setters
    # ------------------------------------------------------------------

    def storage(
        self,
        backend: str = "local",
        upload_async: bool = True,
        best_upload_freq: int = 1,
        **backend_kwargs: Any,
    ) -> "FrameworkConfigMixin":
        """Configure the storage backend (chain-friendly).

        Args:
            backend: ``"local"``, ``"minio"``, or ``"s3"``.
            upload_async: When ``True``, upload to the remote backend in a background
                thread so training is not blocked.
            best_upload_freq: Upload the best model to the remote backend every *N*
                improvements.  ``1`` uploads on every improvement; ``0`` disables
                remote upload of the best model.
            **backend_kwargs: Forwarded to the backend constructor.
                For MinIO: ``endpoint``, ``access_key``, ``secret_key``, ``bucket``.
                For S3: ``bucket``, optionally ``prefix``, ``region_name``.

        Returns:
            This config (for method chaining).
        """
        validators.validate_backend(backend, ["local", "minio", "s3"])
        self._storage_backend = backend
        self._storage_backend_config = backend_kwargs
        self._storage_configured = True
        self._storage_upload_async = upload_async
        self._best_upload_freq = max(0, best_upload_freq)
        return self

    def metrics(
        self,
        reporters: list[str] | None = None,
        reporter_configs: dict[str, dict[str, Any]] | None = None,
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

    def framework_checkpointing(
        self,
        freq: int = 0,
        local_dir: str = "./checkpoints",
    ) -> "FrameworkConfigMixin":
        """Configure automatic checkpoint saving.

        Args:
            freq: Save a checkpoint every *freq* training iterations.
                ``0`` disables automatic checkpointing.
            local_dir: Local directory to save checkpoints into.

        Returns:
            This config (for method chaining).
        """
        validators.validate_non_negative_int(freq, "checkpoint_freq")
        if not isinstance(local_dir, str) or not local_dir.strip():
            from rlframework.utils.exceptions import ValidationError

            raise ValidationError(
                "checkpoint local_dir must be a non-empty string",
                field="checkpointing.local_dir",
                value=local_dir,
            )
        self._checkpoint_freq = freq
        self._checkpoint_local_dir = local_dir
        return self

    # ------------------------------------------------------------------
    # Helper: build reporters from config
    # ------------------------------------------------------------------

    def build_reporters(self) -> list[BaseReporter]:
        """Instantiate and return the configured reporter objects."""
        from rlframework.observability.reporters import (
            FileReporter,
            InfluxDBReporter,
            PrometheusReporter,
        )

        built: list[BaseReporter] = []
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

    def build_checkpoint_manager(self) -> CheckpointManager | None:
        """Instantiate and return a :class:`~rlframework.storage.CheckpointManager`.

        Returns:
            A CheckpointManager if the user called :meth:`storage`, otherwise ``None``.
            When ``None``, checkpoint upload is skipped in callbacks.
        """
        if not self._storage_configured:
            return None
        return CheckpointManager(
            backend=self._storage_backend,
            backend_config=self._storage_backend_config,
            upload_async=self._storage_upload_async,
        )

    def _apply_framework_runtime_config(self) -> None:
        """Wire up runtime objects after all config has been set.

        This method is called by the algorithm's ``setup()`` method.
        It configures the callback class with reporters and checkpoint manager
        unless the user has already provided a custom callback class.

        Subclasses can override this to add additional wiring.
        """
        self._validate_framework_config()
        reporters = self.build_reporters()
        ckpt_mgr = self.build_checkpoint_manager()

        existing = getattr(self, "callbacks_class", RLlibCallback)

        if existing is RLlibCallback:
            # No custom callback — wire everything into a fresh FrameworkCallback.
            best_local_dir = self._checkpoint_local_dir + "/best"
            cast(Any, self).callbacks(
                FrameworkCallback.with_reporters(
                    reporters,
                    checkpoint_manager=ckpt_mgr,
                    checkpoint_freq=self._checkpoint_freq,
                    checkpoint_local_dir=self._checkpoint_local_dir,
                    best_local_dir=best_local_dir,
                    best_upload_freq=self._best_upload_freq,
                )
            )
        elif isinstance(existing, partial) and existing.func is FrameworkCallback:
            # User passed FrameworkCallback.with_reporters(reporters) —
            # inject checkpointing config.  Only inject checkpoint_manager if we
            # actually built one; preserve the user's checkpoint_manager if any.
            best_local_dir = self._checkpoint_local_dir + "/best"
            merged_kwargs: dict[str, Any] = dict(
                existing.keywords or {},
                checkpoint_freq=self._checkpoint_freq,
                checkpoint_local_dir=self._checkpoint_local_dir,
                best_local_dir=best_local_dir,
                best_upload_freq=self._best_upload_freq,
            )
            if ckpt_mgr is not None:
                merged_kwargs["checkpoint_manager"] = ckpt_mgr
            user_reporters = merged_kwargs.pop("reporters")
            cast(Any, self).callbacks(
                FrameworkCallback.with_reporters(user_reporters, **merged_kwargs)
            )
