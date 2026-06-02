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

import os
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, cast

from ray.rllib.callbacks.callbacks import RLlibCallback

from rlframework.callbacks import FrameworkCallback
from rlframework.config import validators
from rlframework.observability.reporters import BaseReporter
from rlframework.storage import CheckpointManager
from rlframework.utils.exceptions import CheckpointError, ConfigurationError, ValidationError


@dataclass(frozen=True)
class RunLayout:
    """Resolved directory layout for one training run."""

    run_dir: Path
    rllib_log_dir: Path
    checkpoint_dir: Path
    best_checkpoint_dir: Path
    metrics_dir: Path
    storage_dir: Path


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
        self._checkpoint_local_dir: str | None = None
        self._checkpoint_local_dir_configured: bool = False

        # --- storage section ---
        self._storage_upload_async: bool = True
        self._best_upload_freq: int = 1  # upload best model every N improvements

        # --- run layout section ---
        self._run_name: str | None = None
        self._run_root_dir: str = "./runs"
        self._run_id: str | None = None
        self._run_auto_logger: bool = True
        self._run_strict_layout: bool = False
        self._framework_run_layout: RunLayout | None = None
        self._framework_logger_factory: Callable[[Any, RunLayout], Any] | None = None
        self._framework_managed_logger_creator: Callable[[Any], Any] | None = None
        self._warned_custom_logger_creator: bool = False
        self._resume_checkpoint_path: str | None = None

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

    def framework_run(
        self,
        name: str,
        root_dir: str = "./runs",
        run_id: str | None = None,
        auto_logger: bool = True,
        strict_layout: bool = False,
    ) -> "FrameworkConfigMixin":
        """Configure the managed directory layout for one training run.

        Args:
            name: Experiment name under ``root_dir``.
            root_dir: Root directory for framework-managed run artifacts.
            run_id: Optional subdirectory for a concrete run / variant.
            auto_logger: When ``True``, install a default RLlib ``UnifiedLogger``
                under the run layout if the user did not provide one.
            strict_layout: When ``True``, reject raw ``logger_creator`` functions
                because they can bypass the managed logger directory.

        Returns:
            This config (for method chaining).
        """
        if not isinstance(name, str) or not name.strip():
            raise ValidationError(
                "framework_run name must be a non-empty string",
                field="framework_run.name",
                value=name,
            )
        if not isinstance(root_dir, str) or not root_dir.strip():
            raise ValidationError(
                "framework_run root_dir must be a non-empty string",
                field="framework_run.root_dir",
                value=root_dir,
            )
        if run_id is not None and (not isinstance(run_id, str) or not run_id.strip()):
            raise ValidationError(
                "framework_run run_id must be a non-empty string when provided",
                field="framework_run.run_id",
                value=run_id,
            )

        timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self._run_name = f"{name.strip()}_{timestamp}"
        self._run_root_dir = root_dir.strip()
        self._run_id = run_id.strip() if isinstance(run_id, str) else None
        self._run_auto_logger = bool(auto_logger)
        self._run_strict_layout = bool(strict_layout)
        self._framework_run_layout = None
        return self

    def framework_logger_creator(
        self,
        factory: Callable[[Any, RunLayout], Any],
    ) -> "FrameworkConfigMixin":
        """Configure a custom RLlib logger factory that receives the run layout.

        The factory is wrapped into RLlib's one-argument ``logger_creator``
        shape during build.
        """
        if not callable(factory):
            raise ValidationError(
                "framework_logger_creator factory must be callable",
                field="framework_logger_creator.factory",
                value=factory,
            )
        self._framework_logger_factory = factory
        return self

    def resume_from(
        self,
        checkpoint_path: str | os.PathLike[str],
    ) -> "FrameworkConfigMixin":
        """Restore the built RLlib algorithm from an existing checkpoint path.

        The restore happens automatically after ``config.build()`` creates the
        algorithm, so this can be used in the normal fluent config chain.

        Args:
            checkpoint_path: Local RLlib checkpoint directory/file, or a URI
                supported by RLlib's ``restore_from_path``.

        Returns:
            This config (for method chaining).
        """
        if isinstance(checkpoint_path, os.PathLike):
            checkpoint_path = os.fspath(checkpoint_path)
        if not isinstance(checkpoint_path, str) or not checkpoint_path.strip():
            raise ValidationError(
                "resume_from checkpoint_path must be a non-empty string or path",
                field="resume_from.checkpoint_path",
                value=checkpoint_path,
            )
        self._resume_checkpoint_path = checkpoint_path.strip()
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
        local_dir: str | None = None,
    ) -> "FrameworkConfigMixin":
        """Configure automatic checkpoint saving.

        Args:
            freq: Save a checkpoint every *freq* training iterations.
                ``0`` disables automatic checkpointing.
            local_dir: Local directory to save checkpoints into. If omitted
                and ``framework_run`` is enabled, uses the managed run layout.

        Returns:
            This config (for method chaining).
        """
        validators.validate_non_negative_int(freq, "checkpoint_freq")
        if local_dir is not None and (not isinstance(local_dir, str) or not local_dir.strip()):
            raise ValidationError(
                "checkpoint local_dir must be a non-empty string",
                field="checkpointing.local_dir",
                value=local_dir,
            )
        self._checkpoint_freq = freq
        self._checkpoint_local_dir = local_dir
        self._checkpoint_local_dir_configured = local_dir is not None
        return self

    @property
    def framework_layout(self) -> RunLayout | None:
        """Return the resolved framework run layout, if ``framework_run`` is enabled."""
        return self._resolve_framework_run_layout()

    # ------------------------------------------------------------------
    # Helper: build reporters from config
    # ------------------------------------------------------------------

    def build_reporters(self) -> list[BaseReporter]:
        """Instantiate and return the configured reporter objects."""
        self._apply_framework_path_defaults(self._resolve_framework_run_layout())

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
        self._apply_framework_path_defaults(self._resolve_framework_run_layout())
        if not self._storage_configured:
            return None
        return CheckpointManager(
            backend=self._storage_backend,
            backend_config=self._storage_backend_config,
            upload_async=self._storage_upload_async,
        )

    def _resolve_framework_run_layout(self) -> RunLayout | None:
        """Resolve the configured run layout without creating directories."""
        if self._run_name is None:
            return None
        if self._framework_run_layout is not None:
            return self._framework_run_layout

        run_dir = Path(self._run_root_dir).expanduser() / self._run_name
        if self._run_id is not None:
            run_dir = run_dir / self._run_id

        layout = RunLayout(
            run_dir=run_dir,
            rllib_log_dir=run_dir / "rllib_logs",
            checkpoint_dir=run_dir / "checkpoints",
            best_checkpoint_dir=run_dir / "checkpoints" / "best",
            metrics_dir=run_dir / "metrics",
            storage_dir=run_dir / "storage",
        )
        self._framework_run_layout = layout
        return layout

    def _apply_framework_run_layout(self, layout: RunLayout | None) -> None:
        """Apply managed run-layout defaults while preserving explicit user paths."""
        self._apply_framework_path_defaults(layout)
        if layout is None:
            return

        self._apply_framework_logger_layout(layout)

    def _apply_framework_path_defaults(self, layout: RunLayout | None) -> None:
        """Apply non-logger path defaults derived from the run layout."""
        if layout is None:
            if self._checkpoint_local_dir is None:
                self._checkpoint_local_dir = "./checkpoints"
            return

        if not self._checkpoint_local_dir_configured:
            self._checkpoint_local_dir = str(layout.checkpoint_dir)

        if "file" in self._metrics_reporters:
            reporter_configs = dict(self._metrics_reporter_configs)
            file_config = dict(reporter_configs.get("file", {}))
            file_config.setdefault("filepath", str(layout.metrics_dir / "metrics.jsonl"))
            reporter_configs["file"] = file_config
            self._metrics_reporter_configs = reporter_configs

        if self._storage_configured and self._storage_backend == "local":
            self._storage_backend_config.setdefault("root", str(layout.storage_dir))

    def _apply_framework_logger_layout(self, layout: RunLayout) -> None:
        """Install or validate the RLlib logger creator for a managed run."""
        current_logger_creator = getattr(self, "logger_creator", None)

        if self._framework_logger_factory is not None:
            logger_creator = self._make_framework_logger_creator(
                self._framework_logger_factory,
                layout,
            )
            self._framework_managed_logger_creator = logger_creator
            cast(Any, self).debugging(logger_creator=logger_creator)
            return

        if not self._run_auto_logger:
            return

        if current_logger_creator is None:
            logger_creator = self._make_default_unified_logger_creator(layout.rllib_log_dir)
            self._framework_managed_logger_creator = logger_creator
            cast(Any, self).debugging(logger_creator=logger_creator)
            return

        if current_logger_creator is self._framework_managed_logger_creator:
            return

        message = (
            "Custom logger_creator bypasses framework_run layout. Use "
            "framework_logger_creator() if you want a custom logger under the "
            "managed run directory."
        )
        if self._run_strict_layout:
            raise ConfigurationError(
                message,
                field="framework_run.logger_creator",
            )
        if not self._warned_custom_logger_creator:
            warnings.warn(message, UserWarning, stacklevel=3)
            self._warned_custom_logger_creator = True

    @staticmethod
    def _make_default_unified_logger_creator(
        log_dir: Path,
    ) -> Callable[[Any], Any]:
        """Create RLlib's deprecated one-argument logger creator in one place."""
        resolved_log_dir = log_dir.expanduser().resolve()

        def logger_creator(config: Any) -> Any:
            from ray.tune.logger import UnifiedLogger

            os.makedirs(resolved_log_dir, exist_ok=True)
            return UnifiedLogger(config, str(resolved_log_dir), loggers=None)

        return logger_creator

    @staticmethod
    def _make_framework_logger_creator(
        factory: Callable[[Any, RunLayout], Any],
        layout: RunLayout,
    ) -> Callable[[Any], Any]:
        """Wrap a framework-aware two-argument factory for RLlib."""

        def logger_creator(config: Any) -> Any:
            return factory(config, layout)

        return logger_creator

    def _apply_framework_runtime_config(self) -> None:
        """Wire up runtime objects after all config has been set.

        This method is called by the algorithm's ``setup()`` method.
        It configures the callback class with reporters and checkpoint manager
        unless the user has already provided a custom callback class.

        Subclasses can override this to add additional wiring.
        """
        layout = self._resolve_framework_run_layout()
        self._apply_framework_run_layout(layout)
        self._validate_framework_config()
        self._resolve_resume_checkpoint_path()
        reporters = self.build_reporters()
        ckpt_mgr = self.build_checkpoint_manager()

        existing = getattr(self, "callbacks_class", RLlibCallback)

        if existing is RLlibCallback:
            # No custom callback — wire everything into a fresh FrameworkCallback.
            checkpoint_local_dir = self._checkpoint_local_dir or "./checkpoints"
            best_local_dir = (
                str(layout.best_checkpoint_dir)
                if layout is not None and not self._checkpoint_local_dir_configured
                else os.path.join(checkpoint_local_dir, "best")
            )
            cast(Any, self).callbacks(
                FrameworkCallback.with_reporters(
                    reporters,
                    checkpoint_manager=ckpt_mgr,
                    checkpoint_freq=self._checkpoint_freq,
                    checkpoint_local_dir=checkpoint_local_dir,
                    best_local_dir=best_local_dir,
                    best_upload_freq=self._best_upload_freq,
                )
            )
        elif (
            isinstance(existing, partial)
            and isinstance(existing.func, type)
            and issubclass(existing.func, FrameworkCallback)
        ):
            # User passed FrameworkCallback.with_reporters(...) or the same
            # factory on a FrameworkCallback subclass. Inject framework-managed
            # checkpointing while preserving subclass-specific kwargs.
            checkpoint_local_dir = self._checkpoint_local_dir or "./checkpoints"
            best_local_dir = (
                str(layout.best_checkpoint_dir)
                if layout is not None and not self._checkpoint_local_dir_configured
                else os.path.join(checkpoint_local_dir, "best")
            )
            merged_kwargs: dict[str, Any] = dict(
                existing.keywords or {},
                checkpoint_freq=self._checkpoint_freq,
                checkpoint_local_dir=checkpoint_local_dir,
                best_local_dir=best_local_dir,
                best_upload_freq=self._best_upload_freq,
            )
            if ckpt_mgr is not None:
                merged_kwargs["checkpoint_manager"] = ckpt_mgr
            existing_reporters = merged_kwargs.pop("reporters", None)
            user_reporters = existing_reporters if existing_reporters else reporters
            cast(Any, self).callbacks(existing.func.with_reporters(user_reporters, **merged_kwargs))

    def _restore_framework_checkpoint(self, algorithm: Any) -> Any:
        """Restore *algorithm* from the configured resume checkpoint, if any."""
        checkpoint_path = self._resolve_resume_checkpoint_path()
        if checkpoint_path is None:
            return algorithm

        restore_from_path = getattr(algorithm, "restore_from_path", None)
        if not callable(restore_from_path):
            raise ConfigurationError(
                f"Configured algorithm does not support restore_from_path(): {checkpoint_path}",
                field="resume_from.checkpoint_path",
            )

        restore_from_path(checkpoint_path)
        return algorithm

    def _resolve_resume_checkpoint_path(self) -> str | None:
        """Return the configured resume checkpoint path after local validation."""
        if self._resume_checkpoint_path is None:
            return None

        checkpoint_path = self._resume_checkpoint_path
        if "://" in checkpoint_path:
            return checkpoint_path

        local_path = Path(checkpoint_path).expanduser()
        if not local_path.exists():
            raise CheckpointError(
                "resume checkpoint path does not exist",
                checkpoint_path=str(local_path),
            )
        return str(local_path)
