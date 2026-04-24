"""FrameworkCallback - integrates all reporters into the RLlib callback system.

Usage::

    from rlframework.logging import FrameworkCallback
    from rlframework.logging.reporters import InfluxDBReporter, FileReporter

    reporters = [
        InfluxDBReporter(url="http://influxdb:8086", org="rl",
                         bucket="metrics", token="..."),
        FileReporter(filepath="./metrics.json"),
    ]

    config = (
        CustomPPOConfig()
        .environment("CartPole-v1")
        .callbacks(FrameworkCallback.with_reporters(reporters))
    )

Checkpointing is handled by :class:`~rlframework.storage.AutoCheckpoint`
instead of this callback.  See ``examples/`` for a concrete usage pattern.
"""


import logging
import os
from functools import partial

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN

logger = logging.getLogger(__name__)


class FrameworkCallback(RLlibCallback):
    """RLlib callback that fans out training metrics.

    The reporters receive a flat metric dict after every training iteration.
    Resource metrics (CPU / memory) are optionally collected via *psutil*.
    Checkpointing is handled by :class:`~rlframework.storage.AutoCheckpoint`.

    Args:
        reporters: List of :class:`~rlframework.logging.reporters.BaseReporter`
            instances to send metrics to.
        collect_resource_stats: When ``True``, attach process-level CPU and
            memory stats to every report.
        checkpoint_manager: Pre-configured :class:`~rlframework.storage.CheckpointManager`.
            When supplied, checkpointing is enabled with this manager.
        checkpoint_freq: Checkpoint save frequency (iterations). ``0`` disables.
        checkpoint_local_dir: Local directory for checkpoint files.
    """

    def __init__(
        self,
        reporters: list | None = None,
        collect_resource_stats: bool = False,
        checkpoint_manager=None,
        checkpoint_freq: int = 0,
        checkpoint_local_dir: str = "./checkpoints",
    ):
        super().__init__()
        self._reporters = reporters or []
        self._collect_resource = collect_resource_stats
        self._ckpt_manager = checkpoint_manager
        self._ckpt_freq = checkpoint_freq
        self._ckpt_local_dir = checkpoint_local_dir

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_reporters"] = []  # exclude reporters on serialization
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    @classmethod
    def with_reporters(cls, reporters: list, **kwargs):
        """Return a callback factory compatible with ``config.callbacks(...)``."""
        return partial(cls, reporters=reporters, **kwargs)

    # ------------------------------------------------------------------
    # RLlib hooks
    # ------------------------------------------------------------------

    def on_episode_end(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env=None,
        env_index: int = 0,
        rl_module=None,
        **kwargs,
    ) -> None:
        """Hook reserved for future per-episode metric injection."""
        pass

    def on_train_result(
        self, *, result: dict, algorithm=None, metrics_logger=None, **kwargs
    ) -> None:
        metrics = self._extract_metrics(result)
        metrics["phase"] = "train"
        if self._collect_resource:
            metrics.update(self._resource_stats())
        iteration = result.get("training_iteration", 0)
        self._fan_out(metrics, iteration=iteration, phase="train")

    def on_evaluate_end(
        self,
        *,
        algorithm=None,
        metrics_logger=None,
        evaluation_metrics: dict,
        **kwargs,
    ) -> None:
        """Report evaluation metrics separately with ``phase="eval"``."""
        metrics = self._extract_eval_metrics(evaluation_metrics)
        if self._collect_resource:
            metrics.update(self._resource_stats())
        iteration = evaluation_metrics.get("training_iteration", 0)
        self._fan_out(metrics, iteration=iteration, phase="eval")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fan_out(self, metrics: dict, *, iteration: int, phase: str) -> None:
        """Send *metrics* to every registered reporter."""
        for reporter in self._reporters:
            try:
                reporter.report(metrics, iteration=iteration, phase=phase)
            except Exception as exc:
                logger.warning(
                    "Reporter %s failed: %s", reporter, exc
                )

    # ------------------------------------------------------------------
    # Metric extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_metrics(result: dict) -> dict:
        """Flatten the most relevant keys from the training result dict."""
        flat: dict = {}
        flat["training_iteration"] = result.get("training_iteration", 0)
        flat["time_total_s"] = result.get("time_total_s", 0.0)

        env_runners = result.get(ENV_RUNNER_RESULTS, {})
        flat["episode_return_mean"] = env_runners.get(EPISODE_RETURN_MEAN, 0.0)
        flat["episode_len_mean"] = env_runners.get("episode_len_mean", 0.0)
        flat["num_env_steps_sampled_lifetime"] = result.get(
            "num_env_steps_sampled_lifetime", 0
        )
        flat["num_env_steps_trained_lifetime"] = result.get(
            "num_env_steps_trained_lifetime", 0
        )

        for key, value in env_runners.items():
            if key in {EPISODE_RETURN_MEAN, "episode_len_mean"}:
                continue
            if isinstance(value, (int, float)):
                flat[key] = value

        learner = result.get("learner_results", {})
        for module_id, module_stats in learner.items():
            for k, v in module_stats.items():
                if isinstance(v, (int, float)):
                    flat[f"learner/{module_id}/{k}"] = v

        known_top_level = {
            "training_iteration",
            "time_total_s",
            "num_env_steps_sampled_lifetime",
            "num_env_steps_trained_lifetime",
            ENV_RUNNER_RESULTS,
            "learner_results",
            "evaluation",
        }
        for key, value in result.items():
            if key in known_top_level:
                continue
            if isinstance(value, (int, float)):
                flat[key] = value

        return flat

    @staticmethod
    def _extract_eval_metrics(evaluation_metrics: dict) -> dict:
        """Flatten evaluation-specific result dict with ``eval/`` prefix."""
        flat: dict = {}

        env_runners = evaluation_metrics.get(ENV_RUNNER_RESULTS, {})
        flat["eval/episode_return_mean"] = env_runners.get(EPISODE_RETURN_MEAN, 0.0)
        flat["eval/episode_len_mean"] = env_runners.get("episode_len_mean", 0.0)

        for key, value in env_runners.items():
            if key in {EPISODE_RETURN_MEAN, "episode_len_mean"}:
                continue
            if isinstance(value, (int, float)):
                flat[f"eval/{key}"] = value

        for key, value in evaluation_metrics.items():
            if key in {ENV_RUNNER_RESULTS, "training_iteration"}:
                continue
            if isinstance(value, (int, float)):
                flat[f"eval/{key}"] = value

        return flat

    @staticmethod
    def _resource_stats() -> dict:
        try:
            import psutil
            proc = psutil.Process(os.getpid())
            mem = proc.memory_info()
            return {
                "system/process_cpu_percent": proc.cpu_percent(interval=0.05),
                "system/process_memory_rss_mb": mem.rss / (1024 ** 2),
                "system/system_memory_percent": psutil.virtual_memory().percent,
            }
        except Exception:
            return {}
