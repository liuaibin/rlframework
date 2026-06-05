"""FrameworkCallback - integrates reporters into the RLlib callback system.

Usage::

    from rlframework.callbacks import FrameworkCallback
    from rlframework.observability.reporters import InfluxDBReporter, FileReporter

    reporters = [
        InfluxDBReporter(url="http://influxdb:8086", org="rl",
                         bucket="metrics", token="..."),
        FileReporter(filepath="./metrics.json"),
    ]

    config = (
        CustomPPOConfig()
        .environment("CartPole-v1")
        .evaluation(evaluation_interval=5)
        .callbacks(FrameworkCallback.with_reporters(reporters))
    )

Periodic checkpointing is driven by
:meth:`~rlframework.config.FrameworkConfigMixin.checkpointing`.
Best-model saving is driven by :meth:`on_evaluate_end` and requires evaluation
to be enabled.  See ``examples/`` for a concrete usage pattern.
"""

import logging
import os
from functools import partial
from typing import Any

from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS, EPISODE_RETURN_MEAN

from rlframework.observability.reporters import BaseReporter
from rlframework.storage import CheckpointManager

logger = logging.getLogger(__name__)


class FrameworkCallback(RLlibCallback):
    """RLlib callback that fans out training metrics and manages checkpoints.

    The reporters receive a flat metric dict after every training iteration.
    Resource metrics (CPU / memory) are optionally collected via *psutil*.

    Checkpoint behaviour is independent of the ``checkpoint_manager``:

    - **Local checkpoint saving** — always active when ``checkpoint_freq > 0``,
      controlled by :meth:`~rlframework.config.FrameworkConfigMixin.checkpointing`.
      Saves model snapshots to ``checkpoint_local_dir`` on disk.
      **Never** uploaded to the remote backend.
    - **Best model** — after each evaluation round (``on_evaluate_end``), the model
      is saved locally as ``best/`` if the eval return exceeds the best seen so far.
      Remote upload to the backend is performed every ``best_upload_freq`` improvements
      (or skipped if ``0``).  Requires evaluation to be enabled in the config.

    Args:
        reporters: List of reporter instances to send metrics to.
        collect_resource_stats: When ``True``, attach process-level CPU and
            memory stats to every report.
        checkpoint_manager: Pre-configured :class:`~rlframework.storage.CheckpointManager`.
            When supplied, best-model snapshots are uploaded to the
            configured remote backend.  Optional — local saving still works without it.
        checkpoint_freq: Save a local checkpoint every *N* training iterations.
            ``0`` disables local checkpointing.
        checkpoint_local_dir: Local directory for periodic checkpoint snapshots.
        best_local_dir: Local directory for the best-model snapshot.
        best_upload_freq: Upload the best model to the remote backend every *N*
            improvements.  ``1`` (default) uploads on every improvement; ``5`` uploads
            every 5 improvements; ``0`` disables remote upload of the best model.
    """

    def __init__(
        self,
        reporters: list[BaseReporter] | None = None,
        collect_resource_stats: bool = False,
        checkpoint_manager: CheckpointManager | None = None,
        checkpoint_freq: int = 0,
        checkpoint_local_dir: str = "./checkpoints",
        best_local_dir: str | None = None,
        best_upload_freq: int = 1,
    ) -> None:
        super().__init__()
        self._reporters = reporters or []
        self._collect_resource = collect_resource_stats
        self._ckpt_manager = checkpoint_manager
        self._ckpt_freq = checkpoint_freq
        # Convert to absolute paths so PyArrow (used by RLlib's save_to_path)
        # can handle them.
        self._ckpt_local_dir = os.path.abspath(checkpoint_local_dir)
        self._best_local_dir = os.path.abspath(best_local_dir or (checkpoint_local_dir + "/best"))
        self._best_reward = float("-inf")
        self._best_upload_freq = max(0, best_upload_freq)
        self._best_improvement_count = 0

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_reporters"] = []  # exclude reporters on serialization
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    @classmethod
    def with_reporters(cls, reporters: list[BaseReporter], **kwargs: Any) -> Any:
        """Return a callback factory compatible with ``config.callbacks(...)``."""
        return partial(cls, reporters=reporters, **kwargs)

    def on_episode_end(
        self,
        *,
        episode: Any,
        env_runner: Any = None,
        metrics_logger: Any = None,
        env: Any = None,
        env_index: int = 0,
        rl_module: Any = None,
        **kwargs: Any,
    ) -> None:
        """Hook reserved for future per-episode metric injection."""
        pass

    def on_train_result(
        self,
        *,
        result: dict[str, Any],
        algorithm: Any = None,
        metrics_logger: Any = None,
        **kwargs: Any,
    ) -> None:
        metrics = self._extract_metrics(result)
        metrics["phase"] = "train"
        if self._collect_resource:
            metrics.update(self._resource_stats())
        iteration = int(result.get("training_iteration", 0))
        self._fan_out(metrics, iteration=iteration, phase="train")
        self._save_periodic_checkpoint(algorithm, iteration)

    def on_evaluate_end(
        self,
        *,
        algorithm: Any = None,
        metrics_logger: Any = None,
        evaluation_metrics: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Report evaluation metrics and update the best model if improved."""
        metrics = self._extract_eval_metrics(evaluation_metrics)
        if self._collect_resource:
            metrics.update(self._resource_stats())
        iteration = algorithm.training_iteration if algorithm is not None else 0
        self._fan_out(metrics, iteration=iteration, phase="eval")
        self._update_best_model_if_improved(algorithm, metrics, iteration)

    def _fan_out(self, metrics: dict[str, Any], *, iteration: int, phase: str) -> None:
        """Send *metrics* to every registered reporter."""
        for reporter in self._reporters:
            try:
                reporter.report(metrics, iteration=iteration, phase=phase)
            except Exception as exc:
                logger.warning("Reporter %s failed: %s", reporter, exc)

    def _save_periodic_checkpoint(self, algorithm: Any, iteration: int) -> None:
        """Save a periodic checkpoint if *iteration* hits configured frequency."""
        if algorithm is None:
            return
        if self._ckpt_freq <= 0:
            return
        if iteration % self._ckpt_freq != 0:
            return
        ckpt_name = f"iter_{iteration:06d}"
        ckpt_path = os.path.join(self._ckpt_local_dir, ckpt_name)
        os.makedirs(self._ckpt_local_dir, exist_ok=True)
        saved_path = algorithm.save_to_path(ckpt_path)
        logger.info("Periodic checkpoint saved: %s", saved_path)

    def _update_best_model_if_improved(
        self, algorithm: Any, eval_metrics: dict[str, Any], iteration: int
    ) -> None:
        """Save and optionally upload the current model if eval reward improves."""
        if algorithm is None:
            return
        raw_eval_return = eval_metrics.get("eval/episode_return_mean", float("-inf"))
        eval_return = (
            float(raw_eval_return) if isinstance(raw_eval_return, (int, float)) else float("-inf")
        )
        if eval_return <= self._best_reward:
            return
        self._best_reward = eval_return
        self._best_improvement_count += 1
        os.makedirs(self._best_local_dir, exist_ok=True)
        local_path = algorithm.save_to_path(self._best_local_dir)
        logger.info(
            "New best model (eval_reward=%.2f, iter=%d) saved to %s.",
            eval_return,
            iteration,
            local_path,
        )
        if self._ckpt_manager is not None and self._best_upload_freq > 0:
            if self._best_improvement_count % self._best_upload_freq == 0:
                self._ckpt_manager.upload(local_path, "best.tar")
                logger.info(
                    "Best model upload triggered (improvement #%d, freq=%d).",
                    self._best_improvement_count,
                    self._best_upload_freq,
                )

    @staticmethod
    def _extract_metrics(result: dict[str, Any]) -> dict[str, Any]:
        """Flatten the most relevant keys from the training result dict."""
        flat: dict[str, Any] = {}
        flat["training_iteration"] = result.get("training_iteration", 0)
        flat["time_total_s"] = result.get("time_total_s", 0.0)
        flat["time_this_iter_s"] = result.get("time_this_iter_s", 0.0)

        env_runners = result.get(ENV_RUNNER_RESULTS, {})
        flat["episode_return_mean"] = env_runners.get(EPISODE_RETURN_MEAN, 0.0)
        flat["episode_len_mean"] = env_runners.get("episode_len_mean", 0.0)
        flat["num_env_steps_sampled_lifetime"] = result.get("num_env_steps_sampled_lifetime", 0)
        flat["num_env_steps_trained_lifetime"] = result.get("num_env_steps_trained_lifetime", 0)

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
            "time_this_iter_s",
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
    def _extract_eval_metrics(evaluation_metrics: dict[str, Any]) -> dict[str, Any]:
        """Flatten evaluation-specific result dict with ``eval/`` prefix."""
        flat: dict[str, Any] = {}

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
    def _resource_stats() -> dict[str, float]:
        try:
            import psutil

            proc = psutil.Process(os.getpid())
            mem = proc.memory_info()
            return {
                "system/process_cpu_percent": proc.cpu_percent(interval=0.05),
                "system/process_memory_rss_mb": mem.rss / (1024**2),
                "system/system_memory_percent": psutil.virtual_memory().percent,
            }
        except Exception:
            return {}
