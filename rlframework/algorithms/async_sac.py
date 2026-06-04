"""AsyncCustomSAC - SAC with configurable async EnvRunner/Learner pipeline.

Ray version detection and dual-path support:
- Ray >= 2.54.0: can use foreach_env_runner_async_fetch_ready for non-blocking
  sampling and learner_group.update(async_update=True) for non-blocking training.
- Ray < 2.54.0: delegates to CustomSAC's parent synchronous training loop.

Async controls:
- env_sampling="sync"|"async"|"auto"
- learner_training="sync"|"async"|"auto"

The main staged rollout modes are:
- async EnvRunner sampling + sync Learner training.
- async EnvRunner sampling + async Learner training.

Architecture (Ray >= 2.54.0)::

    EnvRunner 1..N  --async sample-->  episodes
          ^                                  |
          | (sync weights + connector states)  | add
          |                                  v
    LearnerGroup <--------------------  EpisodeReplayBuffer (local)
          |
          | async_update=True
          v
    RLModule weights

Architecture (Ray < 2.54.0)::

    EnvRunner 1..N  --sync sample-->  episodes
                                        | add
                                        v
    LearnerGroup <--------------------  EpisodeReplayBuffer (local)
          |
          | sync update
          v
    RLModule weights

Usage::

    from rlframework.algorithms.async_sac import AsyncCustomSACConfig, AsyncCustomSAC

    config = (
        AsyncCustomSACConfig()
        .environment("Pendulum-v1")
        .training(
            replay_buffer_config={"type": "EpisodeReplayBuffer", "capacity": 100_000},
            train_batch_size_per_learner=256,
            num_steps_sampled_before_learning_starts=1500,
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
        )
        .env_runners(
            num_env_runners=2,
            num_envs_per_env_runner=1,
            rollout_fragment_length=16,
            max_requests_in_flight_per_env_runner=1,
        )
        .learners(
            num_learners=1,
            num_gpus_per_learner=0,
            max_requests_in_flight_per_learner=1,
        )
        .algorithm_options(
            {
                "env_sampling": "async",
                "learner_training": "async",
                "pipeline_log_interval": 100,
            }
        )
    )
    algo = config.build()
    for _ in range(100):
        results = algo.train()

Prerequisites:
- replay_buffer_config["type"] must be EpisodeReplayBuffer or
  BatchEvictEpisodeReplayBuffer. Prioritization is NOT supported in this async
  pipeline (v1).
"""

import logging
from typing import Any, Literal, cast

import ray
from ray.rllib.algorithms.dqn.dqn import calculate_rr_weights
from ray.rllib.core import (
    COMPONENT_ENV_TO_MODULE_CONNECTOR,
    COMPONENT_MODULE_TO_ENV_CONNECTOR,
)
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    ALL_MODULES,
    ENV_RUNNER_RESULTS,
    ENV_RUNNER_SAMPLING_TIMER,
    LEARNER_RESULTS,
    NUM_AGENT_STEPS_SAMPLED_LIFETIME,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    REPLAY_BUFFER_ADD_DATA_TIMER,
    REPLAY_BUFFER_RESULTS,
    REPLAY_BUFFER_SAMPLE_TIMER,
    TD_ERROR_KEY,
    TIMERS,
)
from ray.rllib.utils.replay_buffers.episode_replay_buffer import EpisodeReplayBuffer

from rlframework.algorithms.sac import (
    CustomSAC,
    CustomSACConfig,
    _resolve_replay_buffer_type,
)
from rlframework.utils.exceptions import ValidationError
from rlframework.utils.replay_buffers import BatchEvictEpisodeReplayBuffer

_ASYNC_SAMPLE_TAG = "async_sac_sample"
AsyncPipelineMode = Literal["auto", "sync", "async"]
_ASYNC_PIPELINE_MODES = {"auto", "sync", "async"}
logger = logging.getLogger(__name__)
_SUPPORTED_ASYNC_REPLAY_BUFFER_TYPES = (
    EpisodeReplayBuffer,
    BatchEvictEpisodeReplayBuffer,
)
_SUPPORTED_ASYNC_REPLAY_BUFFER_NAMES = {
    "EpisodeReplayBuffer",
    f"{EpisodeReplayBuffer.__module__}.{EpisodeReplayBuffer.__name__}",
    "BatchEvictEpisodeReplayBuffer",
    f"{BatchEvictEpisodeReplayBuffer.__module__}.{BatchEvictEpisodeReplayBuffer.__name__}",
}


def _supports_async_env_runner_fetch_ready() -> bool:
    """Return whether the installed Ray has the Ray 2.54 async EnvRunner API."""
    try:
        from ray.rllib.env.env_runner_group import EnvRunnerGroup
    except Exception:
        return False
    return hasattr(EnvRunnerGroup, "foreach_env_runner_async_fetch_ready")


def _is_supported_async_replay_buffer(buffer_type: Any) -> bool:
    """Return whether the configured replay buffer is safe for AsyncCustomSAC."""
    if isinstance(buffer_type, type):
        return buffer_type in _SUPPORTED_ASYNC_REPLAY_BUFFER_TYPES

    resolved = _resolve_replay_buffer_type(buffer_type)
    if resolved is not None:
        return resolved in _SUPPORTED_ASYNC_REPLAY_BUFFER_TYPES

    if isinstance(buffer_type, str):
        return buffer_type in _SUPPORTED_ASYNC_REPLAY_BUFFER_NAMES

    return False


def _replay_buffer_type_label(buffer_type: Any) -> str:
    """Format replay buffer type for validation errors."""
    if isinstance(buffer_type, type):
        return f"{buffer_type.__module__}.{buffer_type.__name__}"
    return repr(buffer_type)


class AsyncCustomSACConfig(CustomSACConfig):
    """Configuration for AsyncCustomSAC.

    Inherits from CustomSACConfig, which inherits from SACConfig.
    Adds async-specific validation but otherwise preserves SACConfig defaults.
    """

    def __init__(self) -> None:
        super().__init__()
        self.algo_class = AsyncCustomSAC

        self.async_env_sampling: AsyncPipelineMode = "auto"
        self.async_learner_training: AsyncPipelineMode = "auto"
        self.async_pipeline_log_interval: int = 0

        # AsyncCustomSAC owns EnvRunner state/weight sync only when the async
        # EnvRunner sampling path is selected. Older Ray versions and explicit sync
        # mode keep RLlib's auto-sync.
        self._update_auto_sync_env_runner_states()

    def _set_async_pipeline_options(
        self,
        *,
        env_sampling: AsyncPipelineMode | None = None,
        learner_training: AsyncPipelineMode | None = None,
    ) -> None:
        """Configure the async pipeline pieces from algorithm options."""
        if env_sampling is not None:
            self.async_env_sampling = self._validate_async_pipeline_mode(
                "env_sampling",
                env_sampling,
            )
        if learner_training is not None:
            self.async_learner_training = self._validate_async_pipeline_mode(
                "learner_training",
                learner_training,
            )
        self._update_auto_sync_env_runner_states()

    def _apply_algorithm_options(
        self,
        options: dict[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        supported = {
            "env_sampling",
            "learner_training",
            "pipeline_log_interval",
        }
        unknown = set(options) - supported
        if unknown and strict:
            unknown_list = ", ".join(sorted(unknown))
            raise ValidationError(
                f"Unknown AsyncCustomSAC algorithm option(s): {unknown_list}",
                field="algorithm_options",
                value=options,
            )

        self._set_async_pipeline_options(
            env_sampling=options.get("env_sampling"),
            learner_training=options.get("learner_training"),
        )
        if "pipeline_log_interval" in options:
            self.async_pipeline_log_interval = self._validate_non_negative_int_option(
                "pipeline_log_interval",
                options["pipeline_log_interval"],
            )

    @staticmethod
    def _validate_async_pipeline_mode(
        name: str,
        mode: str,
    ) -> AsyncPipelineMode:
        if mode not in _ASYNC_PIPELINE_MODES:
            raise ValueError(f"`{name}` must be one of {_ASYNC_PIPELINE_MODES}. Got {mode!r}.")
        return cast(AsyncPipelineMode, mode)

    @staticmethod
    def _validate_non_negative_int_option(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValidationError(
                f"`{name}` must be a non-negative integer.",
                field=f"algorithm_options.{name}",
                value=value,
            )
        return value

    def _uses_async_env_sampling_for_config(self) -> bool:
        if self.async_env_sampling == "sync":
            return False
        if self.async_env_sampling == "async":
            return True
        return _supports_async_env_runner_fetch_ready()

    def _update_auto_sync_env_runner_states(self) -> None:
        self._dont_auto_sync_env_runner_states = self._uses_async_env_sampling_for_config()

    @override(CustomSACConfig)
    def validate(self) -> None:
        buffer_type = self.replay_buffer_config.get("type")
        if not _is_supported_async_replay_buffer(buffer_type):
            raise ValueError(
                "AsyncCustomSAC only supports non-prioritized episode replay buffers: "
                "`EpisodeReplayBuffer` or "
                "`rlframework.utils.replay_buffers.BatchEvictEpisodeReplayBuffer`. "
                f"Got `replay_buffer_config['type']="
                f"{_replay_buffer_type_label(buffer_type)}`. "
                "Set `.training(replay_buffer_config={...})` explicitly. "
                "Prioritized replay buffers are not supported because async priority "
                "updates require a remote buffer actor and concurrency handling."
            )

        self.async_env_sampling = self._validate_async_pipeline_mode(
            "env_sampling",
            self.async_env_sampling,
        )
        self.async_learner_training = self._validate_async_pipeline_mode(
            "learner_training",
            self.async_learner_training,
        )
        self.async_pipeline_log_interval = self._validate_non_negative_int_option(
            "pipeline_log_interval",
            self.async_pipeline_log_interval,
        )
        self._update_auto_sync_env_runner_states()

        supports_async_env_sampling = _supports_async_env_runner_fetch_ready()
        if self.async_env_sampling == "async" and not supports_async_env_sampling:
            raise ValueError(
                "AsyncCustomSAC was configured with `env_sampling='async'`, but "
                "the installed Ray version does not expose "
                "`EnvRunnerGroup.foreach_env_runner_async_fetch_ready`. Use Ray "
                ">= 2.54.0, or set "
                "`.algorithm_options({'env_sampling': 'sync'})`."
            )

        if self.async_learner_training == "async":
            if self.async_env_sampling == "sync":
                raise ValueError(
                    "AsyncCustomSAC `learner_training='async'` requires async "
                    "EnvRunner sampling in this first pipeline version. Set "
                    "`env_sampling='async'` or `env_sampling='auto'` on Ray >= 2.54.0."
                )
            if not supports_async_env_sampling:
                raise ValueError(
                    "AsyncCustomSAC `learner_training='async'` requires Ray >= 2.54.0 "
                    "because it depends on the async EnvRunner pipeline."
                )
            if self.num_learners <= 0:
                raise ValueError(
                    "AsyncCustomSAC `learner_training='async'` requires "
                    "`num_learners > 0`, because `LearnerGroup.update(async_update=True)` "
                    "is only supported with remote Learner actors. Configure "
                    "`.learners(num_learners=1, ...)`, or set "
                    "`.algorithm_options({'learner_training': 'sync'})`."
                )

        super().validate()


class AsyncCustomSAC(CustomSAC):
    """Soft Actor-Critic with asynchronous IMPALA-style training pipeline.

    Ray >= 2.54.0 path (async EnvRunner sampling):
    - EnvRunner sampling uses foreach_env_runner_async_fetch_ready (non-blocking)
      instead of synchronous_parallel_sample (blocking).
    - Learner updates are configurable:
      - sync: learner_group.update(async_update=False), while EnvRunners already
        sample the next fragment.
      - async: learner_group.update(async_update=True), returning immediately and
        fetching prior results.
    - Train credit system prevents training on stale replay when no new samples arrived.
    - Connector states are collected from EnvRunners and broadcast on weight sync.
    - No priority updates (uses plain EpisodeReplayBuffer).

    Ray < 2.54.0 path (sync fallback):
    - Delegates to CustomSAC/SAC/DQN's parent synchronous training_step().
    - Train credit is not used in this path.

    NOTE: The local replay buffer is NOT thread-safe. On Ray >= 2.54.0, async
    sampling writes to it from the main training thread only (foreach_env_runner_async_fetch_ready
    returns in the same thread), so this is safe. On Ray < 2.54.0, sampling is
    synchronous so this is also safe. The buffer should NOT be shared across threads.
    """

    def __init__(self, config: AsyncCustomSACConfig, *args: Any, **kwargs: Any) -> None:
        super().__init__(config, *args, **kwargs)
        runtime_config = cast(Any, self.config)

        # Detect Ray version for API routing.
        # foreach_env_runner_async_fetch_ready is only available in Ray >= 2.54.
        # Using hasattr as the detection mechanism rather than ray.__version__
        # string parsing, since the API presence is the actual requirement.
        self._supports_async_env_sampling: bool = hasattr(
            self.env_runner_group, "foreach_env_runner_async_fetch_ready"
        )
        self._use_async_env_sampling = self._resolve_async_env_sampling_runtime(
            runtime_config,
        )
        self._use_async_learner_training = self._resolve_async_learner_training_runtime(
            runtime_config,
        )
        # Backward-compatible internal alias used by older tests/extensions.
        self._is_async_sampling = self._use_async_env_sampling

        if self._use_async_learner_training and runtime_config.num_learners <= 0:
            raise ValueError(
                "AsyncCustomSAC async learner training requires `num_learners > 0`. "
                "Configure `.learners(num_learners=1, ...)`, or set "
                "`.algorithm_options({'learner_training': 'sync'})`."
            )

        self._train_credit = 0.0
        self._latest_connector_states_by_actor: dict[Any, Any] = {}
        self._async_training_step_count = 0

    @classmethod
    @override(CustomSAC)
    def get_default_config(cls) -> AsyncCustomSACConfig:
        return AsyncCustomSACConfig()

    # =======================================================================
    # Runtime mode resolution
    # =======================================================================

    def _resolve_async_env_sampling_runtime(self, runtime_config: Any) -> bool:
        mode = getattr(runtime_config, "async_env_sampling", "auto")
        if mode == "sync":
            return False
        if mode == "async":
            if not self._supports_async_env_sampling:
                raise ValueError(
                    "AsyncCustomSAC was configured with `env_sampling='async'`, "
                    "but this Ray runtime does not expose "
                    "`foreach_env_runner_async_fetch_ready`."
                )
            return True
        return self._supports_async_env_sampling

    def _resolve_async_learner_training_runtime(self, runtime_config: Any) -> bool:
        mode = getattr(runtime_config, "async_learner_training", "auto")
        if mode == "sync":
            return False
        if mode == "async":
            if not self._use_async_env_sampling:
                raise ValueError(
                    "AsyncCustomSAC `learner_training='async'` requires async EnvRunner sampling."
                )
            if runtime_config.num_learners <= 0:
                raise ValueError(
                    "AsyncCustomSAC `learner_training='async'` requires `num_learners > 0`."
                )
            return True

        # Auto enables async Learner training only when Env sampling is async and
        # Learners are remote. A local Learner falls back to sync training.
        return self._use_async_env_sampling and runtime_config.num_learners > 0

    # =======================================================================
    # Ray >= 2.54.0: Async EnvRunner sampling path
    # =======================================================================

    def _drain_learner_results(self) -> None:
        """Drain any previously-ready learner results without issuing a new update.

        This prevents learner results from piling up when sampling is faster than
        training (credit ran out). We drain them here so that:
        1. Metrics are eventually logged.
        2. Any new RLModule weights are propagated to EnvRunners promptly.

        This does NOT sample from the replay buffer or consume train credit.
        """
        if not self._use_async_learner_training:
            return

        runtime_config = cast(Any, self.config)
        learner_group = cast(Any, self.learner_group)

        if runtime_config.num_learners <= 0:
            return

        worker_manager = getattr(learner_group, "_worker_manager", None)
        if worker_manager is None:
            return

        before_inflight = worker_manager.num_outstanding_async_reqs()
        remote_results = worker_manager.fetch_ready_async_reqs(
            timeout_seconds=0.0,
        )
        ready_results = learner_group._get_results(remote_results)
        metrics_logger = cast(Any, self.metrics)
        metrics_logger.log_value(
            "async_sac_learner_results_drained",
            len(ready_results),
            window=1,
        )
        metrics_logger.log_value(
            "async_sac_learner_inflight_before_drain",
            before_inflight,
            window=1,
        )
        metrics_logger.log_value(
            "async_sac_learner_inflight_after_drain",
            worker_manager.num_outstanding_async_reqs(),
            window=1,
        )
        if ready_results:
            self._process_learner_results(ready_results)

    def training_step(self) -> None:
        """One training step, using async when available and parent sync otherwise."""
        if not self._use_async_env_sampling:
            return super().training_step()

        self.on_before_training_step()

        self._training_step_async()

        metrics_logger = cast(Any, self.metrics)
        result = metrics_logger.peek()
        existing_metric_keys = set(result) if isinstance(result, dict) else set()
        result = self.on_after_training_step(result)
        if result:
            for key, value in result.items():
                if key not in existing_metric_keys and isinstance(value, (int, float)):
                    metrics_logger.log_value(key, value, window=1)

    def _training_step_async(self) -> None:
        """Async EnvRunner training step for Ray >= 2.54.0.

        Pipeline:
        1. Fetch ready EnvRunner samples and issue next async sample.
           Track connector_states for later broadcast.
        2. If Learner training is async, drain previously-ready learner results.
        3. Accumulate train credit only after warmup.
        4. With available credit: run or issue at most one learner update per call.
        5. Call framework hooks and return results.

        Only one learner update per training_step prevents in-flight saturation
        and keeps EnvRunner ready-result polling frequent.
        """
        self._async_training_step_count += 1
        new_env_steps = self._fetch_ready_samples_and_reissue()

        if self._use_async_learner_training:
            self._drain_learner_results()

        runtime_config = cast(Any, self.config)
        current_ts = self._current_sampled_timesteps()
        credit_added = 0.0
        warmup_blocked = current_ts < runtime_config.num_steps_sampled_before_learning_starts
        cast(Any, self.metrics).log_value(
            "async_sac_warmup_blocked",
            int(warmup_blocked),
            window=1,
        )

        if not warmup_blocked:
            credit_added = self._calc_credit_increment(new_env_steps)
            self._train_credit += credit_added

            if self._use_async_learner_training:
                self._maybe_issue_async_learner_update()
            else:
                self._maybe_run_sync_learner_update()

        self._record_async_pipeline_metrics(
            new_env_steps=new_env_steps,
            current_ts=current_ts,
            credit_added=credit_added,
        )
        self._maybe_log_async_pipeline_snapshot(
            new_env_steps=new_env_steps,
            current_ts=current_ts,
        )

    def _maybe_run_sync_learner_update(self) -> None:
        """Run one synchronous Learner update if train credit is available."""
        metrics_logger = cast(Any, self.metrics)
        if self._train_credit < 1.0:
            metrics_logger.log_value(
                "async_sac_train_credit_blocked",
                1,
                window=1,
            )
            return
        metrics_logger.log_value(
            "async_sac_train_credit_blocked",
            0,
            window=1,
        )

        learner_group = cast(Any, self.learner_group)
        episodes = self._sample_from_replay_buffer()

        learner_results = learner_group.update(
            episodes=episodes,
            async_update=False,
            return_state=True,
            timesteps=self._learner_timesteps(),
        )

        self._process_learner_results(learner_results)
        self._train_credit -= 1.0

        metrics_logger.log_value(
            "async_sac_sync_learner_updates",
            1,
            window=1,
        )
        metrics_logger.log_value(
            "async_sac_train_credit_spent",
            1.0,
            window=1,
        )

    def _maybe_issue_async_learner_update(self) -> None:
        """Issue one asynchronous Learner update if credit and capacity allow."""
        metrics_logger = cast(Any, self.metrics)
        if self._train_credit < 1.0:
            metrics_logger.log_value(
                "async_sac_train_credit_blocked",
                1,
                window=1,
            )
            return
        metrics_logger.log_value(
            "async_sac_train_credit_blocked",
            0,
            window=1,
        )

        runtime_config = cast(Any, self.config)
        learner_group = cast(Any, self.learner_group)

        # Guard: don't overflow learner in-flight queue. With
        # max_requests_in_flight_per_learner=1 and async_update=True, a second
        # concurrent update would be rejected by FaultTolerantActorManager.
        max_in_flight = getattr(runtime_config, "max_requests_in_flight_per_learner", 1)
        before_inflight = learner_group._worker_manager.num_outstanding_async_reqs()
        metrics_logger.log_value(
            "async_sac_learner_inflight_before_issue",
            before_inflight,
            window=1,
        )
        metrics_logger.log_value(
            "async_sac_learner_max_inflight",
            max_in_flight,
            window=1,
        )
        blocked_by_inflight = before_inflight >= max_in_flight
        metrics_logger.log_value(
            "async_sac_learner_blocked_by_inflight",
            int(blocked_by_inflight),
            window=1,
        )
        if blocked_by_inflight:
            return

        episodes = self._sample_from_replay_buffer()

        learner_results = learner_group.update(
            episodes=episodes,
            async_update=True,
            return_state=True,
            timesteps=self._learner_timesteps(),
        )

        self._process_learner_results(learner_results)
        self._train_credit -= 1.0

        metrics_logger.log_value(
            "async_sac_async_learner_updates_issued",
            1,
            window=1,
        )
        metrics_logger.log_value(
            "async_sac_train_credit_spent",
            1.0,
            window=1,
        )
        metrics_logger.log_value(
            "async_sac_learner_inflight_after_issue",
            learner_group._worker_manager.num_outstanding_async_reqs(),
            window=1,
        )

    def _learner_timesteps(self) -> dict[str, Any]:
        """Build the timestep metadata passed into LearnerGroup.update()."""
        metrics_logger = cast(Any, self.metrics)
        return {
            NUM_ENV_STEPS_SAMPLED_LIFETIME: metrics_logger.peek(
                (ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME),
                default=0,
            ),
            NUM_AGENT_STEPS_SAMPLED_LIFETIME: metrics_logger.peek(
                (ENV_RUNNER_RESULTS, NUM_AGENT_STEPS_SAMPLED_LIFETIME),
                default={},
            ),
        }

    def _record_async_pipeline_metrics(
        self,
        *,
        new_env_steps: int,
        current_ts: int,
        credit_added: float,
    ) -> None:
        """Record lightweight pipeline health metrics on each async step."""
        metrics_logger = cast(Any, self.metrics)
        metrics_logger.log_value(
            "async_sac_pipeline_step",
            self._async_training_step_count,
            window=1,
        )
        metrics_logger.log_value(
            "async_sac_env_sampling_async",
            int(self._use_async_env_sampling),
            window=1,
        )
        metrics_logger.log_value(
            "async_sac_learner_training_async",
            int(self._use_async_learner_training),
            window=1,
        )
        metrics_logger.log_value(
            "async_sac_current_sampled_timesteps",
            current_ts,
            window=1,
        )
        metrics_logger.log_value(
            "async_sac_train_credit",
            self._train_credit,
            window=1,
        )
        metrics_logger.log_value(
            "async_sac_train_credit_added",
            credit_added,
            window=1,
        )
        metrics_logger.log_value(
            "async_sac_new_env_steps",
            new_env_steps,
            window=1,
        )
        metrics_logger.log_value(
            "async_sac_cached_connector_states",
            len(self._latest_connector_states_by_actor),
            window=1,
        )
        metrics_logger.log_value(
            "async_sac_env_sample_inflight",
            self._env_sample_inflight(),
            window=1,
        )
        metrics_logger.log_value(
            "async_sac_learner_inflight",
            self._learner_inflight(),
            window=1,
        )

    def _maybe_log_async_pipeline_snapshot(
        self,
        *,
        new_env_steps: int,
        current_ts: int,
    ) -> None:
        """Emit an optional driver log line for async pipeline diagnosis."""
        runtime_config = cast(Any, self.config)
        interval = getattr(runtime_config, "async_pipeline_log_interval", 0)
        if interval <= 0 or self._async_training_step_count % interval != 0:
            return

        logger.info(
            "AsyncCustomSAC pipeline step=%s env_async=%s learner_async=%s "
            "new_env_steps=%s sampled_ts=%s train_credit=%.3f "
            "env_inflight=%s learner_inflight=%s cached_connector_states=%s",
            self._async_training_step_count,
            self._use_async_env_sampling,
            self._use_async_learner_training,
            new_env_steps,
            current_ts,
            self._train_credit,
            self._env_sample_inflight(),
            self._learner_inflight(),
            len(self._latest_connector_states_by_actor),
        )

    def _env_sample_inflight(self) -> int:
        """Return outstanding async EnvRunner sample requests, if available."""
        env_runner_group = cast(Any, self.env_runner_group)
        worker_manager = getattr(env_runner_group, "_worker_manager", None)
        if worker_manager is None:
            return 0
        try:
            return worker_manager.num_outstanding_async_reqs(tag=_ASYNC_SAMPLE_TAG)
        except TypeError:
            return worker_manager.num_outstanding_async_reqs()

    def _learner_inflight(self) -> int:
        """Return outstanding async Learner update requests, if available."""
        if not self._use_async_learner_training:
            return 0
        learner_group = cast(Any, self.learner_group)
        worker_manager = getattr(learner_group, "_worker_manager", None)
        if worker_manager is None:
            return 0
        return worker_manager.num_outstanding_async_reqs()

    def _fetch_ready_samples_and_reissue(self) -> int:
        """Fetch previously ready EnvRunner samples and immediately issue the next
        async sample call.

        This is the core of the IMPALA-style pipeline: we never block waiting
        for a sample to finish. Instead, each training_step call:
        1. Collects whatever episodes are ready from the prior async sample.
        2. Starts the next async sample (to be collected next call).
        3. Collects connector states for later weight broadcast.

        With max_requests_in_flight_per_env_runner=1, each EnvRunner always
        has exactly one pending sample request.

        Returns:
            Total number of env steps newly added to the replay buffer this call.
        """
        total_new_env_steps = 0
        ready_connector_state_count = 0
        env_runner_group = cast(Any, self.env_runner_group)
        metrics_logger = cast(Any, self.metrics)
        replay_buffer = cast(Any, self.local_replay_buffer)
        num_healthy_remote_workers = env_runner_group.num_healthy_remote_workers()

        metrics_logger.log_value(
            "async_sac_remote_env_runners_healthy",
            num_healthy_remote_workers,
            window=1,
        )

        if num_healthy_remote_workers > 0:
            with metrics_logger.log_time((TIMERS, ENV_RUNNER_SAMPLING_TIMER)):
                results = env_runner_group.foreach_env_runner_async_fetch_ready(
                    func="sample_get_state_and_metrics",
                    tag=_ASYNC_SAMPLE_TAG,
                    timeout_seconds=0.0,
                    return_actor_ids=True,
                )

            # Fetch and process results.
            for actor_id, (episodes_ref, connector_state, metrics) in results:
                episodes = ray.get(episodes_ref)
                env_step_count = self._count_episode_env_steps(episodes)
                total_new_env_steps += env_step_count
                self._latest_connector_states_by_actor[actor_id] = connector_state
                ready_connector_state_count += 1

                with metrics_logger.log_time((TIMERS, REPLAY_BUFFER_ADD_DATA_TIMER)):
                    replay_buffer.add(episodes)

                metrics_logger.aggregate([metrics], key=ENV_RUNNER_RESULTS)
        else:
            env_runner = cast(Any, self.env_runner)
            if env_runner is None:
                return 0

            # No remote workers: sample locally (synchronous, non-blocking).
            with metrics_logger.log_time((TIMERS, ENV_RUNNER_SAMPLING_TIMER)):
                episodes = env_runner.sample()
                env_runner_metrics = env_runner.get_metrics()
                env_step_count = self._count_episode_env_steps(episodes)
                total_new_env_steps += env_step_count

            with metrics_logger.log_time((TIMERS, REPLAY_BUFFER_ADD_DATA_TIMER)):
                replay_buffer.add(episodes)

            metrics_logger.aggregate([env_runner_metrics], key=ENV_RUNNER_RESULTS)

            local_connector_state = env_runner.get_state(
                components=[
                    COMPONENT_ENV_TO_MODULE_CONNECTOR,
                    COMPONENT_MODULE_TO_ENV_CONNECTOR,
                ]
            )
            self._latest_connector_states_by_actor["local"] = local_connector_state
            ready_connector_state_count = 1

        metrics_logger.log_value(
            "async_sac_env_steps_added",
            total_new_env_steps,
            window=1,
        )
        metrics_logger.log_value(
            "async_sac_env_ready_results",
            ready_connector_state_count,
            window=1,
        )
        metrics_logger.log_value(
            "async_sac_env_sample_inflight_after_reissue",
            self._env_sample_inflight(),
            window=1,
        )
        return total_new_env_steps

    def _sample_from_replay_buffer(self) -> list[Any]:
        """Sample a batch of episodes from the local replay buffer."""
        runtime_config = cast(Any, self.config)
        metrics_logger = cast(Any, self.metrics)
        replay_buffer = cast(Any, self.local_replay_buffer)

        with metrics_logger.log_time((TIMERS, REPLAY_BUFFER_SAMPLE_TIMER)):
            episodes = replay_buffer.sample(
                num_items=runtime_config.total_train_batch_size,
                n_step=runtime_config.n_step,
                batch_length_T=(
                    self._module_is_stateful * runtime_config.model_config.get("max_seq_len", 0)
                ),
                lookback=int(self._module_is_stateful),
                min_batch_length_T=(
                    runtime_config.burn_in_len if hasattr(runtime_config, "burn_in_len") else 0
                ),
                gamma=runtime_config.gamma,
                sample_episodes=True,
            )

            replay_buffer_results = replay_buffer.get_metrics()
            metrics_logger.aggregate([replay_buffer_results], key=REPLAY_BUFFER_RESULTS)

        metrics_logger.log_value(
            "async_sac_replay_sampled_episodes",
            len(episodes),
            window=1,
        )
        return episodes

    def _process_learner_results(self, learner_results: list[Any]) -> None:
        """Handle learner update results: aggregate metrics and sync weights.

        When async_update=True:
        - learner_results contains results from the PRIOR update call.
        - The current update call was queued and will return next iteration.
        This creates a pipeline: training overlaps with the next sampling.

        After processing, connector states collected during sampling are
        broadcast to all EnvRunners so that filters / ConnectorV2 state
        stays in sync across workers.
        """
        if not learner_results:
            return

        rl_module_state = None

        for result_from_learner in learner_results:
            rl_module_state = result_from_learner.pop(
                "_rl_module_state_after_update",
                rl_module_state,
            )

            for module_id, module_result in list(result_from_learner.items()):
                if module_id in (ALL_MODULES, "__all__"):
                    continue
                if isinstance(module_result, dict):
                    module_result.pop(TD_ERROR_KEY, None)

        metrics_logger = cast(Any, self.metrics)
        env_runner_group = cast(Any, self.env_runner_group)
        runtime_config = cast(Any, self.config)
        metrics_logger.aggregate(learner_results, key=LEARNER_RESULTS)
        metrics_logger.log_value(
            "async_sac_learner_results_processed",
            len(learner_results),
            window=1,
        )

        # Use the latest connector states observed from each EnvRunner. Learner
        # results may become ready on a step with no ready EnvRunner samples.
        connector_states = list(self._latest_connector_states_by_actor.values())
        if rl_module_state is not None:
            metrics_logger.log_value(
                "async_sac_weight_syncs",
                1,
                window=1,
            )
            metrics_logger.log_value(
                "async_sac_connector_states_synced",
                len(connector_states),
                window=1,
            )
            env_runner_group.sync_env_runner_states(
                config=runtime_config,
                connector_states=connector_states,
                rl_module_state=rl_module_state,
                env_steps_sampled=metrics_logger.peek(
                    (ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME),
                    default=0,
                ),
                env_to_module=self.env_to_module_connector,
                module_to_env=self.module_to_env_connector,
            )

    def _calc_credit_increment(self, new_env_steps: int) -> float:
        """Convert newly-sampled env steps into a fractional train credit.

        One "credit" represents the right to run one train batch (total_train_batch_size).

        - If ``training_intensity`` is explicitly set by the user, credit is computed
          directly as ``new_env_steps * training_intensity / total_train_batch_size``.
          This avoids the mismatch between RLlib's ``calculate_rr_weights`` (which
          assumes synchronous parallel sampling across all workers) and our async
          pipeline (which only uses remote EnvRunners).
        - Otherwise, we fall back to the ``calculate_rr_weights`` ratio, using only
          the remote worker count for ``rollout_steps_per_store``.
        """
        if new_env_steps <= 0:
            return 0.0

        runtime_config = cast(Any, self.config)
        env_runner_group = cast(Any, self.env_runner_group)

        if runtime_config.training_intensity is not None:
            return (
                new_env_steps
                * runtime_config.training_intensity
                / runtime_config.total_train_batch_size
            )

        store_weight, train_weight = calculate_rr_weights(runtime_config)

        rollout_steps_per_store = (
            runtime_config.get_rollout_fragment_length()
            * runtime_config.num_envs_per_env_runner
            * max(env_runner_group.num_healthy_remote_workers(), 1)
        )
        env_steps_per_cycle = rollout_steps_per_store * max(store_weight, 1)
        if env_steps_per_cycle == 0:
            return 0.0

        return new_env_steps * train_weight / env_steps_per_cycle

    # =======================================================================
    # Shared helpers
    # =======================================================================

    def _count_episode_env_steps(self, episodes: list[Any]) -> int:
        """Count total env steps across a list of episodes.

        SingleAgentEpisode / MultiAgentEpisode expose ``env_steps()`` as a method,
        not a ``length`` property. Falls back to ``len(ep)`` for generic sequences.
        """
        total = 0
        for ep in episodes:
            if hasattr(ep, "env_steps"):
                steps = ep.env_steps
                total += steps() if callable(steps) else steps
            else:
                total += len(ep)
        return total

    def _current_sampled_timesteps(self) -> int:
        """Return the total env or agent steps sampled so far."""
        runtime_config = cast(Any, self.config)
        metrics_logger = cast(Any, self.metrics)

        if runtime_config.count_steps_by == "agent_steps":
            agent_steps = metrics_logger.peek(
                (ENV_RUNNER_RESULTS, NUM_AGENT_STEPS_SAMPLED_LIFETIME),
                default={},
            )
            return sum(agent_steps.values()) if isinstance(agent_steps, dict) else agent_steps

        return metrics_logger.peek(
            (ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME),
            default=0,
        )
