"""AsyncCustomSAC - Asynchronous SAC using IMPALA-style pipeline.

Ray version detection and dual-path support:
- Ray >= 2.54.0: uses foreach_env_runner_async_fetch_ready for non-blocking sampling
  and learner_group.update(async_update=True) for non-blocking training.
- Ray < 2.54.0: delegates to CustomSAC's parent synchronous training loop.

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
    )
    algo = config.build()
    for _ in range(100):
        results = algo.train()

Prerequisites:
- replay_buffer_config["type"] must be EpisodeReplayBuffer or
  BatchEvictEpisodeReplayBuffer. Prioritization is NOT supported in this async
  pipeline (v1).
"""

from typing import Any

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
from rlframework.utils.replay_buffers import BatchEvictEpisodeReplayBuffer

_ASYNC_SAMPLE_TAG = "async_sac_sample"
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

        # AsyncCustomSAC owns EnvRunner state/weight sync on Ray versions that expose
        # the tagged async EnvRunner API. Older Ray versions keep RLlib's auto-sync.
        self._dont_auto_sync_env_runner_states = _supports_async_env_runner_fetch_ready()

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

        super().validate()

        if _supports_async_env_runner_fetch_ready() and self.num_learners <= 0:
            raise ValueError(
                "AsyncCustomSAC on Ray versions with async EnvRunner sampling "
                "requires `num_learners > 0`, because "
                "`LearnerGroup.update(async_update=True)` is only supported with "
                "remote Learner actors. Configure `.learners(num_learners=1, ...)`, "
                "or use CustomSAC for synchronous local-learner training."
            )


class AsyncCustomSAC(CustomSAC):
    """Soft Actor-Critic with asynchronous IMPALA-style training pipeline.

    Ray >= 2.54.0 path (async sampling):
    - EnvRunner sampling uses foreach_env_runner_async_fetch_ready (non-blocking)
      instead of synchronous_parallel_sample (blocking).
    - Learner updates use learner_group.update(async_update=True), returning
      immediately and fetching prior results, creating a training/sampling overlap.
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

    def __init__(self, config: AsyncCustomSACConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        # Detect Ray version for API routing.
        # foreach_env_runner_async_fetch_ready is only available in Ray >= 2.54.
        # Using hasattr as the detection mechanism rather than ray.__version__
        # string parsing, since the API presence is the actual requirement.
        self._is_async_sampling: bool = hasattr(
            self.env_runner_group, "foreach_env_runner_async_fetch_ready"
        )
        if self._is_async_sampling and self.config.num_learners <= 0:
            raise ValueError(
                "AsyncCustomSAC async sampling requires `num_learners > 0`. "
                "Configure `.learners(num_learners=1, ...)`, or use CustomSAC "
                "for synchronous local-learner training."
            )

        self._train_credit = 0.0
        self._pending_connector_states = []

    @classmethod
    @override(CustomSAC)
    def get_default_config(cls) -> AsyncCustomSACConfig:
        return AsyncCustomSACConfig()

    # =======================================================================
    # Ray >= 2.54.0: Async sampling + async learner path
    # =======================================================================

    def _drain_learner_results(self) -> None:
        """Drain any previously-ready learner results without issuing a new update.

        This prevents learner results from piling up when sampling is faster than
        training (credit ran out). We drain them here so that:
        1. Metrics are eventually logged.
        2. Any new RLModule weights are propagated to EnvRunners promptly.

        This does NOT sample from the replay buffer or consume train credit.
        """
        if self.config.num_learners <= 0:
            return

        worker_manager = getattr(self.learner_group, "_worker_manager", None)
        if worker_manager is None:
            return

        remote_results = worker_manager.fetch_ready_async_reqs(
            timeout_seconds=0.0,
        )
        ready_results = self.learner_group._get_results(remote_results)
        if ready_results:
            self._process_learner_results(ready_results)

    def training_step(self) -> None:
        """One training step, using async when available and parent sync otherwise."""
        if not self._is_async_sampling:
            return super().training_step()

        self.on_before_training_step()

        self._training_step_async()

        result = self.metrics.peek()
        existing_metric_keys = set(result) if isinstance(result, dict) else set()
        result = self.on_after_training_step(result)
        if result:
            for key, value in result.items():
                if key not in existing_metric_keys and isinstance(value, (int, float)):
                    self.metrics.log_value(key, value, window=1)

    def _training_step_async(self) -> None:
        """Async training step for Ray >= 2.54.0.

        Pipeline:
        1. Fetch ready EnvRunner samples and issue next async sample.
           Track connector_states for later broadcast.
        2. Drain any previously-ready learner results (weight sync).
        3. Accumulate train credit only after warmup.
        4. With available credit: issue at most one learner update per call.
        5. Call framework hooks and return results.

        Only one learner update per training_step prevents in-flight saturation
        when max_requests_in_flight_per_learner=1.
        """
        new_env_steps = self._fetch_ready_samples_and_reissue()

        self._drain_learner_results()

        current_ts = self._current_sampled_timesteps()
        if current_ts >= self.config.num_steps_sampled_before_learning_starts:
            self._train_credit += self._calc_credit_increment(new_env_steps)

            if self._train_credit >= 1.0:
                # Guard: don't overflow learner in-flight queue.
                # With max_requests_in_flight_per_learner=1 and async_update=True,
                # a second concurrent update would be rejected by FaultTolerantActorManager.
                # Skip this iteration and let drain handle the result next time.
                if (
                    self.learner_group._worker_manager.num_outstanding_async_reqs()
                    >= self.config.max_requests_in_flight_per_learner
                ):
                    return

                episodes = self._sample_from_replay_buffer()

                learner_results = self.learner_group.update(
                    episodes=episodes,
                    async_update=True,
                    return_state=True,
                    timesteps={
                        NUM_ENV_STEPS_SAMPLED_LIFETIME: self.metrics.peek(
                            (ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME),
                            default=0,
                        ),
                        NUM_AGENT_STEPS_SAMPLED_LIFETIME: self.metrics.peek(
                            (ENV_RUNNER_RESULTS, NUM_AGENT_STEPS_SAMPLED_LIFETIME),
                            default={},
                        ),
                    },
                )

                self._process_learner_results(learner_results)
                self._train_credit -= 1.0

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
        connector_states: list = []

        if self.env_runner_group.num_healthy_remote_workers() > 0:
            with self.metrics.log_time((TIMERS, ENV_RUNNER_SAMPLING_TIMER)):
                results = self.env_runner_group.foreach_env_runner_async_fetch_ready(
                    func="sample_get_state_and_metrics",
                    tag=_ASYNC_SAMPLE_TAG,
                    timeout_seconds=0.0,
                    return_actor_ids=True,
                )

            # Fetch and process results.
            pending_by_actor: dict = {}
            for actor_id, (episodes_ref, connector_state, metrics) in results:
                episodes = ray.get(episodes_ref)
                env_step_count = self._count_episode_env_steps(episodes)
                total_new_env_steps += env_step_count
                pending_by_actor[actor_id] = connector_state

                with self.metrics.log_time((TIMERS, REPLAY_BUFFER_ADD_DATA_TIMER)):
                    self.local_replay_buffer.add(episodes)

                self.metrics.aggregate([metrics], key=ENV_RUNNER_RESULTS)

            connector_states = list(pending_by_actor.values())
        else:
            if self.env_runner is None:
                self._pending_connector_states = []
                return 0

            # No remote workers: sample locally (synchronous, non-blocking).
            with self.metrics.log_time((TIMERS, ENV_RUNNER_SAMPLING_TIMER)):
                episodes = self.env_runner.sample()
                env_runner_metrics = self.env_runner.get_metrics()
                env_step_count = self._count_episode_env_steps(episodes)
                total_new_env_steps += env_step_count

            with self.metrics.log_time((TIMERS, REPLAY_BUFFER_ADD_DATA_TIMER)):
                self.local_replay_buffer.add(episodes)

            self.metrics.aggregate([env_runner_metrics], key=ENV_RUNNER_RESULTS)

            connector_states.append(
                self.env_runner.get_state(
                    components=[
                        COMPONENT_ENV_TO_MODULE_CONNECTOR,
                        COMPONENT_MODULE_TO_ENV_CONNECTOR,
                    ]
                )
            )

        self._pending_connector_states = connector_states
        return total_new_env_steps

    def _sample_from_replay_buffer(self) -> list:
        """Sample a batch of episodes from the local replay buffer."""
        with self.metrics.log_time((TIMERS, REPLAY_BUFFER_SAMPLE_TIMER)):
            episodes = self.local_replay_buffer.sample(
                num_items=self.config.total_train_batch_size,
                n_step=self.config.n_step,
                batch_length_T=(
                    self._module_is_stateful * self.config.model_config.get("max_seq_len", 0)
                ),
                lookback=int(self._module_is_stateful),
                min_batch_length_T=(
                    self.config.burn_in_len if hasattr(self.config, "burn_in_len") else 0
                ),
                gamma=self.config.gamma,
                sample_episodes=True,
            )

            replay_buffer_results = self.local_replay_buffer.get_metrics()
            self.metrics.aggregate([replay_buffer_results], key=REPLAY_BUFFER_RESULTS)

        return episodes

    def _process_learner_results(self, learner_results: list) -> None:
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

        self.metrics.aggregate(learner_results, key=LEARNER_RESULTS)

        # _pending_connector_states is populated by _fetch_ready_samples_and_reissue
        # in the SAME training_step call, so it represents the connector states from
        # the SAME sampling round as the learner results (results from the prior update).
        # On the very first call, _pending_connector_states is [] — weight sync still
        # happens (without connector state broadcast), which is fine.
        if rl_module_state is not None:
            self.env_runner_group.sync_env_runner_states(
                config=self.config,
                connector_states=self._pending_connector_states,
                rl_module_state=rl_module_state,
                env_steps_sampled=self.metrics.peek(
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

        if self.config.training_intensity is not None:
            return (
                new_env_steps * self.config.training_intensity / self.config.total_train_batch_size
            )

        store_weight, train_weight = calculate_rr_weights(self.config)

        rollout_steps_per_store = (
            self.config.get_rollout_fragment_length()
            * self.config.num_envs_per_env_runner
            * max(self.env_runner_group.num_healthy_remote_workers(), 1)
        )
        env_steps_per_cycle = rollout_steps_per_store * max(store_weight, 1)
        if env_steps_per_cycle == 0:
            return 0.0

        return new_env_steps * train_weight / env_steps_per_cycle

    # =======================================================================
    # Shared helpers
    # =======================================================================

    def _count_episode_env_steps(self, episodes: list) -> int:
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
        if self.config.count_steps_by == "agent_steps":
            agent_steps = self.metrics.peek(
                (ENV_RUNNER_RESULTS, NUM_AGENT_STEPS_SAMPLED_LIFETIME),
                default={},
            )
            return sum(agent_steps.values()) if isinstance(agent_steps, dict) else agent_steps

        return self.metrics.peek(
            (ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME),
            default=0,
        )
