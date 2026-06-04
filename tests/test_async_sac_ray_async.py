"""Ray-backed async SAC smoke tests.

These tests exercise the real Ray/RLlib async EnvRunner and Learner APIs. They
are skipped on Ray versions that do not expose Ray 2.54's async EnvRunner fetch
API, so the normal Ray 2.49 compatibility test suite remains lightweight.
"""

import time

import pytest
import ray

from rlframework.algorithms.async_sac import (
    AsyncCustomSACConfig,
    _supports_async_env_runner_fetch_ready,
)

pytestmark = pytest.mark.skipif(
    not _supports_async_env_runner_fetch_ready(),
    reason="requires Ray >= 2.54 async EnvRunner API",
)


@pytest.fixture
def ray_runtime():
    if ray.is_initialized():
        ray.shutdown()
    ray.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        log_to_driver=False,
        num_cpus=4,
    )
    try:
        yield
    finally:
        ray.shutdown()


def _async_sac_config(*, learner_training: str, num_learners: int) -> AsyncCustomSACConfig:
    return (
        AsyncCustomSACConfig()
        .environment("Pendulum-v1")
        .training(
            actor_lr=3e-4,
            critic_lr=3e-4,
            alpha_lr=3e-4,
            train_batch_size_per_learner=4,
            num_steps_sampled_before_learning_starts=0,
            training_intensity=1.0,
            replay_buffer_config={
                "type": "EpisodeReplayBuffer",
                "capacity": 200,
            },
        )
        .env_runners(
            num_env_runners=1,
            num_envs_per_env_runner=1,
            rollout_fragment_length=4,
            max_requests_in_flight_per_env_runner=1,
        )
        .learners(
            num_learners=num_learners,
            num_gpus_per_learner=0,
            max_requests_in_flight_per_learner=1,
        )
        .algorithm_options(
            {
                "env_sampling": "async",
                "learner_training": learner_training,
                "pipeline_log_interval": 0,
            }
        )
    )


def _wait_for_async_sample(algo, *, timeout_s: float = 10.0) -> int:
    """Prime the async sampler, then fetch the first ready result without reissuing."""
    total_steps = algo._fetch_ready_samples_and_reissue()
    if algo._latest_connector_states_by_actor:
        return total_steps

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        remaining = max(deadline - time.monotonic(), 0.0)
        total_steps += algo._fetch_ready_samples_only(
            timeout_seconds=min(0.5, remaining),
        )
        if algo._latest_connector_states_by_actor:
            return total_steps

    raise AssertionError(
        f"Timed out waiting for async EnvRunner sample; inflight={algo._env_sample_inflight()}"
    )


def _drain_async_samples_before_stop(algo, *, timeout_s: float = 5.0) -> None:
    """Drain the in-flight async sample request so Ray actor shutdown stays bounded."""
    deadline = time.monotonic() + timeout_s
    while algo._env_sample_inflight() > 0 and time.monotonic() < deadline:
        remaining = max(deadline - time.monotonic(), 0.0)
        algo._fetch_ready_samples_only(timeout_seconds=min(0.5, remaining))


def _stop_algo(algo) -> None:
    try:
        _drain_async_samples_before_stop(algo)
    finally:
        algo.stop()


def test_ray_async_env_runner_fetches_ready_samples(ray_runtime):
    config = _async_sac_config(learner_training="sync", num_learners=0)
    algo = config.build()
    try:
        sampled_steps = _wait_for_async_sample(algo)

        assert algo._use_async_env_sampling is True
        assert algo._use_async_learner_training is False
        assert sampled_steps > 0
        assert algo._latest_connector_states_by_actor
        assert algo._env_sample_inflight() >= 0
    finally:
        _stop_algo(algo)


def test_ray_async_learner_update_creates_inflight_request(ray_runtime):
    config = _async_sac_config(learner_training="async", num_learners=1)
    algo = config.build()
    try:
        sampled_steps = _wait_for_async_sample(algo)
        assert sampled_steps > 0

        algo._train_credit = 1.0
        algo._maybe_issue_async_learner_update()

        assert algo._use_async_env_sampling is True
        assert algo._use_async_learner_training is True
        assert algo._train_credit == 0.0
        assert algo._learner_inflight() == 1
    finally:
        _stop_algo(algo)
