"""Unit tests for AsyncCustomSAC's async pipeline scheduling helpers."""

from types import SimpleNamespace


class _Timer:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _Metrics:
    def __init__(self, peeks=None):
        self.peeks = peeks or {}
        self.values = []
        self.aggregates = []

    def log_time(self, key):
        return _Timer()

    def log_value(self, key, value, window=None):
        self.values.append((key, value, window))

    def aggregate(self, items, key=None):
        self.aggregates.append((items, key))

    def peek(self, key=None, default=None):
        if key is None:
            return {}
        return self.peeks.get(key, default)

    def latest(self, key):
        for logged_key, value, _window in reversed(self.values):
            if logged_key == key:
                return value
        raise KeyError(key)


class _Episode:
    def __init__(self, env_steps):
        self._env_steps = env_steps

    def env_steps(self):
        return self._env_steps


class _ReplayBuffer:
    def __init__(self):
        self.added = []

    def add(self, episodes):
        self.added.append(episodes)


class _WorkerManager:
    def __init__(self, outstanding=0, ready_results=None):
        self.outstanding = outstanding
        self.ready_results = ready_results or []
        self.fetch_calls = []

    def num_outstanding_async_reqs(self, tag=None):
        return self.outstanding

    def fetch_ready_async_reqs(self, **kwargs):
        self.fetch_calls.append(kwargs)
        return self.ready_results


class _EnvRunnerGroup:
    def __init__(self, *, ready_results=None, healthy_remote_workers=1, worker_manager=None):
        self.ready_results = ready_results or []
        self.healthy_remote_workers = healthy_remote_workers
        self._worker_manager = worker_manager or _WorkerManager()
        self.async_fetch_calls = []
        self.fetch_ready_calls = []
        self.sync_calls = []

    def num_healthy_remote_workers(self):
        return self.healthy_remote_workers

    def foreach_env_runner_async_fetch_ready(self, **kwargs):
        self.async_fetch_calls.append(kwargs)
        return self.ready_results

    def fetch_ready_async_reqs(self, **kwargs):
        self.fetch_ready_calls.append(kwargs)
        return self.ready_results

    def sync_env_runner_states(self, **kwargs):
        self.sync_calls.append(kwargs)


class _LearnerGroup:
    def __init__(
        self,
        *,
        worker_manager=None,
        update_results=None,
        ready_results=None,
        state=None,
    ):
        self._worker_manager = worker_manager or _WorkerManager()
        self.update_results = update_results if update_results is not None else []
        self.ready_results = ready_results if ready_results is not None else []
        self.state = state if state is not None else {}
        self.update_calls = []
        self.get_state_calls = []

    def update(self, **kwargs):
        self.update_calls.append(kwargs)
        if kwargs.get("async_update"):
            self._worker_manager.outstanding += 1
        return self.update_results

    def _get_results(self, remote_results):
        return self.ready_results

    def get_state(self, **kwargs):
        self.get_state_calls.append(kwargs)
        return self.state


def _make_algo():
    from rlframework.algorithms.async_sac import AsyncCustomSAC

    algo = object.__new__(AsyncCustomSAC)
    algo.config = SimpleNamespace(
        num_learners=1,
        max_requests_in_flight_per_learner=1,
        async_max_sync_learner_updates_per_step=None,
        async_pipeline_log_interval=0,
    )
    algo.metrics = _Metrics()
    algo._train_credit = 0.0
    algo._use_async_env_sampling = True
    algo._use_async_learner_training = True
    algo._latest_connector_states_by_actor = {}
    algo._async_training_step_count = 0
    algo.env_to_module_connector = object()
    algo.module_to_env_connector = object()
    return algo


def test_async_fetch_ready_samples_adds_to_replay_and_caches_connector_state(monkeypatch):
    from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS

    from rlframework.algorithms import async_sac
    from rlframework.algorithms.async_sac import AsyncCustomSAC

    episodes = [_Episode(3), _Episode(4)]
    connector_state = {"connector": "state-1"}
    env_metrics = {"episode_return_mean": 1.0}
    env_group = _EnvRunnerGroup(
        ready_results=[("actor-1", (episodes, connector_state, env_metrics))],
        worker_manager=_WorkerManager(outstanding=1),
    )
    replay_buffer = _ReplayBuffer()
    algo = _make_algo()
    algo.env_runner_group = env_group
    algo.local_replay_buffer = replay_buffer

    monkeypatch.setattr(async_sac.ray, "get", lambda ref: ref)

    new_steps = AsyncCustomSAC._fetch_ready_samples_and_reissue(algo)

    assert new_steps == 7
    assert replay_buffer.added == [episodes]
    assert algo._latest_connector_states_by_actor == {"actor-1": connector_state}
    assert env_group.async_fetch_calls == [
        {
            "func": "sample_get_state_and_metrics",
            "tag": "async_sac_sample",
            "timeout_seconds": 0.0,
            "return_actor_ids": True,
        }
    ]
    assert algo.metrics.aggregates == [([env_metrics], ENV_RUNNER_RESULTS)]
    assert algo.metrics.latest("async_sac_env_ready_results") == 1
    assert algo.metrics.latest("async_sac_env_steps_added") == 7
    assert algo.metrics.latest("async_sac_env_sample_inflight_after_reissue") == 1


def test_async_fetch_ready_samples_batches_replay_adds(monkeypatch):
    from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS

    from rlframework.algorithms import async_sac
    from rlframework.algorithms.async_sac import AsyncCustomSAC

    episodes_1 = [_Episode(3)]
    episodes_2 = [_Episode(4), _Episode(5)]
    connector_state_1 = {"connector": "state-1"}
    connector_state_2 = {"connector": "state-2"}
    env_metrics_1 = {"episode_return_mean": 1.0}
    env_metrics_2 = {"episode_return_mean": 2.0}
    env_group = _EnvRunnerGroup(
        ready_results=[
            ("actor-1", (episodes_1, connector_state_1, env_metrics_1)),
            ("actor-2", (episodes_2, connector_state_2, env_metrics_2)),
        ],
        worker_manager=_WorkerManager(outstanding=2),
    )
    replay_buffer = _ReplayBuffer()
    algo = _make_algo()
    algo.env_runner_group = env_group
    algo.local_replay_buffer = replay_buffer

    monkeypatch.setattr(async_sac.ray, "get", lambda ref: ref)

    new_steps = AsyncCustomSAC._fetch_ready_samples_and_reissue(algo)

    assert new_steps == 12
    assert replay_buffer.added == [episodes_1 + episodes_2]
    assert algo._latest_connector_states_by_actor == {
        "actor-1": connector_state_1,
        "actor-2": connector_state_2,
    }
    assert algo.metrics.aggregates == [
        ([env_metrics_1], ENV_RUNNER_RESULTS),
        ([env_metrics_2], ENV_RUNNER_RESULTS),
    ]
    assert algo.metrics.latest("async_sac_env_ready_results") == 2
    assert algo.metrics.latest("async_sac_env_steps_added") == 12


def test_async_fetch_ready_samples_only_does_not_reissue(monkeypatch):
    from ray.rllib.utils.metrics import ENV_RUNNER_RESULTS

    from rlframework.algorithms import async_sac
    from rlframework.algorithms.async_sac import AsyncCustomSAC

    episodes = [_Episode(2)]
    connector_state = {"connector": "state-1"}
    env_metrics = {"episode_return_mean": 1.0}
    env_group = _EnvRunnerGroup(
        ready_results=[("actor-1", (episodes, connector_state, env_metrics))],
        worker_manager=_WorkerManager(outstanding=1),
    )
    replay_buffer = _ReplayBuffer()
    algo = _make_algo()
    algo.env_runner_group = env_group
    algo.local_replay_buffer = replay_buffer

    monkeypatch.setattr(async_sac.ray, "get", lambda ref: ref)

    new_steps = AsyncCustomSAC._fetch_ready_samples_only(algo, timeout_seconds=0.5)

    assert new_steps == 2
    assert replay_buffer.added == [episodes]
    assert algo._latest_connector_states_by_actor == {"actor-1": connector_state}
    assert env_group.async_fetch_calls == []
    assert env_group.fetch_ready_calls == [
        {
            "tags": "async_sac_sample",
            "timeout_seconds": 0.5,
        }
    ]
    assert algo.metrics.aggregates == [([env_metrics], ENV_RUNNER_RESULTS)]


def test_async_env_sync_learner_uses_sync_update_when_credit_available():
    from rlframework.algorithms.async_sac import AsyncCustomSAC

    episodes = [_Episode(5)]
    learner_group = _LearnerGroup(update_results=[{"learner_stats": {"loss": 1.0}}])
    processed = []
    algo = _make_algo()
    algo._use_async_learner_training = False
    algo._train_credit = 1.0
    algo.learner_group = learner_group
    algo._sample_from_replay_buffer = lambda: episodes
    algo._process_learner_results = processed.append

    AsyncCustomSAC._maybe_run_sync_learner_update(algo)

    assert learner_group.update_calls[0]["episodes"] == episodes
    assert learner_group.update_calls[0]["async_update"] is False
    assert learner_group.update_calls[0]["return_state"] is True
    assert processed == [[{"learner_stats": {"loss": 1.0}}]]
    assert algo._train_credit == 0.0
    assert algo.metrics.latest("async_sac_sync_learner_updates") == 1
    assert algo.metrics.latest("async_sac_train_credit_spent") == 1.0


def test_sync_learner_drains_multiple_train_credits_and_processes_once():
    from rlframework.algorithms.async_sac import AsyncCustomSAC

    learner_group = _LearnerGroup(update_results=[{"learner_stats": {"loss": 1.0}}])
    processed = []
    algo = _make_algo()
    algo._use_async_learner_training = False
    algo._train_credit = 2.5
    algo.learner_group = learner_group
    algo._sample_from_replay_buffer = lambda: [_Episode(5)]
    algo._process_learner_results = processed.append

    AsyncCustomSAC._maybe_run_sync_learner_update(algo)

    assert len(learner_group.update_calls) == 2
    assert all(call["async_update"] is False for call in learner_group.update_calls)
    assert len(processed) == 1
    assert len(processed[0]) == 2
    assert algo._train_credit == 0.5
    assert algo.metrics.latest("async_sac_sync_learner_updates") == 2
    assert algo.metrics.latest("async_sac_train_credit_spent") == 2.0
    assert algo.metrics.latest("async_sac_sync_learner_update_limit_reached") == 0


def test_sync_learner_respects_per_step_update_limit():
    from rlframework.algorithms.async_sac import AsyncCustomSAC

    learner_group = _LearnerGroup(update_results=[{"learner_stats": {"loss": 1.0}}])
    algo = _make_algo()
    algo.config.async_max_sync_learner_updates_per_step = 2
    algo._use_async_learner_training = False
    algo._train_credit = 3.0
    algo.learner_group = learner_group
    algo._sample_from_replay_buffer = lambda: [_Episode(5)]
    algo._process_learner_results = lambda _results: None

    AsyncCustomSAC._maybe_run_sync_learner_update(algo)

    assert len(learner_group.update_calls) == 2
    assert algo._train_credit == 1.0
    assert algo.metrics.latest("async_sac_sync_learner_updates") == 2
    assert algo.metrics.latest("async_sac_sync_learner_update_limit_reached") == 1


def test_async_env_async_learner_issues_update_when_inflight_has_capacity():
    from rlframework.algorithms.async_sac import AsyncCustomSAC

    episodes = [_Episode(5)]
    learner_group = _LearnerGroup(
        worker_manager=_WorkerManager(outstanding=0),
        update_results=[],
    )
    processed = []
    algo = _make_algo()
    algo._train_credit = 1.0
    algo.learner_group = learner_group
    algo._sample_from_replay_buffer = lambda: episodes
    algo._process_learner_results = processed.append

    AsyncCustomSAC._maybe_issue_async_learner_update(algo)

    assert learner_group.update_calls[0]["episodes"] == episodes
    assert learner_group.update_calls[0]["async_update"] is True
    assert learner_group.update_calls[0]["return_state"] is True
    assert processed == [[]]
    assert algo._train_credit == 0.0
    assert algo.metrics.latest("async_sac_learner_blocked_by_inflight") == 0
    assert algo.metrics.latest("async_sac_async_learner_updates_issued") == 1
    assert algo.metrics.latest("async_sac_learner_inflight_after_issue") == 1


def test_async_learner_issues_until_inflight_capacity_is_full():
    from rlframework.algorithms.async_sac import AsyncCustomSAC

    learner_group = _LearnerGroup(
        worker_manager=_WorkerManager(outstanding=0),
        update_results=[],
    )
    processed = []
    algo = _make_algo()
    algo.config.max_requests_in_flight_per_learner = 2
    algo._train_credit = 3.0
    algo.learner_group = learner_group
    algo._sample_from_replay_buffer = lambda: [_Episode(5)]
    algo._process_learner_results = processed.append

    AsyncCustomSAC._maybe_issue_async_learner_update(algo)

    assert len(learner_group.update_calls) == 2
    assert all(call["async_update"] is True for call in learner_group.update_calls)
    assert processed == [[]]
    assert algo._train_credit == 1.0
    assert algo.metrics.latest("async_sac_async_learner_updates_issued") == 2
    assert algo.metrics.latest("async_sac_train_credit_spent") == 2.0
    assert algo.metrics.latest("async_sac_learner_blocked_by_inflight") == 1
    assert algo.metrics.latest("async_sac_learner_inflight_after_issue") == 2


def test_async_learner_does_not_issue_update_when_inflight_is_full():
    from rlframework.algorithms.async_sac import AsyncCustomSAC

    learner_group = _LearnerGroup(worker_manager=_WorkerManager(outstanding=1))
    algo = _make_algo()
    algo._train_credit = 1.0
    algo.learner_group = learner_group
    algo._sample_from_replay_buffer = lambda: (_ for _ in ()).throw(
        AssertionError("replay sample should not run when learner is full")
    )

    AsyncCustomSAC._maybe_issue_async_learner_update(algo)

    assert learner_group.update_calls == []
    assert algo._train_credit == 1.0
    assert algo.metrics.latest("async_sac_learner_blocked_by_inflight") == 1


def test_async_learner_drain_processes_ready_results_and_records_inflight_metrics():
    from rlframework.algorithms.async_sac import AsyncCustomSAC

    worker_manager = _WorkerManager(outstanding=1, ready_results=["remote-result"])
    learner_results = [{"learner_stats": {"loss": 1.0}}]
    learner_group = _LearnerGroup(
        worker_manager=worker_manager,
        ready_results=learner_results,
    )
    processed = []
    algo = _make_algo()
    algo.config.num_learners = 1
    algo.learner_group = learner_group
    algo._process_learner_results = processed.append

    AsyncCustomSAC._drain_learner_results(algo)

    assert worker_manager.fetch_calls == [{"timeout_seconds": 0.0}]
    assert processed == [learner_results]
    assert algo.metrics.latest("async_sac_learner_results_drained") == 1
    assert algo.metrics.latest("async_sac_learner_inflight_before_drain") == 1


def test_process_learner_results_syncs_latest_connector_states():
    from ray.rllib.utils.metrics import (
        ENV_RUNNER_RESULTS,
        LEARNER_RESULTS,
        NUM_ENV_STEPS_SAMPLED_LIFETIME,
        TD_ERROR_KEY,
    )

    from rlframework.algorithms.async_sac import AsyncCustomSAC

    metrics = _Metrics(peeks={(ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME): 123})
    env_group = _EnvRunnerGroup(healthy_remote_workers=0)
    algo = _make_algo()
    algo.metrics = metrics
    algo.env_runner_group = env_group
    algo._latest_connector_states_by_actor = {
        "actor-1": {"connector": "state-1"},
        "actor-2": {"connector": "state-2"},
    }
    learner_result = {
        "_rl_module_state_after_update": {"weights": "state"},
        "default_policy": {TD_ERROR_KEY: [1.0], "loss": 2.0},
    }

    AsyncCustomSAC._process_learner_results(algo, [learner_result])

    assert learner_result == {"default_policy": {"loss": 2.0}}
    assert metrics.aggregates == [([learner_result], LEARNER_RESULTS)]
    assert metrics.latest("async_sac_learner_results_processed") == 1
    assert metrics.latest("async_sac_weight_syncs") == 1
    assert metrics.latest("async_sac_connector_states_synced") == 2
    assert env_group.sync_calls[0]["connector_states"] == [
        {"connector": "state-1"},
        {"connector": "state-2"},
    ]
    assert env_group.sync_calls[0]["rl_module_state"] == {"weights": "state"}
    assert env_group.sync_calls[0]["env_steps_sampled"] == 123


def test_process_learner_results_syncs_only_latest_rl_module_state():
    from rlframework.algorithms.async_sac import AsyncCustomSAC

    env_group = _EnvRunnerGroup(healthy_remote_workers=0)
    algo = _make_algo()
    algo.env_runner_group = env_group
    algo._latest_connector_states_by_actor = {"actor-1": {"connector": "state-1"}}
    learner_results = [
        {"_rl_module_state_after_update": {"weights": "old"}, "default_policy": {}},
        {"_rl_module_state_after_update": {"weights": "new"}, "default_policy": {}},
    ]

    AsyncCustomSAC._process_learner_results(algo, learner_results)

    assert len(env_group.sync_calls) == 1
    assert env_group.sync_calls[0]["rl_module_state"] == {"weights": "new"}


def test_process_learner_results_fetches_state_when_sac_learner_omits_return_state():
    from ray.rllib.core import COMPONENT_LEARNER, COMPONENT_RL_MODULE
    from ray.rllib.utils.metrics import WEIGHTS_SEQ_NO

    from rlframework.algorithms.async_sac import AsyncCustomSAC

    env_group = _EnvRunnerGroup(healthy_remote_workers=0)
    learner_rl_module_state = {
        COMPONENT_RL_MODULE: {"weights": "state"},
        WEIGHTS_SEQ_NO: 7,
    }
    learner_group = _LearnerGroup(
        state={COMPONENT_LEARNER: learner_rl_module_state},
    )
    algo = _make_algo()
    algo._use_async_learner_training = False
    algo.env_runner_group = env_group
    algo.learner_group = learner_group
    algo._latest_connector_states_by_actor = {"actor-1": {"connector": "state-1"}}
    learner_result = {"default_policy": {"loss": 2.0}}

    AsyncCustomSAC._process_learner_results(algo, [learner_result])

    assert learner_group.get_state_calls == [
        {
            "components": f"{COMPONENT_LEARNER}/{COMPONENT_RL_MODULE}",
            "inference_only": True,
        }
    ]
    assert algo.metrics.latest("async_sac_rl_module_state_missing") == 1
    assert algo.metrics.latest("async_sac_rl_module_state_fallback_fetches") == 1
    assert env_group.sync_calls[0]["rl_module_state"] == learner_rl_module_state


def test_process_learner_results_defers_state_fetch_while_async_update_inflight():
    from ray.rllib.core import COMPONENT_LEARNER, COMPONENT_RL_MODULE

    from rlframework.algorithms.async_sac import AsyncCustomSAC

    env_group = _EnvRunnerGroup(healthy_remote_workers=0)
    learner_group = _LearnerGroup(
        worker_manager=_WorkerManager(outstanding=1),
        state={COMPONENT_LEARNER: {COMPONENT_RL_MODULE: {"weights": "state"}}},
    )
    algo = _make_algo()
    algo.env_runner_group = env_group
    algo.learner_group = learner_group
    algo._latest_connector_states_by_actor = {"actor-1": {"connector": "state-1"}}

    AsyncCustomSAC._process_learner_results(algo, [{"default_policy": {"loss": 2.0}}])

    assert learner_group.get_state_calls == []
    assert env_group.sync_calls == []
    assert algo.metrics.latest("async_sac_rl_module_state_missing") == 1
    assert algo.metrics.latest("async_sac_rl_module_state_fallback_fetches") == 0
