"""Tests for rlframework.config.validators and rlframework.utils."""

import pytest

# ---------------------------------------------------------------------------
# validators
# ---------------------------------------------------------------------------


class TestValidateLR:
    # --- Fixed value: valid cases ---
    def test_fixed_float_ok(self):
        from rlframework.config.validators import validate_lr

        validate_lr(3e-4)
        validate_lr(1.0)
        validate_lr(1)  # int is fine

    def test_fixed_float_zero_raises(self):
        from rlframework.config.validators import validate_lr
        from rlframework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="must be positive"):
            validate_lr(0.0)

    def test_fixed_float_negative_raises(self):
        from rlframework.config.validators import validate_lr
        from rlframework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="must be positive"):
            validate_lr(-1e-4)

    def test_fixed_float_too_high_raises(self):
        from rlframework.config.validators import validate_lr
        from rlframework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match=r">1\.0"):
            validate_lr(2.0)

    # --- Schedule: valid cases ---
    def test_schedule_list_of_lists_ok(self):
        from rlframework.config.validators import validate_lr

        validate_lr([[0, 3e-4], [100000, 1e-4]])
        validate_lr([[0, 1e-3], [50000, 5e-4], [200000, 1e-5]])

    def test_schedule_list_of_tuples_ok(self):
        from rlframework.config.validators import validate_lr

        validate_lr([(0, 3e-4), (100000, 1e-4)])

    # --- Schedule: error cases ---
    def test_schedule_too_short_raises(self):
        from rlframework.config.validators import validate_lr
        from rlframework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="at least 2 entries"):
            validate_lr([[0, 3e-4]])

    def test_schedule_wrong_entry_type_raises(self):
        from rlframework.config.validators import validate_lr
        from rlframework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match=r"\[timestep, lr_value\] pair"):
            validate_lr([[0, 3e-4], "not a pair"])

    def test_schedule_entry_wrong_length_raises(self):
        from rlframework.config.validators import validate_lr
        from rlframework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match=r"\[timestep, lr_value\] pair"):
            validate_lr([[0, 3e-4], [100000]])  # missing lr_value

    def test_schedule_negative_timestep_raises(self):
        from rlframework.config.validators import validate_lr
        from rlframework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="non-negative int"):
            validate_lr([[0, 3e-4], [-1, 1e-4]])

    def test_schedule_float_timestep_raises(self):
        from rlframework.config.validators import validate_lr
        from rlframework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="non-negative int"):
            validate_lr([[0, 3e-4], [1.5, 1e-4]])

    def test_schedule_negative_lr_raises(self):
        from rlframework.config.validators import validate_lr
        from rlframework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="positive number"):
            validate_lr([[0, -1e-4]])

    def test_schedule_lr_too_high_raises(self):
        from rlframework.config.validators import validate_lr
        from rlframework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match=r"> 1\.0"):
            validate_lr([[0, 3.0], [100000, 1.0]])

    def test_schedule_nonincreasing_ts_raises(self):
        from rlframework.config.validators import validate_lr
        from rlframework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="strictly increasing"):
            validate_lr([[0, 3e-4], [50000, 1e-4], [50000, 5e-5]])

    def test_schedule_first_ts_not_zero_raises(self):
        from rlframework.config.validators import validate_lr
        from rlframework.utils.exceptions import ValidationError

        # Multi-element schedule where first entry's ts != 0 hits the sentinel.
        with pytest.raises(ValidationError, match="timestep 0"):
            validate_lr([[1000, 3e-4], [2000, 1e-4]])

    # --- Wrong top-level type ---
    def test_wrong_type_raises(self):
        from rlframework.config.validators import validate_lr
        from rlframework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="schedule list"):
            validate_lr("3e-4")

    def test_field_name_in_error(self):
        from rlframework.config.validators import validate_lr
        from rlframework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            validate_lr(0.0, field="custom.lr")
        assert "custom.lr" in str(exc_info.value)


class TestValidateGamma:
    def test_valid(self):
        from rlframework.config.validators import validate_gamma

        validate_gamma(0.99)
        validate_gamma(1.0)  # upper bound inclusive

    def test_too_low_raises(self):
        from rlframework.config.validators import validate_gamma
        from rlframework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="range"):
            validate_gamma(0.0)

    def test_too_high_raises(self):
        from rlframework.config.validators import validate_gamma
        from rlframework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="range"):
            validate_gamma(1.01)


# ---------------------------------------------------------------------------
# data_utils
# ---------------------------------------------------------------------------


class TestFlattenDict:
    def test_flat_passthrough(self):
        from rlframework.utils.data_utils import flatten_dict

        d = {"a": 1, "b": 2}
        assert flatten_dict(d) == {"a": 1, "b": 2}

    def test_nested_one_level(self):
        from rlframework.utils.data_utils import flatten_dict

        d = {"outer": {"inner": 42}}
        result = flatten_dict(d)
        assert result["outer/inner"] == 42

    def test_nested_deep(self):
        from rlframework.utils.data_utils import flatten_dict

        d = {"a": {"b": {"c": 99}}}
        result = flatten_dict(d)
        assert result["a/b/c"] == 99

    def test_custom_separator(self):
        from rlframework.utils.data_utils import flatten_dict

        d = {"a": {"b": 1}}
        result = flatten_dict(d, sep=".")
        assert "a.b" in result

    def test_prefix(self):
        from rlframework.utils.data_utils import flatten_dict

        d = {"x": 1}
        result = flatten_dict(d, prefix="ns")
        assert "ns/x" in result


class TestUnflattenDict:
    def test_single_key(self):
        from rlframework.utils.data_utils import unflatten_dict

        flat = {"a": 1}
        assert unflatten_dict(flat) == {"a": 1}

    def test_nested(self):
        from rlframework.utils.data_utils import unflatten_dict

        flat = {"a/b": 2}
        result = unflatten_dict(flat)
        assert result == {"a": {"b": 2}}

    def test_roundtrip(self):
        from rlframework.utils.data_utils import flatten_dict, unflatten_dict

        original = {"env": {"reward": 1.0, "steps": 10}, "loss": 0.5}
        assert unflatten_dict(flatten_dict(original)) == original


class TestSafeMean:
    def test_empty_returns_default(self):
        from rlframework.utils.data_utils import safe_mean

        assert safe_mean([]) == 0.0
        assert safe_mean([], default=float("nan")) != safe_mean([], default=float("nan"))

    def test_basic(self):
        from rlframework.utils.data_utils import safe_mean

        assert safe_mean([1.0, 2.0, 3.0]) == pytest.approx(2.0)

    def test_single_element(self):
        from rlframework.utils.data_utils import safe_mean

        assert safe_mean([5.0]) == 5.0


class TestDeepMerge:
    def test_non_overlapping(self):
        from rlframework.utils.data_utils import deep_merge

        result = deep_merge({"a": 1}, {"b": 2})
        assert result == {"a": 1, "b": 2}

    def test_override_scalar(self):
        from rlframework.utils.data_utils import deep_merge

        result = deep_merge({"a": 1}, {"a": 99})
        assert result["a"] == 99

    def test_nested_merge(self):
        from rlframework.utils.data_utils import deep_merge

        base = {"x": {"y": 1, "z": 2}}
        override = {"x": {"y": 10}}
        result = deep_merge(base, override)
        assert result["x"]["y"] == 10
        assert result["x"]["z"] == 2

    def test_base_not_mutated(self):
        from rlframework.utils.data_utils import deep_merge

        base = {"a": {"b": 1}}
        deep_merge(base, {"a": {"b": 99}})
        assert base["a"]["b"] == 1


# ---------------------------------------------------------------------------
# replay_buffers
# ---------------------------------------------------------------------------


class TestReplayBuffers:
    @staticmethod
    def _make_episode(
        episode_id: str,
        start: int,
        length: int,
        *,
        terminated: bool = True,
        truncated: bool = False,
    ):
        from ray.rllib.env.single_agent_episode import SingleAgentEpisode

        return SingleAgentEpisode(
            id_=episode_id,
            observations=list(range(start, start + length + 1)),
            actions=list(range(start, start + length)),
            rewards=[1.0] * length,
            terminated=terminated,
            truncated=truncated,
            t_started=start,
            len_lookback_buffer=0,
        )

    @staticmethod
    def _make_numpy_episode(episode_id: str, length: int = 2):
        import numpy as np
        from ray.rllib.env.single_agent_episode import SingleAgentEpisode

        return SingleAgentEpisode(
            id_=episode_id,
            observations=np.arange((length + 1) * 2, dtype=np.float32).reshape(length + 1, 2),
            infos=[{} for _ in range(length + 1)],
            actions=np.arange(length * 2, dtype=np.float32).reshape(length, 2),
            rewards=np.ones(length, dtype=np.float32),
            terminated=True,
            len_lookback_buffer=0,
        )

    def test_public_replay_buffer_exports(self):
        import rlframework.utils as utils
        from rlframework.utils import replay_buffers

        assert "BatchEvictEpisodeReplayBuffer" in replay_buffers.__all__
        assert "FastSampleEpisodeReplayBuffer" in replay_buffers.__all__
        assert "NumpyIndexedFastSampleEpisodeReplayBuffer" in replay_buffers.__all__
        assert "PrioritizedSumTreeBuffer" in replay_buffers.__all__
        assert "ReservoirReplayBuffer" not in replay_buffers.__all__
        assert hasattr(utils, "BatchEvictEpisodeReplayBuffer")
        assert hasattr(utils, "FastSampleEpisodeReplayBuffer")
        assert hasattr(utils, "NumpyIndexedFastSampleEpisodeReplayBuffer")
        assert not hasattr(replay_buffers, "ReservoirReplayBuffer")
        assert not hasattr(utils, "ReservoirReplayBuffer")

    def test_batch_evict_add_matches_episode_replay_buffer_state(self):
        from ray.rllib.utils.replay_buffers.episode_replay_buffer import EpisodeReplayBuffer

        from rlframework.utils.replay_buffers import BatchEvictEpisodeReplayBuffer

        base_buffer = EpisodeReplayBuffer(capacity=6)
        batch_evict_buffer = BatchEvictEpisodeReplayBuffer(capacity=6)

        add_inputs = [
            self._make_episode("A", 0, 2, terminated=False),
            self._make_episode("B", 0, 2),
            self._make_episode("A", 2, 2),
            [
                self._make_episode("C", 0, 2),
                self._make_episode("D", 0, 2),
                self._make_episode("E", 0, 2),
            ],
        ]

        for episodes in add_inputs:
            assert base_buffer.add(episodes) == batch_evict_buffer.add(episodes)
            assert base_buffer.get_state() == batch_evict_buffer.get_state()
            assert base_buffer.get_num_episodes() == batch_evict_buffer.get_num_episodes()
            assert base_buffer.get_num_timesteps() == batch_evict_buffer.get_num_timesteps()
            assert (
                base_buffer.get_num_episodes_evicted()
                == batch_evict_buffer.get_num_episodes_evicted()
            )
        assert base_buffer.get_metrics() == batch_evict_buffer.get_metrics()

    def test_batch_evict_rebuild_handles_fragmented_episode_indices(self):
        from rlframework.utils.replay_buffers import BatchEvictEpisodeReplayBuffer

        buffer = BatchEvictEpisodeReplayBuffer(capacity=6)
        buffer.add(self._make_episode("A", 0, 2, terminated=False))
        buffer.add(self._make_episode("B", 0, 2))
        buffer.add(self._make_episode("A", 2, 2))

        # Episode A's indices are split around episode B before eviction.
        assert buffer._indices == [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3)]

        buffer.add(self._make_episode("C", 0, 2))

        assert "A" not in buffer.episode_id_to_index
        assert buffer.episode_id_to_index == {"B": 1, "C": 2}
        assert buffer._indices == [(1, 0), (1, 1), (2, 0), (2, 1)]
        assert buffer.get_num_timesteps() == 4

    def test_replay_buffer_can_skip_copying_owned_episodes(self):
        from rlframework.utils.replay_buffers import NumpyIndexedFastSampleEpisodeReplayBuffer

        copied_episode = self._make_episode("A", 0, 2)
        copied_buffer = NumpyIndexedFastSampleEpisodeReplayBuffer(capacity=6)
        copied_buffer.add(copied_episode)

        owned_episode = self._make_episode("B", 0, 2)
        owned_buffer = NumpyIndexedFastSampleEpisodeReplayBuffer(
            capacity=6,
            copy_episodes_on_add=False,
        )
        owned_buffer.add(owned_episode)

        assert copied_buffer.episodes[0] is not copied_episode
        assert owned_buffer.episodes[0] is owned_episode

    @pytest.mark.parametrize(
        "buffer_name",
        ["FastSampleEpisodeReplayBuffer", "NumpyIndexedFastSampleEpisodeReplayBuffer"],
    )
    def test_eviction_diagnostics_report_released_backing(self, buffer_name):
        import gc

        from rlframework.utils import replay_buffers

        buffer_type = getattr(replay_buffers, buffer_name)
        buffer = buffer_type(
            capacity=2,
            copy_episodes_on_add=False,
            track_evicted_episode_refs=True,
        )
        episode = self._make_numpy_episode("A")
        observation_backing = episode.observations.data
        buffer.add(episode)

        del observation_backing
        del episode
        buffer.add(self._make_numpy_episode("B"))
        gc.collect()

        stats = buffer.get_evicted_episode_release_stats()
        assert stats["tracked_container_refs"] >= 4
        assert stats["pending_container_refs"] == 0
        assert stats["tracked_array_refs"] >= 3
        assert stats["pending_array_refs"] == 0

    @pytest.mark.parametrize(
        "buffer_name",
        ["FastSampleEpisodeReplayBuffer", "NumpyIndexedFastSampleEpisodeReplayBuffer"],
    )
    def test_eviction_diagnostics_detect_sampled_view_retention(self, buffer_name):
        import gc

        import numpy as np

        from rlframework.utils import replay_buffers

        buffer_type = getattr(replay_buffers, buffer_name)
        buffer = buffer_type(
            capacity=2,
            copy_episodes_on_add=False,
            track_evicted_episode_refs=True,
        )
        episode = self._make_numpy_episode("A")
        observation_backing = episode.observations.data
        buffer.add(episode)
        buffer.rng = np.random.default_rng(123)
        sample = buffer.sample(
            sample_episodes=True,
            batch_size_B=1,
            batch_length_T=None,
            n_step=1,
            lookback=0,
        )
        assert np.shares_memory(
            sample[0].get_observations(0),
            observation_backing,
        )

        del observation_backing
        del episode
        buffer.add(self._make_numpy_episode("B"))
        gc.collect()

        retained_stats = buffer.get_evicted_episode_release_stats()
        assert retained_stats["pending_container_refs"] == 0
        assert retained_stats["pending_array_refs"] > 0

        del sample
        gc.collect()

        released_stats = buffer.get_evicted_episode_release_stats()
        assert released_stats["pending_array_refs"] == 0

    @pytest.mark.parametrize(
        "buffer_name",
        ["FastSampleEpisodeReplayBuffer", "NumpyIndexedFastSampleEpisodeReplayBuffer"],
    )
    def test_eviction_diagnostics_are_disabled_by_default(self, buffer_name):
        from rlframework.utils import replay_buffers

        buffer_type = getattr(replay_buffers, buffer_name)
        buffer = buffer_type(capacity=2, copy_episodes_on_add=False)
        buffer.add(self._make_numpy_episode("A"))
        buffer.add(self._make_numpy_episode("B"))

        assert buffer.get_evicted_episode_release_stats() == {
            "tracked_container_refs": 0,
            "pending_container_refs": 0,
            "tracked_array_refs": 0,
            "pending_array_refs": 0,
        }

    def test_fast_sample_matches_batch_evict_transition_sampling(self):
        import numpy as np

        from rlframework.utils.replay_buffers import (
            BatchEvictEpisodeReplayBuffer,
            FastSampleEpisodeReplayBuffer,
        )

        base_buffer = BatchEvictEpisodeReplayBuffer(capacity=10)
        fast_buffer = FastSampleEpisodeReplayBuffer(capacity=10)
        episodes = [
            self._make_episode("A", 0, 3),
            self._make_episode("B", 10, 2, terminated=False, truncated=True),
        ]

        base_buffer.add(episodes)
        fast_buffer.add(episodes)
        base_buffer.rng = np.random.default_rng(123)
        fast_buffer.rng = np.random.default_rng(123)

        base_sample = base_buffer.sample(
            sample_episodes=True,
            batch_size_B=8,
            batch_length_T=None,
            n_step=1,
            lookback=0,
        )
        fast_sample = fast_buffer.sample(
            sample_episodes=True,
            batch_size_B=8,
            batch_length_T=None,
            n_step=1,
            lookback=0,
        )

        assert len(fast_sample) == len(base_sample)
        for fast_episode, base_episode in zip(fast_sample, base_sample, strict=True):
            assert fast_episode.id_ == base_episode.id_
            assert fast_episode.t_started == base_episode.t_started
            assert fast_episode.is_terminated == base_episode.is_terminated
            assert fast_episode.is_truncated == base_episode.is_truncated
            assert fast_episode.get_observations() == base_episode.get_observations()
            assert fast_episode.get_actions() == base_episode.get_actions()
            assert fast_episode.get_rewards() == base_episode.get_rewards()
            assert fast_episode.get_infos() == base_episode.get_infos()
            assert fast_episode.get_extra_model_outputs(
                "n_step"
            ) == base_episode.get_extra_model_outputs("n_step")
            assert fast_episode.get_extra_model_outputs(
                "weights"
            ) == base_episode.get_extra_model_outputs("weights")
        assert fast_buffer.sampled_timesteps == 8

    def test_fast_sample_marks_done_only_at_episode_end(self):
        from rlframework.utils.replay_buffers import FastSampleEpisodeReplayBuffer

        class FixedRng:
            def __init__(self, values):
                self.values = iter(values)

            def integers(self, high):
                return next(self.values)

        buffer = FastSampleEpisodeReplayBuffer(capacity=6)
        buffer.add(self._make_episode("A", 0, 3))
        buffer.rng = FixedRng([0, 2])

        sample = buffer.sample(
            sample_episodes=True,
            batch_size_B=2,
            batch_length_T=None,
            n_step=1,
            lookback=0,
        )

        assert sample[0].t_started == 0
        assert not sample[0].is_terminated
        assert not sample[0].is_truncated
        assert sample[1].t_started == 2
        assert sample[1].is_terminated
        assert not sample[1].is_truncated

    def test_fast_sample_falls_back_for_unsupported_modes(self, monkeypatch):
        from rlframework.utils.replay_buffers import (
            BatchEvictEpisodeReplayBuffer,
            FastSampleEpisodeReplayBuffer,
        )

        def fake_parent_sample(self, *args, **kwargs):
            return ["fallback"]

        monkeypatch.setattr(
            BatchEvictEpisodeReplayBuffer,
            "_sample_episodes",
            fake_parent_sample,
            raising=False,
        )

        fallback_cases = [
            {"batch_length_T": 2, "n_step": 1, "lookback": 0},
            {"batch_length_T": None, "n_step": 2, "lookback": 0},
            {"batch_length_T": None, "n_step": (1, 3), "lookback": 0},
            {"batch_length_T": None, "n_step": 1, "lookback": 1},
            {
                "batch_length_T": None,
                "n_step": 1,
                "lookback": 0,
                "include_extra_model_outputs": True,
            },
            {"batch_length_T": None, "n_step": 1, "lookback": 0, "to_numpy": True},
            {
                "batch_length_T": None,
                "n_step": 1,
                "lookback": 0,
                "min_batch_length_T": 1,
            },
        ]

        for params in fallback_cases:
            buffer = FastSampleEpisodeReplayBuffer(capacity=6)
            assert buffer._sample_episodes(
                batch_size_B=1,
                **params,
            ) == ["fallback"]

    def test_numpy_indexed_buffer_tracks_indices_and_eviction(self):
        from rlframework.utils.replay_buffers import NumpyIndexedFastSampleEpisodeReplayBuffer

        buffer = NumpyIndexedFastSampleEpisodeReplayBuffer(capacity=6)
        buffer.add(self._make_episode("A", 0, 2, terminated=False))
        buffer.add(self._make_episode("B", 0, 2))
        buffer.add(self._make_episode("A", 2, 2))

        assert buffer.get_num_timesteps() == 6
        assert buffer._num_indices == 6
        assert buffer._index_episode[: buffer._num_indices].tolist() == [0, 0, 1, 1, 0, 0]
        assert buffer._index_timestep[: buffer._num_indices].tolist() == [0, 1, 0, 1, 2, 3]

        buffer.add(self._make_episode("C", 0, 2))

        assert "A" not in buffer.episode_id_to_index
        assert buffer.episode_id_to_index == {"B": 1, "C": 2}
        assert buffer.get_num_timesteps() == 4
        assert buffer._index_episode[: buffer._num_indices].tolist() == [1, 1, 2, 2]
        assert buffer._index_timestep[: buffer._num_indices].tolist() == [0, 1, 0, 1]
        assert buffer._indices == []

    def test_numpy_indexed_fast_sample_matches_fast_sample(self):
        import numpy as np

        from rlframework.utils.replay_buffers import (
            FastSampleEpisodeReplayBuffer,
            NumpyIndexedFastSampleEpisodeReplayBuffer,
        )

        fast_buffer = FastSampleEpisodeReplayBuffer(capacity=10)
        numpy_buffer = NumpyIndexedFastSampleEpisodeReplayBuffer(capacity=10)
        episodes = [
            self._make_episode("A", 0, 3),
            self._make_episode("B", 10, 2, terminated=False, truncated=True),
        ]

        fast_buffer.add(episodes)
        numpy_buffer.add(episodes)
        fast_buffer.rng = np.random.default_rng(123)
        numpy_buffer.rng = np.random.default_rng(123)

        fast_sample = fast_buffer.sample(
            sample_episodes=True,
            batch_size_B=8,
            batch_length_T=None,
            n_step=1,
            lookback=0,
        )
        numpy_sample = numpy_buffer.sample(
            sample_episodes=True,
            batch_size_B=8,
            batch_length_T=None,
            n_step=1,
            lookback=0,
        )

        assert numpy_buffer._indices == []
        assert len(numpy_sample) == len(fast_sample)
        for numpy_episode, fast_episode in zip(numpy_sample, fast_sample, strict=True):
            assert numpy_episode.id_ == fast_episode.id_
            assert numpy_episode.t_started == fast_episode.t_started
            assert numpy_episode.is_terminated == fast_episode.is_terminated
            assert numpy_episode.is_truncated == fast_episode.is_truncated
            assert numpy_episode.get_observations() == fast_episode.get_observations()
            assert numpy_episode.get_actions() == fast_episode.get_actions()
            assert numpy_episode.get_rewards() == fast_episode.get_rewards()
            assert numpy_episode.get_infos() == fast_episode.get_infos()
            assert numpy_episode.get_extra_model_outputs(
                "n_step"
            ) == fast_episode.get_extra_model_outputs("n_step")
            assert numpy_episode.get_extra_model_outputs(
                "weights"
            ) == fast_episode.get_extra_model_outputs("weights")

    def test_numpy_indexed_buffer_materializes_indices_for_fallback(self, monkeypatch):
        from rlframework.utils.replay_buffers import (
            BatchEvictEpisodeReplayBuffer,
            NumpyIndexedFastSampleEpisodeReplayBuffer,
        )

        observed_indices = []

        def fake_parent_sample(self, *args, **kwargs):
            observed_indices.extend(self._indices)
            return ["fallback"]

        monkeypatch.setattr(
            BatchEvictEpisodeReplayBuffer,
            "_sample_episodes",
            fake_parent_sample,
            raising=False,
        )

        buffer = NumpyIndexedFastSampleEpisodeReplayBuffer(capacity=6)
        buffer.add(self._make_episode("A", 0, 2))

        assert buffer._sample_episodes(
            batch_size_B=1,
            batch_length_T=None,
            n_step=2,
            lookback=0,
        ) == ["fallback"]
        assert observed_indices == [(0, 0), (0, 1)]

    def test_numpy_indexed_buffer_state_restore_and_legacy_indices(self):
        import numpy as np

        from rlframework.utils.replay_buffers import (
            BatchEvictEpisodeReplayBuffer,
            NumpyIndexedFastSampleEpisodeReplayBuffer,
        )

        buffer = NumpyIndexedFastSampleEpisodeReplayBuffer(capacity=6)
        buffer.add([self._make_episode("A", 0, 2), self._make_episode("B", 10, 2)])
        state = buffer.get_state()

        restored = NumpyIndexedFastSampleEpisodeReplayBuffer(capacity=6)
        restored.set_state(state)
        restored.rng = np.random.default_rng(123)

        assert restored.get_num_timesteps() == buffer.get_num_timesteps()
        assert restored._index_episode[: restored._num_indices].tolist() == [0, 0, 1, 1]
        assert restored._index_timestep[: restored._num_indices].tolist() == [0, 1, 0, 1]
        assert len(restored.sample(sample_episodes=True, batch_size_B=2, n_step=1, lookback=0)) == 2

        legacy_buffer = BatchEvictEpisodeReplayBuffer(capacity=6)
        legacy_buffer.add(self._make_episode("C", 0, 2))
        migrated = NumpyIndexedFastSampleEpisodeReplayBuffer(capacity=6)
        migrated.set_state(legacy_buffer.get_state())

        assert migrated.get_num_timesteps() == 2
        assert migrated._index_episode[: migrated._num_indices].tolist() == [0, 0]
        assert migrated._index_timestep[: migrated._num_indices].tolist() == [0, 1]


# ---------------------------------------------------------------------------
# torch_utils
# ---------------------------------------------------------------------------


class TestTorchUtils:
    @pytest.fixture
    def simple_model(self):
        import torch.nn as nn

        return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))

    def test_count_parameters_all(self, simple_model):
        from rlframework.utils.torch_utils import count_parameters

        n = count_parameters(simple_model, trainable_only=False)
        # Linear(4,8): 4*8+8=40; Linear(8,2): 8*2+2=18 → 58
        assert n == 58

    def test_count_parameters_trainable_only(self, simple_model):
        from rlframework.utils.torch_utils import count_parameters, freeze_parameters

        freeze_parameters(simple_model)
        n = count_parameters(simple_model, trainable_only=True)
        assert n == 0

    def test_freeze_unfreeze(self, simple_model):
        from rlframework.utils.torch_utils import freeze_parameters, unfreeze_parameters

        freeze_parameters(simple_model)
        for p in simple_model.parameters():
            assert not p.requires_grad

        unfreeze_parameters(simple_model)
        for p in simple_model.parameters():
            assert p.requires_grad

    def test_polyak_update(self):
        import torch
        import torch.nn as nn

        from rlframework.utils.torch_utils import polyak_update

        source = nn.Linear(4, 4)
        target = nn.Linear(4, 4)

        # Make weights clearly different
        with torch.no_grad():
            source.weight.fill_(1.0)
            target.weight.fill_(0.0)

        polyak_update(source, target, tau=1.0)

        # tau=1.0 means full copy
        assert torch.allclose(target.weight, source.weight)

    def test_polyak_update_partial(self):
        import torch
        import torch.nn as nn

        from rlframework.utils.torch_utils import polyak_update

        source = nn.Linear(2, 2)
        target = nn.Linear(2, 2)

        with torch.no_grad():
            source.weight.fill_(1.0)
            target.weight.fill_(0.0)

        polyak_update(source, target, tau=0.5)
        # target = 0.5 * source + 0.5 * target = 0.5
        assert torch.allclose(target.weight, torch.full_like(target.weight, 0.5))


# ---------------------------------------------------------------------------
# models/components
# ---------------------------------------------------------------------------
