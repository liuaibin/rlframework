"""Tests for rlframework.algorithms — mixin hooks."""

# ---------------------------------------------------------------------------
# Mixin hooks — default no-ops
# ---------------------------------------------------------------------------


class TestFrameworkAlgorithmMixin:
    def _make_mixin(self):
        from rlframework.algorithms.base import FrameworkAlgorithmMixin

        class _Concrete(FrameworkAlgorithmMixin):
            pass

        return _Concrete()

    def test_on_before_training_step_noop(self):
        mixin = self._make_mixin()
        # Should not raise and return None
        result = mixin.on_before_training_step()
        assert result is None

    def test_on_after_training_step_noop(self):
        mixin = self._make_mixin()
        fake_result = {"episode_return_mean": 1.0}
        mixin.on_after_training_step(fake_result)  # should not raise

    def test_hooks_can_be_overridden(self):
        from rlframework.algorithms.base import FrameworkAlgorithmMixin

        called = []

        class _Custom(FrameworkAlgorithmMixin):
            def on_before_training_step(self):
                called.append("before")

            def on_after_training_step(self, result):
                called.append("after")

        obj = _Custom()
        obj.on_before_training_step()
        obj.on_after_training_step({})
        assert called == ["before", "after"]


class TestAlgorithmConfigs:
    def test_custom_ppo_rl_module_spec_uses_new_api(self):
        import importlib.util
        from pathlib import Path

        import gymnasium as gym
        import torch
        from ray.rllib.core.columns import Columns
        from ray.rllib.core.rl_module.rl_module import RLModuleSpec

        example_path = Path(__file__).resolve().parents[1] / "examples/06_custom_rl_module.py"
        spec = importlib.util.spec_from_file_location("example_06_custom_rl_module", example_path)
        assert spec is not None and spec.loader is not None
        module_file = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module_file)

        module = RLModuleSpec(
            module_class=module_file.MinimalPPOModule,
            observation_space=gym.spaces.Box(-1.0, 1.0, shape=(4,)),
            action_space=gym.spaces.Discrete(2),
            model_config={"fcnet_hiddens": [16], "fcnet_activation": "relu"},
        ).build()

        batch = {Columns.OBS: torch.randn(3, 4)}
        fwd_out = module.forward_train(batch)
        values = module.compute_values(batch, embeddings=fwd_out[Columns.EMBEDDINGS])

        assert fwd_out[Columns.ACTION_DIST_INPUTS].shape == (3, 2)
        assert values.shape == (3,)
        assert hasattr(module, "state_dict")

    def test_ppo_framework_models_preserves_new_api_model_config(self):
        from rlframework.algorithms.ppo import CustomPPOConfig
        from rlframework.models.catalog import PPOCompositeCatalog

        cfg = (
            CustomPPOConfig()
            .rl_module(
                model_config={
                    "custom_model_config": {
                        "critic_head": {
                            "value_scale": 0.5,
                        }
                    }
                }
            )
            .framework_models(
                encoder="enc",
                actor_head="pi",
                critic_head="vf",
            )
        )

        assert cfg.model_config["_framework_custom_config"] == {
            "custom_encoder": "enc",
            "custom_actor_head": "pi",
            "custom_critic_head": "vf",
        }
        assert cfg.model_config["custom_model_config"]["critic_head"]["value_scale"] == 0.5
        assert cfg._rl_module_spec.catalog_class is PPOCompositeCatalog

    def test_sac_framework_models_sets_rl_module_spec(self):
        from rlframework.algorithms.sac import CustomSACConfig
        from rlframework.models.catalog import SACCompositeCatalog

        cfg = CustomSACConfig().framework_models(
            encoder="enc",
            actor_head="pi",
            critic_head="vf",
            q_head="qf",
        )

        assert cfg.model_config["_framework_custom_config"] == {
            "custom_encoder": "enc",
            "custom_actor_head": "pi",
            "custom_critic_head": "vf",
            "custom_q_head": "qf",
        }
        assert cfg._rl_module_spec.catalog_class is SACCompositeCatalog

    def test_sac_validate_accepts_custom_replay_buffer_import_path(self):
        from rlframework.algorithms.sac import CustomSACConfig

        cfg = CustomSACConfig().training(
            replay_buffer_config={
                "type": "rlframework.utils.replay_buffers.PrioritizedSumTreeBuffer",
                "capacity": 1000,
            }
        )
        # Should not raise on valid EpisodeReplayBuffer subclass path.
        cfg.validate()

    def test_async_sac_preserves_sac_defaults_but_requires_supported_replay_buffer(self):
        import pytest
        from ray.rllib.algorithms.sac import SACConfig

        from rlframework.algorithms.async_sac import AsyncCustomSACConfig

        cfg = AsyncCustomSACConfig().learners(num_learners=1, num_gpus_per_learner=0)

        assert (
            cfg.num_steps_sampled_before_learning_starts
            == SACConfig().num_steps_sampled_before_learning_starts
        )
        assert cfg.replay_buffer_config == SACConfig().replay_buffer_config

        with pytest.raises(ValueError, match="only supports non-prioritized"):
            cfg.validate()

    def test_async_sac_validate_accepts_supported_replay_buffers(self):
        from ray.rllib.utils.replay_buffers.episode_replay_buffer import EpisodeReplayBuffer

        from rlframework.algorithms.async_sac import AsyncCustomSACConfig
        from rlframework.utils.replay_buffers import BatchEvictEpisodeReplayBuffer

        for buffer_type in (
            "EpisodeReplayBuffer",
            EpisodeReplayBuffer,
            BatchEvictEpisodeReplayBuffer,
            "rlframework.utils.replay_buffers.BatchEvictEpisodeReplayBuffer",
        ):
            cfg = (
                AsyncCustomSACConfig()
                .training(
                    replay_buffer_config={
                        "type": buffer_type,
                        "capacity": 1000,
                    }
                )
                .learners(num_learners=1, num_gpus_per_learner=0)
            )
            cfg.validate()

    def test_async_sac_algorithm_options_rejects_invalid_mode(self):
        import pytest

        from rlframework.algorithms.async_sac import AsyncCustomSACConfig

        with pytest.raises(ValueError, match="env_sampling"):
            AsyncCustomSACConfig().algorithm_options({"env_sampling": "invalid"})

    def test_algorithm_options_rejects_unknown_base_option(self):
        import pytest

        from rlframework.algorithms.ppo import CustomPPOConfig
        from rlframework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="Unknown algorithm option"):
            CustomPPOConfig().algorithm_options({"unknown": True})

    def test_async_sac_algorithm_options_rejects_unknown_option(self):
        import pytest

        from rlframework.algorithms.async_sac import AsyncCustomSACConfig
        from rlframework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="Unknown AsyncCustomSAC algorithm option"):
            AsyncCustomSACConfig().algorithm_options({"unknown": True})

    def test_async_sac_algorithm_options_accepts_pipeline_log_interval(self):
        from rlframework.algorithms.async_sac import AsyncCustomSACConfig

        cfg = AsyncCustomSACConfig().algorithm_options({"pipeline_log_interval": 25})

        assert cfg.async_pipeline_log_interval == 25

    def test_async_sac_algorithm_options_accepts_sync_learner_update_limit(self):
        from rlframework.algorithms.async_sac import AsyncCustomSACConfig

        cfg = AsyncCustomSACConfig().algorithm_options({"max_sync_learner_updates_per_step": 3})

        assert cfg.async_max_sync_learner_updates_per_step == 3

    def test_async_sac_algorithm_options_rejects_bad_pipeline_log_interval(self):
        import pytest

        from rlframework.algorithms.async_sac import AsyncCustomSACConfig
        from rlframework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="pipeline_log_interval"):
            AsyncCustomSACConfig().algorithm_options({"pipeline_log_interval": -1})

    def test_async_sac_algorithm_options_rejects_bad_sync_learner_update_limit(self):
        import pytest

        from rlframework.algorithms.async_sac import AsyncCustomSACConfig
        from rlframework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="max_sync_learner_updates_per_step"):
            AsyncCustomSACConfig().algorithm_options({"max_sync_learner_updates_per_step": 0})

    def test_async_sac_explicit_async_env_requires_ray_254_api(self):
        import pytest

        from rlframework.algorithms.async_sac import AsyncCustomSACConfig

        cfg = (
            AsyncCustomSACConfig()
            .training(replay_buffer_config={"type": "EpisodeReplayBuffer", "capacity": 1000})
            .algorithm_options({"env_sampling": "async", "learner_training": "sync"})
        )

        with pytest.raises(ValueError, match="env_sampling='async'"):
            cfg.validate()

    def test_async_sac_async_env_sync_learner_allows_local_learner(self, monkeypatch):
        from rlframework.algorithms import async_sac

        monkeypatch.setattr(async_sac, "_supports_async_env_runner_fetch_ready", lambda: True)

        cfg = (
            async_sac.AsyncCustomSACConfig()
            .training(replay_buffer_config={"type": "EpisodeReplayBuffer", "capacity": 1000})
            .learners(num_learners=0, num_gpus_per_learner=0)
            .algorithm_options({"env_sampling": "async", "learner_training": "sync"})
        )

        cfg.validate()
        assert cfg._dont_auto_sync_env_runner_states is True

    def test_async_sac_async_learner_requires_remote_learner(self, monkeypatch):
        import pytest

        from rlframework.algorithms import async_sac

        monkeypatch.setattr(async_sac, "_supports_async_env_runner_fetch_ready", lambda: True)

        cfg = (
            async_sac.AsyncCustomSACConfig()
            .training(replay_buffer_config={"type": "EpisodeReplayBuffer", "capacity": 1000})
            .learners(num_learners=0, num_gpus_per_learner=0)
            .algorithm_options({"env_sampling": "async", "learner_training": "async"})
        )

        with pytest.raises(ValueError, match="num_learners > 0"):
            cfg.validate()

    def test_async_sac_sync_env_disables_manual_env_state_sync(self, monkeypatch):
        from rlframework.algorithms import async_sac

        monkeypatch.setattr(async_sac, "_supports_async_env_runner_fetch_ready", lambda: True)

        cfg = (
            async_sac.AsyncCustomSACConfig()
            .training(replay_buffer_config={"type": "EpisodeReplayBuffer", "capacity": 1000})
            .algorithm_options({"env_sampling": "sync", "learner_training": "sync"})
        )

        cfg.validate()
        assert cfg._dont_auto_sync_env_runner_states is False

    def test_metrics_auto_wires_framework_callback_factory(self, tmp_dir):
        from rlframework.algorithms.ppo import CustomPPOConfig
        from rlframework.callbacks import FrameworkCallback

        metrics_file = tmp_dir / "metrics.jsonl"
        cfg = CustomPPOConfig().metrics(
            reporters=["file"],
            reporter_configs={"file": {"filepath": str(metrics_file)}},
        )
        cfg._apply_framework_runtime_config()

        callback = cfg.callbacks_class()
        assert isinstance(callback, FrameworkCallback)
        assert len(callback._reporters) == 1

    def test_metrics_does_not_override_custom_callback(self, tmp_dir):
        from ray.rllib.callbacks.callbacks import RLlibCallback

        from rlframework.algorithms.ppo import CustomPPOConfig

        class _UserCallback(RLlibCallback):
            pass

        cfg = (
            CustomPPOConfig()
            .callbacks(_UserCallback)
            .metrics(
                reporters=["file"],
                reporter_configs={"file": {"filepath": str(tmp_dir / "m.jsonl")}},
            )
        )
        cfg._apply_framework_runtime_config()
        assert cfg.callbacks_class is _UserCallback

    def test_framework_run_layout_defaults_are_absolute(self, tmp_dir, monkeypatch):
        from pathlib import Path

        from rlframework.algorithms.ppo import CustomPPOConfig
        from rlframework.callbacks import FrameworkCallback

        project_dir = tmp_dir / "external_project"
        project_dir.mkdir()
        monkeypatch.chdir(project_dir)

        cfg = (
            CustomPPOConfig()
            .framework_run("probe", root_dir="./runs")
            .framework_checkpointing(freq=5)
            .metrics(reporters=["file"])
            .storage()
        )
        cfg._apply_framework_runtime_config()
        layout = cfg.framework_layout
        assert layout is not None

        assert layout.run_dir.is_absolute()
        assert layout.run_dir.parent == (project_dir / "runs").resolve()
        assert layout.run_dir.name.startswith("probe_")
        assert Path(cfg._checkpoint_local_dir) == layout.checkpoint_dir
        assert Path(cfg._metrics_reporter_configs["file"]["filepath"]).is_absolute()
        assert Path(cfg._metrics_reporter_configs["file"]["filepath"]).parent == layout.metrics_dir
        assert Path(cfg._storage_backend_config["root"]) == layout.storage_dir

        callback = cfg.callbacks_class()
        assert isinstance(callback, FrameworkCallback)
        assert Path(callback._ckpt_local_dir) == layout.checkpoint_dir

    def test_checkpointing_upload_async_is_passed_to_manager(self):
        from rlframework.algorithms.ppo import CustomPPOConfig

        cfg = CustomPPOConfig().storage(upload_async=False).framework_checkpointing(freq=5)
        manager = cfg.build_checkpoint_manager()
        try:
            assert manager._upload_async is False
        finally:
            manager.shutdown()

    def test_checkpointing_auto_wires_callback_with_manager(self):
        from rlframework.algorithms.ppo import CustomPPOConfig
        from rlframework.callbacks import FrameworkCallback

        cfg = CustomPPOConfig().storage().framework_checkpointing(freq=10, local_dir="/tmp/ckpts")
        cfg._apply_framework_runtime_config()

        callback = cfg.callbacks_class()
        assert isinstance(callback, FrameworkCallback)
        assert callback._ckpt_freq == 10
        assert callback._ckpt_local_dir == "/tmp/ckpts"
        assert callback._ckpt_manager is not None
        try:
            callback._ckpt_manager.shutdown()
        except Exception:
            pass

    def test_checkpointing_reuses_user_provided_manager(self):
        from unittest.mock import MagicMock

        from rlframework.algorithms.ppo import CustomPPOConfig
        from rlframework.callbacks import FrameworkCallback

        user_manager = MagicMock()
        cfg = (
            CustomPPOConfig()
            .framework_checkpointing(freq=5, local_dir="/tmp/c")
            .callbacks(
                FrameworkCallback.with_reporters(
                    [],
                    checkpoint_manager=user_manager,
                    checkpoint_freq=5,
                    checkpoint_local_dir="/tmp/c",
                )
            )
        )
        cfg._apply_framework_runtime_config()

        callback = cfg.callbacks_class()
        assert callback._ckpt_manager is user_manager

    def test_resume_from_restores_after_build(self, tmp_dir, monkeypatch):
        from ray.rllib.algorithms.ppo import PPOConfig

        from rlframework.algorithms.ppo import CustomPPOConfig

        checkpoint_dir = tmp_dir / "iter_000020"
        checkpoint_dir.mkdir()
        restored_paths = []

        class _FakeAlgorithm:
            def restore_from_path(self, path):
                restored_paths.append(path)

        def fake_build(self, *args, **kwargs):
            return _FakeAlgorithm()

        monkeypatch.setattr(PPOConfig, "build", fake_build)

        algo = CustomPPOConfig().resume_from(checkpoint_dir).build()

        assert isinstance(algo, _FakeAlgorithm)
        assert restored_paths == [str(checkpoint_dir)]

    def test_resume_from_validates_checkpoint_path(self):
        import pytest

        from rlframework.algorithms.ppo import CustomPPOConfig
        from rlframework.utils.exceptions import ValidationError

        with pytest.raises(ValidationError, match="resume_from checkpoint_path"):
            CustomPPOConfig().resume_from("")

    def test_resume_from_rejects_missing_local_checkpoint(self, tmp_dir, monkeypatch):
        import pytest
        from ray.rllib.algorithms.ppo import PPOConfig

        from rlframework.algorithms.ppo import CustomPPOConfig
        from rlframework.utils.exceptions import CheckpointError

        build_called = False

        def fake_build(self, *args, **kwargs):
            nonlocal build_called
            build_called = True
            return object()

        monkeypatch.setattr(PPOConfig, "build", fake_build)

        with pytest.raises(CheckpointError, match="resume checkpoint path does not exist"):
            CustomPPOConfig().resume_from(tmp_dir / "missing").build()
        assert build_called is False

    def test_resume_from_requires_algorithm_restore_method(self, tmp_dir, monkeypatch):
        import pytest
        from ray.rllib.algorithms.ppo import PPOConfig

        from rlframework.algorithms.ppo import CustomPPOConfig
        from rlframework.utils.exceptions import ConfigurationError

        checkpoint_dir = tmp_dir / "iter_000020"
        checkpoint_dir.mkdir()

        def fake_build(self, *args, **kwargs):
            return object()

        monkeypatch.setattr(PPOConfig, "build", fake_build)

        with pytest.raises(ConfigurationError, match="restore_from_path"):
            CustomPPOConfig().resume_from(checkpoint_dir).build()
