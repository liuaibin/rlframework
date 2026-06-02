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
