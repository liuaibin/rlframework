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

    def test_compute_grads_postprocess_identity(self):
        mixin = self._make_mixin()
        grads = [1.0, 2.0]
        returned = mixin.compute_grads_postprocess(grads)
        assert returned == grads

    def test_loss_postprocess_identity(self):
        mixin = self._make_mixin()
        total_loss = 0.5
        components = {"policy_loss": 0.5}
        returned = mixin.loss_postprocess(total_loss, components)
        assert returned == total_loss

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
    def test_sac_framework_models_sets_rl_module_spec(self):
        from rlframework.algorithms.sac import CustomSACConfig
        from rlframework.models.catalog import SACCompositeCatalog

        cfg = CustomSACConfig().framework_models(
            encoder="enc",
            actor_head="pi",
            critic_head="vf",
            q_head="qf",
        )

        assert cfg.model["_framework_custom_config"] == {
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

    def test_metrics_auto_wires_framework_callback_factory(self, tmp_dir):
        from rlframework.algorithms.ppo import CustomPPOConfig
        from rlframework.logging.callbacks import FrameworkCallback

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

        cfg = CustomPPOConfig().checkpointing(freq=5, upload_async=False)
        manager = cfg.build_checkpoint_manager()
        try:
            assert manager._upload_async is False
        finally:
            manager.shutdown()

    def test_checkpointing_auto_wires_callback_with_manager(self):
        from rlframework.algorithms.ppo import CustomPPOConfig
        from rlframework.logging.callbacks import FrameworkCallback

        cfg = CustomPPOConfig().checkpointing(freq=10, local_dir="/tmp/ckpts")
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
        from rlframework.logging.callbacks import FrameworkCallback

        user_manager = MagicMock()
        cfg = (
            CustomPPOConfig()
            .checkpointing(freq=5, local_dir="/tmp/c")
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
