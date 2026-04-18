"""Tests for rlframework.algorithms — mixin hooks."""

import pytest

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
