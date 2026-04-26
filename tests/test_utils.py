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
