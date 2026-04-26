"""Configuration validation utilities.

Provides validators for common configuration patterns with helpful error messages.
"""

from typing import Any

from rlframework.utils.exceptions import ConfigurationError, ValidationError


def validate_lr(lr: Any, field: str = "learning_rate") -> None:
    """Validate learning rate: either a fixed float or a schedule.

    Args:
        lr: Either a single float (fixed LR) or a schedule list.
            Schedule format mirrors RLlib's ``LearningRateOrSchedule``:
            ``[[timestep0, lr0], [timestep1, lr1], ...]`` where timestep
            values are monotonically increasing ints and lr values are
            positive floats. The first entry must start at timestep 0.
        field: Dot-prefixed config path for error messages.

    Raises:
        ValidationError: If the value does not match either accepted format.
    """
    # --- Fixed float / int ---
    if isinstance(lr, (int, float)):
        if lr <= 0:
            raise ValidationError(
                "Learning rate must be positive",
                field=field,
                value=lr,
            )
        if lr > 1:
            raise ValidationError(
                "Learning rate seems unusually high (>1.0). Did you mean a smaller value?",
                field=field,
                value=lr,
            )
        return

    # --- Schedule (list of [timestep, lr_value] pairs) ---
    if not isinstance(lr, (list, tuple)):
        raise ValidationError(
            f"Learning rate must be a number or a schedule list, got {type(lr).__name__}",
            field=field,
            value=lr,
        )

    # --- Validate each entry (handles both short and long schedules) ---
    for i, entry in enumerate(lr):
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            raise ValidationError(
                f"Each schedule entry must be a [timestep, lr_value] pair "
                f"(2-element list/tuple), but entry {i} is: {entry!r}",
                field=field,
                value=lr,
            )

        ts, val = entry

        # Timestep must be a non-negative int
        if not isinstance(ts, int) or ts < 0:
            raise ValidationError(
                f"Schedule timestep (entry {i}) must be a non-negative int, "
                f"got {type(ts).__name__}={ts}",
                field=field,
                value=lr,
            )

        # Value must be a positive number (warn on > 1.0)
        if not isinstance(val, (int, float)) or val <= 0:
            raise ValidationError(
                f"Schedule lr_value (entry {i}) must be a positive number, "
                f"got {type(val).__name__}={val}",
                field=field,
                value=lr,
            )
        if val > 1:
            raise ValidationError(
                f"Schedule lr_value at entry {i} ({val}) is > 1.0. "
                "Did you mean a smaller value?",
                field=field,
                value=lr,
            )

        # Timestamps must be strictly increasing
        if i > 0:
            prev_ts = lr[i - 1][0]
            if ts <= prev_ts:
                raise ValidationError(
                    f"Schedule timesteps must be strictly increasing; "
                    f"entry {i} has ts={ts} which is not > prev_ts={prev_ts}",
                    field=field,
                    value=lr,
                )

    # --- Length and first-entry sentinel ---
    # Check first entry's timestep=0 BEFORE the length guard so that a
    # single-element schedule with a wrong starting timestep gets the more
    # specific "must start at timestep 0" message rather than "need ≥2 entries".
    if lr[0][0] != 0:
        raise ValidationError(
            f"Learning rate schedule must start at timestep 0, but first "
            f"entry has timestep={lr[0][0]}. Example: [[0, 3e-4], [100000, 1e-4]]",
            field=field,
            value=lr,
        )
    if len(lr) < 2:
        raise ValidationError(
            f"Learning rate schedule must have at least 2 entries, got {len(lr)}. "
            "Example: [[0, 3e-4], [100000, 1e-4]]",
            field=field,
            value=lr,
        )


def validate_gamma(gamma: Any, field: str = "gamma") -> None:
    """Validate discount factor is in valid range (0, 1]."""
    if not isinstance(gamma, (int, float)):
        raise ValidationError(
            f"Discount factor must be a number, got {type(gamma).__name__}",
            field=field,
            value=gamma,
        )
    if not 0 < gamma <= 1:
        raise ValidationError(
            "Discount factor (gamma) must be in range (0, 1]",
            field=field,
            value=gamma,
        )


def validate_positive_int(value: int, field: str, min_value: int = 1) -> None:
    """Validate a positive integer value."""
    if not isinstance(value, int):
        raise ValidationError(
            f"{field} must be an integer, got {type(value).__name__}",
            field=field,
            value=value,
        )
    if value < min_value:
        raise ValidationError(
            f"{field} must be at least {min_value}, got {value}",
            field=field,
            value=value,
        )


def validate_non_negative_int(value: int, field: str) -> None:
    """Validate a non-negative integer value."""
    if not isinstance(value, int):
        raise ValidationError(
            f"{field} must be an integer, got {type(value).__name__}",
            field=field,
            value=value,
        )
    if value < 0:
        raise ValidationError(
            f"{field} must be non-negative, got {value}",
            field=field,
            value=value,
        )


def validate_env_config(env_config: dict[str, Any]) -> None:
    """Validate environment configuration."""
    if not isinstance(env_config, dict):
        raise ValidationError(
            "env_config must be a dictionary",
            value=type(env_config).__name__,
            field="env_config",
        )


def validate_model_config(model_config: dict[str, Any]) -> None:
    """Validate model configuration."""
    if not isinstance(model_config, dict):
        raise ValidationError(
            "model_config must be a dictionary",
            value=type(model_config).__name__,
            field="model_config",
        )


def validate_backend(backend: str, valid_backends: list[str]) -> None:
    """Validate storage backend name."""
    if backend not in valid_backends:
        raise ValidationError(
            f"Unknown backend: '{backend}'. Available: {valid_backends}",
            field="backend",
            value=backend,
        )


def validate_workers(num_workers: int, field: str = "num_workers") -> None:
    """Validate worker count configuration."""
    validate_non_negative_int(num_workers, field)
    if num_workers > 64:
        raise ConfigurationError(
            f"Worker count ({num_workers}) is very high. Consider using Ray cluster for >64 workers.",
            field=field,
            hint="For large-scale distributed training, use ray.init(address='auto') to connect to a Ray cluster",
        )


def validate_gpu_config(num_gpus: float) -> None:
    """Validate GPU configuration."""
    if not isinstance(num_gpus, (int, float)):
        raise ValidationError(
            f"num_gpus must be a number, got {type(num_gpus).__name__}",
            field="num_gpus",
            value=num_gpus,
        )
    if num_gpus < 0:
        raise ValidationError(
            "num_gpus cannot be negative",
            field="num_gpus",
            value=num_gpus,
        )
    if num_gpus > 1 and not isinstance(num_gpus, int):
        raise ValidationError(
            "num_gpus > 1 must be an integer (whole GPUs)",
            field="num_gpus",
            value=num_gpus,
        )
