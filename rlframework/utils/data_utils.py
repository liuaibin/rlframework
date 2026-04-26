"""Data / dict utility helpers used across the framework."""

from collections.abc import Iterable
from typing import Any


def flatten_dict(
    d: dict[str, Any],
    sep: str = "/",
    prefix: str = "",
) -> dict[str, Any]:
    """Recursively flatten a nested dict using *sep* as a key separator.

    Example::

        >>> flatten_dict({"a": {"b": 1, "c": 2}, "d": 3})
        {"a/b": 1, "a/c": 2, "d": 3}

    Args:
        d: Input (possibly nested) dict.
        sep: Separator string between nesting levels.
        prefix: Key prefix prepended at the top level.

    Returns:
        Flat dict.
    """
    result: dict[str, Any] = {}
    for k, v in d.items():
        full_key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            result.update(flatten_dict(v, sep=sep, prefix=full_key))
        else:
            result[full_key] = v
    return result


def unflatten_dict(
    flat: dict[str, Any],
    sep: str = "/",
) -> dict[str, Any]:
    """Reverse of :func:`flatten_dict` – rebuild nested structure.

    Example::

        >>> unflatten_dict({"a/b": 1, "a/c": 2, "d": 3})
        {"a": {"b": 1, "c": 2}, "d": 3}

    Args:
        flat: Flat dict produced by :func:`flatten_dict`.
        sep: Key separator used during flattening.

    Returns:
        Nested dict.
    """
    result: dict[str, Any] = {}
    for key, value in flat.items():
        parts = key.split(sep)
        node = result
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return result


def safe_mean(values: Iterable, default: float = 0.0) -> float:
    """Return the mean of *values*, or *default* if the sequence is empty.

    Args:
        values: Iterable of numeric values.
        default: Value returned when *values* is empty.

    Returns:
        The mean, or *default*.
    """
    lst = [v for v in values if v is not None]
    return sum(lst) / len(lst) if lst else default


def deep_merge(base: dict, overrides: dict) -> dict:
    """Recursively merge *overrides* into *base* and return the result.

    Lists and non-dict values in *overrides* replace those in *base*.

    Args:
        base: Base dictionary.
        overrides: Dictionary whose values take precedence.

    Returns:
        Merged dict (new object; *base* is not mutated).
    """
    import copy

    result = copy.deepcopy(base)
    for k, v in overrides.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result
