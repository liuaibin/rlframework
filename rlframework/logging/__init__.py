"""Backward-compatible logging namespace.

Prefer using ``rlframework.callbacks`` and ``rlframework.observability``.
"""

from rlframework.callbacks import FrameworkCallback

__all__ = ["FrameworkCallback"]
