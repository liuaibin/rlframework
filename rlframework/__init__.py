"""RLframework - A company-level RL framework built on top of RLlib.

Design Principles:
- Extend RLlib natively (inherit, not wrap)
- Pluggable backends for storage/logging/metrics
- Minimal abstraction layers
"""

import logging

from rlframework.logging_config import setup_logging

__version__ = "0.1.0"

_logger = logging.getLogger(__name__)
if not any(isinstance(handler, logging.NullHandler) for handler in _logger.handlers):
    _logger.addHandler(logging.NullHandler())

__all__ = ["__version__", "setup_logging"]
