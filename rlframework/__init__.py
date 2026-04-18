"""RLframework - A company-level RL framework built on top of RLlib.

Design Principles:
- Extend RLlib natively (inherit, not wrap)
- Pluggable backends for storage/logging/metrics
- Minimal abstraction layers
"""

import logging

__version__ = "0.1.0"

_rh = logging.StreamHandler()
_rh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
logging.root.addHandler(_rh)
logging.root.setLevel(logging.INFO)
logging.getLogger("ray").setLevel(logging.WARNING)
