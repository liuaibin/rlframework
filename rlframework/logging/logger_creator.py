"""Factory for custom RLlib logger_creator callables.

Usage inside AlgorithmConfig::

    from rlframework.logging import make_logger_creator

    config = (
        CustomPPOConfig()
        .environment("CartPole-v1")
        .debugging(
            logger_creator=make_logger_creator(logdir="/tmp/my_exp"),
        )
    )
"""

import os
import time


def make_logger_creator(
    logdir: str | None = None,
    prefix: str = "trial",
):
    """Return a ``logger_creator`` callable for use in ``config.debugging()``.

    The returned creator produces a :class:`ray.tune.logger.UnifiedLogger`
    that writes JSON, CSV, and TensorBoard event files under *logdir*.

    Args:
        logdir: Directory to write logs into.  Defaults to
            ``~/ray_results/<prefix>_<timestamp>``.
        prefix: Directory name prefix when *logdir* is not specified.

    Returns:
        A callable with signature ``(config) -> Logger``.
    """
    from ray.tune.logger import UnifiedLogger

    if logdir is None:
        ts = time.strftime("%Y-%m-%d_%H-%M-%S")
        logdir = os.path.expanduser(f"~/ray_results/{prefix}_{ts}")

    os.makedirs(logdir, exist_ok=True)

    def creator(config):
        return UnifiedLogger(config, logdir, loggers=None)

    return creator
