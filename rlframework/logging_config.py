"""Logging helpers for applications using rlframework."""

from __future__ import annotations

import logging
from collections.abc import Mapping

DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_RLFRAMEWORK_HANDLER_ATTR = "_rlframework_handler"


def _find_rlframework_handler(logger: logging.Logger) -> logging.Handler | None:
    for handler in logger.handlers:
        if getattr(handler, _RLFRAMEWORK_HANDLER_ATTR, False):
            return handler
    return None


def _has_configured_handler(logger: logging.Logger) -> bool:
    return any(not isinstance(handler, logging.NullHandler) for handler in logger.handlers)


def setup_logging(
    level: int | str = logging.INFO,
    *,
    configure_root: bool = True,
    ray_level: int | str | None = logging.WARNING,
    logger_levels: Mapping[str, int | str] | None = None,
    force: bool = False,
) -> None:
    """Configure logging explicitly for rlframework applications.

    Importing rlframework does not configure process-wide logging. Call this
    helper from an application entry point when rlframework should provide a
    default console logging setup.
    """
    logger = logging.getLogger() if configure_root else logging.getLogger("rlframework")

    if force:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    logger.setLevel(level)

    handler = _find_rlframework_handler(logger)
    if handler is None and not _has_configured_handler(logger):
        handler = logging.StreamHandler()
        setattr(handler, _RLFRAMEWORK_HANDLER_ATTR, True)
        logger.addHandler(handler)

    if handler is not None:
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(DEFAULT_LOG_FORMAT))

    if not configure_root:
        logger.propagate = False

    if ray_level is not None:
        logging.getLogger("ray").setLevel(ray_level)

    for logger_name, logger_level in (logger_levels or {}).items():
        logging.getLogger(logger_name).setLevel(logger_level)
