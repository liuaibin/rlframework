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


'''
# 1. 默认控制台日志
  setup_logging()

  # 2. 打开 debug
  setup_logging(level="DEBUG")

  # 3. 只配置 rlframework，不影响业务项目 root logger
  setup_logging(configure_root=False, ray_level=None)

  # 4. 单独控制某些 logger
  setup_logging(
      level="INFO",
      logger_levels={
          "rlframework": "DEBUG",
          "ray": "ERROR",
      },
  )

  # 5. 强制重配已有 handler
  setup_logging(force=True)
'''

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
        for existing_handler in list(logger.handlers):
            logger.removeHandler(existing_handler)
            existing_handler.close()

    logger.setLevel(level)

    handler = _find_rlframework_handler(logger)
    if handler is None and (force or not _has_configured_handler(logger)):
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
