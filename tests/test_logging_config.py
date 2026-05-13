"""Tests for rlframework logging configuration helpers."""

import logging
import subprocess
import sys
import textwrap
from pathlib import Path

from rlframework.logging_config import setup_logging

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_python(script: str) -> None:
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_import_does_not_configure_global_logging() -> None:
    _run_python(
        """
        import logging

        root = logging.getLogger()
        before = (root.level, len(root.handlers), logging.getLogger("ray").level)

        import rlframework

        after = (root.level, len(root.handlers), logging.getLogger("ray").level)
        assert after == before, f"{before=!r} {after=!r}"
        assert any(
            isinstance(handler, logging.NullHandler)
            for handler in logging.getLogger("rlframework").handlers
        )
        """
    )


def test_setup_logging_configures_root_once() -> None:
    _run_python(
        """
        import logging

        from rlframework.logging_config import setup_logging

        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)
        logging.getLogger("ray").setLevel(logging.NOTSET)

        setup_logging()
        setup_logging()

        handlers = [
            handler
            for handler in root.handlers
            if getattr(handler, "_rlframework_handler", False)
        ]
        assert len(handlers) == 1
        assert root.level == logging.INFO
        assert handlers[0].level == logging.INFO
        assert logging.getLogger("ray").level == logging.WARNING
        """
    )


def test_setup_logging_can_target_package_only() -> None:
    root = logging.getLogger()
    package_logger = logging.getLogger("rlframework")
    ray_logger = logging.getLogger("ray")

    original_root_level = root.level
    original_root_handlers = list(root.handlers)
    original_package_level = package_logger.level
    original_package_handlers = list(package_logger.handlers)
    original_package_propagate = package_logger.propagate
    original_ray_level = ray_logger.level

    try:
        root.handlers.clear()
        root.setLevel(logging.ERROR)
        package_logger.handlers.clear()
        package_logger.addHandler(logging.NullHandler())
        package_logger.propagate = True
        ray_logger.setLevel(logging.NOTSET)

        setup_logging("DEBUG", configure_root=False, ray_level=None)

        handlers = [
            handler
            for handler in package_logger.handlers
            if getattr(handler, "_rlframework_handler", False)
        ]
        assert root.level == logging.ERROR
        assert len(root.handlers) == 0
        assert package_logger.level == logging.DEBUG
        assert package_logger.propagate is False
        assert len(handlers) == 1
        assert ray_logger.level == logging.NOTSET
    finally:
        root.handlers[:] = original_root_handlers
        root.setLevel(original_root_level)
        package_logger.handlers[:] = original_package_handlers
        package_logger.setLevel(original_package_level)
        package_logger.propagate = original_package_propagate
        ray_logger.setLevel(original_ray_level)
