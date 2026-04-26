"""CheckpointManager - save / load algorithm checkpoints with pluggable backends.

Usage::

    from rlframework.storage import CheckpointManager

    # Local backend (default)
    mgr = CheckpointManager()

    # MinIO backend
    mgr = CheckpointManager(
        backend="minio",
        backend_config=dict(
            endpoint="minio.example.com:9000",
            access_key="minioadmin",
            secret_key="minioadmin",
            bucket="rl-checkpoints",
        ),
    )

    # Upload with the configured async/sync mode
    future = mgr.upload(local_path, remote_name="exp1/iter_100")
"""

import logging
import os
import shutil
import time
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, cast

from rlframework.storage.backends import BaseBackend, get_backend

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages checkpoint persistence with sync-local + async-remote upload.

    Args:
        backend: Either a :class:`~rlframework.storage.backends.BaseBackend`
            instance **or** a backend name string (``"local"``, ``"minio"``,
            ``"s3"``).  Passing an instance lets you share a pre-configured
            backend across multiple managers.
        backend_config: Keyword arguments forwarded to the backend constructor
            when *backend* is provided as a string.
        upload_async: When ``True`` (default), uploads run in a background
            thread pool.  Set to ``False`` for fully synchronous uploads.
        upload_workers: Thread-pool size for async uploads.
        upload_retries: Number of retries on upload failure.
    """

    def __init__(
        self,
        backend: str | BaseBackend = "local",
        backend_config: dict[str, Any] | None = None,
        upload_async: bool = True,
        upload_workers: int = 2,
        upload_retries: int = 3,
    ) -> None:
        self._backend: BaseBackend
        if isinstance(backend, str):
            self._backend = get_backend(backend, backend_config or {})
        else:
            # Accept any duck-typed backend (BaseBackend subclass or mock)
            self._backend = backend
        self._upload_async = upload_async
        self._upload_workers = upload_workers
        self._executor: ThreadPoolExecutor | None = None
        self._retries = upload_retries

    def _get_executor(self) -> ThreadPoolExecutor:
        """Lazily create the thread pool (avoids pickling ThreadPoolExecutor)."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=self._upload_workers)
        return self._executor

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_executor"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload(
        self,
        local_path: str,
        remote_name: str | None = None,
        async_mode: bool | None = None,
    ) -> Future[str] | str:
        """Upload a checkpoint with optional per-call sync/async override.

        Args:
            local_path: Path to the local checkpoint file or directory.
            remote_name: Object / key name in the remote store.  Defaults to
                the basename of *local_path*.
            async_mode: Per-call override of async behavior.
                - ``None`` (default): follow manager-level ``upload_async``.
                - ``True``: force async and return a Future.
                - ``False``: force sync and return a string path.

        Returns:
            A :class:`concurrent.futures.Future` (async mode) or uploaded
            remote path string (forced sync mode).
        """
        if remote_name is None:
            remote_name = os.path.basename(local_path.rstrip("/"))
        remote_name = self._normalize_remote_name(local_path, remote_name)
        run_async = self._upload_async if async_mode is None else async_mode

        if run_async:
            return self._get_executor().submit(self._upload_with_retry, local_path, remote_name)

        # Preserve legacy behavior: default sync mode still returns a Future.
        result = self._upload_with_retry(local_path, remote_name)
        if async_mode is False:
            return result

        future: Future[str] = Future()
        future.set_result(result)
        return future

    def upload_sync(self, local_path: str, remote_name: str | None = None) -> str:
        """Deprecated: use ``upload(..., async_mode=False)`` instead."""
        warnings.warn(
            "upload_sync() is deprecated; use upload(..., async_mode=False).",
            DeprecationWarning,
            stacklevel=2,
        )
        result = self.upload(
            local_path=local_path,
            remote_name=remote_name,
            async_mode=False,
        )
        return cast(str, result)

    def download(self, remote_name: str, local_path: str) -> str:
        """Download *remote_name* from the backend to *local_path*.

        Returns:
            Absolute local path of the downloaded file/directory.
        """
        return self._backend.download(remote_name, local_path)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the upload thread pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=wait)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _upload_with_retry(self, local_path: str, remote_name: str) -> str:
        """Upload with automatic retries and directory-to-tarball conversion."""
        upload_path = local_path
        tmp_tar: str | None = None

        if os.path.isdir(local_path):
            tar_base = local_path.rstrip(os.sep)
            upload_path = shutil.make_archive(tar_base, "tar", local_path)
            tmp_tar = upload_path

        self._fsync(upload_path)
        last_exc: Exception | None = None

        try:
            for attempt in range(1, self._retries + 1):
                try:
                    logger.info(f"Uploading {upload_path} to {remote_name}")
                    result = self._backend.upload(upload_path, remote_name)
                    return result
                except Exception as exc:
                    last_exc = exc
                    time.sleep(attempt)

            raise RuntimeError(
                f"Upload failed after {self._retries} attempts: {last_exc}"
            ) from last_exc
        finally:
            if tmp_tar and os.path.exists(tmp_tar):
                os.remove(tmp_tar)

    @staticmethod
    def _normalize_remote_name(local_path: str, remote_name: str) -> str:
        """Normalize remote object name for directory uploads."""
        if os.path.isdir(local_path) and not remote_name.endswith(".tar"):
            return f"{remote_name}.tar"
        return remote_name

    @staticmethod
    def _fsync(path: str) -> None:
        """Best-effort fsync before upload."""
        try:
            if os.path.isfile(path):
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        except Exception:
            pass
