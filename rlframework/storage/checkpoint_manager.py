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
from concurrent.futures import Future, ThreadPoolExecutor

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
        backend: "str | BaseBackend" = "local",
        backend_config: dict | None = None,
        upload_async: bool = True,
        upload_workers: int = 2,
        upload_retries: int = 3,
    ):
        if isinstance(backend, str):
            self._backend = get_backend(backend, backend_config or {})
        else:
            # Accept any duck-typed backend (BaseBackend subclass or mock)
            self._backend: BaseBackend = backend  # type: ignore[assignment]
        self._upload_async = upload_async
        self._executor = ThreadPoolExecutor(max_workers=upload_workers)
        self._retries = upload_retries

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload(
        self, local_path: str, remote_name: str | None = None
    ) -> Future:
        """Upload a checkpoint, sync or async depending on *upload_async*.

        Args:
            local_path: Path to the local checkpoint file or directory.
            remote_name: Object / key name in the remote store.  Defaults to
                the basename of *local_path*.

        Returns:
            A :class:`concurrent.futures.Future`.
        """
        if remote_name is None:
            remote_name = os.path.basename(local_path.rstrip("/"))
        if self._upload_async:
            return self._executor.submit(
                self._upload_with_retry, local_path, remote_name
            )
        # upload_async=False: run synchronously, then wrap result in a Future
        self._upload_with_retry(local_path, remote_name)
        future: Future = Future()
        future.set_result(remote_name)
        return future

    def upload_sync(
        self, local_path: str, remote_name: str | None = None
    ) -> str:
        """Explicitly run synchronously, ignoring *upload_async*."""
        if remote_name is None:
            remote_name = os.path.basename(local_path.rstrip("/"))
        return self._upload_with_retry(local_path, remote_name)

    def download(self, remote_name: str, local_path: str) -> str:
        """Download *remote_name* from the backend to *local_path*.

        Returns:
            Absolute local path of the downloaded file/directory.
        """
        return self._backend.download(remote_name, local_path)

    def shutdown(self, wait: bool = True) -> None:
        """Shutdown the upload thread pool."""
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
            remote_name = "m_" + remote_name + ".tar"
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
