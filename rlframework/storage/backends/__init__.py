"""Storage backends - abstract interface and concrete implementations.

Provides
--------
- :class:`BaseBackend`   – interface all backends must implement
- :class:`LocalBackend`  – copy to a local directory
- :class:`MinIOBackend`  – upload / download via the MinIO Python SDK
- :class:`S3Backend`     – upload / download via boto3

Factory
-------
Use :func:`get_backend` to instantiate a backend by name::

    from rlframework.storage.backends import get_backend

    backend = get_backend("minio", {
        "endpoint": "minio.example.com:9000",
        "access_key": "minioadmin",
        "secret_key": "minioadmin",
        "bucket": "rl-checkpoints",
    })
"""

import os
import shutil
from abc import ABC, abstractmethod

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

class BaseBackend(ABC):
    """Minimal interface every storage backend must implement."""

    @abstractmethod
    def upload(self, local_path: str, remote_name: str) -> str:
        """Upload *local_path* and return the remote URI / key."""

    @abstractmethod
    def download(self, remote_name: str, local_path: str) -> str:
        """Download *remote_name* to *local_path* and return local path."""


# ---------------------------------------------------------------------------
# LocalBackend
# ---------------------------------------------------------------------------

class LocalBackend(BaseBackend):
    """Copies files/directories inside a local *root* directory.

    Args:
        root: Destination root directory.  Created if absent.
    """

    def __init__(self, root: str = "./checkpoints"):
        self._root = root
        os.makedirs(root, exist_ok=True)

    def upload(self, local_path: str, remote_name: str) -> str:
        dest = os.path.join(self._root, remote_name)
        os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
        if os.path.isdir(local_path):
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(local_path, dest)
        else:
            shutil.copy2(local_path, dest)
        return dest

    def download(self, remote_name: str, local_path: str) -> str:
        src = os.path.join(self._root, remote_name)
        if os.path.isdir(src):
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
            shutil.copytree(src, local_path)
        else:
            shutil.copy2(src, local_path)
        return local_path


# ---------------------------------------------------------------------------
# MinIOBackend
# ---------------------------------------------------------------------------

class MinIOBackend(BaseBackend):
    """Stores checkpoints in a MinIO / S3-compatible object store.

    Args:
        endpoint: MinIO endpoint, e.g. ``"minio.example.com:9000"``.
        access_key: Access key (or set env ``MINIO_ACCESS_KEY``).
        secret_key: Secret key (or set env ``MINIO_SECRET_KEY``).
        bucket: Target bucket name.  Created automatically if absent.
        secure: Use HTTPS when ``True`` (default ``False`` for internal MinIO).
    """

    def __init__(
        self,
        endpoint: str,
        access_key: str = "",
        secret_key: str = "",
        bucket: str = "rl-checkpoints",
        secure: bool = False,
    ):
        import os as _os
        _ak = access_key or _os.getenv("MINIO_ACCESS_KEY", "")
        _sk = secret_key or _os.getenv("MINIO_SECRET_KEY", "")
        try:
            from minio import Minio  # type: ignore
            self._client = Minio(endpoint, access_key=_ak, secret_key=_sk, secure=secure)
            if not self._client.bucket_exists(bucket):
                self._client.make_bucket(bucket)
        except ImportError as exc:
            raise ImportError(
                "MinIOBackend requires the 'minio' package.  "
                "Install with: pip install minio"
            ) from exc
        self._bucket = bucket

    def upload(self, local_path: str, remote_name: str) -> str:
        self._client.fput_object(self._bucket, remote_name, local_path)
        return f"minio://{self._bucket}/{remote_name}"

    def download(self, remote_name: str, local_path: str) -> str:
        self._client.fget_object(self._bucket, remote_name, local_path)
        return local_path


# ---------------------------------------------------------------------------
# S3Backend
# ---------------------------------------------------------------------------

class S3Backend(BaseBackend):
    """Stores checkpoints in AWS S3.

    Credentials are resolved via the standard boto3 chain
    (env vars / ~/.aws/credentials / IAM role).

    Args:
        bucket: Target S3 bucket name.
        prefix: Optional key prefix prepended to every remote name.
        region_name: AWS region (optional).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region_name: str = "",
    ):
        try:
            import boto3  # type: ignore
            kwargs = {}
            if region_name:
                kwargs["region_name"] = region_name
            self._s3 = boto3.client("s3", **kwargs)
        except ImportError as exc:
            raise ImportError(
                "S3Backend requires the 'boto3' package.  "
                "Install with: pip install boto3"
            ) from exc
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")

    def _key(self, remote_name: str) -> str:
        return f"{self._prefix}/{remote_name}" if self._prefix else remote_name

    def upload(self, local_path: str, remote_name: str) -> str:
        key = self._key(remote_name)
        self._s3.upload_file(local_path, self._bucket, key)
        return f"s3://{self._bucket}/{key}"

    def download(self, remote_name: str, local_path: str) -> str:
        key = self._key(remote_name)
        self._s3.download_file(self._bucket, key, local_path)
        return local_path


# ---------------------------------------------------------------------------
# Registry / factory
# ---------------------------------------------------------------------------

_BACKEND_REGISTRY: dict[str, type[BaseBackend]] = {
    "local": LocalBackend,
    "minio": MinIOBackend,
    "s3": S3Backend,
}


def get_backend(name: str, config: dict) -> BaseBackend:
    """Instantiate a backend by name.

    Args:
        name: One of ``"local"``, ``"minio"``, ``"s3"``.
        config: Keyword arguments forwarded to the backend constructor.

    Raises:
        KeyError: When *name* is not registered.
    """
    if name not in _BACKEND_REGISTRY:
        raise KeyError(
            f"Unknown storage backend '{name}'. "
            f"Available: {list(_BACKEND_REGISTRY)}"
        )
    return _BACKEND_REGISTRY[name](**config)
