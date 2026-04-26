"""Tests for rlframework.storage — backends and CheckpointManager."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class TestLocalBackend:
    def test_upload_copies_file(self, tmp_dir):
        from rlframework.storage.backends import get_backend

        src = tmp_dir / "model.pt"
        src.write_bytes(b"fake weights")

        backend = get_backend("local", {"root": str(tmp_dir / "store")})
        result = backend.upload(str(src), "models/model.pt")

        dest = Path(result)
        assert dest.exists()
        assert dest.read_bytes() == b"fake weights"

    def test_upload_creates_parent_dirs(self, tmp_dir):
        from rlframework.storage.backends import get_backend

        src = tmp_dir / "a.bin"
        src.write_bytes(b"data")
        backend = get_backend("local", {"root": str(tmp_dir / "store")})
        result = backend.upload(str(src), "deep/nested/a.bin")
        assert Path(result).exists()

    def test_download_copies_file(self, tmp_dir):
        from rlframework.storage.backends import get_backend

        store = tmp_dir / "store"
        store.mkdir()
        src_in_store = store / "model.pt"
        src_in_store.write_bytes(b"weights")

        backend = get_backend("local", {"root": str(store)})
        dest = str(tmp_dir / "downloaded.pt")
        result = backend.download("model.pt", dest)
        assert Path(result).read_bytes() == b"weights"

    def test_get_backend_unknown_raises(self):
        from rlframework.storage.backends import get_backend

        with pytest.raises((KeyError, ValueError)):
            get_backend("nonexistent_backend", {})


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

class TestCheckpointManager:
    def test_upload_sync_calls_backend(self, tmp_dir, mock_backend):
        from rlframework.storage.checkpoint_manager import CheckpointManager

        src = tmp_dir / "checkpoint.pt"
        src.write_bytes(b"checkpoint data")

        manager = CheckpointManager(backend=mock_backend, upload_async=False)
        result = manager.upload_sync(str(src), "runs/ckpt.pt")

        mock_backend.upload.assert_called_once()
        assert result == "mock://runs/ckpt.pt"

    def test_upload_async_returns_future(self, tmp_dir, mock_backend):
        from concurrent.futures import Future

        from rlframework.storage.checkpoint_manager import CheckpointManager

        src = tmp_dir / "ckpt.pt"
        src.write_bytes(b"data")

        manager = CheckpointManager(backend=mock_backend, upload_async=True)
        future = manager.upload(str(src), "async/ckpt.pt")
        assert isinstance(future, Future)
        manager.shutdown()

    def test_upload_sync_mode_future_contains_backend_result(self, tmp_dir, mock_backend):
        from rlframework.storage.checkpoint_manager import CheckpointManager

        src = tmp_dir / "sync.pt"
        src.write_bytes(b"data")

        manager = CheckpointManager(backend=mock_backend, upload_async=False)
        future = manager.upload(str(src), "sync/ckpt.pt")

        assert future.result() == "mock://sync/ckpt.pt"

    def test_upload_directory_creates_tar(self, tmp_dir, mock_backend):
        from rlframework.storage.checkpoint_manager import CheckpointManager

        ckpt_dir = tmp_dir / "ckpt_dir"
        ckpt_dir.mkdir()
        (ckpt_dir / "policy.pt").write_bytes(b"policy")
        (ckpt_dir / "value.pt").write_bytes(b"value")

        manager = CheckpointManager(backend=mock_backend, upload_async=False)
        manager.upload(str(ckpt_dir), "runs/ckpt_dir.tar")
        # backend.upload should have been called with a .tar path
        call_args = mock_backend.upload.call_args[0]
        assert call_args[0].endswith(".tar")
        assert call_args[1] == "runs/ckpt_dir.tar"

    def test_upload_directory_appends_tar_suffix_when_missing(self, tmp_dir, mock_backend):
        from rlframework.storage.checkpoint_manager import CheckpointManager

        ckpt_dir = tmp_dir / "ckpt_dir"
        ckpt_dir.mkdir()
        (ckpt_dir / "weights.pt").write_bytes(b"weights")

        manager = CheckpointManager(backend=mock_backend, upload_async=False)
        manager.upload(str(ckpt_dir), "runs/ckpt_dir")
        call_args = mock_backend.upload.call_args[0]
        assert call_args[1] == "runs/ckpt_dir.tar"

    def test_retry_on_failure(self, tmp_dir):
        from rlframework.storage.checkpoint_manager import CheckpointManager

        fail_backend = MagicMock()
        fail_backend.upload.side_effect = [OSError("fail"), OSError("fail"), "mock://ok"]

        src = tmp_dir / "f.pt"
        src.write_bytes(b"x")

        manager = CheckpointManager(backend=fail_backend, upload_async=False, upload_retries=3)
        result = manager.upload_sync(str(src), "f.pt")
        assert result == "mock://ok"
        assert fail_backend.upload.call_count == 3


