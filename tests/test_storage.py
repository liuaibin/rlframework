"""Tests for rlframework.storage — backends, CheckpointManager, ModelManager."""

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
        result = manager.upload(str(src), "runs/ckpt.pt")

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

    def test_retry_on_failure(self, tmp_dir):
        from rlframework.storage.checkpoint_manager import CheckpointManager

        fail_backend = MagicMock()
        fail_backend.upload.side_effect = [OSError("fail"), OSError("fail"), "mock://ok"]

        src = tmp_dir / "f.pt"
        src.write_bytes(b"x")

        manager = CheckpointManager(backend=fail_backend, upload_async=False, upload_retries=3)
        result = manager.upload(str(src), "f.pt")
        assert result == "mock://ok"
        assert fail_backend.upload.call_count == 3


# ---------------------------------------------------------------------------
# ModelManager
# ---------------------------------------------------------------------------

class TestModelManager:
    def test_register_and_list(self, tmp_dir):
        from rlframework.storage.model_manager import ModelManager

        mgr = ModelManager(catalogue_path=str(tmp_dir / "cat.json"))
        mgr.register("sac", "v1", "/path/v1.tar", metrics={"reward": 10.0})
        mgr.register("sac", "v2", "/path/v2.tar", metrics={"reward": 20.0})

        versions = mgr.list_versions("sac")
        assert len(versions) == 2
        assert {v["version"] for v in versions} == {"v1", "v2"}

    def test_latest_returns_last_registered(self, tmp_dir):
        from rlframework.storage.model_manager import ModelManager

        mgr = ModelManager(catalogue_path=str(tmp_dir / "cat.json"))
        mgr.register("ppo", "v1", "/p/v1.tar")
        mgr.register("ppo", "v2", "/p/v2.tar")
        latest = mgr.latest("ppo")
        assert latest["version"] == "v2"

    def test_best_max(self, tmp_dir):
        from rlframework.storage.model_manager import ModelManager

        mgr = ModelManager(catalogue_path=str(tmp_dir / "cat.json"))
        mgr.register("ppo", "v1", "/p/v1.tar", metrics={"reward": 5.0})
        mgr.register("ppo", "v2", "/p/v2.tar", metrics={"reward": 15.0})
        mgr.register("ppo", "v3", "/p/v3.tar", metrics={"reward": 12.0})

        best = mgr.best("ppo", metric="reward", mode="max")
        assert best["version"] == "v2"

    def test_best_min(self, tmp_dir):
        from rlframework.storage.model_manager import ModelManager

        mgr = ModelManager(catalogue_path=str(tmp_dir / "cat.json"))
        mgr.register("ppo", "v1", "/p/v1.tar", metrics={"loss": 0.4})
        mgr.register("ppo", "v2", "/p/v2.tar", metrics={"loss": 0.1})
        best = mgr.best("ppo", metric="loss", mode="min")
        assert best["version"] == "v2"

    def test_catalogue_persists_to_disk(self, tmp_dir):
        from rlframework.storage.model_manager import ModelManager

        path = str(tmp_dir / "cat.json")
        mgr = ModelManager(catalogue_path=path)
        mgr.register("net", "v1", "/net/v1.tar", metrics={"acc": 0.9})

        # Reload from disk
        mgr2 = ModelManager(catalogue_path=path)
        assert mgr2.latest("net")["version"] == "v1"

    def test_list_models(self, tmp_dir):
        from rlframework.storage.model_manager import ModelManager

        mgr = ModelManager(catalogue_path=str(tmp_dir / "cat.json"))
        mgr.register("modelA", "v1", "/a.tar")
        mgr.register("modelB", "v1", "/b.tar")
        models = mgr.list_models()
        assert set(models) == {"modelA", "modelB"}
