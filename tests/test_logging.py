"""Tests for rlframework.logging — reporters and FrameworkCallback."""

import json
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# FileReporter
# ---------------------------------------------------------------------------

class TestFileReporter:
    def test_report_writes_json_line(self, tmp_dir):
        from rlframework.logging.reporters import FileReporter

        path = tmp_dir / "metrics.jsonl"
        reporter = FileReporter(filepath=str(path))
        reporter.report({"reward": 10.0, "loss": 0.5}, iteration=1)
        reporter.close()

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["reward"] == 10.0
        assert record["iteration"] == 1

    def test_report_appends_multiple_lines(self, tmp_dir):
        from rlframework.logging.reporters import FileReporter

        path = tmp_dir / "metrics.jsonl"
        reporter = FileReporter(filepath=str(path))
        for i in range(5):
            reporter.report({"step": i}, iteration=i)
        reporter.close()

        lines = path.read_text().strip().splitlines()
        assert len(lines) == 5

    def test_flush_every(self, tmp_dir):
        from rlframework.logging.reporters import FileReporter

        path = tmp_dir / "metrics.jsonl"
        reporter = FileReporter(filepath=str(path), flush_every=2)
        reporter.report({"v": 1}, iteration=0)
        reporter.report({"v": 2}, iteration=1)
        # After 2 reports the file should be flushed
        lines = path.read_text().strip().splitlines()
        assert len(lines) == 2
        reporter.close()

    def test_creates_parent_dirs(self, tmp_dir):
        from rlframework.logging.reporters import FileReporter

        path = tmp_dir / "deep" / "nested" / "metrics.jsonl"
        reporter = FileReporter(filepath=str(path))
        reporter.report({"x": 1}, iteration=0)
        reporter.close()
        assert path.exists()


# ---------------------------------------------------------------------------
# InfluxDBReporter (mocked HTTP)
# ---------------------------------------------------------------------------

class TestInfluxDBReporter:
    def test_report_sends_post_request(self, sample_metrics):
        from rlframework.logging.reporters import InfluxDBReporter

        with patch("requests.post") as mock_post:
            mock_post.return_value = MagicMock(status_code=204)
            reporter = InfluxDBReporter(
                url="http://fake:8086",
                org="org",
                bucket="bucket",
                token="token",
                measurement="test",
            )
            reporter.report({"reward": -200.0}, iteration=1)
            mock_post.assert_called_once()

    def test_report_failure_does_not_raise(self, sample_metrics):
        from rlframework.logging.reporters import InfluxDBReporter

        with patch("requests.Session.post") as mock_post:
            mock_post.side_effect = ConnectionError("refused")
            reporter = InfluxDBReporter(
                url="http://fake:8086",
                org="org",
                bucket="bucket",
                token="token",
                measurement="test",
            )
            # Should log warning but not raise
            reporter.report({"reward": -200.0}, iteration=1)


# ---------------------------------------------------------------------------
# FrameworkCallback — _extract_metrics
# ---------------------------------------------------------------------------

class TestFrameworkCallback:
    def test_extract_metrics_flattens_dict(self, sample_metrics):
        from rlframework.logging.callbacks import FrameworkCallback

        cb = FrameworkCallback.with_reporters([])
        flat = cb._extract_metrics(sample_metrics)

        assert isinstance(flat, dict)
        # Should contain flattened keys
        assert any("episode_return_mean" in k for k in flat)

    def test_with_reporters_accepts_empty_list(self):
        from rlframework.logging.callbacks import FrameworkCallback

        cb = FrameworkCallback.with_reporters([])
        assert cb is not None

    def test_reporters_receive_metrics(self, sample_metrics):
        from rlframework.logging.callbacks import FrameworkCallback

        reporter = MagicMock()
        cb = FrameworkCallback.with_reporters([reporter])

        # Add training_iteration so the callback can pass it to reporters
        sample_metrics["training_iteration"] = 5
        fake_algo = MagicMock()
        cb.on_train_result(algorithm=fake_algo, metrics_logger=None, result=sample_metrics)

        reporter.report.assert_called_once()
        # iteration should be 5 (from result["training_iteration"])
        call_args = reporter.report.call_args
        iteration_arg = call_args[1].get("iteration") or call_args[0][1]
        assert iteration_arg == 5
