"""Metric reporters - pluggable backends for reporting training metrics.

Each reporter implements a simple two-method interface:

    reporter.report(metrics: dict, iteration: int) -> None
    reporter.close() -> None

Built-in reporters
------------------
- :class:`FileReporter`      – append JSON lines to a local file
- :class:`InfluxDBReporter`  – write to InfluxDB v2 via HTTP
- :class:`PrometheusReporter` – expose metrics on a Prometheus push-gateway
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Any


class BaseReporter(ABC):
    """Abstract base class for all reporters."""

    @abstractmethod
    def report(self, metrics: dict[str, Any], iteration: int = 0, phase: str = "train") -> None:
        """Send *metrics* to the backend.

        Args:
            metrics: Flat dict of metric name → numeric value.
            iteration: Current training iteration number.
            phase: ``"train"`` or ``"eval"`` – indicates where the metrics
                originated.

        This method must **not** raise; swallow errors internally and log a
        warning at most so a failing reporter never crashes the training loop.
        """

    def close(self) -> None:  # noqa: B027  (intentionally optional override)
        """Release any backend connections or file handles (optional)."""


# ---------------------------------------------------------------------------
# FileReporter
# ---------------------------------------------------------------------------

class FileReporter(BaseReporter):
    """Appends one JSON line per call to *filepath*.

    Args:
        filepath: Destination file path.  Parent directories are created
            automatically.
        flush_every: Flush the file buffer after every *n* writes (default 1).
    """

    def __init__(self, filepath: str, flush_every: int = 1):
        import os

        self._filepath = filepath
        self._flush_every = flush_every
        self._write_count = 0
        self._fh = None  # 延迟打开

    def _ensure_file(self):
        """惰性打开文件句柄 (ray 需要对象可序列化)."""
        if self._fh is None:
            import os

            os.makedirs(
                os.path.dirname(os.path.abspath(self._filepath)), exist_ok=True
            )
            self._fh = open(self._filepath, "a", encoding="utf-8")

    def report(self, metrics: dict[str, Any], iteration: int = 0, phase: str = "train") -> None:
        self._ensure_file()
        record = {"timestamp": time.time(), "iteration": iteration, "phase": phase, **metrics}
        self._fh.write(json.dumps(record) + "\n")
        self._write_count += 1
        if self._write_count % self._flush_every == 0:
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def __del__(self):
        self.close()

    # ray 序列化支持
    def __getstate__(self):
        state = self.__dict__.copy()
        state["_fh"] = None  # 序列化时排除文件句柄
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


# ---------------------------------------------------------------------------
# InfluxDBReporter
# ---------------------------------------------------------------------------

class InfluxDBReporter(BaseReporter):
    """Writes metrics to an InfluxDB v2 instance using the line protocol.

    Args:
        url: Base URL of the InfluxDB instance, e.g. ``http://localhost:8086``.
        org: InfluxDB organisation name.
        bucket: Destination bucket name.
        token: Authentication token (``INFLUXDB_TOKEN`` env-var is a safe
            alternative – set ``token=None`` to read from env).
        measurement: InfluxDB measurement name (default ``"rl_training"``).
        timeout: HTTP request timeout in seconds.
    """

    def __init__(
        self,
        url: str,
        org: str,
        bucket: str,
        token: str | None = None,
        measurement: str = "rl_training",
        timeout: float = 5.0,
    ):
        import os
        self._url = f"{url.rstrip('/')}/api/v2/write"
        self._params = {"org": org, "bucket": bucket, "precision": "s"}
        _token = token or os.getenv("INFLUXDB_TOKEN", "")
        self._headers = {
            "Authorization": f"Token {_token}",
            "Content-Type": "text/plain; charset=utf-8",
        }
        self._measurement = measurement
        self._timeout = timeout

    def report(self, metrics: dict[str, Any], iteration: int = 0, phase: str = "train") -> None:
        try:
            import requests
        except ImportError:
            return  # silently skip if requests not installed

        ts = int(time.time())
        fields = ",".join(
            f"{k}={v}"
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and k != "training_iteration"
        )
        if not fields:
            return

        line = (
            f"{self._measurement},"
            f"iteration={iteration},phase={phase} "
            f"{fields} "
            f"{ts}"
        )
        try:
            requests.post(
                self._url,
                params=self._params,
                headers=self._headers,
                data=line.encode(),
                timeout=self._timeout,
            )
        except Exception:
            pass   # never crash the training loop


# ---------------------------------------------------------------------------
# PrometheusReporter
# ---------------------------------------------------------------------------

class PrometheusReporter(BaseReporter):
    """Pushes metrics to a Prometheus Push Gateway.

    Args:
        gateway: Push-gateway URL, e.g. ``http://pushgateway:9091``.
        job: Prometheus job label.
        grouping_key: Extra labels added to every push (optional).
    """

    def __init__(
        self,
        gateway: str,
        job: str = "rl_training",
        grouping_key: dict[str, str] | None = None,
    ):
        self._gateway = gateway
        self._job = job
        self._grouping_key = grouping_key or {}

    def report(self, metrics: dict[str, Any], iteration: int = 0, phase: str = "train") -> None:
        try:
            from prometheus_client import (
                CollectorRegistry,
                Gauge,
                push_to_gateway,
            )
        except ImportError:
            return  # silently skip if prometheus_client not installed

        registry = CollectorRegistry()
        for name, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            safe_name = name.replace("/", "_").replace("-", "_")
            g = Gauge(safe_name, safe_name, registry=registry)
            g.set(value)
        grouping_key = {**self._grouping_key, "phase": phase}
        try:
            push_to_gateway(
                self._gateway,
                job=self._job,
                registry=registry,
                grouping_key=grouping_key,
            )
        except Exception:
            pass
