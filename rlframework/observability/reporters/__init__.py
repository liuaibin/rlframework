"""Metric reporters - pluggable backends for reporting training metrics."""

import json
import time
from abc import ABC, abstractmethod
from typing import Any, TextIO


class BaseReporter(ABC):
    """Abstract base class for all reporters."""

    @abstractmethod
    def report(self, metrics: dict[str, Any], iteration: int = 0, phase: str = "train") -> None:
        """Send *metrics* to the backend."""

    def close(self) -> None:  # noqa: B027
        """Release backend connections or file handles (optional)."""


class FileReporter(BaseReporter):
    """Append one JSON line per report call."""

    def __init__(self, filepath: str, flush_every: int = 1) -> None:
        self._filepath = filepath
        self._flush_every = flush_every
        self._write_count = 0
        self._fh: TextIO | None = None

    def _ensure_file(self) -> TextIO:
        if self._fh is None:
            import os

            os.makedirs(os.path.dirname(os.path.abspath(self._filepath)), exist_ok=True)
            self._fh = open(self._filepath, "a", encoding="utf-8")
        return self._fh

    def report(self, metrics: dict[str, Any], iteration: int = 0, phase: str = "train") -> None:
        fh = self._ensure_file()
        record = {
            "timestamp": time.time(),
            "iteration": iteration,
            "phase": phase,
            **metrics,
        }
        fh.write(json.dumps(record) + "\n")
        self._write_count += 1
        if self._write_count % self._flush_every == 0:
            fh.flush()

    def close(self) -> None:
        if self._fh is not None and not self._fh.closed:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def __del__(self) -> None:
        self.close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_fh"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class InfluxDBReporter(BaseReporter):
    """Write metrics to an InfluxDB v2 instance using line protocol."""

    def __init__(
        self,
        url: str,
        org: str,
        bucket: str,
        token: str | None = None,
        measurement: str = "rl_training",
        timeout: float = 5.0,
    ) -> None:
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
            return

        ts = int(time.time())
        fields = ",".join(
            f"{k}={v}"
            for k, v in metrics.items()
            if isinstance(v, (int, float)) and k != "training_iteration"
        )
        if not fields:
            return
        line = f"{self._measurement},iteration={iteration},phase={phase} {fields} {ts}"
        try:
            requests.post(
                self._url,
                params=self._params,
                headers=self._headers,
                data=line.encode(),
                timeout=self._timeout,
            )
        except Exception:
            pass


class PrometheusReporter(BaseReporter):
    """Push metrics to a Prometheus Push Gateway."""

    def __init__(
        self,
        gateway: str,
        job: str = "rl_training",
        grouping_key: dict[str, str] | None = None,
    ) -> None:
        self._gateway = gateway
        self._job = job
        self._grouping_key = grouping_key or {}

    def report(self, metrics: dict[str, Any], iteration: int = 0, phase: str = "train") -> None:
        try:
            from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
        except ImportError:
            return

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
