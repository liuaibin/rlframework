# rlframework Examples

Quick-start examples showing how to use each part of the framework.

## Examples Overview

| File | Algorithm | Key Features |
|------|-----------|--------------|
| [01_ppo_cartpole.py](01_ppo_cartpole.py) | PPO | Minimal setup, local file reporter, local checkpoint |
| [02_sac_pendulum.py](02_sac_pendulum.py) | SAC | InfluxDB reporter (optional), MinIO upload (optional) |
| [03_custom_algorithm.py](03_custom_algorithm.py) | Custom PPO | Hook overrides, curriculum learning, gradient clipping |
| [04_full_production.py](04_full_production.py) | SAC | MinIO + InfluxDB + Prometheus, ModelManager catalogue |

---

## Running the Examples

### Example 1 — PPO CartPole (no external services needed)

```bash
python rlframework/examples/01_ppo_cartpole.py
```

Output files:
- `./logs/cartpole_metrics.jsonl` — JSON-lines metrics
- `./checkpoints/cartpole/` — local checkpoints

---

### Example 2 — SAC Pendulum

```bash
# Without external services
python rlframework/examples/02_sac_pendulum.py

# With InfluxDB
INFLUXDB_URL=http://localhost:8086 \
INFLUXDB_TOKEN=my-token \
python rlframework/examples/02_sac_pendulum.py

# With MinIO
MINIO_ENDPOINT=localhost:9000 \
python rlframework/examples/02_sac_pendulum.py
```

---

### Example 3 — Custom Algorithm

Demonstrates subclassing `CustomPPO` and overriding lifecycle hooks:

```python
class CurriculumPPO(CustomPPO):
    def on_before_training_step(self) -> None:
        # runs before every training_step()
        ...

    def on_after_training_step(self, result) -> None:
        # runs after every training_step(), can inject extra metrics
        ...
```

```bash
python rlframework/examples/03_custom_algorithm.py
```

---

### Example 4 — Full Production Setup

Requires MinIO, InfluxDB, and Prometheus push gateway running locally.

```bash
# Start services
docker-compose up -d   # see file header for docker-compose snippet

# Run experiment
python rlframework/examples/04_full_production.py
```

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MINIO_ENDPOINT` | `localhost:9000` | MinIO endpoint |
| `MINIO_ACCESS_KEY` | `minioadmin` | MinIO access key |
| `MINIO_SECRET_KEY` | `minioadmin` | MinIO secret key |
| `MINIO_BUCKET` | `rl-checkpoints` | MinIO bucket name |
| `INFLUXDB_URL` | `http://localhost:8086` | InfluxDB v2 URL |
| `INFLUXDB_ORG` | `rl` | InfluxDB org |
| `INFLUXDB_BUCKET` | `metrics` | InfluxDB bucket |
| `INFLUXDB_TOKEN` | `my-super-secret-token` | InfluxDB auth token |
| `PROMETHEUS_GW` | `localhost:9091` | Prometheus push gateway |

---

## Common Patterns

### Add a reporter at any time

```python
from rlframework.logging.reporters import InfluxDBReporter, FileReporter

reporters = [
    FileReporter("./logs/metrics.jsonl"),
    InfluxDBReporter(url="http://influxdb:8086", org="rl", bucket="runs", token="..."),
]
```

### Switch storage backend

```python
from rlframework.storage.backends import get_backend

# Local
backend = get_backend("local", {"root": "/tmp/checkpoints"})

# MinIO
backend = get_backend("minio", {
    "endpoint": "minio:9000",
    "access_key": "admin",
    "secret_key": "password",
    "bucket": "rl",
})

# S3
backend = get_backend("s3", {
    "bucket": "my-rl-bucket",
    "prefix": "experiments/",
    "region_name": "us-east-1",
})
```

### Look up best model version

```python
from rlframework.storage.model_manager import ModelManager

mgr = ModelManager("./logs/catalogue.json")
best = mgr.best("sac_pendulum", metric="episode_return_mean", mode="max")
print(best["path"])  # remote path in MinIO/S3
```
