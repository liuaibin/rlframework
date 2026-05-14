# rlframework Examples

Quick-start examples showing how to use each part of the framework.

## Examples Overview

| File | Algorithm | Key Features |
|------|-----------|--------------|
| [01_ppo_cartpole.py](01_ppo_cartpole.py) | PPO | Minimal setup, managed run layout, local metrics/checkpoints |
| [02_sac_pendulum.py](02_sac_pendulum.py) | SAC | File metrics, optional InfluxDB, managed checkpointing |
| [03_custom_algorithm.py](03_custom_algorithm.py) | Custom PPO | Hook overrides, curriculum learning, gradient clipping |
| [04_full_production.py](04_full_production.py) | SAC | MinIO + InfluxDB + Prometheus production setup |

---

## Running the Examples

### Example 1 — PPO CartPole (no external services needed)

```bash
python rlframework/examples/01_ppo_cartpole.py
```

Output files:
- `./runs/cartpole/rllib_logs/` — RLlib JSON/CSV/TensorBoard logs
- `./runs/cartpole/metrics/metrics.jsonl` — JSON-lines metrics
- `./runs/cartpole/checkpoints/` — local checkpoints
- `./runs/cartpole/storage/` — local storage backend artifacts

---

### Example 2 — SAC Pendulum

```bash
# Without external services
python rlframework/examples/02_sac_pendulum.py

# With InfluxDB
INFLUXDB_URL=http://localhost:8086 \
INFLUXDB_TOKEN=my-token \
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

### Configure reporters

```python
config = (
    CustomPPOConfig()
    .framework_run("cartpole", root_dir="./runs")
    .metrics(
        reporters=["file", "influxdb"],
        reporter_configs={
            "influxdb": {
                "url": "http://influxdb:8086",
                "org": "rl",
                "bucket": "runs",
                "token": "...",
            }
        },
    )
)
```

### Configure storage backend

```python
# Local
config = CustomPPOConfig().framework_run("cartpole").storage()

# MinIO
config = (
    CustomPPOConfig()
    .framework_run("cartpole")
    .storage(
        backend="minio",
        endpoint="minio:9000",
        access_key="admin",
        secret_key="password",
        bucket="rl",
    )
)

# S3
config = (
    CustomPPOConfig()
    .framework_run("cartpole")
    .storage(
        backend="s3",
        bucket="my-rl-bucket",
        prefix="experiments/",
        region_name="us-east-1",
    )
)
```

### Manually upload a checkpoint when needed

```python
from rlframework.storage.checkpoint_manager import CheckpointManager

ckpt_mgr = CheckpointManager(
    backend="minio",
    backend_config={"endpoint": "minio:9000", "bucket": "rl"},
    upload_async=True,
)
ckpt_mgr.upload("./runs/cartpole/checkpoints/iter_100", "cartpole/iter_100.tar")
```
