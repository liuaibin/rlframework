"""
Example 04: Full Production Setup — MinIO + InfluxDB + Prometheus
==================================================================
Demonstrates a production-grade training run with:
- SAC on Pendulum-v1
- MinIO checkpoint storage with async upload
- InfluxDB v2 metrics reporter
- Prometheus push gateway reporter
- Custom resource metrics (CPU/memory via psutil)

Prerequisites (docker-compose snippet):

    services:
      minio:
        image: minio/minio
        command: server /data --console-address ":9001"
        environment:
          MINIO_ROOT_USER: minioadmin
          MINIO_ROOT_PASSWORD: minioadmin
        ports: ["9000:9000", "9001:9001"]

      influxdb:
        image: influxdb:2.7
        environment:
          DOCKER_INFLUXDB_INIT_MODE: setup
          DOCKER_INFLUXDB_INIT_USERNAME: admin
          DOCKER_INFLUXDB_INIT_PASSWORD: adminpass
          DOCKER_INFLUXDB_INIT_ORG: rl
          DOCKER_INFLUXDB_INIT_BUCKET: metrics
          DOCKER_INFLUXDB_INIT_ADMIN_TOKEN: my-super-secret-token
        ports: ["8086:8086"]

Run:
    docker-compose up -d
    python rlframework/examples/04_full_production.py
"""

import os

import ray

from rlframework.algorithms.sac import CustomSACConfig
from rlframework.logging.callbacks import FrameworkCallback
from rlframework.logging.reporters import (
    FileReporter,
    InfluxDBReporter,
    PrometheusReporter,
)
from rlframework.storage.backends import get_backend
from rlframework.storage.checkpoint_manager import CheckpointManager

# ---------------------------------------------------------------------------
# Configuration (override via env vars in CI/CD)
# ---------------------------------------------------------------------------
MINIO_ENDPOINT   = os.environ.get("MINIO_ENDPOINT",   "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET     = os.environ.get("MINIO_BUCKET",     "rl-checkpoints")

INFLUXDB_URL     = os.environ.get("INFLUXDB_URL",    "http://localhost:8086")
INFLUXDB_ORG     = os.environ.get("INFLUXDB_ORG",    "rl")
INFLUXDB_BUCKET  = os.environ.get("INFLUXDB_BUCKET", "metrics")
INFLUXDB_TOKEN   = os.environ.get("INFLUXDB_TOKEN",  "my-super-secret-token")

PROMETHEUS_GW    = os.environ.get("PROMETHEUS_GW",   "localhost:9091")

MODEL_NAME       = "sac_pendulum"
EXPERIMENT_NAME  = "pendulum_production_v1"
TOTAL_ITERATIONS = 300
CHECKPOINT_FREQ  = 20

# ---------------------------------------------------------------------------
# 1. Init Ray
# ---------------------------------------------------------------------------
ray.init(ignore_reinit_error=True)

# ---------------------------------------------------------------------------
# 2. Reporters
# ---------------------------------------------------------------------------
os.makedirs("./logs", exist_ok=True)

reporters = [
    # Always write to file as safety net
    FileReporter(filepath=f"./logs/{EXPERIMENT_NAME}.jsonl"),
    # InfluxDB for Grafana dashboards
    InfluxDBReporter(
        url=INFLUXDB_URL,
        org=INFLUXDB_ORG,
        bucket=INFLUXDB_BUCKET,
        token=INFLUXDB_TOKEN,
        measurement=MODEL_NAME,
    ),
    # Prometheus for alerting
    PrometheusReporter(
        gateway=PROMETHEUS_GW,
        job=EXPERIMENT_NAME,
    ),
]

# ---------------------------------------------------------------------------
# 3. Storage: MinIO backend with async upload
# ---------------------------------------------------------------------------
backend = get_backend(
    "minio",
    {
        "endpoint":   MINIO_ENDPOINT,
        "access_key": MINIO_ACCESS_KEY,
        "secret_key": MINIO_SECRET_KEY,
        "bucket":     MINIO_BUCKET,
        "secure":     False,
    },
)
ckpt_manager = CheckpointManager(
    backend=backend,
    upload_async=True,
    upload_retries=3,
)

# ---------------------------------------------------------------------------
# 4. SAC config
# ---------------------------------------------------------------------------
config = (
    CustomSACConfig()
    .environment("Pendulum-v1")
    .training(
        lr=3e-4,
        train_batch_size=256,
        replay_buffer_config={
            "type": "MultiAgentReplayBuffer",
            "capacity": 200_000,
        },
        target_entropy="auto",
        tau=0.005,
    )
    .env_runners(num_env_runners=2, rollout_fragment_length=1)
    .callbacks(
        FrameworkCallback.with_reporters(reporters, collect_resource_stats=True)
    )
)

# ---------------------------------------------------------------------------
# 5. Training loop
# ---------------------------------------------------------------------------
algo = config.build()
best_reward = float("-inf")

for iteration in range(TOTAL_ITERATIONS):
    result = algo.train()
    mean_reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
    print(f"[{EXPERIMENT_NAME}][iter {iteration:04d}] reward={mean_reward:.3f}")

    # Periodic checkpoint
    if (iteration + 1) % CHECKPOINT_FREQ == 0:
        local_path = algo.save_to_path(
            f"./checkpoints/{EXPERIMENT_NAME}/iter_{iteration+1}"
        )
        remote_name = f"{EXPERIMENT_NAME}/iter_{iteration+1}.tar"
        ckpt_manager.upload(local_path, remote_name)
        print(f"  -> checkpoint uploaded: {remote_name}")

    # Track best model
    if mean_reward > best_reward:
        best_reward = mean_reward
        local_best = algo.save_to_path(f"./checkpoints/{EXPERIMENT_NAME}/best")
        ckpt_manager.upload(local_best, f"{EXPERIMENT_NAME}/best.tar")

# ---------------------------------------------------------------------------
# 6. Finalize
# ---------------------------------------------------------------------------
ckpt_manager.shutdown()

for reporter in reporters:
    reporter.close()

algo.stop()
ray.shutdown()
print(f"Experiment complete. Best reward: {best_reward:.3f}")
