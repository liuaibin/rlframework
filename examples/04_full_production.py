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
from rlframework.callbacks import FrameworkCallback

# ---------------------------------------------------------------------------
# Configuration (override via env vars in CI/CD)
# ---------------------------------------------------------------------------
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "rl-checkpoints")

INFLUXDB_URL = os.environ.get("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_ORG = os.environ.get("INFLUXDB_ORG", "rl")
INFLUXDB_BUCKET = os.environ.get("INFLUXDB_BUCKET", "metrics")
INFLUXDB_TOKEN = os.environ.get("INFLUXDB_TOKEN", "my-super-secret-token")

PROMETHEUS_GW = os.environ.get("PROMETHEUS_GW", "localhost:9091")

MODEL_NAME = "sac_pendulum"
EXPERIMENT_NAME = "pendulum_production_v1"
TOTAL_ITERATIONS = 300
CHECKPOINT_FREQ = 20

# ---------------------------------------------------------------------------
# 1. Init Ray
# ---------------------------------------------------------------------------
ray.init(ignore_reinit_error=True)

# ---------------------------------------------------------------------------
# 2. Reporter configs
# ---------------------------------------------------------------------------
metric_reporters = ["file", "influxdb", "prometheus"]
reporter_configs = {
    "influxdb": {
        "url": INFLUXDB_URL,
        "org": INFLUXDB_ORG,
        "bucket": INFLUXDB_BUCKET,
        "token": INFLUXDB_TOKEN,
        "measurement": MODEL_NAME,
    },
    "prometheus": {
        "gateway": PROMETHEUS_GW,
        "job": EXPERIMENT_NAME,
    },
}

# ---------------------------------------------------------------------------
# 3. SAC config
# ---------------------------------------------------------------------------
config = (
    CustomSACConfig()
    .framework_run(EXPERIMENT_NAME, root_dir="./runs")
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
    .evaluation(evaluation_interval=CHECKPOINT_FREQ)
    .checkpointing(freq=CHECKPOINT_FREQ)
    .metrics(reporters=metric_reporters, reporter_configs=reporter_configs)
    .storage(
        backend="minio",
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        bucket=MINIO_BUCKET,
        secure=False,
        upload_async=True,
        best_upload_freq=1,
    )
    .callbacks(FrameworkCallback.with_reporters([], collect_resource_stats=True))
)

# ---------------------------------------------------------------------------
# 4. Training loop
# ---------------------------------------------------------------------------
algo = config.build()

for iteration in range(TOTAL_ITERATIONS):
    result = algo.train()
    mean_reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
    print(f"[{EXPERIMENT_NAME}][iter {iteration:04d}] reward={mean_reward:.3f}")

# ---------------------------------------------------------------------------
# 5. Finalize
# ---------------------------------------------------------------------------
algo.stop()
ray.shutdown()
layout = config.framework_layout
assert layout is not None
print(f"Experiment complete. Local run artifacts: {layout.run_dir}")
