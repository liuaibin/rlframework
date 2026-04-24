"""
Example 02: SAC on Pendulum-v1
================================
Demonstrates:
- CustomSAC with continuous action space
- InfluxDB metrics reporter (optional — falls back to FileReporter)
- PrometheusReporter for Grafana integration
- AutoCheckpoint for periodic saves (best practice)

Run:
    # Without external services (file reporter only)
    python rlframework/examples/02_sac_pendulum.py

    # With InfluxDB
    INFLUXDB_URL=http://localhost:8086 \
    INFLUXDB_TOKEN=my-token \
    python rlframework/examples/02_sac_pendulum.py
"""

import os

import ray
from rlframework.algorithms.sac import CustomSACConfig
from rlframework.logging.callbacks import FrameworkCallback
from rlframework.logging.reporters import FileReporter, InfluxDBReporter
from rlframework.storage import AutoCheckpoint, CheckpointManager
from rlframework.storage.backends import get_backend

# ---------------------------------------------------------------------------
# 1. Init Ray
# ---------------------------------------------------------------------------
ray.init(ignore_reinit_error=True)

# ---------------------------------------------------------------------------
# 2. Build reporters based on env vars
# ---------------------------------------------------------------------------
os.makedirs("./logs", exist_ok=True)
reporters = [FileReporter(filepath="./logs/pendulum_metrics.jsonl")]

influxdb_url = os.environ.get("INFLUXDB_URL")
if influxdb_url:
    reporters.append(
        InfluxDBReporter(
            url=influxdb_url,
            org=os.environ.get("INFLUXDB_ORG", "rl"),
            bucket=os.environ.get("INFLUXDB_BUCKET", "metrics"),
            token=os.environ.get("INFLUXDB_TOKEN", ""),
            measurement="sac_pendulum",
        )
    )
    print(f"InfluxDB reporter enabled: {influxdb_url}")

# ---------------------------------------------------------------------------
# 3. Storage backend — MinIO if configured, otherwise local
# ---------------------------------------------------------------------------
minio_endpoint = os.environ.get("MINIO_ENDPOINT")
if minio_endpoint:
    backend = get_backend(
        "minio",
        {
            "endpoint": minio_endpoint,
            "access_key": os.environ.get("MINIO_ACCESS_KEY", "minioadmin"),
            "secret_key": os.environ.get("MINIO_SECRET_KEY", "minioadmin"),
            "bucket": "rl-checkpoints",
            "secure": False,
        },
    )
    print(f"MinIO backend enabled: {minio_endpoint}")
else:
    backend = get_backend("local", {"root": "./checkpoints/pendulum"})

# ---------------------------------------------------------------------------
# 5. AutoCheckpoint — independent of callbacks, full control in the loop
# ---------------------------------------------------------------------------
ckpt_manager = CheckpointManager(backend=backend, upload_async=True, upload_retries=3)
auto = AutoCheckpoint(
    ckpt_manager,
    freq=20,
    local_dir="./checkpoints/pendulum",
)

# ---------------------------------------------------------------------------
# 6. Configure SAC — reporters wired automatically, no callback checkpointing
# ---------------------------------------------------------------------------
config = (
    CustomSACConfig()
    .environment("Pendulum-v1")
    .training(
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        train_batch_size=256,
        replay_buffer_config={
            "type": "MultiAgentReplayBuffer",
            "capacity": 100_000,
        },
        target_entropy="auto",
        tau=0.005,
        target_update_interval=1,
    )
    .env_runners(num_env_runners=1, rollout_fragment_length=1)
    .checkpointing(freq=20, local_dir="./checkpoints/pendulum")
    .callbacks(FrameworkCallback.with_reporters(reporters))
)

# ---------------------------------------------------------------------------
# 7. Train
# ---------------------------------------------------------------------------
algo = config.build()

best_reward = float("-inf")
for iteration in range(200):
    result = algo.train()
    mean_reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
    print(f"[iter {iteration:03d}] mean_reward={mean_reward:.2f}")

    # Periodic checkpoint — explicit and visible in the loop
    auto.step(algo, iteration=iteration, metrics=result)

    # Save best checkpoint
    if mean_reward > best_reward:
        best_reward = mean_reward
        ckpt_path = algo.save_to_path("./checkpoints/pendulum/best")
        ckpt_manager.upload(ckpt_path, "pendulum/best.tar")
        print(f"  -> new best checkpoint: {mean_reward:.2f}")

# Wait for any pending uploads
ckpt_manager.shutdown()

algo.stop()
ray.shutdown()
print(f"Done. Best reward: {best_reward:.2f}")
