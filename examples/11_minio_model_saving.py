"""
Example 11: MinIO Model Saving
================================
Demonstrates:
- Configuring MinIO as the checkpoint storage backend
- Using CheckpointManager for upload/download

Prerequisites:
    # Start a local MinIO server
    docker run -d --name minio -p 9000:9000 -p 9001:9001 \
        -e MINIO_ROOT_USER=minioadmin \
        -e MINIO_ROOT_PASSWORD=minioadmin \
        minio/minio server /data --console-address ":9001"

Run:
    python rlframework/examples/11_minio_model_saving.py
"""

import os

import ray

from rlframework.algorithms.ppo import CustomPPOConfig
from rlframework.storage.checkpoint_manager import CheckpointManager

# =========================================================================
# 1. Configuration — override via env vars in CI / production
# =========================================================================
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET = os.environ.get("MINIO_BUCKET", "rl-checkpoints")

MODEL_NAME = "ppo_cartpole_example"
TOTAL_ITERATIONS = 20
SAVE_FREQ = 5  # save checkpoint every 5 iterations

# =========================================================================
# 2. Create CheckpointManager with MinIO backend
# =========================================================================
ckpt_manager = CheckpointManager(
    backend="minio",
    backend_config={
        "endpoint": MINIO_ENDPOINT,
        "access_key": MINIO_ACCESS_KEY,
        "secret_key": MINIO_SECRET_KEY,
        "bucket": MINIO_BUCKET,
    },
    upload_async=True,
    upload_retries=3,
)

# =========================================================================
# 3. Train and periodically save checkpoints to MinIO
# =========================================================================
ray.init(ignore_reinit_error=True)

config = (
    CustomPPOConfig()
    .framework_run(MODEL_NAME, root_dir="./runs")
    .environment("CartPole-v1")
    .training(lr=3e-4, train_batch_size=4000, num_epochs=10)
    .env_runners(num_env_runners=2)
    .metrics(reporters=["file"])
)
algo = config.build()
layout = config.framework_layout
assert layout is not None

for iteration in range(TOTAL_ITERATIONS):
    result = algo.train()
    mean_reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
    print(f"[iter {iteration:03d}] reward={mean_reward:.2f}")

    # Save checkpoint every SAVE_FREQ iterations
    if (iteration + 1) % SAVE_FREQ == 0:
        version = f"iter_{iteration + 1}"
        ckpt_dir = algo.save_to_path(str(layout.checkpoint_dir / version))
        remote_name = f"{MODEL_NAME}/{version}.tar"
        ckpt_manager.upload(ckpt_dir, remote_name)
        print(f"  -> saved {version} to MinIO (reward={mean_reward:.2f})")

algo.stop()
ckpt_manager.shutdown()

ray.shutdown()
print("\nDone. Checkpoints uploaded to MinIO.")
