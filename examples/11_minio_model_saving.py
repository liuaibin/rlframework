"""
Example 11: MinIO Model Saving
================================
Demonstrates:
- Configuring MinIO as the checkpoint storage backend
- Using CheckpointManager for upload/download
- Using ModelStore for unified save + metadata tracking
- Using ModelManager to query best version by metric

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
import tempfile

import ray

from rlframework.algorithms.ppo import CustomPPOConfig
from rlframework.storage.model_store import ModelStore

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
# 2. Create ModelStore with MinIO backend
# =========================================================================
store = ModelStore(
    backend="minio",
    backend_config=dict(
        endpoint=MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        bucket=MINIO_BUCKET,
    ),
    catalogue_path="./logs/model_catalogue.json",
    upload_async=True,
    upload_workers=2,
    upload_retries=3,
)

# =========================================================================
# 3. Train and periodically save checkpoints to MinIO
# =========================================================================
ray.init(ignore_reinit_error=True)

config = (
    CustomPPOConfig()
    .environment("CartPole-v1")
    .training(lr=3e-4, train_batch_size=4000, num_epochs=10)
    .env_runners(num_env_runners=2)
)
algo = config.build()

best_reward = float("-inf")

for iteration in range(TOTAL_ITERATIONS):
    result = algo.train()
    mean_reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
    print(f"[iter {iteration:03d}] reward={mean_reward:.2f}")

    # Save checkpoint every SAVE_FREQ iterations
    if (iteration + 1) % SAVE_FREQ == 0:
        # Save RLlib checkpoint to a temp directory
        ckpt_dir = algo.save(tempfile.mkdtemp(prefix="rl_ckpt_")).checkpoint.path

        version = f"iter_{iteration + 1}"
        metrics = {"episode_return_mean": mean_reward}

        # Upload to MinIO + register in catalogue
        store.save(
            name=MODEL_NAME,
            version=version,
            local_path=ckpt_dir,
            metrics=metrics,
            metadata={"iteration": iteration + 1},
            upload_async=True,
        )
        print(f"  -> saved {version} to MinIO (reward={mean_reward:.2f})")

        # Also save as "best" if this is the highest reward
        if mean_reward > best_reward:
            best_reward = mean_reward
            store.save_best(
                name=MODEL_NAME,
                local_path=ckpt_dir,
                metrics=metrics,
            )
            print(f"  -> updated 'best' (reward={best_reward:.2f})")

algo.stop()

# =========================================================================
# 4. Query the catalogue for the best model
# =========================================================================
best_info = store.get_best(MODEL_NAME, metric="episode_return_mean", mode="max")
if best_info:
    print(f"\nBest model: version={best_info.get('version')}, "
          f"metrics={best_info.get('metrics')}")

# To download the best model later:
#   local_path = store.load_best(MODEL_NAME, metric="episode_return_mean",
#                                local_dir="./loaded_models")

ray.shutdown()
print("\nDone. Checkpoints uploaded to MinIO, metadata in model_catalogue.json.")
