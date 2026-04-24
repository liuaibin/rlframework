"""
Example 01: PPO on CartPole-v1 (minimal usage)
================================================
Demonstrates the simplest possible usage of rlframework:
- CustomPPO with default settings
- FileReporter for local metrics logging
- AutoCheckpoint for periodic saves (best practice)
- Manual best-model save in training loop

Run:
    python rlframework/examples/01_ppo_cartpole.py
"""
import glob
import os

import ray
from rlframework.algorithms.ppo import CustomPPOConfig
from rlframework.logging.callbacks import FrameworkCallback
from rlframework.logging.reporters import FileReporter
from rlframework.storage import AutoCheckpoint, CheckpointManager
from rlframework.storage.backends import get_backend

# ---------------------------------------------------------------------------
# 1. Init Ray
# ---------------------------------------------------------------------------
ray.init(ignore_reinit_error=True)

# ---------------------------------------------------------------------------
# 2. Build reporters
# ---------------------------------------------------------------------------
os.makedirs("./logs", exist_ok=True)
reporters = [FileReporter(filepath="./logs/cartpole_metrics.jsonl")]

# ---------------------------------------------------------------------------
# 3. Storage backend for remote uploads (optional)
# ---------------------------------------------------------------------------
backend = get_backend("local", {"root": "./checkpoints/cartpole"})

# ---------------------------------------------------------------------------
# 5. AutoCheckpoint — independent of callbacks, full control in the loop
# ---------------------------------------------------------------------------
ckpt_manager = CheckpointManager(backend=backend, upload_async=True)
auto = AutoCheckpoint(
    ckpt_manager,
    freq=5,
    local_dir="./checkpoints/cartpole",
)

# ---------------------------------------------------------------------------
# 6. Configure algorithm — reporters wired automatically, no callback checkpointing
# ---------------------------------------------------------------------------
config = (
    CustomPPOConfig()
    .environment("CartPole-v1")
    .training(
        lr=5e-5,
        train_batch_size=4000,
        num_epochs=10,
        minibatch_size=128,
    )
    .env_runners(num_env_runners=2)
    .checkpointing(freq=5, local_dir="./checkpoints/cartpole")
    .callbacks(FrameworkCallback.with_reporters(reporters))
)

# ---------------------------------------------------------------------------
# 7. Helper: find latest checkpoint
# ---------------------------------------------------------------------------
def find_latest_checkpoint(checkpoint_dir: str):
    """Find the most recent checkpoint by modification time."""
    pattern = os.path.join(checkpoint_dir, "iter_*")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

# ---------------------------------------------------------------------------
# 8. Train or Resume
# ---------------------------------------------------------------------------
algo = config.build()

checkpoint_base = os.path.abspath("./checkpoints/cartpole")
latest_ckpt = find_latest_checkpoint(checkpoint_base)
print(latest_ckpt)
if latest_ckpt:
    print(f"Found existing checkpoint: {latest_ckpt}")
    print("Restoring model from checkpoint...")
    algo.restore_from_path(latest_ckpt)
    iter_num = int(os.path.basename(latest_ckpt).split("_")[1])
    start_iter = iter_num
    print(f"Resuming from iteration {start_iter}")
else:
    print("No existing checkpoint found, starting fresh training.")
    start_iter = 0

best_reward = float("-inf")
for iteration in range(start_iter, 15):
    result = algo.train()
    mean_reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
    print(f"[iter {iteration:03d}] mean_reward={mean_reward:.2f}")

    # Periodic checkpoint — now explicit and visible in the loop
    auto.step(algo, iteration=iteration, metrics=result)

    # Best model save
    if mean_reward > best_reward:
        best_reward = mean_reward
        ckpt_path = algo.save_to_path(
            os.path.abspath(f"./checkpoints/cartpole/best")
        )
        ckpt_manager.upload(ckpt_path, "best.tar")
        print(f"  -> new best checkpoint: {mean_reward:.2f}")

# Shutdown the shared manager to flush async uploads
ckpt_manager.shutdown()
algo.stop()
ray.shutdown()
print("Done. Metrics written to ./logs/cartpole_metrics.jsonl")
