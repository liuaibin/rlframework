"""
Example 01: PPO on CartPole-v1 (minimal usage)
================================================
Demonstrates the simplest possible usage of rlframework:
- CustomPPO with default settings
- FileReporter for local metrics logging
- Local checkpoint storage
- Model reload and resume training

Run:
    python rlframework/examples/01_ppo_cartpole.py
"""
import os
import glob

import ray
from rlframework.algorithms.ppo import CustomPPOConfig
from rlframework.logging.callbacks import FrameworkCallback
from rlframework.logging.reporters import FileReporter
from rlframework.storage.backends import get_backend
from rlframework.storage.checkpoint_manager import CheckpointManager

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
# 3. Build CheckpointManager (local backend)
# ---------------------------------------------------------------------------
backend = get_backend("local", {"root": "./checkpoints/cartpole"})
ckpt_manager = CheckpointManager(backend=backend, upload_async=True)
# ---------------------------------------------------------------------------
# 4. Configure algorithm
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
    .callbacks(lambda: FrameworkCallback(reporters=reporters))
)

# ---------------------------------------------------------------------------
# 5. Helper: find latest checkpoint
# ---------------------------------------------------------------------------
def find_latest_checkpoint(checkpoint_dir: str):
    """Find the most recent checkpoint by modification time."""
    pattern = os.path.join(checkpoint_dir, "iter_*")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)

# ---------------------------------------------------------------------------
# 6. Train or Resume
# ---------------------------------------------------------------------------
algo = config.build()

# Check for existing checkpoints to resume training
checkpoint_base = os.path.abspath("./checkpoints/cartpole")
latest_ckpt = find_latest_checkpoint(checkpoint_base)
print(latest_ckpt)
if latest_ckpt:
    print(f"Found existing checkpoint: {latest_ckpt}")
    print("Restoring model from checkpoint...")
    algo.restore_from_path(latest_ckpt)
    # Extract iteration number from checkpoint path for resuming
    iter_num = int(os.path.basename(latest_ckpt).split("_")[1])
    start_iter = iter_num
    print(f"Resuming from iteration {start_iter}")
else:
    print("No existing checkpoint found, starting fresh training.")
    start_iter = 0

for iteration in range(start_iter, 15):
    result = algo.train()
    mean_reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
    print(f"[iter {iteration:03d}] mean_reward={mean_reward:.2f}")

    # Save checkpoint every 5 iterations
    if (iteration + 1) % 5 == 0:
        # 使用绝对路径 (ray 需要完整路径)
        ckpt_path = algo.save_to_path(
            os.path.abspath(f"./checkpoints/cartpole/iter_{iteration+1}")
        )
        ckpt_manager.upload(ckpt_path, f"iter_{iteration+1}")
        print(f"  -> checkpoint saved: {ckpt_path}")

algo.stop()
ray.shutdown()
print("Done. Metrics written to ./logs/cartpole_metrics.jsonl")
