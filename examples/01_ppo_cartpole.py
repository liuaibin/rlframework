"""
Example 01: PPO on CartPole-v1 (minimal usage)
================================================
Demonstrates the simplest possible usage of rlframework:
- CustomPPO with default settings
- FileReporter for local metrics logging
- Automatic periodic checkpoint saving (every N iterations)
- Automatic best-model saving (on eval improvement, requires evaluation)

Everything checkpoint-related is driven by the config — no manual save logic
in the training loop needed.

Run:
    python rlframework/examples/01_ppo_cartpole.py
"""

import glob
import os

import ray

from rlframework.algorithms.ppo import CustomPPOConfig
from rlframework.callbacks import FrameworkCallback
from rlframework.observability.reporters import FileReporter

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
# 3. Configure algorithm
# ---------------------------------------------------------------------------
# All checkpointing is handled automatically by FrameworkCallback:
#   - Periodic save every 5 iterations  (framework_checkpointing freq=5, local only)
#   - Best-model save on eval improvement (requires .evaluation())
#   - Best-model upload every 5 improvements via .storage()
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
    .evaluation(evaluation_interval=5)
    .framework_checkpointing(freq=5, local_dir="./checkpoints/cartpole")
    .storage(upload_async=True, best_upload_freq=1)
    .callbacks(FrameworkCallback.with_reporters(reporters))
)


# ---------------------------------------------------------------------------
# 4. Helper: find latest checkpoint for resume
# ---------------------------------------------------------------------------
def find_latest_checkpoint(checkpoint_dir: str):
    """Find the most recent checkpoint by modification time."""
    pattern = os.path.join(checkpoint_dir, "iter_*")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# 5. Build algorithm
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

# ---------------------------------------------------------------------------
# 6. Training loop — no manual checkpoint logic needed
# ---------------------------------------------------------------------------
for iteration in range(start_iter, 15):
    result = algo.train()
    mean_reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
    print(f"[iter {iteration:03d}] mean_reward={mean_reward:.2f}")
    # Periodic checkpointing + eval-driven best-model saving happen automatically
    # inside FrameworkCallback.

algo.stop()
ray.shutdown()
print("Done. Metrics written to ./logs/cartpole_metrics.jsonl")
