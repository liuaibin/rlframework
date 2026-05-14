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

# ---------------------------------------------------------------------------
# 1. Init Ray
# ---------------------------------------------------------------------------
ray.init(ignore_reinit_error=True)

# ---------------------------------------------------------------------------
# 2. Configure algorithm
# ---------------------------------------------------------------------------
# Run artifacts are managed by rlframework:
#   - RLlib logs: runs/cartpole/rllib_logs/
#   - Metrics: runs/cartpole/metrics/metrics.jsonl
#   - Checkpoints: runs/cartpole/checkpoints/
#   - Local storage uploads: runs/cartpole/storage/
#
# Checkpointing is handled automatically by FrameworkCallback:
#   - Periodic save every 5 iterations  (framework_checkpointing freq=5, local only)
#   - Best-model save on eval improvement (requires .evaluation())
#   - Best-model upload on every improvement via .storage(best_upload_freq=1)
# ---------------------------------------------------------------------------
config = (
    CustomPPOConfig()
    .framework_run("cartpole_1", root_dir="./runs")
    .environment("CartPole-v1")
    .training(
        lr=5e-5,
        train_batch_size=4000,
        num_epochs=10,
        minibatch_size=128,
    )
    .env_runners(num_env_runners=2)
    .evaluation(evaluation_interval=5)
    .framework_checkpointing(freq=5)
    .metrics(reporters=["file"])
    .storage(upload_async=True, best_upload_freq=1)
)


# ---------------------------------------------------------------------------
# 3. Helper: find latest checkpoint for resume
# ---------------------------------------------------------------------------
def find_latest_checkpoint(checkpoint_dir: str):
    """Find the most recent checkpoint by modification time."""
    pattern = os.path.join(checkpoint_dir, "iter_*")
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    return max(checkpoints, key=os.path.getmtime)


# ---------------------------------------------------------------------------
# 4. Build algorithm
# ---------------------------------------------------------------------------
algo = config.build()

layout = config.framework_layout
assert layout is not None
checkpoint_base = str(layout.checkpoint_dir.resolve())
latest_ckpt = find_latest_checkpoint(checkpoint_base)
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
# 5. Training loop — no manual checkpoint logic needed
# ---------------------------------------------------------------------------
for iteration in range(start_iter, 15):
    result = algo.train()
    mean_reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
    print(f"[iter {iteration:03d}] mean_reward={mean_reward:.2f}")
    # Periodic checkpointing + eval-driven best-model saving happen automatically
    # inside FrameworkCallback.

algo.stop()
ray.shutdown()
print(f"Done. Metrics written to {layout.metrics_dir / 'metrics.jsonl'}")
