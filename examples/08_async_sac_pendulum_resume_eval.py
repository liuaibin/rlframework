"""
Example 08: Async SAC on Pendulum-v1 with parallel evaluation and resume
========================================================================
Demonstrates:
- AsyncCustomSACConfig for the asynchronous SAC pipeline.
- RLlib parallel evaluation using separate evaluation EnvRunners.
- Optional resume from a previous RLlib checkpoint via ``.resume_from(...)``.

Run fresh:
    python examples/08_async_sac_pendulum_resume_eval.py

Resume from a checkpoint:
    RESUME_FROM=./runs/async_sac_pendulum_2026_06_02_17_30_00/checkpoints/iter_000020 \\
    python examples/08_async_sac_pendulum_resume_eval.py

Notes:
- On Ray >= 2.54, AsyncCustomSAC uses non-blocking EnvRunner sampling and async
  Learner updates. Older Ray versions fall back to the parent SAC training loop.
- ``evaluation_parallel_to_training=True`` overlaps evaluation with training.
  ``evaluation_duration="auto"`` keeps evaluation from intentionally running
  longer than the parallel training iteration.
"""

import math
import os
import time

import ray

from rlframework.algorithms.async_sac import AsyncCustomSACConfig

EXPERIMENT_NAME = "async_sac_pendulum"
TRAINING_ITERS = int(os.environ.get("TRAINING_ITERS", "100"))
RESUME_FROM = os.environ.get("RESUME_FROM", "").strip()


def _format_reward(value: object) -> str:
    """Format reward values that may be missing from early RLlib results."""
    if isinstance(value, (int, float)) and not math.isnan(float(value)):
        return f"{float(value):.2f}"
    return "n/a"


# ---------------------------------------------------------------------------
# 1. Init Ray
# ---------------------------------------------------------------------------
ray.init(ignore_reinit_error=True)


# ---------------------------------------------------------------------------
# 2. Configure Async SAC
# ---------------------------------------------------------------------------
config = (
    AsyncCustomSACConfig()
    .framework_run(EXPERIMENT_NAME, root_dir="./runs")
    .environment("Pendulum-v1")
    .training(
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        train_batch_size_per_learner=256,
        num_steps_sampled_before_learning_starts=1500,
        replay_buffer_config={
            "type": "EpisodeReplayBuffer",
            "capacity": 100_000,
        },
        target_entropy="auto",
        tau=0.005,
    )
    .env_runners(
        num_env_runners=1,
        num_envs_per_env_runner=1,
        rollout_fragment_length=16,
        max_requests_in_flight_per_env_runner=1,
    )
    .learners(
        num_learners=1,
        num_gpus_per_learner=0,
        max_requests_in_flight_per_learner=1,
    )
    .evaluation(
        evaluation_interval=1,
        evaluation_parallel_to_training=True,
        evaluation_duration="auto",
        evaluation_duration_unit="timesteps",
        evaluation_force_reset_envs_before_iteration=True,
        evaluation_num_env_runners=1,
    )
    .checkpointing(freq=20)
    .metrics(reporters=["file"])
    .storage(upload_async=True, best_upload_freq=5)
)

if RESUME_FROM:
    print(f"Resuming from checkpoint: {RESUME_FROM}")
    config = config.resume_from(RESUME_FROM)
else:
    print("No RESUME_FROM set, starting fresh training.")


# ---------------------------------------------------------------------------
# 3. Build and train
# ---------------------------------------------------------------------------
algo = config.build()
layout = config.framework_layout
assert layout is not None

for iteration in range(TRAINING_ITERS):
    iter_start = time.perf_counter()
    result = algo.train()
    iter_time_s = time.perf_counter() - iter_start

    train_reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
    eval_reward = (
        result.get("evaluation", {}).get("env_runners", {}).get("episode_return_mean", float("nan"))
    )

    print(
        f"[iter {iteration:03d}] "
        f"time={iter_time_s:.2f}s "
        f"train_reward={_format_reward(train_reward)} "
        f"eval_reward={_format_reward(eval_reward)}"
    )

algo.stop()
ray.shutdown()

print(f"Done. Run directory: {layout.run_dir}")
print(f"Metrics written to: {layout.metrics_dir / 'metrics.jsonl'}")
print(f"Checkpoints written to: {layout.checkpoint_dir}")
