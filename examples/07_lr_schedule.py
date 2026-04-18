"""
Example 07: Dynamic Learning Rate Schedule
============================================
Demonstrates:
- LR schedule via RLlib's built-in [[timestep, lr], ...] format
- Validator support for schedule format checking
- Observing LR changes during training

Run:
    python rlframework/examples/07_lr_schedule.py
"""

import ray
from rlframework.algorithms.ppo import CustomPPOConfig
from rlframework.logging.callbacks import FrameworkCallback
from rlframework.logging.reporters import FileReporter

# ---------------------------------------------------------------------------
# 1. Init Ray
# ---------------------------------------------------------------------------
ray.init(ignore_reinit_error=True)

# ---------------------------------------------------------------------------
# 2. Define a learning rate schedule
#    Format: [[timestep, lr_value], ...]
#    RLlib linearly interpolates between entries.
# ---------------------------------------------------------------------------
lr_schedule = [
    [0,     1e-3],    # start at 1e-3
    [20000, 5e-4],    # decay to 5e-4 by 20k steps
    [50000, 1e-4],    # decay to 1e-4 by 50k steps
    [80000, 1e-5],    # decay to 1e-5 by 80k steps
]

# ---------------------------------------------------------------------------
# 3. Configure
# ---------------------------------------------------------------------------
reporters = [FileReporter(filepath="./logs/lr_schedule_metrics.jsonl")]

config = (
    CustomPPOConfig()
    .environment("CartPole-v1")
    .training(
        lr=lr_schedule,               # <-- pass schedule instead of fixed float
        train_batch_size=4000,
        num_epochs=10,
        minibatch_size=128,
    )
    .env_runners(num_env_runners=2)
    .callbacks(lambda:FrameworkCallback.with_reporters(reporters))
)

# ---------------------------------------------------------------------------
# 4. Train and observe LR changes
# ---------------------------------------------------------------------------
algo = config.build()

for iteration in range(50):
    result = algo.train()

    mean_reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
    total_steps = int(result.get("num_env_steps_sampled_lifetime", 0))

    # Read current LR via foreach_learner -> metrics.peek()
    # Structure: roe.get() -> {module_id: {"default_optimizer_learning_rate": float, ...}}
    current_lr = "N/A"
    try:
        learner_results = algo.learner_group.foreach_learner(
            lambda learner: learner.metrics.peek()
        )
        for item in learner_results:
            roe = item.result_or_error
            if not roe.ok:
                continue
            metrics_dict = roe.get()
            for module_id, stats in metrics_dict.items():
                if module_id.startswith("__"):
                    continue
                if isinstance(stats, dict) and "default_optimizer_learning_rate" in stats:
                    lr_val = stats["default_optimizer_learning_rate"]
                    try:
                        current_lr = f"{float(lr_val):.6f}"
                    except (TypeError, ValueError):
                        pass
                if current_lr != "N/A":
                    break
            if current_lr != "N/A":
                break
    except Exception:
        pass

    print(
        f"[iter {iteration:03d}] "
        f"steps={total_steps:>7d}  "
        f"lr={current_lr}  "
        f"reward={mean_reward:.2f}"
    )

algo.stop()
ray.shutdown()
print("\nDone. LR decayed according to schedule.")
