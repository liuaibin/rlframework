"""
Example 06: Custom Evaluation Metrics
======================================
Demonstrates:
- Adding custom metrics during training AND evaluation episodes
- Using on_episode_end to compute episode-level metrics (e.g. success rate)
- Separate train/eval metric reporting via FrameworkCallback
- Evaluation running every 5 training iterations

Run:
    python rlframework/examples/06_custom_eval_metrics.py
"""

import ray

from rlframework.algorithms.ppo import CustomPPOConfig
from rlframework.logging.callbacks import FrameworkCallback
from rlframework.logging.reporters import FileReporter


# ---------------------------------------------------------------------------
# 1. Custom callback that adds episode-level metrics
# ---------------------------------------------------------------------------

class EvalMetricsCallback(FrameworkCallback):
    """Adds success rate and episode efficiency metrics via MetricsLogger."""

    def on_episode_end(
        self, *, episode, env_runner=None, metrics_logger=None, env=None, **kwargs
    ):
        # Compute episode-level stats
        ep_return = episode.get_return()
        ep_length = episode.env_steps()

        # Log custom metrics through MetricsLogger (the RLlib-correct way)
        if metrics_logger is not None:
            # Custom metric: did the episode achieve a high return?
            metrics_logger.log_value("success", float(ep_return > 195.0), reduce="mean")

            # Custom metric: reward efficiency (return per step)
            efficiency = ep_return / ep_length if ep_length > 0 else 0.0
            metrics_logger.log_value("reward_efficiency", efficiency, reduce="mean")


# ---------------------------------------------------------------------------
# 2. Init Ray
# ---------------------------------------------------------------------------
ray.init(ignore_reinit_error=True)

# ---------------------------------------------------------------------------
# 3. Configure with evaluation enabled
# ---------------------------------------------------------------------------
reporters = [FileReporter(filepath="./logs/eval_metrics.jsonl")]

config = (
    CustomPPOConfig()
    .environment("CartPole-v1")
    .training(lr=3e-4, train_batch_size=4000, num_epochs=10, minibatch_size=128)
    .env_runners(num_env_runners=2)
    .evaluation(
        evaluation_interval=5,          # evaluate every 5 training iterations
        evaluation_duration=10,         # run 10 evaluation episodes
        evaluation_num_env_runners=1,
    )
    .callbacks(lambda: EvalMetricsCallback.with_reporters(reporters))
)

# ---------------------------------------------------------------------------
# 4. Train and observe separate train/eval metrics
# ---------------------------------------------------------------------------
algo = config.build()

for iteration in range(30):
    result = algo.train()

    # Training metrics
    env_runners = result.get("env_runners", {})
    mean_reward = env_runners.get("episode_return_mean", float("nan"))
    train_success = env_runners.get("success", "N/A")

    print(f"[iter {iteration:03d}] TRAIN  reward={mean_reward:.2f}  success={train_success}")

    # Evaluation metrics (only present every 5 iterations)
    eval_results = result.get("evaluation", {})
    if eval_results:
        eval_env = eval_results.get("env_runners", {})
        eval_reward = eval_env.get("episode_return_mean", float("nan"))
        eval_success = eval_env.get("success", "N/A")
        print(f"           EVAL   reward={eval_reward:.2f}  success={eval_success}")

algo.stop()
ray.shutdown()
print("\nDone. Metrics (with phase=train/eval) in ./logs/eval_metrics.jsonl")
