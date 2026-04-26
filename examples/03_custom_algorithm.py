"""
Example 03: Custom Algorithm — overriding hooks
================================================
Demonstrates:
- Subclassing CustomPPO to override mixin hooks
- Custom training step logic (e.g. curriculum learning)

Run:
    python rlframework/examples/03_custom_algorithm.py
"""

from typing import Any

import ray

from rlframework.algorithms.ppo import CustomPPO, CustomPPOConfig
from rlframework.callbacks import FrameworkCallback
from rlframework.observability.reporters import FileReporter
from ray.rllib.utils.typing import ResultDict

# ---------------------------------------------------------------------------
# 1. Define a custom PPO subclass
# ---------------------------------------------------------------------------

class CurriculumPPO(CustomPPO):
    """PPO with simple curriculum: increase env difficulty over time."""

    def setup(self, config):
        super().setup(config)
        self._curriculum_level = 0

    # ------------------------------------------------------------------
    # Hook: called before every training step
    # ------------------------------------------------------------------
    def on_before_training_step(self) -> None:
        iteration = self.training_iteration
        # Advance curriculum every 20 iterations
        new_level = min(iteration // 2, 5)
        if new_level != self._curriculum_level:
            self._curriculum_level = new_level
            print(f"[CurriculumPPO] curriculum level -> {self._curriculum_level}")

    # ------------------------------------------------------------------
    # Hook: called after every training step
    # ------------------------------------------------------------------
    def on_after_training_step(self, result: dict[str, Any]) -> ResultDict:
        result["custom/curriculum_level"] = self._curriculum_level
        return result


# ---------------------------------------------------------------------------
# 2. Custom config extending CustomPPOConfig
# ---------------------------------------------------------------------------

class CurriculumPPOConfig(CustomPPOConfig):
    def __init__(self):
        super().__init__(algo_class=CurriculumPPO)


# ---------------------------------------------------------------------------
# 3. Train
# ---------------------------------------------------------------------------

ray.init(ignore_reinit_error=True)

reporters = [FileReporter(filepath="./logs/curriculum_ppo_metrics.jsonl")]

config = (
    CurriculumPPOConfig()
    .environment("CartPole-v1")
    .training(lr=3e-4, train_batch_size=2000)
    .env_runners(num_env_runners=2)
    .callbacks(FrameworkCallback.with_reporters(reporters))
)

algo = config.build()

for iteration in range(60):
    result = algo.train()
    mean_reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
    curriculum = result.get("custom/curriculum_level", 0)
    print(f"[iter {iteration:03d}] reward={mean_reward:.2f} | curriculum={curriculum}")

algo.stop()
ray.shutdown()
print("Done.")
