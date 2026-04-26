"""
Example 09: Custom Replay Buffer (PrioritizedSumTreeBuffer)
============================================================
Demonstrates:
- Plugging a custom prioritized replay buffer into SAC
- Using PrioritizedSumTreeBuffer from rlframework.utils.replay_buffers
- Tuning alpha (priority) and beta (IS-correction) hyper-parameters
- Inspecting replay buffer stats (size, added/sampled timesteps) during training

Run:
    python rlframework/examples/09_custom_replay_buffer.py
"""

import ray

from rlframework.algorithms.sac import CustomSACConfig
from rlframework.callbacks import FrameworkCallback
from rlframework.observability.reporters import FileReporter
from rlframework.utils.replay_buffers import PrioritizedSumTreeBuffer

# ===========================================================================
# Init
# ===========================================================================
ray.init(ignore_reinit_error=True)

reporters = [FileReporter(filepath="./logs/custom_buffer_metrics.jsonl")]

# ===========================================================================
# Configure SAC with a custom prioritized replay buffer
# ===========================================================================
config = (
    CustomSACConfig()
    .environment("Pendulum-v1")
    .training(
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        train_batch_size_per_learner=256,
        replay_buffer_config={
            "type": PrioritizedSumTreeBuffer,  # use the framework's buffer
            "capacity": 50_000,
            "alpha": 0.6,  # priority exponent (0=uniform, 1=full priority)
            "beta": 0.4,  # IS-weight exponent (higher = less correction)
        },
    )
    .env_runners(num_env_runners=2)
    .callbacks(lambda: FrameworkCallback.with_reporters(reporters))
)

# ===========================================================================
# Train
# ===========================================================================
algo = config.build()

for iteration in range(40):
    result = algo.train()
    mean_reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
    rb = algo.local_replay_buffer
    assert isinstance(rb, PrioritizedSumTreeBuffer), (
        f"Expected PrioritizedSumTreeBuffer, got {type(rb)}"
    )
    print(
        f"[verified] buffer is PrioritizedSumTreeBuffer, _epsilon={rb._epsilon}, _beta_override={rb._beta_override}"
    )
    # Inspect replay buffer stats from the local replay buffer attached to the algorithm
    buffer_size = "N/A"
    added_steps = "N/A"
    sampled_steps = "N/A"
    try:
        rb = algo.local_replay_buffer
        if rb is not None:
            buffer_size = len(rb)
            added_steps = rb.get_added_timesteps()
            sampled_steps = rb.get_sampled_timesteps()
    except Exception:
        pass

    print(
        f"[iter {iteration:03d}] "
        f"reward={mean_reward:.2f}  "
        f"buf_size={buffer_size}  "
        f"added={added_steps}  "
        f"sampled={sampled_steps}"
    )

algo.stop()
ray.shutdown()
print("\nDone. SAC trained with PrioritizedSumTreeBuffer.")
