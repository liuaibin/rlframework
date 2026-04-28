"""
Example 02: SAC on Pendulum-v1
================================
Demonstrates:
- CustomSAC with continuous action space
- InfluxDB metrics reporter (optional — falls back to FileReporter)
- Automatic periodic checkpoint saving + best-model saving (on eval improvement)

Everything checkpoint-related is driven by the config — no manual save logic
in the training loop needed.

Run:
    # Without external services (file reporter only)
    python rlframework/examples/02_sac_pendulum.py

    # With InfluxDB
    INFLUXDB_URL=http://localhost:8086 \
    INFLUXDB_TOKEN=my-token \
    python rlframework/examples/02_sac_pendulum.py
"""

import os

import ray

from rlframework.algorithms.sac import CustomSACConfig
from rlframework.callbacks import FrameworkCallback
from rlframework.observability.reporters import FileReporter, InfluxDBReporter

# ---------------------------------------------------------------------------
# 1. Init Ray
# ---------------------------------------------------------------------------
ray.init(ignore_reinit_error=True)

# ---------------------------------------------------------------------------
# 2. Build reporters based on env vars
# ---------------------------------------------------------------------------
os.makedirs("./logs", exist_ok=True)
reporters = [FileReporter(filepath="./logs/pendulum_metrics.jsonl")]

influxdb_url = os.environ.get("INFLUXDB_URL")
if influxdb_url:
    reporters.append(
        InfluxDBReporter(
            url=influxdb_url,
            org=os.environ.get("INFLUXDB_ORG", "rl"),
            bucket=os.environ.get("INFLUXDB_BUCKET", "metrics"),
            token=os.environ.get("INFLUXDB_TOKEN", ""),
            measurement="sac_pendulum",
        )
    )
    print(f"InfluxDB reporter enabled: {influxdb_url}")

# ---------------------------------------------------------------------------
# 3. Configure SAC
# ---------------------------------------------------------------------------
config = (
    CustomSACConfig()
    .environment("Pendulum-v1")
    .training(
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
        train_batch_size=256,
        replay_buffer_config={
            "type": "EpisodeReplayBuffer",
            "capacity": 100_000,
        },
        target_entropy="auto",
        tau=0.005,
    )
    .env_runners(num_env_runners=1, rollout_fragment_length=1)
    .evaluation(evaluation_interval=20)
    .checkpointing(freq=20, local_dir="./checkpoints/pendulum")
    .storage(upload_async=True, best_upload_freq=10)
    .callbacks(FrameworkCallback.with_reporters(reporters))
)

# ---------------------------------------------------------------------------
# 4. Train — no manual checkpoint logic needed
# ---------------------------------------------------------------------------
algo = config.build()

for iteration in range(200):
    result = algo.train()
    mean_reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
    print(f"[iter {iteration:03d}] mean_reward={mean_reward:.2f}")
    # Periodic checkpointing + eval-driven best-model saving happen automatically
    # inside FrameworkCallback.

algo.stop()
ray.shutdown()
print("Done. Metrics written to ./logs/pendulum_metrics.jsonl")
