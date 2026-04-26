"""
Example 12: Grafana Metrics (InfluxDB + Prometheus reporters)
==============================================================
Demonstrates:
- Configuring InfluxDBReporter for Grafana dashboards
- Configuring PrometheusReporter for push-gateway scraping
- Using FileReporter as a local fallback
- Training/eval phase tagging via FrameworkCallback

Prerequisites:
    # docker-compose.yml snippet:
    services:
      influxdb:
        image: influxdb:2.7
        environment:
          DOCKER_INFLUXDB_INIT_MODE: setup
          DOCKER_INFLUXDB_INIT_USERNAME: admin
          DOCKER_INFLUXDB_INIT_PASSWORD: adminpass
          DOCKER_INFLUXDB_INIT_ORG: rl
          DOCKER_INFLUXDB_INIT_BUCKET: metrics
          DOCKER_INFLUXDB_INIT_ADMIN_TOKEN: my-super-secret-token
        ports: ["8086:8086"]

      pushgateway:
        image: prom/pushgateway
        ports: ["9091:9091"]

      grafana:
        image: grafana/grafana-oss
        ports: ["3000:3000"]

    # Then add InfluxDB and Prometheus as data sources in Grafana UI.

Run:
    docker-compose up -d
    python rlframework/examples/12_grafana_metrics.py
"""

import os

import ray

from rlframework.algorithms.ppo import CustomPPOConfig
from rlframework.callbacks import FrameworkCallback
from rlframework.observability.reporters import (
    FileReporter,
    InfluxDBReporter,
    PrometheusReporter,
)

# =========================================================================
# 1. Reporter configuration — override via env vars in CI / production
# =========================================================================
INFLUXDB_URL = os.environ.get("INFLUXDB_URL", "http://localhost:8086")
INFLUXDB_ORG = os.environ.get("INFLUXDB_ORG", "rl")
INFLUXDB_BUCKET = os.environ.get("INFLUXDB_BUCKET", "metrics")
INFLUXDB_TOKEN = os.environ.get("INFLUXDB_TOKEN", "my-super-secret-token")

PROMETHEUS_GW = os.environ.get("PROMETHEUS_GW", "localhost:9091")

EXPERIMENT = "ppo_cartpole_grafana"

# =========================================================================
# 2. Build reporter list
# =========================================================================
os.makedirs("./logs", exist_ok=True)

reporters = [
    # Local file — always available, no external deps
    FileReporter(filepath=f"./logs/{EXPERIMENT}.jsonl"),
    # InfluxDB — powers Grafana time-series dashboards
    InfluxDBReporter(
        url=INFLUXDB_URL,
        org=INFLUXDB_ORG,
        bucket=INFLUXDB_BUCKET,
        token=INFLUXDB_TOKEN,
        measurement="rl_training",
    ),
    # Prometheus Push Gateway — powers Grafana Prometheus panels
    PrometheusReporter(
        gateway=PROMETHEUS_GW,
        job="rl_training",
        grouping_key={"experiment": EXPERIMENT},
    ),
]

# =========================================================================
# 3. Configure PPO with evaluation enabled
# =========================================================================
ray.init(ignore_reinit_error=True)

config = (
    CustomPPOConfig()
    .environment("CartPole-v1")
    .training(lr=3e-4, train_batch_size=4000, num_epochs=10, minibatch_size=128)
    .env_runners(num_env_runners=2)
    .evaluation(
        evaluation_interval=5,
        evaluation_num_env_runners=1,
        evaluation_duration=10,
    )
    .callbacks(FrameworkCallback.with_reporters(reporters))
)

# =========================================================================
# 4. Train — metrics are pushed to InfluxDB, Prometheus, and local file
# =========================================================================
algo = config.build()

for iteration in range(30):
    result = algo.train()
    mean_reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
    phase_info = ""
    eval_reward = result.get("evaluation", {}).get("env_runners", {}).get("episode_return_mean")
    if eval_reward is not None:
        phase_info = f"  eval_reward={eval_reward:.2f}"
    print(f"[iter {iteration:03d}] train_reward={mean_reward:.2f}{phase_info}")

algo.stop()

# Close reporters to flush buffers
for r in reporters:
    r.close()

ray.shutdown()
print("\nDone. Open Grafana at http://localhost:3000 to view dashboards.")
print(f"Local log: ./logs/{EXPERIMENT}.jsonl")
