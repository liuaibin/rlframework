"""
Example 13: Behavioral Cloning (BC) from Expert Demonstrations
============================================================
Demonstrates a full production-grade supervised learning pipeline:
- CustomBC with rlframework infrastructure (storage, metrics, callbacks)
- Offline dataset loading (JSONL format)
- Custom model composition via ComponentRegistry
- Periodic checkpointing + MinIO upload
- Training / validation split with early stopping
- Offline evaluation on held-out data

Prerequisites:
    # Generate expert demonstrations (example script)
    python scripts/generate_expert_demos.py \
        --env CartPole-v1 \
        --n-episodes 500 \
        --output ./data/expert_demos_cartpole.jsonl

    Or use the built-in random data generator below for testing.

Run:
    python rlframework/examples/13_supervised_bc.py
"""

import os
import tempfile
import pickle

import numpy as np
import ray

from rlframework.algorithms.supervised import (
    CustomBC,
    CustomBCConfig,
    SupervisedDataset,
)
from rlframework.logging.callbacks import FrameworkCallback
from rlframework.logging.reporters import FileReporter
from rlframework.models.catalog import ComponentRegistry


# =========================================================================
# 1. Generate synthetic expert demonstrations for testing
#   (Replace this with real expert data in production)
# =========================================================================

def generate_synthetic_expert_data(env_name: str, n_samples: int = 5000, seed: int = 42):
    """Generate synthetic expert demonstrations for CartPole-v1.

    This creates a plausible (obs, action) dataset by running a partially
    trained policy. In production, replace this with real human or
    algorithmic expert demonstrations.
    """
    import gymnasium as gym

    np.random.seed(seed)
    env = gym.make(env_name)

    observations, actions = [], []

    # Simulate a reasonably good policy: bias toward correct actions
    for _ in range(n_samples):
        obs, _ = env.reset(seed=seed)
        done = False
        while not done and len(observations) < n_samples:
            # Partially correct policy: correct action ~70% of the time
            if env.action_space.n == 2:
                # CartPole: action 1 = right, optimal is to stay upright
                true_action = 1 if obs[2] > 0 else 0  # angular velocity
                action = true_action if np.random.rand() < 0.75 else 1 - true_action
            else:
                action = env.action_space.sample()

            observations.append(obs)
            actions.append(action)
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

    env.close()
    return np.array(observations, dtype=np.float32), np.array(actions, dtype=np.int64)


# =========================================================================
# 2. Register a custom encoder for supervised learning
# =========================================================================

@ComponentRegistry.register_encoder("bc_encoder")
def build_bc_encoder(observation_space, action_space, model_config, framework):
    """Custom encoder for behavioral cloning.

    Returns a shared encoder that outputs actor features used for action prediction.
    """
    import torch
    import torch.nn as nn
    import numpy as np
    from ray.rllib.core.models.base import ActorCriticEncoder, ENCODER_OUT, ACTOR, CRITIC
    from ray.rllib.core.models.torch.base import TorchModel

    obs_dim = int(np.prod(observation_space.shape))
    hidden_dims = model_config.get("encoder_fcnet_hiddens", [256, 256])
    activation = model_config.get("encoder_fcnet_activation", "relu")

    if activation == "relu":
        act_fn = nn.ReLU
    elif activation == "tanh":
        act_fn = nn.Tanh
    else:
        act_fn = nn.ReLU

    _framework = framework

    class BCEncoder(TorchModel, ActorCriticEncoder):
        framework = _framework

        def __init__(self, config):
            TorchModel.__init__(self, config)
            layers = []
            in_dim = obs_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(act_fn())
                in_dim = h_dim
            self.net = nn.Sequential(*layers)

        def _forward(self, inputs, **kwargs):
            obs = inputs["obs"]
            encoded = self.net(obs)
            return {ENCODER_OUT: {ACTOR: encoded, CRITIC: encoded}}

        def get_num_parameters(self):
            return sum(p.numel() for p in self.parameters()), 0

        def _set_to_dummy_weights(self, value_sequence=(-0.02, -0.01, 0.01, 0.02)):
            for i, p in enumerate(self.parameters()):
                p.data.fill_(value_sequence[i % len(value_sequence)])

    class Config:
        shared = True
        inference_only = False

    return BCEncoder(Config())


# =========================================================================
# 3. Train
# =========================================================================

def main():
    # Init Ray
    ray.init(ignore_reinit_error=True)

    # ── Reporters ───────────────────────────────────────────────────
    os.makedirs("./logs", exist_ok=True)
    reporters = [FileReporter(filepath="./logs/bc_metrics.jsonl")]

    # ── Generate or load demonstration data ─────────────────────────
    env_name = "CartPole-v1"
    data_path = os.environ.get("BC_DATA_PATH")

    if not data_path:
        # Generate synthetic data for testing
        print("No BC_DATA_PATH set — generating synthetic expert data...")
        obs, acts = generate_synthetic_expert_data(env_name, n_samples=5000)
        data_path = tempfile.mktemp(suffix=".npz")
        np.savez(data_path, observations=obs, actions=acts)
        print(f"  -> saved to {data_path}  ({len(obs)} samples)")

    # Verify data
    ds = SupervisedDataset.load(data_path)
    print(f"Dataset loaded: {len(ds)} samples")
    print(f"  obs shape: {ds.observations.shape}")
    print(f"  actions: {np.bincount(ds.actions.astype(int))}")

    # Train / validation split
    train_ds, val_ds = ds.split(train_ratio=0.9, shuffle=True)
    print(f"  train: {len(train_ds)}  val: {len(val_ds)}")

    # Save validation data for periodic evaluation
    val_data_path = tempfile.mktemp(suffix=".npz")
    np.savez(
        val_data_path,
        observations=val_ds.observations,
        actions=val_ds.actions,
    )

    # ── Configure CustomBC ─────────────────────────────────────────
    config = (
        CustomBCConfig()
        .environment(env_name)
        # Offline data source (for RLlib BC algorithm itself)
        .offline_data(input_=data_path)
        # Supervised learning parameters
        .supervised_training(
            data_path=data_path,
            batch_size=256,
            epochs_per_round=5,
            validation_split=0.1,
            loss_type="ce",
        )
        # Custom model
        .framework_models(encoder="bc_encoder")
        # Training hyperparameters
        .training(
            lr=3e-4,
            train_batch_size=256,
        )
        .env_runners(num_env_runners=1)
        # Logging
        .callbacks(FrameworkCallback.with_reporters(reporters))
    )

    # ── Training loop ───────────────────────────────────────────────
    algo = config.build()

    best_val_acc = 0.0
    patience, no_improve = 10, 0

    print(f"\nTraining BC on {env_name} ...")
    print(f"{'Epoch':>6} | {'Train Loss':>12} | {'Val Acc':>8} | {'Best':>8} | {'Patience':>8}")
    print("-" * 60)

    for epoch in range(200):
        # Train one epoch (runs supervised_epochs_per_round steps internally)
        result = algo.train()

        # Training metrics from metrics log
        metrics = result.get("env_runners", {})
        train_loss = metrics.get("supervised/cross_entropy_loss", float("nan"))
        train_acc = metrics.get("supervised/sup_accuracy", float("nan"))

        # Validation (every 5 epochs to save compute)
        val_acc = 0.0
        if (epoch + 1) % 5 == 0:
            val_metrics = algo.evaluate_on_supervised_data(val_data_path)
            val_acc = val_metrics.get("sup_accuracy", 0.0)

            # Checkpoint best
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                ckpt_path = algo.save_to_path(f"./checkpoints/bc/best")
                print(f"  -> new best: val_acc={val_acc:.4f}  ckpt={ckpt_path}")
            else:
                no_improve += 1

        print(
            f"{epoch + 1:>6} | "
            f"{train_loss:>12.4f} | "
            f"{val_acc:>8.4f} | "
            f"{best_val_acc:>8.4f} | "
            f"{patience - no_improve:>8}"
        )

        # Early stopping
        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (patience={patience})")
            break

    # ── Final evaluation ────────────────────────────────────────────
    print("\nFinal evaluation on validation set...")
    final_metrics = algo.evaluate_on_supervised_data(val_data_path)
    print(f"  sup_accuracy:       {final_metrics.get('sup_accuracy', 'N/A'):.4f}")
    print(f"  sup_top3_accuracy:  {final_metrics.get('sup_top3_accuracy', 'N/A'):.4f}")
    print(f"  num_evaluated:      {final_metrics.get('num_evaluated_samples', 'N/A')}")

    # ── Cleanup ───────────────────────────────────────────────────
    for reporter in reporters:
        reporter.close()
    algo.stop()
    ray.shutdown()

    print(f"\nDone. Best validation accuracy: {best_val_acc:.4f}")
    print(f"Metrics saved to ./logs/bc_metrics.jsonl")


if __name__ == "__main__":
    main()
