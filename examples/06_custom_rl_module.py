"""Example: Using Custom RLModule for fully custom architectures.

This example shows how to use CustomPPORLModule when you need complete
control over the forward pass and training logic, beyond what Catalog
component substitution can provide.

Run:
    python rlframework/examples/06_custom_rl_module.py
"""

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ray
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.annotations import override
from rlframework.algorithms.ppo import CustomPPOConfig
from rlframework.models.rl_module import CustomPPORLModule


# =============================================================================
# Define a fully custom PPO RLModule
# =============================================================================


class MinimalPPOModule(CustomPPORLModule):
    """Minimal custom PPO module that reimplements the standard MLP actor-critic.

    This exists as a REFERENCE IMPLEMENTATION showing the full RLModule contract.
    For most use cases, prefer PPOCompositeCatalog + ComponentRegistry
    (see examples/05_custom_model_composition.py).

    This implementation shows:
    - How to set up encoder, pi_head, vf_head as sub-components
    - How to implement _forward() for inference/exploration
    - How to implement _forward_train() to return loss inputs
    """

    @staticmethod
    def _get_hidden_activation(name: str):
        """Convert activation string to nn.Module."""
        activations = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "elu": nn.ELU,
            "leaky_relu": nn.LeakyReLU,
        }
        return activations.get(name.lower(), nn.ReLU)

    @staticmethod
    def _build_mlp(input_dim: int, hidden_dims: list[int], output_dim: int,
                   activation: str = "relu", output_activation=None) -> nn.Sequential:
        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(MinimalPPOModule._get_hidden_activation(activation)())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        if output_activation:
            layers.append(output_activation)
        return nn.Sequential(*layers)

    @override(CustomPPORLModule)
    def setup(self):
        model_config = self._model_config or {}
        hiddens = model_config.get("fcnet_hiddens", [256, 256])
        activation = model_config.get("fcnet_activation", "relu")
        obs_dim = int(np.prod(self.observation_space.shape))
        action_dim = self.action_space.n

        self.encoder = self._build_mlp(obs_dim, hiddens, 128, activation)
        self.pi_head = self._build_mlp(128, [], action_dim)
        self.vf_head = self._build_mlp(128, [], 1)

    @override(CustomPPORLModule)
    def _forward(self, batch, **kwargs):
        obs = batch["obs"]
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs).float()
        h = self.encoder(obs)
        action_logits = self.pi_head(h)
        return {
            "action_dist_inputs": action_logits,
            "vf_preds": self.vf_head(h).squeeze(-1),
        }

    @override(CustomPPORLModule)
    def _forward_train(self, batch, **kwargs):
        return self._forward(batch, **kwargs)



# =============================================================================
# Use the custom RLModule in config
# =============================================================================


def main():
    ray.init(ignore_reinit_error=True)

    # Configure PPO to use the custom RLModule class
    # No catalog_class needed — the RLModule builds everything itself
    config = (
        CustomPPOConfig()
        .environment("CartPole-v1")
        .framework("torch")
        .training(
            lr=3e-4,
            train_batch_size=2000,
            num_epochs=5,
        )
        .env_runners(num_env_runners=1)
        .rl_module(
            rl_module_spec=RLModuleSpec(
                rl_module_class=MinimalPPOModule,
                # catalog_class is omitted — MinimalPPOModule is fully self-contained
                model_config_dict={
                    "fcnet_hiddens": [256, 256],
                    "fcnet_activation": "relu",
                },
            )
        )
    )

    print("Building algorithm with custom RLModule...")
    algo = config.build()

    module = algo.get_module()
    print(f"\nModule type: {type(module).__name__}")
    print(f"  - Encoder: {type(module.encoder).__name__}")
    print(f"  - Pi head: {type(module.pi_head).__name__}")
    print(f"  - VF head: {type(module.vf_head).__name__}")

    # Check forward pass
    obs = torch.randn(4, 4)
    batch = {"obs": obs}
    fwd_out = module(batch)
    print(f"\nForward pass output keys: {list(fwd_out.keys())}")
    print(f"  action_logits shape: {fwd_out['action_dist_inputs'].shape}")
    print(f"  vf_preds shape:     {fwd_out['vf_preds'].shape}")

    print("\nTraining for 5 iterations...")
    for i in range(5):
        result = algo.train()
        mean_reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
        print(f"[iter {i:02d}] mean_reward={mean_reward:.2f}")

    algo.stop()
    ray.shutdown()
    print("\nDone!")


if __name__ == "__main__":
    main()
