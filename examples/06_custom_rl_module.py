"""Example: Using Custom RLModule for fully custom architectures.

This example shows how to use CustomPPORLModule when you need complete
control over the forward pass and training logic, beyond what Catalog
component substitution can provide.

Run:
    python examples/06_custom_rl_module.py
"""

import numpy as np
import ray
import torch
import torch.nn as nn
from ray.rllib.core import DEFAULT_MODULE_ID
from ray.rllib.core.columns import Columns
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
    def _build_mlp(
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        activation: str = "relu",
        output_activation=None,
    ) -> nn.Sequential:
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

    def _encode(self, batch):
        obs = batch[Columns.OBS]
        if not isinstance(obs, torch.Tensor):
            obs = torch.from_numpy(obs)
        obs = obs.float()
        return self.encoder(obs.reshape(obs.shape[0], -1))

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
        h = self._encode(batch)
        action_logits = self.pi_head(h)
        return {
            Columns.ACTION_DIST_INPUTS: action_logits,
        }

    @override(CustomPPORLModule)
    def _forward_train(self, batch, **kwargs):
        h = self._encode(batch)
        return {
            Columns.ACTION_DIST_INPUTS: self.pi_head(h),
            Columns.EMBEDDINGS: h,
        }

    @override(CustomPPORLModule)
    def compute_values(self, batch, embeddings=None):
        h = embeddings if embeddings is not None else self._encode(batch)
        return self.vf_head(h).squeeze(-1)


# =============================================================================
# Use the custom RLModule in config
# =============================================================================


def main():
    ray.init(ignore_reinit_error=True)

    # Configure PPO to use the custom RLModule class
    # No catalog_class needed — the RLModule builds everything itself
    config = (
        CustomPPOConfig()
        .framework_run("custom_rl_module", root_dir="./runs")
        .environment("CartPole-v1")
        .framework("torch")
        .training(
            lr=3e-4,
            train_batch_size=2000,
            num_epochs=5,
        )
        .env_runners(num_env_runners=1)
        .rl_module(
            model_config={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
            },
            rl_module_spec=RLModuleSpec(
                module_class=MinimalPPOModule,
                # catalog_class is omitted — MinimalPPOModule is fully self-contained
            ),
        )
        .metrics(reporters=["file"])
    )

    print("Building algorithm with custom RLModule...")
    algo = config.build()

    def get_local_learner_module(algo):
        """Return the learner module when the learner runs in this process."""
        learner_group = getattr(algo, "learner_group", None)
        if learner_group is None or not getattr(learner_group, "is_local", False):
            return None

        module = learner_group._learner.module
        return module.get(DEFAULT_MODULE_ID) if hasattr(module, "get") else module

    learner_module = get_local_learner_module(algo)
    module = learner_module or algo.get_module()
    print(f"\nModule type: {type(module).__name__}")
    print(f"  - Inspecting {'learner' if learner_module is not None else 'env-runner'} module")
    print(f"  - Encoder: {type(module.encoder).__name__}")
    print(f"  - Pi head: {type(module.pi_head).__name__}")
    print(f"  - VF head: {type(module.vf_head).__name__}")

    # Check forward pass
    obs = torch.randn(4, 4)
    batch = {Columns.OBS: obs}
    fwd_out = (
        module.forward_exploration(batch)
        if getattr(module, "inference_only", False)
        else module.forward_train(batch)
    )
    values = module.compute_values(batch, embeddings=fwd_out.get(Columns.EMBEDDINGS))
    print(f"\nForward pass output keys: {list(fwd_out.keys())}")
    print(f"  action_logits shape: {fwd_out[Columns.ACTION_DIST_INPUTS].shape}")
    print(f"  values shape:        {values.shape}")

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
