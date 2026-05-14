"""
Example 05: Custom Model Composition
====================================
Demonstrates how to use the new FrameworkCatalog component composition system.

This example shows:
- Registering custom encoder, actor_head, critic_head
- Mixing custom components with default ones
- Using framework_models() API in config

Run:
    python examples/05_custom_model_composition.py
"""

import gymnasium as gym
import numpy as np
import ray
import torch.nn as nn
from ray.rllib.core import DEFAULT_MODULE_ID

from rlframework.algorithms.ppo import CustomPPOConfig
from rlframework.models.catalog import ComponentRegistry

# =============================================================================
# 1. Define custom model components
# =============================================================================


# Register a custom encoder
@ComponentRegistry.register_encoder("my_mlp_encoder")
def build_my_encoder(observation_space, action_space, model_config, framework):
    """Custom MLP encoder that extracts features from observations.

    Returns an ActorCriticEncoder that:
    - Takes dict input ({"obs": tensor})
    - Returns dict output {"encoder_out": {"actor": tensor, "critic": tensor}}
    """
    from ray.rllib.core.models.base import ACTOR, CRITIC, ENCODER_OUT, ActorCriticEncoder
    from ray.rllib.core.models.torch.base import TorchModel

    hidden_dims = model_config.get("encoder_fcnet_hiddens", [256, 256])
    activation = model_config.get("encoder_fcnet_activation", "relu")

    if activation == "relu":
        act_fn = nn.ReLU
    elif activation == "tanh":
        act_fn = nn.Tanh
    else:
        act_fn = nn.ReLU

    if isinstance(observation_space, gym.spaces.Box):
        obs_dim = int(np.prod(observation_space.shape))
    else:
        obs_dim = observation_space.n

    _framework = framework  # captured by closure

    class CustomActorCriticEncoder(TorchModel, ActorCriticEncoder):
        framework = _framework

        def __init__(self, config):
            TorchModel.__init__(self, config)
            self.config = config

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
            return {
                ENCODER_OUT: {
                    ACTOR: encoded,
                    CRITIC: encoded,
                }
            }

        def get_num_parameters(self):
            return sum(p.numel() for p in self.parameters()), 0

        def _set_to_dummy_weights(self, value_sequence=(-0.02, -0.01, 0.01, 0.02)):
            for i, p in enumerate(self.parameters()):
                p.data.fill_(value_sequence[i % len(value_sequence)])

    class Config:
        shared = True
        inference_only = False

    return CustomActorCriticEncoder(Config())


# Register a custom actor head (policy head)
@ComponentRegistry.register_actor_head("my_actor_head")
def build_my_actor_head(input_dims, action_space, model_config, framework):
    """Custom actor head that computes policy logits."""
    hidden_dims = model_config.get("head_fcnet_hiddens", [256, 256])
    activation = model_config.get("head_fcnet_activation", "relu")

    if activation == "relu":
        act_fn = nn.ReLU
    elif activation == "tanh":
        act_fn = nn.Tanh
    else:
        act_fn = nn.ReLU

    if isinstance(action_space, gym.spaces.Discrete):
        action_dim = action_space.n
    elif isinstance(action_space, gym.spaces.Box):
        action_dim = int(np.prod(action_space.shape))
    else:
        action_dim = action_space.n

    class CustomActorHead(nn.Module):
        def __init__(self, input_dim, hidden_dims, action_dim, act_fn):
            super().__init__()
            layers = []
            in_dim = int(input_dim) if isinstance(input_dim, tuple) else input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(act_fn())
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, action_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    input_dim = int(input_dims[0]) if isinstance(input_dims, tuple) else input_dims
    return CustomActorHead(input_dim, hidden_dims, action_dim, act_fn)


# Register a custom critic head (value head)
@ComponentRegistry.register_critic_head("my_critic_head")
def build_my_critic_head(input_dims, model_config, framework):
    """Custom critic head that computes value function."""
    hidden_dims = model_config.get("head_fcnet_hiddens", [256, 256])
    activation = model_config.get("head_fcnet_activation", "relu")
    critic_config = model_config.get("custom_model_config", {}).get("critic_head", {})
    value_scale = critic_config.get("value_scale", 1.0)

    if activation == "relu":
        act_fn = nn.ReLU
    elif activation == "tanh":
        act_fn = nn.Tanh
    else:
        act_fn = nn.ReLU

    class CustomCriticHead(nn.Module):
        def __init__(self, input_dim, hidden_dims, act_fn, value_scale):
            super().__init__()
            self.value_scale = value_scale
            layers = []
            in_dim = int(input_dim) if isinstance(input_dim, tuple) else input_dim
            for h_dim in hidden_dims:
                layers.append(nn.Linear(in_dim, h_dim))
                layers.append(act_fn())
                in_dim = h_dim
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x) * self.value_scale

    input_dim = int(input_dims[0]) if isinstance(input_dims, tuple) else input_dims
    return CustomCriticHead(input_dim, hidden_dims, act_fn, value_scale)


# =============================================================================
# 2. Train with custom model composition
# =============================================================================


def main():
    ray.init(ignore_reinit_error=True)

    # List registered components
    print("Registered encoders:", ComponentRegistry.list_encoders())
    print("Registered actor heads:", ComponentRegistry.list_actor_heads())

    # Configure PPO with custom model components.
    # Custom component options can be passed through the new RLModule model_config.
    config = (
        CustomPPOConfig()
        .framework_run("custom_model_composition", root_dir="./runs")
        .environment("CartPole-v1")
        .framework("torch")
        .training(
            lr=3e-4,
            train_batch_size=2000,
            num_epochs=5,
        )
        .rl_module(
            model_config={
                "custom_model_config": {
                    "critic_head": {
                        # Test parameter consumed by build_my_critic_head().
                        "value_scale": 0.5,
                    }
                }
            }
        )
        .env_runners(num_env_runners=2)
        # Use custom model components!
        .framework_models(
            encoder="my_mlp_encoder",
            actor_head="my_actor_head",
            critic_head="my_critic_head",
        )
        .metrics(reporters=["file"])
    )

    print("\nBuilding algorithm with custom model components...")
    algo = config.build()

    def get_local_learner_module(algo):
        """Return the learner module when the learner runs in this process."""
        learner_group = getattr(algo, "learner_group", None)
        if learner_group is None or not getattr(learner_group, "is_local", False):
            return None

        # RLlib exposes Algorithm.get_module() for the env-runner module, which can
        # be inference-only and omit the value-function head. The local learner
        # module is the one that contains encoder + pi + vf for PPO training.
        module = learner_group._learner.module
        return module.get(DEFAULT_MODULE_ID) if hasattr(module, "get") else module

    learner_module = get_local_learner_module(algo)
    module = learner_module or algo.get_module()
    print("\n✓ Module built successfully!")
    print(f"  - Inspecting {'learner' if learner_module is not None else 'env-runner'} module")
    print(f"  - Encoder: {type(module.encoder).__name__}")
    print(f"  - Pi head: {type(module.pi).__name__}")
    if hasattr(module, "vf"):
        print(f"  - Vf head: {type(module.vf).__name__}")

    def print_model_structure(model, name, indent=2):
        print(f"\n{' ' * indent}[{name}] {type(model).__name__}")
        if hasattr(model, "net"):
            print(f"{' ' * indent}  net: {model.net}")
        if hasattr(model, "value_scale"):
            print(f"{' ' * indent}  value_scale: {model.value_scale}")
        if hasattr(model, "parameters"):
            total = sum(p.numel() for p in model.parameters())
            print(f"{' ' * indent}  params: {total:,}")

    def verify_custom_model(model, name, expected_name):
        actual_name = type(model).__name__
        assert actual_name == expected_name, f"{name} should be {expected_name}, got {actual_name}"
        print(f"  ✓ {name} verified: {actual_name}")

    if hasattr(module, "encoder"):
        print_model_structure(module.encoder, "encoder")
        verify_custom_model(module.encoder, "encoder", "CustomActorCriticEncoder")
    if hasattr(module, "pi"):
        print_model_structure(module.pi, "pi")
        verify_custom_model(module.pi, "pi", "CustomActorHead")
    if hasattr(module, "vf"):
        print_model_structure(module.vf, "vf")
        verify_custom_model(module.vf, "vf", "CustomCriticHead")
        assert module.vf.value_scale == 0.5, (
            f"vf value_scale should be 0.5, got {module.vf.value_scale}"
        )
        print(f"  ✓ vf custom model_config verified: value_scale={module.vf.value_scale}")
    else:
        print(f"\n{' ' * 2}- Vf head is omitted on RLlib's inference-only env-runner module.")

    print("\nTraining for 10 iterations...")
    for i in range(10):
        result = algo.train()
        mean_reward = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
        print(f"[iter {i:02d}] mean_reward={mean_reward:.2f}")

    algo.stop()
    ray.shutdown()
    print("\nDone!")


if __name__ == "__main__":
    main()
