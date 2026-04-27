"""Custom RLModule implementations.

This module provides placeholder implementations for fully custom RLModules,
to be used when the Catalog-based component composition is insufficient.

When to use this vs Catalog:
- Catalog (build_*)  → Replace individual parts (encoder, head, distribution)
- Custom RLModule    → Redefine the entire forward pass and training logic

Usage::

    from rlframework.models.rl_module import CustomPPORLModule

    # Recommended: fully self-contained, no catalog needed
    config = CustomPPOConfig()
    config.rl_module(
        model_config={...},
        rl_module_spec=RLModuleSpec(
            module_class=MyFullyCustomPPOModule,
        ),
    )

    # Advanced: inject catalog for hybrid use (keep some default components)
    config.rl_module(
        model_config={"_framework_custom_config": {...}},
        rl_module_spec=RLModuleSpec(
            module_class=MyHybridPPOModule,
            catalog_class=PPOCompositeCatalog,
        ),
    )

Available RLModules:
- CustomPPORLModule: Fully custom PPO module (encoder + pi + vf + action_dist)
- CustomSACRLModule: Fully custom SAC module (encoder + policy + q-functions)
- CustomDQNRLModule: Fully custom DQN module (encoder + q-head)
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override

# ----------------------------------------------------------------------
# Base class for custom RLModules
# ----------------------------------------------------------------------


class CustomTorchRLModule(TorchRLModule):
    """Base class for custom Torch-based RLModules.

    Provides a clean starting point for implementing fully custom modules.
    Subclasses must implement:
    - _forward()
    - _forward_inference()
    - _forward_exploration()
    - _forward_train()

    Optional to override:
    - get_non_inference_attributes() [for inference-only optimization]
    """

    framework: str = "torch"

    @override(RLModule)
    def get_initial_state(self) -> dict[str, Any]:
        """Return initial hidden states for RNN-based modules.

        Override this if your module uses recurrent layers.
        Return an empty dict for stateless modules.
        """
        return {}

    @override(RLModule)
    def _forward(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Generic forward pass used in all phases.

        Override for custom forward behavior that should be shared
        across inference, exploration, and training.
        """
        raise NotImplementedError

    @override(RLModule)
    def _forward_inference(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Forward pass during inference (evaluation, greedy acting).

        Override when inference behavior differs from exploration/training.
        By default, falls back to _forward().
        """
        return self._forward(batch, **kwargs)

    @override(RLModule)
    def _forward_exploration(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Forward pass during exploration (action sampling).

        Override to add exploration noise or stochasticity.
        By default, falls back to _forward().
        """
        return self._forward(batch, **kwargs)

    @override(RLModule)
    def _forward_train(self, batch: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Forward pass during training (computing loss inputs).

        Override to compute extra outputs needed for loss computation.
        By default, falls back to _forward().
        """
        return self._forward(batch, **kwargs)


# ----------------------------------------------------------------------
# PPO Custom RLModule
# ----------------------------------------------------------------------


class CustomPPORLModule(CustomTorchRLModule, ValueFunctionAPI):
    """Fully custom PPO RLModule.

    Use this when you need to:
    - Completely redefine the actor-critic architecture
    - Implement custom forward passes (e.g., auxiliary losses)
    - Add shared representations with unusual branching
    - Use non-standard action distributions

    If you only need to swap the encoder or heads, prefer using
    PPOCompositeCatalog with ComponentRegistry instead.

    Example - Minimal Implementation::

        import torch.nn as nn
        from ray.rllib.core.models.config import ModelConfig

        class MyPPOModule(CustomPPORLModule):
            def setup(self):
                obs_dim = self.observation_space.shape[0]
                action_dim = self.action_space.n

                self.encoder = nn.Sequential(
                    nn.Linear(obs_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                )
                self.actor = nn.Linear(128, action_dim)
                self.critic = nn.Linear(128, 1)

            def _forward(self, batch, **kwargs):
                obs = batch["obs"]
                h = self.encoder(obs)
                return {
                    "actions": torch.argmax(self.actor(h), dim=-1),
                    "action_dist": None,  # set in _forward_train for training
                    "vf_preds": self.critic(h).squeeze(-1),
                }

            def _forward_train(self, batch, **kwargs):
                obs = batch["obs"]
                h = self.encoder(obs)
                action_logits = self.actor(h)
                return {
                    "action_dist_inputs": action_logits,
                    "vf_preds": self.critic(h).squeeze(-1),
                }
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        inference_only: bool = False,
        learner_only: bool = False,
        model_config: dict | None = None,
        catalog_class: type[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        # Store spaces before calling super().__init__ so setup() can use them
        self._observation_space = observation_space
        self._action_space = action_space
        self._model_config = model_config or {}

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            catalog_class=catalog_class,
            **kwargs,
        )

    @override(RLModule)
    def setup(self) -> None:
        """Set up the subcomponents of this module.

        Called automatically during __init__ via super().__init__().

        Override this method to:
        - Create encoder, pi_head, vf_head, action_dist as self.* attributes
        - These will be automatically serialized by get_state()/set_state()
        """
        raise NotImplementedError(
            "CustomPPORLModule.setup() must be implemented. "
            "Create encoder, pi_head, vf_head, and action_dist as attributes."
        )

    def get_non_inference_attributes(self) -> list[str]:
        """Return components not needed during inference-only mode.

        By default, the value function head is not needed for inference.
        Override to customize.
        """
        return ["vf_head"]

    @override(ValueFunctionAPI)
    def compute_values(self, batch: dict[str, Any], embeddings: Any | None = None) -> Any:
        """Compute value estimates for PPO losses.

        PPO's new Learner API requires modules to implement ValueFunctionAPI.
        """
        raise NotImplementedError("CustomPPORLModule.compute_values() must be implemented.")


# ----------------------------------------------------------------------
# SAC Custom RLModule
# ----------------------------------------------------------------------


class CustomSACRLModule(CustomTorchRLModule):
    """Fully custom SAC RLModule for continuous control.

    Use this when you need complete control over:
    - The policy network architecture
    - Q-function architecture (e.g., ensemble Q-networks)
    - How target networks are managed

    If you only need to swap individual components, prefer using
    SACCompositeCatalog instead.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        inference_only: bool = False,
        learner_only: bool = False,
        model_config: dict | None = None,
        catalog_class: type[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._observation_space = observation_space
        self._action_space = action_space
        self._model_config = model_config or {}

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            catalog_class=catalog_class,
            **kwargs,
        )

    @override(RLModule)
    def setup(self) -> None:
        """Set up the subcomponents: encoder, policy, Q-networks.

        Override to create:
        - self.encoder: Feature extraction network
        - self.policy: Action distribution head (e.g., SquashedGaussian)
        - self.q_net: Q-function network(s)
        - self.q_target: Target Q-network(s)
        """
        raise NotImplementedError(
            "CustomSACRLModule.setup() must be implemented. "
            "Create encoder, policy, q_net, and q_target as attributes."
        )

    def get_non_inference_attributes(self) -> list[str]:
        """Components not needed during inference.

        For SAC, Q-networks are not needed for inference.
        """
        return ["q_net", "q_target"]


# ----------------------------------------------------------------------
# DQN Custom RLModule
# ----------------------------------------------------------------------


class CustomDQNRLModule(CustomTorchRLModule):
    """Fully custom DQN RLModule.

    Use this for:
    - Rainbow-style algorithms (dueling, distributional, noisy)
    - Custom Q-network architectures
    - Non-standard target network update schedules

    For simple Q-head swaps, use DQNCompositeCatalog instead.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        inference_only: bool = False,
        learner_only: bool = False,
        model_config: dict | None = None,
        catalog_class: type[Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._observation_space = observation_space
        self._action_space = action_space
        self._model_config = model_config or {}

        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            inference_only=inference_only,
            learner_only=learner_only,
            model_config=model_config,
            catalog_class=catalog_class,
            **kwargs,
        )

    @override(RLModule)
    def setup(self) -> None:
        """Set up the subcomponents: encoder, Q-head.

        Override to create:
        - self.encoder: Feature extraction network
        - self.q_head: Q-value head
        """
        raise NotImplementedError(
            "CustomDQNRLModule.setup() must be implemented. "
            "Create encoder and q_head as attributes."
        )

    def get_non_inference_attributes(self) -> list[str]:
        """No special inference-only optimization for basic DQN."""
        return []
