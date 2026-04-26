from rlframework.models.catalog import (
    ComponentRegistry,
    DQNCompositeCatalog,
    PPOCompositeCatalog,
    SACCompositeCatalog,
    register_action_dist,
    register_actor_head,
    register_critic_head,
    register_encoder,
    register_q_head,
)
from rlframework.models.rl_module import (
    CustomDQNRLModule,
    CustomPPORLModule,
    CustomSACRLModule,
    CustomTorchRLModule,
)

__all__ = [
    # Catalog
    "ComponentRegistry",
    "CustomDQNRLModule",
    "CustomPPORLModule",
    "CustomSACRLModule",
    # RLModule
    "CustomTorchRLModule",
    "DQNCompositeCatalog",
    "PPOCompositeCatalog",
    "SACCompositeCatalog",
    "register_action_dist",
    "register_actor_head",
    "register_critic_head",
    "register_encoder",
    "register_q_head",
]
