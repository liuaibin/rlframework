"""CustomPPO - extends RLlib's PPO with framework hooks.

Inherit from this class (not directly from RLlib's PPO) when you need to
customise PPO behaviour inside rlframework::

    from rlframework.algorithms.ppo import CustomPPO

    class MyPPO(CustomPPO):

        def on_after_training_step(self, result):
            result["my_custom_metric"] = 42.0
            return result
"""

from typing import Any

from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.annotations import override

from rlframework.algorithms.base import FrameworkAlgorithmMixin
from rlframework.config.framework_config import FrameworkConfigMixin
from rlframework.models.catalog import (
    ComponentRegistry,
    PPOCompositeCatalog,
)


class CustomPPOConfig(PPOConfig, FrameworkConfigMixin):
    """PPOConfig extended with rlframework-specific settings.

    All standard PPO knobs are inherited; this class adds a ``framework``
    section to hold storage / logging preferences without polluting the main
    algorithm config.

    Usage::

        config = (
            CustomPPOConfig()
            .environment("CartPole-v1")
            .storage(backend="minio", endpoint="minio:9000",
                     access_key="admin", secret_key="admin",
                     bucket="rl-models")
            .metrics(reporters=["influxdb", "file"])
            .framework_checkpointing(freq=10, upload_async=True)
            .training(lr=1e-4)
        )
        algo = config.build()
    """

    def __init__(self, algo_class: type[Any] | None = None) -> None:
        super().__init__(algo_class=algo_class or CustomPPO)
        # Initialize the framework mixin
        self._init_framework_mixin()

    @override(PPOConfig)
    def build(self, *args: Any, **kwargs: Any) -> Any:
        self._apply_framework_runtime_config()
        return super().build(*args, **kwargs)

    def framework_models(
        self,
        encoder: str | None = None,
        actor_head: str | None = None,
        critic_head: str | None = None,
    ) -> "CustomPPOConfig":
        """Configure custom model components for PPO.

        This allows you to mix and match custom components with default ones.
        For example, use a custom encoder with the default actor/critic heads.

        Args:
            encoder: Name of a registered custom encoder, or None/"default" for default.
            actor_head: Name of a registered custom actor head, or None/"default" for default.
            critic_head: Name of a registered custom critic head, or None/"default" for default.

        Usage:
            # Use custom encoder, default heads
            config.framework_models(encoder="my_encoder")

            # Use custom encoder + custom actor head, default critic
            config.framework_models(
                encoder="my_encoder",
                actor_head="my_actor_head"
            )

            # Use all defaults
            config.framework_models()  # or .framework_models("default", "default", "default")
        """
        # Build the custom config dict
        custom_config = {}

        def resolve_registered_component(component: Any, getter: Any) -> Any:
            if isinstance(component, str):
                return getter(component) or component
            return component

        if encoder and encoder != "default":
            # Resolve registered names to callables when available so Ray can
            # serialize them to remote workers.
            custom_config["custom_encoder"] = resolve_registered_component(
                encoder,
                ComponentRegistry.get_encoder,
            )

        if actor_head and actor_head != "default":
            custom_config["custom_actor_head"] = resolve_registered_component(
                actor_head,
                ComponentRegistry.get_actor_head,
            )

        if critic_head and critic_head != "default":
            custom_config["custom_critic_head"] = resolve_registered_component(
                critic_head,
                ComponentRegistry.get_critic_head,
            )

        # If we have custom components, configure RLModule to use CompositeCatalog
        if custom_config:
            # Keep using RLlib's new API stack model_config.  This preserves any
            # prior `.rl_module(model_config={...})` values while injecting the
            # component choices consumed by PPOCompositeCatalog.
            model_config = dict(self.model) | dict(self.model_config)
            framework_custom_config = dict(model_config.get("_framework_custom_config", {}))
            framework_custom_config.update(custom_config)
            model_config["_framework_custom_config"] = framework_custom_config

            # Set catalog class to PPOCompositeCatalog
            self.rl_module(
                model_config=model_config,
                rl_module_spec=RLModuleSpec(
                    catalog_class=PPOCompositeCatalog,
                ),
            )

        return self


class CustomPPO(FrameworkAlgorithmMixin, PPO):
    """PPO with rlframework extension hooks.

    All :py:class:`FrameworkAlgorithmMixin` hook methods are available.
    The mixin methods are called *around* the standard RLlib PPO
    ``training_step``; untouched hooks are no-ops.
    """

    @classmethod
    @override(PPO)
    def get_default_config(cls) -> CustomPPOConfig:
        return CustomPPOConfig()

    @override(PPO)
    def setup(self, config: CustomPPOConfig) -> None:
        super().setup(config)

    @override(PPO)
    def training_step(self) -> None:
        self.on_before_training_step()
        super().training_step()
        metrics = self.metrics
        if metrics is None:
            return
        result = metrics.peek()
        result = self.on_after_training_step(result)
        if result:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    if key in metrics:
                        metrics.log_value(key, value)
                    else:
                        metrics.log_value(key, value, window=1)
