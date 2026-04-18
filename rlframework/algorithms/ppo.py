"""CustomPPO - extends RLlib's PPO with framework hooks.

Inherit from this class (not directly from RLlib's PPO) when you need to
customise PPO behaviour inside rlframework::

    from rlframework.algorithms.ppo import CustomPPO

    class MyPPO(CustomPPO):

        def build_extra_model_config(self):
            return {"custom_model": "my_model", "custom_model_config": {}}

        def on_after_training_step(self, result):
            result["my_custom_metric"] = 42.0
            return result
"""

from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.annotations import override

from rlframework.algorithms.base import FrameworkAlgorithmMixin
from rlframework.config.framework_config import FrameworkConfigMixin
from rlframework.models.catalog import (
    PPOCompositeCatalog,
)
from rlframework.utils.replay_buffers import ReservoirReplayBuffer


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
            .checkpointing(freq=10, upload_async=True)
            .training(lr=1e-4)
        )
        algo = config.build()
    """

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or CustomPPO)
        # Initialize the framework mixin
        self._init_framework_mixin()

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

        if encoder and encoder != "default":
            # 支持 string name 或直接传入 builder 函数
            custom_config["custom_encoder"] = encoder

        if actor_head and actor_head != "default":
            custom_config["custom_actor_head"] = actor_head

        if critic_head and critic_head != "default":
            custom_config["custom_critic_head"] = critic_head

        # If we have custom components, configure RLModule to use CompositeCatalog
        if custom_config:
            # Update model config
            self.model.update({
                "_framework_custom_config": custom_config,
            })

            # Set catalog class to PPOCompositeCatalog
            self.rl_module(
                rl_module_spec=RLModuleSpec(
                    catalog_class=PPOCompositeCatalog,
                    model_config=self.model,
                )
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
    def setup(self, config: CustomPPOConfig):
        # Merge any extra model config from user subclass
        extra = self.build_extra_model_config()
        if extra:
            config.model.update(extra)
        super().setup(config)

    @override(PPO)
    def training_step(self) -> None:
        self.on_before_training_step()
        super().training_step()
        # 新 API stack: 从 metrics 中提取当前编译结果, 传给 hook, 再写回 metrics
        result = self.metrics.peek()
        result = self.on_after_training_step(result)
        if result:
            for key, value in result.items():
                # 已有 key：push 值走原有 reduce/window 聚合逻辑
                # 新 key（custom/* 等）：默认 EMA=0.01 会严重平滑阶梯值，改用 window=1
                if isinstance(value, (int, float)):
                    if key in self.metrics:
                        self.metrics.log_value(key, value)
                    else:
                        self.metrics.log_value(key, value, window=1)




class CustomReplayBufferPPO(CustomPPO):
    """PPO with rlframework extension hooks.

    All :py:class:`FrameworkAlgorithmMixin` hook methods are available.
    The mixin methods are called *around* the standard RLlib PPO
    ``training_step``; untouched hooks are no-ops.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_replay_buffer = ReservoirReplayBuffer(capacity=1000)