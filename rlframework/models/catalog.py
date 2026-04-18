"""FrameworkCatalog - 组合式模型定义系统.

提供模型组件的注册和组合能力，让用户可以混用自定义组件和标准组件。

Usage:

    # 1. 注册自定义组件
    from rlframework.models.catalog import ComponentRegistry

    @ComponentRegistry.register_encoder("my_encoder")
    class MyEncoder:
        def __init__(self, config):
            self.config = config
        def __call__(self, obs):
            # custom encoding logic
            return encoded

    @ComponentRegistry.register_actor_head("my_actor_head")
    class MyActorHead:
        def __init__(self, config):
            self.config = config

    # 2. 在算法中使用
    from rlframework.algorithms.ppo import CustomPPOConfig

    ppo_config = (
        CustomPPOConfig()
        .framework_models(
            encoder="my_encoder",
            actor_head="my_actor_head",
        )
    )
"""

from __future__ import annotations

import gymnasium as gym
from typing import Any, Callable

from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.sac.sac_catalog import SACCatalog
from ray.rllib.algorithms.dqn.dqn_catalog import DQNCatalog


# 存储全局 builder 函数，供跨进程使用
_registered_builders: dict = {}

# 类型定义
EncoderBuilder = Callable[..., Any]
HeadBuilder = Callable[..., Any]


class ComponentRegistry:
    """组件注册中心.

    提供装饰器方式注册自定义模型组件:
    - encoder: 观察空间编码器
    - actor_head: 策略头 (用于 PPO/SAC)
    - critic_head: 价值头 (用于 PPO/SAC/DQN)
    - q_head: Q 网络头 (用于 SAC/DQN)
    """

    _encoders: dict[str, EncoderBuilder] = {}
    _actor_heads: dict[str, HeadBuilder] = {}
    _critic_heads: dict[str, HeadBuilder] = {}
    _q_heads: dict[str, HeadBuilder] = {}
    _vf_heads: dict[str, HeadBuilder] = {}
    _action_dists: dict[str, Callable] = {}

    @classmethod
    def register_encoder(cls, name: str = None):
        """注册自定义 encoder.

        Usage:
            @ComponentRegistry.register_encoder("my_encoder")
            def build_my_encoder(observation_space, action_space, model_config, framework):
                return MyEncoder(...)
        """
        def decorator(func: EncoderBuilder):
            encoder_name = name or func.__name__
            cls._encoders[encoder_name] = func
            _registered_builders[encoder_name] = func  # 存到全局，供跨进程使用
            return func
        return decorator

    @classmethod
    def register_actor_head(cls, name: str = None):
        """注册自定义 actor head (策略头).

        Usage:
            @ComponentRegistry.register_actor_head("my_pi_head")
            def build_pi_head(input_dim, action_space, model_config, framework):
                return MyPiHead(...)
        """
        def decorator(func: HeadBuilder):
            head_name = name or func.__name__
            cls._actor_heads[head_name] = func
            _registered_builders[head_name] = func  # 存到全局，供跨进程使用
            return func
        return decorator

    @classmethod
    def register_critic_head(cls, name: str = None):
        """注册自定义 critic head (价值头).

        Usage:
            @ComponentRegistry.register_critic_head("my_vf_head")
            def build_vf_head(input_dim, model_config, framework):
                return MyVfHead(...)
        """
        def decorator(func: HeadBuilder):
            head_name = name or func.__name__
            cls._critic_heads[head_name] = func
            _registered_builders[head_name] = func  # 存到全局，供跨进程使用
            return func
        return decorator

    @classmethod
    def register_q_head(cls, name: str = None):
        """注册自定义 Q-network head (用于 SAC/DQN).

        Usage:
            @ComponentRegistry.register_q_head("my_q_head")
            def build_q_head(input_dim, action_space, model_config, framework):
                return MyQHead(...)
        """
        def decorator(func: HeadBuilder):
            head_name = name or func.__name__
            cls._q_heads[head_name] = func
            return func
        return decorator

    @classmethod
    def register_vf_head(cls, name: str = None):
        """注册自定义 Value Function head (用于 DQN dueling architecture).

        Usage:
            @ComponentRegistry.register_vf_head("my_vf_head")
            def build_vf_head(input_dim, model_config, framework):
                return MyVfHead(...)
        """
        def decorator(func: HeadBuilder):
            head_name = name or func.__name__
            cls._vf_heads[head_name] = func
            return func
        return decorator

    @classmethod
    def register_action_dist(cls, name: str = None):
        """注册自定义动作分布.

        Usage:
            @ComponentRegistry.register_action_dist("my_dist")
            def build_action_dist(action_space, model_config, framework):
                return MyDistribution(...)
        """
        def decorator(func: Callable):
            dist_name = name or func.__name__
            cls._action_dists[dist_name] = func
            return func
        return decorator

    @classmethod
    def get_encoder(cls, name: str) -> EncoderBuilder | None:
        """获取注册的 encoder builder."""
        if name == "default":
            return None
        # 先从跨进程全局 dict 查
        if name in _registered_builders:
            return _registered_builders[name]
        return cls._encoders.get(name)

    @classmethod
    def get_actor_head(cls, name: str) -> HeadBuilder | None:
        """获取注册的 actor head builder."""
        if name == "default":
            return None
        if name in _registered_builders:
            return _registered_builders[name]
        return cls._actor_heads.get(name)

    @classmethod
    def get_critic_head(cls, name: str) -> HeadBuilder | None:
        """获取注册的 critic head builder."""
        if name == "default":
            return None
        if name in _registered_builders:
            return _registered_builders[name]
        return cls._critic_heads.get(name)

    @classmethod
    def get_q_head(cls, name: str) -> HeadBuilder | None:
        """获取注册的 Q head builder."""
        if name == "default":
            return None
        return cls._q_heads.get(name)

    @classmethod
    def get_vf_head(cls, name: str) -> HeadBuilder | None:
        """获取注册的 VF head builder."""
        if name == "default":
            return None
        return cls._vf_heads.get(name)

    @classmethod
    def get_action_dist(cls, name: str) -> Callable | None:
        """获取注册的动作分布 builder."""
        if name == "default":
            return None
        return cls._action_dists.get(name)

    @classmethod
    def list_encoders(cls) -> list[str]:
        """列出所有注册的 encoder."""
        return list(cls._encoders.keys())

    @classmethod
    def list_actor_heads(cls) -> list[str]:
        """列出所有注册的 actor head."""
        return list(cls._actor_heads.keys())

    @classmethod
    def list_critic_heads(cls) -> list[str]:
        """列出所有注册的 critic head."""
        return list(cls._critic_heads.keys())

    @classmethod
    def list_q_heads(cls) -> list[str]:
        """列出所有注册的 Q head."""
        return list(cls._q_heads.keys())

    @classmethod
    def list_vf_heads(cls) -> list[str]:
        """列出所有注册的 VF head."""
        return list(cls._vf_heads.keys())


# ----------------------------------------------------------------------
# 组合式 Catalog 实现
# ----------------------------------------------------------------------


class PPOCompositeCatalog(PPOCatalog):
    """PPO 的可组合 Catalog.

    允许用户自定义 encoder, actor_head, critic_head 中的任意组件。

    Usage:
        config = CustomPPOConfig()
        config.rl_module(
            policy_map_fn=lambda *args, **kwargs: RLModuleSpec(
                catalog_class=PPOCompositeCatalog,
                model_config_dict={"_framework_custom_config": {
                    "custom_encoder": "my_encoder"
                }}
            )
        )
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict,
    ):
        # 提取框架自定义配置
        self._framework_custom_config = model_config_dict.get(
            "_framework_custom_config", {}
        )
        super().__init__(observation_space, action_space, model_config_dict)

    def build_actor_critic_encoder(self, framework: str):
        """构建 Actor-Critic Encoder."""
        custom_encoder = self._framework_custom_config.get("custom_encoder")

        # 支持直接传 builder 函数（可序列化）或字符串名字（需注册表）
        if callable(custom_encoder):
            return custom_encoder(
                self.observation_space,
                self.action_space,
                self._model_config_dict,
                framework,
            )

        encoder_builder = ComponentRegistry.get_encoder(custom_encoder)
        if encoder_builder is not None:
            return encoder_builder(
                self.observation_space,
                self.action_space,
                self._model_config_dict,
                framework,
            )
        return super().build_actor_critic_encoder(framework)

    def build_pi_head(self, framework: str):
        """构建 Policy Head."""
        custom_head = self._framework_custom_config.get("custom_actor_head")

        if callable(custom_head):
            return custom_head(
                self.latent_dims,
                self.action_space,
                self._model_config_dict,
                framework,
            )

        head_builder = ComponentRegistry.get_actor_head(custom_head)
        if head_builder is not None:
            return head_builder(
                self.latent_dims,
                self.action_space,
                self._model_config_dict,
                framework,
            )
        return super().build_pi_head(framework)

    def build_vf_head(self, framework: str):
        """构建 Value Function Head."""
        custom_head = self._framework_custom_config.get("custom_critic_head")

        if callable(custom_head):
            return custom_head(
                self.latent_dims,
                self._model_config_dict,
                framework,
            )

        head_builder = ComponentRegistry.get_critic_head(custom_head)
        if head_builder is not None:
            return head_builder(
                self.latent_dims,
                self._model_config_dict,
                framework,
            )
        return super().build_vf_head(framework)


class SACCompositeCatalog(SACCatalog):
    """SAC 的可组合 Catalog.

    允许用户自定义 encoder, actor_head, critic_head, q_head。
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict,
    ):
        self._framework_custom_config = model_config_dict.get(
            "_framework_custom_config", {}
        )
        super().__init__(observation_space, action_space, model_config_dict)

    def build_encoder(self, framework: str):
        """构建 Encoder (Policy network)."""
        custom_encoder = self._framework_custom_config.get("custom_encoder")
        encoder_builder = ComponentRegistry.get_encoder(custom_encoder)

        if encoder_builder is not None:
            return encoder_builder(
                self.observation_space,
                self.action_space,
                self._model_config_dict,
                framework,
            )
        return super().build_encoder(framework)

    def build_pi_head(self, framework: str):
        """构建 Policy Head (Squashed Gaussian for continuous, Categorical for discrete)."""
        custom_head = self._framework_custom_config.get("custom_actor_head")
        head_builder = ComponentRegistry.get_actor_head(custom_head)

        if head_builder is not None:
            return head_builder(
                self.latent_dims,
                self.action_space,
                self._model_config_dict,
                framework,
            )
        return super().build_pi_head(framework)

    def build_qf_head(self, framework: str):
        """构建 Q-Function Head."""
        custom_q_head = self._framework_custom_config.get("custom_q_head")
        q_builder = ComponentRegistry.get_q_head(custom_q_head)

        if q_builder is not None:
            return q_builder(
                self.latent_dims,
                self.action_space,
                self._model_config_dict,
                framework,
            )
        return super().build_qf_head(framework)


class DQNCompositeCatalog(DQNCatalog):
    """DQN 的可组合 Catalog.

    允许用户自定义 encoder, q_head。
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        model_config_dict: dict,
    ):
        self._framework_custom_config = model_config_dict.get(
            "_framework_custom_config", {}
        )
        super().__init__(observation_space, action_space, model_config_dict)

    def build_encoder(self, framework: str):
        """构建 Encoder."""
        custom_encoder = self._framework_custom_config.get("custom_encoder")
        encoder_builder = ComponentRegistry.get_encoder(custom_encoder)

        if encoder_builder is not None:
            return encoder_builder(
                self.observation_space,
                self.action_space,
                self._model_config_dict,
                framework,
            )
        return super().build_encoder(framework)

    def build_af_head(self, framework: str):
        """构建 Advantage / Q-Function Head (af_head = Q-function head)."""
        custom_q_head = self._framework_custom_config.get("custom_q_head")
        q_builder = ComponentRegistry.get_q_head(custom_q_head)

        if q_builder is not None:
            return q_builder(
                self.latent_dims,
                self.action_space,
                self._model_config_dict,
                framework,
            )
        return super().build_af_head(framework)

    def build_vf_head(self, framework: str):
        """构建 Value Function Head (for dueling architecture)."""
        custom_vf_head = self._framework_custom_config.get("custom_vf_head")
        vf_builder = ComponentRegistry.get_vf_head(custom_vf_head)

        if vf_builder is not None:
            return vf_builder(
                self.latent_dims,
                self._model_config_dict,
                framework,
            )
        return super().build_vf_head(framework)


# ----------------------------------------------------------------------
# 便捷函数
# ----------------------------------------------------------------------

def register_encoder(name: str = None):
    """ComponentRegistry.register_encoder 的便捷别名."""
    return ComponentRegistry.register_encoder(name)


def register_actor_head(name: str = None):
    """ComponentRegistry.register_actor_head 的便捷别名."""
    return ComponentRegistry.register_actor_head(name)


def register_critic_head(name: str = None):
    """ComponentRegistry.register_critic_head 的便捷别名."""
    return ComponentRegistry.register_critic_head(name)


def register_q_head(name: str = None):
    """ComponentRegistry.register_q_head 的便捷别名."""
    return ComponentRegistry.register_q_head(name)


def register_action_dist(name: str = None):
    """ComponentRegistry.register_action_dist 的便捷别名."""
    return ComponentRegistry.register_action_dist(name)

