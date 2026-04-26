"""CustomSAC - extends RLlib's SAC with framework hooks.

Usage::

    from rlframework.algorithms.sac import CustomSAC

    class MySAC(CustomSAC):

        def on_after_training_step(self, result):
            result["my_metric"] = compute_my_metric()
            return result
"""

import importlib
from typing import Any

from ray.rllib.algorithms.sac import SAC, SACConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.rllib.utils.annotations import override
from ray.rllib.utils.replay_buffers.episode_replay_buffer import EpisodeReplayBuffer

from rlframework.algorithms.base import FrameworkAlgorithmMixin
from rlframework.config.framework_config import FrameworkConfigMixin
from rlframework.models.catalog import (
    SACCompositeCatalog,
)


def _resolve_replay_buffer_type(buffer_type: Any) -> type | None:
    """Resolve replay buffer type from class object or import-path string."""
    if isinstance(buffer_type, type):
        return buffer_type
    if not isinstance(buffer_type, str) or "." not in buffer_type:
        return None
    module_name, _, attr_name = buffer_type.rpartition(".")
    if not module_name or not attr_name:
        return None
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    resolved = getattr(module, attr_name, None)
    return resolved if isinstance(resolved, type) else None


class CustomSACConfig(SACConfig, FrameworkConfigMixin):
    """SACConfig extended with rlframework-specific settings.

    Usage::

        config = (
            CustomSACConfig()
            .environment("Pendulum-v1")
            .storage(backend="minio", endpoint="minio:9000",
                     access_key="admin", secret_key="admin",
                     bucket="rl-models")
            .metrics(reporters=["influxdb", "file"])
            .checkpointing(freq=10, upload_async=True)
            .training(lr=3e-4)
        )
        algo = config.build()
    """

    def __init__(self):
        super().__init__(algo_class=CustomSAC)
        # Initialize the framework mixin
        self._init_framework_mixin()

    def checkpointing(
        self,
        freq: int = 0,
        local_dir: str = "./checkpoints",
    ) -> "CustomSACConfig":
        """Store checkpointing preferences for use with :class:`~rlframework.storage.AutoCheckpoint`."""
        self.framework_checkpointing(freq=freq, local_dir=local_dir)
        return self

    @override(SACConfig)
    def build(self, *args, **kwargs):
        self._apply_framework_runtime_config()
        return super().build(*args, **kwargs)

    @override(SACConfig)
    def validate(self) -> None:
        # Bypass the hard-coded whitelist in SACConfig.validate() that only
        # accepts four built-in EpisodeReplayBuffer subclasses.  Our
        # PrioritizedSumTreeBuffer inherits from PrioritizedEpisodeReplayBuffer,
        # so it IS a valid EpisodeReplayBuffer subclass -- the parent's check
        # fails only because it matches the type against hard-coded strings /
        # module paths rather than doing a proper issubclass() test.
        # We perform the same replay-buffer validation as the parent but with
        # a correct issubclass() check, then skip the parent's ValueError.
        buffer_type = self.replay_buffer_config.get("type")

        def _is_valid_episode_buffer() -> bool:
            if buffer_type is None:
                return True
            if isinstance(buffer_type, str) and "Episode" in buffer_type:
                return True
            resolved = _resolve_replay_buffer_type(buffer_type)
            if isinstance(resolved, type) and issubclass(
                resolved, EpisodeReplayBuffer
            ):
                return True
            return False

        # Run the parent's logic.  If it raises ValueError because the type is
        # not in its hard-coded whitelist, we re-check with issubclass() and
        # only let the error through if it is genuinely invalid.
        try:
            super().validate()
        except ValueError as e:
            if (
                "EpisodeReplayBuffer" in str(e)
                and self.enable_env_runner_and_connector_v2
                and _is_valid_episode_buffer()
            ):
                # The buffer is a valid EpisodeReplayBuffer subclass;
                # skip the whitelist error raised by the parent.
                return
            raise

    def framework_models(
        self,
        encoder: str | None = None,
        actor_head: str | None = None,
        critic_head: str | None = None,
        q_head: str | None = None,
    ) -> "CustomSACConfig":
        """Configure custom model components for SAC.

        This allows you to mix and match custom components with default ones.

        Args:
            encoder: Name of a registered custom encoder, or None/"default" for default.
            actor_head: Name of a registered custom actor head, or None/"default" for default.
            critic_head: Name of a registered custom critic head, or None/"default" for default.
            q_head: Name of a registered custom Q head, or None/"default" for default.

        Usage:
            config.framework_models(
                encoder="my_encoder",
                actor_head="my_actor_head",
                q_head="my_q_head"
            )
        """
        # Build the custom config dict
        custom_config = {}

        if encoder and encoder != "default":
            custom_config["custom_encoder"] = encoder
        if actor_head and actor_head != "default":
            custom_config["custom_actor_head"] = actor_head
        if critic_head and critic_head != "default":
            custom_config["custom_critic_head"] = critic_head
        if q_head and q_head != "default":
            custom_config["custom_q_head"] = q_head

        # If we have custom components, configure RLModule to use CompositeCatalog
        if custom_config:
            # Update model config
            self.model.update({
                "_framework_custom_config": custom_config,
            })

            # Set catalog class to SACCompositeCatalog
            self.rl_module(
                rl_module_spec=RLModuleSpec(
                    catalog_class=SACCompositeCatalog,
                    model_config=self.model,
                )
            )

        return self


class CustomSAC(FrameworkAlgorithmMixin, SAC):
    """SAC with rlframework extension hooks.

    All :py:class:`FrameworkAlgorithmMixin` hook methods are available.
    """

    @classmethod
    @override(SAC)
    def get_default_config(cls) -> CustomSACConfig:
        return CustomSACConfig()

    @override(SAC)
    def setup(self, config: CustomSACConfig):
        super().setup(config)

    @override(SAC)
    def _create_local_replay_buffer_if_necessary(self, config):
        # The parent Algorithm._create_local_replay_buffer_if_necessary() has a
        # broken "in" check (line ~4066) that crashes when replay_buffer_config["type"]
        # is a class rather than a string.  We replicate the parent's logic but
        # handle the case where type is a class.
        if not config.get("replay_buffer_config") or config["replay_buffer_config"].get(
            "no_local_replay_buffer"
        ):
            return None

        buffer_type = config["replay_buffer_config"].get("type", "")

        # Determine if this is an Episode-based buffer (supports SingleAgentEpisode).
        # The parent's check "EpisodeReplayBuffer" in str(type) fails for subclasses
        # because str(class) returns "<class '...'>".
        from ray.rllib.utils.replay_buffers.episode_replay_buffer import (
            EpisodeReplayBuffer,
        )

        def _is_episode_buffer(t: Any) -> bool:
            if isinstance(t, str):
                if "EpisodeReplayBuffer" in t:
                    return True
                t = _resolve_replay_buffer_type(t)
            if isinstance(t, type):
                return issubclass(t, EpisodeReplayBuffer)
            return False

        if _is_episode_buffer(buffer_type):
            config["replay_buffer_config"][
                "metrics_num_episodes_for_smoothing"
            ] = self.config.metrics_num_episodes_for_smoothing

        # Use from_config to instantiate the replay buffer.
        from ray.rllib.utils.from_config import from_config
        from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer

        return from_config(ReplayBuffer, config["replay_buffer_config"])

    @override(SAC)
    def training_step(self) -> None:
        self.on_before_training_step()
        super().training_step()
        result = self.metrics.peek()
        result = self.on_after_training_step(result)
        if result:
            for key, value in result.items():
                if isinstance(value, (int, float)):
                    if key in self.metrics:
                        self.metrics.log_value(key, value)
                    else:
                        self.metrics.log_value(key, value, window=1)

