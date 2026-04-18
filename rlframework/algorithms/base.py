"""Base mixin that adds common extension hooks to any RLlib Algorithm subclass.

Design: We do NOT introduce a parallel class hierarchy. Instead, this mixin
is applied to a concrete RLlib Algorithm (e.g. PPO, SAC) so that framework
users only need to override the hook methods they care about.
"""

from typing import Any

from ray.rllib.utils.typing import ResultDict

from rlframework.config.validators import (
    validate_lr,
    validate_gamma,
    validate_gpu_config,
    validate_workers,
)
from rlframework.utils.exceptions import ConfigurationError


class FrameworkAlgorithmMixin:
    """Mixin providing standard extension points for RLlib algorithms.

    Override any combination of the methods below in your subclass.  All
    default implementations delegate back to the parent RLlib behaviour, so
    untouched methods cost nothing.

    Typical usage::

        class MyPPO(FrameworkAlgorithmMixin, CustomPPO):
            def build_extra_model_config(self) -> dict:
                return {"custom_model_config": {"hidden": 256}}

            def compute_grads_postprocess(self, grads):
                # clip gradients after PPO default processing
                return {k: v.clamp(-1, 1) for k, v in grads.items()}
    """

    # ------------------------------------------------------------------
    # Model construction hook
    # ------------------------------------------------------------------

    def build_extra_model_config(self) -> dict[str, Any]:
        """Return extra ``model_config`` keys to merge into the algorithm config.

        Called once during :py:meth:`setup`.  Subclasses should return a
        plain dict; the base implementation returns ``{}``.
        """
        return {}

    # ------------------------------------------------------------------
    # Loss hooks
    # ------------------------------------------------------------------

    def loss_postprocess(
        self,
        total_loss,
        loss_components: dict[str, Any],
    ):
        """Post-process the final loss tensor before backward pass.

        Args:
            total_loss: The summed loss tensor produced by the default learner.
            loss_components: Named sub-losses (e.g. ``{"policy_loss": ...,
                "vf_loss": ...}``).

        Returns:
            Modified ``total_loss`` tensor (same device / dtype).
        """
        return total_loss

    # ------------------------------------------------------------------
    # Training-step hooks
    # ------------------------------------------------------------------

    def on_before_training_step(self) -> None:
        """Called at the very start of :py:meth:`training_step`."""

    def on_after_training_step(self, result: ResultDict) -> ResultDict:
        """Called at the end of :py:meth:`training_step` with the result dict.

        Subclasses may add extra keys to *result* here.

        Returns:
            The (possibly modified) result dict.
        """
        return result

    # ------------------------------------------------------------------
    # Gradient hooks
    # ------------------------------------------------------------------

    def compute_grads_postprocess(self, grads: dict | None) -> dict | None:
        """Hook to modify gradients *after* they are computed.

        Args:
            grads: Gradient dict keyed by parameter name.  May be ``None``
                when the algorithm does not expose gradients directly.

        Returns:
            The (possibly modified) gradient dict.
        """
        return grads

    # ------------------------------------------------------------------
    # Configuration validation hooks
    # ------------------------------------------------------------------

    def validate_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Validate and sanitize configuration before training starts.

        Override this method to add custom validation logic for your algorithm.
        This is called during setup() before the algorithm is built.

        Args:
            config: The algorithm configuration dictionary.

        Returns:
            The validated (possibly modified) configuration dictionary.

        Raises:
            ConfigurationError: If configuration is invalid.
        """
        return config

    @staticmethod
    def _validate_training_config(config: dict[str, Any]) -> None:
        """Validate common training hyperparameters.

        Args:
            config: The training configuration dictionary.

        Raises:
            ConfigurationError: If any validation fails.
        """
        # Validate learning rate
        if "lr" in config and config["lr"] is not None:
            try:
                validate_lr(config["lr"], "training.lr")
            except Exception as e:
                raise ConfigurationError(
                    f"Invalid learning rate: {e}",
                    field="training.lr",
                ) from e

        # Validate discount factor
        if "gamma" in config and config["gamma"] is not None:
            try:
                validate_gamma(config["gamma"], "training.gamma")
            except Exception as e:
                raise ConfigurationError(
                    f"Invalid discount factor: {e}",
                    field="training.gamma",
                ) from e


        # Validate GPU config
        if "num_gpus" in config:
            try:
                validate_gpu_config(config["num_gpus"])
            except Exception as e:
                raise ConfigurationError(
                    str(e),
                    field="num_gpus",
                ) from e
