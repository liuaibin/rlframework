"""Base mixin that adds common extension hooks to any RLlib Algorithm subclass.

Design: We do NOT introduce a parallel class hierarchy. Instead, this mixin
is applied to a concrete RLlib Algorithm (e.g. PPO, SAC) so that framework
users only need to override the hook methods they care about.
"""

from ray.rllib.utils.typing import ResultDict


class FrameworkAlgorithmMixin:
    """Mixin providing standard extension points for RLlib algorithms.

    Override any combination of the methods below in your subclass.  All
    default implementations delegate back to the parent RLlib behaviour, so
    untouched methods cost nothing.
    """

    def on_before_training_step(self) -> None:
        """Called at the very start of :py:meth:`training_step`."""

    def on_after_training_step(self, result: ResultDict) -> ResultDict:
        """Called at the end of :py:meth:`training_step` with the result dict.

        Subclasses may add extra keys to *result* here.

        Returns:
            The (possibly modified) result dict.
        """
        return result
