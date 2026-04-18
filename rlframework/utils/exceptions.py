"""Custom exceptions for rlframework.

Provides user-friendly error messages with actionable guidance.
"""

from typing import Any


class RLFrameworkError(Exception):
    """Base exception for all rlframework errors."""

    def __init__(self, message: str, hint: str | None = None):
        self.message = message
        self.hint = hint
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        if self.hint:
            return f"{self.message}\n\nHint: {self.hint}"
        return self.message


class ConfigurationError(RLFrameworkError):
    """Raised when algorithm or framework configuration is invalid."""

    def __init__(self, message: str, field: str | None = None, hint: str | None = None):
        self.field = field
        full_message = f"Configuration error: {message}"
        if field:
            full_message = f"Configuration error in '{field}': {message}"
        super().__init__(full_message, hint=hint)


class EnvironmentError(RLFrameworkError):
    """Raised when environment setup or registration fails."""

    def __init__(self, message: str, env_name: str | None = None, hint: str | None = None):
        self.env_name = env_name
        full_message = f"Environment error: {message}"
        if env_name:
            full_message = f"Environment error for '{env_name}': {message}"
        hint = hint or "Make sure the environment is properly registered with gymnasium.register()"
        super().__init__(full_message, hint=hint)


class ModelError(RLFrameworkError):
    """Raised when model configuration or building fails."""

    def __init__(self, message: str, model_name: str | None = None, hint: str | None = None):
        self.model_name = model_name
        full_message = f"Model error: {message}"
        if model_name:
            full_message = f"Model error for '{model_name}': {message}"
        hint = hint or "Register your custom model using @ComponentRegistry.register_encoder()"
        super().__init__(full_message, hint=hint)


class StorageError(RLFrameworkError):
    """Raised when storage operations fail."""

    def __init__(
        self,
        message: str,
        path: str | None = None,
        backend: str | None = None,
        hint: str | None = None,
    ):
        self.path = path
        self.backend = backend
        full_message = f"Storage error: {message}"
        if path:
            full_message = f"Storage error at '{path}': {message}"
        if backend:
            full_message += f" (backend: {backend})"
        hint = hint or "Check that the storage backend is properly configured and accessible"
        super().__init__(full_message, hint=hint)


class AlgorithmError(RLFrameworkError):
    """Raised when algorithm execution fails."""

    def __init__(
        self,
        message: str,
        algorithm: str | None = None,
        hint: str | None = None,
    ):
        self.algorithm = algorithm
        full_message = f"Algorithm error: {message}"
        if algorithm:
            full_message = f"Algorithm error in '{algorithm}': {message}"
        hint = hint or "Check algorithm configuration and custom hook implementations"
        super().__init__(full_message, hint=hint)


class ValidationError(RLFrameworkError):
    """Raised when input validation fails."""

    def __init__(self, message: str, value: Any = None, field: str | None = None):
        self.value = value
        self.field = field
        full_message = f"Validation error: {message}"
        if field:
            full_message = f"Validation error for '{field}': {message}"
        if value is not None:
            full_message += f"\n  Got: {value!r}"
        super().__init__(full_message)


class RayInitError(RLFrameworkError):
    """Raised when Ray initialization fails."""

    def __init__(self, message: str, hint: str | None = None):
        hint = hint or (
            "Make sure Ray is properly installed: pip install ray[rllib]\n"
            "For GPU support, ensure CUDA is configured: ray start --gpu"
        )
        super().__init__(f"Ray initialization failed: {message}", hint=hint)


class CheckpointError(RLFrameworkError):
    """Raised when checkpoint operations fail."""

    def __init__(self, message: str, checkpoint_path: str | None = None, hint: str | None = None):
        self.checkpoint_path = checkpoint_path
        full_message = f"Checkpoint error: {message}"
        if checkpoint_path:
            full_message = f"Checkpoint error for '{checkpoint_path}': {message}"
        hint = hint or "Ensure the checkpoint exists and was created by a compatible algorithm version"
        super().__init__(full_message, hint=hint)
