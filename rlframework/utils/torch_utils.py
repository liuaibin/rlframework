"""PyTorch utility helpers used across the framework."""

from collections.abc import Iterable

import torch
import torch.nn as nn


def get_device(prefer_gpu: bool = True) -> torch.device:
    """Return the best available device.

    Args:
        prefer_gpu: Use CUDA if available and *prefer_gpu* is ``True``.

    Returns:
        A :class:`torch.device` instance.
    """
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count the total number of parameters in a model.

    Args:
        model: The PyTorch module to inspect.
        trainable_only: When ``True`` count only trainable parameters.

    Returns:
        Total parameter count.
    """
    params: Iterable[nn.Parameter] = (
        model.parameters() if not trainable_only
        else filter(lambda p: p.requires_grad, model.parameters())
    )
    return sum(p.numel() for p in params)


def freeze_parameters(model: nn.Module) -> None:
    """Set ``requires_grad = False`` for all parameters of *model*."""
    for p in model.parameters():
        p.requires_grad_(False)


def unfreeze_parameters(model: nn.Module) -> None:
    """Set ``requires_grad = True`` for all parameters of *model*."""
    for p in model.parameters():
        p.requires_grad_(True)


def polyak_update(
    source: nn.Module,
    target: nn.Module,
    tau: float = 0.005,
) -> None:
    """Soft (Polyak) update of *target* towards *source*.

    ``target = tau * source + (1 - tau) * target``

    Args:
        source: Network whose weights are the source of truth.
        target: Network to update in-place.
        tau: Interpolation coefficient (0 = no update, 1 = hard copy).
    """
    with torch.no_grad():
        for src_p, tgt_p in zip(source.parameters(), target.parameters(), strict=False):
            tgt_p.data.mul_(1.0 - tau).add_(src_p.data * tau)
