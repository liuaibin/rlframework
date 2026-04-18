"""Reusable neural-network building blocks.

All components are plain ``torch.nn.Module`` subclasses so they can be
composed freely both inside and outside of RLlib models.
"""



import torch
import torch.nn as nn


class MLP(nn.Module):
    """Fully-connected network with configurable depth, width and activations.

    Args:
        input_dim: Input feature dimension.
        hidden_dims: Sizes of hidden layers.
        output_dim: Output feature dimension (``None`` → no output layer).
        activation: Activation class applied after every hidden layer.
        output_activation: Activation applied after the output layer (optional).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int | None = None,
        activation: nn.Module = nn.ReLU,
        output_activation: nn.Module | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), activation()]
            in_dim = h
        if output_dim is not None:
            layers.append(nn.Linear(in_dim, output_dim))
            if output_activation is not None:
                layers.append(output_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualMLP(nn.Module):
    """MLP with residual (skip) connections between every pair of hidden layers.

    Args:
        input_dim: Input feature dimension.
        hidden_dim: Width of every hidden layer (must all be equal for skip connections).
        num_blocks: Number of residual blocks (each block = 2 linear layers).
        output_dim: Final projection dimension (``None`` → return hidden repr).
    """

    class _Block(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.fc2 = nn.Linear(dim, dim)
            self.act = nn.ReLU()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.act(x + self.fc2(self.act(self.fc1(x))))

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_blocks: int = 2,
        output_dim: int | None = None,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [self._Block(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if output_dim is not None
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)
