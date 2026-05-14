"""
models/mlp.py — MLP architectures for plasticity experiments.

Supported configurations (from the experimental setting):
    - 2 hidden layers, width 512 or 1024
    - 3 hidden layers, width 256
    - 4 hidden layers, width 256 or 512

Optional:
    - Layer Normalization after each linear layer
    - Configurable activation function (ReLU, GeLU, LeakyReLU)
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _make_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name in ("leaky_relu", "leakyrelu"):
        return nn.LeakyReLU(negative_slope=0.01)
    else:
        raise ValueError(f"Unknown activation '{name}'. Choose: relu, gelu, leaky_relu")


class MLP(nn.Module):
    """
    Fully connected network with configurable depth, width, activation,
    and optional Layer Normalization.

    Args:
        d_in:           Input dimension (flattened).
        n_classes:      Number of output classes.
        hidden_sizes:   List of hidden layer widths, e.g. [512, 512].
        activation:     Activation name: "relu" | "gelu" | "leaky_relu".
        use_layer_norm: If True, LayerNorm is inserted after each linear layer
                        (before the activation).
        bias:           Whether linear layers have bias terms.
    """

    def __init__(
        self,
        d_in: int,
        n_classes: int,
        hidden_sizes: list[int],
        activation: str = "relu",
        use_layer_norm: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.flatten = nn.Flatten()

        layers: list[nn.Module] = []
        in_f = d_in
        for h in hidden_sizes:
            layers.append(nn.Linear(in_f, h, bias=bias))
            if use_layer_norm:
                layers.append(nn.LayerNorm(h))
            layers.append(_make_activation(activation))
            in_f = h

        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(in_f, n_classes, bias=bias)

        self._init_weights(activation)

    def _init_weights(self, activation: str) -> None:
        nonlinearity = "leaky_relu" if "leaky" in activation else "relu"
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        return self.head(self.body(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return last-hidden-layer activations."""
        return self.body(self.flatten(x))


# ── Named configs from the experimental setting ───────────────────────────────

MLP_CONFIGS: dict[str, dict] = {
    "mlp_2x512":   {"hidden_sizes": [512, 512]},
    "mlp_2x1024":  {"hidden_sizes": [1024, 1024]},
    "mlp_3x256":   {"hidden_sizes": [256, 256, 256]},
    "mlp_4x256":   {"hidden_sizes": [256, 256, 256, 256]},
    "mlp_4x512":   {"hidden_sizes": [512, 512, 512, 512]},
}
