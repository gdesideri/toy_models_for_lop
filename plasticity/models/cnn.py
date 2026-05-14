"""
models/cnn.py — CNN architecture for plasticity experiments.

Architecture (from experimental setting):
    4 conv layers  (32 channels each, 3×3 kernel, same padding)
    2 FC layers    (width 256)
    Output head    (n_classes)

Optional:
    - Layer Normalization after each conv / FC layer
    - Configurable activation (relu, gelu, leaky_relu)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.mlp import _make_activation


class CNN(nn.Module):
    """
    4-conv + 2-FC convolutional network.

    Args:
        in_channels:    Input image channels (1 for MNIST, 3 for CIFAR-10).
        input_size:     Spatial size of input (assumed square), e.g. 32 for CIFAR-10.
        n_classes:      Number of output classes.
        activation:     "relu" | "gelu" | "leaky_relu".
        use_layer_norm: LayerNorm after each conv (uses spatial size) and each FC.
        n_channels:     Number of conv filters per layer (default 32).
        fc_width:       Width of both FC layers (default 256).
    """

    def __init__(
        self,
        in_channels: int,
        input_size: int,
        n_classes: int,
        activation: str = "relu",
        use_layer_norm: bool = False,
        n_channels: int = 32,
        fc_width: int = 256,
    ):
        super().__init__()

        conv_layers: list[nn.Module] = []
        ch_in = in_channels
        spatial = input_size

        for _ in range(4):
            conv_layers.append(nn.Conv2d(ch_in, n_channels, kernel_size=3, padding=1))
            if use_layer_norm:
                # LayerNorm over (C, H, W) after this conv
                conv_layers.append(nn.LayerNorm([n_channels, spatial, spatial]))
            conv_layers.append(_make_activation(activation))
            conv_layers.append(nn.MaxPool2d(2))
            spatial = spatial // 2
            ch_in = n_channels

        self.conv_body = nn.Sequential(*conv_layers)
        self.flatten = nn.Flatten()

        flat_dim = n_channels * spatial * spatial

        fc_layers: list[nn.Module] = []
        in_f = flat_dim
        for _ in range(2):
            fc_layers.append(nn.Linear(in_f, fc_width))
            if use_layer_norm:
                fc_layers.append(nn.LayerNorm(fc_width))
            fc_layers.append(_make_activation(activation))
            in_f = fc_width

        self.fc_body = nn.Sequential(*fc_layers)
        self.head = nn.Linear(fc_width, n_classes)

        self._init_weights(activation)

    def _init_weights(self, activation: str) -> None:
        nonlinearity = "leaky_relu" if "leaky" in activation else "relu"
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_uniform_(m.weight, nonlinearity=nonlinearity)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_body(x)
        x = self.flatten(x)
        return self.head(self.fc_body(x))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Return last FC-body activations."""
        x = self.conv_body(x)
        return self.fc_body(self.flatten(x))
