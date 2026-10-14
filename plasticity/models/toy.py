"""
models/toy.py — Tiny scalar-output networks for the low-dimensional toy experiments.

Build them through models.factory.build_model ("minimal", "shallow_64", ...).

All networks:
  - Use ReLU activations
  - No batch norm, dropout, or residual connections
  - Support bias-free and with-bias variants
  - Output a single logit for binary classification
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ── Base helper ──────────────────────────────────────────────────────────────

def _linear(in_f: int, out_f: int, bias: bool) -> nn.Linear:
    layer = nn.Linear(in_f, out_f, bias=bias)
    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


# ── Minimal model ────────────────────────────────────────────────────────────

class MinimalNet(nn.Module):
    """
    One hidden layer, tiny width (default 4).
    Bias-free by default for clean geometric analysis.
    """
    def __init__(self, d_in: int, hidden: int = 4, bias: bool = False):
        super().__init__()
        self.fc1 = _linear(d_in, hidden, bias)
        self.fc2 = _linear(hidden, 1, bias)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x))).squeeze(-1)

    def hidden_activations(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc1(x))


# ── Shallow model ────────────────────────────────────────────────────────────

class ShallowNet(nn.Module):
    """
    One hidden layer, width in {32, 64, 128, 256}.
    """
    def __init__(self, d_in: int, hidden: int = 64, bias: bool = True):
        super().__init__()
        self.fc1 = _linear(d_in, hidden, bias)
        self.fc2 = _linear(hidden, 1, bias)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x))).squeeze(-1)

    def hidden_activations(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc1(x))


# ── Deep model ───────────────────────────────────────────────────────────────

class DeepNet(nn.Module):
    """
    Four hidden layers, moderate width.
    No batch norm / dropout / residual.
    """
    def __init__(self, d_in: int, hidden: int = 64, n_layers: int = 4,
                 bias: bool = True):
        super().__init__()
        layers: list[nn.Module] = []
        in_f = d_in
        for _ in range(n_layers):
            layers.append(_linear(in_f, hidden, bias))
            layers.append(nn.ReLU())
            in_f = hidden
        self.body = nn.Sequential(*layers)
        self.head = _linear(hidden, 1, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x)).squeeze(-1)

    def hidden_activations(self, x: torch.Tensor) -> torch.Tensor:
        """Activations at the last hidden layer."""
        return self.body(x)


# ── Linear baseline ──────────────────────────────────────────────────────────

class LinearNet(nn.Module):
    """Single linear layer — useful to check if effects need nonlinearity."""
    def __init__(self, d_in: int, bias: bool = True):
        super().__init__()
        self.fc = _linear(d_in, 1, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x).squeeze(-1)

    def hidden_activations(self, x: torch.Tensor) -> torch.Tensor:
        return x  # identity — no hidden representation


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
