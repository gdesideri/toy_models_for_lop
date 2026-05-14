"""
models/factory.py — Central factory for models and optimizers.

    model  = build_model("mlp_2x512", d_in=3072, n_classes=10,
                         activation="gelu", use_layer_norm=True)

    optim  = build_optimizer(model, "adam", lr=1e-3)
    optim  = build_optimizer(model, "sgd",  lr=0.01, momentum=0.9)
    # EWC is not an optimizer — use the EWC class directly after training each task.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from models.mlp import MLP, MLP_CONFIGS
from models.cnn import CNN


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(
    arch: str,
    d_in: int | None = None,
    in_channels: int | None = None,
    input_size: int | None = None,
    n_classes: int = 10,
    activation: str = "relu",
    use_layer_norm: bool = False,
    **kwargs: Any,
) -> nn.Module:
    """
    Build a model by name.

    MLP archs (require d_in):
        "mlp_2x512"   — 2 hidden layers, width 512
        "mlp_2x1024"  — 2 hidden layers, width 1024
        "mlp_3x256"   — 3 hidden layers, width 256
        "mlp_4x256"   — 4 hidden layers, width 256
        "mlp_4x512"   — 4 hidden layers, width 512

    CNN arch (requires in_channels and input_size):
        "cnn"         — 4 conv (32ch) + 2 FC (256)

    Common kwargs forwarded to the constructor:
        activation    — "relu" | "gelu" | "leaky_relu"  (default "relu")
        use_layer_norm — bool (default False)
        n_classes     — int (default 10)
    """
    arch_lower = arch.lower()

    if arch_lower in MLP_CONFIGS:
        if d_in is None:
            raise ValueError(f"MLP arch '{arch}' requires d_in (flattened input size).")
        cfg = MLP_CONFIGS[arch_lower]
        return MLP(
            d_in=d_in,
            n_classes=n_classes,
            hidden_sizes=cfg["hidden_sizes"],
            activation=activation,
            use_layer_norm=use_layer_norm,
            **kwargs,
        )

    if arch_lower == "cnn":
        if in_channels is None or input_size is None:
            raise ValueError("CNN arch requires in_channels and input_size.")
        return CNN(
            in_channels=in_channels,
            input_size=input_size,
            n_classes=n_classes,
            activation=activation,
            use_layer_norm=use_layer_norm,
            **kwargs,
        )

    raise ValueError(
        f"Unknown arch '{arch}'. Available: {list(MLP_CONFIGS.keys()) + ['cnn']}"
    )


# ── Optimizer factory ─────────────────────────────────────────────────────────

def build_optimizer(
    model: nn.Module,
    optimizer: str,
    lr: float = 1e-3,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """
    Build an optimizer for `model`.

    Args:
        model:     The network whose parameters will be optimized.
        optimizer: "adam" | "sgd".
                   Note: EWC is a regularizer, not an optimizer — import
                   models.ewc.EWC directly and add its penalty to the loss.
        lr:        Learning rate.
        **kwargs:  Forwarded to the optimizer constructor.
                   Useful keys: weight_decay, momentum (SGD), betas (Adam).

    Returns:
        A torch.optim.Optimizer instance.
    """
    name = optimizer.lower()

    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, **kwargs)

    if name in ("sgd", "sgd_momentum"):
        momentum = kwargs.pop("momentum", 0.9 if name == "sgd_momentum" else 0.0)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, **kwargs)

    if name == "ewc":
        raise ValueError(
            "EWC is a continual-learning regularizer, not an optimizer. "
            "Use `build_optimizer(..., optimizer='adam')` or 'sgd' for the base optimizer, "
            "then create `from models.ewc import EWC` and call `ewc.penalty(model)` "
            "as an additive term in your loss."
        )

    raise ValueError(
        f"Unknown optimizer '{optimizer}'. Available: adam, sgd. "
        "For EWC regularization see models.ewc.EWC."
    )


# ── Convenience ───────────────────────────────────────────────────────────────

ACTIVATIONS = ("relu", "gelu", "leaky_relu")
OPTIMIZERS  = ("adam", "sgd")
ARCHS       = list(MLP_CONFIGS.keys()) + ["cnn"]


def list_archs() -> list[str]:
    return ARCHS


def list_activations() -> list[str]:
    return list(ACTIVATIONS)


def list_optimizers() -> list[str]:
    return list(OPTIMIZERS)
