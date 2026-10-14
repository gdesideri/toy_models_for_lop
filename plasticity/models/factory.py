"""
models/factory.py — Central factory for models and optimizers.

    model  = build_model("mlp_2x512", d_in=3072, n_classes=10,
                         activation="gelu", use_layer_norm=True)

    optim  = build_optimizer(model, "adam", lr=1e-3)
    optim  = build_optimizer(model, "sgd",  lr=0.01, momentum=0.9)
    # EWC is not an optimizer — use the EWC class directly after training each task.
"""
from __future__ import annotations

import re
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim

from models.mlp import MLP, MLP_CONFIGS
from models.cnn import CNN
from models.toy import MinimalNet, ShallowNet, DeepNet, LinearNet


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(
    arch: str,
    d_in: int | None = None,
    in_channels: int | None = None,
    input_size: int | None = None,
    n_classes: int = 10,
    activation: str = "relu",
    use_layer_norm: bool = False,
    bias: bool = True,
    **kwargs: Any,
) -> nn.Module:
    """
    Build a model by name. The single entry point for every network in the repo.

    MLP archs (require d_in) — any "mlp_<n_layers>x<width>" name works,
    e.g. "mlp_2x100"; the named presets are:
        "mlp_2x512"   — 2 hidden layers, width 512
        "mlp_2x1024"  — 2 hidden layers, width 1024
        "mlp_3x256"   — 3 hidden layers, width 256
        "mlp_4x256"   — 4 hidden layers, width 256
        "mlp_4x512"   — 4 hidden layers, width 512

    CNN arch (requires in_channels and input_size):
        "cnn"         — 4 conv (32ch) + 2 FC (256)

    Toy archs (require d_in; single scalar output, n_classes is ignored):
        "minimal"     — 1 hidden layer of 4, bias-free (bias is forced off)
        "minimal_2"   — 1 hidden layer of 2, bias-free
        "minimal_8"   — 1 hidden layer of 8, bias-free
        "shallow_<h>" — 1 hidden layer of width h
        "deep_<h>"    — 4 hidden layers of width h
        "linear"      — a single linear layer

    Common kwargs forwarded to the constructor:
        activation    — "relu" | "gelu" | "leaky_relu"  (default "relu")
        use_layer_norm — bool (default False)
        n_classes     — int (default 10)
        bias          — bool (default True; MLP and toy archs)
    """
    arch_lower = arch.lower()

    hidden_sizes = None
    if arch_lower in MLP_CONFIGS:
        hidden_sizes = MLP_CONFIGS[arch_lower]["hidden_sizes"]
    else:
        # any "mlp_<n_layers>x<width>" name, e.g. "mlp_2x100"
        m = re.fullmatch(r"mlp_(\d+)x(\d+)", arch_lower)
        if m:
            hidden_sizes = [int(m.group(2))] * int(m.group(1))

    if hidden_sizes is not None:
        if d_in is None:
            raise ValueError(f"MLP arch '{arch}' requires d_in (flattened input size).")
        return MLP(
            d_in=d_in,
            n_classes=n_classes,
            hidden_sizes=hidden_sizes,
            activation=activation,
            use_layer_norm=use_layer_norm,
            bias=bias,
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

    toy = _build_toy(arch_lower, d_in, bias)
    if toy is not None:
        return toy

    raise ValueError(
        f"Unknown arch '{arch}'. Available: {ARCHS}, "
        "or any 'mlp_<n_layers>x<width>', 'shallow_<h>' or 'deep_<h>' name."
    )


def _build_toy(arch: str, d_in: int | None, bias: bool) -> nn.Module | None:
    """The scalar-output toy networks of models/toy.py, or None if `arch` is not one."""
    fixed = {"minimal": 4, "minimal_2": 2, "minimal_8": 8}
    is_toy = (arch in fixed or arch == "linear"
              or re.fullmatch(r"(shallow|deep)_\d+", arch) is not None)
    if not is_toy:
        return None
    if d_in is None:
        raise ValueError(f"Toy arch '{arch}' requires d_in.")

    if arch in fixed:
        return MinimalNet(d_in, hidden=fixed[arch], bias=False)   # bias-free for geometric tests
    if arch.startswith("shallow_"):
        return ShallowNet(d_in, hidden=int(arch.split("_")[1]), bias=bias)
    if arch.startswith("deep_"):
        return DeepNet(d_in, hidden=int(arch.split("_")[1]), bias=bias)
    return LinearNet(d_in, bias=bias)


# ── Head utilities ────────────────────────────────────────────────────────────

def reset_head(model: nn.Module, activation: str = "relu") -> nn.Module:
    """Re-draws `model.head` with the init every classifier here uses (kaiming, zero bias)."""
    nonlinearity = "leaky_relu" if "leaky" in activation else "relu"
    nn.init.kaiming_uniform_(model.head.weight, nonlinearity=nonlinearity)
    if model.head.bias is not None:
        nn.init.zeros_(model.head.bias)
    return model


def freeze_body(model: nn.Module) -> nn.Module:
    """Freezes every parameter except those of `model.head`."""
    head = {id(p) for p in model.head.parameters()}
    for p in model.parameters():
        p.requires_grad_(id(p) in head)
    return model


# ── Optimizer factory ─────────────────────────────────────────────────────────

def build_optimizer(
    model: nn.Module,
    optimizer: str = "sgd",
    lr: float = 1e-3,
    **kwargs: Any,
) -> torch.optim.Optimizer:
    """
    Build an optimizer for `model`.

    Args:
        model:     The network whose parameters will be optimized.
        optimizer: "sgd" (default, momentum 0) | "adam".
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
ARCHS       = list(MLP_CONFIGS.keys()) + ["cnn", "minimal", "minimal_2", "minimal_8",
                                         "shallow_<h>", "deep_<h>", "linear"]


def list_archs() -> list[str]:
    return ARCHS


def list_activations() -> list[str]:
    return list(ACTIVATIONS)


def list_optimizers() -> list[str]:
    return list(OPTIMIZERS)
