"""
utils/checkpoints.py — Save and load model checkpoints.

A checkpoint holds the weights plus whatever is needed to rebuild the network
from nothing, so a snapshot can be reloaded without the script that made it:

    save_checkpoint(path, model, model_kwargs=dict(arch="mlp_2x30", d_in=12288, n_classes=2),
                    task=3, epoch=7)
    load_checkpoint(path, model)                    # into an existing model
    model, ckpt = load_model_from_checkpoint(path)  # rebuild through models.factory
"""
from __future__ import annotations

from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer=None, model_kwargs=None, **meta):
    """
    Writes {"model_state", "optimizer_state", "model_kwargs", "meta"} to `path`.

    model_kwargs: the build_model() arguments, needed by load_model_from_checkpoint.
    meta        : anything else worth keeping (task index, epoch, metrics, ...).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "model_kwargs": model_kwargs,
        "meta": meta,
    }, path)
    return path


def load_checkpoint(path, model=None, optimizer=None, map_location="cpu"):
    """Loads a checkpoint; restores it into `model` / `optimizer` if given. Returns the dict."""
    ckpt = torch.load(path, map_location=map_location)
    if model is not None:
        model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and ckpt.get("optimizer_state") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


def load_model_from_checkpoint(path, map_location="cpu", weights_only=True):
    """
    Rebuilds the network with models.factory.build_model and loads its weights.
    weights_only=False for our own checkpoints whose meta holds numpy scalars.
    """
    from models.factory import build_model

    ckpt = torch.load(path, map_location=map_location, weights_only=weights_only)
    if not ckpt.get("model_kwargs"):
        raise ValueError(f"{path} has no model_kwargs; use load_checkpoint with a built model")
    model = build_model(**ckpt["model_kwargs"])
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt
