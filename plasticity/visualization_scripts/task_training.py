"""
visualization_scripts/task_training.py — Figures for experiments/run_basic_task_training.py.

    plot_history(history, params, out_path)         training loss + train/test accuracy
    save_sample_images(dataset, out_path, params)   original vs permuted example images

Both use the shared plotstyle setup in visualization_scripts/common.py.
"""
from __future__ import annotations


import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless: figures are written to disk
import matplotlib.pyplot as plt

from tasks.permuted_pixels import PermutedPixels

from visualization_scripts import common


def _suptitle(params, extra=""):
    return (f"{params.dataset_kind} / {params.model_kind}"
            + r" — " + rf"${100 * params.perm_percentage:.0f}\%$ permuted pixels" + extra)


def plot_history(history, params, out_path):
    """Train loss over time (left) and train/test accuracy per epoch (right)."""
    common.set_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.plot(history["step"], history["step_loss"],
            **common.style_for(0), alpha=0.35, linewidth=1.0)
    # epoch means, placed at the last step of each epoch
    steps_per_epoch = len(history["step"]) / max(len(history["epoch"]), 1)
    epoch_x = [e * steps_per_epoch for e in history["epoch"]]
    ax.plot(epoch_x, history["train_loss"], **common.style_for(1),
            marker=common.MARKERS[0], label=r"epoch mean")
    ax.set_xlabel(r"step")
    ax.set_ylabel(r"train loss")
    ax.set_title(r"Training loss")
    ax.legend()

    ax = axes[1]
    ax.plot(history["epoch"], history["train_acc"], **common.style_for(0),
            marker=common.MARKERS[0], label=r"train")
    ax.plot(history["epoch"], history["test_acc"], **common.style_for(1),
            marker=common.MARKERS[1], label=r"test")
    ax.set_xlabel(r"epoch")
    ax.set_ylabel(r"accuracy")
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.set_title(r"Accuracy")
    ax.legend()

    common.bold_suptitle(fig, _suptitle(params))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_sample_images(dataset, out_path, params, n=4, rng=None):
    """
    Saves n example images of the task: the original image on top, the same
    image after the pixel permutation below it.

    Each panel is min-max rescaled for display, because the tensors the model
    sees are normalised (Shapes3D lands in [-1, 1]) and would otherwise clip.
    """
    common.set_style()
    base = dataset.dataset if isinstance(dataset, PermutedPixels) else dataset
    rng = rng or np.random.default_rng(0)
    idx = rng.choice(len(dataset), size=min(n, len(dataset)), replace=False)

    def _show(ax, x):
        x = x.detach().cpu().float()
        x = (x - x.min()) / max(float(x.max() - x.min()), 1e-12)
        x = x.permute(1, 2, 0).numpy()
        ax.imshow(x.squeeze(-1) if x.shape[-1] == 1 else x,
                  cmap="gray" if x.shape[-1] == 1 else None,
                  interpolation="nearest")
        ax.set_axis_off()

    fig, axes = plt.subplots(2, len(idx), figsize=(3.0 * len(idx), 6.4),
                             squeeze=False)
    for col, i in enumerate(idx):
        x_orig, y = base[int(i)]
        x_perm, _ = dataset[int(i)]
        _show(axes[0][col], x_orig)
        _show(axes[1][col], x_perm)
        axes[0][col].set_title(rf"$y = {int(y)}$")

    axes[0][0].text(-0.12, 0.5, r"original", transform=axes[0][0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=16)
    axes[1][0].text(-0.12, 0.5, r"permuted", transform=axes[1][0].transAxes,
                    rotation=90, va="center", ha="right", fontsize=16)

    common.bold_suptitle(fig, _suptitle(params, rf" (seed ${params.seed}$)"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
