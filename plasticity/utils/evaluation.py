"""
utils/evaluation.py — Loss and accuracy of a classifier over a DataLoader, the
representations a frozen network gives a dataset, and how many hidden units of a
ReLU MLP are still alive.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def evaluate(model, loader, criterion, device, with_precision=False):
    """(loss, accuracy) over `loader`; (loss, accuracy, precision of class 1) with_precision."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    true_pos, pred_pos = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item() * y.size(0)
            correct += n_correct(logits, y, criterion)
            n += y.size(0)
            if with_precision:
                tp, pp = positive_counts(logits, y)
                true_pos, pred_pos = true_pos + tp, pred_pos + pp
    if with_precision:
        return total_loss / n, correct / n, precision(true_pos, pred_pos)
    return total_loss / n, correct / n


def positive_counts(logits, y):
    """(true positives, predicted positives) of class 1 in a batch, by argmax."""
    predicted = logits.argmax(dim=1) == 1
    return (predicted & (y == 1)).sum().item(), predicted.sum().item()


def precision(true_pos, pred_pos):
    """TP / (TP + FP) of class 1; 0 when nothing is predicted positive."""
    return true_pos / pred_pos if pred_pos else 0.0


def n_correct(logits, y, criterion=None):
    """
    Number of correct predictions in a batch: argmax over the classes, unless the
    criterion defines its own n_correct (e.g. one target per latent factor).
    """
    if criterion is not None and hasattr(criterion, "n_correct"):
        return criterion.n_correct(logits, y)
    return (logits.argmax(dim=1) == y).sum().item()


def extract_features(encoder, loader, device):
    """
    (z, targets) over a whole DataLoader: z = encoder(x) for every input, in loader
    order, under no_grad, both concatenated on `device`. For a frozen encoder this
    is the cache of its representation of the dataset.
    """
    encoder.eval()
    zs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            zs.append(encoder(x.to(device)))
            ys.append(torch.as_tensor(y).to(device))
    return torch.cat(zs), torch.cat(ys)


def alive_units(model, x, batch_size=1024):
    """
    Per hidden layer of a ReLU MLP (models.mlp.MLP), the number of units whose
    output is > 0 for at least one input in `x`. A unit that is 0 on every input
    is dead: it passes no signal forward and receives no gradient.
    """
    return [int(m.sum()) for m in alive_masks(model, x, batch_size)]


def alive_masks(model, x, batch_size=1024):
    """Per hidden layer, a boolean tensor: which units are alive (> 0 on some input of `x`)."""
    return alive_masks_over(model, (x[i:i + batch_size] for i in range(0, len(x), batch_size)))


def alive_masks_over(model, batches):
    """alive_masks over an iterable of input batches, e.g. a whole training set streamed batch by batch."""
    if not hasattr(model, "body"):
        raise ValueError(f"{type(model).__name__} has no .body of hidden layers")
    alive = None
    model.eval()
    with torch.no_grad():
        for xb in batches:
            h = model.flatten(xb) if hasattr(model, "flatten") else xb
            batch = []
            for layer in model.body:
                h = layer(h)
                if isinstance(layer, nn.ReLU):
                    batch.append((h > 0).any(dim=0))
            alive = batch if alive is None else [a | b for a, b in zip(alive, batch)]
    return [a.cpu() for a in alive]


def activation_rates_over(model, batches):
    """
    Per hidden layer of a ReLU MLP, the fraction of the inputs in `batches` on which
    each unit is active (output > 0): 0 = dead on this data, 1 = active on every input.
    """
    if not hasattr(model, "body"):
        raise ValueError(f"{type(model).__name__} has no .body of hidden layers")
    counts, n = None, 0
    model.eval()
    with torch.no_grad():
        for xb in batches:
            h = model.flatten(xb) if hasattr(model, "flatten") else xb
            batch = []
            for layer in model.body:
                h = layer(h)
                if isinstance(layer, nn.ReLU):
                    batch.append((h > 0).sum(dim=0))
            counts = batch if counts is None else [a + b for a, b in zip(counts, batch)]
            n += len(xb)
    return [(c.double() / n).cpu() for c in counts]


_ACTIVATIONS = (nn.ReLU, nn.LeakyReLU, nn.GELU)


def _preactivations(model, x):
    """(the input of each activation of model.body for the batch x, the body's output)."""
    pre, h = [], model.flatten(x) if hasattr(model, "flatten") else x
    for layer in model.body:
        if isinstance(layer, _ACTIVATIONS):
            pre.append(h)
        h = layer(h)
    return pre, h


def effective_rank(h):
    """exp(entropy of the normalised singular values of h): how many directions the rows of h use."""
    s = torch.linalg.svdvals(h.detach().double().cpu())
    p = s / s.sum()
    p = p[p > 0]
    return float(torch.exp(-(p * p.log()).sum()))


def unit_health(model, batches, probe=None, rare=0.01):
    """
    The state of the hidden units of an MLP over the inputs in `batches`, as plain numbers:
      layers         per hidden layer: units, dead (pre-activation <= 0 on every input), rare
                     (> 0 on under `rare` of them; dead units included), median and mean rate
      n_images       how many inputs were measured
      weight_norms   Frobenius norm of every Linear layer, input to output
      effective_rank of the last hidden layer's output on `probe` (a tensor of inputs), if given
    Activity is read before the activation, so ReLU, LeakyReLU and GELU units count alike.
    Changes nothing: no gradients, no random draws, the model's mode is restored.
    """
    if not hasattr(model, "body"):
        raise ValueError(f"{type(model).__name__} has no .body of hidden layers")
    was_training = model.training
    model.eval()
    counts, n = None, 0
    with torch.no_grad():
        for xb in batches:
            pre, _ = _preactivations(model, xb)
            batch = [(p > 0).sum(dim=0) for p in pre]
            counts = batch if counts is None else [a + b for a, b in zip(counts, batch)]
            n += len(xb)
        layers = []
        for c in counts:
            rate = (c.double() / n).cpu()
            layers.append(dict(units=len(rate), dead=int((c == 0).sum()), rare=int((rate < rare).sum()),
                               median_rate=float(rate.median()), mean_rate=float(rate.mean())))
        out = dict(layers=layers, n_images=n,
                   weight_norms=[float(m.weight.norm()) for m in model.modules() if isinstance(m, nn.Linear)])
        if probe is not None:
            out["effective_rank"] = effective_rank(_preactivations(model, probe)[1])
    model.train(was_training)
    return out
