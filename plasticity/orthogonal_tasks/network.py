"""
The network: D → width ReLU units (trainable hidden bias) → 1 logit (trainable output bias), float64,
BCE with logits. Built with models.factory ("shallow_<width>").

Init: the factory's uniform Kaiming weights; the hidden bias drawn from U(−1/√D, 1/√D) (or zero with
bias_init="zero"); then every unit silent on all train inputs is redrawn (weights only, same Kaiming
distribution) until it fires on at least one, so every unit starts alive.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from models.factory import build_model, build_optimizer
from orthogonal_tasks.config import Config, seed_for

DTYPE = torch.float64


def make_net(cfg: Config, seed: int, X: torch.Tensor) -> torch.nn.Module:
    """A net initialised from `seed` with every hidden unit active on some row of X; the number of
    redraws is kept in `net.init_redraws`."""
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        net = build_model(f"shallow_{cfg.width}", d_in=cfg.input_dim, hidden_bias=True, output_bias=True)
    net = net.to(DTYPE)
    if cfg.bias_init == "uniform":      # PyTorch's default bias range, drawn from its own stream
        g = torch.Generator().manual_seed(seed_for(seed, "bias-init"))
        bound = 1.0 / cfg.input_dim ** 0.5
        with torch.no_grad():
            net.fc1.bias.copy_((torch.rand(cfg.width, generator=g, dtype=DTYPE) * 2 - 1) * bound)
    net.init_redraws = redraw_silent_units(net, X, seed)
    return net


def redraw_silent_units(net, X: torch.Tensor, seed: int, max_tries: int = 100_000) -> int:
    """Redraw the weights of every unit silent on all rows of X, from the uniform Kaiming distribution
    (bound sqrt(6 / fan_in)), until it fires on at least one row. Active units are left alone. Returns
    the number of redraws."""
    W, b = net.fc1.weight, net.fc1.bias
    bound = (6.0 / W.shape[1]) ** 0.5
    g = torch.Generator().manual_seed(seed_for(seed, "live-init"))
    redraws = 0
    with torch.no_grad():
        for h in range(W.shape[0]):
            for _ in range(max_tries):
                if (X @ W[h] + b[h]).max() > 0:
                    break
                W[h] = (torch.rand(W.shape[1], generator=g, dtype=W.dtype) * 2 - 1) * bound
                redraws += 1
            else:
                raise RuntimeError(f"unit {h}: no active direction found in {max_tries} draws")
    return redraws


def make_optimizer(cfg: Config, net: torch.nn.Module) -> torch.optim.Optimizer:
    if cfg.optimizer == "sgd":
        return build_optimizer(net, "sgd", lr=cfg.lr, momentum=cfg.momentum)
    return build_optimizer(net, cfg.optimizer, lr=cfg.lr)


def params(net):
    """(W (H, D), b (H,), a (H,), c ()) as detached tensors."""
    return net.fc1.weight.detach(), net.fc1.bias.detach(), net.fc2.weight.detach()[0], net.fc2.bias.detach()[0]


def full_batch_grads(W, b, a, c, X, Y):
    """Exact gradients of the mean BCE of every task (rows of Y, (K, N)) at (W, b, a, c), with ReLU'(0) = 0
    as in autograd. Returns grad_W (K, H, D), grad_b (K, H), grad_a (K, H), grad_c (K,), loss (K,), acc (K,).
    grad_b[k, h] = a_h · mean over inputs of (p − y)·1[unit h active]: the push on unit h's bias."""
    P = X @ W.T + b
    Hid = torch.relu(P)
    f = Hid @ a + c
    R = (torch.sigmoid(f)[None, :] - Y) / X.shape[0]            # dL/df per task
    M = (P > 0).to(X.dtype)
    grad_a = R @ Hid
    grad_c = R.sum(1)
    grad_b = a[None, :] * (R @ M)
    grad_W = a[None, :, None] * torch.einsum("kn,nh,nd->khd", R, M, X)
    loss = torch.stack([F.binary_cross_entropy_with_logits(f, y) for y in Y])
    acc = ((f[None, :] > 0).to(Y.dtype) == Y).to(X.dtype).mean(1)
    return grad_W, grad_b, grad_a, grad_c, loss, acc


class BatchStream:
    """Minibatch indices of the fixed train set: reshuffled each epoch, from one seed."""

    def __init__(self, n: int, batch_size: int, seed: int):
        self.n, self.bs = n, batch_size
        self.g = torch.Generator().manual_seed(seed)
        self.perm, self.pos = torch.randperm(n, generator=self.g), 0

    def next(self) -> torch.Tensor:
        if self.pos + self.bs > self.n:
            self.perm, self.pos = torch.randperm(self.n, generator=self.g), 0
        idx = self.perm[self.pos:self.pos + self.bs]
        self.pos += self.bs
        return idx


def train_step(net, opt, xb, yb) -> float:
    """One optimizer step on a minibatch; returns its loss."""
    loss = F.binary_cross_entropy_with_logits(net(xb), yb)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()
