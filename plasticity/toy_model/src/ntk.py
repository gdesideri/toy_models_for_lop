"""
src/ntk.py — Empirical NTK computation utilities.

The empirical NTK between two inputs x_i and x_j is:

    K(x_i, x_j) = ∑_k  ∂f(x_i)/∂θ_k  ·  ∂f(x_j)/∂θ_k
                = J[i, :] · J[j, :]

where J is the (N, P) Jacobian matrix.

Per-layer NTK: contribution of each parameter group to the full NTK.
Full NTK = ∑_l  K^(l)   (sum over layers).

The NTK governs learning dynamics in function space:
    d r / d t = -η K r
where r = y - f(x) are the residuals.
Eigendecompose K = V Λ V^T: component of r along v_k decays at rate η λ_k.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Tuple


def _output_scalar(model: nn.Module, x_single: torch.Tensor) -> torch.Tensor:
    """Model output for a single input (squeeze to scalar)."""
    return model(x_single.unsqueeze(0)).squeeze()


def compute_jacobian(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Full Jacobian J[i, k] = ∂f(x_i)/∂θ_k.

    Args:
        model: nn.Module (output must be scalar per sample)
        x:     (N, d_in) input batch

    Returns:
        J: (N, P) Jacobian, P = total number of parameters
    """
    params = [p for p in model.parameters() if p.requires_grad]
    rows   = []
    for xi in x:
        grads = torch.autograd.grad(
            model(xi.unsqueeze(0)).squeeze(),
            params,
            create_graph=False,
            retain_graph=False,
        )
        rows.append(torch.cat([g.reshape(-1) for g in grads]))
    return torch.stack(rows)   # (N, P)


def compute_jacobian_per_layer(
    model: nn.Module, x: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Per-layer Jacobians: for each named parameter, J_l[i, k] = ∂f(x_i)/∂θ_k^l.

    Returns:
        dict mapping param name → (N, P_l) Jacobian
    """
    named_params = [(name, p) for name, p in model.named_parameters()
                    if p.requires_grad]
    layer_jacs   = {name: [] for name, _ in named_params}

    for xi in x:
        out   = model(xi.unsqueeze(0)).squeeze()
        grads = torch.autograd.grad(out, [p for _, p in named_params],
                                    create_graph=False, retain_graph=False)
        for (name, _), g in zip(named_params, grads):
            layer_jacs[name].append(g.reshape(-1))

    return {name: torch.stack(rows) for name, rows in layer_jacs.items()}


def jacobian_to_ntk(J: torch.Tensor) -> torch.Tensor:
    """NTK from a Jacobian: K = J @ J^T, shape (N, N)."""
    return J @ J.T


def compute_ntk(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Full empirical NTK matrix K[i, j] = ∇f(x_i)·∇f(x_j). Shape (N, N)."""
    J = compute_jacobian(model, x)
    return jacobian_to_ntk(J)


def compute_ntk_per_layer(
    model: nn.Module, x: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Per-layer NTK matrices.

    Returns:
        dict mapping param name → (N, N) NTK contribution
        The full NTK = sum of all values.
    """
    layer_jacs = compute_jacobian_per_layer(model, x)
    return {name: jacobian_to_ntk(J) for name, J in layer_jacs.items()}


def ntk_eigendecomposition(
    K: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Eigendecompose a symmetric NTK matrix.

    Returns:
        eigenvalues:  (N,) sorted descending
        eigenvectors: (N, N) columns are eigenvectors
    """
    # Use eigh for symmetric matrices (more stable)
    vals, vecs = torch.linalg.eigh(K)       # ascending
    idx        = torch.argsort(vals, descending=True)
    return vals[idx], vecs[:, idx]


def residual_alignment(
    residuals: torch.Tensor,
    eigenvectors: torch.Tensor,
) -> torch.Tensor:
    """
    Alignment of residual vector r with each NTK eigenvector.

    alignment_k = (v_k^T r)^2 / ||r||^2

    Args:
        residuals:    (N,) residual vector  y - f(x)
        eigenvectors: (N, N) columns = eigenvectors (descending eigenvalue order)

    Returns:
        alignments: (N,) alignment coefficient per eigenvector
    """
    r_norm_sq  = (residuals ** 2).sum() + 1e-12
    projections = (eigenvectors.T @ residuals) ** 2   # (N,) squared projections
    return projections / r_norm_sq
