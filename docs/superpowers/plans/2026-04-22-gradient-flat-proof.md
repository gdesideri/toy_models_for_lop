# Gradient Cancellation Proof Notebook — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained Jupyter notebook that proves, both analytically and empirically, that an antipodal ReLU neuron pair has zero combined gradient under a symmetric distribution + even target, and that this collapses an m-neuron network's effective capacity to m-2.

**Architecture:** Single notebook `gradient_flat_proof.ipynb` with 5 sections: shared setup, target visualisation, Part A (algebraic proof + autograd), Part B (frozen pair + capacity), Part C (LoP replay). All cells are independent within each section; shared helpers defined in Section 0.

**Tech Stack:** Python 3, PyTorch (autograd), NumPy, Matplotlib. No training framework needed — raw Adam via `torch.optim`.

---

## File map

| File | Role |
|------|------|
| `gradient_flat_proof.ipynb` | The entire notebook — created from scratch |

---

## Task 1: Notebook skeleton + Section 0 (imports and helpers)

**Files:**
- Create: `gradient_flat_proof.ipynb`

- [ ] **Step 1: Create the notebook file**

Write the following JSON to `gradient_flat_proof.ipynb`. This creates the skeleton with all section header markdown cells plus one empty code cell per section. Subsequent tasks will fill in the code cells.

```json
{
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "cells": [
  {"cell_type": "markdown", "id": "s0-title", "metadata": {}, "source": ["# Gradient Cancellation Proof\n", "Proves that an antipodal ReLU pair has zero combined gradient under a symmetric distribution and even target."]},
  {"cell_type": "code",     "id": "s0-imports", "metadata": {}, "source": ["# Section 0 — imports and helpers (filled in Task 1)"], "outputs": [], "execution_count": null},
  {"cell_type": "markdown", "id": "s1-title", "metadata": {}, "source": ["## Section 1 — Target function"]},
  {"cell_type": "code",     "id": "s1-target", "metadata": {}, "source": ["# Section 1 (filled in Task 2)"], "outputs": [], "execution_count": null},
  {"cell_type": "markdown", "id": "s2-title", "metadata": {}, "source": ["## Part A — Algebraic proof + autograd verification"]},
  {"cell_type": "code",     "id": "s2-derive", "metadata": {}, "source": ["# Part A derivation cell (filled in Task 3)"], "outputs": [], "execution_count": null},
  {"cell_type": "code",     "id": "s2-sweep",  "metadata": {}, "source": ["# Part A sweep (filled in Task 4)"], "outputs": [], "execution_count": null},
  {"cell_type": "code",     "id": "s2-plot",   "metadata": {}, "source": ["# Part A plot (filled in Task 5)"], "outputs": [], "execution_count": null},
  {"cell_type": "markdown", "id": "s3-title", "metadata": {}, "source": ["## Part B — Frozen pair + capacity argument"]},
  {"cell_type": "code",     "id": "s3-net",    "metadata": {}, "source": ["# Part B network (filled in Task 6)"], "outputs": [], "execution_count": null},
  {"cell_type": "code",     "id": "s3-train",  "metadata": {}, "source": ["# Part B training (filled in Task 7)"], "outputs": [], "execution_count": null},
  {"cell_type": "code",     "id": "s3-counter","metadata": {}, "source": ["# Part B counterfactual (filled in Task 8)"], "outputs": [], "execution_count": null},
  {"cell_type": "code",     "id": "s3-plot",   "metadata": {}, "source": ["# Part B plot (filled in Task 9)"], "outputs": [], "execution_count": null},
  {"cell_type": "markdown", "id": "s4-title", "metadata": {}, "source": ["## Part C — Full LoP replay"]},
  {"cell_type": "code",     "id": "s4-load",   "metadata": {}, "source": ["# Part C loading (filled in Task 10)"], "outputs": [], "execution_count": null},
  {"cell_type": "code",     "id": "s4-trap",   "metadata": {}, "source": ["# Part C trap (filled in Task 11)"], "outputs": [], "execution_count": null},
  {"cell_type": "code",     "id": "s4-plot",   "metadata": {}, "source": ["# Part C plot (filled in Task 12)"], "outputs": [], "execution_count": null}
 ]
}
```

- [ ] **Step 2: Fill in cell `s0-imports` with all shared helpers**

Replace the source of cell `s0-imports` with:

```python
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

torch.manual_seed(42)
np.random.seed(42)
matplotlib.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

# ── Data ──────────────────────────────────────────────────────────────────────
def sample_uniform(n: int = 4096) -> torch.Tensor:
    """x ~ Uniform(-1, 1), shape (n,)."""
    return torch.empty(n).uniform_(-1, 1)

# ── Target y*(x) = |x| + relu(|x|-0.3) + relu(|x|-0.7) ──────────────────────
def target(x: torch.Tensor) -> torch.Tensor:
    ax = x.abs()
    return ax + torch.relu(ax - 0.3) + torch.relu(ax - 0.7)

# ── Fixed-neuron contribution (pairs 2 & 3 from target table) ─────────────────
_W_FIX = torch.tensor([ 1., -1.,  1., -1.])
_B_FIX = torch.tensor([-0.3, -0.3, -0.7, -0.7])
_V_FIX = torch.tensor([ 1.,  1.,  1.,  1.])

def f_fixed(x: torch.Tensor) -> torch.Tensor:
    """Contribution of the 4 analytically optimal neurons (no grad)."""
    pre = x.unsqueeze(1) * _W_FIX + _B_FIX   # (N, 4)
    return (torch.relu(pre) * _V_FIX).sum(1)   # (N,)

# ── Gradient norm helper ──────────────────────────────────────────────────────
def grad_norm(*tensors) -> float:
    return sum(t.grad.norm().item() for t in tensors if t.grad is not None)

print("Helpers ready.")
```

- [ ] **Step 3: Verify the notebook file is valid JSON**

```bash
python -c "import json; json.load(open('gradient_flat_proof.ipynb')); print('valid')"
```

Expected output: `valid`

- [ ] **Step 4: Commit**

```bash
git add gradient_flat_proof.ipynb
git commit -m "feat: notebook skeleton + Section 0 helpers"
```

---

## Task 2: Section 1 — Target function visualisation

**Files:**
- Modify: `gradient_flat_proof.ipynb` cell `s1-target`

- [ ] **Step 1: Fill in cell `s1-target`**

```python
x_plot = torch.linspace(-1, 1, 400)
y_plot = target(x_plot)

# Kink locations
kinks = [-0.7, -0.3, 0.0, 0.3, 0.7]

fig, ax = plt.subplots(figsize=(7, 3))
ax.plot(x_plot.numpy(), y_plot.numpy(), lw=2, label=r"$y^*(x)$")
for k in kinks:
    ax.axvline(k, color="gray", ls="--", lw=0.8, alpha=0.6)
ax.scatter(kinks, target(torch.tensor(kinks)).numpy(),
           zorder=5, color="tab:orange", label="kinks")
ax.set_xlabel("x")
ax.set_title(r"$y^*(x) = |x| + \mathrm{relu}(|x|-0.3) + \mathrm{relu}(|x|-0.7)$  — 5 kinks, even function")
ax.legend()
plt.tight_layout()
plt.savefig("fig_target.png", dpi=150)
plt.show()
print(f"y*(0.5)  = {target(torch.tensor(0.5)):.4f}")
print(f"y*(-0.5) = {target(torch.tensor(-0.5)):.4f}  (must match — even function)")
```

- [ ] **Step 2: Run cell and verify**

Execute the cell. Expected:
- Plot shows a W-shaped even function with 5 marked kink positions.
- Printed values `y*(0.5)` and `y*(-0.5)` must be identical.

- [ ] **Step 3: Commit**

```bash
git add gradient_flat_proof.ipynb fig_target.png
git commit -m "feat: Section 1 target visualisation"
```

---

## Task 3: Part A — Mathematical derivation (markdown cell)

**Files:**
- Modify: `gradient_flat_proof.ipynb` cell `s2-derive`

- [ ] **Step 1: Convert cell `s2-derive` to markdown and fill it**

Change `cell_type` of `s2-derive` to `"markdown"` and set its source to:

```markdown
### Derivation

For the antipodal pair $(w_+, w_- = -w_+)$ with $b_+=b_-=0$, $v_+=v_-=1$, the MSE loss is

$$L = \mathbb{E}_x\!\left[(f(x) - y^*(x))^2\right]$$

The gradient w.r.t. $w_+$:

$$\frac{\partial L}{\partial w_+}
= \mathbb{E}\!\left[2(f-y^*)\cdot\mathbf{1}[w_+ x>0]\cdot x\right]$$

The gradient w.r.t. $w_-$ (chain rule, $w_- = -a$):

$$\frac{\partial L}{\partial w_-}
= \mathbb{E}\!\left[2(f-y^*)\cdot\mathbf{1}[w_- x>0]\cdot x\right]
= \mathbb{E}\!\left[2(f-y^*)\cdot\mathbf{1}[-w_+ x>0]\cdot x\right]$$

Since $\mathbf{1}[w_+x>0]$ and $\mathbf{1}[-w_+x>0]$ are complementary indicators (one is 1 whenever the other is 0, a.s.), their sum equals 1. Therefore:

$$\frac{\partial L}{\partial w_+} + \frac{\partial L}{\partial w_-}
= \mathbb{E}\!\left[2(f(x)-y^*(x))\cdot x\right]$$

**Key observation:** The pair's output is $v(\mathrm{relu}(w_+x)+\mathrm{relu}(-w_+x)) = v\,w_+\,|x|$ — an **even** function. The 4 fixed neurons also produce an even function (symmetric pairs). So $f(x)$ is even. The target $y^*(x)$ is even. Therefore $f(x)-y^*(x)$ is even, i.e. $g(-x)=g(x)$.

Under **any** zero-mean symmetric distribution $p(x)=p(-x)$:

$$\mathbb{E}[g(x)\cdot x] = \int g(x)\,x\,p(x)\,dx = 0$$

because the integrand $g(x)\cdot x$ is an **odd** function (even × odd = odd) and integrates to zero over any symmetric domain.

**Conclusion:** $\nabla_{w_+}L + \nabla_{w_-}L = 0$ exactly, for any value of $a$, any symmetric distribution, and any even target.
```

- [ ] **Step 2: Commit**

```bash
git add gradient_flat_proof.ipynb
git commit -m "feat: Part A mathematical derivation cell"
```

---

## Task 4: Part A — Autograd sweep

**Files:**
- Modify: `gradient_flat_proof.ipynb` cell `s2-sweep`

- [ ] **Step 1: Fill in cell `s2-sweep`**

```python
# Part A — numerical verification of ∂L/∂w+ + ∂L/∂w- = 0
x_big = sample_uniform(n=500_000)   # large batch → low MC noise
y_big = target(x_big)

print(f"{'a':>6}  {'∂L/∂w+':>14}  {'∂L/∂w-':>14}  {'|sum|':>14}")
print("-" * 56)

a_sweep = [0.1, 0.3, 0.5, 0.7, 0.9]
sweep_results = []   # saved for the plot in Task 5

for a in a_sweep:
    w_pos = torch.tensor( a, requires_grad=True)
    w_neg = torch.tensor(-a, requires_grad=True)

    # Forward: antipodal pair + 4 fixed optimal neurons
    f_pair = torch.relu(w_pos * x_big) + torch.relu(w_neg * x_big)  # (N,)
    f_all  = f_pair + f_fixed(x_big)                                 # (N,)
    loss   = ((f_all - y_big) ** 2).mean()

    gp, gn = torch.autograd.grad(loss, [w_pos, w_neg])
    gsum = (gp + gn).abs().item()
    sweep_results.append((a, gp.item(), gn.item(), gsum))

    print(f"{a:>6.1f}  {gp.item():>14.8f}  {gn.item():>14.8f}  {gsum:>14.2e}")

print()
all_pass = all(r[3] < 1e-12 for r in sweep_results)
print(f"All |sum| < 1e-12: {'✓ PASS' if all_pass else '✗ FAIL'}")
assert all_pass, "Gradient sum is not machine zero — check implementation."
```

- [ ] **Step 2: Run the cell and verify**

Expected output:
```
     a      ∂L/∂w+      ∂L/∂w-           |sum|
--------------------------------------------------------
   0.1    ...          ...          <1e-12
   0.3    ...          ...          <1e-12
   ...
All |sum| < 1e-12: ✓ PASS
```
The two gradient columns must be exact negatives of each other.

- [ ] **Step 3: Commit**

```bash
git add gradient_flat_proof.ipynb
git commit -m "feat: Part A autograd sweep — gradient sum verified machine zero"
```

---

## Task 5: Part A — Plots

**Files:**
- Modify: `gradient_flat_proof.ipynb` cell `s2-plot`

- [ ] **Step 1: Fill in cell `s2-plot`**

```python
# Three-panel plot:
# (left)  ∂L/∂w+ and ∂L/∂w- vs a  (mirror images)
# (mid)   (f(x)-y*(x)) vs x        (even function)
# (right) (f(x)-y*(x))·x vs x      (odd function, expectation = 0)

a_vals = [r[0] for r in sweep_results]
gp_vals = [r[1] for r in sweep_results]
gn_vals = [r[2] for r in sweep_results]

# Use a=0.5 for the function plots
a_demo = 0.5
x_p = torch.linspace(-1, 1, 600)
f_demo = (torch.relu(a_demo * x_p) + torch.relu(-a_demo * x_p)
          + f_fixed(x_p)).detach()
err    = f_demo - target(x_p)
err_x  = err * x_p

fig, axes = plt.subplots(1, 3, figsize=(14, 4))

# Left
axes[0].plot(a_vals, gp_vals, "o-", label=r"$\partial L/\partial w_+$")
axes[0].plot(a_vals, gn_vals, "s--", label=r"$\partial L/\partial w_-$")
axes[0].axhline(0, color="k", lw=0.8, ls=":")
axes[0].set_xlabel("a")
axes[0].set_title("Gradients are exact mirrors")
axes[0].legend()

# Mid
axes[1].plot(x_p.numpy(), err.numpy(), lw=2, color="tab:orange")
axes[1].axhline(0, color="k", lw=0.8, ls=":")
axes[1].set_xlabel("x")
axes[1].set_title(r"$f(x)-y^*(x)$ — even function  $[a=0.5]$")

# Right
axes[2].plot(x_p.numpy(), err_x.numpy(), lw=2, color="tab:green")
axes[2].axhline(0, color="k", lw=0.8, ls=":")
axes[2].axhline(err_x.mean().item(), color="tab:red", ls="--",
                label=f"mean = {err_x.mean().item():.2e}")
axes[2].set_xlabel("x")
axes[2].set_title(r"$(f(x)-y^*(x))\cdot x$ — odd → $\mathbb{E}[\cdot]=0$")
axes[2].legend(fontsize=10)

plt.tight_layout()
plt.savefig("fig_partA.png", dpi=150)
plt.show()
```

- [ ] **Step 2: Run and verify**

Expected:
- Left panel: two curves that are perfect reflections across the x-axis.
- Middle panel: symmetric W-shaped curve (even function).
- Right panel: antisymmetric curve crossing zero, mean label ≈ 0 (< 1e-5).

- [ ] **Step 3: Commit**

```bash
git add gradient_flat_proof.ipynb fig_partA.png
git commit -m "feat: Part A three-panel plot"
```

---

## Task 6: Part B — Network class

**Files:**
- Modify: `gradient_flat_proof.ipynb` cell `s3-net`

- [ ] **Step 1: Fill in cell `s3-net`**

```python
class SixNeuronNet(torch.nn.Module):
    """
    6-neuron 1-D ReLU network.
    Neurons 0-1 can be frozen as the antipodal pair.
    Neurons 2-5 are always learnable.
    """
    def __init__(self, freeze_pair: bool = True):
        super().__init__()
        self.freeze_pair = freeze_pair

        # Antipodal pair (neurons 0-1) — always initialised at (0.5, -0.5)
        w_pair = torch.tensor([0.5, -0.5])
        b_pair = torch.tensor([0.0,  0.0])
        v_pair = torch.tensor([1.0,  1.0])

        if freeze_pair:
            self.register_buffer("w_pair", w_pair)
            self.register_buffer("b_pair", b_pair)
            self.register_buffer("v_pair", v_pair)
        else:
            self.w_pair = torch.nn.Parameter(w_pair.clone())
            self.b_pair = torch.nn.Parameter(b_pair.clone())
            self.v_pair = torch.nn.Parameter(v_pair.clone())

        # Free neurons (2-5) — random Kaiming init
        self.w_free = torch.nn.Parameter(torch.randn(4) * (2 / 4) ** 0.5)
        self.b_free = torch.nn.Parameter(torch.zeros(4))
        self.v_free = torch.nn.Parameter(torch.randn(4) * (2 / 4) ** 0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre_p = x.unsqueeze(1) * self.w_pair + self.b_pair  # (N, 2)
        f_p   = (torch.relu(pre_p) * self.v_pair).sum(1)    # (N,)
        pre_f = x.unsqueeze(1) * self.w_free + self.b_free  # (N, 4)
        f_f   = (torch.relu(pre_f) * self.v_free).sum(1)    # (N,)
        return f_p + f_f

    @torch.no_grad()
    def pair_grad_norm(self, x: torch.Tensor, y: torch.Tensor) -> float:
        """Compute gradient norm for the pair without updating anything."""
        w = self.w_pair.clone().requires_grad_(True)
        b = self.b_pair.clone().requires_grad_(True)
        v = self.v_pair.clone().requires_grad_(True)
        pre = x.unsqueeze(1) * w + b
        f   = (torch.relu(pre) * v).sum(1) + (
               torch.relu(x.unsqueeze(1) * self.w_free + self.b_free)
               * self.v_free).sum(1)
        loss = ((f - y) ** 2).mean()
        grads = torch.autograd.grad(loss, [w, b, v])
        return sum(g.norm().item() for g in grads)

# Quick sanity check
_net = SixNeuronNet(freeze_pair=True)
_x   = sample_uniform(16)
_out = _net(_x)
assert _out.shape == (16,), f"Expected shape (16,), got {_out.shape}"
print("SixNeuronNet forward pass OK. Output shape:", _out.shape)
```

- [ ] **Step 2: Run and verify**

Expected: `SixNeuronNet forward pass OK. Output shape: torch.Size([16])`

- [ ] **Step 3: Commit**

```bash
git add gradient_flat_proof.ipynb
git commit -m "feat: Part B SixNeuronNet with frozen-pair support"
```

---

## Task 7: Part B — Training loop

**Files:**
- Modify: `gradient_flat_proof.ipynb` cell `s3-train`

- [ ] **Step 1: Fill in cell `s3-train`**

```python
STEPS_B   = 10_000
LR_B      = 1e-3
LOG_EVERY = 200

net_b   = SixNeuronNet(freeze_pair=True)
opt_b   = torch.optim.Adam(
    [net_b.w_free, net_b.b_free, net_b.v_free], lr=LR_B
)

log_step_b, log_loss_b = [], []
log_grad_pair_b, log_grad_free_b = [], []

for step in range(STEPS_B):
    x = sample_uniform()
    y = target(x)

    loss = ((net_b(x) - y) ** 2).mean()
    opt_b.zero_grad()
    loss.backward()

    # Gradient norm of free neurons (before step)
    gf = (net_b.w_free.grad.norm()
        + net_b.b_free.grad.norm()
        + net_b.v_free.grad.norm()).item()

    opt_b.step()

    if step % LOG_EVERY == 0:
        with torch.no_grad():
            x_eval = sample_uniform(50_000)
            eval_loss = ((net_b(x_eval) - target(x_eval)) ** 2).mean().item()
        gp = net_b.pair_grad_norm(x, y)
        log_step_b.append(step)
        log_loss_b.append(eval_loss)
        log_grad_pair_b.append(gp)
        log_grad_free_b.append(gf)

print(f"Final loss (frozen pair): {log_loss_b[-1]:.6f}")
print(f"Final pair gradient norm: {log_grad_pair_b[-1]:.2e}")
assert log_loss_b[-1] > 0.01, "Loss should plateau above 0.01 — 4 neurons cannot fit 5 kinks"
assert log_grad_pair_b[-1] < 1e-6, "Pair gradient norm should be machine zero"
print("✓ PASS: loss plateau verified, pair gradient verified flat")
```

- [ ] **Step 2: Run and verify**

Expected:
- `Final loss (frozen pair)` > 0.01 (training stagnates)
- `Final pair gradient norm` < 1e-6
- Prints `✓ PASS`

- [ ] **Step 3: Commit**

```bash
git add gradient_flat_proof.ipynb
git commit -m "feat: Part B frozen-pair training loop with gradient tracking"
```

---

## Task 8: Part B — Counterfactual (unfreeze and converge)

**Files:**
- Modify: `gradient_flat_proof.ipynb` cell `s3-counter`

- [ ] **Step 1: Fill in cell `s3-counter`**

```python
STEPS_CF = 2_000

# Clone the frozen-pair net into a fully trainable one, copying all weights
net_cf = SixNeuronNet(freeze_pair=False)
with torch.no_grad():
    net_cf.w_pair.copy_(net_b.w_pair)
    net_cf.b_pair.copy_(net_b.b_pair)
    net_cf.v_pair.copy_(net_b.v_pair)
    net_cf.w_free.copy_(net_b.w_free)
    net_cf.b_free.copy_(net_b.b_free)
    net_cf.v_free.copy_(net_b.v_free)

opt_cf = torch.optim.Adam(net_cf.parameters(), lr=LR_B)

log_step_cf, log_loss_cf = [], []

for step in range(STEPS_CF):
    x = sample_uniform()
    y = target(x)
    loss = ((net_cf(x) - y) ** 2).mean()
    opt_cf.zero_grad()
    loss.backward()
    opt_cf.step()

    if step % 100 == 0:
        with torch.no_grad():
            x_eval = sample_uniform(50_000)
            eval_loss = ((net_cf(x_eval) - target(x_eval)) ** 2).mean().item()
        log_step_cf.append(STEPS_B + step)   # offset for plotting
        log_loss_cf.append(eval_loss)

final_cf_loss = log_loss_cf[-1]
print(f"Counterfactual final loss: {final_cf_loss:.2e}")
assert final_cf_loss < 1e-4, f"Expected near-zero loss after unfreeze, got {final_cf_loss}"
print("✓ PASS: unfreezing allows loss to collapse — stagnation was capacity-driven")
```

- [ ] **Step 2: Run and verify**

Expected:
- `Counterfactual final loss` < 1e-4
- Prints `✓ PASS`

- [ ] **Step 3: Commit**

```bash
git add gradient_flat_proof.ipynb
git commit -m "feat: Part B counterfactual — unfreeze confirms capacity-driven stagnation"
```

---

## Task 9: Part B — 4-panel plot

**Files:**
- Modify: `gradient_flat_proof.ipynb` cell `s3-plot`

- [ ] **Step 1: Fill in cell `s3-plot`**

```python
x_p = torch.linspace(-1, 1, 600)
y_p = target(x_p)

with torch.no_grad():
    f_plateau = net_b(x_p)
    f_cf      = net_cf(x_p)

fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# (a) loss over steps
ax_a = fig.add_subplot(gs[0, 0])
ax_a.semilogy(log_step_b, log_loss_b, label="frozen pair")
ax_a.semilogy(log_step_cf, log_loss_cf, label="unfrozen (CF)", ls="--")
ax_a.axvline(STEPS_B, color="k", ls=":", lw=1, label="unfreeze point")
ax_a.set_xlabel("step")
ax_a.set_ylabel("MSE loss")
ax_a.set_title("(a) Loss — plateau then collapse")
ax_a.legend(fontsize=9)

# (b) gradient norms
ax_b = fig.add_subplot(gs[0, 1])
ax_b.semilogy(log_step_b, log_grad_pair_b, label="pair grad norm")
ax_b.semilogy(log_step_b, log_grad_free_b, label="free grad norm")
ax_b.set_xlabel("step")
ax_b.set_ylabel("gradient norm")
ax_b.set_title("(b) Gradient norms — pair is machine zero")
ax_b.legend(fontsize=9)

# (c) fit at plateau
ax_c = fig.add_subplot(gs[1, 0])
ax_c.plot(x_p.numpy(), y_p.numpy(), "k-", lw=2, label=r"$y^*$")
ax_c.plot(x_p.numpy(), f_plateau.numpy(), "tab:orange", ls="--", label="4-neuron fit")
ax_c.set_xlabel("x")
ax_c.set_title("(c) Fit at plateau — 4 neurons miss the 5th kink")
ax_c.legend(fontsize=9)

# (d) fit after unfreeze
ax_d = fig.add_subplot(gs[1, 1])
ax_d.plot(x_p.numpy(), y_p.numpy(), "k-", lw=2, label=r"$y^*$")
ax_d.plot(x_p.numpy(), f_cf.numpy(), "tab:green", ls="--", label="6-neuron fit")
ax_d.set_xlabel("x")
ax_d.set_title("(d) Fit after unfreeze — near-perfect")
ax_d.legend(fontsize=9)

plt.suptitle("Part B — Frozen Pair + Capacity Argument", fontweight="bold")
plt.savefig("fig_partB.png", dpi=150)
plt.show()
```

- [ ] **Step 2: Run and verify**

Expected:
- Panel (c) has a visible mismatch near one of the outer kinks.
- Panel (d) is near-indistinguishable from the target.

- [ ] **Step 3: Commit**

```bash
git add gradient_flat_proof.ipynb fig_partB.png
git commit -m "feat: Part B 4-panel plot"
```

---

## Task 10: Part C — Loading phases

**Files:**
- Modify: `gradient_flat_proof.ipynb` cell `s4-load`

- [ ] **Step 1: Fill in cell `s4-load`**

```python
STEPS_LOAD = 3_000
LR_C       = 1e-3

# Fresh 6-neuron network, all free
class FlexNet(torch.nn.Module):
    """6-neuron net with all parameters learnable."""
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.randn(6) * 0.3)
        self.b = torch.nn.Parameter(torch.zeros(6))
        self.v = torch.nn.Parameter(torch.randn(6) * 0.3)

    def forward(self, x):
        pre = x.unsqueeze(1) * self.w + self.b   # (N, 6)
        return (torch.relu(pre) * self.v).sum(1)  # (N,)

net_c = FlexNet()
opt_c = torch.optim.Adam(net_c.parameters(), lr=LR_C)

# ── Phase 1: y_pos = relu(x) ──────────────────────────────────────────────────
log_c_step, log_c_loss, log_c_w = [], [], []   # log w values for trajectory

for step in range(STEPS_LOAD):
    x  = sample_uniform()
    y  = torch.relu(x)
    loss = ((net_c(x) - y) ** 2).mean()
    opt_c.zero_grad(); loss.backward(); opt_c.step()
    if step % 50 == 0:
        log_c_step.append(step)
        log_c_loss.append(loss.item())
        log_c_w.append(net_c.w.detach().clone())

phase1_end = len(log_c_step)
print(f"Phase 1 done. w values: {net_c.w.detach().numpy().round(3)}")

# ── Phase 2: y_neg = relu(-x) ─────────────────────────────────────────────────
for step in range(STEPS_LOAD):
    x  = sample_uniform()
    y  = torch.relu(-x)
    loss = ((net_c(x) - y) ** 2).mean()
    opt_c.zero_grad(); loss.backward(); opt_c.step()
    if step % 50 == 0:
        log_c_step.append(STEPS_LOAD + step)
        log_c_loss.append(loss.item())
        log_c_w.append(net_c.w.detach().clone())

phase2_end = len(log_c_step)
print(f"Phase 2 done. w values: {net_c.w.detach().numpy().round(3)}")

# ── Identify organic antipodal pair ───────────────────────────────────────────
w_final = net_c.w.detach()
idx_pos  = w_final.argmax().item()
idx_neg  = w_final.argmin().item()
w_max    = w_final[idx_pos].item()
w_min    = w_final[idx_neg].item()
asymmetry = abs(w_max + w_min) / (abs(w_max) + 1e-8)
print(f"Antipodal pair: neuron {idx_pos} (w={w_max:.3f}), neuron {idx_neg} (w={w_min:.3f})")
print(f"Antipodal asymmetry ratio: {asymmetry:.3f}  (< 0.5 = roughly antipodal)")
```

- [ ] **Step 2: Run and verify**

Expected:
- After Phase 1, some `w` values should be positive (> 0.1).
- After Phase 2, some `w` values should be negative (< -0.1).
- Asymmetry ratio < 0.5.

- [ ] **Step 3: Commit**

```bash
git add gradient_flat_proof.ipynb
git commit -m "feat: Part C loading phases + antipodal pair identification"
```

---

## Task 11: Part C — Trap phase + baseline

**Files:**
- Modify: `gradient_flat_proof.ipynb` cell `s4-trap`

- [ ] **Step 1: Fill in cell `s4-trap`**

```python
STEPS_TRAP = 3_000

# ── Baseline: fresh net trained directly on y* ────────────────────────────────
net_baseline = FlexNet()
opt_base     = torch.optim.Adam(net_baseline.parameters(), lr=LR_C)
log_base_step, log_base_loss = [], []

for step in range(STEPS_TRAP):
    x = sample_uniform()
    y = target(x)
    loss = ((net_baseline(x) - y) ** 2).mean()
    opt_base.zero_grad(); loss.backward(); opt_base.step()
    if step % 50 == 0:
        log_base_step.append(step)
        log_base_loss.append(loss.item())

print(f"Baseline final loss: {log_base_loss[-1]:.4f}")

# ── Trap phase: loaded net on y* ─────────────────────────────────────────────
log_grad_c   = {i: [] for i in range(6)}   # per-neuron gradient norms
log_trap_step, log_trap_loss = [], []

for step in range(STEPS_TRAP):
    x = sample_uniform()
    y = target(x)
    loss = ((net_c(x) - y) ** 2).mean()
    opt_c.zero_grad()
    loss.backward()

    # Per-neuron gradient norm (w component only — most informative)
    for i in range(6):
        log_grad_c[i].append(net_c.w.grad[i].abs().item())

    opt_c.step()
    if step % 50 == 0:
        log_trap_step.append(2 * STEPS_LOAD + step)
        log_trap_loss.append(loss.item())

print(f"Loaded net final loss (Phase 3): {log_trap_loss[-1]:.4f}")
print(f"Baseline final loss:             {log_base_loss[-1]:.4f}")

trap_steps_logged = list(range(0, STEPS_TRAP, 1))   # all steps
pair_grad_trap  = [(log_grad_c[idx_pos][s] + log_grad_c[idx_neg][s]) / 2
                   for s in range(STEPS_TRAP)]
print(f"Mean |pair grad| in Phase 3: {np.mean(pair_grad_trap):.2e}  (should be < 1e-4)")
```

- [ ] **Step 2: Run and verify**

Expected:
- Loaded net Phase 3 final loss > baseline final loss (loaded net stagnates more).
- Mean pair grad < 1e-4.

- [ ] **Step 3: Commit**

```bash
git add gradient_flat_proof.ipynb
git commit -m "feat: Part C trap phase + baseline comparison"
```

---

## Task 12: Part C — 4-panel plot

**Files:**
- Modify: `gradient_flat_proof.ipynb` cell `s4-plot`

- [ ] **Step 1: Fill in cell `s4-plot`**

```python
x_p = torch.linspace(-1, 1, 600)
y_p = target(x_p)
with torch.no_grad():
    f_loaded   = net_c(x_p)
    f_baseline = net_baseline(x_p)

# Weight trajectory: stack log_c_w into (T, 6)
w_traj = torch.stack(log_c_w).numpy()   # (T, 6)

fig = plt.figure(figsize=(14, 9))
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

# (a) loss across all phases
ax_a = fig.add_subplot(gs[0, 0])
all_loaded_steps = log_c_step[:phase1_end] + log_c_step[phase1_end:phase2_end] + log_trap_step
all_loaded_loss  = (log_c_loss[:phase1_end]
                    + log_c_loss[phase1_end:phase2_end]
                    + log_trap_loss)
ax_a.semilogy(all_loaded_steps, all_loaded_loss, label="loaded net")
baseline_offset = [2 * STEPS_LOAD + s for s in log_base_step]
ax_a.semilogy(baseline_offset, log_base_loss, label="fresh net (baseline)", ls="--")
ax_a.axvline(STEPS_LOAD,       color="gray", ls=":", lw=1)
ax_a.axvline(2 * STEPS_LOAD,   color="k",    ls=":", lw=1, label="phase boundaries")
ax_a.set_xlabel("step")
ax_a.set_ylabel("MSE loss")
ax_a.set_title("(a) Loss across all phases")
ax_a.legend(fontsize=9)
ax_a.text(STEPS_LOAD / 2,    ax_a.get_ylim()[1] * 0.5, "Phase 1\nrelu(x)",   ha="center", fontsize=8)
ax_a.text(1.5 * STEPS_LOAD,  ax_a.get_ylim()[1] * 0.5, "Phase 2\nrelu(-x)",  ha="center", fontsize=8)

# (b) per-neuron gradient norms in Phase 3
ax_b = fig.add_subplot(gs[0, 1])
steps_trap_all = list(range(STEPS_TRAP))
for i in range(6):
    lw   = 2.5 if i in (idx_pos, idx_neg) else 0.8
    ls   = "-"  if i in (idx_pos, idx_neg) else "--"
    lbl  = f"neuron {i} (pair)" if i in (idx_pos, idx_neg) else f"neuron {i}"
    ax_b.semilogy(steps_trap_all, log_grad_c[i], lw=lw, ls=ls, label=lbl)
ax_b.set_xlabel("Phase-3 step")
ax_b.set_ylabel("|∂L/∂w_k|")
ax_b.set_title("(b) Per-neuron gradient in Phase 3")
ax_b.legend(fontsize=7, ncol=2)

# (c) weight trajectory for the antipodal pair
ax_c = fig.add_subplot(gs[1, 0])
all_steps_traj = log_c_step[:phase2_end]
ax_c.plot(all_steps_traj, w_traj[:phase2_end, idx_pos], label=f"w[{idx_pos}] (→ +)", lw=2)
ax_c.plot(all_steps_traj, w_traj[:phase2_end, idx_neg], label=f"w[{idx_neg}] (→ -)", lw=2)
ax_c.axvline(STEPS_LOAD, color="gray", ls=":", lw=1)
ax_c.axhline(0, color="k", lw=0.8, ls=":")
ax_c.set_xlabel("step")
ax_c.set_ylabel("weight value")
ax_c.set_title("(c) Weight trajectory of the antipodal pair")
ax_c.legend(fontsize=9)

# (d) final function fit
ax_d = fig.add_subplot(gs[1, 1])
ax_d.plot(x_p.numpy(), y_p.numpy(),           "k-",  lw=2,   label=r"$y^*$")
ax_d.plot(x_p.numpy(), f_loaded.numpy(),      "tab:orange", ls="--", lw=1.5, label="loaded net")
ax_d.plot(x_p.numpy(), f_baseline.numpy(),    "tab:green",  ls=":",  lw=1.5, label="fresh net")
ax_d.set_xlabel("x")
ax_d.set_title("(d) Final fit — loaded vs fresh")
ax_d.legend(fontsize=9)

plt.suptitle("Part C — Full LoP Replay", fontweight="bold")
plt.savefig("fig_partC.png", dpi=150)
plt.show()
```

- [ ] **Step 2: Run and verify**

Expected:
- Panel (a): loaded net loss stagnates in Phase 3; baseline converges faster.
- Panel (b): antipodal pair neurons have consistently lower gradient than the others.
- Panel (c): two neurons diverge in opposite directions across the loading phases.
- Panel (d): fresh network fit is visibly closer to $y^*$ than the loaded network.

- [ ] **Step 3: Final commit**

```bash
git add gradient_flat_proof.ipynb fig_partC.png
git commit -m "feat: Part C 4-panel plot — LoP replay complete"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task that covers it |
|-----------------|---------------------|
| Even target $y^*$ with 5 kinks requiring 6 neurons | Task 2 |
| Part A: derive gradient sum identity in markdown | Task 3 |
| Part A: autograd sweep over $a$ values, sum < 1e-12 | Task 4 |
| Part A: mirror-image gradient plot + odd-function plot | Task 5 |
| Part B: 6-neuron network, 2 frozen as antipodal pair | Task 6 |
| Part B: training loop tracking loss + gradient norms | Task 7 |
| Part B: loss plateau > 0.01, pair grad < 1e-6 | Task 7 |
| Part B: counterfactual unfreeze → loss < 1e-4 | Task 8 |
| Part B: 4-panel plot | Task 9 |
| Part C: Phase 1 relu(x) loading | Task 10 |
| Part C: Phase 2 relu(-x) loading | Task 10 |
| Part C: antipodal pair identification | Task 10 |
| Part C: Phase 3 trap on even target | Task 11 |
| Part C: baseline comparison | Task 11 |
| Part C: 4-panel plot with all panels | Task 12 |

**Placeholder scan:** All code cells contain complete runnable code. No TBDs. ✓

**Type consistency:**
- `sample_uniform(n)` → `torch.Tensor` shape `(n,)` used consistently throughout.
- `target(x)` → same shape as input, used in all tasks.
- `f_fixed(x)` → same shape, used in Task 4 and defined in Task 1.
- `SixNeuronNet` → `forward(x)` returns `(N,)`, used in Tasks 7-9.
- `FlexNet` → same interface, used in Tasks 10-12.
- `log_c_step`, `log_c_w` defined in Task 10 and used in Task 12. ✓
- `idx_pos`, `idx_neg` defined in Task 10 and used in Tasks 11-12. ✓
- `log_grad_c` defined in Task 11 and used in Task 12. ✓
- `phase1_end`, `phase2_end` defined in Task 10 and used in Task 12. ✓
