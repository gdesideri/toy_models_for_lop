"""
experiments/run_factor_rule_stream.py — Continual learning of a changing decision rule
on a fixed representation.

    1. pretrain   a 3-hidden-layer MLP (12288 -> n -> n -> n, ReLU) on Shapes3D with six
                  temporary factor heads (one Linear(n, 57) = six Linear(n, K_f) blocks),
                  loss = sum of the six cross entropies, plain SGD
    2. freeze     keep the body z = phi(x) (output of the last ReLU), drop the heads,
                  requires_grad=False, eval mode; cache z for train / val / test
    3. probe      one linear probe per factor on the frozen z (diagnostic only)
    4. stream     ONE Linear(n, 2) head and ONE SGD optimizer, built once before task 0
                  and carried through all 800 tasks; task t is labelled by the upper half
                  of factor t % 6 (tasks.factor_rules). After every task the head is
                  evaluated under all six rules on the test split.

Only the label rule changes between tasks: the images, their split, and z are fixed.

Variants:
    --encoder_from <run>/checkpoints/encoder.pt   skip stage 1 and reuse that run's pretrained
                  MLP (same seed, so same split; checked against its stored head accuracies).
                  The file is copied into the new run and never written to.
    --unfreeze_layers K   only the first 3 - K hidden layers are frozen and cached; copies of
                  the top K (Linear + ReLU each) are trained with the head by the same SGD
                  optimizer (--lr_head) through all 800 tasks. K = 1 (alias
                  --unfreeze_last_layer) trains the layer feeding the head; K = 3 the full
                  network, on cached flattened images. The adapted MLP is saved separately.

Run with:
    uv run python -m experiments.run_factor_rule_stream --n 256 --pretrain-epochs 20 \
        --lr-pretrain 0.01 --lr-head 0.01 --steps-per-task 500 --batch_size 256 --seed 0

Writes results/shapes3d_factor_rule_stream/factorrule_shapes3d_mlp_3x<n>_perm000_seed<k>/:
    config.json                         the arguments
    checkpoints/encoder.pt              the pretrained MLP (body = the frozen encoder; its head
                                        = the six temporary factor heads, not used afterwards)
    pretrain/                           history.json, metrics.csv (test_* columns = validation split)
    representation_probe_results.csv    per factor: train / val / test loss and accuracy
    continual_training.csv              one row per SGD step of the head
    task_end_metrics.csv                one row per task
    all_rule_evaluation.csv             800 x 6 rows: the head after each task, under each rule
    checkpoints/head_final.pt           the head and its optimizer after task 799
  with --unfreeze_layers K > 0, also:
    checkpoints/encoder_adapted.pt      the pretrained MLP with its top K hidden layers adapted
    checkpoints/trainable_layers_final.pt   the adapted top K hidden layers alone
    last_layer_trajectory.pt            weight and bias of the last hidden layer at the end of every task
    representation_probe_results_after_stream.csv   probes on the adapted representation
    head_trajectory.pt                  the head's weight and bias at the end of every task
    summary.json                        split sizes, pretraining / probe results, sanity checks
    *.png                               figures (visualization_scripts/factor_rule_stream.py)
"""
import copy
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.args import parse_factor_rule_stream_args
from data.factory import (build_dataset, build_eval_loader, build_loaders, build_tensor_loader,
                          relabel_dataset)
from data.download import SHAPES3D_FACTOR_N_VALUES
from tasks.factor_rules import (FACTORS, N_FACTORS, FactorCrossEntropy, all_task_labels,
                                task_distance, task_factor, task_schedule)
from models.factory import build_model, build_optimizer
from utils.checkpoints import load_model_from_checkpoint, save_checkpoint
from utils.evaluation import evaluate, extract_features
from utils.results import CsvLog, run_dir_for, save_csv, save_results, save_summary
from experiments.run_basic_task_training import train

NUM_TASKS = 800
IMAGE_SHAPE = (64, 64, 3)
D_IN = 64 * 64 * 3


# ── setup ─────────────────────────────────────────────────────────────────────

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return np.random.default_rng(seed)


def load_splits(params, rng):
    """
    train / val / test Shapes3D splits, labelled by all six factor indices, images in [0, 1].
    The repo's split (get_shapes3d) gives train / test; val is carved out of train with
    the same rng. These three splits are used by every phase.
    """
    train_set, test_set = build_dataset("shapes3d", n_samples=params.n_samples, target_factor=None,
                                        rng=rng, root=params.data_path, normalize=False)
    perm = rng.permutation(len(train_set))
    n_val = int(round(len(train_set) * params.val_frac))
    val_set, train_set = train_set.subset(np.sort(perm[:n_val])), train_set.subset(np.sort(perm[n_val:]))
    return train_set, val_set, test_set


def check_dataset(splits):
    """Sanity check 1: image shape, six factors, their cardinalities, and [0, 1] inputs."""
    for name, ds in splits.items():
        assert ds.images.shape[1:] == IMAGE_SHAPE, f"{name}: images {ds.images.shape[1:]}"
        assert ds.latent_idx.shape[1] == N_FACTORS, f"{name}: {ds.latent_idx.shape[1]} factors"
    for f, k in enumerate(SHAPES3D_FACTOR_N_VALUES):
        values = np.unique(splits["train"].latent_idx[:, f])
        assert np.array_equal(values, np.arange(k)), f"{FACTORS[f]}: values {values}, expected 0..{k - 1}"
    x, y = splits["train"][0]
    assert x.shape == (3, 64, 64) and 0.0 <= float(x.min()) and float(x.max()) <= 1.0
    assert y.shape == (N_FACTORS,)
    return {"image_shape": list(IMAGE_SHAPE), "d_in": D_IN,
            "factor_cardinalities": dict(zip(FACTORS, SHAPES3D_FACTOR_N_VALUES)),
            "split_sizes": {k: len(v) for k, v in splits.items()}}


# ── stage 1: representation pretraining ───────────────────────────────────────

def pretrain_representation(params, splits, device, run_dir):
    """
    The MLP with a Linear(n, 57) head = the six temporary factor heads, trained jointly on
    the summed factor cross entropies with the repo's train() (plain SGD, momentum 0).
    """
    criterion = FactorCrossEntropy()
    model_kwargs = dict(arch=params.model_kind, d_in=D_IN, n_classes=criterion.n_outputs,
                        activation="relu", use_layer_norm=False)
    model = build_model(**model_kwargs)

    train_loader, val_loader = build_loaders(splits["train"], splits["val"], params.batch_size,
                                             params.num_workers)
    print(f"── stage 1: pretraining {params.model_kind} on all six factors "
          f"({params.num_epochs} epochs, SGD lr {params.learning_rate}; 'test' below = validation)")
    history = train(model, train_loader, val_loader, params, criterion=criterion)
    save_results(run_dir / "pretrain", params, history, model, np.arange(0), save_model=False,
                 extra={"monitor_split": "val", "accuracy": "mean over the six factors"})

    per_factor = factor_head_metrics(model, splits, params, device)
    for f in FACTORS:
        print(f"  pretraining head {f:15s} test acc {per_factor['test'][f]['acc']:.4f}")

    save_checkpoint(run_dir / "checkpoints" / "encoder.pt", model, model_kwargs=model_kwargs,
                    stage="pretrained", encoder="model.body (+ model.flatten)",
                    head="six temporary factor heads, logits blocks of " + str(SHAPES3D_FACTOR_N_VALUES),
                    factor_names=FACTORS, per_factor=per_factor)
    return model, per_factor, model_kwargs


def factor_head_metrics(model, splits, params, device):
    """Per-factor loss / accuracy of the six temporary heads on each split."""
    criterion = FactorCrossEntropy()
    per_factor = {}
    model.eval()
    for split in ("train", "val", "test"):
        sums = np.zeros((N_FACTORS, 2))
        with torch.no_grad():
            for x, y in build_eval_loader(splits[split], params.batch_size, params.num_workers):
                x, y = x.to(device), y.to(device)
                for f, (loss, correct) in enumerate(criterion.per_factor(model(x), y)):
                    sums[f] += (loss.item() * len(y), correct)
        per_factor[split] = {FACTORS[f]: {"loss": sums[f, 0] / len(splits[split]),
                                          "acc": sums[f, 1] / len(splits[split])}
                             for f in range(N_FACTORS)}
    return per_factor


def load_pretrained(params, splits, device, run_dir):
    """
    Stage 1 replaced by an earlier run's pretrained MLP. The MLP is rebuilt from the
    checkpoint (a fresh copy in memory) and the file is copied into this run unchanged.
    Its six heads must score exactly what they scored in that run, which confirms the
    same seed gave the same train / val / test split.
    """
    model, ckpt = load_model_from_checkpoint(params.encoder_from, weights_only=False)   # our own file
    model = model.to(device)
    assert ckpt["model_kwargs"]["arch"] == params.model_kind, \
        f"{params.encoder_from} is a {ckpt['model_kwargs']['arch']}, not a {params.model_kind}"
    per_factor = factor_head_metrics(model, splits, params, device)
    stored = ckpt["meta"]["per_factor"]
    for split in stored:
        for f in FACTORS:
            assert abs(per_factor[split][f]["acc"] - stored[split][f]["acc"]) < 1e-6, \
                f"{split} {f}: acc {per_factor[split][f]['acc']} != stored {stored[split][f]['acc']} (different split?)"
    print(f"── stage 1: pretrained {params.model_kind} loaded from {params.encoder_from} "
          f"(per-factor accuracies match the stored ones on train / val / test)")
    for f in FACTORS:
        print(f"  pretraining head {f:15s} test acc {per_factor['test'][f]['acc']:.4f}")
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    shutil.copy2(params.encoder_from, run_dir / "checkpoints" / "encoder.pt")
    return model, per_factor, ckpt["model_kwargs"]


# ── stage 2: freeze and cache ─────────────────────────────────────────────────

def freeze_encoder(model, unfreeze_layers=0):
    """
    The body as a standalone frozen encoder x -> z; the temporary heads are dropped.
    With unfreeze_layers = K > 0 the encoder stops before the last K Linear + ReLU pairs,
    and a deep copy of those is returned as the trainable top (None otherwise). With
    K = 3 the encoder is only the flatten.
    """
    layers = list(model.body)
    top = None
    if unfreeze_layers:
        k = 2 * unfreeze_layers
        assert len(layers) == 6 and all(isinstance(m, nn.Linear) for m in layers[-k::2]) \
            and all(isinstance(m, nn.ReLU) for m in layers[-k + 1::2])
        top = copy.deepcopy(nn.Sequential(*layers[-k:]))
        for p in top.parameters():
            p.requires_grad_(True)
            p.grad = None
        layers = layers[:-k]
    encoder = nn.Sequential(model.flatten, *layers)
    for p in encoder.parameters():
        p.requires_grad_(False)
        p.grad = None                   # left over from the last pretraining step
    encoder.eval()
    reference = {k: v.detach().clone() for k, v in encoder.state_dict().items()}
    return encoder, reference, top


def represent(top, cache):
    """The cache seen through the trainable top: z = top(h) per split (no gradient)."""
    with torch.no_grad():
        return {s: {"z": torch.cat([top(h) for h in c["z"].split(65536)]), "latent": c["latent"]}
                for s, c in cache.items()}


def check_encoder_unchanged(encoder, reference, when):
    """Sanity check 3: same weights (bit for bit) as when frozen, no gradients, eval mode."""
    state = encoder.state_dict()
    assert state.keys() == reference.keys(), f"{when}: encoder parameters renamed"
    changed = [k for k in reference if not torch.equal(state[k], reference[k])]
    assert not changed, f"{when}: encoder parameters changed: {changed}"
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters()), \
        f"{when}: encoder parameters trainable or holding gradients"
    assert not encoder.training, f"{when}: encoder not in eval mode"


# ── stage 3: linear probes ────────────────────────────────────────────────────

def run_probes(params, cache, n, device):
    """One Linear(n, K_f) per factor on the frozen z, trained with the repo's train()."""
    probe_params = type(params)(**{**vars(params), "num_epochs": params.probe_epochs,
                                   "learning_rate": params.lr_probe, "optimizer": "sgd"})
    criterion = nn.CrossEntropyLoss()
    rows = []
    print(f"── stage 3: linear probes on the frozen representation "
          f"({params.probe_epochs} epochs, SGD lr {params.lr_probe})")
    for f, (name, k) in enumerate(zip(FACTORS, SHAPES3D_FACTOR_N_VALUES)):
        torch.manual_seed(params.seed + 1000 + f)
        probe = nn.Linear(n, k)
        z_y = {s: (cache[s]["z"], cache[s]["latent"][:, f]) for s in cache}
        gen = torch.Generator().manual_seed(params.seed + f)
        train_loader = build_tensor_loader(*z_y["train"], batch_size=params.batch_size,
                                           shuffle=True, generator=gen)
        eval_loaders = {s: build_tensor_loader(*z_y[s], batch_size=4096) for s in cache}
        train(probe, train_loader, eval_loaders["val"], probe_params, criterion=criterion)
        row = {"factor": name, "classes": k}
        for s in ("train", "val", "test"):
            row[f"{s}_loss"], row[f"{s}_accuracy"] = evaluate(probe, eval_loaders[s], criterion, device)
        rows.append(row)
        del probe                                 # diagnostic only, discarded
    print(f"  {'factor':15s} {'classes':>7s} {'train_acc':>9s} {'val_acc':>9s} {'test_acc':>9s}")
    for r in rows:
        print(f"  {r['factor']:15s} {r['classes']:7d} {r['train_accuracy']:9.4f} "
              f"{r['val_accuracy']:9.4f} {r['test_accuracy']:9.4f}")
    return rows


# ── stage 4/5: the continual stream ───────────────────────────────────────────

class HeadFactory:
    """Builds the persistent head and its optimizer, and counts how often each was built."""

    def __init__(self):
        self.n_heads = 0
        self.n_optimizers = 0

    def head(self, n, device):
        self.n_heads += 1
        head = nn.Linear(n, 2)
        # the init of every classifier head in the repo (models.factory.reset_head)
        nn.init.kaiming_uniform_(head.weight, nonlinearity="relu")
        nn.init.zeros_(head.bias)
        return head.to(device)

    def optimizer(self, head, lr):
        self.n_optimizers += 1
        return build_optimizer(head, "sgd", lr=lr, momentum=0.0)


class BatchStream:
    """Minibatch indices of the training split: random permutations, consumed in order."""

    def __init__(self, n, batch_size, generator, device):
        self.n, self.batch_size, self.generator, self.device = n, batch_size, generator, device
        self._refill()

    def _refill(self):
        self.perm = torch.randperm(self.n, generator=self.generator, device=self.device)
        self.pos = 0

    def next(self):
        if self.pos + self.batch_size > self.n:
            self._refill()
        idx = self.perm[self.pos:self.pos + self.batch_size]
        self.pos += self.batch_size
        return idx


def run_stream(params, cache, encoder, encoder_ref, n, device, run_dir, checks, top=None):
    """top: the trainable top hidden layers (--unfreeze_layers), cache z = its input."""
    num_tasks, steps, bs = params.num_tasks, params.steps_per_task, params.batch_size
    z_train, z_test = cache["train"]["z"], cache["test"]["z"]
    y_train = all_task_labels(cache["train"]["latent"])        # (N, 6): labels of every rule
    y_test = all_task_labels(cache["test"]["latent"])
    # sanity check 11: binary targets
    assert set(torch.unique(y_train).tolist()) == {0, 1} and set(torch.unique(y_test).tolist()) == {0, 1}
    checks["11_binary_targets"] = {"train_values": [0, 1], "test_values": [0, 1],
                                   "class1_fraction_test": dict(zip(FACTORS, y_test.float().mean(0).tolist()))}

    schedule = task_schedule(num_tasks)
    # sanity checks 8 and 9: the cycle and its counts
    assert all(schedule[t] == t % N_FACTORS for t in range(num_tasks))
    assert [FACTORS[f] for f in schedule[:7]] == FACTORS + [FACTORS[0]]
    counts = {name: schedule.count(f) for f, name in enumerate(FACTORS)}
    assert list(counts.values()) == [134, 134, 133, 133, 133, 133], counts
    checks["8_feature_cycle"] = {f"task_{t}": FACTORS[schedule[t]] for t in range(8)}
    checks["9_task_counts"] = counts

    # ── the persistent head and optimizer, built once, here, before task 0 ───
    factory = HeadFactory()
    classifier = factory.head(n, device)
    # what the optimizer trains: the head, or the top hidden layers followed by the head
    learner = classifier if top is None else nn.Sequential(top.to(device), classifier)
    optimizer = factory.optimizer(learner, params.lr_head)
    head_id, opt_id = id(classifier), id(optimizer)
    head_params = list(learner.parameters())
    if top is not None:
        # trainable hidden layers, numbered as in the MLP (1..3)
        top_linears = dict(zip(range(4 - len(top) // 2, 4), top[::2]))
        top_weight0 = {L: m.weight.detach().clone() for L, m in top_linears.items()}
        top_trajectory = torch.empty(num_tasks, n, n + 1)          # the last hidden layer only

    # sanity check 4: the optimizer holds exactly the head's parameters, plain SGD
    opt_params = [p for g in optimizer.param_groups for p in g["params"]]
    assert ({id(p) for p in opt_params} == {id(p) for p in head_params}
            and len(opt_params) == len(head_params))
    assert not ({id(p) for p in opt_params} & {id(p) for p in encoder.parameters()})
    assert isinstance(optimizer, torch.optim.SGD)
    assert all(g["momentum"] == 0.0 and not g["nesterov"] for g in optimizer.param_groups)
    assert classifier.out_features == 2 and classifier.in_features == n
    checks["4_optimizer_isolation"] = {"optimizer": "SGD", "momentum": 0.0,
                                       "trains": "head" if top is None else
                                                 f"top {len(top) // 2} hidden layer(s) + head",
                                       "param_shapes": [list(p.shape) for p in opt_params]}

    criterion = nn.CrossEntropyLoss()
    gen = torch.Generator(device=device).manual_seed(params.seed)
    batches = BatchStream(len(z_train), bs, gen, device)

    step_log = CsvLog(run_dir / "continual_training.csv")
    task_rows, eval_rows = [], []
    trajectory = torch.empty(num_tasks, 2, n + 1)
    params_after_prev = None
    n_continuity_checks, n_encoder_checks = 0, 0
    global_step, total_seen = 0, 0
    t0 = time.time()
    print(f"── stage 5: {num_tasks} tasks x {steps} SGD steps (batch {bs}, lr {params.lr_head}) "
          f"with one persistent Linear({n}, 2) head"
          + ("" if top is None else f" on {len(top) // 2} trainable hidden layer(s)"))

    for t, f in enumerate(schedule):
        # sanity checks 5 and 7: the same head and optimizer objects throughout
        assert id(classifier) == head_id and id(optimizer) == opt_id
        assert factory.n_heads == 1 and factory.n_optimizers == 1
        learner.train()
        losses = torch.empty(steps, device=device)
        accs = torch.empty(steps, device=device)
        for s in range(steps):
            idx = batches.next()
            z, y = z_train[idx], y_train[idx, f]                # cached z: no encoder gradient
            logits = learner(z)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            if s == 0 and params_after_prev is not None:
                # sanity check 6: task t starts exactly where task t-1 ended
                assert all(torch.equal(p, q) for p, q in zip(learner.parameters(), params_after_prev)), \
                    f"head changed between the end of task {t - 1} and the first step of task {t}"
                n_continuity_checks += 1
            optimizer.step()
            losses[s] = loss.detach()
            accs[s] = (logits.argmax(dim=1) == y).float().mean()

        params_after_prev = [p.detach().clone() for p in learner.parameters()]
        trajectory[t] = torch.cat([classifier.weight.detach(), classifier.bias.detach()[:, None]], 1).cpu()
        if top is not None:
            top_trajectory[t] = torch.cat([top[-2].weight.detach(), top[-2].bias.detach()[:, None]], 1).cpu()

        # per-step log of the task
        losses, accs = losses.cpu().tolist(), accs.cpu().tolist()
        step_log.write({"task_id": t, "relevant_factor": FACTORS[f], "step_within_task": s + 1,
                        "global_step": global_step + s + 1, "examples_seen_in_task": (s + 1) * bs,
                        "total_examples_seen": total_seen + (s + 1) * bs,
                        "train_loss": losses[s], "train_accuracy": accs[s]} for s in range(steps))
        global_step += steps
        total_seen += steps * bs

        # the head after task t, under all six rules, on the test split (no training)
        learner.eval()
        with torch.no_grad():
            logits = learner(z_test)
            if top is not None:
                h_test, h = {}, z_test                 # post-ReLU activations of each trainable layer
                for L, (lin, act) in zip(top_linears, zip(top[::2], top[1::2])):
                    h = h_test[L] = act(lin(h))
            ev = torch.stack([torch.stack([F.cross_entropy(logits, y_test[:, g]),
                                           (logits.argmax(1) == y_test[:, g]).float().mean()])
                              for g in range(N_FACTORS)]).cpu().tolist()
        eval_rows += [{"after_task": t, "trained_on_factor": FACTORS[f], "evaluated_factor": FACTORS[g],
                       "is_current_rule": int(g == f), "test_loss": ev[g][0], "test_accuracy": ev[g][1]}
                      for g in range(N_FACTORS)]
        task_rows.append({
            "task_id": t, "relevant_factor": FACTORS[f],
            "task_distance": task_distance(schedule[t - 1] if t > 0 else None, f),
            "steps": steps, "total_examples_seen": total_seen, "global_step": global_step,
            "task_mean_train_loss": float(np.mean(losses)), "task_mean_train_accuracy": float(np.mean(accs)),
            "last_step_train_loss": losses[-1], "last_step_train_accuracy": accs[-1],
            "current_task_test_loss": ev[f][0], "current_task_test_accuracy": ev[f][1],
            "classifier_weight_norm": classifier.weight.detach().norm().item(),
            "classifier_bias_norm": classifier.bias.detach().norm().item(),
        })
        if top is not None:
            for L, m in top_linears.items():
                w = m.weight.detach()
                task_rows[-1].update({
                    f"layer{L}_weight_norm": w.norm().item(),
                    f"layer{L}_weight_change": (w - top_weight0[L]).norm().item(),   # from the pretrained layer
                    f"layer{L}_dead_units_test": (h_test[L] > 0).sum(0).eq(0).sum().item(),
                    f"layer{L}_active_fraction_test": (h_test[L] > 0).float().mean().item(),
                })

        # sanity check 3: the encoder is untouched
        if (t + 1) % params.check_encoder_every == 0 or t == num_tasks - 1:
            check_encoder_unchanged(encoder, encoder_ref, f"after task {t}")
            n_encoder_checks += 1
        if (t + 1) % params.print_every == 0 or t == 0 or t == num_tasks - 1:
            r = task_rows[-1]
            extra = "" if top is None else "".join(
                f"  L{L} |dW| {r[f'layer{L}_weight_change']:.3f} dead {r[f'layer{L}_dead_units_test']}/{n}"
                for L in top_linears)
            print(f"  task {t:3d} {FACTORS[f]:15s} test acc {r['current_task_test_accuracy']:.4f}  "
                  f"all rules {' '.join(f'{a:.2f}' for _, a in ev)}  |W| {r['classifier_weight_norm']:.3f}{extra}  "
                  f"({time.time() - t0:.0f}s)")

    step_log.close()
    assert factory.n_heads == 1 and factory.n_optimizers == 1
    assert n_continuity_checks == num_tasks - 1
    checks["3_frozen_encoder"] = {"n_checks": n_encoder_checks, "unchanged": True}
    checks["5_single_head"] = {"heads_built": factory.n_heads, "same_object_all_tasks": True}
    checks["6_head_continuity"] = {"transitions_checked": n_continuity_checks, "all_equal": True}
    checks["7_single_optimizer"] = {"optimizers_built": factory.n_optimizers, "same_object_all_tasks": True}

    save_csv(run_dir / "task_end_metrics.csv", task_rows)
    save_csv(run_dir / "all_rule_evaluation.csv", eval_rows)
    save_checkpoint(run_dir / "checkpoints" / "head_final.pt", classifier, optimizer,
                    after_task=num_tasks - 1, in_features=n, out_features=2)
    torch.save({"weight_and_bias": trajectory, "factors": [FACTORS[f] for f in schedule],
                "layout": "(task, class, n weights + bias) at the end of each task"},
               run_dir / "head_trajectory.pt")
    if top is not None:
        save_checkpoint(run_dir / "checkpoints" / "trainable_layers_final.pt", top, after_task=num_tasks - 1,
                        layers=f"hidden layers {list(top_linears)} (Linear + ReLU each), adapted with the head")
        torch.save({"weight_and_bias": top_trajectory, "factors": [FACTORS[f] for f in schedule],
                    "layout": "(task, unit, n weights + bias) of the last hidden layer at the end of each task"},
                   run_dir / "last_layer_trajectory.pt")
    return task_rows


# ── sanity check 10: same image, changing target ──────────────────────────────

def same_image_changing_target(splits, cache, encoder, device, i=0):
    """
    Image i of the training split under each of the six rules: the image (and so z)
    is the same object every time, only its label changes. Uses the repo's relabelling
    (data.factory.relabel_dataset with the upper_half rule) for the labels.
    """
    base = splits["train"]
    x0, latent = base[i]
    with torch.no_grad():
        z = encoder(x0[None].to(device))[0]
    assert torch.allclose(z, cache["train"]["z"][i], atol=1e-5), "cached z differs from encoder(x)"
    rows = {}
    for t in range(N_FACTORS):
        f = task_factor(t)
        task_view = relabel_dataset(base, f, "upper_half")
        x, y = task_view[i]
        assert task_view.images is base.images and torch.equal(x, x0)
        assert y == int(cache["train"]["latent"][i, f] >= SHAPES3D_FACTOR_N_VALUES[f] / 2.0)
        rows[f"task_{t}"] = {"relevant_factor": FACTORS[f], "factor_index": int(latent[f]),
                             "num_values": SHAPES3D_FACTOR_N_VALUES[f], "label": int(y)}
    print(f"── same image (train index {i}), same z (|z| = {z.norm():.3f}), label per task:")
    for t, r in rows.items():
        print(f"  {t}: {r['relevant_factor']:15s} index {r['factor_index']:2d}/{r['num_values']}  -> y = {r['label']}")
    return {"train_index": i, "image_sum": float(x0.sum()), "z_norm": float(z.norm()),
            "z_matches_cache": True, "tasks": rows}


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    params = parse_factor_rule_stream_args()
    if params.num_tasks != NUM_TASKS:
        raise SystemExit(f"--num_tasks must be {NUM_TASKS} for this experiment")
    if params.optimizer != "sgd" or params.activation != "relu" or params.use_layer_norm:
        raise SystemExit("this experiment is plain SGD (momentum 0) with a ReLU MLP and no LayerNorm")
    if params.perm_percentage != 0.0:
        raise SystemExit("this experiment does not permute pixels")
    rng = seed_everything(params.seed)
    device = torch.device(params.device)
    checks = {}

    run_dir = run_dir_for(params, prefix="factorrule_")
    save_summary(run_dir, vars(params), name="config.json")
    print(f"run dir: {run_dir.resolve()}")

    splits = dict(zip(("train", "val", "test"), load_splits(params, rng)))
    checks["1_dataset"] = check_dataset(splits)
    print(f"shapes3d train {len(splits['train'])} / val {len(splits['val'])} / test {len(splits['test'])}, "
          f"images in [0, 1], flattened to {D_IN}")

    # stage 1
    if params.encoder_from:
        model, pretrain_per_factor, model_kwargs = load_pretrained(params, splits, device, run_dir)
    else:
        model, pretrain_per_factor, model_kwargs = pretrain_representation(params, splits, device, run_dir)

    # stage 2
    encoder, encoder_ref, top = freeze_encoder(model, params.unfreeze_layers)
    pretrained = copy.deepcopy(model) if top is not None else None   # base of encoder_adapted.pt
    del model                                        # the temporary factor heads go with it
    x, _ = next(iter(build_eval_loader(splits["train"], params.batch_size)))
    z = encoder(x.to(device))
    z_dim = D_IN if params.unfreeze_layers == 3 else params.n          # full network: z = the image
    assert z.shape == (len(x), z_dim) and not z.requires_grad          # sanity check 2
    checks["2_encoder_output"] = {"batch_shape": list(z.shape), "n": params.n}
    cache = {}
    for s, ds in splits.items():
        zs, latent = extract_features(encoder, build_eval_loader(ds, params.batch_size, params.num_workers),
                                      device)
        cache[s] = {"z": zs, "latent": latent.long()}
    check_encoder_unchanged(encoder, encoder_ref, "after caching z")
    print(f"── stage 2: encoder frozen, z cached ({z_dim} dims; "
          f"{(cache['train']['z'] > 0).float().mean():.3f} of train z entries > 0)")

    # stage 3 (sanity check 12): probes on the representation the head starts from
    probe_rows = run_probes(params, cache if top is None else represent(top.to(device), cache),
                            params.n, device)
    save_csv(run_dir / "representation_probe_results.csv", probe_rows)
    checks["12_representation_quality"] = {r["factor"]: r["test_accuracy"] for r in probe_rows}
    check_encoder_unchanged(encoder, encoder_ref, "after the probes")

    checks["10_same_image_changing_target"] = same_image_changing_target(splits, cache, encoder, device)

    # stages 4-5
    if top is not None:
        top_ref = {k: v.detach().clone() for k, v in top.state_dict().items()}
    task_rows = run_stream(params, cache, encoder, encoder_ref, params.n, device, run_dir, checks, top=top)

    # the cached z still is the frozen encoder's representation
    zs, _ = extract_features(encoder, build_eval_loader(splits["test"], params.batch_size), device)
    assert torch.allclose(zs, cache["test"]["z"], atol=1e-5)
    checks["3_frozen_encoder"]["cache_matches_encoder_at_end"] = True

    extra = {}
    if top is not None:
        # the pretrained MLP is still the loaded one; the adapted copy is saved next to it
        k = len(top)
        pretrained_top = nn.Sequential(*list(pretrained.body)[-k:]).state_dict()
        assert all(torch.equal(v, pretrained_top[name]) for name, v in top_ref.items()), \
            "the pretrained MLP changed"
        assert all(not torch.equal(top_ref[f"{i}.weight"], top[i].weight) for i in range(0, k, 2)), \
            "a trainable hidden layer was not trained"
        adapted = copy.deepcopy(pretrained)
        nn.Sequential(*list(adapted.body)[-k:]).load_state_dict(top.state_dict())
        save_checkpoint(run_dir / "checkpoints" / "encoder_adapted.pt", adapted, model_kwargs=model_kwargs,
                        stage=f"top {k // 2} hidden layer(s) adapted through the stream",
                        encoder=f"model.body (+ model.flatten); body[-{k}:] trained with the head",
                        head="six temporary factor heads of the pretrained MLP (not adapted)",
                        factor_names=FACTORS, after_task=params.num_tasks - 1)
        probes_after = run_probes(params, represent(top, cache), params.n, device)
        save_csv(run_dir / "representation_probe_results_after_stream.csv", probes_after)
        extra = {"representation_probes_after_stream": probes_after,
                 "trainable_layers_final": {c: task_rows[-1][c] for c in task_rows[-1] if c.startswith("layer")}}

    save_summary(run_dir, {
        "split_sizes": checks["1_dataset"]["split_sizes"],
        "pretraining_heads_per_factor": pretrain_per_factor,
        "representation_probes": probe_rows,
        "final_current_task_test_accuracy": task_rows[-1]["current_task_test_accuracy"],
        **extra,
        "sanity_checks": checks,
    })
    print("all sanity checks passed")

    from visualization_scripts.factor_rule_stream import plot_all
    plot_all(run_dir)
    print(f"done -> {run_dir.resolve()}")
