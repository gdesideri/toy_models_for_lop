"""
experiments/replay_stream_unit_activity.py — Per-epoch activity of every hidden unit
over a window of consecutive tasks t, t+1, ... of a finished task stream
(experiments/run_tasks_stream.py).

The stream only kept snapshots at a few epochs, so the two tasks are replayed: the
network is loaded from the end of task t-1 together with the torch RNG state saved
there, and tasks t .. t+window-1 are retrained exactly as the stream trained them (same
split, same saved pixel permutations, same batch order, same optimizer). At epochs
where the stream kept a snapshot the replayed weights are compared with it.

After every epoch, for each unit, the fraction of the current task's training data
on which the unit is active (output > 0), always on that task's own data: D_n during
the epochs of task n.

Run with (GPU):
    uv run python -m experiments.replay_stream_unit_activity \
        --run_dir results/shapes3d_stream_perm100_sgd/stream_..._seed0 --tasks 20 70 300 400 --window 5
Writes   : <run_dir>/unit_activity_t<t>_w<window>.json
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.factory import build_dataset, build_loaders
from tasks.permuted_pixels import permute_dataset
from models.factory import build_model
from utils.checkpoints import load_checkpoint
from utils.evaluation import activation_rates_over
from utils.stream_probe import task_batches
from experiments.run_basic_task_training import train


def replay(run_dir: Path, t: int, device: str, window: int = 2):
    a = json.load(open(run_dir / "args.json"))
    params = argparse.Namespace(**{**a, "device": device})
    train_set, test_set = build_dataset(a["dataset_kind"], n_samples=a["n_samples"], target_factor=a["target_factor"],
                                        rng=np.random.default_rng(a["seed"]), label_rule=a["label_rule"])
    images = torch.from_numpy(np.ascontiguousarray(train_set.images)).to(device)

    start = run_dir / f"task_{t - 1:03d}" / "checkpoints" / f"epoch_{a['num_epochs']:03d}.pt"
    ckpt = torch.load(start, map_location="cpu", weights_only=False)
    model = build_model(**ckpt["model_kwargs"])
    load_checkpoint(start, model)
    torch.set_rng_state(ckpt["meta"]["torch_rng_state"])

    out = {"run_dir": str(run_dir), "t": t, "window": window, "n_images": len(images), "epochs": [], "task": [],
           "rates": {}, "snapshot_check": {}}
    for n in range(t, t + window):
        perm = np.load(run_dir / f"task_{n:03d}" / "task_pixel_permutation.npy")
        loaders = build_loaders(permute_dataset(train_set, perm), permute_dataset(test_set, perm),
                                a["batch_size"], a["num_workers"])
        perm_t = torch.from_numpy(perm).long()

        def on_epoch_end(epoch, model, n=n, perm_t=perm_t):
            rates = activation_rates_over(model, task_batches(images, perm_t))
            out["epochs"].append(len(out["epochs"]) + 1)
            out["task"].append(n)
            for k, r in enumerate(rates, start=1):
                out["rates"].setdefault(f"layer_{k}", []).append(r.tolist())
            snap = run_dir / f"task_{n:03d}" / "checkpoints" / f"epoch_{epoch:03d}.pt"
            if snap.exists():   # the stream kept this epoch: the replay must match it
                saved = torch.load(snap, map_location="cpu", weights_only=False)["model_state"]
                diff = max(float((v.detach().cpu() - saved[k]).abs().max()) for k, v in model.state_dict().items())
                out["snapshot_check"][f"task {n} epoch {epoch}"] = diff

        print(f"── replaying task {n}" + (f" (from the end of task {t - 1})" if n == t else ""))
        train(model, *loaders, params, on_epoch_end=on_epoch_end)
    json.dump(out, open(run_dir / f"unit_activity_t{t}_w{window}.json", "w"))
    print(f"   snapshot check (max |replayed - saved| weight): {out['snapshot_check']}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True)
    p.add_argument("--tasks", type=int, nargs="+", required=True)
    p.add_argument("--window", type=int, default=2, help="consecutive tasks to replay from each t")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    for t in args.tasks:
        replay(Path(args.run_dir), t, args.device, args.window)
