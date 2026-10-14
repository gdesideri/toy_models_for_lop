""""
- carica il dataset 
- seleziona che mod vuoi 
- carica il modello
- smapla un task
- addestra il modello sul task
- salva train e test acc e train loss over time
- stampale
- salva i risultati in una cartella
"""

import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.args import parse_basic_task_args
from data.factory import build_dataset, build_loaders, dataset_spec
from tasks.permuted_pixels import sample_pixel_permutation, permute_dataset
from models.factory import build_model, build_optimizer
from utils.evaluation import evaluate, n_correct
from utils.results import run_dir_for, save_results
from visualization_scripts.task_training import plot_history, save_sample_images


def train(model, train_loader, test_loader, params, on_epoch_end=None, criterion=None):
    """
    Trains on the sampled task, recording train loss / train acc / test acc over time.
    on_epoch_end(epoch, model), if given, is called after each epoch (e.g. to snapshot).
    criterion defaults to cross entropy; one with an n_correct method also defines
    the accuracy (utils.evaluation.n_correct).
    """
    device = torch.device(params.device)
    model = model.to(device)
    criterion = criterion if criterion is not None else nn.CrossEntropyLoss()
    optimizer = build_optimizer(model, params.optimizer, lr=params.learning_rate)

    history = {
        "step": [], "step_loss": [],
        "epoch": [], "train_loss": [], "train_acc": [], "test_loss": [], "test_acc": [],
    }

    step = 0
    t0 = time.time()
    for epoch in range(1, params.num_epochs + 1):
        model.train()
        running_loss, correct, n = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            history["step"].append(step)
            history["step_loss"].append(loss.item())

            running_loss += loss.item() * y.size(0)
            correct += n_correct(logits, y, criterion)
            n += y.size(0)

        train_loss, train_acc = running_loss / n, correct / n
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

        print(f"  epoch {epoch:3d}/{params.num_epochs}  "
              f"train_loss {train_loss:.4f}  train_acc {train_acc:.4f}  "
              f"test_loss {test_loss:.4f}  test_acc {test_acc:.4f}  "
              f"({time.time() - t0:.1f}s)")

        if on_epoch_end is not None:
            on_epoch_end(epoch, model)

    return history


if __name__ == "__main__":
    params = parse_basic_task_args()
    rng = np.random.default_rng(params.seed)
    torch.manual_seed(params.seed)

    # load the dataset
    train_set, test_set = build_dataset(params.dataset_kind, n_samples=params.n_samples,
                                        target_factor=params.target_factor, rng=rng,
                                        label_rule=params.label_rule)
    (c, h, w), n_classes = dataset_spec(train_set)

    # sample a task: one pixel permutation, applied to both splits
    task_perm = sample_pixel_permutation(h * w, params.perm_percentage, rng)
    train_set = permute_dataset(train_set, task_perm)
    test_set = permute_dataset(test_set, task_perm)

    run_dir = run_dir_for(params)
    save_sample_images(train_set, run_dir / "sample_images.png", params,
                       n=params.n_sample_images, rng=np.random.default_rng(params.seed))
    if params.samples_only:
        sys.exit(0)

    train_loader, test_loader = build_loaders(train_set, test_set,
                                              params.batch_size, params.num_workers)

    # load the model
    model = build_model(params.model_kind, d_in=c * h * w, in_channels=c, input_size=h,
                        n_classes=n_classes, activation=params.activation,
                        use_layer_norm=params.use_layer_norm)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{params.dataset_kind} {c}x{h}x{w} ({len(train_set)}/{len(test_set)}), "
          f"{n_classes} classes, {100 * params.perm_percentage:.0f}% permuted | "
          f"{params.model_kind} ({n_params} params) on {params.device}")

    # train the model on the task
    history = train(model, train_loader, test_loader, params)

    # save the results and plot them
    save_results(run_dir, params, history, model, task_perm, extra={
        "n_train": len(train_set), "n_test": len(test_set),
        "image_shape": [c, h, w], "n_classes": n_classes, "n_params": n_params,
    })
    plot_history(history, params, run_dir / "curves.png")
    print(f"final test_acc {history['test_acc'][-1]:.4f} -> {run_dir.resolve()}")
