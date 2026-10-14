"""
utils/args.py — Command-line arguments of the experiments in experiments/.
"""
from __future__ import annotations

import argparse

import torch


def basic_task_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_kind', type=str, default='mlp_2x512', help='Model kind to use (mlp_2x512, mlp_3x256, ..., cnn)')
    parser.add_argument('--dataset_kind', type=str, default='mnist', help='Dataset kind to use')
    parser.add_argument('--perm_percentage', type=float, default=0.0, help='Percentage of permuted pixels in the dataset')

    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--save_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    parser.add_argument('--optimizer', type=str, default='sgd', help='Optimizer to use (sgd with momentum 0, adam)')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function (relu, gelu, leaky_relu)')
    parser.add_argument('--use_layer_norm', action='store_true', help='Insert LayerNorm after each linear/conv layer')
    parser.add_argument('--n_samples', type=int, default=None, help='Shapes3D only: number of images to load (the full set is ~5.9 GB)')
    parser.add_argument('--target_factor', type=int, default=4, help='Shapes3D only: latent factor used as the label')
    parser.add_argument('--label_rule', type=str, default=None, choices=['parity', 'upper_half'], help='Shapes3D only: relabel the target factor (parity: even index -> 1, odd -> 0; upper_half: index >= K/2 -> 1)')
    parser.add_argument('--num_workers', type=int, default=0, help='DataLoader worker processes')
    parser.add_argument('--run_name', type=str, default=None, help='Name of the results sub-directory (default: auto)')
    parser.add_argument('--n_sample_images', type=int, default=4, help='How many example images of the task to save')
    parser.add_argument('--samples_only', action='store_true', help='Write the sample image figure and exit, without training')


    # ── task: a partial permutation of the pixel grid ─────────────────────────────

    def sample_task(n_pixels, perm_percentage, rng):
        """
        Samples one task: a permutation of the pixel grid in which a
        `perm_percentage` fraction of the positions is shuffled among itself
        (perm_percentage=0 → identity, 1 → a fully permuted image).

        The permutation is drawn once and applied to BOTH splits, otherwise the
        test set would carry a different task than the one the model was trained on.
        """
        perm = np.arange(n_pixels)
        n_perm = int(round(n_pixels * perm_percentage))
        if n_perm > 1:
            idx = rng.choice(n_pixels, n_perm, replace=False)
            perm[idx] = idx[rng.permutation(n_perm)]
        return perm


    class PermutedPixels(torch.utils.data.Dataset):
        """Wraps a dataset and re-orders the pixels of every image with `perm`."""

        def __init__(self, dataset, perm):
            self.dataset = dataset
            self.perm = torch.from_numpy(np.asarray(perm)).long()

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, i):
            x, y = self.dataset[i]
            shape = x.shape
            x = x.reshape(-1, self.perm.numel())[:, self.perm].reshape(shape)
            return x, y


    def permute_dataset(dataset, perm):
        """Applies the sampled task to a dataset (no-op for the identity permutation)."""
        if np.array_equal(perm, np.arange(len(perm))):
            return dataset
        return PermutedPixels(dataset, perm)


    # ── dataset introspection ─────────────────────────────────────────────────────

    def dataset_spec(train, test):
        """(C, H, W) of one image and the number of classes, inferred from the data."""
        x, _ = train[0]
        if x.dim() == 3:
            c, h, w = x.shape
        elif x.dim() == 2:
            c, (h, w) = 1, x.shape
        else:
            raise ValueError(f"Expected image-shaped samples, got shape {tuple(x.shape)}")

        n_classes = getattr(train, "n_classes", None)
        if n_classes is None and getattr(train, "classes", None) is not None:
            n_classes = len(train.classes)
        if n_classes is None:
            labels = [int(train[i][1]) for i in range(min(len(train), 10_000))]
            n_classes = max(labels) + 1
        return (c, h, w), int(n_classes)


    def load_dataset(params, rng):
        """Loads (train, test) for the requested dataset kind."""
        kwargs = {}
        if params.dataset_kind == "shapes3d":
            kwargs = dict(n_samples=params.n_samples,
                          target_factor=params.target_factor,
                          rng=rng)
        out = get_dataset(params.dataset_kind, **kwargs)
        if not (isinstance(out, tuple) and len(out) == 2):
            raise TypeError(
                f"get_dataset('{params.dataset_kind}') did not return a (train, test) split."
            )
        return out


    # ── train / eval ──────────────────────────────────────────────────────────────

    def evaluate(model, loader, criterion, device):
        model.eval()
        total_loss, correct, n = 0.0, 0, 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                total_loss += criterion(logits, y).item() * y.size(0)
                correct += (logits.argmax(dim=1) == y).sum().item()
                n += y.size(0)
        return total_loss / n, correct / n


    def train(model, train_loader, test_loader, params):
        """Trains on the sampled task, recording train loss / train acc / test acc over time."""
        device = torch.device(params.device)
        model = model.to(device)
        criterion = nn.CrossEntropyLoss()
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
                correct += (logits.argmax(dim=1) == y).sum().item()
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

        return history





    # ── results ───────────────────────────────────────────────────────────────────

    def save_results(run_dir, params, history, task_perm, extra):
        run_dir.mkdir(parents=True, exist_ok=True)

        with open(run_dir / "args.json", "w") as f:
            json.dump(vars(params), f, indent=2)

        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f)

        with open(run_dir / "summary.json", "w") as f:
            json.dump({
                **extra,
                "final_train_loss": history["train_loss"][-1],
                "final_train_acc":  history["train_acc"][-1],
                "final_test_loss":  history["test_loss"][-1],
                "final_test_acc":   history["test_acc"][-1],
                "best_test_acc":    max(history["test_acc"]),
            }, f, indent=2)

        with open(run_dir / "metrics.csv", "w") as f:
            f.write("epoch,train_loss,train_acc,test_loss,test_acc\n")
            for i, e in enumerate(history["epoch"]):
                f.write(f"{e},{history['train_loss'][i]},{history['train_acc'][i]},"
                        f"{history['test_loss'][i]},{history['test_acc'][i]}\n")

        np.save(run_dir / "task_pixel_permutation.npy", task_perm)


    def main():
        params = parser.parse_args()

        rng = np.random.default_rng(params.seed)
        torch.manual_seed(params.seed)

        # ── load dataset ──────────────────────────────────────────────────────────
        try:
            train_set, test_set = load_dataset(params, rng)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print(f"Available datasets: {list_datasets()}")
            return

        (c, h, w), n_classes = dataset_spec(train_set, test_set)
        print(f"dataset   : {params.dataset_kind}  "
              f"train {len(train_set)} / test {len(test_set)}  "
              f"images {c}x{h}x{w}  classes {n_classes}")

        # ── sample a task and apply it to both splits ─────────────────────────────
        task_perm = sample_task(h * w, params.perm_percentage, rng)
        train_set = permute_dataset(train_set, task_perm)
        test_set  = permute_dataset(test_set, task_perm)
        print(f"task      : {100 * params.perm_percentage:.0f}% of the {h * w} pixel "
              f"positions permuted")

        # Python 3.14 starts worker processes with forkserver, which PICKLES the
        # dataset into every worker -- for an in-memory Shapes3D split that is
        # ~6 GB copied per worker and a certain OOM. fork shares the pages instead
        # (the workers never touch CUDA, so forking after init is safe here).
        mp_context = "fork" if params.num_workers > 0 else None

        # ── save a sample of what the task looks like ─────────────────────────────
        # the label is part of the identity of a run: without it an object_azimuth
        # sweep would silently overwrite the object_type one
        label = ""
        if params.dataset_kind == "shapes3d" and params.target_factor is not None:
            label = f"_{SHAPES3D_FACTOR_NAMES[params.target_factor]}"
        run_name = params.run_name or (
            f"{params.dataset_kind}_{params.model_kind}_"
            f"perm{int(round(100 * params.perm_percentage)):03d}{label}_seed{params.seed}"
        )
        run_dir = Path(params.save_dir) / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        save_sample_images(train_set, run_dir / "sample_images.png", params,
                           n=params.n_sample_images,
                           rng=np.random.default_rng(params.seed))
        print(f"samples   : {run_dir / 'sample_images.png'}")
        if params.samples_only:
            return

        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=params.batch_size, shuffle=True,
            num_workers=params.num_workers, multiprocessing_context=mp_context,
            persistent_workers=params.num_workers > 0)
        test_loader = torch.utils.data.DataLoader(
            test_set, batch_size=max(params.batch_size, 256), shuffle=False,
            num_workers=params.num_workers, multiprocessing_context=mp_context,
            persistent_workers=params.num_workers > 0)

        # ── build model ───────────────────────────────────────────────────────────
        model = build_model(
            params.model_kind,
            d_in=c * h * w,
            in_channels=c,
            input_size=h,
            n_classes=n_classes,
            activation=params.activation,
            use_layer_norm=params.use_layer_norm,
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"model     : {params.model_kind}  ({n_params} parameters)  "
              f"optimizer {params.optimizer}  lr {params.learning_rate}  "
              f"device {params.device}")
    return parser


def parse_basic_task_args(argv=None) -> argparse.Namespace:
    return basic_task_parser().parse_args(argv)


def repr_distance_parser() -> argparse.ArgumentParser:
    """The basic-task arguments, plus the head-retraining budget."""
    parser = basic_task_parser()
    parser.add_argument('--head_epochs', type=int, default=None,
                        help='Epochs for retraining the head on task 2 (default: --num_epochs)')
    parser.add_argument('--task1_factors', type=int, nargs='+', default=None,
                        help='Shapes3D only: task 1 = parity of the summed indices of these features '
                             '(tasks.feature_streams.FeatureParityTask); needs --task2_factors')
    parser.add_argument('--task2_factors', type=int, nargs='+', default=None,
                        help='Shapes3D only: task 2 = parity of these features')
    parser.add_argument('--set_rule', type=str, default='parity', choices=['parity', 'sum'],
                        help='Label rule of --task1_factors / --task2_factors tasks: parity of the summed '
                             'indices, or sum above median (notes/sum_above_median_labels.md)')
    parser.add_argument('--task2_target_factor', type=int, default=None,
                        help='Shapes3D only: label task 2 by this factor instead of --target_factor '
                             '(--label_rule applies to both)')
    # two tasks with 0% permuted pixels would be the same task
    parser.set_defaults(perm_percentage=1.0)
    return parser


def parse_repr_distance_args(argv=None) -> argparse.Namespace:
    return repr_distance_parser().parse_args(argv)


def tasks_stream_parser() -> argparse.ArgumentParser:
    """The basic-task arguments, plus the length of the stream and the snapshot rate."""
    parser = basic_task_parser()
    parser.add_argument('--n_tasks', type=int, default=10,
                        help='Number of tasks in the stream, trained one after the other')
    parser.add_argument('--snapshot_every', type=int, default=1,
                        help='Save a checkpoint every this many epochs within each task')
    parser.add_argument('--task_factors', type=int, nargs='+', default=None,
                        help='Shapes3D only: task n is labelled by factor task_factors[n %% len] '
                             '(relabelled with --label_rule); default: every task uses --target_factor')
    parser.add_argument('--features_per_task', type=int, default=1,
                        help='With --task_factors: task n is labelled by the parity of task_factors[n], ...,'
                             ' task_factors[n + k - 1] (k = 2: each switch swaps one of the two features)')
    parser.add_argument('--keep_features', type=int, default=None,
                        help='With --features_per_task 3: features kept per switch, 1 or 2 (default 2)')
    parser.add_argument('--pca_components', type=int, default=6,
                        help='With --pca_bins: how many principal components of the raw train pixels '
                             'serve as factors')
    parser.add_argument('--pca_bins', type=int, default=None,
                        help='Use the first --pca_components principal components as factors, each cut '
                             'into this many equal-count bins (even); task n is the parity of its bin on '
                             'component task_factors[n] (default task_factors: every component)')
    parser.add_argument('--resume', action='store_true',
                        help='Continue an existing run directory after its last completed task')
    parser.add_argument('--deadline_h', type=float, default=None,
                        help='Stop cleanly (between tasks) before this many hours of wall time')
    # consecutive tasks with 0% permuted pixels would all be the same task
    parser.set_defaults(perm_percentage=1.0)
    return parser


def parse_tasks_stream_args(argv=None) -> argparse.Namespace:
    return tasks_stream_parser().parse_args(argv)


def factor_rule_stream_parser() -> argparse.ArgumentParser:
    """
    The basic-task arguments, plus those of experiments/run_factor_rule_stream.py:
    pretrain a 3-hidden-layer MLP on all six Shapes3D factors, freeze it, and train one
    persistent Linear(n, 2) head through a cycle of binary factor rules.

    The existing --learning_rate / --num_epochs / --save_dir are the pretraining
    learning rate / pretraining epochs / output directory; the experiment's own names
    for them (--lr-pretrain, --pretrain-epochs, --output-dir) are aliases.
    """
    parser = basic_task_parser()
    parser.add_argument('--n', type=int, default=256, help='Width of each of the 3 hidden layers (= dim of z)')
    parser.add_argument('--lr_pretrain', '--lr-pretrain', dest='learning_rate', type=float, default=argparse.SUPPRESS,
                        help='Alias of --learning_rate: SGD learning rate of the representation pretraining')
    parser.add_argument('--pretrain_epochs', '--pretrain-epochs', dest='num_epochs', type=int, default=argparse.SUPPRESS,
                        help='Alias of --num_epochs: epochs of representation pretraining')
    parser.add_argument('--output_dir', '--output-dir', dest='save_dir', type=str, default=argparse.SUPPRESS,
                        help='Alias of --save_dir')
    parser.add_argument('--data_path', '--data-path', type=str, default=None,
                        help='Directory holding 3dshapes.h5 (default: data/_cache/shapes3d or $PLASTICITY_DATA_DIR)')
    parser.add_argument('--val_frac', type=float, default=0.1, help='Fraction of the training split held out for validation')
    parser.add_argument('--lr_head', '--lr-head', type=float, default=0.01, help='SGD learning rate of the persistent Linear(n, 2) head')
    parser.add_argument('--lr_probe', '--lr-probe', type=float, default=None, help='SGD learning rate of the linear probes (default: --lr_head)')
    parser.add_argument('--probe_epochs', '--probe-epochs', type=int, default=10, help='Epochs of each linear probe')
    parser.add_argument('--steps_per_task', '--steps-per-task', type=int, default=500, help='SGD steps of the head per task')
    parser.add_argument('--num_tasks', '--num-tasks', type=int, default=800, help='Number of tasks (fixed at 800 for this experiment)')
    parser.add_argument('--check_encoder_every', type=int, default=100, help='Verify the frozen encoder every this many tasks')
    parser.add_argument('--print_every', type=int, default=50, help='Print progress every this many tasks')
    parser.add_argument('--encoder_from', '--encoder-from', type=str, default=None,
                        help='checkpoints/encoder.pt of an earlier run with the same seed: reuse that '
                             'pretrained MLP (a copy; the file is not modified) instead of pretraining')
    parser.add_argument('--unfreeze_layers', '--unfreeze-layers', type=int, default=0, choices=[0, 1, 2, 3],
                        help='Train the top this many hidden layers (Linear + ReLU) together with the head, '
                             'with the same SGD optimizer and --lr_head (0: frozen encoder, 3: full network)')
    parser.add_argument('--unfreeze_last_layer', '--unfreeze-last-layer', dest='unfreeze_layers',
                        action='store_const', const=1, help='Alias of --unfreeze_layers 1')
    parser.set_defaults(dataset_kind='shapes3d', perm_percentage=0.0, target_factor=None,
                        num_epochs=30, learning_rate=0.003, batch_size=256,
                        save_dir='results/shapes3d_factor_rule_stream')
    return parser


def parse_factor_rule_stream_args(argv=None) -> argparse.Namespace:
    params = factor_rule_stream_parser().parse_args(argv)
    params.model_kind = f"mlp_3x{params.n}"
    if params.lr_probe is None:
        params.lr_probe = params.lr_head
    return params
