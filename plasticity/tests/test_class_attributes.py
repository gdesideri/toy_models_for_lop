"""tasks/class_attributes.py and the --class_tasks option of experiments/run_tasks_stream.py."""
import numpy as np
import pytest
import torch

from experiments.run_tasks_stream import changed_args, resolve_task_factors
from tasks.class_attributes import ATTRIBUTES, CIFAR10_CLASSES, ClassAttributeTask, attribute_names
from utils.args import parse_tasks_stream_args
from utils.results import run_dir_for


class Labelled:
    """A CIFAR-like dataset: `targets` and images that encode their own index."""

    def __init__(self, targets):
        self.targets = list(targets)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return torch.tensor([float(i)]), self.targets[i]


def cifar_like(per_class=100):
    return Labelled(np.repeat(np.arange(10), per_class))


def test_cifar10_attributes_name_the_right_classes():
    named = {a: sorted(CIFAR10_CLASSES[c] for c in cs) for a, cs in ATTRIBUTES["cifar10"].items()}
    assert named["animal"] == ["bird", "cat", "deer", "dog", "frog", "horse"]
    assert named["four_legs"] == ["cat", "deer", "dog", "frog", "horse"]
    assert named["fur"] == ["cat", "deer", "dog", "horse"]
    assert named["wings"] == ["airplane", "bird"]
    assert named["wheels"] == ["automobile", "truck"]


@pytest.mark.parametrize("name", ["animal", "four_legs", "fur", "wings", "wheels"])
def test_task_is_balanced_and_labels_follow_the_class(name):
    data = cifar_like()
    task = ClassAttributeTask("cifar10", name, per_label=10_000, seed=0)
    labelled = task.apply(data)
    labels = np.array([labelled[i][1] for i in range(len(labelled))])
    assert labels.mean() == 0.5
    positive = ATTRIBUTES["cifar10"][name]
    for i in range(len(labelled)):
        x, y = labelled[i]
        assert y == int(data.targets[int(x.item())] in positive)


def test_per_label_caps_the_size():
    labelled = ClassAttributeTask("cifar10", "four_legs", per_label=30, seed=0).apply(cifar_like())
    assert len(labelled) == 60


def test_same_seed_same_subset_other_seed_other_subset():
    def images(seed):
        labelled = ClassAttributeTask("cifar10", "fur", per_label=50, seed=seed).apply(cifar_like())
        return [labelled[i][0].item() for i in range(len(labelled))]
    assert images(0) == images(0)
    assert images(0) != images(1)


def test_unknown_attribute_or_dataset_rejected():
    with pytest.raises(ValueError, match="unknown attribute"):
        ClassAttributeTask("cifar10", "flies", 10, 0)
    with pytest.raises(ValueError, match="no class attributes"):
        attribute_names("mnist")


def test_resolve_accepts_known_tasks_and_rejects_the_rest():
    p = parse_tasks_stream_args(["--dataset_kind", "cifar10", "--class_tasks", "animal", "wings"])
    assert resolve_task_factors(p) is None
    with pytest.raises(SystemExit, match="unknown"):
        resolve_task_factors(parse_tasks_stream_args(["--dataset_kind", "cifar10", "--class_tasks", "flies"]))
    with pytest.raises(SystemExit, match="no class attributes"):
        resolve_task_factors(parse_tasks_stream_args(["--dataset_kind", "mnist", "--class_tasks", "animal"]))
    with pytest.raises(SystemExit, match="only one task source"):
        resolve_task_factors(parse_tasks_stream_args(["--dataset_kind", "cifar10", "--class_tasks", "animal",
                                                      "--cluster_factors", "6"]))


def test_run_name(tmp_path):
    p = parse_tasks_stream_args(["--dataset_kind", "cifar10", "--perm_percentage", "0", "--model_kind",
                                 "mlp_2x30", "--class_tasks", "animal", "four_legs", "fur", "wings", "wheels",
                                 "--seed", "0", "--save_dir", str(tmp_path)])
    assert run_dir_for(p, prefix="stream_").name == \
        "stream_cifar10_mlp_2x30_perm000_attrs-animal-four_legs-fur-wings-wheels_seed0"
    p.per_label = 5000
    assert run_dir_for(p, prefix="stream_").name.endswith("_attrs-animal-four_legs-fur-wings-wheels_n5000_seed0")


def test_old_runs_still_resume_and_other_tasks_refused():
    p = parse_tasks_stream_args(["--dataset_kind", "cifar10", "--cluster_factors", "6"])
    saved = {k: v for k, v in vars(p).items() if k not in ("class_tasks", "per_label")}
    assert changed_args(saved, p) == []
    saved = vars(parse_tasks_stream_args(["--dataset_kind", "cifar10", "--class_tasks", "animal"]))
    p = parse_tasks_stream_args(["--dataset_kind", "cifar10", "--class_tasks", "wings"])
    assert changed_args(saved, p) == ["class_tasks"]


def test_precision_of_class_1():
    from utils.evaluation import evaluate, positive_counts, precision
    logits = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]])   # predicts 1, 1, 1, 0
    y = torch.tensor([1, 0, 1, 1])
    assert positive_counts(logits, y) == (2, 3)
    assert precision(2, 3) == pytest.approx(2 / 3)
    assert precision(0, 0) == 0.0

    model = torch.nn.Identity()
    loader = [(logits[:2], y[:2]), (logits[2:], y[2:])]
    loss, acc, prec = evaluate(model, loader, torch.nn.CrossEntropyLoss(), "cpu", with_precision=True)
    assert acc == 0.5 and prec == pytest.approx(2 / 3)
    assert len(evaluate(model, loader, torch.nn.CrossEntropyLoss(), "cpu")) == 2
