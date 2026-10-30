"""
orthogonal_tasks — continual learning on K binary tasks over one fixed, centred input cloud, with a
one-hidden-layer ReLU net whose hidden units are tracked one by one.

    data.py      the input cloud and the tasks (half-plane, stripe or XOR labels)
    config.py    the run configuration, its folder name and the seed streams
    network.py   the net (trainable hidden bias, random or zero bias init), exact full-batch gradients
    train.py     one seed: the continued net and a fresh net per phase on the same minibatches
    summary.py   per-unit and per-phase summaries, the aggregate over seeds, loading saved runs
    run.py       command line

Run from plasticity/:  uv run python -m orthogonal_tasks.run --help
"""
