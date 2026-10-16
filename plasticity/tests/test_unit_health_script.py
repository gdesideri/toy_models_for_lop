"""visualization_scripts/unit_health.py: post-hoc health equals logged health; the report runs."""
import json

import pytest

from tests.test_stream_shift_health import HAS_MNIST, run_stream
from visualization_scripts import common
from visualization_scripts import unit_health as uh


@pytest.mark.skipif(not HAS_MNIST, reason="MNIST not downloaded")
def test_post_hoc_health_equals_the_logged_health_and_the_report_runs(tmp_path, monkeypatch, capsys):
    run = run_stream(tmp_path / "stream", "--center_inputs", "1", "--health_every", "1")
    logged = json.load(open(run / "summary.json"))

    measured = uh.compute(tmp_path / "stream", 0, every=1, device="cpu")
    assert measured["init_health"] == pytest.approx(logged["init_health"])
    assert [t["task"] for t in measured["tasks"]] == [t["task"] for t in logged["tasks"]]
    for m, t in zip(measured["tasks"], logged["tasks"]):
        assert m["health"] == pytest.approx(t["health"])
    assert measured["input_stats"] == pytest.approx(logged["input_stats"])

    monkeypatch.setattr(common, "RESULTS", tmp_path)
    groups = {"centred": uh.load_group(tmp_path / "stream")}
    uh.report(groups, "test")
    assert (tmp_path / "unit_health_test.png").exists()
    assert "centred" in capsys.readouterr().out
