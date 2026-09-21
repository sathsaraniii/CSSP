"""E8 reporting uses matched saved trials and never reruns algorithms."""
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "pipeline_tradeoff", Path(__file__).resolve().parents[1] / "experiments/plot_pipeline_tradeoff.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def fixture_run(path, count=6, trials=2, ks=(2, 3)):
    path.mkdir()
    config = dict(experiment="E7", status="complete", pipelines=count, ks=list(ks),
                  trials=trials, seed=42, dataset="fixture", matrix_shape=[9, 7])
    (path / "config.json").write_text(json.dumps(config))
    rows = []
    for k in ks:
        for trial in range(trials):
            for method in list(report.METHODS)[:count]:
                rows.append(dict(dataset="fixture", k=k, trial=trial, seed=42 + trial,
                                 method=method, basis_seconds=0.01, selection_seconds=0.02,
                                 reconstruction_seconds=0.03 + trial * 0.02,
                                 algorithm_seconds=0.06 + trial * 0.02,
                                 relative_frobenius_error=0.2 + trial * 0.1,
                                 optimal_relative_error=0.1))
    write_rows(path, rows)
    return rows


def write_rows(path, rows):
    with (path / "raw_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.mark.parametrize("count", [4, 6])
def test_statistics_from_same_trials(tmp_path, count):
    fixture_run(tmp_path / "run", count=count)
    summary, source = report.load_run(tmp_path / "run")
    assert len(summary) == count * 2
    assert len(source["raw_results_sha256"]) == 64
    for row in summary:
        assert row["relative_error_mean"] == pytest.approx(0.25)
        assert row["relative_error_std"] == pytest.approx(np.std([0.2, 0.3], ddof=1))
        assert row["algorithm_seconds_median"] == pytest.approx(0.07)
        assert row["algorithm_seconds_p25"] == pytest.approx(0.065)
        assert row["algorithm_seconds_p75"] == pytest.approx(0.075)


@pytest.mark.parametrize("case", ["duplicate", "missing", "dataset", "seed", "time", "nan", "baseline"])
def test_reject_mismatched_or_invalid_rows(tmp_path, case):
    path = tmp_path / "run"
    rows = fixture_run(path)
    if case == "duplicate": rows.append(rows[0].copy())
    elif case == "missing": rows.pop()
    elif case == "dataset": rows[0]["dataset"] = "different"
    elif case == "seed": rows[0]["seed"] = 999
    elif case == "time": rows[0]["algorithm_seconds"] = 1
    elif case == "nan": rows[0]["relative_frobenius_error"] = float("nan")
    elif case == "baseline": rows[0]["optimal_relative_error"] = 0.15
    write_rows(path, rows)
    with pytest.raises(ValueError): report.load_run(path)


def test_reject_incomplete_run(tmp_path):
    path = tmp_path / "run"
    fixture_run(path)
    config = json.loads((path / "config.json").read_text())
    config["status"] = "incomplete"
    (path / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="complete E7"):
        report.load_run(path)


def test_single_trial_has_no_sample_sd(tmp_path):
    fixture_run(tmp_path / "run", trials=1)
    summary, _ = report.load_run(tmp_path / "run")
    assert all(np.isnan(row["relative_error_std"]) for row in summary)


def test_output_separates_runs_and_ranks_and_preserves_sources(tmp_path):
    pytest.importorskip("matplotlib")
    paths = [tmp_path / "first", tmp_path / "second"]
    for path in paths: fixture_run(path, trials=1)
    before = {file: file.read_bytes() for path in paths for file in path.iterdir()}
    out = report.generate_report(paths, tmp_path / "output")
    assert len(list((out / "plots").glob("*.png"))) == 4
    config = json.loads((out / "config.json").read_text())
    assert config["status"] == "complete" and config["analysis_only"]
    assert len(config["sources"]) == 2
    with (out / "summary.csv").open() as stream: rows = list(csv.DictReader(stream))
    assert len(rows) == 24
    assert {r["source_id"] for r in rows} == {"1", "2"}
    assert all(file.read_bytes() == data for file, data in before.items())
    with pytest.raises(ValueError, match="already exists"):
        report.generate_report(paths, out)
    with pytest.raises(ValueError, match="duplicate"):
        report.generate_report([paths[0], paths[0]], tmp_path / "duplicate")


def test_plot_failure_marks_report_incomplete(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    fixture_run(tmp_path / "run")
    def fail(*args):
        raise RuntimeError("plot failed")
    monkeypatch.setattr(report, "plot_rank", fail)
    with pytest.raises(RuntimeError, match="plot failed"):
        report.generate_report([tmp_path / "run"], tmp_path / "output")
    assert json.loads((tmp_path / "output/config.json").read_text())["status"] == "incomplete"
