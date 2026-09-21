"""E3 design, numerical reporting, and output safety regressions."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "e3", Path(__file__).resolve().parents[1] / "experiments/run_randomization_variability.py"
)
e3 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e3)


def arguments(**kwargs):
    return SimpleNamespace(**dict(dict(k=2, trials=3, trial_seed=0, data_seed=42), **kwargs))


def test_fixed_basis_counts_and_reproducibility(monkeypatch):
    A = np.random.default_rng(4).normal(size=(9, 7))
    before = A.copy()
    svd = np.linalg.svd
    calls, bases = [], []
    def tracked_svd(*args, **kwargs):
        calls.append(1)
        return svd(*args, **kwargs)
    monkeypatch.setattr(np.linalg, "svd", tracked_svd)
    for name in ("arp_select", "rejection_arp_select"):
        original = getattr(e3, name)
        def tracked(V, _select=original, **kwargs):
            bases.append(V)
            return _select(V, **kwargs)
        monkeypatch.setattr(e3, name, tracked)
    rows = list(e3.dataset_rows(A, "synthetic", None, arguments()))
    assert len(calls) == 1
    assert len(rows) == 7
    assert all(V is bases[0] for V in bases)
    assert rows == list(e3.dataset_rows(A, "synthetic", None, arguments()))
    np.testing.assert_array_equal(A, before)
    assert len({row["selector_seed"] for row in rows[1:]}) == 6
    for row in rows:
        J = json.loads(row["indices"])
        Q, _ = np.linalg.qr(A[:, J], mode="reduced")
        np.testing.assert_allclose(row["frobenius_error"], np.linalg.norm(A - Q @ Q.T @ A, "fro"))
        assert row["svd_optimality_ratio"] >= 1 - 1e-12


def test_dna_labels_and_zero_baseline():
    rows = list(e3.dataset_rows(np.eye(2), "dna", ["a", "b"], arguments()))
    for row in rows:
        assert row["data_seed"] is None
        assert set(json.loads(row["sample_ids"])) == {"a", "b"}
        assert np.isnan(row["svd_optimality_ratio"])
        assert row["comparison_vs_osinsky"] == "tied"
    assert all(np.isnan(row["svd_optimality_ratio_mean"]) for row in e3.summarize(rows))


def test_summary_and_tolerance():
    rows = []
    for value in (1., 2., 3.):
        rows.append(dict(dataset="synthetic", k=2, method="arp", osinsky_error=2.,
                         comparison_tolerance=0.1, comparison_vs_osinsky=e3.classify(value - 2, 0.1),
                         **{metric: value for metric in e3.METRICS}))
    result = e3.summarize(rows)[0]
    assert result["relative_error_mean"] == 2
    assert result["relative_error_std"] == 1
    assert result["relative_error_p05"] == pytest.approx(1.1)
    assert result["relative_error_p95"] == pytest.approx(2.9)
    assert result["better_fraction"] == result["tied_fraction"] == result["worse_fraction"] == 1 / 3
    assert e3.classify(-0.1, 0.1) == e3.classify(0.1, 0.1) == "tied"


@pytest.mark.parametrize("options", [["--k", "0"], ["--trials", "1"], ["--run-name", "../bad"],
                                     ["--run-name", ""], ["--noise", "nan"], ["--datasets", "dna", "dna"]])
def test_invalid_cli(options):
    with pytest.raises(SystemExit):
        e3.parse_args(options)


def test_csv_output_and_no_overwrite(tmp_path, monkeypatch):
    args = e3.parse_args(["--datasets", "synthetic", "--m", "8", "--n", "6", "--k", "2",
                         "--trials", "2", "--no-plots", "--run-name", "pilot"])
    monkeypatch.setattr(e3, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(e3, "parse_args", lambda: args)
    e3.main()
    output = tmp_path / "results/e3/pilot"
    assert json.loads((output / "config.json").read_text())["status"] == "complete"
    original = (output / "raw_results.csv").read_bytes()
    with (output / "raw_results.csv").open() as stream:
        assert len(list(e3.csv.DictReader(stream))) == 5
    with pytest.raises(SystemExit, match="already exists"):
        e3.main()
    assert (output / "raw_results.csv").read_bytes() == original
