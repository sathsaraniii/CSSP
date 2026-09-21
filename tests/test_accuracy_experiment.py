"""Bounded checks of the E2 experiment design and aggregation."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

spec = importlib.util.spec_from_file_location(
    "e2", Path(__file__).resolve().parents[1] / "experiments/run_accuracy_vs_k.py"
)
e2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e2)


def test_trial_counts_seeds_and_reconstruction():
    A = np.random.default_rng(4).normal(size=(9, 7))
    args = SimpleNamespace(ks=[2, 3], trials=2, trial_seed=0, data_seed=4, p=1, q=1)
    rows = list(e2.dataset_rows(A, "synthetic", None, args))
    repeat = list(e2.dataset_rows(A, "synthetic", None, args))
    assert rows == repeat
    assert len(rows) == 2 * (1 + 5 * 2)
    summaries = e2.summarize(rows)
    assert len(summaries) == 12
    for row in summaries:
        assert row["runs"] == (1 if row["method"] == "deterministic_osinsky" else 2)
    for row in rows:
        J = e2.json.loads(row["indices"])
        Q, _ = np.linalg.qr(A[:, J], mode="reduced")
        expected = np.linalg.norm(A - Q @ Q.T @ A, "fro")
        np.testing.assert_allclose(row["frobenius_error"], expected)
        assert row["svd_optimality_ratio"] >= 1 - 1e-12
    for k in args.ks:
        for trial in range(args.trials):
            randomized = [row for row in rows if row["k"] == k and row["trial"] == trial
                          and row["method"].startswith("randomized")]
            assert len({row["basis_seed"] for row in randomized}) == 1
            assert len(set(e2.trial_seeds(0, 0, k, trial))) == 3


def test_exact_svd_only_once_and_random_basis_shared(monkeypatch):
    A = np.random.default_rng(5).normal(size=(9, 7))
    args = SimpleNamespace(ks=[2, 3], trials=2, trial_seed=0, data_seed=5, p=1, q=1)
    exact_calls = []
    randomized_calls = []
    svd = np.linalg.svd
    def tracked_svd(matrix, **kwargs):
        if matrix is A:
            exact_calls.append(1)
        return svd(matrix, **kwargs)
    def fake_rsvd(matrix, k, **kwargs):
        randomized_calls.append((k, kwargs["random_state"]))
        U, s, Vt = svd(matrix, full_matrices=False)
        return U[:, :k], s[:k], Vt[:k]
    monkeypatch.setattr(np.linalg, "svd", tracked_svd)
    monkeypatch.setattr(e2, "randomized_svd", fake_rsvd)
    list(e2.dataset_rows(A, "synthetic", None, args))
    assert len(exact_calls) == 1
    assert len(randomized_calls) == 4


def test_dna_labels_and_zero_baseline_summary():
    args = SimpleNamespace(ks=[2], trials=2, trial_seed=0, data_seed=42, p=0, q=0)
    rows = list(e2.dataset_rows(np.eye(2), "dna", ["sample_a", "sample_b"], args))
    for row in rows:
        assert row["data_seed"] is None
        assert set(e2.json.loads(row["sample_ids"])) == {"sample_a", "sample_b"}
        assert np.isnan(row["svd_optimality_ratio"])
    for row in e2.summarize(rows):
        assert np.isnan(row["svd_optimality_ratio_mean"])


def test_summary_sample_standard_deviation():
    rows = [dict(dataset="synthetic", k=1, method="arp", svd_relative_error=0.1,
                 **{metric: value for metric in e2.METRICS}) for value in (1., 3.)]
    summary = e2.summarize(rows)[0]
    assert summary["relative_error_mean"] == 2
    np.testing.assert_allclose(summary["relative_error_std"], np.sqrt(2))
