"""E4 timing boundaries, repeat design, and benchmark reporting."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "e4", Path(__file__).resolve().parents[1] / "experiments/run_runtime_scaling.py"
)
e4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e4)


def arguments():
    return SimpleNamespace(k=2, p=1, q=1, repeats=3, threads=1, signal_rank=3,
                           relative_noise=.02, data_seed=42, trial_seed=0)


@pytest.mark.parametrize("method", list(e4.SIX_PIPELINE_CONFIGS))
def test_stage_accounting_and_only_one_extraction(monkeypatch, method):
    A = np.eye(4)
    clock = [0.0]
    calls = []
    monkeypatch.setattr(e4, "perf_counter", lambda: clock[0])
    def basis(A, k, **kwargs):
        calls.append("basis")
        clock[0] += 2
        return A[:, :k], np.ones(k), A[:k]
    def select(*args, **kwargs):
        calls.append("select")
        clock[0] += 3
        return np.array([0, 1])
    def reconstruct(A, J):
        calls.append("reconstruct_including_extraction")
        clock[0] += 5
        return A.copy()
    monkeypatch.setattr(e4, "deterministic_svd", basis)
    monkeypatch.setattr(e4, "randomized_svd", basis)
    for name in ("arp_select", "osinsky_select", "rejection_arp_select"):
        monkeypatch.setattr(e4, name, select)
    monkeypatch.setattr(e4, "orthogonal_cssp_approximation", reconstruct)
    row = e4.measure(A, method, arguments(), 12, 34)
    assert [row[name] for name in e4.TIMINGS] == [2, 3, 5, 10]
    assert calls == ["basis", "select", "reconstruct_including_extraction"]


def test_warmups_repetitions_and_reproducible_order(monkeypatch):
    calls = []
    def fake(A, method, args, basis_seed, selector_seed):
        calls.append((method, basis_seed, selector_seed, id(A)))
        return dict(method=method, basis_seed=basis_seed, selector_seed=selector_seed,
                    **{name: 1.0 for name in e4.TIMINGS})
    monkeypatch.setattr(e4, "measure", fake)
    A = np.eye(5)
    rows = list(e4.benchmark_rows(A, arguments()))
    assert len(calls) == 6 * (1 + 3)
    assert len(rows) == 18
    assert len({call[3] for call in calls}) == 1
    assert rows == list(e4.benchmark_rows(A, arguments()))
    for repetition in range(3):
        batch = [row for row in rows if row["repetition"] == repetition]
        assert {row["method"] for row in batch} == set(e4.SIX_PIPELINE_CONFIGS)
        assert [row["execution_position"] for row in batch] == list(range(6))
    assert e4.seeds(0, 5, 0, 0) != e4.seeds(0, 5, 0, 1)


def test_normalized_noise_and_signal_spectrum():
    for size in (10, 20):
        signal = e4.make_matrix(size, 3, 0, 42)
        noisy = e4.make_matrix(size, 3, .02, 42)
        np.testing.assert_array_equal(noisy, e4.make_matrix(size, 3, .02, 42))
        np.testing.assert_allclose(np.linalg.norm(noisy - signal) / np.linalg.norm(signal), .02)
        np.testing.assert_allclose(np.linalg.svd(signal, compute_uv=False)[:3], np.geomspace(20, 1, 3))


def test_summary_and_thread_verification():
    rows = [dict(size=5, k=2, method="arp", **{name: value for name in e4.TIMINGS}) for value in (1., 2., 3.)]
    result = e4.summarize(rows)[0]
    assert result["algorithm_seconds_median"] == 2
    assert result["algorithm_seconds_p25"] == 1.5
    assert result["algorithm_seconds_p75"] == 2.5
    assert result["algorithm_seconds_std"] == 1
    for info in ([], [{"user_api": "blas", "num_threads": 4}]):
        with pytest.raises(RuntimeError, match="Cannot verify"):
            e4.verified_blas(info, 1)
    assert e4.verified_blas([{"user_api": "blas", "num_threads": 1}], 1)


@pytest.mark.parametrize("options", [["--sizes", "10"], ["--repeats", "1"],
                                     ["--threads", "0"], ["--relative-noise", "nan"],
                                     ["--run-name", "../bad"], ["--sizes", "100", "100"]])
def test_invalid_cli(options):
    with pytest.raises(SystemExit):
        e4.parse_args(options)
