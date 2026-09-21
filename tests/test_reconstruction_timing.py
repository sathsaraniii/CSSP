"""Deterministic timing checks; no sleeps or wall-clock thresholds.

Verify that algorithm seconds = basis seconds + selection_seconds + reconstruction seconds.

They also verify that:
- Making the unused reconstruction slower does not change the reported algorithm time.
- Baseline calculation does not enter that algorithm time.
- Both reconstruction outputs remain available.
- Reconstruction aliases select the correct result.

These use a controlled fake clock to test accounting reliably. They do not measure real algorithm speed."""

from __future__ import annotations

import numpy as np
import pytest

import cssp.cssp_pipeline as pipeline


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        now = 0.0

        def __init__(self):
            self.calls = []
            self.costs = {
                "deterministic_svd": 2.0,
                "randomized_svd": 3.0,
                "osinsky_select": 5.0,
                "arp_select": 5.0,
                "rejection_arp_select": 5.0,
                "selected_columns": 7.0,
                "orthogonal_cssp_approximation": 11.0,
                "oblique_cssp_approximation": 13.0,
                "evaluate_approximation": 17.0,
                "optimal_rank_k_error": 101.0,
            }

        def wrap(self, name, original):
            def timed(*args, **kwargs):
                self.calls.append(name)
                self.now += self.costs[name]
                return original(*args, **kwargs)
            return timed

    clock = Clock()
    monkeypatch.setattr(pipeline, "perf_counter", lambda: clock.now)
    for name in clock.costs:
        monkeypatch.setattr(pipeline, name, clock.wrap(name, getattr(pipeline, name)))
    return clock


@pytest.mark.parametrize(
    "runner", [pipeline.run_cssp_pipeline, pipeline.run_four_pipelines, pipeline.run_six_pipelines]
)
@pytest.mark.parametrize(
    "requested,canonical", [("orthogonal", "orthogonal"), ("least-squares", "orthogonal"),
                             ("oblique", "oblique"), ("interpolatory", "oblique")]
)
def test_only_requested_reconstruction_is_counted_in_algorithm_time(clock, runner, requested, canonical):
    output = runner(np.diag([3.0, 2.0, 1.0]), 1, reconstruction=requested, random_state=0)
    results = [output] if runner is pipeline.run_cssp_pipeline else list(output.values())

    for result in results:
        timings = result["timings"]
        basis = 2.0 if result["row_space_method"] == "deterministic" else 3.0
        selected_method = 11.0 if canonical == "orthogonal" else 13.0
        assert timings["basis_seconds"] == basis
        assert timings["selection_seconds"] == 5.0
        assert timings["column_extraction_seconds"] == 7.0
        assert timings["orthogonal_reconstruction_seconds"] == 11.0
        assert timings["oblique_reconstruction_seconds"] == 13.0
        assert timings["reconstruction_seconds"] == 7.0 + selected_method
        assert timings["algorithm_seconds"] == basis + 5.0 + 7.0 + selected_method
        assert timings["evaluation_seconds"] == 51.0
        # Keep the prior total's coverage; the 101-second baseline is excluded.
        assert timings["total_seconds"] == basis + 5.0 + 7.0 + 11.0 + 13.0 + 51.0
        assert "optimal_baseline_seconds" not in timings
        assert "comparison_total_seconds" not in timings
        assert result["approximation"] is result[f"{canonical}_approximation"]
        assert result["metrics"] is result[f"{canonical}_metrics"]

    reconstruction_calls = [name for name in clock.calls if name.endswith("cssp_approximation")]
    other = "oblique" if canonical == "orthogonal" else "orthogonal"
    assert reconstruction_calls == [
        f"{canonical}_cssp_approximation", f"{other}_cssp_approximation"
    ] * len(results)


@pytest.mark.parametrize("reconstruction", ["orthogonal", "oblique"])
def test_slower_alternative_does_not_inflate_chosen_algorithm_time(clock, reconstruction):
    A = np.diag([3.0, 2.0, 1.0])
    first = pipeline.run_cssp_pipeline(A, 1, reconstruction=reconstruction)
    other = "oblique" if reconstruction == "orthogonal" else "orthogonal"
    clock.costs[f"{other}_cssp_approximation"] += 1000.0
    second = pipeline.run_cssp_pipeline(A, 1, reconstruction=reconstruction)

    assert first["timings"]["algorithm_seconds"] == second["timings"]["algorithm_seconds"]
    assert second["timings"]["total_seconds"] == first["timings"]["total_seconds"] + 1000.0
    np.testing.assert_array_equal(first["approximation"], second["approximation"])


def test_optimal_baseline_stays_outside_reported_times(clock):
    A = np.diag([3.0, 2.0, 1.0])
    with_baseline = pipeline.run_cssp_pipeline(A, 1, compute_optimal=True)
    without_baseline = pipeline.run_cssp_pipeline(A, 1, compute_optimal=False)

    assert with_baseline["timings"] == without_baseline["timings"]
    assert clock.calls.count("optimal_rank_k_error") == 1
