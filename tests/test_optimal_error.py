"""Regression tests for the exact singular-value-tail error baseline."""

from __future__ import annotations

import numpy as np
import pytest

import cssp.cssp_pipeline as pipeline
from cssp.metrics.evaluation import optimal_rank_k_error


METRIC_KEYS = (
    "metrics",
    "orthogonal_metrics",
    "oblique_metrics",
    "row_space_metrics",
)


def _assert_baseline(result: dict, expected: float) -> None:
    for key in METRIC_KEYS:
        # A positive absolute tolerance could hide a lost, very small tail.
        np.testing.assert_allclose(
            result[key]["optimal_rank_k_error"], expected, rtol=1e-12, atol=0.0
        )


@pytest.mark.parametrize(
    "row_space_method,column_selection_method",
    list(pipeline.SIX_PIPELINE_CONFIGS.values()),
    ids=list(pipeline.SIX_PIPELINE_CONFIGS),
)
def test_single_pipeline_preserves_small_nonzero_optimal_error(
    row_space_method: str, column_selection_method: str
) -> None:
    A = np.diag([1.0, 1e-8])
    result = pipeline.run_cssp_pipeline(
        A,
        1,
        row_space_method,
        column_selection_method,
        random_state=0,
    )

    _assert_baseline(result, 1e-8)
    np.testing.assert_array_equal(result["indices"], [0])
    ratio = result["metrics"]["frobenius_optimality_ratio"]
    assert np.isfinite(ratio)
    np.testing.assert_allclose(ratio, 1.0, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("runner", [pipeline.run_four_pipelines, pipeline.run_six_pipelines])
@pytest.mark.parametrize(
    "diagonal,k,expected",
    [
        ([1.0, 1e-8], 1, 1e-8),
        ([1.0, 4e-8, 3e-8], 1, 5e-8),
        ([1.0, 2.0], 2, 0.0),
    ],
    ids=["small-tail", "multiple-tail-values", "full-rank"],
)
def test_comparison_runners_share_exact_optimal_error(runner, diagonal, k, expected) -> None:
    results = runner(np.diag(diagonal), k, random_state=0)

    for result in results.values():
        _assert_baseline(result, expected)
        if expected == 0.0:
            for key in METRIC_KEYS:
                assert np.isnan(result[key]["frobenius_optimality_ratio"])
        if k == 1:
            np.testing.assert_array_equal(result["indices"], [0])
            ratio = result["metrics"]["frobenius_optimality_ratio"]
            assert np.isfinite(ratio)
            np.testing.assert_allclose(ratio, 1.0, rtol=1e-12, atol=0.0)


@pytest.mark.parametrize("row_space_method", ["deterministic", "randomized"])
@pytest.mark.parametrize("column_selection_method", ["osinsky", "arp", "rejection_arp"])
def test_single_pipeline_full_rank_has_zero_optimal_error(
    row_space_method: str, column_selection_method: str
) -> None:
    result = pipeline.run_cssp_pipeline(
        np.diag([1.0, 2.0]), 2, row_space_method, column_selection_method, random_state=0
    )

    _assert_baseline(result, 0.0)
    for key in METRIC_KEYS:
        assert np.isnan(result[key]["frobenius_optimality_ratio"])


@pytest.mark.parametrize("shape", [(4, 3), (3, 4)], ids=["tall", "wide"])
@pytest.mark.parametrize("k,expected", [(1, 5e-8), (2, 3e-8), (3, 0.0)])
def test_optimal_error_uses_all_discarded_singular_values(shape, k, expected) -> None:
    A = np.zeros(shape)
    np.fill_diagonal(A, [1.0, 4e-8, 3e-8])

    np.testing.assert_allclose(
        optimal_rank_k_error(A, k), expected, rtol=1e-12, atol=0.0
    )


@pytest.mark.parametrize("row_space_method", ["deterministic", "randomized"])
def test_disabling_optimal_error_skips_baseline_computation(
    monkeypatch, row_space_method: str
) -> None:
    def unexpected_baseline(*args, **kwargs):
        pytest.fail("compute_optimal=False must not compute the optimal baseline")

    monkeypatch.setattr(pipeline, "optimal_rank_k_error", unexpected_baseline)
    result = pipeline.run_cssp_pipeline(
        np.diag([1.0, 1e-8]),
        1,
        row_space_method,
        random_state=0,
        compute_optimal=False,
    )

    for key in METRIC_KEYS:
        assert "optimal_rank_k_error" not in result[key]
        assert "frobenius_optimality_ratio" not in result[key]
