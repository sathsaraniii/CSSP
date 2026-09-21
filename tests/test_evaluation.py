"""Regression tests for reporting errors with a zero optimal baseline."""

from __future__ import annotations

import numpy as np
import pytest

from cssp.metrics.evaluation import evaluate_approximation, optimal_rank_k_error


@pytest.mark.parametrize("scale", [1e-16, 1.0, 1e16])
@pytest.mark.parametrize("retained_fraction", [0.0, 0.5, 1.0])
def test_zero_optimal_baseline_has_no_ratio_at_any_scale(
    scale: float, retained_fraction: float
) -> None:
    A = np.diag([scale, 0.0])
    approximation = retained_fraction * A
    optimal_error = optimal_rank_k_error(A, 1)

    metrics = evaluate_approximation(A, approximation, optimal_error)

    assert metrics["optimal_rank_k_error"] == 0.0
    assert np.isnan(metrics["frobenius_optimality_ratio"])
    np.testing.assert_allclose(
        metrics["frobenius_error"],
        scale * (1.0 - retained_fraction),
        rtol=1e-12,
        atol=0.0,
    )
    np.testing.assert_allclose(
        metrics["relative_frobenius_error"],
        1.0 - retained_fraction,
        rtol=1e-12,
        atol=0.0,
    )


def test_zero_baseline_does_not_classify_roundoff_as_an_optimality_ratio() -> None:
    A = np.diag([1.0, 0.0])
    approximation = A.copy()
    approximation[0, 0] = np.nextafter(1.0, 0.0)

    metrics = evaluate_approximation(A, approximation, optimal_error=0.0)

    assert 0.0 < metrics["frobenius_error"] < np.finfo(float).eps
    assert np.isnan(metrics["frobenius_optimality_ratio"])


@pytest.mark.parametrize("exact", [True, False])
def test_zero_matrix_retains_error_metrics_but_has_no_optimality_ratio(exact: bool) -> None:
    A = np.zeros((2, 2))
    approximation = A.copy() if exact else np.eye(2)

    metrics = evaluate_approximation(A, approximation, optimal_error=0.0)

    assert np.isnan(metrics["frobenius_optimality_ratio"])
    if exact:
        assert metrics["frobenius_error"] == 0.0
        assert metrics["relative_frobenius_error"] == 0.0
    else:
        np.testing.assert_allclose(metrics["frobenius_error"], np.sqrt(2.0))
        assert metrics["relative_frobenius_error"] == float("inf")


@pytest.mark.parametrize("tail", [1e-16, 1e-8, 0.5])
def test_positive_optimal_baseline_keeps_the_normal_ratio(tail: float) -> None:
    A = np.diag([1.0, tail])
    approximation = np.diag([1.0, 0.0])

    metrics = evaluate_approximation(A, approximation, optimal_error=tail)

    np.testing.assert_allclose(
        metrics["frobenius_optimality_ratio"], 1.0, rtol=1e-12, atol=0.0
    )


def test_absent_optimal_baseline_omits_optional_metrics() -> None:
    metrics = evaluate_approximation(np.eye(2), np.eye(2))

    assert "optimal_rank_k_error" not in metrics
    assert "frobenius_optimality_ratio" not in metrics


@pytest.mark.parametrize("optimal_error", [-1.0, np.nan, np.inf, -np.inf])
def test_invalid_optimal_baselines_are_rejected(optimal_error: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        evaluate_approximation(np.eye(2), np.zeros((2, 2)), optimal_error)
