"""Tests for the row-space methods, selectors, and four pipelines."""

from __future__ import annotations

import numpy as np

from cssp.basis.deterministic_svd import deterministic_svd
from cssp.basis.randomized_svd import randomized_svd
from cssp.column_selection.arp import arp_select
from cssp.column_selection.osinsky import osinsky_select
from cssp.cssp_pipeline import PIPELINE_CONFIGS, run_four_pipelines


def test_row_space_methods_return_orthonormal_right_bases() -> None:
    rng = np.random.default_rng(3)
    A = rng.standard_normal((30, 20))

    for decomposition in (
        deterministic_svd(A, 5),
        randomized_svd(A, 5, p=4, q=1, random_state=3),
    ):
        U, singular_values, Vt = decomposition
        assert U.shape == (30, 5)
        assert singular_values.shape == (5,)
        assert Vt.shape == (5, 20)
        np.testing.assert_allclose(Vt @ Vt.T, np.eye(5), atol=1e-10)


def test_arp_is_reproducible_and_selects_a_nonsingular_subset() -> None:
    rng = np.random.default_rng(7)
    V, _ = np.linalg.qr(rng.standard_normal((25, 6)), mode="reduced")

    first = arp_select(V, random_state=19)
    second = arp_select(V, random_state=19)

    np.testing.assert_array_equal(first, second)
    assert np.unique(first).size == 6
    assert np.linalg.matrix_rank(V[first, :]) == 6


def test_osinsky_selects_a_nonsingular_subset() -> None:
    rng = np.random.default_rng(11)
    A = rng.standard_normal((35, 25))
    _, _, Vt = deterministic_svd(A, 6)
    V = Vt.T

    indices = osinsky_select(A, V)

    assert indices.shape == (6,)
    assert np.unique(indices).size == 6
    assert np.linalg.matrix_rank(V[indices, :]) == 6


def test_all_four_pipelines_run_and_return_comparable_results() -> None:
    rng = np.random.default_rng(23)
    left = rng.standard_normal((40, 8))
    right = rng.standard_normal((8, 30))
    A = left @ right + 0.01 * rng.standard_normal((40, 30))

    results = run_four_pipelines(
        A,
        5,
        random_state=23,
        oversampling=5,
        power_iterations=1,
    )

    assert set(results) == set(PIPELINE_CONFIGS)
    np.testing.assert_allclose(
        results["deterministic_osinsky"]["V"],
        results["deterministic_arp"]["V"],
    )
    np.testing.assert_allclose(
        results["randomized_osinsky"]["V"],
        results["randomized_arp"]["V"],
    )

    for result in results.values():
        assert result["C"].shape == (40, 5)
        assert result["approximation"].shape == A.shape
        assert np.unique(result["indices"]).size == 5
        assert np.isfinite(result["relative_error"])
        assert result["relative_error"] >= 0.0
        # Orthogonal projection is never worse than the interpolatory one.
        assert (
            result["orthogonal_metrics"]["frobenius_error"]
            <= result["oblique_metrics"]["frobenius_error"] + 1e-10
        )
