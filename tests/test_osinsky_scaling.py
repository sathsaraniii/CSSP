"""Scale robustness and MATLAB-reference checks for Osinsky selection.

These compare the selected indices against an independent Python translation of
the authors’ MATLAB procedure.

They also check that:
- Multiplying the input matrix by very small or large factors does not change the selection.
- Intermediate calculations avoid overflow or invalid division in the tested cases.
- The zero matrix is handled.
- Input arrays are not modified.
- Returned columns and approximations retain the original matrix’s scale.

This directly checks both reference agreement and the numerical-scaling fix."""

from __future__ import annotations

import numpy as np
import pytest

from cssp.column_selection.osinsky import osinsky_cssp, osinsky_select


def _matlab_reference_indices(A: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Translate the reference's unsquared norm criterion and reflection.

    Source: Alice94/ARP, selection_methods/get_columns_osinsky.m.
    This deliberately does not call production selector helpers. The
    random, ordinary-scale fixtures avoid the reference's zero-pivot case.
    """
    basis = V.T.copy()
    residual = A - A @ basis.T @ basis
    selected: list[int] = []
    for step in range(basis.shape[0]):
        # MATLAB masks selected columns after division, including 0/0
        # values produced by their exactly eliminated residuals/basis rows.
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = np.linalg.norm(residual, axis=0) / np.linalg.norm(
                basis[step:, :], axis=0
            )
        scores[selected] = np.inf
        assert np.count_nonzero(~np.isfinite(scores)) == len(selected)
        pivot = int(np.argmin(scores))
        selected.append(pivot)
        reflector = basis[step:, pivot].copy()
        reflector[0] -= np.linalg.norm(reflector)
        if np.sum(np.abs(reflector)) == 0.0:
            continue
        reflector /= np.linalg.norm(reflector)
        basis[step:, :] -= 2.0 * np.outer(
            reflector, reflector @ basis[step:, :]
        )
        residual -= np.outer(
            residual[:, pivot], basis[step, :] / basis[step, pivot]
        )
        residual[:, selected] = 0.0
    return np.asarray(selected)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("shape,k", [((12, 9), 3), ((8, 13), 4)])
@pytest.mark.parametrize("basis_type", ["svd", "orthonormal"])
def test_osinsky_matches_matlab_norm_criterion(seed, shape, k, basis_type):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal(shape)
    if basis_type == "svd":
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        V = Vt[:k, :].T
    else:
        V, _ = np.linalg.qr(rng.standard_normal((shape[1], k)))

    np.testing.assert_array_equal(
        osinsky_select(A, V), _matlab_reference_indices(A, V)
    )


@pytest.mark.parametrize("scale", [1e-170, 1e-100, 1.0, 1e100, 1e160])
def test_osinsky_indices_and_normalized_scores_are_scale_invariant(scale):
    rng = np.random.default_rng(103)
    A = rng.standard_normal((17, 12))
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt[:4, :].T
    expected_indices, expected_scores = osinsky_select(A, V, return_scores=True)
    scaled_A = A * scale
    original_A = scaled_A.copy()
    original_V = V.copy()

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        indices, scores = osinsky_select(scaled_A, V, return_scores=True)

    np.testing.assert_array_equal(indices, expected_indices)
    for actual, expected in zip(scores, expected_scores):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=0.0)
    np.testing.assert_array_equal(scaled_A, original_A)
    np.testing.assert_array_equal(V, original_V)
    assert np.linalg.matrix_rank(V[indices, :]) == V.shape[1]


def test_osinsky_handles_zero_matrix_without_dividing_by_zero():
    rng = np.random.default_rng(104)
    V, _ = np.linalg.qr(rng.standard_normal((8, 3)))
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        indices, scores = osinsky_select(np.zeros((5, 8)), V, return_scores=True)

    assert np.unique(indices).size == 3
    assert np.linalg.matrix_rank(V[indices, :]) == 3
    for iteration_scores in scores:
        np.testing.assert_array_equal(
            iteration_scores[np.isfinite(iteration_scores)], 0.0
        )


def test_osinsky_wrapper_preserves_original_matrix_units():
    rng = np.random.default_rng(105)
    A = 17.0 * rng.standard_normal((11, 8))
    result = osinsky_cssp(A, 3)
    J = result["indices"]
    V = result["V"]

    np.testing.assert_array_equal(result["C"], A[:, J])
    coefficients = np.linalg.solve(V[J, :].T, V.T)
    np.testing.assert_allclose(result["approximation"], A[:, J] @ coefficients)
    np.testing.assert_allclose(result["approximation"][:, J], A[:, J])
