"""Public inputs must fail clearly before numerical work or lossy coercion.

These ensure incorrect inputs fail clearly rather than silently producing misleading results.

Examples include:
- Invalid ranks, oversampling values, iteration counts, and block sizes.
- Empty, incorrectly shaped, complex, or nonfinite matrices.
- Nonorthonormal bases.
- Duplicate, out-of-range, Boolean, or fractional column indices.
- Valid NumPy integer values that should be accepted."""

from __future__ import annotations

import numpy as np
import pytest

from cssp.basis.deterministic_svd import deterministic_svd
from cssp.basis.randomized_svd import randomized_svd
from cssp.column_selection.arp import arp_cssp, arp_select
from cssp.column_selection.osinsky import osinsky_cssp, osinsky_select
from cssp.column_selection.rejection_arp import rejection_arp_select
from cssp.cssp_pipeline import run_cssp_pipeline, run_four_pipelines, run_six_pipelines
from cssp.metrics.evaluation import (
    evaluate_approximation,
    frobenius_error,
    optimal_rank_k_error,
    relative_frobenius_error,
)
from cssp.reconstruction.cssp_approximation import (
    oblique_cssp_approximation,
    orthogonal_cssp_approximation,
    reconstruct_cssp,
    selected_columns,
)


RANK_METHODS = [
    deterministic_svd,
    randomized_svd,
    optimal_rank_k_error,
    arp_cssp,
    osinsky_cssp,
    run_cssp_pipeline,
    run_four_pipelines,
    run_six_pipelines,
]


@pytest.mark.parametrize("method", RANK_METHODS)
@pytest.mark.parametrize("k", [True, np.bool_(False), 1.0, 1.5, "1", 0, -1, 4])
def test_invalid_ranks_are_rejected_before_svd(method, k, monkeypatch) -> None:
    def unexpected_svd(*args, **kwargs):
        pytest.fail("SVD was called before validating k")

    monkeypatch.setattr(np.linalg, "svd", unexpected_svd)
    with pytest.raises(ValueError, match="k must"):
        method(np.eye(3), k)


@pytest.mark.parametrize("parameter", ["p", "q"])
@pytest.mark.parametrize("value", [True, np.bool_(True), 1.0, 1.5, "1", -1])
@pytest.mark.parametrize("size", [1, 4])
def test_randomized_parameters_are_validated_even_when_sketch_is_clipped(
    parameter, value, size, monkeypatch
) -> None:
    def unexpected_rng(*args, **kwargs):
        pytest.fail("Random sampling began before parameter validation")

    monkeypatch.setattr(np.random, "default_rng", unexpected_rng)
    with pytest.raises(ValueError, match=rf"{parameter} must"):
        randomized_svd(np.eye(size), 1, **{parameter: value})


@pytest.mark.parametrize("runner", [run_cssp_pipeline, run_four_pipelines, run_six_pipelines])
@pytest.mark.parametrize("parameter", ["oversampling", "power_iterations"])
@pytest.mark.parametrize("value", [True, 1.5, -1])
def test_pipeline_parameters_are_validated_before_either_basis(
    runner, parameter, value, monkeypatch
) -> None:
    def unexpected_svd(*args, **kwargs):
        pytest.fail("SVD was called before validating pipeline parameters")

    monkeypatch.setattr(np.linalg, "svd", unexpected_svd)
    with pytest.raises(ValueError, match=rf"{parameter} must"):
        runner(np.eye(3), 1, **{parameter: value})


@pytest.mark.parametrize("block_size", [True, np.bool_(True), 1.0, 1.5, "1", 0, -1])
def test_invalid_rejection_block_sizes_are_rejected(block_size) -> None:
    with pytest.raises(ValueError, match="block_size must"):
        rejection_arp_select(np.eye(3), block_size=block_size)


@pytest.mark.parametrize("method", RANK_METHODS)
def test_numpy_integer_ranks_are_supported(method) -> None:
    method(np.diag([3.0, 2.0, 1.0]), np.int64(1))


def test_numpy_integer_randomized_parameters_are_supported() -> None:
    U, singular_values, Vt = randomized_svd(
        np.eye(3), np.int64(1), p=np.uint64(1), q=np.int32(0), random_state=0
    )
    assert U.shape == (3, 1)
    assert singular_values.shape == (1,)
    assert Vt.shape == (1, 3)
    assert rejection_arp_select(
        np.eye(3), block_size=np.int64(2), random_state=0
    ).size == 3


INVALID_MATRICES = [
    pytest.param(np.ones(3), id="one-dimensional"),
    pytest.param(np.ones((1, 1, 1)), id="three-dimensional"),
    pytest.param(np.empty((0, 3)), id="empty-rows"),
    pytest.param(np.empty((3, 0)), id="empty-columns"),
    pytest.param(np.array([[np.nan]]), id="nan"),
    pytest.param(np.array([[np.inf]]), id="infinity"),
    pytest.param(np.array([[-np.inf]]), id="negative-infinity"),
    pytest.param(np.array([[1.0 + 1.0j]]), id="complex"),
]

MATRIX_METHODS = [
    pytest.param(lambda A: deterministic_svd(A, 1), id="deterministic-svd"),
    pytest.param(lambda A: randomized_svd(A, 1), id="randomized-svd"),
    pytest.param(lambda A: optimal_rank_k_error(A, 1), id="optimal-error"),
    pytest.param(lambda A: arp_cssp(A, 1), id="arp-wrapper"),
    pytest.param(lambda A: osinsky_cssp(A, 1), id="osinsky-wrapper"),
    pytest.param(lambda A: osinsky_select(A, np.eye(3)), id="osinsky-selector"),
    pytest.param(lambda A: run_cssp_pipeline(A, 1), id="single-pipeline"),
    pytest.param(lambda A: run_four_pipelines(A, 1), id="four-pipelines"),
    pytest.param(lambda A: run_six_pipelines(A, 1), id="six-pipelines"),
    pytest.param(lambda A: selected_columns(A, [0]), id="selected-columns"),
    pytest.param(lambda A: orthogonal_cssp_approximation(A, [0]), id="orthogonal"),
    pytest.param(
        lambda A: oblique_cssp_approximation(A, np.eye(3), [0]), id="oblique"
    ),
    pytest.param(lambda A: reconstruct_cssp(A, [0]), id="reconstruction-dispatch"),
    pytest.param(lambda A: frobenius_error(A, A), id="frobenius-error"),
    pytest.param(lambda A: relative_frobenius_error(A, A), id="relative-error"),
    pytest.param(lambda A: evaluate_approximation(A, A), id="evaluation"),
]


@pytest.mark.parametrize("method", MATRIX_METHODS)
@pytest.mark.parametrize("A", INVALID_MATRICES)
def test_public_matrix_inputs_are_nonempty_finite_and_real(method, A) -> None:
    with pytest.raises(ValueError):
        method(A)


@pytest.mark.parametrize("V", INVALID_MATRICES)
@pytest.mark.parametrize(
    "method",
    [
        arp_select,
        rejection_arp_select,
        lambda V: osinsky_select(np.eye(3), V),
        lambda V: oblique_cssp_approximation(np.eye(3), V, [0]),
    ],
)
def test_public_basis_inputs_are_nonempty_finite_and_real(method, V) -> None:
    with pytest.raises(ValueError):
        method(V)


@pytest.mark.parametrize("approximation", INVALID_MATRICES)
@pytest.mark.parametrize(
    "method", [frobenius_error, relative_frobenius_error, evaluate_approximation]
)
def test_metric_approximation_is_also_validated(method, approximation) -> None:
    with pytest.raises(ValueError):
        method(np.eye(3), approximation)


RECONSTRUCTION_METHODS = [
    selected_columns,
    orthogonal_cssp_approximation,
    reconstruct_cssp,
    lambda A, indices: oblique_cssp_approximation(A, np.eye(3), indices),
]


@pytest.mark.parametrize("method", RECONSTRUCTION_METHODS)
@pytest.mark.parametrize(
    "indices",
    [[1.9], [-0.1], [1.0], [True], [np.bool_(False)], [0, True], ["2"], [1 + 0j]],
)
def test_indices_are_checked_before_lossy_conversion(method, indices) -> None:
    with pytest.raises(ValueError, match="only integers"):
        method(np.eye(3), indices)


@pytest.mark.parametrize("method", RECONSTRUCTION_METHODS)
@pytest.mark.parametrize("indices", [[], [[0]], 1, [0, 0]])
def test_empty_multidimensional_and_duplicate_indices_are_rejected(method, indices) -> None:
    with pytest.raises(ValueError):
        method(np.eye(3), indices)


@pytest.mark.parametrize("method", RECONSTRUCTION_METHODS)
@pytest.mark.parametrize("indices", [[-1], [3], [2**100], np.array([2**64 - 1], dtype=np.uint64)])
def test_index_bounds_are_checked_before_machine_integer_conversion(method, indices) -> None:
    with pytest.raises(IndexError, match="outside"):
        method(np.eye(3), indices)


@pytest.mark.parametrize("indices", [[2, 0, 1], np.array([2, 0, 1], dtype=np.uint64)])
def test_valid_integer_indices_keep_their_order(indices) -> None:
    A = np.diag([3.0, 2.0, 1.0])
    np.testing.assert_array_equal(selected_columns(A, indices), A[:, [2, 0, 1]])
    np.testing.assert_allclose(orthogonal_cssp_approximation(A, indices), A)
    np.testing.assert_allclose(oblique_cssp_approximation(A, np.eye(3), indices), A)


@pytest.mark.parametrize("method", [arp_select, rejection_arp_select])
def test_nonorthonormal_bases_are_still_rejected(method) -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        method(np.ones((3, 2)))
