"""Bounded checks against independent formulations from the primary papers.

ARP: https://arxiv.org/html/2412.13992v2 (prototype and Householder algorithms).
RSVD: Halko--Martinsson--Tropp, Algorithms 4.4 and 5.1.
Rejection ARP: https://arxiv.org/html/2510.02513v2 (volume sampling, Section 4).
These are Python checks, not executions of the authors' MATLAB programs.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from cssp.basis.randomized_svd import randomized_svd
from cssp.column_selection.arp import arp_select
from cssp.column_selection.rejection_arp import rejection_arp_select
from cssp.cssp_pipeline import run_six_pipelines


@pytest.mark.parametrize("seed", [0, 13, 29])
def test_arp_probabilities_match_direct_projection_reference(seed: int) -> None:
    V, _ = np.linalg.qr(np.random.default_rng(23).standard_normal((9, 3)))
    actual_indices, actual_probabilities = arp_select(
        V, random_state=seed, return_probabilities=True
    )
    residual_basis = V.copy()
    rng = np.random.default_rng(seed)
    selected = []

    for step in range(V.shape[1]):
        weights = np.sum(residual_basis**2, axis=1)
        weights[selected] = 0.0
        probabilities = weights / np.sum(weights)
        np.testing.assert_allclose(
            actual_probabilities[step], probabilities, rtol=1e-12, atol=1e-14
        )
        pivot = int(rng.choice(V.shape[0], p=probabilities))
        selected.append(pivot)
        direction = residual_basis[pivot] / np.linalg.norm(residual_basis[pivot])
        residual_basis -= np.outer(residual_basis @ direction, direction)

    np.testing.assert_array_equal(actual_indices, selected)


@pytest.mark.parametrize("shape", [(12, 9), (9, 12)])
@pytest.mark.parametrize("power_iterations", [0, 1, 2])
def test_randomized_svd_matches_powered_range_reference(shape, power_iterations) -> None:
    A = np.random.default_rng(11).standard_normal(shape)
    k, p, seed = 3, 2, 17
    U, s, Vt = randomized_svd(A, k, p=p, q=power_iterations, random_state=seed)

    # Algebraically equivalent unorthogonalized powers are an independent
    # oracle on these small, well-scaled cases (not a production replacement).
    omega = np.random.default_rng(seed).standard_normal((shape[1], k + p))
    Y = np.linalg.matrix_power(A @ A.T, power_iterations) @ A @ omega
    Q, _ = np.linalg.qr(Y, mode="reduced")
    projected_U, expected_s, expected_Vt = np.linalg.svd(Q.T @ A, full_matrices=False)
    expected = (Q @ projected_U[:, :k] * expected_s[:k]) @ expected_Vt[:k]

    np.testing.assert_allclose(s, expected_s[:k], rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose((U * s) @ Vt, expected, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize("block_size", [None, 1, 2, 7], ids=["arp", "block1", "block2", "block7"])
def test_selectors_match_small_volume_sampling_distribution(block_size) -> None:
    V, _ = np.linalg.qr(np.array([[1., 0.], [0., 1.], [1., 1.], [2., -1.]]))
    subsets = list(combinations(range(V.shape[0]), V.shape[1]))
    expected = np.array([np.linalg.det(V[list(J), :]) ** 2 for J in subsets])
    np.testing.assert_allclose(np.sum(expected), 1.0, atol=1e-14)
    counts = dict.fromkeys(subsets, 0)
    rng = np.random.default_rng(101)
    trials = 2000

    for _ in range(trials):
        if block_size is None:
            indices = arp_select(V, random_state=rng)
        else:
            indices = rejection_arp_select(V, random_state=rng, block_size=block_size)
        counts[tuple(sorted(indices))] += 1

    observed = np.array([counts[J] / trials for J in subsets])
    # A seeded statistical sanity check, not a proof of distributional identity.
    tolerance = 6.0 * np.sqrt(expected * (1.0 - expected) / trials) + 2.0 / trials
    assert np.all(np.abs(observed - expected) <= tolerance)


def test_all_six_reconstructions_match_projection_and_interpolation_references() -> None:
    A = np.random.default_rng(37).standard_normal((12, 9))
    results = run_six_pipelines(A, 3, oversampling=2, power_iterations=1, random_state=19)
    expected_optimal = np.sqrt(np.sum(np.linalg.svd(A, compute_uv=False)[3:] ** 2))

    for result in results.values():
        J = result["indices"]
        Q, _ = np.linalg.qr(A[:, J], mode="reduced")
        orthogonal = result["orthogonal_approximation"]
        oblique = result["oblique_approximation"]
        np.testing.assert_allclose(orthogonal, Q @ (Q.T @ A), rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(A[:, J].T @ (A - orthogonal), 0.0, atol=1e-11)
        np.testing.assert_allclose(oblique[:, J], A[:, J], rtol=1e-11, atol=1e-12)
        np.testing.assert_allclose(
            result["metrics"]["optimal_rank_k_error"], expected_optimal, rtol=1e-12
        )
        np.testing.assert_allclose(
            result["relative_error"],
            np.linalg.norm(A - orthogonal, "fro") / np.linalg.norm(A, "fro"),
            rtol=1e-12,
        )
