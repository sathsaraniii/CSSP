"""Osinsky's deterministic column subset selection algorithm.

This is a NumPy translation of ``ARP/ARP/selection_methods/``
``get_columns_osinsky.m`` from the authors' reference repository. The
selector accepts a precomputed ``n x k`` row-space basis so it can be paired
fairly with either deterministic or randomized SVD.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cssp.column_selection.arp import _as_orthonormal_basis, _householder_vector


Array = NDArray[np.float64]


def osinsky_select(
    A: Array,
    V: Array,
    verbose: bool = False,
    return_scores: bool = False,
) -> Array | tuple[Array, list[Array]]:
    """Select ``k`` columns deterministically using Osinsky's criterion.

    Parameters
    ----------
    A:
        Input matrix with shape ``(m, n)``.
    V:
        Orthonormal row-space basis with shape ``(n, k)``.
    verbose:
        Print the selected index and score at every step.
    return_scores:
        If true, also return the score vector from each iteration.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2-D matrix.")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must contain only finite values.")

    V_basis = _as_orthonormal_basis(V)
    if V_basis.shape[0] != A.shape[1]:
        raise ValueError(
            "A and V are incompatible: "
            f"A has {A.shape[1]} columns but V has {V_basis.shape[0]} rows."
        )

    n, k = V_basis.shape
    # The MATLAB reference stores the working basis as k x n.
    V_work = V_basis.T.copy()
    residual = A - (A @ V_basis) @ V_basis.T

    selected: list[int] = []
    score_history: list[Array] = []

    for step in range(k):
        residual_norm_sq = np.sum(residual * residual, axis=0)
        basis_norm_sq = np.sum(V_work[step:, :] ** 2, axis=0)

        scores = np.full(n, np.inf, dtype=float)
        threshold = np.finfo(float).eps * max(1.0, float(np.max(basis_norm_sq)))
        valid = basis_norm_sq > threshold
        if selected:
            valid[np.asarray(selected, dtype=int)] = False
        scores[valid] = residual_norm_sq[valid] / basis_norm_sq[valid]

        if not np.any(np.isfinite(scores)):
            raise RuntimeError("No finite Osinsky selection score remains.")

        score_history.append(scores.copy())
        pivot = int(np.argmin(scores))
        selected.append(pivot)

        if verbose:
            print(
                f"Osinsky step {step + 1}/{k}: column={pivot}, "
                f"score={scores[pivot]:.6e}"
            )

        # Rotate the selected trailing basis column onto coordinate `step`.
        reflector = _householder_vector(V_work[step:, pivot])
        if reflector is not None:
            product = reflector @ V_work[step:, :]
            V_work[step:, :] -= 2.0 * np.outer(reflector, product)

        pivot_value = float(V_work[step, pivot])
        pivot_tolerance = np.finfo(float).eps * max(
            1.0, float(np.linalg.norm(V_work[step, :]))
        )
        if abs(pivot_value) <= pivot_tolerance:
            raise RuntimeError("Osinsky encountered a numerically zero pivot.")

        # This is the dimensionally correct residual update from the MATLAB
        # reference: (m x 1) outer (1 x n), not (m x 1) outer (1 x k).
        residual -= np.outer(
            residual[:, pivot], V_work[step, :] / pivot_value
        )
        residual[:, np.asarray(selected, dtype=int)] = 0.0

    indices = np.asarray(selected, dtype=int)
    if np.unique(indices).size != k:
        raise RuntimeError("Osinsky selection produced duplicate indices.")
    if np.linalg.matrix_rank(V_basis[indices, :]) < k:
        raise RuntimeError(
            "Osinsky selected a numerically singular V[indices, :]."
        )

    if return_scores:
        return indices, score_history
    return indices


def osinsky_cssp(
    A: Array,
    k: int,
    verbose: bool = False,
    V: Array | None = None,
) -> dict:
    """Backward-compatible end-to-end wrapper for deterministic CSSP.

    If ``V`` is not supplied, this function computes the exact top-``k``
    right singular basis. The returned approximation is the oblique
    interpolatory approximation used in the paper's analysis.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2-D matrix.")
    if not 1 <= k <= min(A.shape):
        raise ValueError(f"k must satisfy 1 <= k <= min(A.shape); got {k}.")

    if V is None:
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        V_basis = Vt[:k, :].T
    else:
        V_basis = _as_orthonormal_basis(V)
        if V_basis.shape != (A.shape[1], k):
            raise ValueError(
                f"V must have shape {(A.shape[1], k)}; got {V_basis.shape}."
            )

    indices, scores = osinsky_select(
        A, V_basis, verbose=verbose, return_scores=True
    )
    C = A[:, indices]
    coefficients = np.linalg.solve(V_basis[indices, :].T, V_basis.T)
    approximation = C @ coefficients
    error = float(np.linalg.norm(A - approximation, ord="fro"))
    norm_A = float(np.linalg.norm(A, ord="fro"))

    return {
        "indices": indices,
        "C": C,
        "V": V_basis,
        "scores": scores,
        "approximation": approximation,
        "error": error,
        "relative_error": error / norm_A if norm_A else 0.0,
    }


__all__ = ["osinsky_select", "osinsky_cssp"]
