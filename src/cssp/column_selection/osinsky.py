"""
Osinsky-style deterministic Column Subset Selection (CSSP).

Based on:
A. Osinsky, "Volume-based subset selection", Numerical Linear Algebra
with Applications, 2024.

The deterministic selection procedure can also be viewed as the
derandomization of Adaptive Randomized Pivoting (ARP), as described in:

A. Cortinovis and D. Kressner,
"Adaptive Randomized Pivoting for Column Subset Selection, DEIM,
and Low-Rank Approximation", SIAM J. Matrix Anal. Appl., 2026.

Input:
    A : m x n matrix
    k : number of columns to select

Output:
    indices : selected column indices
    C       : selected columns A[:, indices]
    A_hat   : CSSP approximation
    error   : relative Frobenius reconstruction error

The implementation assumes k <= min(m, n).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


def _validate_input(A: Array, k: int) -> None:
    """Validate CSSP input."""
    if A.ndim != 2:
        raise ValueError("A must be a 2-D matrix.")

    m, n = A.shape

    if not 1 <= k <= min(m, n):
        raise ValueError(
            f"k must satisfy 1 <= k <= min(m, n). "
            f"Got k={k}, matrix shape={A.shape}."
        )


def _orthonormal_row_space(A: Array, k: int) -> Array:
    """
    Compute V_k, an n x k orthonormal basis for the dominant
    right singular subspace of A.

    A = U Sigma V^T

    Returns:
        V_k : n x k matrix
    """
    _, _, Vt = np.linalg.svd(A, full_matrices=False)
    return Vt[:k, :].T


def _project_out_selected_row(
    V: Array,
    j: int,
) -> Array:
    """
    Remove the selected row direction from V.

    Given V with orthonormal columns and the selected row v_j,
    project every row of V onto the orthogonal complement of v_j.

        V_new = V - (V v_j^T / ||v_j||^2) v_j

    The resulting matrix has one fewer effective dimension.

    A small SVD is then used to obtain an orthonormal basis for
    the remaining subspace.

    This is the conceptual equivalent of the row-elimination /
    Householder update used in the paper.
    """
    row = V[j, :]
    row_norm_sq = np.dot(row, row)

    if row_norm_sq <= np.finfo(float).eps:
        raise RuntimeError(
            "Selected row has numerically zero norm. "
            "The CSSP basis may be rank deficient."
        )

    # Project every row onto the orthogonal complement of row.
    V_projected = V - np.outer(V @ row, row) / row_norm_sq

    # Obtain an orthonormal basis for the projected column space.
    U, S, _ = np.linalg.svd(
        V_projected,
        full_matrices=False,
    )

    # Keep numerically nonzero singular directions.
    tol = max(V.shape) * np.finfo(float).eps * S[0]

    rank = np.sum(S > tol)

    if rank == 0:
        return np.empty((V.shape[0], 0))

    # U contains an orthonormal basis for Col(V_projected).
    return U[:, :rank]


def _compute_approximation(
    A: Array,
    V: Array,
    indices: list[int],
) -> Array:
    """
    Construct

        A_hat = A[:, J] V[J, :]^{-T} V^T

    as described in the Osinsky formulation.

    For numerical stability we use np.linalg.pinv rather than
    explicitly computing an inverse.
    """
    J = np.asarray(indices, dtype=int)

    VJ = V[J, :]

    # A(:,J) V(J,:)^{-T} V^T
    interpolation = np.linalg.pinv(VJ.T)

    return A[:, J] @ interpolation @ V.T


def osinsky_cssp(
    A: Array,
    k: int,
    verbose: bool = False,
) -> dict:
    """
    Deterministic Osinsky CSSP.

    Parameters
    ----------
    A:
        Input matrix of shape (m, n).

    k:
        Number of columns to select.

    verbose:
        Print intermediate selection steps.

    Returns
    -------
    result:
        Dictionary containing:

        indices
            Selected column indices.

        C
            Selected matrix A[:, indices].

        V
            Top-k right singular subspace.

        approximation
            Rank-k approximation generated from selected columns.

        relative_error
            ||A - A_hat||_F / ||A||_F
    """
    A = np.asarray(A, dtype=float)

    _validate_input(A, k)

    m, n = A.shape

    if verbose:
        print("=" * 60)
        print("OSINSKY DETERMINISTIC CSSP")
        print("=" * 60)
        print(f"Matrix shape : {A.shape}")
        print(f"Target rank  : {k}")
        print()

    # ---------------------------------------------------------
    # Step 1:
    # Compute an orthonormal basis V for the dominant
    # right singular subspace of A.
    # ---------------------------------------------------------
    V = _orthonormal_row_space(A, k)

    if verbose:
        print("[1] Dominant right singular subspace")
        print(f"    V shape = {V.shape}")
        print()

    # ---------------------------------------------------------
    # Working copies.
    #
    # E is the remaining / residual matrix.
    # V_work is the remaining right-space information.
    # ---------------------------------------------------------
    E = A.copy()
    V_work = V.copy()

    selected: list[int] = []

    # ---------------------------------------------------------
    # Step 2:
    # Deterministically select one column at a time.
    #
    # At each iteration choose j minimizing
    #
    #       ||E[:,j]||² / ||V_work[j,:]||²
    #
    # This is the deterministic counterpart to ARP's
    # leverage-score sampling rule.
    # ---------------------------------------------------------
    for step in range(k):
        if V_work.shape[1] == 0:
            break

        residual_norm_sq = np.sum(E * E, axis=0)
        row_norm_sq = np.sum(V_work * V_work, axis=1)

        # Do not select an already selected column.
        for j in selected:
            row_norm_sq[j] = 0.0

        valid = row_norm_sq > np.finfo(float).eps

        if not np.any(valid):
            raise RuntimeError(
                "No valid candidate column remains. "
                "The basis may be numerically rank deficient."
            )

        scores = np.full(n, np.inf)

        scores[valid] = (
            residual_norm_sq[valid]
            / row_norm_sq[valid]
        )

        j = int(np.argmin(scores))

        selected.append(j)

        if verbose:
            print(f"[Step {step + 1}/{k}]")
            print(f"    Selected column : {j}")
            print(f"    Residual norm²  : {residual_norm_sq[j]:.6e}")
            print(f"    Basis norm²     : {row_norm_sq[j]:.6e}")
            print(f"    Selection score : {scores[j]:.6e}")
            print()

        # -----------------------------------------------------
        # Update residual matrix.
        #
        # This removes the component represented by the
        # selected column in the current projection.
        # -----------------------------------------------------
        row = V_work[j, :]
        denom = np.dot(row, row)

        if denom <= np.finfo(float).eps:
            raise RuntimeError(
                "Selected row is numerically zero."
            )

        E -= np.outer(
            E[:, j],
            row,
        ) / denom

        # Force numerical zero in selected column.
        E[:, j] = 0.0

        # -----------------------------------------------------
        # Update the remaining right-space basis.
        # -----------------------------------------------------
        if step < k - 1:
            V_work = _project_out_selected_row(
                V_work,
                j,
            )

    # ---------------------------------------------------------
    # Construct final CSSP approximation.
    # ---------------------------------------------------------
    C = A[:, selected]

    VJ = V[selected, :]

    if np.linalg.matrix_rank(VJ) < k:
        raise RuntimeError(
            "Selected columns do not form a numerically "
            "full-rank CSSP basis."
        )

    A_hat = _compute_approximation(
        A,
        V,
        selected,
    )

    fro_error = np.linalg.norm(
        A - A_hat,
        ord="fro",
    )

    relative_error = fro_error / np.linalg.norm(
        A,
        ord="fro",
    )

    if verbose:
        print("=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        print(f"Selected indices       : {selected}")
        print(f"Selected matrix shape  : {C.shape}")
        print(f"Approximation shape    : {A_hat.shape}")
        print(f"Frobenius error        : {fro_error:.6e}")
        print(f"Relative Frobenius err : {relative_error:.6e}")
        print("=" * 60)

    return {
        "indices": np.asarray(selected, dtype=int),
        "C": C,
        "V": V,
        "approximation": A_hat,
        "error": fro_error,
        "relative_error": relative_error,
    }


if __name__ == "__main__":
    # Small sanity test
    rng = np.random.default_rng(42)

    A = rng.normal(size=(8, 10))
    k = 3

    result = osinsky_cssp(
        A,
        k,
        verbose=True,
    )
