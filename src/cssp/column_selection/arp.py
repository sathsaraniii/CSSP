"""
Adaptive Randomized Pivoting (ARP) for Column Subset Selection.

Based on:

A. Cortinovis and D. Kressner,
"Adaptive Randomized Pivoting for Column Subset Selection,
DEIM, and Low-Rank Approximation",
SIAM Journal on Matrix Analysis and Applications,
Vol. 47, Issue 1, 2026.

ARP can be viewed as a randomization of Osinsky's deterministic
CSSP algorithm.

Core idea:

    1. Compute an orthonormal basis V for the dominant right
       singular subspace of A.

    2. At each step, compute adaptive leverage probabilities

          p_j = ||V_work[j, :]||² / remaining_rank

    3. Randomly select one column according to p_j.

    4. Remove the selected direction from V_work.

    5. Repeat until k columns are selected.

The same residual/basis update structure is used as the
deterministic Osinsky procedure.

Input:
    A : m x n matrix
    k : number of columns to select

Output:
    indices : selected column indices
    C       : selected columns
    A_hat    : CSSP approximation
    error    : reconstruction error

A random_state argument is provided so that experiments can be
reproduced.
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
    Compute V_k, an n x k orthonormal matrix containing
    the dominant right singular subspace of A.
    """
    _, _, Vt = np.linalg.svd(
        A,
        full_matrices=False,
    )

    return Vt[:k, :].T


def _project_out_selected_row(
    V: Array,
    j: int,
) -> Array:
    """
    Remove the selected row direction from V.

    If v = V[j,:], then

        V_new = V - (V v^T / ||v||²) v

    projects the remaining row-space information onto the
    orthogonal complement of the selected row.

    An SVD is then used to obtain an orthonormal basis for
    the remaining subspace.
    """
    row = V[j, :]

    row_norm_sq = np.dot(
        row,
        row,
    )

    if row_norm_sq <= np.finfo(float).eps:
        raise RuntimeError(
            "Selected row has numerically zero norm."
        )

    V_projected = (
        V
        - np.outer(V @ row, row)
        / row_norm_sq
    )

    U, S, _ = np.linalg.svd(
        V_projected,
        full_matrices=False,
    )

    if len(S) == 0:
        return np.empty(
            (V.shape[0], 0)
        )

    tol = (
        max(V.shape)
        * np.finfo(float).eps
        * S[0]
    )

    rank = np.sum(S > tol)

    if rank == 0:
        return np.empty(
            (V.shape[0], 0)
        )

    return U[:, :rank]


def _compute_approximation(
    A: Array,
    V: Array,
    indices: list[int],
) -> Array:
    """
    Construct the CSSP approximation

        A_hat = A[:,J] V[J,:]^{-T} V^T.

    The pseudoinverse is used for numerical stability.
    """
    J = np.asarray(
        indices,
        dtype=int,
    )

    VJ = V[J, :]

    interpolation = np.linalg.pinv(
        VJ.T
    )

    return (
        A[:, J]
        @ interpolation
        @ V.T
    )


def arp_cssp(
    A: Array,
    k: int,
    random_state: int | None = None,
    verbose: bool = False,
) -> dict:
    """
    Adaptive Randomized Pivoting for CSSP.

    Parameters
    ----------
    A:
        Input matrix of shape (m, n).

    k:
        Number of columns to select.

    random_state:
        Seed for reproducibility.

    verbose:
        Print intermediate steps and probabilities.

    Returns
    -------
    result:
        Dictionary containing:

        indices
            Selected column indices.

        C
            Selected matrix A[:, indices].

        V
            Dominant right singular subspace.

        probabilities
            Sampling probabilities used at each step.

        approximation
            CSSP approximation.

        error
            Frobenius reconstruction error.

        relative_error
            Relative Frobenius reconstruction error.
    """
    A = np.asarray(
        A,
        dtype=float,
    )

    _validate_input(
        A,
        k,
    )

    m, n = A.shape

    rng = np.random.default_rng(
        random_state
    )

    if verbose:
        print("=" * 60)
        print("ADAPTIVE RANDOMIZED PIVOTING (ARP)")
        print("=" * 60)
        print(f"Matrix shape : {A.shape}")
        print(f"Target rank  : {k}")
        print(f"Random seed  : {random_state}")
        print()

    # ---------------------------------------------------------
    # Step 1:
    #
    # Compute dominant right singular subspace.
    # ---------------------------------------------------------
    V = _orthonormal_row_space(
        A,
        k,
    )

    if verbose:
        print("[1] Dominant right singular subspace")
        print(f"    V shape = {V.shape}")
        print()

    # Working copy of V.
    V_work = V.copy()

    selected: list[int] = []

    probability_history: list[Array] = []

    # ---------------------------------------------------------
    # Step 2:
    # Adaptive randomized selection.
    # ---------------------------------------------------------
    for step in range(k):

        remaining_rank = V_work.shape[1]

        if remaining_rank == 0:
            break

        # -----------------------------------------------------
        # Compute leverage-score probabilities.
        #
        # p_j =
        #       ||V_work[j,:]||²
        #       ----------------
        #       remaining_rank
        #
        # Because the columns of V_work are orthonormal,
        # sum_j ||V_work[j,:]||² = remaining_rank.
        # Therefore the probabilities sum to 1.
        # -----------------------------------------------------
        row_norm_sq = np.sum(
            V_work * V_work,
            axis=1,
        )

        probabilities = (
            row_norm_sq
            / remaining_rank
        )

        # Already selected columns must have probability zero.
        probabilities[selected] = 0.0

        probability_sum = probabilities.sum()

        if probability_sum <= np.finfo(float).eps:
            raise RuntimeError(
                "No valid sampling probability remains."
            )

        # Normalize to eliminate tiny floating-point errors.
        probabilities /= probability_sum

        probability_history.append(
            probabilities.copy()
        )

        # -----------------------------------------------------
        # Random selection.
        # -----------------------------------------------------
        j = int(
            rng.choice(
                n,
                p=probabilities,
            )
        )

        selected.append(j)

        if verbose:
            print(f"[Step {step + 1}/{k}]")
            print(
                f"    Remaining rank : "
                f"{remaining_rank}"
            )
            print(
                f"    Selected column: "
                f"{j}"
            )
            print(
                f"    Probability    : "
                f"{probabilities[j]:.6e}"
            )
            print(
                f"    Max probability: "
                f"{probabilities.max():.6e}"
            )

            # For a small matrix show all probabilities.
            if n <= 20:
                print(
                    "    Probabilities:"
                )
                for idx, p in enumerate(
                    probabilities
                ):
                    print(
                        f"        column {idx}: "
                        f"{p:.6f}"
                    )

            print()

        # -----------------------------------------------------
        # Update V_work.
        #
        # The selected row is projected out. This is the key
        # adaptive step: probabilities in the next iteration
        # are therefore different from the current ones.
        # -----------------------------------------------------
        if step < k - 1:
            V_work = _project_out_selected_row(
                V_work,
                j,
            )

    # ---------------------------------------------------------
    # Final selected matrix.
    # ---------------------------------------------------------
    C = A[:, selected]

    VJ = V[selected, :]

    if np.linalg.matrix_rank(VJ) < k:
        raise RuntimeError(
            "ARP selected a numerically rank-deficient "
            "subset. Try another random seed."
        )

    # ---------------------------------------------------------
    # Construct CSSP approximation.
    # ---------------------------------------------------------
    A_hat = _compute_approximation(
        A,
        V,
        selected,
    )

    fro_error = np.linalg.norm(
        A - A_hat,
        ord="fro",
    )

    relative_error = (
        fro_error
        / np.linalg.norm(
            A,
            ord="fro",
        )
    )

    if verbose:
        print("=" * 60)
        print("FINAL RESULT")
        print("=" * 60)
        print(
            f"Selected indices       : "
            f"{selected}"
        )
        print(
            f"Selected matrix shape  : "
            f"{C.shape}"
        )
        print(
            f"Approximation shape    : "
            f"{A_hat.shape}"
        )
        print(
            f"Frobenius error        : "
            f"{fro_error:.6e}"
        )
        print(
            f"Relative Frobenius err : "
            f"{relative_error:.6e}"
        )
        print("=" * 60)

    return {
        "indices": np.asarray(
            selected,
            dtype=int,
        ),
        "C": C,
        "V": V,
        "probabilities": probability_history,
        "approximation": A_hat,
        "error": fro_error,
        "relative_error": relative_error,
    }


if __name__ == "__main__":
    # Small reproducible sanity test.
    rng = np.random.default_rng(42)

    A = rng.normal(
        size=(8, 10)
    )

    k = 3

    result = arp_cssp(
        A,
        k,
        random_state=42,
        verbose=True,
    )
