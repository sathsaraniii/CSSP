"""Adaptive Randomized Pivoting (ARP) for column subset selection.

The selector follows Algorithm 2.2 of Cortinovis and Kressner. Its main
input is an already-computed ``n x k`` orthonormal row-space basis ``V``.
Keeping basis construction outside the selector lets the same algorithm be
used with either an exact truncated SVD or a randomized SVD.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cssp._validation import as_real_matrix, validate_rank

Array = NDArray[np.float64]


def _as_orthonormal_basis(V: Array) -> Array:
    """Validate and copy an ``n x k`` matrix with orthonormal columns."""
    V = as_real_matrix(V, name="V")
    n, k = V.shape
    if not 1 <= k <= n:
        raise ValueError(
            f"V must have shape (n, k) with 1 <= k <= n; got {V.shape}."
        )
    gram_error = np.linalg.norm(V.T @ V - np.eye(k), ord="fro")
    tolerance = 1e-8 * max(1, k)
    if gram_error > tolerance:
        raise ValueError(
            "The columns of V must be orthonormal. "
            f"||V.T @ V - I||_F = {gram_error:.3e}."
        )
    return V.copy()


def _householder_vector(x: Array) -> Array | None:
    """Return a stable unit vector defining a Householder reflector."""
    norm_x = float(np.linalg.norm(x))
    if norm_x == 0.0:
        return None

    v = np.asarray(x, dtype=float).copy()
    alpha = -np.copysign(norm_x, v[0])
    v[0] -= alpha
    norm_v = float(np.linalg.norm(v))
    if norm_v <= np.finfo(float).eps * norm_x:
        return None
    return v / norm_v


def arp_select(
    V: Array,
    random_state: int | np.random.Generator | None = None,
    verbose: bool = False,
    return_probabilities: bool = False,
) -> Array | tuple[Array, list[Array]]:
    """Select exactly ``k`` row indices from an ``n x k`` basis using ARP.

    At step ``s``, index ``j`` is sampled with probability proportional to
    ``||V_work[j, s:k]||_2^2``. A Householder update then eliminates the
    sampled direction from all later probability distributions.

    Parameters
    ----------
    V:
        Orthonormal row-space basis of shape ``(n, k)``.
    random_state:
        Seed or NumPy random generator used for sampling.
    verbose:
        Print the selected index and probability at each step.
    return_probabilities:
        If true, also return a copy of every sampling distribution.
    """
    V_work = _as_orthonormal_basis(V)
    n, k = V_work.shape

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    selected: list[int] = []
    probability_history: list[Array] = []

    for step in range(k):
        trailing = V_work[:, step:]
        weights = np.sum(trailing * trailing, axis=1)
        weights = np.maximum(weights, 0.0)
        if selected:
            weights[np.asarray(selected, dtype=int)] = 0.0

        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= np.finfo(float).eps:
            raise RuntimeError("No valid ARP sampling probability remains.")

        probabilities = weights / total
        probability_history.append(probabilities.copy())
        pivot = int(rng.choice(n, p=probabilities))
        selected.append(pivot)

        if verbose:
            print(
                f"ARP step {step + 1}/{k}: column={pivot}, "
                f"probability={probabilities[pivot]:.6e}"
            )

        if step < k - 1:
            reflector = _householder_vector(V_work[pivot, step:])
            if reflector is not None:
                product = V_work[:, step:] @ reflector
                V_work[:, step:] -= 2.0 * np.outer(product, reflector)

            # Zero in exact arithmetic; enforce it to prevent reselection from
            # accumulated floating-point roundoff.
            V_work[pivot, step + 1 :] = 0.0

    indices = np.asarray(selected, dtype=int)
    if np.unique(indices).size != k:
        raise RuntimeError("ARP produced duplicate indices.")
    if np.linalg.matrix_rank(np.asarray(V)[indices, :]) < k:
        raise RuntimeError("ARP selected a numerically singular V[indices, :].")

    if return_probabilities:
        return indices, probability_history
    return indices


def arp_cssp(
    A: Array,
    k: int,
    random_state: int | np.random.Generator | None = None,
    verbose: bool = False,
    V: Array | None = None,
) -> dict:
    """Backward-compatible convenience wrapper around :func:`arp_select`.

    If ``V`` is omitted, an exact top-``k`` right singular basis is computed.
    The returned approximation is the paper's oblique interpolatory one.
    New experiments should use ``arp_select`` or ``run_cssp_pipeline``.
    """
    A = as_real_matrix(A)
    k = validate_rank(k, A.shape)

    if V is None:
        _, _, Vt = np.linalg.svd(A, full_matrices=False)
        V_basis = Vt[:k, :].T
    else:
        V_basis = _as_orthonormal_basis(V)
        if V_basis.shape != (A.shape[1], k):
            raise ValueError(
                f"V must have shape {(A.shape[1], k)}; got {V_basis.shape}."
            )

    indices, probabilities = arp_select(
        V_basis,
        random_state=random_state,
        verbose=verbose,
        return_probabilities=True,
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
        "probabilities": probabilities,
        "approximation": approximation,
        "error": error,
        "relative_error": error / norm_A if norm_A else 0.0,
    }


__all__ = ["arp_select", "arp_cssp"]
