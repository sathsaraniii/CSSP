"""Reconstruction routines for column subset selection."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cssp._validation import as_real_matrix

Array = NDArray[np.float64]


def _validate_indices(A: Array, indices: Array | list[int]) -> Array:
    # Preserve each original value: ordinary NumPy coercion turns [0, True]
    # into integers before we can reject the Boolean index.
    raw_indices = np.asarray(indices, dtype=object)
    if raw_indices.ndim != 1 or raw_indices.size == 0:
        raise ValueError("indices must be a non-empty one-dimensional sequence.")
    if any(
        isinstance(index, (bool, np.bool_))
        or not isinstance(index, (int, np.integer))
        for index in raw_indices
    ):
        raise ValueError("indices must contain only integers (not Booleans).")
    # Check before casting to np.intp, so oversized Python/unsigned integers
    # cannot wrap around to a different, valid index.
    if any(index < 0 or index >= A.shape[1] for index in raw_indices):
        raise IndexError("A selected column index is outside the matrix.")
    J = np.asarray(raw_indices, dtype=np.intp)
    if np.unique(J).size != J.size:
        raise ValueError("Selected column indices must be unique.")
    return J


def selected_columns(A: Array, indices: Array | list[int]) -> Array:
    """Return the selected-column matrix ``C = A[:, indices]``."""
    A = as_real_matrix(A)
    J = _validate_indices(A, indices)
    return A[:, J]


def orthogonal_cssp_approximation(
    A: Array, indices: Array | list[int]
) -> Array:
    """Return the least-squares/orthogonal CSSP approximation ``C C^+ A``.

    This is the reconstruction used for the CSSP experiments in the ARP
    repository. ``lstsq`` also handles a numerically rank-deficient selected
    column matrix more safely than an unpivoted QR projection.
    """
    A = as_real_matrix(A)
    J = _validate_indices(A, indices)
    C = A[:, J]
    coefficients, _, _, _ = np.linalg.lstsq(C, A, rcond=None)
    return C @ coefficients


def oblique_cssp_approximation(
    A: Array,
    V: Array,
    indices: Array | list[int],
) -> Array:
    """Return ``A[:, J] @ V[J, :]^{-T} @ V.T``.

    This interpolatory approximation is the quantity controlled directly by
    the ARP expectation theorem. It requires exactly as many selected indices
    as columns in ``V`` and a nonsingular ``V[J, :]``.
    """
    A = as_real_matrix(A)
    V = as_real_matrix(V, name="V")
    J = _validate_indices(A, indices)
    if V.shape[0] != A.shape[1]:
        raise ValueError(
            f"V must have shape (A.shape[1], k); got {V.shape}."
        )
    if V.shape[1] != J.size:
        raise ValueError(
            "The number of selected columns must equal the number of columns in V."
        )

    try:
        coefficients = np.linalg.solve(V[J, :].T, V.T)
    except np.linalg.LinAlgError as exc:
        raise np.linalg.LinAlgError(
            "V[indices, :] is singular; the oblique reconstruction is undefined."
        ) from exc
    return A[:, J] @ coefficients


def reconstruct_cssp(
    A: Array,
    indices: Array | list[int],
    method: str = "orthogonal",
    V: Array | None = None,
) -> Array:
    """Reconstruct ``A`` using selected columns and the requested method."""
    normalized = method.lower().replace("-", "_")
    if normalized in {"orthogonal", "projection", "least_squares"}:
        return orthogonal_cssp_approximation(A, indices)
    if normalized in {"oblique", "interpolatory"}:
        if V is None:
            raise ValueError("V is required for oblique reconstruction.")
        return oblique_cssp_approximation(A, V, indices)
    raise ValueError("method must be 'orthogonal' or 'oblique'.")


__all__ = [
    "selected_columns",
    "orthogonal_cssp_approximation",
    "oblique_cssp_approximation",
    "reconstruct_cssp",
]
