"""Evaluation metrics shared by all CSSP pipelines."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cssp._validation import as_real_matrix, validate_rank

Array = NDArray[np.float64]


def _matching_matrices(A: Array, approximation: Array) -> tuple[Array, Array]:
    A = as_real_matrix(A)
    approximation = as_real_matrix(approximation, name="approximation")
    if A.shape != approximation.shape:
        raise ValueError(
            f"A and approximation must have the same shape; got "
            f"{A.shape} and {approximation.shape}."
        )
    return A, approximation


def frobenius_error(A: Array, approximation: Array) -> float:
    """Return ``||A - approximation||_F``."""
    A, approximation = _matching_matrices(A, approximation)
    return float(np.linalg.norm(A - approximation, ord="fro"))


def relative_frobenius_error(A: Array, approximation: Array) -> float:
    """Return the Frobenius error relative to ``||A||_F``."""
    A, approximation = _matching_matrices(A, approximation)
    norm_A = float(np.linalg.norm(A, ord="fro"))
    error = float(np.linalg.norm(A - approximation, ord="fro"))
    if norm_A == 0.0:
        return 0.0 if error == 0.0 else float("inf")
    return error / norm_A


def optimal_rank_k_error(A: Array, k: int) -> float:
    """Return the norm of the discarded singular values of ``A``.

    This matches ``sqrt(sum(sv(k+1:end).^2))`` in the authors' MATLAB
    CSSP experiments. Use the full, exact singular spectrum, not a
    truncated/randomized spectrum or a difference of squared norms:
    subtracting nearly equal energies can erase a small nonzero tail.
    """
    A = as_real_matrix(A)
    k = validate_rank(k, A.shape)
    singular_values = np.linalg.svd(A, compute_uv=False)
    return float(np.linalg.norm(singular_values[k:]))


def evaluate_approximation(
    A: Array,
    approximation: Array,
    optimal_error: float | None = None,
) -> dict[str, float]:
    """Compute reconstruction errors and an optional optimality ratio.

    The reference MATLAB CSSP experiments report Frobenius errors relative
    to ``||A||_F`` and plot the optimal SVD error separately. The additional
    ``frobenius_optimality_ratio`` here divides by a supplied finite,
    nonnegative ``optimal_error`` only when it is strictly positive.

    For a zero optimal error, this repository reports ``nan`` (undefined),
    regardless of the reconstruction error. This is not a convention from
    the paper: use the absolute/relative errors in that case, rather than
    assigning a ratio using a rounding tolerance. No optional optimality
    metrics are returned when ``optimal_error`` is ``None``.
    """
    A, approximation = _matching_matrices(A, approximation)
    residual = A - approximation

    fro_error = float(np.linalg.norm(residual, ord="fro"))
    spectral_error = float(np.linalg.norm(residual, ord=2))
    norm_fro = float(np.linalg.norm(A, ord="fro"))
    norm_spectral = float(np.linalg.norm(A, ord=2))

    metrics = {
        "frobenius_error": fro_error,
        "relative_frobenius_error": (
            fro_error / norm_fro if norm_fro else (0.0 if fro_error == 0.0 else float("inf"))
        ),
        "spectral_error": spectral_error,
        "relative_spectral_error": (
            spectral_error / norm_spectral
            if norm_spectral
            else (0.0 if spectral_error == 0.0 else float("inf"))
        ),
    }

    if optimal_error is not None:
        optimal_error = float(optimal_error)
        if not np.isfinite(optimal_error) or optimal_error < 0.0:
            raise ValueError("optimal_error must be finite and non-negative.")
        metrics["optimal_rank_k_error"] = optimal_error
        metrics["frobenius_optimality_ratio"] = (
            fro_error / optimal_error
            if optimal_error > 0.0
            else float("nan")
        )

    return metrics


__all__ = [
    "frobenius_error",
    "relative_frobenius_error",
    "optimal_rank_k_error",
    "evaluate_approximation",
]
