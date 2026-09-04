"""Evaluation metrics shared by all CSSP pipelines."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


Array = NDArray[np.float64]


def _matching_matrices(A: Array, approximation: Array) -> tuple[Array, Array]:
    A = np.asarray(A, dtype=float)
    approximation = np.asarray(approximation, dtype=float)
    if A.ndim != 2 or approximation.ndim != 2:
        raise ValueError("A and approximation must both be 2-D matrices.")
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
    """Return the optimal rank-``k`` Frobenius error from singular values."""
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2-D matrix.")
    if not 1 <= k <= min(A.shape):
        raise ValueError(f"k must satisfy 1 <= k <= min(A.shape); got {k}.")
    singular_values = np.linalg.svd(A, compute_uv=False)
    return float(np.linalg.norm(singular_values[k:]))


def evaluate_approximation(
    A: Array,
    approximation: Array,
    optimal_error: float | None = None,
) -> dict[str, float]:
    """Compute reconstruction errors used to compare the four pipelines."""
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
        if optimal_error < 0.0:
            raise ValueError("optimal_error must be non-negative.")
        metrics["optimal_rank_k_error"] = optimal_error
        metrics["frobenius_optimality_ratio"] = (
            fro_error / optimal_error
            if optimal_error
            else (1.0 if fro_error <= 1e-14 * max(1.0, norm_fro) else float("inf"))
        )

    return metrics


__all__ = [
    "frobenius_error",
    "relative_frobenius_error",
    "optimal_rank_k_error",
    "evaluate_approximation",
]
