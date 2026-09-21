"""Shared validation for the real-valued CSSP implementation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def as_real_matrix(A: object, *, name: str = "A") -> NDArray[np.float64]:
    """Return a finite, nonempty real matrix without discarding imaginary parts."""
    try:
        matrix = np.asarray(A)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a non-empty 2-D real matrix.") from exc
    if matrix.ndim != 2 or 0 in matrix.shape:
        raise ValueError(f"{name} must be a non-empty 2-D real matrix.")
    if np.iscomplexobj(matrix):
        raise ValueError(f"{name} must be real-valued; complex matrices are unsupported.")
    try:
        matrix = np.asarray(matrix, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain real numeric values.") from exc
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def validate_integer(
    value: object,
    *,
    name: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Validate an integer parameter, rejecting Boolean and fractional values."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer (not a Boolean).")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        bounds = (
            f"{minimum} <= {name} <= {maximum}"
            if maximum is not None
            else f"{name} >= {minimum}"
        )
        raise ValueError(f"{name} must satisfy {bounds}; got {value}.")
    return value


def validate_rank(k: object, shape: tuple[int, int]) -> int:
    """Validate a target rank for a nonempty matrix."""
    return validate_integer(k, name="k", minimum=1, maximum=min(shape))
