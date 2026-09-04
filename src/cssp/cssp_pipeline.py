"""End-to-end pipelines for the 2 x 2 CSSP experiment.

The four supported combinations are:

1. deterministic truncated SVD + deterministic Osinsky selection;
2. deterministic truncated SVD + randomized ARP selection;
3. randomized SVD + deterministic Osinsky selection;
4. randomized SVD + randomized ARP selection.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cssp.basis.deterministic_svd import deterministic_svd
from cssp.basis.randomized_svd import randomized_svd
from cssp.column_selection.arp import arp_select
from cssp.column_selection.osinsky import osinsky_select
from cssp.metrics.evaluation import evaluate_approximation, optimal_rank_k_error
from cssp.reconstruction.cssp_approximation import (
    oblique_cssp_approximation,
    orthogonal_cssp_approximation,
    selected_columns,
)


Array = NDArray[np.float64]

PIPELINE_CONFIGS: dict[str, tuple[str, str]] = {
    "deterministic_osinsky": ("deterministic", "osinsky"),
    "deterministic_arp": ("deterministic", "arp"),
    "randomized_osinsky": ("randomized", "osinsky"),
    "randomized_arp": ("randomized", "arp"),
}


def _validate_matrix_rank(A: Array, k: int) -> Array:
    A = np.asarray(A, dtype=float)
    if A.ndim != 2:
        raise ValueError("A must be a 2-D matrix.")
    if not np.all(np.isfinite(A)):
        raise ValueError("A must contain only finite values.")
    if not 1 <= k <= min(A.shape):
        raise ValueError(f"k must satisfy 1 <= k <= min(A.shape); got {k}.")
    return A


def _normalize_row_space_method(method: str) -> str:
    method = method.lower().replace("-", "_").replace(" ", "_")
    if method in {"deterministic", "exact", "truncated_svd", "svd"}:
        return "deterministic"
    if method in {"randomized", "randomized_svd", "rsvd", "sketch"}:
        return "randomized"
    raise ValueError("row_space_method must be 'deterministic' or 'randomized'.")


def _normalize_selector(method: str) -> str:
    method = method.lower().replace("-", "_").replace(" ", "_")
    if method in {"osinsky", "deterministic"}:
        return "osinsky"
    if method in {"arp", "randomized"}:
        return "arp"
    raise ValueError("column_selection_method must be 'osinsky' or 'arp'.")


def _compute_row_space(
    A: Array,
    k: int,
    method: str,
    oversampling: int,
    power_iterations: int,
    random_state: int | None,
) -> tuple[Array, Array, Array, float]:
    started = perf_counter()
    if method == "deterministic":
        U, singular_values, Vt = deterministic_svd(A, k)
    else:
        U, singular_values, Vt = randomized_svd(
            A,
            k,
            p=oversampling,
            q=power_iterations,
            random_state=random_state,
        )
    elapsed = perf_counter() - started
    return U, singular_values, Vt, elapsed


def _run_with_basis(
    A: Array,
    k: int,
    U: Array,
    singular_values: Array,
    Vt: Array,
    basis_seconds: float,
    row_space_method: str,
    column_selection_method: str,
    random_state: int | None,
    reconstruction: str,
    optimal_error: float | None,
    oversampling: int,
    power_iterations: int,
    verbose: bool,
) -> dict[str, Any]:
    V = Vt.T

    selection_started = perf_counter()
    if column_selection_method == "arp":
        indices, history = arp_select(
            V,
            random_state=random_state,
            verbose=verbose,
            return_probabilities=True,
        )
        selection_details = {"probability_history": history}
    else:
        indices, history = osinsky_select(
            A,
            V,
            verbose=verbose,
            return_scores=True,
        )
        selection_details = {"score_history": history}
    selection_seconds = perf_counter() - selection_started

    reconstruction_started = perf_counter()
    C = selected_columns(A, indices)
    orthogonal_approximation = orthogonal_cssp_approximation(A, indices)
    oblique_approximation = oblique_cssp_approximation(A, V, indices)
    reconstruction_seconds = perf_counter() - reconstruction_started

    reconstruction_name = reconstruction.lower().replace("-", "_")
    if reconstruction_name in {"orthogonal", "projection", "least_squares"}:
        approximation = orthogonal_approximation
        reconstruction_name = "orthogonal"
    elif reconstruction_name in {"oblique", "interpolatory"}:
        approximation = oblique_approximation
        reconstruction_name = "oblique"
    else:
        raise ValueError("reconstruction must be 'orthogonal' or 'oblique'.")

    evaluation_started = perf_counter()
    row_space_approximation = (A @ V) @ V.T
    orthogonal_metrics = evaluate_approximation(
        A, orthogonal_approximation, optimal_error=optimal_error
    )
    oblique_metrics = evaluate_approximation(
        A, oblique_approximation, optimal_error=optimal_error
    )
    row_space_metrics = evaluate_approximation(
        A, row_space_approximation, optimal_error=optimal_error
    )
    metrics = (
        orthogonal_metrics if reconstruction_name == "orthogonal" else oblique_metrics
    )
    condition_number = float(np.linalg.cond(V[indices, :]))
    selected_rank = int(np.linalg.matrix_rank(C))
    evaluation_seconds = perf_counter() - evaluation_started

    pipeline_name = f"{row_space_method}_{column_selection_method}"
    algorithm_seconds = basis_seconds + selection_seconds + reconstruction_seconds

    return {
        "name": pipeline_name,
        "row_space_method": row_space_method,
        "column_selection_method": column_selection_method,
        "reconstruction_method": reconstruction_name,
        "indices": indices,
        "C": C,
        "U": U,
        "singular_values": singular_values,
        "Vt": Vt,
        "V": V,
        "approximation": approximation,
        "orthogonal_approximation": orthogonal_approximation,
        "oblique_approximation": oblique_approximation,
        "row_space_approximation": row_space_approximation,
        "metrics": metrics,
        "orthogonal_metrics": orthogonal_metrics,
        "oblique_metrics": oblique_metrics,
        "row_space_metrics": row_space_metrics,
        "condition_number_VJ": condition_number,
        "selected_column_rank": selected_rank,
        "selection_details": selection_details,
        "timings": {
            "basis_seconds": basis_seconds,
            "selection_seconds": selection_seconds,
            "reconstruction_seconds": reconstruction_seconds,
            "algorithm_seconds": algorithm_seconds,
            "evaluation_seconds": evaluation_seconds,
            "total_seconds": algorithm_seconds + evaluation_seconds,
        },
        "parameters": {
            "k": k,
            "oversampling": oversampling if row_space_method == "randomized" else None,
            "power_iterations": (
                power_iterations if row_space_method == "randomized" else None
            ),
            "random_state": random_state,
        },
        # Convenient compatibility aliases.
        "error": metrics["frobenius_error"],
        "relative_error": metrics["relative_frobenius_error"],
    }


def run_cssp_pipeline(
    A: Array,
    k: int,
    row_space_method: str = "deterministic",
    column_selection_method: str = "osinsky",
    *,
    random_state: int | None = None,
    oversampling: int = 10,
    power_iterations: int = 0,
    reconstruction: str = "orthogonal",
    compute_optimal: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Run one chosen row-space/column-selection combination."""
    A = _validate_matrix_rank(A, k)
    row_space_method = _normalize_row_space_method(row_space_method)
    column_selection_method = _normalize_selector(column_selection_method)

    U, singular_values, Vt, basis_seconds = _compute_row_space(
        A,
        k,
        row_space_method,
        oversampling,
        power_iterations,
        random_state,
    )

    if compute_optimal:
        if row_space_method == "deterministic":
            tail_sq = float(np.linalg.norm(A, ord="fro") ** 2 - np.sum(singular_values**2))
            optimal_error = float(np.sqrt(max(0.0, tail_sq)))
        else:
            optimal_error = optimal_rank_k_error(A, k)
    else:
        optimal_error = None

    return _run_with_basis(
        A,
        k,
        U,
        singular_values,
        Vt,
        basis_seconds,
        row_space_method,
        column_selection_method,
        random_state,
        reconstruction,
        optimal_error,
        oversampling,
        power_iterations,
        verbose,
    )


def run_four_pipelines(
    A: Array,
    k: int,
    *,
    random_state: int | None = 0,
    oversampling: int = 10,
    power_iterations: int = 0,
    reconstruction: str = "orthogonal",
    verbose: bool = False,
) -> dict[str, dict[str, Any]]:
    """Run all four pipelines while sharing each computed row-space basis.

    Sharing the bases ensures Osinsky and ARP see exactly the same ``V`` in
    their corresponding comparison. It also avoids computing each SVD twice.
    """
    A = _validate_matrix_rank(A, k)

    U_d, S_d, Vt_d, deterministic_seconds = _compute_row_space(
        A, k, "deterministic", oversampling, power_iterations, random_state
    )
    tail_sq = float(np.linalg.norm(A, ord="fro") ** 2 - np.sum(S_d**2))
    optimal_error = float(np.sqrt(max(0.0, tail_sq)))

    U_r, S_r, Vt_r, randomized_seconds = _compute_row_space(
        A, k, "randomized", oversampling, power_iterations, random_state
    )

    basis_data = {
        "deterministic": (U_d, S_d, Vt_d, deterministic_seconds),
        "randomized": (U_r, S_r, Vt_r, randomized_seconds),
    }

    results: dict[str, dict[str, Any]] = {}
    for name, (row_method, selector) in PIPELINE_CONFIGS.items():
        U, singular_values, Vt, basis_seconds = basis_data[row_method]
        results[name] = _run_with_basis(
            A,
            k,
            U,
            singular_values,
            Vt,
            basis_seconds,
            row_method,
            selector,
            random_state,
            reconstruction,
            optimal_error,
            oversampling,
            power_iterations,
            verbose,
        )
    return results


def deterministic_svd_osinsky(A: Array, k: int, **kwargs: Any) -> dict[str, Any]:
    """Pipeline 1: deterministic truncated SVD + Osinsky."""
    return run_cssp_pipeline(A, k, "deterministic", "osinsky", **kwargs)


def deterministic_svd_arp(A: Array, k: int, **kwargs: Any) -> dict[str, Any]:
    """Pipeline 2: deterministic truncated SVD + ARP."""
    return run_cssp_pipeline(A, k, "deterministic", "arp", **kwargs)


def randomized_svd_osinsky(A: Array, k: int, **kwargs: Any) -> dict[str, Any]:
    """Pipeline 3: randomized SVD + Osinsky."""
    return run_cssp_pipeline(A, k, "randomized", "osinsky", **kwargs)


def randomized_svd_arp(A: Array, k: int, **kwargs: Any) -> dict[str, Any]:
    """Pipeline 4: randomized SVD + ARP."""
    return run_cssp_pipeline(A, k, "randomized", "arp", **kwargs)


__all__ = [
    "PIPELINE_CONFIGS",
    "run_cssp_pipeline",
    "run_four_pipelines",
    "deterministic_svd_osinsky",
    "deterministic_svd_arp",
    "randomized_svd_osinsky",
    "randomized_svd_arp",
]
