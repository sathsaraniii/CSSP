"""Fast adaptive randomized pivoting by rejection sampling.

This module implements the rejection-sampling RPQR method described in
Algorithm 3 of Epperly, "Adaptive randomized pivoting and volume sampling".
For an orthonormal ``n x k`` basis ``V``, it samples the same volume-sampling
distribution as the sequential ARP selector while avoiding an update of all
``n`` rows at every pivot.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cssp._validation import validate_integer
from cssp.column_selection.arp import _as_orthonormal_basis


Array = NDArray[np.float64]


def _project_out(Q: Array, X: Array) -> Array:
    """Project columns of ``X`` away from ``range(Q)`` with reorthogonalization."""
    if Q.shape[1] == 0:
        return X.copy()

    residual = X - Q @ (Q.T @ X)
    # A second pass limits loss of orthogonality near the final pivots.
    residual -= Q @ (Q.T @ residual)
    return residual


def rejection_arp_select(
    V: Array,
    random_state: int | np.random.Generator | None = None,
    *,
    block_size: int | None = None,
    verbose: bool = False,
    return_diagnostics: bool = False,
) -> Array | tuple[Array, dict[str, object]]:
    """Select ``k`` indices from an ``n x k`` basis using RejectionRPQR.

    Candidate indices are proposed from the fixed leverage-score distribution
    ``||V[j, :]||_2^2 / k``. A candidate is accepted with probability equal to
    its residual squared norm divided by its original leverage score. Proposals
    are processed in blocks so their residual Gram matrix is formed with
    matrix-matrix operations.

    The returned subset has the same volume-sampling distribution as
    :func:`cssp.column_selection.arp.arp_select`; only the implementation is
    different.

    Parameters
    ----------
    V:
        Orthonormal row-space basis with shape ``(n, k)``.
    random_state:
        Seed or NumPy random generator used for proposals and acceptance tests.
    block_size:
        Number of proposals per block. The paper's default of ``k`` is used
        when this is omitted.
    verbose:
        Print the number of accepted candidates after every proposal block.
    return_diagnostics:
        If true, also return leverage scores, proposal counts, block counts,
        and the observed acceptance rate.
    """
    if block_size is not None:
        block_size = validate_integer(block_size, name="block_size", minimum=1)
    V_basis = _as_orthonormal_basis(V)
    n, k = V_basis.shape
    if block_size is None:
        block_size = k

    rng = (
        random_state
        if isinstance(random_state, np.random.Generator)
        else np.random.default_rng(random_state)
    )

    # RejectionRPQR is stated for B = V.T, whose columns are the candidates.
    B = V_basis.T
    leverage_scores = np.sum(V_basis * V_basis, axis=1)
    leverage_scores = np.maximum(leverage_scores, 0.0)
    leverage_sum = float(np.sum(leverage_scores))
    if not np.isfinite(leverage_sum) or leverage_sum <= np.finfo(float).eps:
        raise RuntimeError("No valid leverage-score proposal distribution remains.")
    proposal_probabilities = leverage_scores / leverage_sum

    selected: list[int] = []
    selected_set: set[int] = set()
    Q_selected = np.empty((k, 0), dtype=float)
    proposals_drawn = 0
    accepted_per_block: list[int] = []

    # Termination occurs almost surely and requires O(k log k) proposals in
    # expectation. This generous guard catches numerical/programming failures
    # without affecting ordinary runs.
    maximum_blocks = max(10_000, 100 * k)

    while len(selected) < k:
        if len(accepted_per_block) >= maximum_blocks:
            raise RuntimeError(
                "Rejection ARP did not terminate; the basis may be numerically "
                "rank deficient."
            )

        proposals = rng.choice(
            n,
            size=block_size,
            replace=True,
            p=proposal_probabilities,
        )
        proposals_drawn += block_size

        residual_candidates = _project_out(Q_selected, B[:, proposals])
        gram = residual_candidates.T @ residual_candidates
        gram = 0.5 * (gram + gram.T)

        accepted_positions: list[int] = []
        accepted_indices_in_block: set[int] = set()

        for position, proposal in enumerate(proposals):
            proposal_index = int(proposal)
            if (
                proposal_index in selected_set
                or proposal_index in accepted_indices_in_block
            ):
                continue

            residual_norm_sq = max(0.0, float(gram[position, position]))
            leverage = float(leverage_scores[proposal_index])
            if leverage <= 0.0:
                continue

            acceptance_probability = min(1.0, residual_norm_sq / leverage)
            if rng.random() >= acceptance_probability:
                continue

            accepted_positions.append(position)
            accepted_indices_in_block.add(proposal_index)

            # Eliminate the accepted direction from the Gram matrix of the
            # remaining proposals. This is Algorithm 2 in the paper.
            if residual_norm_sq > np.finfo(float).eps * leverage:
                trailing_column = gram[position:, position].copy()
                trailing = gram[position:, position:]
                trailing -= (
                    np.outer(trailing_column, trailing_column) / residual_norm_sq
                )
                trailing[:] = 0.5 * (trailing + trailing.T)
                diagonal = np.diag(trailing).copy()
                np.fill_diagonal(trailing, np.maximum(diagonal, 0.0))

        remaining = k - len(selected)
        accepted_positions = accepted_positions[:remaining]
        new_indices = [int(proposals[pos]) for pos in accepted_positions]

        if new_indices:
            new_vectors = _project_out(Q_selected, B[:, new_indices])
            Q_new, R_new = np.linalg.qr(new_vectors, mode="reduced")

            scale = max(1.0, float(np.linalg.norm(new_vectors, ord=2)))
            tolerance = (
                100.0
                * np.finfo(float).eps
                * max(new_vectors.shape)
                * scale
            )
            if np.any(np.abs(np.diag(R_new)) <= tolerance):
                raise RuntimeError(
                    "Rejection ARP accepted a numerically dependent pivot block."
                )

            Q_selected = np.column_stack((Q_selected, Q_new))
            selected.extend(new_indices)
            selected_set.update(new_indices)

        accepted_per_block.append(len(new_indices))
        if verbose:
            print(
                f"Rejection ARP block {len(accepted_per_block)}: "
                f"accepted={len(new_indices)}, total={len(selected)}/{k}"
            )

    indices = np.asarray(selected, dtype=int)
    if np.unique(indices).size != k:
        raise RuntimeError("Rejection ARP produced duplicate indices.")
    if np.linalg.matrix_rank(V_basis[indices, :]) < k:
        raise RuntimeError(
            "Rejection ARP selected a numerically singular V[indices, :]."
        )

    if return_diagnostics:
        diagnostics: dict[str, object] = {
            "leverage_scores": leverage_scores.copy(),
            "proposals_drawn": proposals_drawn,
            "blocks": len(accepted_per_block),
            "accepted_per_block": accepted_per_block,
            "acceptance_rate": k / proposals_drawn,
            "block_size": block_size,
        }
        return indices, diagnostics
    return indices


__all__ = ["rejection_arp_select"]
