"""Run and display the four CSSP pipelines on a reproducible matrix.

Usage from the repository root:

    python experiments/run_experiment.py
    python experiments/run_experiment.py --m 500 --n 300 --k 20 --q 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


# This repository uses a src/ layout but is not yet packaged for installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from cssp.cssp_pipeline import run_four_pipelines  # noqa: E402


def make_test_matrix(
    m: int,
    n: int,
    numerical_rank: int,
    noise: float,
    random_state: int,
) -> np.ndarray:
    """Create a low-rank matrix with a decaying spectrum and additive noise."""
    rng = np.random.default_rng(random_state)
    rank = min(numerical_rank, m, n)
    left, _ = np.linalg.qr(rng.standard_normal((m, rank)), mode="reduced")
    right, _ = np.linalg.qr(rng.standard_normal((n, rank)), mode="reduced")
    singular_values = np.geomspace(20.0, 1.0, rank)
    signal = (left * singular_values) @ right.T
    return signal + noise * rng.standard_normal((m, n))


def print_results(results: dict) -> None:
    """Print a compact comparison table suitable for a progress demo."""
    header = (
        f"{'Pipeline':<27} {'Rel. Fro error':>15} "
        f"{'Optimality':>12} {'Basis (s)':>11} {'Select (s)':>12}"
    )
    print(header)
    print("-" * len(header))
    for name, result in results.items():
        metrics = result["metrics"]
        timings = result["timings"]
        print(
            f"{name:<27} "
            f"{metrics['relative_frobenius_error']:>15.6e} "
            f"{metrics['frobenius_optimality_ratio']:>12.6f} "
            f"{timings['basis_seconds']:>11.6f} "
            f"{timings['selection_seconds']:>12.6f}"
        )

    print("\nSelected zero-based column indices:")
    for name, result in results.items():
        print(f"  {name:<25} {result['indices'].tolist()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=160, help="number of rows")
    parser.add_argument("--n", type=int, default=120, help="number of columns")
    parser.add_argument("--k", type=int, default=10, help="selected column count")
    parser.add_argument(
        "--numerical-rank", type=int, default=25, help="signal rank"
    )
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--p", type=int, default=10, help="RSVD oversampling")
    parser.add_argument("--q", type=int, default=1, help="RSVD power iterations")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    A = make_test_matrix(
        args.m, args.n, args.numerical_rank, args.noise, args.seed
    )
    results = run_four_pipelines(
        A,
        args.k,
        random_state=args.seed,
        oversampling=args.p,
        power_iterations=args.q,
    )
    print(f"Matrix shape: {A.shape}; target rank/columns: {args.k}\n")
    print_results(results)


if __name__ == "__main__":
    main()
