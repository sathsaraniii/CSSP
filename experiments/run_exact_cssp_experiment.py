# E1 — Exact CSSP validation by exhaustive column-subset enumeration.
"""Compare Osinsky, ARP, and Rejection ARP with the best actual column subset.

Defaults: one fixed Gaussian 8 x 10 matrix, k=3, all 120 subsets, and
100 seeds per randomized selector. All selectors share the exact SVD basis.
This is an accuracy experiment, not a runtime benchmark. "Exact" means
exhaustive search with floating-point least-squares calculations.

Run from the repository root:
    python experiments/run_exact_cssp_experiment.py
    python experiments/run_exact_cssp_experiment.py --trials 100 --data-seed 42

CSV output includes every enumerated subset and every selector run. Existing
output files are not overwritten; choose another --output path for another run.
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
import json
from math import comb
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cssp.column_selection.arp import arp_select  # noqa: E402
from cssp.column_selection.osinsky import osinsky_select  # noqa: E402
from cssp.column_selection.rejection_arp import rejection_arp_select  # noqa: E402
from cssp.reconstruction.cssp_approximation import (  # noqa: E402
    orthogonal_cssp_approximation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=8)
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--max-subsets", type=int, default=100_000,
                        help="safety limit for the exhaustive search")
    parser.add_argument("--output", type=Path,
                        default=PROJECT_ROOT / "results" / "e1_exact_cssp.csv")
    args = parser.parse_args()
    if not 1 <= args.k <= min(args.m, args.n):
        parser.error("require 1 <= k <= min(m, n)")
    if args.trials < 2:
        parser.error("--trials must be at least 2 to report sample standard deviation")
    if min(args.data_seed, args.seed_start) < 0:
        parser.error("seeds must be nonnegative")
    if args.max_subsets < 1 or comb(args.n, args.k) > args.max_subsets:
        parser.error("subset count exceeds --max-subsets; use a smaller problem")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}; choose another --output")
    return args


def ratio(error: float, baseline: float) -> float:
    """Keep zero-baseline ratios undefined, without thresholding small tails."""
    return error / baseline if baseline > 0 else float("nan")


def run_experiment(args: argparse.Namespace) -> list[dict]:
    """Run exhaustive search and selectors on one unchanged, seeded matrix."""
    A = np.random.default_rng(args.data_seed).standard_normal((args.m, args.n))
    _, singular_values, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt[:args.k].T.copy()
    norm_A = float(np.linalg.norm(A, ord="fro"))
    svd_error = float(np.linalg.norm(singular_values[args.k:]))
    tolerance = 100 * np.finfo(float).eps * max(A.shape) * norm_A

    # Independent least-squares oracle: do not use the production reconstruction
    # helper for exhaustive search, and do not discard rank-deficient subsets.
    exhaustive = []
    for subset in combinations(range(args.n), args.k):
        C = A[:, subset]
        coefficients = np.linalg.lstsq(C, A, rcond=None)[0]
        error = float(np.linalg.norm(A - C @ coefficients, ord="fro"))
        exhaustive.append((subset, error))
    best_indices, cssp_error = min(exhaustive, key=lambda item: item[1])
    if not np.isfinite(cssp_error) or svd_error > cssp_error + tolerance:
        raise RuntimeError("Invalid exhaustive baseline or SVD/CSSP error ordering")

    rows = []

    def record(method: str, indices, error: float, seed=None) -> None:
        if not np.isfinite(error) or error + tolerance < cssp_error:
            raise RuntimeError(f"{method}: error is nonfinite or below exhaustive optimum")
        rows.append({
            "method": method, "matrix_kind": "gaussian", "m": args.m,
            "n": args.n, "k": args.k, "data_seed": args.data_seed,
            "selector_seed": seed, "indices": json.dumps([int(j) for j in indices]),
            "frobenius_error": error, "relative_error": ratio(error, norm_A),
            "exact_cssp_error": cssp_error, "cssp_optimality_ratio": ratio(error, cssp_error),
            "svd_optimal_error": svd_error, "svd_optimality_ratio": ratio(error, svd_error),
            "optimal_within_tolerance": abs(error - cssp_error) <= tolerance,
            "ordering_tolerance": tolerance,
        })

    for subset, error in exhaustive:
        record("exhaustive_subset", subset, error)
    record("exact_cssp", best_indices, cssp_error)

    def evaluate(method: str, indices, seed=None) -> None:
        approximation = orthogonal_cssp_approximation(A, indices)
        error = float(np.linalg.norm(A - approximation, ord="fro"))
        record(method, indices, error, seed)

    evaluate("osinsky", osinsky_select(A, V))
    for seed in range(args.seed_start, args.seed_start + args.trials):
        evaluate("arp", arp_select(V, random_state=seed), seed)
        evaluate("rejection_arp", rejection_arp_select(V, random_state=seed), seed)
    return rows


def print_summary(rows: list[dict]) -> None:
    baseline = next(row for row in rows if row["method"] == "exact_cssp")
    print(f"E1: Gaussian {baseline['m']} x {baseline['n']}, k={baseline['k']}, "
          f"data seed={baseline['data_seed']}")
    print(f"Enumerated subsets: {comb(baseline['n'], baseline['k'])}")
    print(f"SVD baseline: {baseline['svd_optimal_error']:.10g}")
    print(f"Best CSSP error: {baseline['exact_cssp_error']:.10g}; "
          f"one minimizing subset (zero-based): {baseline['indices']}")
    print("\nMethod                 Runs      Mean error        Std error     Mean CSSP ratio")
    for method in ("exact_cssp", "osinsky", "arp", "rejection_arp"):
        runs = [row for row in rows if row["method"] == method]
        errors = np.array([row["frobenius_error"] for row in runs])
        std = float(errors.std(ddof=1)) if len(runs) > 1 else 0.0
        print(f"{method:<22} {len(runs):>4} {errors.mean():>16.8g} "
              f"{std:>16.8g} {ratio(float(errors.mean()), baseline['exact_cssp_error']):>19.8g}")
        if len(runs) > 1:
            best = min(runs, key=lambda row: row["frobenius_error"])
            print(f"  Best observed: error={best['frobenius_error']:.8g}, "
                  f"CSSP ratio={best['cssp_optimality_ratio']:.8g}, "
                  f"seed={best['selector_seed']}, indices={best['indices']}")
    print("\nOrdering checked with numerical tolerance: SVD <= best CSSP <= method error.")
    print("Best observed randomized results are not single-run guarantees.")


def main() -> None:
    args = parse_args()
    rows = run_experiment(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print_summary(rows)
    print(f"\nSaved all subsets and selector runs to {args.output}")


if __name__ == "__main__":
    main()
