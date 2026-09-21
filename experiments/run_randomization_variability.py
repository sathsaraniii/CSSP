# E3 — Selection variability with a fixed matrix, k, and exact SVD basis.
"""Compare ARP/Rejection ARP error distributions against one Osinsky result.

Default: synthetic and full DNA data, k=10, 100 trials per randomized selector.
Only orthogonal reconstruction is evaluated; no randomized SVD or timing study.
Outputs go to results/e3/<run-name> without overwriting existing runs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_dna_experiment import DATASET_PATH, load_gse10072  # noqa: E402
from run_experiment import make_test_matrix  # noqa: E402
from cssp.column_selection.arp import arp_select  # noqa: E402
from cssp.column_selection.osinsky import osinsky_select  # noqa: E402
from cssp.column_selection.rejection_arp import rejection_arp_select  # noqa: E402
from cssp.reconstruction.cssp_approximation import orthogonal_cssp_approximation  # noqa: E402

METRICS = ("frobenius_error", "relative_error", "svd_optimality_ratio", "delta_error_vs_osinsky")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("synthetic", "dna"), default=["synthetic", "dna"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--m", type=int, default=160)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--numerical-rank", type=int, default=25)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--trial-seed", type=int, default=0)
    parser.add_argument("--dna-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if min(args.k, args.m, args.n, args.numerical_rank) < 1 or args.trials < 2:
        parser.error("dimensions/k/signal rank must be positive; trials must be at least 2")
    if min(args.data_seed, args.trial_seed) < 0 or not np.isfinite(args.noise) or args.noise < 0:
        parser.error("seeds must be nonnegative; noise must be finite and nonnegative")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("datasets must not repeat")
    if (not args.run_name or args.run_name in (".", "..")
            or Path(args.run_name).name != args.run_name or "\\" in args.run_name):
        parser.error("run-name must be a single directory name")
    return args


def selector_seeds(master, dataset, k, trial):
    """Independent, reproducible streams, unaffected by dataset execution order."""
    dataset_id = {"synthetic": 0, "dna": 1}[dataset]
    children = np.random.SeedSequence([master, dataset_id, k, trial]).spawn(2)
    return [int(child.generate_state(1)[0]) for child in children]


def classify(delta, tolerance):
    if delta < -tolerance:
        return "better"
    if delta > tolerance:
        return "worse"
    return "tied"


def dataset_rows(A, dataset, sample_ids, args):
    """One SVD, one Osinsky run, and repeated selectors on the very same V."""
    if not 1 <= args.k <= min(A.shape):
        raise ValueError(f"k exceeds dimensions of {dataset}: {A.shape}")
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt[:args.k].T.copy()
    norm_A = float(np.linalg.norm(A, "fro"))
    optimal = float(np.linalg.norm(s[args.k:]))
    # Absolute Frobenius-error tolerance, not a threshold for zero baselines.
    tolerance = 100 * np.finfo(float).eps * max(A.shape) * norm_A

    def evaluate(method, J, seed=None, trial=None, reference=None):
        error = float(np.linalg.norm(A - orthogonal_cssp_approximation(A, J), "fro"))
        if not np.isfinite(error) or error + tolerance < optimal:
            raise RuntimeError("Nonfinite error or violation of SVD lower bound")
        reference = error if reference is None else reference
        delta = error - reference
        return dict(dataset=dataset, m=A.shape[0], n=A.shape[1], k=args.k,
                    method=method, basis="deterministic_svd", reconstruction="orthogonal",
                    trial=trial, selector_seed=seed,
                    data_seed=args.data_seed if dataset == "synthetic" else None,
                    indices=json.dumps(J.tolist()),
                    sample_ids=json.dumps([sample_ids[j] for j in J]) if sample_ids is not None else "",
                    frobenius_error=error,
                    relative_error=error / norm_A if norm_A else (0.0 if error == 0 else float("inf")),
                    svd_optimal_error=optimal,
                    svd_optimality_ratio=error / optimal if optimal > 0 else float("nan"),
                    osinsky_error=reference, delta_error_vs_osinsky=delta,
                    comparison_tolerance=tolerance, comparison_vs_osinsky=classify(delta, tolerance))

    reference = evaluate("osinsky", osinsky_select(A, V))
    yield reference
    for trial in range(args.trials):
        seeds = selector_seeds(args.trial_seed, dataset, args.k, trial)
        for method, select, seed in zip(("arp", "rejection_arp"), (arp_select, rejection_arp_select), seeds):
            yield evaluate(method, select(V, random_state=seed), seed, trial, reference["frobenius_error"])
        if (trial + 1) % 10 == 0 or trial + 1 == args.trials:
            print(f"{dataset}: {trial + 1}/{args.trials} trials complete", flush=True)


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["k"], row["method"]), []).append(row)
    summaries = []
    for (dataset, k, method), group in groups.items():
        result = dict(dataset=dataset, k=k, method=method, runs=len(group),
                      osinsky_error=group[0]["osinsky_error"], comparison_tolerance=group[0]["comparison_tolerance"])
        for metric in METRICS:
            values = np.array([row[metric] for row in group])
            if np.isfinite(values).all():
                stats = [values.mean(), values.std(ddof=1) if len(values) > 1 else 0.0,
                         np.median(values), *np.percentile(values, [5, 95]), values.min(), values.max()]
            else:
                stats = [float("nan")] * 7
            result.update({f"{metric}_{name}": float(value) for name, value in
                           zip(("mean", "std", "median", "p05", "p95", "min", "max"), stats)})
        for label in ("better", "tied", "worse"):
            count = sum(row["comparison_vs_osinsky"] == label for row in group)
            result[f"{label}_count"] = count
            result[f"{label}_fraction"] = count / len(group)
        summaries.append(result)
    return summaries


def write_plots(rows, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir()
    for dataset in dict.fromkeys(row["dataset"] for row in rows):
        subset = [row for row in rows if row["dataset"] == dataset]
        reference = next(row for row in subset if row["method"] == "osinsky")
        fig, ax = plt.subplots(figsize=(8, 5))
        values = [[row["relative_error"] for row in subset if row["method"] == method]
                  for method in ("arp", "rejection_arp")]
        ax.boxplot(values, whis=(5, 95), showmeans=True)
        ax.set_xticks([1, 2], ["ARP", "Rejection ARP"])
        ax.axhline(reference["relative_error"], color="tab:red", linestyle="--", label="Osinsky (fixed)")
        ax.set(ylabel="Relative Frobenius error",
               title=f"E3 — {dataset}, k={reference['k']}, {len(values[0])} trials")
        ax.text(0.5, -0.16, "Boxes: 25–75%; whiskers: 5–95%; points outside shown. Not confidence intervals.",
                transform=ax.transAxes, ha="center", fontsize=8)
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(directory / f"{dataset}_variability.png", dpi=160)
        plt.close(fig)


def main():
    args = parse_args()
    output = PROJECT_ROOT / "results" / "e3" / args.run_name
    if output.exists():
        raise SystemExit(f"Run directory already exists: {output}; choose a new --run-name")
    if not args.no_plots:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            raise SystemExit('Install plotting support: python -m pip install -e ".[experiments]", or use --no-plots')
    datasets = {}
    for name in args.datasets:
        datasets[name] = ((make_test_matrix(args.m, args.n, args.numerical_rank, args.noise, args.data_seed), None)
                          if name == "synthetic" else load_gse10072(args.dna_path))
        if args.k > min(datasets[name][0].shape):
            raise SystemExit(f"k exceeds dimensions of {name}: {datasets[name][0].shape}")
    output.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), dna_path=str(args.dna_path.resolve()), experiment="E3", status="running",
                  numpy_version=np.__version__, python_version=platform.python_version(),
                  basis="fixed_exact_svd", reconstruction="orthogonal",
                  tolerance_rule="100 * float64_eps * max(m,n) * Frobenius_norm(A)",
                  percentile_note="Empirical 5th/95th percentiles, not confidence intervals",
                  datasets_metadata={name: {"shape": list(A.shape), "sha256": hashlib.sha256(A.tobytes()).hexdigest()}
                                     for name, (A, _) in datasets.items()})
    def save_config():
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    save_config()
    rows = []
    try:
        with (output / "raw_results.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = None
            for name, (A, sample_ids) in datasets.items():
                for row in dataset_rows(A, name, sample_ids, args):
                    if writer is None:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    stream.flush()
                    rows.append(row)
        summaries = summarize(rows)
        with (output / "summary.csv").open("x", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        if not args.no_plots:
            write_plots(rows, output / "plots")
    except (Exception, KeyboardInterrupt) as error:
        config.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        save_config()
        raise
    config["status"] = "complete"
    save_config()
    for row in summaries:
        print(f"{row['dataset']} {row['method']}: relative error mean={row['relative_error_mean']:.6g}, "
              f"std={row['relative_error_std']:.6g}; better/tied/worse="
              f"{row['better_count']}/{row['tied_count']}/{row['worse_count']}")
    print(f"Saved {len(rows)} measurements to {output}")


if __name__ == "__main__":
    main()
