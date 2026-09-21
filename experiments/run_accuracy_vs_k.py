# E2 — Accuracy versus selected-column count on synthetic and DNA matrices.
"""Run six CSSP combinations on fixed datasets with independent seed streams.

Example pilot:
    python experiments/run_accuracy_vs_k.py --datasets synthetic --ks 5 10 --trials 3

Default: both datasets, k=5,10,20,30,40, 30 randomized trials. Only orthogonal
reconstruction is evaluated. This is not a runtime benchmark. Each output
directory is new; partial raw results survive if a run is interrupted.
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
from cssp.basis.randomized_svd import randomized_svd  # noqa: E402
from cssp.column_selection.arp import arp_select  # noqa: E402
from cssp.column_selection.osinsky import osinsky_select  # noqa: E402
from cssp.column_selection.rejection_arp import rejection_arp_select  # noqa: E402
from cssp.reconstruction.cssp_approximation import orthogonal_cssp_approximation  # noqa: E402

METRICS = ("frobenius_error", "relative_error", "svd_optimality_ratio")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=("synthetic", "dna"),
                        default=["synthetic", "dna"])
    parser.add_argument("--ks", type=int, nargs="+", default=[5, 10, 20, 30, 40])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--m", type=int, default=160)
    parser.add_argument("--n", type=int, default=120)
    parser.add_argument("--numerical-rank", type=int, default=25)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--trial-seed", type=int, default=0)
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--dna-path", type=Path, default=DATASET_PATH)
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ"))
    parser.add_argument("--no-plots", action="store_true",
                        help="write CSV/config only; matplotlib is otherwise required")
    args = parser.parse_args()
    if min(args.m, args.n, args.numerical_rank) < 1 or args.trials < 2:
        parser.error("dimensions/rank must be positive and trials must be at least 2")
    if min(args.data_seed, args.trial_seed, args.p, args.q) < 0:
        parser.error("seeds, oversampling and power iterations must be nonnegative")
    if not np.isfinite(args.noise) or args.noise < 0:
        parser.error("noise must be finite and nonnegative")
    if min(args.ks) < 1 or len(set(args.ks)) != len(args.ks):
        parser.error("ks must be distinct positive integers")
    if len(set(args.datasets)) != len(args.datasets):
        parser.error("datasets must not repeat")
    if args.run_name in (".", "..") or Path(args.run_name).name != args.run_name or "\\" in args.run_name:
        parser.error("run-name must be a single directory name")
    args.ks.sort()
    return args


def trial_seeds(master, dataset_id, k, trial):
    """Independent basis, ARP and rejection seeds, stable across sweep order."""
    children = np.random.SeedSequence([master, dataset_id, k, trial]).spawn(3)
    return [int(child.generate_state(1)[0]) for child in children]


def dataset_rows(A, dataset, sample_ids, args):
    """Yield runs, sharing one full exact SVD and one randomized basis/trial."""
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    norm_A = float(np.linalg.norm(A, "fro"))
    tolerance = 100 * np.finfo(float).eps * max(A.shape) * norm_A
    dataset_id = {"synthetic": 0, "dna": 1}[dataset]

    def evaluate(k, basis_kind, selector, V, trial, basis_seed, selection_seed):
        if selector == "osinsky":
            J = osinsky_select(A, V)
        elif selector == "arp":
            J = arp_select(V, random_state=selection_seed)
        else:
            J = rejection_arp_select(V, random_state=selection_seed)
        error = float(np.linalg.norm(A - orthogonal_cssp_approximation(A, J), "fro"))
        optimal = float(np.linalg.norm(s[k:]))
        if not np.isfinite(error) or error + tolerance < optimal:
            raise RuntimeError("Nonfinite error or error below SVD baseline beyond tolerance")
        return dict(dataset=dataset, m=A.shape[0], n=A.shape[1], k=k,
                    method=f"{basis_kind}_{selector}", trial=trial,
                    data_seed=args.data_seed if dataset == "synthetic" else None,
                    basis_seed=basis_seed, selection_seed=selection_seed,
                    oversampling=args.p if basis_kind == "randomized" else None,
                    power_iterations=args.q if basis_kind == "randomized" else None,
                    reconstruction="orthogonal", indices=json.dumps(J.tolist()),
                    sample_ids=json.dumps([sample_ids[j] for j in J]) if sample_ids else "",
                    frobenius_error=error, relative_error=error / norm_A if norm_A else (0.0 if error == 0 else float("inf")),
                    svd_optimal_error=optimal, svd_relative_error=optimal / norm_A if norm_A else 0.0,
                    svd_optimality_ratio=error / optimal if optimal > 0 else float("nan"))

    for k in args.ks:
        print(f"{dataset}: k={k}, {args.trials} randomized trials", flush=True)
        exact = Vt[:k].T.copy()
        yield evaluate(k, "deterministic", "osinsky", exact, None, None, None)
        for trial in range(args.trials):
            basis_seed, arp_seed, rejection_seed = trial_seeds(args.trial_seed, dataset_id, k, trial)
            for selector, seed in (("arp", arp_seed), ("rejection_arp", rejection_seed)):
                yield evaluate(k, "deterministic", selector, exact, trial, None, seed)
            _, _, rand_Vt = randomized_svd(A, k, p=args.p, q=args.q, random_state=basis_seed)
            for selector, seed in (("osinsky", None), ("arp", arp_seed), ("rejection_arp", rejection_seed)):
                yield evaluate(k, "randomized", selector, rand_Vt.T, trial, basis_seed, seed)
            print(f"  trial {trial + 1}/{args.trials} complete", flush=True)


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["k"], row["method"]), []).append(row)
    summaries = []
    for (dataset, k, method), group in groups.items():
        summary = dict(dataset=dataset, k=k, method=method, runs=len(group),
                       svd_relative_error=group[0]["svd_relative_error"])
        for metric in METRICS:
            values = np.array([row[metric] for row in group])
            if not np.isfinite(values).all():
                stats = [float("nan")] * 5
            else:
                stats = [float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                         float(np.median(values)), float(values.min()), float(values.max())]
            summary.update({f"{metric}_{name}": value for name, value in zip(("mean", "std", "median", "min", "max"), stats)})
        summaries.append(summary)
    return summaries


def write_plots(summaries, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir()
    for dataset in dict.fromkeys(row["dataset"] for row in summaries):
        subset = [row for row in summaries if row["dataset"] == dataset]
        for metric in ("relative_error", "svd_optimality_ratio"):
            fig, ax = plt.subplots(figsize=(10, 6))
            for method in dict.fromkeys(row["method"] for row in subset):
                runs = sorted((row for row in subset if row["method"] == method), key=lambda row: row["k"])
                ks = [row["k"] for row in runs]
                mean = np.array([row[f"{metric}_mean"] for row in runs])
                std = np.array([row[f"{metric}_std"] for row in runs])
                line, = ax.plot(ks, mean, marker="o", label=method)
                if runs[0]["runs"] > 1:
                    ax.fill_between(ks, mean - std, mean + std, alpha=0.12, color=line.get_color())
            baseline = sorted({row["k"]: row["svd_relative_error"] for row in subset}.items())
            ax.plot([k for k, _ in baseline], [v if metric == "relative_error" else 1 for _, v in baseline],
                    "k--", label="SVD baseline")
            ax.set(xlabel="Selected columns k", ylabel=metric.replace("_", " "),
                   title=f"E2 — {dataset}: means ± sample standard deviation")
            ax.legend(fontsize="small")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(output / f"{dataset}_{metric}.png", dpi=160)
            plt.close(fig)


def main():
    args = parse_args()
    if not args.no_plots:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            raise SystemExit('Install plotting support: python -m pip install -e ".[experiments]", or use --no-plots')
    output = PROJECT_ROOT / "results" / "e2" / args.run_name
    datasets = {}
    for name in args.datasets:
        if name == "synthetic":
            datasets[name] = (make_test_matrix(args.m, args.n, args.numerical_rank, args.noise, args.data_seed), None)
        else:
            datasets[name] = load_gse10072(args.dna_path)
        A, _ = datasets[name]
        if max(args.ks) > min(A.shape):
            raise SystemExit(f"k exceeds min(shape) for {name}: {A.shape}")
    output.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), dna_path=str(args.dna_path.resolve()), experiment="E2", status="running",
                  numpy_version=np.__version__, python_version=platform.python_version(),
                  datasets_metadata={name: {"shape": list(A.shape), "sha256": hashlib.sha256(A.tobytes()).hexdigest()}
                                     for name, (A, _) in datasets.items()})
    def save_config():
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    save_config()
    rows = []
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
        write_plots(summaries, output / "plots")
    config["status"] = "complete"
    save_config()
    print(f"Saved {len(rows)} measurements to {output}")


if __name__ == "__main__":
    main()
