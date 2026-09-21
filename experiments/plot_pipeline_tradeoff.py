# E8 — Full-pipeline accuracy/runtime synthesis from saved, matched E7 trials.
"""Plot existing results only: no dataset loading or pipeline reruns."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
METHODS = {
    "deterministic_osinsky": "Exact SVD + Osinsky",
    "deterministic_arp": "Exact SVD + sequential ARP",
    "randomized_osinsky": "Randomized SVD + Osinsky",
    "randomized_arp": "Randomized SVD + sequential ARP",
    "deterministic_rejection_arp": "Exact SVD + Rejection ARP",
    "randomized_rejection_arp": "Randomized SVD + Rejection ARP",
}
COMPONENTS = ("basis_seconds", "selection_seconds", "reconstruction_seconds")


def load_run(path):
    """Require a complete, balanced E7 run; never join accuracy to other timings."""
    path = Path(path).resolve()
    config_bytes = (path / "config.json").read_bytes()
    raw_bytes = (path / "raw_results.csv").read_bytes()
    config = json.loads(config_bytes)
    if config.get("experiment") != "E7" or config.get("status") != "complete":
        raise ValueError(f"{path}: expected a complete E7 run")
    count = config.get("pipelines")
    if count not in (4, 6):
        raise ValueError(f"{path}: expected four or six pipelines")
    methods = list(METHODS)[:count]
    ks, trials = config["ks"], config["trials"]
    if not ks or len(set(ks)) != len(ks) or min(ks) < 1 or trials < 1:
        raise ValueError(f"{path}: invalid ranks or trial count")
    rows = list(csv.DictReader(raw_bytes.decode("utf-8").splitlines()))
    seen = set()
    groups = {}
    for row in rows:
        k, trial = int(row["k"]), int(row["trial"])
        method = row["method"]
        key = (k, trial, method)
        if (key in seen or k not in ks or not 0 <= trial < trials
                or method not in methods or row["dataset"] != config["dataset"]
                or int(row["seed"]) != config["seed"] + trial):
            raise ValueError(f"{path}: duplicate or mismatched trial {key}")
        seen.add(key)
        for metric in (*COMPONENTS, "algorithm_seconds", "relative_frobenius_error",
                       "optimal_relative_error"):
            row[metric] = float(row[metric])
            if not np.isfinite(row[metric]) or row[metric] < 0:
                raise ValueError(f"{path}: invalid {metric}")
        if not np.isclose(row["algorithm_seconds"], sum(row[x] for x in COMPONENTS),
                          rtol=1e-9, atol=1e-12):
            raise ValueError(f"{path}: algorithm time does not match component sum")
        groups.setdefault((k, method), []).append(row)
    if len(seen) != len(ks) * trials * count:
        raise ValueError(f"{path}: missing pipeline trials")

    summary = []
    for k in sorted(ks):
        baselines = [r["optimal_relative_error"] for m in methods for r in groups[k, m]]
        if not np.allclose(baselines, baselines[0], rtol=1e-10, atol=0):
            raise ValueError(f"{path}: inconsistent SVD baseline for k={k}")
        for method in methods:
            group = groups[k, method]
            errors = np.array([r["relative_frobenius_error"] for r in group])
            times = np.array([r["algorithm_seconds"] for r in group])
            summary.append(dict(
                dataset=config["dataset"], k=k, method=method, trials=trials,
                relative_error_mean=float(errors.mean()),
                relative_error_std=float(errors.std(ddof=1)) if trials > 1 else float("nan"),
                algorithm_seconds_median=float(np.median(times)),
                algorithm_seconds_p25=float(np.percentile(times, 25)),
                algorithm_seconds_p75=float(np.percentile(times, 75)),
                optimal_relative_error=baselines[0],
            ))
    provenance = dict(
        path=str(path), config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        raw_results_sha256=hashlib.sha256(raw_bytes).hexdigest(), config=config,
    )
    return summary, provenance


def plot_rank(rows, config, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6.5))
    try:
        for row in rows:
            method = row["method"]
            x = row["algorithm_seconds_median"] * 1000
            xerr = [[x - row["algorithm_seconds_p25"] * 1000],
                    [row["algorithm_seconds_p75"] * 1000 - x]]
            std = row["relative_error_std"]
            ax.errorbar(
                x, row["relative_error_mean"], xerr=xerr,
                yerr=std if np.isfinite(std) else None,
                fmt="s" if method.startswith("deterministic") else "o",
                color=f"C{list(METHODS).index(method)}", markersize=7, capsize=3,
                label=METHODS[method],
            )
        ax.axhline(rows[0]["optimal_relative_error"], color="black", linestyle="--",
                   label="Optimal rank-k SVD error (accuracy reference only)")
        m, n = config["matrix_shape"]
        ax.set(
            xlabel="Median reported algorithm time (ms) — lower is better",
            ylabel="Mean relative Frobenius error — lower is better",
            title=f"E8 — {config['dataset']}, {m} × {n}, k={rows[0]['k']}\n"
                  f"{rows[0]['trials']} trials on one fixed matrix",
        )
        ax.grid(alpha=0.2)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
        fig.text(0.5, 0.025,
                 "Bars: runtime IQR; error ± sample SD (not confidence intervals).\n"
                 "Descriptive component-sum timings; not controlled standalone wall times.",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        fig.savefig(destination, dpi=160)
    finally:
        plt.close(fig)


def generate_report(runs, output):
    output = Path(output)
    if output.exists():
        raise ValueError(f"Output already exists: {output}; choose a new --run-name")
    paths = [Path(path).resolve() for path in runs]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Provide at least one run, without duplicate paths")
    loaded = [load_run(path) for path in paths]
    # Check the optional dependency before creating the report directory.
    import matplotlib  # noqa: F401

    output.mkdir(parents=True)
    (output / "plots").mkdir()
    config = dict(
        experiment="E8", status="running", analysis_only=True,
        sources=[source for _, source in loaded],
        policy="Each plot uses one source run and one k; no pooling across datasets, ranks or runs. "
               "Mean error and median algorithm time come from the same trial rows. "
               "Bars are sample SD and timing IQR, not confidence intervals or significance tests.",
    )

    def save_config():
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    save_config()
    try:
        combined = []
        for index, (summary, source) in enumerate(loaded, start=1):
            for row in summary:
                combined.append(dict(source_id=index, source_run=source["path"], **row))
            for k in sorted({row["k"] for row in summary}):
                plot_rank([row for row in summary if row["k"] == k], source["config"],
                          output / "plots" / f"source_{index}_k{k}_tradeoff.png")
        with (output / "summary.csv").open("x", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(combined[0]))
            writer.writeheader()
            writer.writerows(combined)
        config["status"] = "complete"
    except (Exception, KeyboardInterrupt) as error:
        config.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save_config()
    print(f"Saved E8 plots and summary to {output}; no pipelines were rerun.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True,
                        help="completed E7 directories containing config.json and raw_results.csv")
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ"))
    args = parser.parse_args()
    if (not args.run_name or args.run_name in (".", "..")
            or Path(args.run_name).name != args.run_name or "\\" in args.run_name):
        parser.error("run-name must be a single directory name")
    try:
        generate_report(args.runs, ROOT / "results/e8" / args.run_name)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    except ImportError:
        parser.error('Install plotting dependencies: python -m pip install -e ".[experiments]"')


if __name__ == "__main__":
    main()
