"""Plot saved E4B summaries as two figures with dense/sparse panels; no reruns."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

METHODS = {"arp": "Sequential ARP", "rejection_arp": "Rejection ARP"}
MODES = {"selection_only": "Selection-only runtime",
         "end_to_end": "Basis + selection + factor-construction runtime"}


def load_results(run):
    config = json.loads((run / "config.json").read_text())
    if config.get("experiment") != "E4 rejection" or config.get("status") != "complete":
        raise ValueError("Provide a complete E4 rejection run containing both datasets (no skips).")
    with (run / "summary.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(d, mode, method, int(k)) for d in ("dense", "sparse")
                for mode in MODES for method in METHODS for k in config["ks"]}
    seen = set()
    for row in rows:
        row["k"] = int(row["k"])
        key = (row["dataset"], row["mode"], row["method"], row["k"])
        if key in seen or key not in expected:
            raise ValueError("Duplicate or unexpected summary row.")
        seen.add(key)
        if int(row["repeats"]) != config["repeats"]:
            raise ValueError("Summary repetition count differs from configuration.")
        values = [float(row[f"algorithm_seconds_{stat}"]) for stat in ("p25", "median", "p75")]
        if not all(math.isfinite(x) and x >= 0 for x in values) or values != sorted(values):
            raise ValueError("Invalid median/IQR timing values.")
        row["low"], row["median"], row["high"] = values
    if not expected or seen != expected:
        raise ValueError("Missing dense/sparse methods, ranks, or modes in summary.")
    for dataset in ("dense", "sparse"):
        shape = config["datasets_metadata"][dataset]["cssp_shape"]
        if len(shape) != 2 or any(int(x) <= 0 for x in shape):
            raise ValueError("Invalid CSSP matrix dimensions.")
    return config, rows


def generate(run, output=None):
    run = Path(run).resolve()
    output = Path(output) if output is not None else run / "combined_plots"
    config, rows = load_results(run)
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new --output directory.")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True)
    for mode, title in MODES.items():
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
        try:
            for ax, dataset in zip(axes, ("dense", "sparse")):
                ks = sorted(config["ks"])
                for index, (method, label) in enumerate(METHODS.items()):
                    group = sorted((r for r in rows if (r["dataset"], r["mode"], r["method"])
                                    == (dataset, mode, method)), key=lambda r: r["k"])
                    med = [r["median"] * 1000 for r in group]
                    errors = [[(r["median"] - r["low"]) * 1000 for r in group],
                              [(r["high"] - r["median"]) * 1000 for r in group]]
                    ax.errorbar([index] if len(ks) == 1 else ks, med, yerr=errors,
                                fmt="o" if len(ks) == 1 else "o-", capsize=5,
                                color=f"C{index}", label=label)
                if len(ks) == 1:
                    ax.set_xticks([0, 1], list(METHODS.values()))
                    ax.set_xlim(-0.5, 1.5)
                    ax.set_xlabel(f"Selected columns k = {ks[0]}")
                else:
                    ax.set_xticks(ks)
                    ax.set_xlabel("Selected columns k")
                    ax.legend()
                m, n = config["datasets_metadata"][dataset]["cssp_shape"]
                ax.set_title(f"{dataset.capitalize()}: {m:,} × {n:,}\n"
                             f"{config['repeats']} measured repetitions per method/k")
                ax.set_ylabel("Runtime (ms): median and 25–75% interval")
                ax.set_ylim(bottom=0)
                ax.grid(axis="y", alpha=0.25)
            fig.suptitle(f"E4B — {title}")
            fig.text(0.5, 0.02,
                     "Panels use independent y-scales. Different matrices, not a matched storage comparison.\n"
                     "Intervals show trial variability, not confidence intervals; factor timing excludes forming full C @ X.",
                     ha="center", fontsize=8)
            fig.tight_layout(rect=(0, 0.10, 1, 0.95))
            fig.savefig(output / f"{mode}_dense_sparse.png", dpi=160)
        finally:
            plt.close(fig)
    print(f"Saved two combined figures to {output}. No benchmarks were rerun.")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Saved E4B run directory")
    parser.add_argument("--output", type=Path, help="New output directory; default: RUN/combined_plots")
    args = parser.parse_args()
    try:
        generate(args.run, args.output)
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    except ImportError:
        parser.error('Install plots: python -m pip install -e ".[experiments]"')


if __name__ == "__main__":
    main()
