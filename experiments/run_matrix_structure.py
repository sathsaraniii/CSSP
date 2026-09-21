# E5 — Matrix structure versus accuracy and standalone runtime at fixed size.
"""Six normalized matrix families; matrix and algorithm randomness kept separate.

Defaults: 500 x 500, k=10, signal rank=25, three instances per family,
ten accuracy trials and ten timed repetitions. Results: results/e5/<run-name>.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_runtime_scaling import controlled_blas, numpy_config, measure, TIMINGS  # noqa: E402
from cssp.cssp_pipeline import SIX_PIPELINE_CONFIGS  # noqa: E402
from cssp.basis.randomized_svd import randomized_svd  # noqa: E402
from cssp.column_selection.arp import arp_select  # noqa: E402
from cssp.column_selection.osinsky import osinsky_select  # noqa: E402
from cssp.column_selection.rejection_arp import rejection_arp_select  # noqa: E402
from cssp.reconstruction.cssp_approximation import orthogonal_cssp_approximation  # noqa: E402

FAMILIES = ("gaussian", "exact_low_rank", "small_noise", "large_noise", "rapid_decay", "slow_decay")
ACCURACY = ("frobenius_error", "relative_error", "svd_optimality_ratio")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=500)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--signal-rank", type=int, default=25)
    p.add_argument("--instances", type=int, default=3)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--p", type=int, default=10)
    p.add_argument("--q", type=int, default=1)
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--trial-seed", type=int, default=0)
    p.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ"))
    args = p.parse_args(argv)
    if not 1 <= args.k < args.signal_rank <= args.size:
        p.error("require 1 <= k < signal-rank <= size (avoid trivial low-rank reconstruction)")
    if args.instances < 1 or min(args.trials, args.repeats) < 2:
        p.error("instances >= 1 and trials/repeats >= 2 required")
    if min(args.p, args.q, args.data_seed, args.trial_seed) < 0:
        p.error("p, q and seeds must be nonnegative")
    if len(set(args.families)) != len(args.families):
        p.error("families must be unique")
    if not args.run_name or args.run_name in (".", "..") or Path(args.run_name).name != args.run_name or "\\" in args.run_name:
        p.error("run-name must be a single directory name")
    return args


def matrix_components(size, rank, data_seed, instance):
    """Shared signal/noise directions and full orthogonal factors per instance."""
    children = np.random.SeedSequence([data_seed, instance]).spawn(4)
    U, _ = np.linalg.qr(np.random.default_rng(children[0]).standard_normal((size, size)))
    V, _ = np.linalg.qr(np.random.default_rng(children[1]).standard_normal((size, size)))
    signal = (U[:, :rank] * np.geomspace(20., 1., rank)) @ V[:, :rank].T
    noise = np.random.default_rng(children[2]).standard_normal((size, size))
    noise *= np.linalg.norm(signal, "fro") / np.linalg.norm(noise, "fro")
    gaussian = np.random.default_rng(children[3]).standard_normal((size, size))
    return U, V, signal, noise, gaussian


def make_family(family, components):
    U, V, signal, noise, gaussian = components
    if family == "gaussian":
        A = gaussian.copy()
    elif family == "exact_low_rank":
        A = signal.copy()
    elif family in ("small_noise", "large_noise"):
        A = signal + (0.01 if family == "small_noise" else 0.5) * noise
    elif family == "rapid_decay":
        A = (U * np.geomspace(1., 1e-8, len(U))) @ V.T
    elif family == "slow_decay":
        A = (U / np.arange(1, len(U) + 1, dtype=float)) @ V.T
    else:
        raise ValueError(f"Unknown family: {family}")
    A /= np.linalg.norm(A, "fro")
    return A


def seeds(args, instance, family, trial, phase):
    children = np.random.SeedSequence([args.trial_seed, instance, FAMILIES.index(family), trial, phase]).spawn(4)
    return [int(child.generate_state(1)[0]) for child in children]


def accuracy_rows(A, s, Vt, args, instance, family):
    exact = Vt[:args.k].T
    optimal = float(np.linalg.norm(s[args.k:]))
    norm_A = float(np.linalg.norm(A, "fro"))
    tolerance = 100 * np.finfo(float).eps * args.size * norm_A

    def evaluate(method, V, trial, bs, ss):
        selector = SIX_PIPELINE_CONFIGS[method][1]
        if selector == "osinsky":
            J = osinsky_select(A, V)
        else:
            J = (arp_select if selector == "arp" else rejection_arp_select)(V, random_state=ss)
        error = float(np.linalg.norm(A - orthogonal_cssp_approximation(A, J), "fro"))
        if not np.isfinite(error) or error + tolerance < optimal:
            raise RuntimeError("Invalid reconstruction error or SVD lower bound violation")
        return dict(method=method, trial=trial, basis_seed=bs, selector_seed=ss,
                    indices=json.dumps(J.tolist()), frobenius_error=error,
                    relative_error=error / norm_A, svd_optimality_ratio=error / optimal if optimal else float("nan"),
                    svd_optimal_error=optimal)

    yield evaluate("deterministic_osinsky", exact, None, None, None)
    for trial in range(args.trials):
        bs, arp_seed, rejection_seed, _ = seeds(args, instance, family, trial, 0)
        yield evaluate("deterministic_arp", exact, trial, None, arp_seed)
        yield evaluate("deterministic_rejection_arp", exact, trial, None, rejection_seed)
        _, _, Vrand = randomized_svd(A, args.k, p=args.p, q=args.q, random_state=bs)
        yield evaluate("randomized_osinsky", Vrand.T, trial, bs, None)
        yield evaluate("randomized_arp", Vrand.T, trial, bs, arp_seed)
        yield evaluate("randomized_rejection_arp", Vrand.T, trial, bs, rejection_seed)


def runtime_rows(A, args, instance, family):
    """Untimed warm-up, then standalone measurements via the verified E4 timer."""
    for repetition in range(-1, args.repeats):
        bs, arp_seed, rejection_seed, order_seed = seeds(args, instance, family, repetition + 1, 1)
        order = np.random.default_rng(order_seed).permutation(list(SIX_PIPELINE_CONFIGS))
        for position, method in enumerate(order):
            ss = rejection_seed if SIX_PIPELINE_CONFIGS[method][1] == "rejection_arp" else arp_seed
            row = measure(A, str(method), args, bs, ss)
            if repetition >= 0:
                yield dict(row, trial=repetition, execution_position=position, order_seed=order_seed)


def summarize(rows):
    """First summarize within each matrix; then equally weight matrix instances."""
    groups = {}
    for row in rows:
        groups.setdefault((row["family"], row["instance"], row["mode"], row["method"]), []).append(row)
    within = []
    across = {}
    for (family, instance, mode, method), group in groups.items():
        for metric in (ACCURACY if mode == "accuracy" else TIMINGS):
            a = np.asarray([row[metric] for row in group], dtype=float)
            if np.isfinite(a).all():
                stats = [a.mean(), a.std(ddof=1) if len(a) > 1 else 0., np.median(a), *np.percentile(a, [25, 75]), a.min(), a.max()]
            else:
                stats = [float("nan")] * 7
            item = dict(family=family, instance=instance, mode=mode, method=method, metric=metric, runs=len(group),
                        **dict(zip(("mean", "std", "median", "p25", "p75", "min", "max"), map(float, stats))))
            within.append(item)
            across.setdefault((family, mode, method, metric), []).append(item)
    summary = []
    for (family, mode, method, metric), items in across.items():
        center = "mean" if mode == "accuracy" else "median"
        values = np.asarray([r[center] for r in items])
        finite = np.isfinite(values).all()
        summary.append(dict(family=family, mode=mode, method=method, metric=metric,
                            instances=len(items), total_runs=sum(r["runs"] for r in items),
                            within_instance_center=center, mean_of_instance_centers=float(values.mean()) if finite else float("nan"),
                            between_instance_std=float(values.std(ddof=1)) if finite and len(values) > 1 else float("nan")))
    return within, summary


def plots(summary, spectra, directory, families):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir()
    fig, ax = plt.subplots(figsize=(9, 5))
    for family in families:
        group = [r for r in spectra if r["family"] == family]
        for instance in sorted({r["instance"] for r in group}):
            values = [r["singular_value"] for r in group if r["instance"] == instance]
            ax.semilogy(range(1, len(values)+1), values, alpha=.6, label=f"{family}, instance {instance}")
    ax.set(xlabel="Singular-value index", ylabel="Singular value (unit Frobenius norm)", title="E5 — measured spectra")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(directory / "spectra.png", dpi=150)
    plt.close(fig)
    for metric, mode in (("relative_error", "accuracy"), ("svd_optimality_ratio", "accuracy"), ("algorithm_seconds", "runtime")):
        fig, ax = plt.subplots(figsize=(11, 6))
        for j, method in enumerate(SIX_PIPELINE_CONFIGS):
            records = [next(r for r in summary if (r["family"],r["mode"],r["method"],r["metric"]) == (f,mode,method,metric)) for f in families]
            x = np.arange(len(families)) + (j-2.5)*.12
            y = [r["mean_of_instance_centers"] for r in records]
            ax.plot(x, y, marker="o", linestyle="none", label=method)
            for xi, yi, r in zip(x, y, records):
                if np.isfinite(r["between_instance_std"]):
                    ax.errorbar(xi, yi, yerr=r["between_instance_std"], fmt="none", ecolor=f"C{j}", capsize=3)
        ax.set_xticks(range(len(families)), families, rotation=20, ha="right")
        ax.set(ylabel=metric.replace("_", " "), title="E5 — means of matrix-instance " + ("means" if mode == "accuracy" else "runtime medians"))
        ax.text(.5, -.32, "Bars: between-matrix-instance sample SD, not trial SD or confidence intervals; omitted for one instance.",
                transform=ax.transAxes, ha="center", fontsize=8)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(directory / f"{metric}.png", dpi=150)
        plt.close(fig)


def write_csv(path, rows):
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    output = ROOT / "results/e5" / args.run_name
    if output.exists():
        raise SystemExit(f"Run exists: {output}; choose another run-name")
    if not args.no_plots:
        import matplotlib  # noqa: F401
    build = numpy_config()
    with controlled_blas(1, build) as snapshot:
        output.mkdir(parents=True)
        config = dict(vars(args), experiment="E5", status="running", numpy_version=np.__version__,
                      python_version=platform.python_version(), platform=platform.platform(),
                      blas=build, thread_control=snapshot(), matrices=[],
                      normalization="Every matrix has Frobenius norm 1",
                      spectra="Signal: geomspace(20,1,signal_rank); rapid: geomspace(1,1e-8,size); slow: 1/i",
                      noise="1% or 50% of signal Frobenius norm before final normalization; shared direction",
                      aggregation="Within-matrix accuracy means or runtime medians, then equally weighted across instances",
                      runtime="standalone, one warm-up per method/matrix, orthogonal reconstruction including extraction; no error/baseline work")
        def save():
            (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        save()
        rows, spectra = [], []
        fields = ["family", "instance", "mode", "size", "k", "data_seed", "trial_seed", "p", "q", "method", "trial",
                  "basis_seed", "selector_seed", "order_seed", "execution_position", "indices", *ACCURACY, "svd_optimal_error", *TIMINGS]
        try:
            with (output / "raw_results.csv").open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for instance in range(args.instances):
                    components = matrix_components(args.size, args.signal_rank, args.data_seed, instance)
                    for family in args.families:
                        A = make_family(family, components)
                        _, s, Vt = np.linalg.svd(A, full_matrices=False)
                        spectra.extend(dict(family=family, instance=instance, index=i+1, singular_value=float(v)) for i,v in enumerate(s))
                        config["matrices"].append(dict(family=family, instance=instance, sha256=hashlib.sha256(A.tobytes()).hexdigest(),
                                                       svd_optimal_error=float(np.linalg.norm(s[args.k:]))))
                        save()
                        for mode in ("accuracy", "runtime"):
                            print(f"{family}, instance {instance+1}/{args.instances}: {mode}", flush=True)
                            iterator = accuracy_rows(A, s, Vt, args, instance, family) if mode == "accuracy" else runtime_rows(A,args,instance,family)
                            for measured in iterator:
                                row = dict(family=family, instance=instance, mode=mode, size=args.size,k=args.k,
                                           data_seed=args.data_seed,trial_seed=args.trial_seed,p=args.p,q=args.q,**measured)
                                writer.writerow(row)
                                stream.flush()
                                rows.append(row)
            within, summary = summarize(rows)
            write_csv(output / "instance_summary.csv", within)
            write_csv(output / "summary.csv", summary)
            write_csv(output / "spectra.csv", spectra)
            if not args.no_plots:
                plots(summary, spectra, output / "plots", args.families)
            config["status"] = "complete"
        except (Exception, KeyboardInterrupt) as error:
            config.update(status="incomplete", error=f"{type(error).__name__}: {error}")
            save()
            raise
        save()
    print(f"Saved {len(rows)} measurements to {output}")


if __name__ == "__main__":
    main()
