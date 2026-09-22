"""Shared E7 output/trial support for the existing DNA and Meszaros runners."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from cssp import run_four_pipelines, run_six_pipelines

ROOT = Path(__file__).resolve().parents[1]
TIMINGS = ("basis_seconds", "selection_seconds", "reconstruction_seconds", "algorithm_seconds")
ERRORS = ("frobenius_error", "relative_frobenius_error", "frobenius_optimality_ratio")


def add_arguments(parser):
    parser.add_argument("--pipelines", type=int, choices=(4, 6), default=4)
    ranks = parser.add_mutually_exclusive_group()
    ranks.add_argument("--k", type=int, help="one selected-column count")
    ranks.add_argument("--ks", type=int, nargs="+", help="multiple selected-column counts")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42, help="trial t uses seed+t for basis and selector")
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ"))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--show-indices", action="store_true", help="also print per-trial selections; always saved in CSV")


def validate_arguments(parser, args, default_k):
    args.ks = args.ks if args.ks is not None else [args.k if args.k is not None else default_k]
    if args.trials < 1 or min(args.ks) < 1 or len(set(args.ks)) != len(args.ks):
        parser.error("trials and ks must be positive; ks must be distinct")
    if min(args.seed, args.p, args.q) < 0:
        parser.error("seed, p and q must be nonnegative")
    if not args.run_name or args.run_name in (".", "..") or Path(args.run_name).name != args.run_name or "\\" in args.run_name:
        parser.error("run-name must be a single directory name")
    args.ks.sort()
    return args


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["k"], row["method"]), []).append(row)
    summary = []
    for (k, method), group in groups.items():
        item = dict(k=k, method=method, trials=len(group), optimal_rank_k_error=group[0]["optimal_rank_k_error"],
                    optimal_relative_error=group[0]["optimal_relative_error"])
        for metric in (*ERRORS, *TIMINGS):
            values = np.asarray([r[metric] for r in group])
            stats = ([values.mean(), values.std(ddof=1) if len(values)>1 else float("nan"),
                      np.median(values), *np.percentile(values,[25,75])]
                     if np.isfinite(values).all() else [float("nan")]*5)
            item.update({f"{metric}_{stat}":float(value) for stat,value in zip(("mean","std","median","p25","p75"),stats)})
        summary.append(item)
    return summary


def plot_comparison(summary, output, dataset):
    """Only two complementary plots: accuracy and descriptive algorithm timing."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir()
    ks=sorted({row["k"] for row in summary})
    methods=list(dict.fromkeys(row["method"] for row in summary))
    for metric,center,low,high,title in (
        ("relative_frobenius_error","mean",None,None,"Relative reconstruction error: mean ± sample SD"),
        ("algorithm_seconds","median","p25","p75","Reported algorithm time: median and interquartile range"),
    ):
        fig,ax=plt.subplots(figsize=(10,6))
        if len(ks)==1:
            ordered=[next(r for r in summary if r["method"]==m) for m in methods]
            for i,r in enumerate(ordered):
                value=r[f"{metric}_{center}"]
                if low:
                    errors=[[value-r[f"{metric}_{low}"]],[r[f"{metric}_{high}"]-value]]
                else:
                    std=r[f"{metric}_std"]; errors=std if np.isfinite(std) else None
                ax.errorbar(value,i,xerr=errors,fmt="o",capsize=4,color=f"C{i}")
            ax.set_yticks(range(len(methods)),methods); ax.invert_yaxis()
            ax.set_xlabel("Seconds" if low else "Relative Frobenius error")
            if not low:
                ax.axvline(summary[0]["optimal_relative_error"],color="black",linestyle="--",label="Optimal rank-k SVD")
                ax.legend(fontsize=8)
        else:
            for method in methods:
                group=sorted((r for r in summary if r["method"]==method),key=lambda r:r["k"])
                values=np.array([r[f"{metric}_{center}"] for r in group])
                line,=ax.plot(ks,values,marker="o",label=method)
                if low:
                    bottom=[r[f"{metric}_{low}"] for r in group]; top=[r[f"{metric}_{high}"] for r in group]
                else:
                    std=np.array([r[f"{metric}_std"] for r in group]); bottom=values-std; top=values+std
                ax.fill_between(ks,bottom,top,alpha=.12,color=line.get_color())
            if not low:
                baseline=[next(r["optimal_relative_error"] for r in summary if r["k"]==k) for k in ks]
                ax.plot(ks,baseline,"k--",label="Optimal rank-k SVD")
            ax.set(xlabel="Selected columns k",ylabel="Seconds" if low else "Relative Frobenius error")
            ax.legend(fontsize=7)
        ax.set_title(f"E7 — {dataset}\n{title}")
        if low:
            ax.text(.5,-.16,"Shared-basis component sums; descriptive timings, not controlled standalone benchmarks.",
                    transform=ax.transAxes,ha="center",fontsize=8)
        ax.grid(alpha=.2); fig.tight_layout()
        fig.savefig(output/("algorithm_time.png" if low else "accuracy.png"),dpi=160); plt.close(fig)


def run_real_data(A, args, dataset, source, original_indices=None, sample_ids=None, preprocessing=None, output=None):
    if max(args.ks)>min(A.shape):
        raise ValueError(f"Requested k exceeds matrix dimensions {A.shape}")
    output=Path(output) if output is not None else ROOT/"results/e7"/dataset/args.run_name
    if output.exists():
        raise SystemExit(f"Run already exists: {output}; choose a new --run-name")
    if not args.no_plots:
        try:
            import matplotlib  # noqa: F401
        except ImportError:
            raise SystemExit('Install plots: python -m pip install -e ".[experiments]", or pass --no-plots')
    if original_indices is None:
        original_indices=np.arange(A.shape[1])
    if len(original_indices)!=A.shape[1] or (sample_ids is not None and len(sample_ids)!=A.shape[1]):
        raise ValueError("Column metadata does not match matrix width")
    output.mkdir(parents=True)
    config=dict(vars(args),experiment="E7",dataset=dataset,source=str(source),status="running",
                matrix_shape=list(A.shape),matrix_sha256=hashlib.sha256(A.tobytes()).hexdigest(),
                numpy_version=np.__version__,python_version=platform.python_version(),platform=platform.platform(),
                preprocessing=preprocessing,original_column_indices=[int(j) for j in original_indices],sample_ids=sample_ids,
                reconstruction="orthogonal primary; existing runner also computes oblique for comparison",
                trial_policy="seed+t; the existing runner passes the same trial seed to basis and selectors; fixed input across all trials",
                timing_policy="existing pipeline component sums, shared basis cost attributed per method; excludes baseline/evaluation/unused reconstruction; no warm-ups/thread control; not standalone wall time",
                deterministic_policy="Exact-SVD + Osinsky repeats unchanged for descriptive timing; repeated errors are not independent accuracy evidence")
    def save():
        (output/"config.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
    save(); rows=[]; norm=float(np.linalg.norm(A,"fro"))
    runner=run_six_pipelines if args.pipelines==6 else run_four_pipelines
    try:
        with (output/"raw_results.csv").open("x",newline="",encoding="utf-8") as stream:
            writer=None
            for k in args.ks:
                for trial in range(args.trials):
                    seed=args.seed+trial
                    results=runner(A,k,random_state=seed,oversampling=args.p,power_iterations=args.q,reconstruction="orthogonal")
                    for method,result in results.items():
                        J=result["indices"]; metrics=result["metrics"]; timings=result["timings"]
                        optimal=metrics["optimal_rank_k_error"]
                        row=dict(dataset=dataset,k=k,trial=trial,seed=seed,method=method,
                                 indices=json.dumps(J.tolist()),original_indices=json.dumps([int(original_indices[j]) for j in J]),
                                 selected_samples=json.dumps([sample_ids[j] for j in J]) if sample_ids is not None else "",
                                 optimal_rank_k_error=optimal,optimal_relative_error=optimal/norm if norm else 0.,
                                 **{name:metrics[name] for name in ERRORS},**{name:timings[name] for name in TIMINGS})
                        if writer is None:
                            writer=csv.DictWriter(stream,fieldnames=list(row)); writer.writeheader()
                        writer.writerow(row); stream.flush(); rows.append(row)
                        if args.show_indices:
                            print(f"{method}, k={k}, trial={trial}: original columns {row['original_indices']} "
                                  f"{row['selected_samples']}",flush=True)
                    del results
                    print(f"{dataset}: k={k}, trial {trial+1}/{args.trials} complete",flush=True)
        summary=summarize(rows)
        with (output/"summary.csv").open("x",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
        if not args.no_plots: plot_comparison(summary,output/"plots",dataset)
        config["status"]="complete"
    except (Exception,KeyboardInterrupt) as error:
        config.update(status="incomplete",error=f"{type(error).__name__}: {error}"); save(); raise
    save()
    print(f"\n{'k':>4} {'Pipeline':<29} {'Mean rel. error':>16} {'Mean SVD ratio':>16} {'Median alg. ms':>16}")
    for r in summary:
        print(f"{r['k']:>4} {r['method']:<29} {r['relative_frobenius_error_mean']:>16.6g} "
              f"{r['frobenius_optimality_ratio_mean']:>16.6g} {r['algorithm_seconds_median']*1000:>16.3f}")
    print(f"Saved {len(rows)} measurements to {output}")
    return output
