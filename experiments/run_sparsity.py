# E6 — Matched dense/CSR storage with sequential ARP and Rejection ARP.
"""Same matrices, sketch seeds and factorized column-selection workflow.

Defaults: 500 x 500, k=10, densities 1/.5/.1/.01, three matrix instances,
ten trials. This compares four storage/selector combinations, not the original
six SVD pipelines. Input conversion, baseline SVD and error evaluation are untimed.
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
from time import perf_counter

import numpy as np
import scipy
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_rejection_benchmark import construct_basis, interpolation, SELECTORS  # noqa: E402
from run_runtime_scaling import controlled_blas, numpy_config  # noqa: E402

TIMINGS = ("basis_seconds", "selection_seconds", "interpolation_seconds", "algorithm_seconds")
METRICS = (*TIMINGS, "relative_error", "svd_optimality_ratio", "input_bytes", "factor_bytes")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=500)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--densities", nargs="+", type=float, default=[1., .5, .1, .01])
    p.add_argument("--instances", type=int, default=3)
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--trial-seed", type=int, default=0)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ"))
    a = p.parse_args(argv)
    if not 1 <= a.k <= a.size or a.instances < 1 or a.trials < 2:
        p.error("require 1 <= k <= size, instances >= 1, trials >= 2")
    if min(a.data_seed, a.trial_seed) < 0:
        p.error("seeds must be nonnegative")
    if len(set(a.densities)) != len(a.densities) or any(not np.isfinite(d) or not 0 < d <= 1 for d in a.densities):
        p.error("densities must be unique finite fractions in (0,1]")
    if min(round(d*a.size*a.size) for d in a.densities) < a.k:
        p.error("each density must retain at least k entries; this alone does not guarantee rank k")
    if not a.run_name or a.run_name in (".", "..") or Path(a.run_name).name != a.run_name or "\\" in a.run_name:
        p.error("run-name must be a single directory name")
    return a


def matrix_source(size, seed, instance):
    children = np.random.SeedSequence([seed, instance]).spawn(2)
    values = np.random.default_rng(children[0]).standard_normal((size, size))
    order = np.random.default_rng(children[1]).permutation(size*size)
    return values, order


def matrix_at_density(values, order, density):
    A = np.zeros_like(values)
    count = round(density*A.size)
    indices = order[:count]
    A.flat[indices] = values.flat[indices]
    norm = np.linalg.norm(A, "fro")
    if norm == 0:
        raise ValueError("Zero matrix cannot be normalized")
    A /= norm
    csr = sparse.csr_matrix(A)
    csr.eliminate_zeros()
    return A, csr


def storage_bytes(A):
    """Numeric buffer bytes, NOT Python overhead, process RSS, or peak memory."""
    return A.data.nbytes + A.indices.nbytes + A.indptr.nbytes if sparse.issparse(A) else A.nbytes


def factor_error(A, C, X, block_columns=128):
    squared = 0.
    for start in range(0, A.shape[1], block_columns):
        block = A[:, start:start+block_columns]
        if sparse.issparse(block):
            block = block.toarray()
        residual = block - C @ X[:, start:start+block_columns]
        squared += float(np.sum(residual*residual))
    return float(np.sqrt(squared))


def measure_case(A, method, k, basis_seed, selector_seed, optimal, evaluate=True):
    t0 = perf_counter()
    V = construct_basis(A, k, basis_seed)
    t1 = perf_counter()
    J = SELECTORS[method](V, random_state=selector_seed)
    t2 = perf_counter()
    X = interpolation(V, J)
    C = A[:, J]
    t3 = perf_counter()
    # Everything below is outside the algorithm timing, including norm/error work.
    if not np.allclose(X[:, J], np.eye(k), atol=1e-8, rtol=1e-8):
        raise RuntimeError("Selected-column interpolation failed")
    error = factor_error(A, C, X) if evaluate else None
    norm = float(np.linalg.norm(A.data if sparse.issparse(A) else A))
    if evaluate and (not np.isfinite(error) or error + 100*np.finfo(float).eps*max(A.shape)*norm < optimal):
        raise RuntimeError("Invalid error or SVD lower bound violation")
    return dict(basis_seconds=t1-t0, selection_seconds=t2-t1, interpolation_seconds=t3-t2,
                algorithm_seconds=t3-t0, indices=json.dumps(J.tolist()), input_bytes=storage_bytes(A),
                factor_bytes=storage_bytes(C)+X.nbytes, frobenius_error=error,
                relative_error=error/norm if evaluate else None,
                svd_optimality_ratio=error/optimal if evaluate and optimal > 0 else float("nan"))


def trial_rows(A, csr, optimal, args, instance, density):
    count = csr.nnz
    for trial in range(-1, args.trials):
        # Same seeds across formats AND across densities for paired comparisons.
        children = np.random.SeedSequence([args.trial_seed, instance, trial+1]).spawn(4)
        bs, arp_seed, rejection_seed, order_seed = [int(c.generate_state(1)[0]) for c in children]
        combos = [(fmt,method) for fmt in ("dense","csr") for method in SELECTORS]
        for position, index in enumerate(np.random.default_rng(order_seed).permutation(4)):
            fmt, method = combos[index]
            ss = arp_seed if method == "arp" else rejection_seed
            measured = measure_case(A if fmt == "dense" else csr, method, args.k, bs, ss, optimal, evaluate=trial>=0)
            if trial >= 0:
                yield dict(instance=instance, density=density, actual_density=count/A.size, nnz=count,
                           size=args.size, k=args.k, storage=fmt, method=method, trial=trial,
                           basis_seed=bs, selector_seed=ss, order_seed=order_seed, execution_position=position,
                           svd_optimal_error=optimal, **measured)
        print(f"instance {instance+1}, density {density}: {'warm-up' if trial<0 else 'trial '+str(trial+1)} complete", flush=True)


def summaries(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["density"],row["instance"],row["storage"],row["method"]),[]).append(row)
    within, across = [], {}
    for (density,instance,storage,method), group in groups.items():
        for metric in METRICS:
            values = np.asarray([r[metric] for r in group], dtype=float)
            stats = [values.mean(), np.median(values), values.std(ddof=1), *np.percentile(values,[25,75])] if np.isfinite(values).all() else [float("nan")]*5
            row = dict(density=density,instance=instance,storage=storage,method=method,metric=metric,trials=len(group),
                       **dict(zip(("mean","median","std","p25","p75"),map(float,stats))))
            within.append(row)
            across.setdefault((density,storage,method,metric),[]).append(row)
    summary = []
    for (density,storage,method,metric), group in across.items():
        center = "median" if metric in TIMINGS else "mean"
        values = np.asarray([r[center] for r in group])
        summary.append(dict(density=density,storage=storage,method=method,metric=metric,instances=len(group),
                            within_instance_center=center,mean_of_instance_centers=float(values.mean()),
                            between_instance_std=float(values.std(ddof=1)) if len(values)>1 else float("nan")))
    return within, summary


def paired_rows(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r["density"],r["instance"],r["method"],r["trial"]),{})[r["storage"]]=r
    paired = []
    for (density,instance,method,trial), pair in groups.items():
        dense, csr = pair["dense"],pair["csr"]
        delta = abs(dense["relative_error"]-csr["relative_error"])
        paired.append(dict(density=density,instance=instance,method=method,trial=trial,
                           same_ordered_indices=dense["indices"]==csr["indices"],
                           relative_error_absolute_difference=delta,
                           errors_close=bool(np.isclose(dense["relative_error"],csr["relative_error"],rtol=1e-8,atol=1e-10)),
                           dense_over_sparse_time=dense["algorithm_seconds"]/csr["algorithm_seconds"]))
    return paired


def write_csv(path, rows):
    with path.open("x",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def plots(summary, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir()
    for metric in ("relative_error","svd_optimality_ratio","algorithm_seconds","input_bytes"):
        fig,ax=plt.subplots(figsize=(9,5))
        for fmt in ("dense","csr"):
            for method in SELECTORS:
                group=sorted((r for r in summary if (r["storage"],r["method"],r["metric"])==(fmt,method,metric)),key=lambda r:r["density"])
                x=[r["density"]*100 for r in group]; y=[r["mean_of_instance_centers"] for r in group]
                line,=ax.plot(x,y,marker="o",linestyle="--" if fmt=="csr" else "-",label=f"{fmt} + {method}")
                for xi,yi,r in zip(x,y,group):
                    if np.isfinite(r["between_instance_std"]):
                        ax.errorbar(xi,yi,yerr=r["between_instance_std"],fmt="none",ecolor=line.get_color(),capsize=3)
        ax.set(xlabel="Nonzero entries (%)",ylabel=metric.replace("_"," "),title="E6 — matched dense/CSR workflows")
        ax.set_xscale("log"); ax.legend(fontsize=8); ax.grid(alpha=.2)
        ax.text(.5,-.23,"Bars: between-instance SD, not trial variability; input bytes are buffer storage, not peak memory.",transform=ax.transAxes,ha="center",fontsize=8)
        fig.tight_layout(); fig.savefig(directory/f"{metric}.png",dpi=150); plt.close(fig)


def main():
    args=parse_args(); output=ROOT/"results/e6"/args.run_name
    if output.exists(): raise SystemExit(f"Run exists: {output}")
    if not args.no_plots:
        import matplotlib  # noqa: F401
    build=numpy_config()
    with controlled_blas(1,build) as snapshot:
        output.mkdir(parents=True)
        config=dict(vars(args),experiment="E6",status="running",numpy_version=np.__version__,scipy_version=scipy.__version__,
                    platform=platform.platform(),blas=build,threads=snapshot(),matrices=[],
                    generation="SeedSequence([data_seed,instance]): independent Gaussian values and permutation; round(density*size^2) nested positions; unit Frobenius norm",
                    workflow="V=orth(A.T@SparseStack); k sketch columns; C=A[:,J], X=solve(V[J,:].T,V.T); factorized oblique CSSP",
                    timing="Standalone basis+selection+factor construction, including validation; excludes conversion, generation, baseline SVD, error evaluation and I/O",
                    storage="Numeric buffer sizes only; CSR data+indices+indptr, dense nbytes; NOT peak process memory. Both representations coexist for matching.",
                    aggregation="Within-instance means for accuracy/storage or medians for timings, then equally weighted instance centers")
        def save(): (output/"config.json").write_text(json.dumps(config,indent=2)+"\n",encoding="utf-8")
        save(); rows=[]
        try:
            with (output/"raw_results.csv").open("x",newline="",encoding="utf-8") as stream:
                writer=None
                for instance in range(args.instances):
                    values,order=matrix_source(args.size,args.data_seed,instance)
                    for density in args.densities:
                        A,csr=matrix_at_density(values,order,density)
                        s=np.linalg.svd(A,compute_uv=False); optimal=float(np.linalg.norm(s[args.k:]))
                        config["matrices"].append(dict(instance=instance,density=density,actual_density=csr.nnz/A.size,nnz=csr.nnz,
                            zero_rows=int(np.sum(~np.any(A,axis=1))),zero_columns=int(np.sum(~np.any(A,axis=0))),
                            dense_bytes=storage_bytes(A),csr_bytes=storage_bytes(csr),svd_optimal_error=optimal,sha256=hashlib.sha256(A.tobytes()).hexdigest()))
                        save()
                        for row in trial_rows(A,csr,optimal,args,instance,density):
                            if writer is None: writer=csv.DictWriter(stream,fieldnames=list(row)); writer.writeheader()
                            writer.writerow(row); stream.flush(); rows.append(row)
            within,summary=summaries(rows); pairs=paired_rows(rows)
            write_csv(output/"instance_summary.csv",within); write_csv(output/"summary.csv",summary); write_csv(output/"paired_results.csv",pairs)
            if not args.no_plots: plots(summary,output/"plots")
            config.update(status="complete",pairs_with_different_indices=sum(not r["same_ordered_indices"] for r in pairs),
                          pairs_with_error_difference=sum(not r["errors_close"] for r in pairs))
        except (Exception,KeyboardInterrupt) as error:
            config.update(status="incomplete",error=f"{type(error).__name__}: {error}"); save(); raise
        save()
    print(f"Saved {len(rows)} runs and {len(pairs)} dense/sparse pairs to {output}")


if __name__=="__main__":
    main()
