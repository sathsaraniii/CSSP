# E4 rejection — Reference-size dense/sparse ARP timing with factorized output.
"""Sparse-safe column selection on transposes of the reference matrices.

Reference commit: 98bc376c247ab2ef133e1b51fac854983c7c3c8b.
Uses existing Python selectors, a SparseStack sketch and X = solve(V[J,:].T,V.T).
Not a performance-identical MATLAB port. No full Ahat or exact SVD is formed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy import sparse, linalg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_runtime_scaling import controlled_blas, numpy_config  # noqa: E402
from cssp.column_selection.arp import arp_select  # noqa: E402
from cssp.column_selection.rejection_arp import rejection_arp_select  # noqa: E402

COMMIT = "98bc376c247ab2ef133e1b51fac854983c7c3c8b"
PROFILES = {"pilot": {"dense": (200, 200), "sparse": (2000, 200)},
            "intermediate": {"dense": (1000, 1000), "sparse": (100000, 1000)},
            "reference": {"dense": (10000, 10000), "sparse": (1000000, 10000)}}
SELECTORS = {"arp": arp_select, "rejection_arp": rejection_arp_select}


def sparse_gaussian(m, n, per_column, seed):
    rng = np.random.default_rng(seed)
    rows = np.concatenate([rng.choice(m, per_column, replace=False) for _ in range(n)])
    values = rng.standard_normal(n * per_column) / (rows.astype(float) + 1) ** 2
    result = sparse.csc_matrix((values, rows, np.arange(n + 1) * per_column), shape=(m, n))
    result.sort_indices()
    return result


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def get_dataset(kind, shape, seed, directory):
    """Generate once with completion metadata; do not overwrite partial datasets."""
    m, n = shape
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{kind}_{m}x{n}_seed{seed}{'.npz' if kind == 'sparse' else '.npy'}"
    meta_path = path.with_suffix(".json")
    if path.exists() or meta_path.exists():
        if not path.exists() or not meta_path.exists():
            raise RuntimeError(f"Partial dataset found: {path}; choose a new seed or inspect it manually")
        meta = json.loads(meta_path.read_text())
        if meta.get("status") != "complete" or meta["sha256"] != file_hash(path):
            raise RuntimeError(f"Dataset incomplete or fingerprint mismatch: {path}")
    else:
        required = m * n * 8 if kind == "dense" else n * 30 * 24
        if shutil.disk_usage(directory).free < required + 256 * 1024**2:
            raise RuntimeError("Insufficient disk space for dataset and safety margin")
        meta = dict(kind=kind, shape=[m, n], seed=seed, dtype="float64", status="generating",
                    numpy_version=np.__version__, reference_commit=COMMIT,
                    row_scaling="1 / (zero_based_row + 1)^2", nonzeros_per_column=30 if kind == "sparse" else None)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        if kind == "sparse":
            sparse.save_npz(path, sparse_gaussian(m, n, 30, seed))
        else:
            rng = np.random.default_rng(seed)
            A = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=shape)
            for start in range(0, m, 128):
                end = min(m, start + 128)
                block = rng.standard_normal((end - start, n))
                block /= np.arange(start + 1, end + 1, dtype=float)[:, None] ** 2
                A[start:end] = block
            A.flush()
            del A
        meta.update(status="complete", sha256=file_hash(path))
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    A = sparse.load_npz(path) if kind == "sparse" else np.load(path, mmap_mode="r")
    # Preserve the stored reference orientation, but expose explicit CSSP input.
    # Dense transpose is a memmap view; sparse transpose stays sparse.
    return A.T, dict(meta, path=str(path), reference_shape=[m, n], cssp_shape=[n, m])


def sparse_stack(n, k, seed, zeta=4):
    """n x k embedding: one signed entry in each of min(4,k) disjoint blocks."""
    rng = np.random.default_rng(seed)
    zeta = min(zeta, k)
    lengths = np.full(zeta, k // zeta)
    lengths[:k % zeta] += 1
    offsets = np.r_[0, np.cumsum(lengths)[:-1]]
    columns = np.column_stack([offset + rng.integers(length, size=n) for offset, length in zip(offsets, lengths)])
    values = rng.choice([-1.0, 1.0], size=n * zeta) / np.sqrt(zeta)
    return sparse.csr_matrix((values, columns.ravel(), np.arange(n + 1) * zeta), shape=(n, k))


def construct_basis(A, k, seed):
    """Right basis V = orth(A.T @ Omega), with exactly k sketch columns."""
    embedding = sparse_stack(A.shape[0], k, seed)
    if sparse.issparse(A):
        # Only the narrow n x k sketch is densified, never the m x n input.
        sketch = (A.T @ embedding).toarray()
    else:
        sketch = np.empty((A.shape[1], k))
        for start in range(0, A.shape[1], 1024):
            block = A[:, start:start + 1024]
            sketch[start:start + block.shape[1]] = (embedding.T @ block).T
    Q, R = np.linalg.qr(sketch, mode="reduced")
    # k x k diagnostic, not an SVD of the input. Stop rather than silently retry.
    if np.linalg.matrix_rank(R) < k:
        raise RuntimeError("Rank-deficient sketch; record failed trial and choose another seed explicitly")
    return Q


def interpolation(V, J):
    """Return k x n X for C @ X, where C=A[:,J]; never form the full product."""
    factors = linalg.lu_factor(V[J].T)
    X = np.empty((len(J), len(V)))
    for start in range(0, len(V), 8192):
        X[:, start:start + 8192] = linalg.lu_solve(factors, V[start:start + 8192].T)
    return X


def estimate_bytes(kind, m, n, k):
    # Conservative planning estimate, not an allocator-enforced memory cap.
    matrix = m * n * 8 if kind == "dense" else n * 30 * 24
    return matrix + 16 * m * k * 8 + 256 * 1024**2


def timed_run(A, method, k, basis_seed, selection_seed):
    t0 = perf_counter()
    Q = construct_basis(A, k, basis_seed)
    t1 = perf_counter()
    J = SELECTORS[method](Q, random_state=selection_seed)
    t2 = perf_counter()
    X = interpolation(Q, J)
    selected = A[:, J]
    t3 = perf_counter()
    # Bounded diagnostics outside the measured intervals.
    if not np.allclose(X[:, J], np.eye(k), atol=1e-8, rtol=1e-8):
        raise RuntimeError("Selected-column interpolation check failed")
    return dict(basis_seconds=t1-t0, selection_seconds=t2-t1,
                interpolation_seconds=t3-t2, algorithm_seconds=t3-t0,
                indices=json.dumps(J.tolist()), factor_shape=json.dumps(list(X.shape)))


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["dataset"], row["k"], row["mode"], row["method"]), []).append(row)
    result = []
    for (dataset, k, mode, method), group in groups.items():
        row = dict(dataset=dataset, k=k, mode=mode, method=method, repeats=len(group))
        for metric in ("basis_seconds", "selection_seconds", "interpolation_seconds", "algorithm_seconds"):
            values = np.array([r[metric] for r in group])
            for stat, value in zip(("median", "p25", "p75"), (np.median(values), *np.percentile(values, [25, 75]))):
                row[f"{metric}_{stat}"] = float(value)
        result.append(row)
    return result


def plot_results(summary, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir()
    for dataset in {row["dataset"] for row in summary}:
        for mode in ("selection_only", "end_to_end"):
            fig, ax = plt.subplots(figsize=(8, 5))
            for method in SELECTORS:
                group = sorted((r for r in summary if (r["dataset"], r["mode"], r["method"]) == (dataset, mode, method)), key=lambda r: r["k"])
                ks = [r["k"] for r in group]
                ax.plot(ks, [r["algorithm_seconds_median"] for r in group], marker="o", label=method)
                ax.fill_between(ks, [r["algorithm_seconds_p25"] for r in group], [r["algorithm_seconds_p75"] for r in group], alpha=.15)
            ax.set(xlabel="Selected columns k", ylabel="Seconds",
                   title=f"E4 rejection — {dataset}, {mode}: median and IQR")
            ax.legend()
            fig.tight_layout()
            fig.savefig(directory / f"{dataset}_{mode}.png", dpi=150)
            plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=PROFILES, default="pilot")
    p.add_argument("--datasets", nargs="+", choices=("dense", "sparse"), default=["dense", "sparse"])
    p.add_argument("--ks", nargs="+", type=int, default=[10])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--trial-seed", type=int, default=0)
    p.add_argument("--memory-gib", type=float, default=4.)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ"))
    args = p.parse_args(argv)
    if args.repeats < 2 or min(args.ks) < 1 or min(args.data_seed, args.trial_seed) < 0:
        p.error("repeats >=2, positive ks, and nonnegative seeds required")
    if len(set(args.ks)) != len(args.ks) or len(set(args.datasets)) != len(args.datasets):
        p.error("duplicate k or dataset")
    if not np.isfinite(args.memory_gib) or args.memory_gib <= 0:
        p.error("memory-gib must be finite and positive")
    if not args.run_name or args.run_name in (".", "..") or Path(args.run_name).name != args.run_name or "\\" in args.run_name:
        p.error("run-name must be one directory name")
    args.ks.sort()
    return args


def main():
    args = parse_args()
    output = ROOT / "results/e4/rejection" / args.run_name
    if output.exists():
        raise SystemExit(f"Run exists: {output}")
    if not args.no_plots:
        import matplotlib  # noqa: F401
    output.mkdir(parents=True)
    config = dict(vars(args), experiment="E4 rejection", status="running", reference_commit=COMMIT,
                  numpy_version=np.__version__, scipy_version=scipy.__version__, platform=platform.platform(),
                  orientation="column selection on A = reference_matrix.T; stored files retain reference orientation",
                  interpolation="C=A[:,J], X=solve(V[J,:].T,V.T), Ahat=C@X; factorized oblique, not orthogonal",
                  memory_note="Planning estimate, not measured peak or allocator-enforced cap",
                  basis="V=orth(A.T@SparseStack), zeta=min(4,k), exactly k columns, no power iterations",
                  limitations="existing Python selectors and their full-basis validation; not identical MATLAB timings",
                  warmups=1, datasets_metadata={}, skipped=[])
    def save():
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save()
    rows = []
    try:
        with controlled_blas(1, numpy_config()) as snapshot, (output / "raw_results.csv").open("x", newline="") as stream:
            config["thread_control"] = snapshot()
            writer = None
            for dataset in args.datasets:
                m, n = PROFILES[args.profile][dataset]
                valid = []
                for k in args.ks:
                    estimate = estimate_bytes(dataset, m, n, k)
                    if k > min(m, n) or estimate > args.memory_gib * 1024**3:
                        config["skipped"].append(dict(dataset=dataset, k=k, estimated_bytes=estimate, reason="dimension or memory budget"))
                    else:
                        valid.append(k)
                save()
                if not valid:
                    continue
                print(f"Preparing {dataset} {m} x {n}", flush=True)
                A, meta = get_dataset(dataset, (m, n), args.data_seed, ROOT / "data/rejection_arp/generated")
                config["datasets_metadata"][dataset] = meta
                save()
                for k in valid:
                    print(f"{dataset} k={k}: estimated peak {estimate_bytes(dataset,m,n,k)/1024**3:.2f} GiB", flush=True)
                    # -1 warm-up uses distinct seed stream and is never saved.
                    for trial in range(-1, args.repeats):
                        children = np.random.SeedSequence([args.trial_seed, int(dataset == "sparse"), k, trial+1]).spawn(3)
                        bs, ss, order = [int(c.generate_state(1)[0]) for c in children]
                        methods = np.random.default_rng(order).permutation(list(SELECTORS))
                        for mode in ("selection_only", "end_to_end"):
                            shared = construct_basis(A, k, bs) if mode == "selection_only" else None
                            for position, method in enumerate(methods):
                                if shared is not None:
                                    start = perf_counter()
                                    J = SELECTORS[method](shared, random_state=ss)
                                    elapsed = perf_counter() - start
                                    measured = dict(basis_seconds=0., selection_seconds=elapsed, interpolation_seconds=0.,
                                                    algorithm_seconds=elapsed, indices=json.dumps(J.tolist()), factor_shape="")
                                else:
                                    measured = timed_run(A, method, k, bs, ss)
                                if trial >= 0:
                                    row = dict(dataset=dataset, m=A.shape[0], n=A.shape[1],
                                               reference_m=m, reference_n=n, k=k, mode=mode, method=str(method),
                                               trial=trial, basis_seed=bs, selector_seed=ss, order_seed=order,
                                               execution_position=position, **measured)
                                    if writer is None:
                                        writer = csv.DictWriter(stream, fieldnames=list(row))
                                        writer.writeheader()
                                    writer.writerow(row)
                                    stream.flush()
                                    rows.append(row)
                            del shared
                        print(f"{dataset} k={k}: {'warm-up' if trial < 0 else 'trial '+str(trial+1)} complete", flush=True)
                del A
        summary = summarize(rows)
        if summary:
            with (output / "summary.csv").open("x", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
            if not args.no_plots:
                plot_results(summary, output / "plots")
        config["status"] = "complete_with_skips" if config["skipped"] else "complete"
    except (Exception, KeyboardInterrupt) as error:
        config.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        save()
        raise
    save()
    print(f"Saved {len(rows)} measurements to {output}")


if __name__ == "__main__":
    main()
