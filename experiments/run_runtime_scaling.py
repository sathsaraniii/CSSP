# E4 — Dense synthetic runtime scaling at fixed selected-column count.
"""Time standalone six-pipeline runs; exclude reference large/sparse examples.

Every call computes its own basis, selects columns, and forms only the
orthogonal approximation. Matrix generation, diagnostics, file I/O, and plotting
are outside timed intervals. Existing component validation remains timed.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import ctypes
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import sys
import warnings
from time import perf_counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cssp.basis.deterministic_svd import deterministic_svd  # noqa: E402
from cssp.basis.randomized_svd import randomized_svd  # noqa: E402
from cssp.column_selection.arp import arp_select  # noqa: E402
from cssp.column_selection.osinsky import osinsky_select  # noqa: E402
from cssp.column_selection.rejection_arp import rejection_arp_select  # noqa: E402
from cssp.cssp_pipeline import SIX_PIPELINE_CONFIGS  # noqa: E402
from cssp.reconstruction.cssp_approximation import orthogonal_cssp_approximation  # noqa: E402

TIMINGS = ("basis_seconds", "selection_seconds", "reconstruction_seconds", "algorithm_seconds")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 250, 500, 1000])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--signal-rank", type=int, default=25)
    parser.add_argument("--relative-noise", type=float, default=0.02,
                        help="Frobenius noise norm / signal norm (not per-entry E2 noise)")
    parser.add_argument("--p", type=int, default=10)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--trial-seed", type=int, default=0)
    parser.add_argument("--run-name", default=datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ"))
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    if min(args.k, args.signal_rank, args.threads) < 1 or args.repeats < 2:
        parser.error("k, signal-rank and threads must be positive; repeats must be >= 2")
    if len(set(args.sizes)) != len(args.sizes) or min(args.sizes) < max(args.k, args.signal_rank):
        parser.error("sizes must be distinct and >= both k and signal-rank")
    if min(args.p, args.q, args.data_seed, args.trial_seed) < 0:
        parser.error("p, q and seeds must be nonnegative")
    if not np.isfinite(args.relative_noise) or args.relative_noise < 0:
        parser.error("relative-noise must be finite and nonnegative")
    if (not args.run_name or args.run_name in (".", "..")
            or Path(args.run_name).name != args.run_name or "\\" in args.run_name):
        parser.error("run-name must be a single directory name")
    args.sizes.sort()
    return args


def seeds(master, size, repetition, phase):
    # Distinct warm-up (phase=0) and measurement (phase=1) streams.
    children = np.random.SeedSequence([master, size, repetition, phase]).spawn(3)
    return [int(child.generate_state(1)[0]) for child in children]


def make_matrix(size, rank, relative_noise, seed):
    rng = np.random.default_rng(np.random.SeedSequence([seed, size]))
    left, _ = np.linalg.qr(rng.standard_normal((size, rank)), mode="reduced")
    right, _ = np.linalg.qr(rng.standard_normal((size, rank)), mode="reduced")
    spectrum = np.geomspace(20.0, 1.0, rank)
    signal = (left * spectrum) @ right.T
    noise = rng.standard_normal((size, size))
    noise *= relative_noise * np.linalg.norm(signal, "fro") / np.linalg.norm(noise, "fro")
    return signal + noise


def measure(A, method, args, basis_seed, selector_seed):
    """Use production helpers. Reconstruction extracts C exactly once internally."""
    basis_kind, selector = SIX_PIPELINE_CONFIGS[method]
    started = perf_counter()
    if basis_kind == "deterministic":
        U, s, Vt = deterministic_svd(A, args.k)
    else:
        U, s, Vt = randomized_svd(A, args.k, p=args.p, q=args.q, random_state=basis_seed)
    V = Vt.T
    basis_done = perf_counter()
    if selector == "osinsky":
        J = osinsky_select(A, V)
    elif selector == "arp":
        J = arp_select(V, random_state=selector_seed)
    else:
        J = rejection_arp_select(V, random_state=selector_seed)
    selection_done = perf_counter()
    approximation = orthogonal_cssp_approximation(A, J)
    reconstruction_done = perf_counter()
    timings = dict(basis_seconds=basis_done - started,
                   selection_seconds=selection_done - basis_done,
                   reconstruction_seconds=reconstruction_done - selection_done)
    timings["algorithm_seconds"] = sum(timings.values())
    # Sanity checks are deliberately after the final timestamp, not full error evaluation.
    if approximation.shape != A.shape or not np.isfinite(approximation).all():
        raise RuntimeError("Invalid orthogonal reconstruction")
    return dict(method=method, basis_seed=basis_seed if basis_kind == "randomized" else None,
                selector_seed=selector_seed if selector != "osinsky" else None,
                indices=json.dumps(J.tolist()), **timings)


def benchmark_rows(A, args):
    size = A.shape[0]
    methods = list(SIX_PIPELINE_CONFIGS)
    basis_seed, selector_seed, order_seed = seeds(args.trial_seed, size, 0, 0)
    for method in np.random.default_rng(order_seed).permutation(methods):
        measure(A, str(method), args, basis_seed, selector_seed)
    print(f"size={size}: warm-up complete", flush=True)
    for repetition in range(args.repeats):
        basis_seed, selector_seed, order_seed = seeds(args.trial_seed, size, repetition, 1)
        order = np.random.default_rng(order_seed).permutation(methods)
        for position, method in enumerate(order):
            measured = measure(A, str(method), args, basis_seed, selector_seed)
            yield dict(size=size, m=size, n=size, k=args.k, repetition=repetition,
                       execution_position=position, order_seed=order_seed, data_seed=args.data_seed,
                       threads=args.threads, signal_rank=args.signal_rank,
                       relative_noise=args.relative_noise, p=args.p, q=args.q,
                       reconstruction="orthogonal", **measured)
        print(f"size={size}: repetition {repetition + 1}/{args.repeats} complete", flush=True)


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["size"], row["method"]), []).append(row)
    results = []
    for (size, method), group in groups.items():
        result = dict(size=size, method=method, repeats=len(group), k=group[0]["k"])
        for name in TIMINGS:
            values = np.array([row[name] for row in group])
            stats = [np.median(values), *np.percentile(values, [25, 75]),
                     values.mean(), values.std(ddof=1) if len(values) > 1 else 0., values.min(), values.max()]
            result.update({f"{name}_{stat}": float(value) for stat, value in
                           zip(("median", "p25", "p75", "mean", "std", "min", "max"), stats)})
        results.append(result)
    return results


def verified_blas(info, threads):
    pools = [pool for pool in info if pool.get("user_api") == "blas"]
    if not pools or any(pool.get("num_threads") != threads for pool in pools):
        raise RuntimeError("Cannot verify requested BLAS thread count. Benchmark stopped; "
                           "use a NumPy BLAS backend supported by threadpoolctl.")
    return pools


def numpy_config():
    # NumPy >=1.26 offers structured configuration without optional PyYAML.
    try:
        return np.show_config(mode="dicts")
    except TypeError:
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            np.show_config()
        return captured.getvalue()


@contextlib.contextmanager
def controlled_blas(threads, build_config):
    """Verify threadpoolctl backends or Apple's per-calling-thread BLAS API."""
    from threadpoolctl import threadpool_info, threadpool_limits
    if sys.platform == "darwin" and "accelerate" in str(build_config).lower():
        if threads != 1:
            raise RuntimeError("Accelerate supports verified single-thread mode here; use --threads 1")
        try:
            lib = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
            setter, getter = lib.BLASSetThreading, lib.BLASGetThreading
        except (OSError, AttributeError) as error:
            raise RuntimeError("Accelerate thread verification requires macOS 15+ BLAS threading API") from error
        setter.argtypes, setter.restype = [ctypes.c_uint], ctypes.c_int
        getter.argtypes, getter.restype = [], ctypes.c_uint
        previous = getter()
        # SDK thread_api.h: MULTI_THREADED=0, SINGLE_THREADED=1.
        if setter(1) != 0:
            raise RuntimeError("Accelerate rejected single-thread mode")
        def snapshot():
            if getter() != 1:
                raise RuntimeError("Accelerate single-thread mode verification failed")
            return [dict(user_api="blas", internal_api="accelerate", num_threads=1,
                         verification="BLASGetThreading", scope="calling thread")]
        try:
            snapshot()
            yield snapshot
        finally:
            setter(previous)
    else:
        with threadpool_limits(limits=threads, user_api="blas"):
            def snapshot():
                return verified_blas(threadpool_info(), threads)
            snapshot()
            yield snapshot


def write_plots(summary, directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir()
    for timing in TIMINGS:
        fig, ax = plt.subplots(figsize=(9, 6))
        for method in SIX_PIPELINE_CONFIGS:
            values = sorted((row for row in summary if row["method"] == method), key=lambda row: row["size"])
            sizes = [row["size"] for row in values]
            line, = ax.plot(sizes, [row[f"{timing}_median"] for row in values], marker="o", label=method)
            ax.fill_between(sizes, [row[f"{timing}_p25"] for row in values],
                            [row[f"{timing}_p75"] for row in values], alpha=0.12, color=line.get_color())
        ax.set(xlabel="Square matrix dimension n (n × n)", ylabel="Seconds",
               title=f"E4 — {timing}: median and 25–75% interval")
        ax.legend(fontsize="small")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(directory / f"{timing.removesuffix('_seconds')}_runtime.png", dpi=160)
        plt.close(fig)


def main():
    args = parse_args()
    output = PROJECT_ROOT / "results/e4" / args.run_name
    if output.exists():
        raise SystemExit(f"Run directory exists: {output}; choose a new --run-name")
    try:
        from threadpoolctl import threadpool_info
        if not args.no_plots:
            import matplotlib  # noqa: F401
    except ImportError:
        raise SystemExit('Install E4 dependencies: python -m pip install -e ".[experiments]"')
    # Force BLAS initialization before discovering and limiting its pools.
    np.ones((2, 2)) @ np.ones((2, 2))
    before = threadpool_info()
    blas_config = numpy_config()
    with controlled_blas(args.threads, blas_config) as thread_snapshot:
        output.mkdir(parents=True, exist_ok=False)
        config = dict(vars(args), experiment="E4", status="running", warmups_per_pipeline_per_size=1,
                      numpy_version=np.__version__, python_version=platform.python_version(),
                      platform=platform.platform(), machine=platform.machine(), processor=platform.processor(),
                      logical_cpu_count=os.cpu_count(), blas_configuration=blas_config,
                      threadpools_before=before, threadpools_limited=thread_snapshot(),
                      data_seed_rule="SeedSequence([data_seed, size])", datasets_metadata={},
                      noise_definition="||noise||_F / ||signal||_F; signal singular values geomspace(20,1,rank)",
                      timing_scope="standalone basis + selection + orthogonal reconstruction, including helper validation",
                      excluded="data generation, warm-ups, sanity checks, baseline SVD, error metrics, oblique reconstruction, I/O",
                      variability="system variation plus seed-dependent algorithm work; no parallel pipeline runs")
        def save_config():
            (output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        save_config()
        rows = []
        try:
            with (output / "raw_results.csv").open("x", newline="", encoding="utf-8") as stream:
                writer = None
                for size in args.sizes:
                    A = make_matrix(size, args.signal_rank, args.relative_noise, args.data_seed)
                    thread_snapshot()
                    config["datasets_metadata"][str(size)] = dict(shape=list(A.shape),
                        sha256=hashlib.sha256(A.tobytes()).hexdigest(), matrix_bytes=A.nbytes)
                    save_config()
                    for row in benchmark_rows(A, args):
                        if writer is None:
                            writer = csv.DictWriter(stream, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                        stream.flush()
                        rows.append(row)
                    del A
            summary = summarize(rows)
            with (output / "summary.csv").open("x", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
                writer.writeheader()
                writer.writerows(summary)
            if not args.no_plots:
                write_plots(summary, output / "plots")
        except (Exception, KeyboardInterrupt) as error:
            config.update(status="incomplete", error=f"{type(error).__name__}: {error}")
            save_config()
            raise
        config["status"] = "complete"
        save_config()
    print(f"Saved {len(rows)} timed runs to {output}")


if __name__ == "__main__":
    main()
