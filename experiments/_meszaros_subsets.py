"""E7 multi-subset reporting; no rejection of ties or adaptive seed selection."""
import argparse
import csv
import itertools
import json

import numpy as np

from _real_data_experiment import run_real_data, ERRORS, TIMINGS


def write_csv(path, rows):
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(summaries, raw):
    groups = {}
    for row in summaries:
        groups.setdefault((int(row["k"]), row["method"]), []).append(row)
    overall = []
    for (k, method), rows in groups.items():
        for metric in (*ERRORS, *TIMINGS):
            center = "median" if metric in TIMINGS else "mean"
            values = np.array([float(r[f"{metric}_{center}"]) for r in rows])
            overall.append(dict(k=k, method=method, metric=metric, subsets=len(rows),
                                within_subset_center=center,
                                mean_of_subset_centers=float(values.mean()),
                                between_subset_std=float(values.std(ddof=1)) if len(values)>1 else float("nan")))
    trials = {}
    for row in raw:
        trials.setdefault((row["subset_seed"], int(row["k"]), int(row["trial"])), {})[row["method"]] = frozenset(json.loads(row["original_indices"]))
    agreement = []
    for (seed, k, trial), methods in trials.items():
        for a, b in itertools.combinations(sorted(methods), 2):
            agreement.append(dict(subset_seed=seed, k=k, trial=trial, method_a=a, method_b=b,
                                  same_column_set=methods[a] == methods[b]))
    paired = {}
    for row in agreement:
        paired.setdefault((row["k"], row["method_a"], row["method_b"]), []).append(row["same_column_set"])
    agreement_summary = [dict(k=k, method_a=a, method_b=b, comparisons=len(v),
                              same_set_count=sum(v), same_set_fraction=float(np.mean(v)))
                         for (k,a,b),v in paired.items()]
    return overall, agreement, agreement_summary


def run_subsets(matrix, args, root, source, sample):
    output = root / "results/e7/meszaros" / args.run_name
    if output.exists():
        raise SystemExit(f"Run exists: {output}; choose a new --run-name")
    output.mkdir(parents=True)
    config = dict(vars(args), experiment="E7 multi-subset", status="running", completed_subsets=[],
                  policy="All predeclared seeds retained, including identical selections/errors. "
                         "Equal subset weights: trial means for errors, trial medians for timings. "
                         "Between-subset SD is separate from within-subset trial SD. "
                         "Subsets may overlap; these are not independent source datasets.",
                  timing="Descriptive shared-basis component sums; not controlled standalone timings.")
    def save():
        (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    save()
    raw, summaries = [], []
    try:
        for seed in args.data_seeds:
            child_args = argparse.Namespace(**vars(args))
            child_args.data_seed = seed
            child_args.data_seeds = None
            A, mapping = sample(matrix, number_of_columns=args.columns, random_state=seed)
            print(f"Subset seed={seed}: {A.shape}", flush=True)
            child = run_real_data(A, child_args, "meszaros", source, original_indices=mapping,
                preprocessing=dict(original_shape=list(matrix.shape), original_nnz=int(matrix.nnz),
                                   data_seed=seed, requested_columns=args.columns,
                                   operations="sample nonzero columns; remove zero rows; dense; no normalization"),
                output=output / "subsets" / f"seed_{seed}")
            for filename, target in (("raw_results.csv", raw), ("summary.csv", summaries)):
                with (child / filename).open(newline="") as stream:
                    target.extend(dict(subset_seed=seed, **r) for r in csv.DictReader(stream))
            config["completed_subsets"].append(seed)
            save()
        overall, agreement, agreement_summary = aggregate(summaries, raw)
        for name, rows in (("raw_results",raw), ("subset_summary",summaries), ("summary",overall),
                           ("selection_agreement",agreement), ("selection_agreement_summary",agreement_summary)):
            write_csv(output / f"{name}.csv", rows)
        config["status"] = "complete"
    except (Exception, KeyboardInterrupt, SystemExit) as error:
        config.update(status="incomplete", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        save()
    print(f"Saved {len(raw)} measurements across {len(args.data_seeds)} subsets to {output}")
    print("Per-subset plots are in subsets/seed_*/plots; aggregate statistics are in summary.csv.")
    for row in overall:
        if row["metric"] == "relative_frobenius_error":
            print(f"k={row['k']} {row['method']}: mean subset error={row['mean_of_subset_centers']:.8g}, "
                  f"between-subset SD={row['between_subset_std']:.4g}")
    return output
