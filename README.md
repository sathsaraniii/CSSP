
# CS4523 Project: A Comparative Study of Deterministic and Randomized Approaches to Column Subset Selection for Low-Rank Matrix Approximation

This repository studies the Column Subset Selection Problem (CSSP): given a
real matrix `A` of shape `m × n`, choose `k` actual columns that represent the
whole matrix well. It implements **six pipelines**, combining two ways to
construct a right singular-vector basis with three column-selection methods.

## Pipeline overview

```text
                         Input A (m × n), target k
                                     │
                  ┌──────────────────┴──────────────────┐
                  ▼                                     ▼
        Deterministic SVD                       Randomized SVD
        Right basis V_d (n × k)                 Right basis V_r (n × k)
                  │                                     │
         ┌────────┼─────────┐                  ┌────────┼─────────┐
         ▼        ▼         ▼                  ▼        ▼         ▼
      Osinsky    ARP    Rejection ARP        Osinsky    ARP    Rejection ARP
        (1)      (2)       (5)                 (3)      (4)       (6)
         │        │         │                  │        │         │
         └────────┴─────────┴──────────────────┴────────┴─────────┘
                                     │
                      For EACH pipeline independently:
                                     ▼
                       k indices J → C = A[:, J]
                                     ▼
                 Orthogonal and oblique reconstructions
                    (orthogonal is the primary default)
                                     ▼
                     Errors, timings, and diagnostics
```

The branches are separate alternatives, not consecutive selection steps.
Osinsky uses both `A` and its basis; ARP and Rejection ARP select from the basis.

| # | Result key | Basis | Selector |
|---|---|---|---|
| 1 | `deterministic_osinsky` | Deterministic SVD | Osinsky (deterministic) |
| 2 | `deterministic_arp` | Deterministic SVD | ARP (randomized) |
| 3 | `randomized_osinsky` | Randomized SVD | Osinsky (deterministic given the basis) |
| 4 | `randomized_arp` | Randomized SVD | ARP (randomized) |
| 5 | `deterministic_rejection_arp` | Deterministic SVD | Rejection ARP (randomized) |
| 6 | `randomized_rejection_arp` | Randomized SVD | Rejection ARP (randomized) |

`run_six_pipelines` shares one deterministic basis across its three selectors
and one randomized basis across its three selectors. `run_four_pipelines`
retains the original combinations (1–4). Experiment scripts still default
to four for compatibility; pass `--pipelines 6` for the complete comparison.
Randomized SVD makes the entire randomized-SVD + Osinsky pipeline randomized,
even though Osinsky itself is deterministic for fixed inputs.

## Setup

The project is packaged with `pyproject.toml` and uses a `src/` layout.
From a fresh clone on Windows, run:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

The editable installation makes `cssp` importable while allowing code changes
to take effect immediately. To include support for loading the paper's MATLAB
dataset files, install both optional dependency groups:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,datasets]"
```

On macOS or Linux:

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -e ".[dev,datasets]"
```

For the commands below, replace `.\.venv\Scripts\python.exe` with
`./.venv/bin/python` and use forward slashes in script paths.

## Run experiments

From the repository root:

```powershell
.\.venv\Scripts\python.exe experiments\run_experiment.py --pipelines 6
```

Example with custom parameters:

```powershell
.\.venv\Scripts\python.exe experiments\run_experiment.py --pipelines 6 --m 500 --n 300 --k 20 --p 10 --q 1 --seed 42
```

Run the original four-pipeline comparison with:

```powershell
.\.venv\Scripts\python.exe experiments\run_experiment.py --pipelines 4
```

The script reports relative Frobenius reconstruction error, error relative
to the optimal rank-`k` truncated SVD, row-space time, selector time, and
selected zero-based column indices.

Reconstruction timing counts only the requested method in `algorithm_seconds`.
`reconstruction_seconds` is column extraction plus that method's time; separate
`orthogonal_reconstruction_seconds` and `oblique_reconstruction_seconds` record
the two reconstruction calls. Both approximations remain available for comparison.
The existing `total_seconds` still includes both reconstructions and evaluation,
but excludes the optimal-error baseline; it is not an end-to-end wall-clock time.

## E2: accuracy versus selected-column count

`experiments/run_accuracy_vs_k.py` compares all six combinations on a fixed
small synthetic matrix and/or the full DNA matrix. It evaluates only orthogonal
reconstruction, using one exact SVD per dataset for both the exact basis and
the optimal-error baseline. Randomized bases are shared across selectors per
trial; basis and selector seed streams are separate. Exact-SVD + Osinsky runs
once per `k`; each other combination runs 30 trials by default.

Install optional plotting support (commands below use an activated environment):

```bash
python -m pip install -e ".[dev,datasets,experiments]"

# Small workflow pilot, not the final statistical experiment
python experiments/run_accuracy_vs_k.py --datasets synthetic --ks 5 10 --trials 3 --run-name synthetic_pilot

# Full synthetic sweep (160 x 120, k=5,10,20,30,40; 30 trials)
python experiments/run_accuracy_vs_k.py --datasets synthetic --run-name synthetic_30_trials

# Full DNA sweep with the same k values and trial count (more expensive)
python experiments/run_accuracy_vs_k.py --datasets dna --run-name dna_30_trials
```

Omitting `--datasets` runs both. Add `--no-plots` for CSV-only output without
Matplotlib. Synthetic generation uses a fixed data seed (42), signal rank 25,
and noise 0.02; randomized SVD uses `--p 10 --q 1`. `--data-seed` controls only
synthetic generation; `--trial-seed` controls independent algorithm seed streams.

Each new run saves to `results/e2/<run-name>/`:

- `config.json`: settings, dataset shapes/hashes, versions, and completion status.
- `raw_results.csv`: every selection, seed, error, and DNA sample identifiers.
- `summary.csv`: mean, sample standard deviation, median, minimum, and maximum.
- `plots/`: relative-error and SVD-optimality-ratio curves, with ±1 standard
  deviation bands for randomized methods (unless `--no-plots`).

Existing run folders are never overwritten. Raw rows are flushed as they are
computed; an interrupted run retains partial data and is not marked complete.
Ratios with zero SVD baseline are `NaN`. The baseline is not the exhaustive CSSP
optimum used in E1. This runner measures accuracy, not runtime performance.

## Python API

After the editable installation above, run all six pipelines with:

```python
import numpy as np
from cssp import run_six_pipelines

rng = np.random.default_rng(42)
A = rng.standard_normal((80, 60))

results = run_six_pipelines(
    A,
    k=10,
    random_state=42,
    oversampling=10,
    power_iterations=1,
)

for name, result in results.items():
    print(name, result["indices"], result["relative_error"])
```

One combination can be run separately:

```python
from cssp import run_cssp_pipeline

result = run_cssp_pipeline(
    A,
    k=10,
    row_space_method="randomized",
    column_selection_method="rejection_arp",
    reconstruction="orthogonal",
    random_state=42,
    oversampling=10,
    power_iterations=1,
)
```

Named wrappers are also available: `deterministic_svd_osinsky`,
`deterministic_svd_arp`, `deterministic_svd_rejection_arp`,
`randomized_svd_osinsky`, `randomized_svd_arp`, and
`randomized_svd_rejection_arp`. The original `run_four_pipelines` API is unchanged.

The Rejection ARP result names are `deterministic_rejection_arp` and
`randomized_rejection_arp`. Their `selection_details` include leverage scores,
number of proposals, number accepted per block, and observed acceptance rate.
Rejection ARP samples the same volume-sampling distribution as sequential ARP;
its purpose is to reduce selection work for large candidate sets.

The Meszaros/large experiment also keeps four pipelines as its default and can
opt into all six with:

```powershell
.\.venv\Scripts\python.exe experiments\run_large_experiment.py --pipelines 6
```

Run all six pipelines on the GSE10072 DNA/gene-expression data with:

```powershell
.\.venv\Scripts\python.exe experiments\run_dna_experiment.py --pipelines 6
```

Each result contains both reconstructions:

- `orthogonal_approximation`: `C @ pinv(C) @ A`, used in the paper's CSSP
  experiments and selected as the default result.
- `oblique_approximation`: `A[:, J] @ V[J, :]^{-T} @ V.T`, controlled by the
  ARP expectation theorem.

## Evaluation metrics

The [reference MATLAB CSSP experiments](https://github.com/Alice94/ARP/blob/main/tests/test_CSS_DNA.m)
plot reconstruction error and the optimal SVD error separately, each normalized
by `||A||_F`. For nonzero `A`, `relative_frobenius_error` uses this normalization;
`optimal_rank_k_error` stores the unnormalized singular-value-tail error.

`frobenius_optimality_ratio` is an additional repository diagnostic:
reconstruction error divided by the optimal error, when the latter is positive.
When the optimal error is exactly zero, the ratio is undefined and is returned
as `NaN` (shown as `nan` in the experiment tables), for both exact and inexact
reconstructions. Inspect the absolute and relative errors instead. This explicit
undefined-ratio policy is not a rule from the paper and does not use a tolerance
to label an approximation optimal. Positive baselines, however small, are not
treated as zero. Supplied baselines must be finite and nonnegative.

## Input validation and reference checks

The numerical API accepts finite, nonempty, real two-dimensional matrices.
Ranks, oversampling, iteration counts, block sizes, and column indices must be
integers in their documented ranges; Boolean and fractional values are rejected.
Indices are checked before conversion, so invalid values cannot silently select
different columns. Complex inputs are rejected rather than losing imaginary parts.

Osinsky selection works on `A / max(abs(A))` (with a separate all-zero case) to
avoid overflow/underflow caused by a common input scale. Its diagnostic scores
are squared ratios in these normalized units. Reconstruction still uses the
original matrix. This does not guarantee arbitrary extreme-scale or
ill-conditioned reconstruction and error calculations are free of roundoff.

Tests compare ARP probabilities with direct residual projection, Osinsky pivots
with a Python translation of the MATLAB criterion, randomized SVD with an
independent powered-range formulation, and reconstructions with projection and
interpolation identities. A seeded small-matrix distribution test checks ARP and
Rejection ARP against determinant-based subset probabilities.

These checks support agreement with the reference algorithms, not a full
reproduction of the published experiments or a proof of correctness. MATLAB
itself has not been run. The HMT randomized-SVD combinations are project
extensions, not the ARP paper's specific oversampled sketch construction.
Timing, diagnostic overhead, and single-trial experiments also differ from the
research benchmarks; no matching performance claim is made.

## Repository structure

| Path | Purpose |
|---|---|
| `src/cssp/cssp_pipeline.py` | Individual, four-pipeline, and six-pipeline orchestration; results and timing. |
| `src/cssp/_validation.py` | Shared matrix and integer-parameter validation. |
| `src/cssp/basis/` | Deterministic and randomized SVD basis construction. |
| `src/cssp/column_selection/` | Osinsky, ARP, and Rejection ARP selectors. |
| `src/cssp/reconstruction/` | Column extraction and orthogonal/oblique approximations. |
| `src/cssp/metrics/` | Approximation errors and exact optimal rank-`k` baseline. |
| `experiments/run_experiment.py` | Seeded synthetic-data comparison. |
| `experiments/run_dna_experiment.py` | GSE10072 gene-expression comparison. |
| `experiments/run_large_experiment.py` | Comparison on a reproducible subset of the sparse Meszaros/large matrix. |
| `data/` | Bundled datasets, provenance, stored SVD data, and MATLAB smile/spiral kernel generators. |
| `tests/` | Automated numerical, reference, validation, loader, and timing tests. |
| `pyproject.toml` / `setup.py` | Package configuration and compatibility entry point. |
| `report-source.md` | Research-source mapping, validation evidence, and limitations. |

The MATLAB generators are not invoked by the Python experiments.
`src/cssp/basis/test_svd.py` is a manual SVD comparison demo, not part of the
default automated suite, which is collected from `tests/`.

## Tests

Install the dependencies and run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q -W error
```

| Test file | What it checks |
|---|---|
| `test_pipelines.py` | Basis shapes/orthonormality, reproducible selectors, nonsingular selections, and four/six-pipeline integration. |
| `test_optimal_error.py` | Singular-value-tail baseline, including nearly low-rank and full-rank cases. |
| `test_evaluation.py` | Error metrics, tiny positive baselines, zero-baseline `NaN`, and invalid baselines. |
| `test_osinsky_scaling.py` | MATLAB-criterion reference pivots, scale invariance, and original-unit outputs. |
| `test_reference_algorithms.py` | Independent ARP/RSVD formulations, determinant-based sampling frequencies, and reconstruction identities. |
| `test_validation.py` | Invalid matrices, ranks, parameters, bases, and indices across public functions. |
| `test_dataset_loaders.py` | GEO parsing and sparse preprocessing using small controlled fixtures. |
| `test_reconstruction_timing.py` | Requested-method timing and exclusion of unused reconstruction/baseline work from algorithm time. |

The suite uses seeded synthetic matrices, analytically known diagonal matrices,
and small loader fixtures. Timing tests use a controlled clock, not speed
benchmarks. Dataset-loader tests do not reproduce full real-data experiments.
On 21 September 2026, **515 parameterized test cases passed** with warnings
treated as errors. This is bounded regression evidence, not proof of correctness
for every input or reproduction of the papers' performance results.
