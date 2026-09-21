
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

## E3: randomization variability at fixed k

`experiments/run_randomization_variability.py` holds each matrix and its exact
SVD basis fixed. It runs Osinsky once and ARP/Rejection ARP 100 times each by
default, at `k=10`. Only selection is randomized; this experiment does not use
randomized SVD or benchmark runtime. Synthetic data defaults match E2, and the
DNA option uses the full matrix.

```bash
# Quick workflow check
python experiments/run_randomization_variability.py --datasets synthetic --trials 3 --run-name synthetic_pilot

# Full 100-trial studies
python experiments/run_randomization_variability.py --datasets synthetic --run-name synthetic_100_trials
python experiments/run_randomization_variability.py --datasets dna --run-name dna_100_trials
```

Omit `--datasets` to run both; change `--k` for another fixed column budget.
`--data-seed` controls synthetic generation and `--trial-seed` derives independent
selector seed streams. Plotting uses the same optional `experiments` dependency
as E2; `--no-plots` writes CSVs without Matplotlib.

Results are saved under `results/e3/<run-name>/`: `config.json`, per-run
`raw_results.csv`, `summary.csv`, and `plots/<dataset>_variability.png`.
The summary includes mean, sample standard deviation, median, empirical 5th/95th
percentiles, min/max, and fractions better/tied/worse than Osinsky. Comparisons
use an absolute error tolerance of `100 * eps * max(m,n) * ||A||_F`, saved with
the results; this tolerance does not redefine a zero SVD baseline. Zero-baseline
ratios remain undefined (`NaN`).

Boxplots show randomized relative errors with a fixed Osinsky reference line;
whiskers mark the empirical 5th/95th percentiles, not confidence intervals.
Best observed errors are not single-run guarantees. Existing folders are not
overwritten; raw rows are flushed during execution and caught failures are
marked incomplete in the configuration.

## E4: dense synthetic runtime scaling

`experiments/run_runtime_scaling.py` measures standalone runs of all six
pipelines on square matrices of sizes 100, 250, 500, and 1000 by default.
It keeps `k=10`, signal rank 25, and randomized SVD `p=10, q=1` fixed. The signal
singular values range geometrically from 20 to 1. Noise is normalized so
`||noise||_F / ||signal||_F = 0.02` across sizes; this is different from E2's
per-entry noise setting. One fixed matrix per size is shared across all runs.

```bash
python -m pip install -e ".[experiments]"

# Pilot: two sizes, three measured repetitions per pipeline
python experiments/run_runtime_scaling.py --sizes 100 250 --repeats 3 --run-name runtime_pilot

# Full initial study: four sizes, ten measured repetitions per pipeline
python experiments/run_runtime_scaling.py --run-name runtime_full
```

Every pipeline computes its own basis and only its orthogonal reconstruction.
One warm-up per method/size is discarded. Measured repetitions shuffle method
order and vary recorded seeds. Timings include existing helper validation and
column extraction (once, inside reconstruction), but exclude generation, error
evaluation, optimal-error baseline work, oblique reconstruction, and I/O.
The algorithm time is the sum of the three measured stage times. Variation
includes both system noise and seed-dependent algorithm work. Summaries report
median, 25th/75th percentiles, mean, sample standard deviation, and min/max;
the median total need not equal the sum of the stage medians.

Execution is sequential with one verified BLAS thread by default. `threadpoolctl`
controls supported BLAS libraries; Apple Accelerate uses its macOS 15+ native
[BLAS threading API](https://developer.apple.com/documentation/accelerate/blassetthreading(_:))
and verifies the setting with `BLASGetThreading`. On Accelerate this runner only
supports `--threads 1`. Unsupported/unverifiable configurations stop rather than
claiming controlled timings. Close other heavy applications during measurement.

Output is `results/e4/<run-name>/`, containing configuration (including platform,
BLAS/thread settings and matrix hashes), raw timings, a summary, and four plots
for algorithm, basis, selection, and reconstruction runtimes. Use `--no-plots`
for CSV-only output. Existing run directories are protected; partial results
are retained on interruption. Optional larger sizes can be supplied via
`--sizes`, subject to available memory and runtime. There is no automatic
resource-limit detection.

The 10,000-square dense and million-row sparse Rejection ARP reference examples
are **not** part of this runner. This is a local dense scaling study, not a
reproduction of the papers' performance results.

## E4 rejection: large reference-family column-selection benchmark

This is separate from the completed six-pipeline size sweep above. It compares
the existing ARP and Rejection ARP selectors with a sparse-sign sketch basis and
**factorized oblique reconstruction**, not the orthogonal reconstruction of E4's
original runner. No original selectors or pipelines are changed.

```bash
python -m pip install -e ".[datasets,experiments]"

# Small verification problem
python experiments/run_rejection_benchmark.py --profile pilot --ks 4 10 --repeats 2 --run-name my_pilot

# Both full reference dimensions, initially k=10
python experiments/run_rejection_benchmark.py --profile reference --ks 10 --repeats 3 --run-name my_reference_k10

# Optional subsequent rank sweep, subject to memory checks
python experiments/run_rejection_benchmark.py --profile reference --ks 10 20 50 100 --repeats 10 --run-name my_rank_sweep
```

Profiles give **stored reference shapes**: `pilot` uses dense 200×200 and sparse
2,000×200; `intermediate` uses dense 1,000×1,000 and sparse 100,000×1,000;
`reference` uses dense 10,000×10,000 and sparse 1,000,000×10,000. The benchmark
transposes these inputs and explicitly selects columns: the full sparse CSSP
input is **10,000×1,000,000**. Neither the original sparse input nor the full
reconstructed matrix is densified. See [dataset details](data/rejection_arp/README.md).

Two measurements are kept separate:

- `selection_only`: both selectors receive the same precomputed basis; basis
  and interpolation times are zero placeholders and not measurements in this mode.
- `end_to_end`: each method constructs its own identically seeded basis, selects
  columns, and builds `C` and `X` for the approximation `C @ X`. It includes input
  validation and the small sketch-rank check, but excludes input generation,
  loading, full error evaluation, output writing, and plotting. This is the
  timed factor-construction workflow, not total script wall time.

One warm-up per method/mode/k is excluded. Methods run in shuffled order with
recorded independent basis/selector seed streams and verified single-thread
BLAS. SparseStack generation follows the pinned reference code; interpolation
uses chunked LU solves instead of the author's selector-returned QR factors.
Full-basis checks and the current Python sampling implementation remain included,
so matching MATLAB speedups is not guaranteed. Sketch rank failures stop and
mark a run incomplete rather than silently retrying or altering the data.

Outputs are `results/e4/rejection/<run-name>/{config.json,raw_results.csv,summary.csv,plots/}`.
Summaries/plots show median and 25th/75th percentiles for each method and mode.
Use `--datasets dense` or `--datasets sparse` for one input; `--no-plots` omits
figures. New run directories are required; raw rows are saved incrementally.
Generated inputs are cached under `data/rejection_arp/generated/` with hashes.

The default `--memory-gib 4` rejects cases whose conservative working-memory
estimate exceeds 4 GiB **before generation**. It is not a measured peak or hard
memory limit. Skips are recorded and the status becomes `complete_with_skips`.
The full sparse input with k=100 exceeds that default estimate and will be
skipped; do not raise the limit without checking available system memory.
Both full reference dimensions completed an initial three-repetition k=10 run
in `results/e4/rejection/reference_k10/`. This is not the full paper rank sweep.

## E5: matrix structure at fixed dimensions

`experiments/run_matrix_structure.py` compares all six pipelines on six matrix
families: Gaussian, exact low-rank, low-rank with 1% or 50% relative noise,
rapid singular-value decay, and slow decay. Defaults are size 500×500, `k=10`,
signal rank 25, three matrix instances per family, ten accuracy trials per
randomized combination, and ten standalone timing repetitions per pipeline.
Randomized SVD uses `p=10, q=1`.

```bash
python -m pip install -e ".[experiments]"

# All six families, small workflow pilot
python experiments/run_matrix_structure.py --size 60 --k 5 --signal-rank 15 --instances 1 --trials 2 --repeats 2 --run-name structure_pilot

# Full planned study
python experiments/run_matrix_structure.py --run-name structure_full
```

Every matrix is normalized to unit Frobenius norm. Within an instance, noise
cases share the same signal and noise direction; their noise norms are 1%/50%
of signal norm before final normalization. Spectral cases share orthogonal
factors: rapid decay is `geomspace(1,1e-8,size)` and slow decay is `1/i`.
The exact low-rank signal uses `geomspace(20,1,signal_rank)`; `k < signal-rank`
is required to avoid a trivial exact reconstruction. Data/algorithm seeds are
separate and recorded; a fixed seed reproduces the inputs on the same environment.

Accuracy uses shared exact/randomized bases as in E2, with deterministic-SVD +
Osinsky evaluated once per matrix. Runtime uses E4's standalone component timer
and verified single-thread control: a fresh basis per method, only orthogonal
reconstruction, one excluded warm-up, shuffled order, and no baseline/error
evaluation inside timed intervals. Matrix-generation cost is excluded.

Outputs under `results/e5/<run-name>/`:

- `config.json`: parameters, matrix fingerprints, environment, and completion status.
- `raw_results.csv`: separate `accuracy` and `runtime` rows; inapplicable fields
  are blank, not zero measurements.
- `instance_summary.csv`: within-matrix statistics for each method/metric.
- `summary.csv`: equally weighted means of matrix-instance accuracy means or
  runtime medians, plus between-instance sample standard deviation.
- `spectra.csv` and `plots/`: measured singular values, relative error,
  SVD-optimality ratios, and algorithm-runtime comparisons.

Plot error bars describe **between-matrix-instance** variation, not variation
across randomized trials or confidence intervals. They are omitted for a
single-instance pilot; within-matrix variability remains in `instance_summary.csv`.
Exact low-rank matrices can have tiny roundoff singular values in measured
spectra. Zero-baseline ratios remain undefined (`NaN`). Use `--no-plots` for
CSV-only output or `--families` for a subset. Existing runs are never overwritten;
raw measurements are flushed incrementally and failed runs are marked incomplete.

## E6: matched dense versus sparse storage

`experiments/run_sparsity.py` compares four combinations: dense/CSR input ×
sequential ARP/Rejection ARP. Each uses the same SparseStack basis and factorized
oblique column approximation as the E4 rejection benchmark. This is **not** the
six exact/randomized-SVD pipelines or orthogonal reconstruction.

```bash
python -m pip install -e ".[datasets,experiments]"

# Four densities and all four combinations, small pilot
python experiments/run_sparsity.py --size 100 --k 5 --instances 1 --trials 2 --run-name sparsity_pilot

# 500 x 500, k=10, three matrices per density, ten trials
python experiments/run_sparsity.py --run-name sparsity_full
```

Default densities are 1, .5, .1, .01. One Gaussian source and one position
permutation per instance yield nested masks with `round(density*size^2)` retained
entries; matrices are normalized to unit Frobenius norm. Dense and CSR formats
contain exactly the same values. Zero rows/columns are retained and counted.
Masking changes the spectrum as well as sparsity, so each matrix has its own
untimed SVD baseline. A rank-deficient sketch stops the run rather than silently
resampling the input or sketch.

Each method independently constructs its basis and interpolation factors,
using identical algorithm seeds for the paired dense/CSR runs. One warm-up per
combination/matrix is excluded; execution order is shuffled and BLAS threading
is verified. Only basis, selection and factor construction are timed. Data
generation, format conversion, baseline SVD and blockwise error evaluation
occur outside those intervals. Full `C @ X` reconstruction is not part of
algorithm time; evaluation forms temporary column blocks after timing.

Outputs under `results/e6/<run-name>/` are `config.json`, `raw_results.csv`,
`instance_summary.csv`, `summary.csv`, `paired_results.csv`, and four plots
(relative error, SVD ratio, runtime, input buffer bytes). Configuration records
matrix hashes, density, nonzero counts, zero rows/columns and library versions.
Pair records show dense/CSR index agreement, error differences and timing ratios;
finite-precision arithmetic can occasionally change random pivot decisions.

Reported storage is **numeric buffer bytes**, not peak RAM: dense `nbytes` versus
CSR `data + indices + indptr`. Factor bytes include selected columns and dense
coefficients; the sketch basis and other workspaces are not included in factor
storage. Both representations coexist in this matched experiment, so process
memory must not be interpreted as the cost of either representation alone.
CSR may use more input storage than dense at high density. Sparse-aware does
not mean every intermediate is sparse: the narrow basis and coefficients are dense.

Instance summaries show trial variation. Overall summaries equally weight
matrix-instance accuracy/storage means or runtime medians; plot bars are
between-instance SD, not confidence intervals. Three matrices with ten trials
are not thirty independent matrix instances. Use `--no-plots` for CSV-only
output. Existing folders are protected and partial results are retained.

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
