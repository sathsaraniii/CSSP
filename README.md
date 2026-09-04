
# CS4523 Project: A Comparative Study of Deterministic and Randomized Approaches to Column Subset Selection for Low-Rank Matrix Approximation

```text
                     INPUT MATRIX A
                          │
             ┌────────────┴────────────┐
             │                         │
      ROW-SPACE STAGE           ROW-SPACE STAGE
      Deterministic              Randomized
      Truncated SVD              Randomized Sketch/SVD
             │                         │
             └────────────┬────────────┘
                          │
                 DOMINANT ROW SPACE
                          │
             ┌────────────┴────────────┐
             │                         │
       COLUMN SELECTION          COLUMN SELECTION
         Osinsky (D)               ARP (R)
             │                         │
             └────────────┬────────────┘
                          │
                 SELECTED COLUMNS C
                          │
                   RECONSTRUCTION
                          │
                       METRICS


```

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

On macOS or Linux, replace `.\.venv\Scripts\python.exe` with
`./.venv/bin/python`.

## Run the progress demo

From the repository root:

```powershell
.\.venv\Scripts\python.exe experiments\run_experiment.py
```

Example with custom parameters:

```powershell
.\.venv\Scripts\python.exe experiments\run_experiment.py --m 500 --n 300 --k 20 --p 10 --q 1 --seed 42
```

The script reports relative Frobenius reconstruction error, error relative
to the optimal rank-`k` truncated SVD, row-space time, selector time, and
selected zero-based column indices.

## Python API

After the editable installation above, run all four pipelines with:

```python
from cssp.cssp_pipeline import run_four_pipelines

results = run_four_pipelines(
    A,
    k=10,
    random_state=42,
    oversampling=10,
    power_iterations=1,
)

for name, result in results.items():
    print(name, result["indices"], result["relative_error"])
```

One combination can be run with `run_cssp_pipeline`, or through the named
functions `deterministic_svd_osinsky`, `deterministic_svd_arp`,
`randomized_svd_osinsky`, and `randomized_svd_arp`.

Each result contains both reconstructions:

- `orthogonal_approximation`: `C @ pinv(C) @ A`, used in the paper's CSSP
  experiments and selected as the default result.
- `oblique_approximation`: `A[:, J] @ V[J, :]^{-T} @ V.T`, controlled by the
  ARP expectation theorem.

## Tests

Install the dependencies and run:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```
