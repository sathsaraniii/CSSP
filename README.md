# Comparative Column Subset Selection Pipelines

This project compares deterministic and randomized approaches to column
subset selection for low-rank matrix approximation. It implements the full
2 x 2 experiment:

| Pipeline | Row-space stage | Column selection |
|---|---|---|
| `deterministic_osinsky` | Exact truncated SVD | Osinsky |
| `deterministic_arp` | Exact truncated SVD | ARP |
| `randomized_osinsky` | Randomized SVD | Osinsky |
| `randomized_arp` | Randomized SVD | ARP |

The ARP Householder update and Osinsky residual update follow the authors'
MATLAB reference implementation under `ARP/ARP/selection_methods`.

## Run the progress demo

From the repository root:

```powershell
py experiments/run_experiment.py
```

Example with custom parameters:

```powershell
py experiments/run_experiment.py --m 500 --n 300 --k 20 --p 10 --q 1 --seed 42
```

The script reports relative Frobenius reconstruction error, error relative
to the optimal rank-`k` truncated SVD, row-space time, selector time, and
selected zero-based column indices.

## Python API

Add `src` to `PYTHONPATH` or install the project as a source-layout package,
then run all four pipelines:

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
py -m pip install -r requirements.txt
py -m pytest -q
```
