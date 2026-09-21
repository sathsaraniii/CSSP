# Rejection ARP reference datasets

These are the two synthetic runtime examples from Ethan N. Epperly,
*Adaptive randomized pivoting and volume sampling*, Section 7.2.
They are generated matrices, not external measured datasets such as DNA.

The generator and benchmark are in
[`experiments/run_rejection_benchmark.py`](../../experiments/run_rejection_benchmark.py).
Generated matrices live under `generated/` with seed/version metadata and SHA-256
fingerprints. The `.npy`/`.npz` binaries are ignored by Git; metadata is retained.
These seeded Python realizations are not entry-for-entry identical to MATLAB.
On 21 September 2026, both full reference-size inputs were generated with seed
42 and an initial `k=10`, three-repetition benchmark completed. Results are in
`results/e4/rejection/reference_k10/`; the paper's full rank sweep is not complete.

## 1. Dense example

- Shape: **10,000 × 10,000**.
- Generate a matrix `G` of independent standard normal entries.
- Scale row `i` by `i**(-2)`, with mathematical indices starting at **1**:
  `A[i-1, :] = G[i-1, :] / i**2`.
- Equivalently: `A = diag(1**(-2), ..., 10000**(-2)) @ G`.
- Store as float64. The matrix alone requires 800,000,000 bytes (800 MB,
  approximately 763 MiB), excluding temporary arrays and algorithm workspaces.
- Generation scales rows in blocks into an `.npy` memory-mapped file; no dense
  diagonal matrix is allocated.

Reference implementation:
[tests/test_arp_dense.m](https://github.com/eepperly/Adaptive-Randomized-Pivoting/blob/main/tests/test_arp_dense.m).

## 2. Sparse example

- Shape: **1,000,000 × 10,000**.
- The paper describes a sparse Gaussian matrix `G` with **30 nonzero entries
  per column**, at random positions.
- Apply the same row scaling: `A[i-1, :] = G[i-1, :] / i**2`.
- The MATLAB helper `random_sparse(m,n,30)` samples positions **without
  replacement within each column**. The Python generator follows this rule and
  applies scaling to stored values directly, retaining CSC storage on disk.
- Thirty stored nonzeros per column gives 300,000 nonzeros in total, rather
  than storing all 10 billion possible entries.
- Keep this matrix sparse. Dense float64 storage alone would require **80 GB**,
  before any SVD, copies, or reconstruction.

Reference implementation:
[tests/test_arp_sparse.m](https://github.com/eepperly/Adaptive-Randomized-Pivoting/blob/main/tests/test_arp_sparse.m).

## Benchmark interpretation

The author's scripts vary the selected rank over ten approximately logarithmic
values from 10 to 1,000 and use MATLAB `timeit`. They benchmark row interpolative
decomposition methods, not this repository's six SVD-based CSSP combinations.

For an equivalent column-selection problem, the mathematical input is `A.T`;
in particular, the sparse row-selection example corresponds to a
**10,000 × 1,000,000** column-selection input. A sparse transpose need not be
densified. Do not pass either orientation through a dense conversion merely to
fit the current pipeline. Matching the data definitions alone does not reproduce
the reference's sparse sketching, interpolation, timing, or speedups.

## Sources

- [Paper, Section 7](https://arxiv.org/html/2510.02513v2#S7).
- [Author's implementation repository](https://github.com/eepperly/Adaptive-Randomized-Pivoting).
- [Dense runtime script](https://github.com/eepperly/Adaptive-Randomized-Pivoting/blob/main/tests/test_arp_dense.m).
- [Sparse runtime script](https://github.com/eepperly/Adaptive-Randomized-Pivoting/blob/main/tests/test_arp_sparse.m).

Implementation checked against commit
[`98bc376c247ab2ef133e1b51fac854983c7c3c8b`](https://github.com/eepperly/Adaptive-Randomized-Pivoting/tree/98bc376c247ab2ef133e1b51fac854983c7c3c8b),
including `utils/random_sparse.m`, `code/sparsestack.c`, and `code/arp.m`.
The Python SparseStack has one signed entry per disjoint block, four blocks
(or `k` when `k<4`), and entries scaled by `1/sqrt(number_of_blocks)`.

## Generate/reuse and benchmark

From the repository root in the activated environment:

```bash
python -m pip install -e ".[datasets,experiments]"
python experiments/run_rejection_benchmark.py --profile reference --ks 10 --repeats 3 --run-name my_reference_k10
```

Use a new run name each time. Completed datasets are reused after fingerprint
verification; partial datasets are not silently overwritten. Different seeds
create additional files (about 800 MB per dense realization).

Input files retain the original reference orientation; the loader returns their
transposes for **explicit column selection**. For sparse data the CSSP matrix is
10,000 × 1,000,000, with 30 nonzeros per row after transposition.
The benchmark stores `C=A[:,J]` and `X=solve(V[J,:].T,V.T)` in memory and times
their construction, but never forms `C @ X`. Large factors are discarded after
each trial, not saved. `V=orth(A.T @ Omega)` is a narrow sparse-sign sketch basis,
not the full-SVD or Gaussian-RSVD basis in the original six pipelines.
