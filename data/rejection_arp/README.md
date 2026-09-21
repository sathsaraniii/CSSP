# Rejection ARP reference datasets

These are the two synthetic runtime examples from Ethan N. Epperly,
*Adaptive randomized pivoting and volume sampling*, Section 7.2.
They are generated matrices, not external measured datasets such as DNA.

**Current status:** this folder records the references and generation definitions
only. Neither large matrix has been generated or downloaded, and no benchmark
has been run. A future generator should record its seed and numerical-library
versions. A new seeded Python realization will not be entry-for-entry identical
to the authors' MATLAB realization.

## 1. Dense example

- Shape: **10,000 × 10,000**.
- Generate a matrix `G` of independent standard normal entries.
- Scale row `i` by `i**(-2)`, with mathematical indices starting at **1**:
  `A[i-1, :] = G[i-1, :] / i**2`.
- Equivalently: `A = diag(1**(-2), ..., 10000**(-2)) @ G`.
- Store as float64. The matrix alone requires 800,000,000 bytes (800 MB,
  approximately 763 MiB), excluding temporary arrays and algorithm workspaces.
- Future generation should scale rows directly rather than allocating a dense
  diagonal matrix.

Reference implementation:
[tests/test_arp_dense.m](https://github.com/eepperly/Adaptive-Randomized-Pivoting/blob/main/tests/test_arp_dense.m).

## 2. Sparse example

- Shape: **1,000,000 × 10,000**.
- The paper describes a sparse Gaussian matrix `G` with **30 nonzero entries
  per column**, at random positions.
- Apply the same row scaling: `A[i-1, :] = G[i-1, :] / i**2`.
- The MATLAB experiment calls `random_sparse(m,n,30)` and left-multiplies by a
  sparse diagonal matrix. Before porting the generator, inspect that helper's
  handling of random positions and possible duplicates for code-level parity.
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

References checked on 21 September 2026. GitHub links above point to `main` and
may change; pin a commit when implementing a reference-reproduction experiment.
