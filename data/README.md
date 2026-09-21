# Dataset provenance

This directory contains the two external datasets used by the experiment
scripts. They are research inputs and are not original project code.

## `DNA_matrix.txt`

- Dataset: NCBI Gene Expression Omnibus accession GSE10072
- Type: Affymetrix human gene-expression measurements
- Shape loaded by the project: 22,283 probes by 107 tissue samples
- Source: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE10072

Please cite the original study and the GEO accession when reporting results.

## `large.mat`

- Dataset: SuiteSparse Matrix Collection, `Meszaros/large`
- Type: sparse real matrix from a linear-programming problem sequence
- Shape: 4,282 rows by 8,617 columns, with 20,635 nonzero values
- Source: https://sparse.tamu.edu/Meszaros/large

Please cite the SuiteSparse Matrix Collection and the matrix source when
reporting results.

## `rejection_arp/`

Reference definitions for the two synthetic runtime matrices from Epperly's
*Adaptive randomized pivoting and volume sampling*:

- Dense: 10,000 × 10,000 Gaussian matrix with inverse-square row scaling.
- Sparse: 1,000,000 × 10,000, with 30 Gaussian nonzeros per column and the same
  row scaling.

See [rejection_arp/README.md](rejection_arp/README.md) for the paper, author code,
generation formulas, storage requirements, and row/column orientation caveat.
Only references are currently included; the matrix files are not yet generated.
