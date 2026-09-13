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
