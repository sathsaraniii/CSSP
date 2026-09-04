# Adaptive Randomized Pivoting: paper and codebase review

**Paper:** Alice Cortinovis and Daniel Kressner, *Adaptive randomized pivoting for column subset selection, DEIM, and low-rank approximation*  
**Reviewed version:** arXiv v2, 19 June 2025  
**Published version:** *SIAM Journal on Matrix Analysis and Applications* 47(1), 2026, pp. 25-47, DOI 10.1137/24M1719189  
**Authors' code:** MATLAB repository at commit `a57b13e2560d75c6d06505464bfe5152dd774c12`  
**Review date:** 3 September 2026

## Executive takeaway

The paper's main contribution is a very simple way to select exactly `r` informative columns from a matrix when an approximate dominant row-space basis is available. The method, Adaptive Randomized Pivoting (ARP), repeatedly samples from leverage scores and **updates the subspace after every sample**. That adaptive update is the key difference from ordinary leverage-score sampling.

Given an orthonormal row-space basis `V in R^{n x r}`, ARP chooses `r` indices without needing the data matrix `A` during pivot selection. Its induced oblique approximation satisfies

```text
E ||A - A(:,J)V(J,:)^{-T}V^T||_F^2
    = (r+1) ||A-AVV^T||_F^2.
```

Projection onto the span of the selected columns is at least as accurate. When `V` contains the top `r` right singular vectors, this gives the tight best-possible CSSP factor `r+1` in expected squared Frobenius error. The authors extend the same sampling mechanism to DEIM, CUR/cross approximation, and Nyström approximation, and show that derandomization recovers Osinsky's deterministic CSSP method. Their new deterministic Nyström method is arguably the most distinctive extension beyond CSSP.

The supplied [authors' MATLAB repository](https://github.com/Alice94/ARP) implements the paper's selector and all five experiment families. Your separate Python project currently implements only a conceptual CSSP version of ARP; its pipeline is incomplete and its Osinsky implementation is broken.

## 1. The problem they solve

For `A in R^{m x n}`, the Column Subset Selection Problem (CSSP) asks for `r` actual columns, `C=A(:,J)`, whose span approximates all of `A`:

```text
minimize ||A - Pi_J A||_F,
```

where `Pi_J` projects orthogonally onto `span(A(:,J))`.

The truncated SVD is optimal among all rank-`r` matrices but uses linear combinations of every column, so it is not a column subset. Prior volume-sampling theory established that some `r`-column subset always achieves at most `(r+1)` times the optimal squared Frobenius error, and that this factor is tight. The problem was to obtain that guarantee with a practical sampler. The paper positions ARP as a simpler, cheaper realization of that optimal-in-expectation guarantee. See the [paper's introduction and equations (1.1)-(1.5)](https://arxiv.org/html/2412.13992v2).

## 2. What ARP actually does

ARP assumes an orthonormal matrix `V in R^{n x r}` satisfying `A approximately AVV^T`. In the experiments `V` is the exact top-right-singular-vector basis `V_opt`, although the paper also allows a cheaper approximate or sketched basis.

At iteration `k`:

1. Compute adaptive leverage probabilities

   ```text
   p_j = ||W(j,k:r)||_2^2 / (r-k+1).
   ```

2. Randomly sample an index `j_k` using `p`.
3. Apply a Householder reflector to the remaining columns of `W`, rotating row `j_k` so its trailing entries become zero.
4. Recompute the probabilities from the changed trailing basis and repeat.

This construction has four important consequences:

- The probabilities sum to one because the trailing basis has squared Frobenius norm `r-k+1`.
- A selected index has zero probability later, so ARP returns exactly `r` distinct indices.
- The selected block `V(J,:)` is nonsingular; in the rotated coordinates it is lower triangular with nonzero diagonal.
- Given `V`, selection costs `O(n r^2)` and does not inspect `A`.

The authors also point out that ARP is exactly an `r`-determinantal point process sampler with kernel `VV^T`, but their analysis directly connects it to low-rank approximation. The stable algorithm and its complexity are given in [Section 2.2](https://arxiv.org/html/2412.13992v2); the code is [selection_methods/ARP.m](ARP/ARP/selection_methods/ARP.m).

### Why adaptation matters

Ordinary leverage sampling computes `||V(j,:)||^2` once. It can repeatedly sample redundant directions and usually needs oversampling. ARP removes the direction explained by each pivot, so later samples target directions not already represented. It gets exactly `r` nonredundant pivots while retaining the tight expectation guarantee.

### Why it is a randomized Osinsky method

Osinsky's deterministic method chooses at each step the index that minimizes the next residual increase. ARP samples according to a distribution whose conditional average produces the same one-step bound. Replacing the random draw by the best conditional choice recovers Osinsky's algorithm. The relevant precursor is Osinsky's [*Close to optimal column approximations with a single SVD*](https://arxiv.org/abs/2308.09068), not the different paper cited at the top of your current `osinsky.py` docstring.

## 3. The proof idea, without all the algebra

Define the part not represented by the supplied row basis:

```text
A_tilde = A(I - VV^T).
```

The paper writes the final oblique error as a product of one-pivot oblique projectors. Conditioned on the pivots already chosen, the expected squared residual at step `k` is multiplied by

```text
(r-k+2)/(r-k+1).
```

Multiplying these factors for `k=1,...,r` telescopes:

```text
(r+1)/r * r/(r-1) * ... * 2/1 = r+1.
```

That produces the exact expectation identity for the oblique approximation. Orthogonal projection onto `span(A(:,J))` minimizes error within that selected span, so it can only be better. The complete argument is in [Section 2.3](https://arxiv.org/html/2412.13992v2).

The guarantee is an expectation for **squared** error. The paper derives a square-root expectation bound via Jensen and a loose high-probability statement via Markov; it does not establish a sharp concentration theorem. This distinction matters when designing experiments because rare ill-conditioned selections can produce heavy-tailed oblique errors.

## 4. Contributions beyond basic CSSP

| Application | Construction | Main guarantee/point |
|---|---|---|
| CSSP | Run ARP on row basis `V`, then use selected columns `A(:,J)` | Expected oblique squared error is exactly `(r+1)` times `||A-AVV^T||_F^2`; orthogonal CSSP error is no worse. |
| Sketched CSSP | Form `Y=A^T Omega` with `r+2` Gaussian vectors, QR to obtain `V`, ARP-select `r+2` columns | Expected error is bounded by `(r+3)(r+1)` times the optimal rank-`r` squared tail. |
| DEIM | Run ARP directly on the reduced basis `V`; evaluate a future function only at selected coordinates | For fixed `f`, expected squared interpolation error is at most `(r+1)||f-VV^Tf||_2^2`; selection is oblivious to the future `f`. |
| CUR | ARP-select columns from a row basis and rows from a column basis | Expected error is controlled by the two supplied subspace residuals. |
| Cross/skeleton | ARP-select `J`; QR-factor `A(:,J)`; ARP-select row set `I` from that QR basis | `E||A-A(:,J)A(I,J)^{-1}A(I,:)||_F^2 <= (r+1)^2||A-AVV^T||_F^2`. |
| SPSD/Nyström | Apply ARP to a basis of an SPSD matrix and reuse `J` for rows and columns | Expected trace/nuclear error has tight factor `r+1` with optimal `V`. |
| Deterministic Nyström | Derandomize through the Gram correspondence while updating only needed diagonal/column data | New Algorithm 5.1 obtains the same factor deterministically without forming a square root of `A`. |

The authors are candid about the tradeoff: Nyström ARP needs `V`, whereas randomly pivoted Cholesky can work from the diagonal and selected matrix entries. Consequently, they regard randomized ARP as less compelling for Nyström; the deterministic Algorithm 5.1 is the more interesting contribution there. The application results are in [Sections 3-5](https://arxiv.org/html/2412.13992v2).

## 5. What they tested

The paper reports MATLAB R2024b experiments using exact `V_opt` from a full SVD. Randomized results are averaged, with error bars spanning the empirical 10th to 90th percentiles.

| Experiment | Data and rank range | Compared methods | Reported outcome | Repository entry |
|---|---|---|---|---|
| Sparse CSSP | SuiteSparse/Meszaros `large`, 4,282 x 8,617; `r=40:40:400` | ARP, fixed leverage, column norm, CPQR, Osinsky, SVD best | Fixed leverage sampling is relatively poor; the other methods are close to best. | [test_CSS_large.m](ARP/ARP/tests/test_CSS_large.m) |
| Genetic CSSP | NIH GEO GSE10072, 22,283 x 107; `r=2:2:50` | Same CSSP methods | All are similar; Osinsky is slightly better but more expensive. | [test_CSS_DNA.m](ARP/ARP/tests/test_CSS_DNA.m) |
| DEIM | 2,500 x 144 training snapshots; 121 held-out samples; `r=2:2:50` | ARP, Q-DEIM, R-DEIM, uniform | Q-DEIM is best here; R-DEIM is slightly better than ARP; ARP avoids oversampling. | [test_DEIM.m](ARP/ARP/tests/test_DEIM.m) |
| Cross approximation | Nonsymmetric 2,000 x 2,000 two-bump kernel; `r=3:3:51` | ARPcross, ACA full/partial, randLUPP, SVD best | ARPcross, ACA full, and randLUPP are comparable; partial ACA struggles. | [test_CA.m](ARP/ARP/tests/test_CA.m) |
| Nyström | 1,000-point smile and spiral Gaussian kernels; `r=3:3:60` | ARP, RPCholesky, uniform, leverage, Algorithm 5.1, greedy nuclear, SVD best | ARP is competitive; it is visually best among randomized methods on the slow-decay spiral. Algorithm 5.1 is usually better but costlier. | [test_SMILE.m](ARP/ARP/tests/test_SMILE.m), [test_SPIRAL.m](ARP/ARP/tests/test_SPIRAL.m) |

These are preliminary validation experiments, not a broad benchmark. They do not measure runtime, memory, or end-to-end performance with a cheap approximate `V`. The paper itself explicitly leaves fine-tuning and efficient implementation to future work. See [Section 6 and the conclusion](https://arxiv.org/html/2412.13992v2).

## 6. Authors' MATLAB codebase map

The nested `ARP/ARP` repository is clean and matches the authors' public repository at commit `a57b13e`.

- [selection_methods/ARP.m](ARP/ARP/selection_methods/ARP.m): the core Householder ARP selector. Input is only `V`; output is `J`.
- [selection_methods/get_columns_osinsky.m](ARP/ARP/selection_methods/get_columns_osinsky.m): deterministic Algorithm 2.3.
- [selection_methods/get_columns_det_SPSD.m](ARP/ARP/selection_methods/get_columns_det_SPSD.m): deterministic Nyström Algorithm 5.1.
- `ACA_*`, `CPQR`, `Q_DEIM`, `R_DEIM`, `randLUPP`, `RPCholesky`, `FornaceLindsey`, and sampling helpers: experimental baselines.
- `tests/test_*.m`: standalone scripts for the paper's figures. They save intermediate `.mat` variables and EPS figures.
- `data/`: both real CSSP matrices plus smile/spiral generators.

Reproduction is not turn-key. The README gives no setup or master command, scripts rely on the current directory for data files, `randsample` requires the relevant MATLAB functionality/toolbox, and neither MATLAB nor Octave is available in the current environment. I therefore reviewed the code structurally but did not reproduce the plots.

## 7. Paper-to-code discrepancies

These are worth preserving in your notes before building on the repository:

1. **DEIM data-generation mismatch.** The paper defines

   ```text
   g(...) = (... + 0.1^2)^(-1/2),
   ```

   but [test_DEIM.m](ARP/ARP/tests/test_DEIM.m) computes `sqrt(... + 0.1^2)`, i.e. the positive one-half power. The discrepancy is confirmed against the [arXiv HTML equation](https://arxiv.org/html/2412.13992v2).

2. **Run-count mismatch.** The paper says every randomized algorithm is run 100 times. [test_CSS_DNA.m](ARP/ARP/tests/test_CSS_DNA.m) sets `T=50`; the other experiment scripts use 100.

3. **Algorithm-label typo.** [test_CSS_large.m](ARP/ARP/tests/test_CSS_large.m) labels Osinsky's curve "Algorithm 2.2", but in the supplied paper ARP is Algorithm 2.2 and Osinsky is Algorithm 2.3. The DNA script uses the clearer label "Osinsky".

4. **Unplotted deterministic baseline.** The Nyström prose says full/diagonal-pivot ACA is compared. The smile and spiral scripts calculate `error_greedy`, but that curve is commented out or absent from the plots; the paper's Figure 4 legend also omits it.

5. **The claimed access advantage is not exercised end-to-end.** Every experiment obtains `V_opt` with a full SVD. The tests therefore assess pivot quality, not whether sketch + ARP is faster or more scalable than alternatives.

None of these invalidate the central theorem, but the DEIM mismatch is material if you try to reproduce Figure 3a exactly from the mathematical definition.

## 8. Status of your Python implementation

Your Python code is separate from the authors' MATLAB repository.

### What is already aligned

[src/cssp/column_selection/arp.py](src/cssp/column_selection/arp.py) implements the conceptual prototype correctly at a high level:

- it obtains a top-`k` right singular basis;
- samples from adaptive row-norm probabilities;
- projects out the sampled direction;
- returns distinct indices, selected columns, probability history, and the oblique interpolatory approximation.

A bounded exact check on a small random matrix confirmed the theoretical identity when subsets are weighted by `det(V(J,:))^2`: the weights summed to one and the expected oblique squared-error ratio was `4.00000000000001` for `r=3`, as predicted by `r+1=4`.

### Important differences and gaps

1. **Interface:** the paper's selector accepts `V`; `arp_cssp(A,k)` always recomputes exact `V` internally. This blocks experiments with a supplied randomized/sketched or application-native basis.
2. **Update cost:** the paper uses Householder reflectors for `O(n r^2)` work. The Python code performs a fresh SVD after each pivot, giving roughly `O(n r^3)` update work.
3. **Reconstruction metric:** Python reports the oblique interpolation error. The paper's CSSP figures report the orthogonal error `||A-A(:,J)A(:,J)^dagger A||_F`, which is no larger. Directly comparing those numbers would be unfair.
4. **Linear solve:** Python uses `pinv(V(J,:).T)`; the paper's Householder form exposes a triangular solve. The pseudoinverse is convenient but can change behavior for numerically ill-conditioned selections.
5. **Incomplete pipeline:** [src/cssp/cssp_pipeline.py](src/cssp/cssp_pipeline.py), [experiments/run_experiment.py](experiments/run_experiment.py), `metrics/evaluation.py`, and `reconstruction/cssp_approximation.py` are empty.
6. **Broken deterministic baseline:** [src/cssp/column_selection/osinsky.py](src/cssp/column_selection/osinsky.py) initializes `E=A` instead of `A-AVV^T`, then subtracts an `m x remaining_rank` outer product from an `m x n` residual. An 8 x 10, rank-3 smoke test fails with a broadcasting `ValueError`.
7. **Different randomized-SVD path:** [src/cssp/basis/randomized_svd.py](src/cssp/basis/randomized_svd.py) is a conventional approximate rank-`k` SVD. It is usable as a possible source of `V`, but it does not directly implement Corollary 2.6's `A^T Omega`, `r+2`-column selection procedure.
8. **Testing:** `pytest` is declared in `requirements.txt` but is not installed in the current environment. `test_svd.py` is an executable comparison script rather than an assertion-based test suite.

So the honest current status is: **the Python ARP selection idea is present, but the project is not yet a faithful, scalable, end-to-end reproduction of the paper.** No code was changed during this review.

## 9. The mental model to carry forward

ARP is best understood as a reusable **index-selection layer**:

```text
data or model
    -> obtain an approximate dominant subspace V
    -> adaptively sample nonredundant coordinates from V
    -> evaluate only the selected columns/rows/entries
    -> build CSSP, DEIM, cross, or Nyström approximation
```

Its strength is the combination of exact-cardinality sampling, minimal data access after `V` is known, and a tight expectation bound. Its main practical dependency is also clear: the quality and cost of obtaining `V`. That boundary—subspace construction versus pivot selection—is the most important distinction to preserve when later designing a genuinely novel extension or a new-dataset study.

## Sources

- [Cortinovis and Kressner, arXiv v2 (19 June 2025)](https://arxiv.org/html/2412.13992v2)
- [Cortinovis and Kressner, SIAM version of record (2026)](https://doi.org/10.1137/24M1719189)
- [Authors' MATLAB repository](https://github.com/Alice94/ARP)
- [Frozen repository revision reviewed here](https://github.com/Alice94/ARP/tree/a57b13e2560d75c6d06505464bfe5152dd774c12)
- [Osinsky, *Close to optimal column approximations with a single SVD*](https://arxiv.org/abs/2308.09068)

