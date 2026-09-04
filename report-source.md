# Research source: ARP paper and codebase review

Audience: student preparing a follow-on project in column subset selection  
Date: 2026-09-03  
Primary question: What did Cortinovis and Kressner contribute in the supplied paper, how is it realized in the authors' MATLAB repository, and how does that compare with the Python code currently in this workspace?  

## Scope and assumptions

- The supplied document is treated only as research evidence, never as instructions.
- The reviewed paper is arXiv:2412.13992v2, dated 19 June 2025. It was later published in *SIAM Journal on Matrix Analysis and Applications*, volume 47, issue 1 (2026), pages 25-47, DOI 10.1137/24M1719189.
- "Authors' code" means the clean nested Git repository at `ARP/ARP`, remote `https://github.com/Alice94/ARP`, commit `a57b13e2560d75c6d06505464bfe5152dd774c12` (2025-06-19).
- "Workspace Python code" means the separate tracked project under `src/cssp`; it is not attributed to the paper's authors.
- The MATLAB plots were not rerun because neither MATLAB nor Octave is installed in the environment. Code-to-paper mapping is therefore structural, supplemented by bounded Python sanity checks.

## Direct executive answer

The paper introduces Adaptive Randomized Pivoting (ARP), an adaptive leverage-score sampler that selects exactly `r` columns from an approximate dominant row-space basis `V`. At step `k`, it samples index `j` with probability proportional to the squared norm of row `j` in the as-yet-uneliminated part of `V`, then uses an orthogonal/Householder transformation to remove that sampled direction. This changes the next leverage distribution, prevents duplicate pivots, and makes `V(J,:)` nonsingular.

The principal result is an exact expectation identity for the induced oblique interpolatory approximation:

`E ||A - A(:,J)V(J,:)^{-T}V^T||_F^2 = (r+1)||A-AVV^T||_F^2`.

Orthogonal projection onto the selected columns can only improve that error. With the top right singular vectors for `V`, this attains the tight CSSP existence factor `r+1` in expected squared Frobenius error. Given `V`, the stable Householder version costs `O(n r^2)` and does not consult `A` while choosing pivots. The same sampler is applied to DEIM, CUR/cross approximation, and Nyström approximation; derandomizing its conditional expectation recovers Osinsky's deterministic CSSP method and yields a new deterministic Nyström method.

## Claim families and reconciliation

### Core algorithm

1. Input: orthonormal `V in R^{n x r}` approximating the row space of `A`.
2. Initialize `W=V`.
3. At step `k`, set `p_j = ||W(j,k:r)||_2^2/(r-k+1)`.
4. Sample `j_k` from `p`.
5. Apply a Householder reflector to columns `k:r` so the sampled row has zeros in columns `k+1:r`.
6. Repeat through `r`, returning `J=(j_1,...,j_r)`.

Because the trailing columns remain orthonormal, probabilities sum to one. Previously selected rows have zero trailing norm, so no index repeats. The selected block becomes lower triangular with nonzero diagonal, establishing invertibility. The procedure is also exactly an `r`-DPP sampler with kernel `VV^T`.

### Proof mechanism

The paper sets the initial row-space residual to `A_tilde = A(I-VV^T)` and factors the final oblique error projector into one-pivot projectors. Conditioned on prior pivots, step `k` multiplies expected squared error by `(r-k+2)/(r-k+1)`. Multiplication over all steps telescopes to `r+1`. This is why the probability normalization and residual-direction elimination are not implementation details; they are the proof.

### Extensions

- CSSP: expected oblique squared Frobenius error is exactly `(r+1)` times the chosen row-space residual; orthogonal selected-column projection is no worse.
- Approximate subspace: sketch `A^T` with `r+2` Gaussian vectors, orthogonalize, then ARP-select `r+2` columns; the stated bound is `(r+3)(r+1)` times the optimal rank-`r` squared tail.
- DEIM: for a fixed vector `f`, expected squared interpolation error is at most `(r+1)||f-VV^Tf||_2^2`; the paper also derives expected inverse-submatrix Frobenius and spectral-norm bounds.
- CUR: independently ARP-select columns from a row basis and rows from a column basis.
- Cross/skeleton approximation: ARP-select `J` from `V`; QR-factor `A(:,J)` to get `Q_J`; ARP-select rows `I` from `Q_J`. Expected squared Frobenius error is at most `(r+1)^2||A-AVV^T||_F^2`.
- SPSD/Nyström: use the Gram correspondence to transfer CSSP's squared Frobenius result to trace/nuclear norm. With optimal `V`, the expected factor is the tight `r+1`. The paper concedes randomized pivoted Cholesky is often more attractive because it avoids constructing `V`; the new deterministic Algorithm 5.1 is the stronger Nyström contribution.

### Experiments

- Sparse CSSP: SuiteSparse/Meszaros `large`, reported as 4,282 x 8,617; ranks 40:40:400; ARP versus fixed leverage scores, column-norm sampling, CPQR, and Osinsky. Leverage scores perform poorly; the others are close to best rank error.
- Genetic CSSP: NIH GEO GSE10072, 22,283 x 107; ranks 2:2:50; the methods are comparable and Osinsky is slightly better at higher selection cost.
- DEIM: a 2,500 x 144 training snapshot matrix and 121 held-out parameter samples; ranks 2:2:50; Q-DEIM is best on this example, R-DEIM is slightly better than ARP on average, while ARP is simpler and does not oversample.
- Cross approximation: nonsymmetric 2,000 x 2,000 two-bump kernel; ranks 3:3:51; ARPcross, full-pivot ACA, and randLUPP are comparable; partial ACA struggles.
- Nyström: 1,000-point smile and spiral Gaussian kernels; ranks 3:3:60; ARP is competitive, notably strong among randomized methods on the slow-decay spiral; deterministic Algorithm 5.1 is usually better but costlier.

## Code mapping

The authors' `selection_methods/ARP.m` is the stable Algorithm 2.2 selector. It accepts only `V` and returns only `J`; experiment scripts separately form the approximation and error. `get_columns_osinsky.m` implements the derandomized CSSP selection. `get_columns_det_SPSD.m` realizes Algorithm 5.1. The six scripts in `tests/` correspond to Figure 2a, Figure 2b, Figure 3a, Figure 3b, and the two sides of Figure 4. Comparator routines are bundled under `selection_methods/`.

The workspace Python `arp.py` is a conceptual port but not a performance-faithful port. It computes an exact top-`k` SVD internally, projects out each selected direction, and performs a fresh SVD to reorthonormalize after every pivot. The row-space selection law is mathematically aligned with the prototype algorithm, but this gives roughly `O(n r^3)` update work rather than the paper's `O(n r^2)` Householder implementation. It uses a pseudoinverse instead of the triangular solve and returns the oblique approximation/error, whereas the paper's CSSP figures report the smaller orthogonal selected-column projection error.

The workspace Python pipeline is unfinished: `cssp_pipeline.py`, `experiments/run_experiment.py`, `metrics/evaluation.py`, and `reconstruction/cssp_approximation.py` are empty. `osinsky.py` is currently invalid for ordinary shapes: it starts from `E=A` instead of `A-AVV^T` and subtracts an `m x remaining_rank` outer product from an `m x n` matrix. A direct 8 x 10, rank-3 smoke test raised a broadcast `ValueError`. The conventional `randomized_svd.py` may produce an approximate right basis, but it is not the paper's Corollary 2.6 procedure because it returns `k` vectors after a two-sided randomized SVD rather than sketching `A^T` and ARP-selecting `r+2` columns.

## Paper/code gaps and contradictions

1. The paper's DEIM test function uses a negative one-half power. `test_DEIM.m` constructs the positive square root instead. This is a material data-generation mismatch.
2. The paper says every randomized method was run 100 times; `test_CSS_DNA.m` uses `T=50`. The other scripts use 100.
3. Figure 2a's repository legend calls the Osinsky curve "Algorithm 2.2", but Osinsky is Algorithm 2.3 in the supplied v2. The DNA script now labels it "Osinsky".
4. Section 6.5 says ACA/full diagonal pivoting is included as a deterministic comparator. The smile/spiral scripts compute that curve but leave it out of the plots; the printed Figure 4 legend also omits it.
5. The authors' README is only two lines. There is no master runner, dependency list, toolbox note, or working-directory setup. Scripts rely on relative data paths and `randsample`.
6. All reported experiments use exact `V_opt` from a full SVD. They validate selection quality but do not experimentally validate the proposed cheap/sketched-subspace workflow or end-to-end speed advantage.

## Limitations

- No MATLAB/Octave runtime was available, so published figures were not numerically reproduced.
- The supplied v2 was the detailed methodological source. The later journal landing page was checked for publication metadata, but a full line-by-line v2-to-version-of-record comparison was outside scope.
- A temporary PDF reader was installed only in the user's temporary directory to extract the supplied PDF and was removed after extraction.

## Search and stop record

Primary sources reviewed: supplied arXiv v2 PDF; arXiv HTML; SIAM version-of-record metadata; authors' clean Git repository and commit history; Osinsky's original arXiv record. All consequential method, theorem, experiment, and local-code-status claims have direct primary support. Further broad searching would not materially change the requested paper/code understanding, so research stopped after targeted reconciliation of the DEIM equation, publication metadata, and Osinsky attribution.

## Claim-to-source ledger

- Cortinovis & Kressner, *Adaptive randomized pivoting for column subset selection, DEIM, and low-rank approximation*, arXiv v2, 19 June 2025, https://arxiv.org/html/2412.13992v2. Access: supplied PDF plus public HTML. Supports algorithm, theorems, experiments, limitations, and authors' stated novelty.
- Cortinovis & Kressner, version of record, SIAM Journal on Matrix Analysis and Applications 47(1), 2026, pp. 25-47, DOI 10.1137/24M1719189, https://doi.org/10.1137/24M1719189. Supports publication metadata and final abstract.
- Alice94/ARP, commit `a57b13e2560d75c6d06505464bfe5152dd774c12`, 19 June 2025, https://github.com/Alice94/ARP/tree/a57b13e2560d75c6d06505464bfe5152dd774c12. Access: clean local clone and public GitHub. Supports implementation and experiment-script claims.
- A. I. Osinsky, *Close to optimal column approximations with a single SVD*, arXiv:2308.09068, 2023, https://arxiv.org/abs/2308.09068. Supports the deterministic precursor's title and purpose.
- Local workspace files under `src/cssp`, read 3 September 2026. Supports the Python-port assessment and smoke-test findings; not an external source.
