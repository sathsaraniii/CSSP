"""Small interactive comparison of exact and randomized SVD."""

from __future__ import annotations

import numpy as np

from cssp.basis.deterministic_svd import deterministic_svd
from cssp.basis.randomized_svd import randomized_svd


def run_experiment(
    m: int, n: int, k: int, p: int, q: int, seed: int
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))

    U_d, S_d, Vt_d = deterministic_svd(A, k)
    U_r, S_r, Vt_r = randomized_svd(
        A, k, p=p, q=q, random_state=seed
    )

    exact_error = float(
        np.linalg.norm(A - (U_d * S_d) @ Vt_d, ord="fro")
    )
    randomized_error = float(
        np.linalg.norm(A - (U_r * S_r) @ Vt_r, ord="fro")
    )
    return exact_error, randomized_error


def main() -> None:
    for q in (0, 1, 2):
        exact_error, randomized_error = run_experiment(
            m=500, n=300, k=20, p=10, q=q, seed=42
        )
        print(f"q = {q}")
        print(f"Deterministic: {exact_error:.6f}")
        print(f"Randomized:    {randomized_error:.6f}\n")


if __name__ == "__main__":
    main()
