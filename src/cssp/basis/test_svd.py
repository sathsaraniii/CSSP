import numpy as np

from deterministic_svd import deterministic_svd
from randomized_svd import randomized_svd


def run_experiment(m, n, k, p, q, seed):
    rng = np.random.default_rng(seed)

    A = rng.standard_normal((m, n))

    U_d, S_d, Vt_d = deterministic_svd(A, k)

    U_r, S_r, Vt_r = randomized_svd(
        A,
        k,
        p=p,
        q=q,
        random_state=seed,
    )

    A_d = U_d @ np.diag(S_d) @ Vt_d
    A_r = U_r @ np.diag(S_r) @ Vt_r

    error_d = np.linalg.norm(A - A_d, ord="fro")
    error_r = np.linalg.norm(A - A_r, ord="fro")

    return error_d, error_r


for q in [0, 1, 2]:
    error_d, error_r = run_experiment(
        m=500,
        n=300,
        k=20,
        p=10,
        q=q,
        seed=42,
    )

    print(f"q = {q}")
    print(f"Deterministic: {error_d:.6f}")
    print(f"Randomized:    {error_r:.6f}")
    print()