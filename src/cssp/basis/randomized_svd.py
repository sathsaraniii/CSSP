import numpy as np


def randomized_svd(
    A: np.ndarray,
    k: int,
    p: int = 10,
    q: int = 0,
    random_state: int | None = None,
):
    """
    Compute a rank-k randomized SVD using Gaussian random projection.

    Parameters
    ----------
    A : np.ndarray
        Input matrix of shape (m, n).
    k : int
        Target rank.
    p : int, default=10
        Oversampling parameter.
    q : int, default=0
        Number of power iterations.
    random_state : int or None, default=None
        Seed for reproducibility.

    Returns
    -------
    U_k : np.ndarray
        Approximate left singular vectors, shape (m, k).
    S_k : np.ndarray
        Approximate singular values, shape (k,).
    Vt_k : np.ndarray
        Approximate right singular vectors, shape (k, n).
    """
    A = np.asarray(A, dtype=float)

    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix.")

    m, n = A.shape

    if not 1 <= k <= min(m, n):
        raise ValueError(
            f"k must satisfy 1 <= k <= min(m, n) = {min(m, n)}."
        )

    if p < 0:
        raise ValueError("Oversampling parameter p must be non-negative.")

    if q < 0:
        raise ValueError("Number of power iterations q must be non-negative.")

    # Number of random samples
    l = min(k + p, min(m, n))

    rng = np.random.default_rng(random_state)

    # Step 1: Generate Gaussian test matrix Ω
    Omega = rng.standard_normal((n, l))

    # Step 2: Form sample matrix Y = AΩ
    Y = A @ Omega

    # Step 3: Basic randomized range finder
    Q, _ = np.linalg.qr(Y, mode="reduced")

    # Optional power iteration
    for _ in range(q):
        Z = A.T @ Q
        Q, _ = np.linalg.qr(Z, mode="reduced")

        Y = A @ Q
        Q, _ = np.linalg.qr(Y, mode="reduced")

    # Step 4: Project A onto the approximate range
    B = Q.T @ A

    # Step 5: SVD of the small matrix
    Uh, S, Vt = np.linalg.svd(B, full_matrices=False)

    # Step 6: Recover approximate left singular vectors
    U = Q @ Uh

    # Truncate to desired rank k
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    return U_k, S_k, Vt_k