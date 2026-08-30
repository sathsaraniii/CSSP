import numpy as np

def deterministic_svd(A: np.ndarray, k: int):
    """
    Compute a deterministic rank-k truncated SVD.

    Parameters
    ----------
    A : np.ndarray
        Input matrix of shape (m, n).
    k : int
        Target rank.

    Returns
    -------
    U_k : np.ndarray
        First k left singular vectors, shape (m, k).
    S_k : np.ndarray
        First k singular values, shape (k,).
    Vt_k : np.ndarray
        First k rows of V^T, shape (k, n).
    """
    A = np.asarray(A, dtype=float)

    if A.ndim != 2:
        raise ValueError("A must be a 2D matrix.")

    m, n = A.shape

    if not 1 <= k <= min(m, n):
        raise ValueError(
            f"k must satisfy 1 <= k <= min(m, n) = {min(m, n)}."
        )

    # Deterministic SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Keep the dominant k components
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    return U_k, S_k, Vt_k