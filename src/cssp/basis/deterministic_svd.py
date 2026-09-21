import numpy as np

from cssp._validation import as_real_matrix, validate_rank


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
    A = as_real_matrix(A)
    k = validate_rank(k, A.shape)

    # Deterministic SVD
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Keep the dominant k components
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]

    return U_k, S_k, Vt_k
