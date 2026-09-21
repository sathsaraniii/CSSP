# E7 — Repeated real-world CSSP evaluation with saved outputs.
"""Run four or six CSSP pipelines on a subset of Meszaros/large
(Linear-programming problem, Shape: 4,282 x 8,617, sparse matrix) """

import argparse
from pathlib import Path
import sys

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix, issparse



PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _real_data_experiment import add_arguments, validate_arguments, run_real_data  # noqa: E402
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "large.mat"
)


def load_large_matrix():
    """Load Problem.A from the authors' MATLAB dataset."""
    matlab_data = loadmat(
        DATASET_PATH,
        squeeze_me=True,
        struct_as_record=False,
    )

    problem = matlab_data["Problem"]
    A_sparse = problem.A.tocsc()

    return A_sparse


def make_reproducible_submatrix(
    A_sparse,
    number_of_columns: int = 400,
    random_state: int = 42,
):
    """Choose nonzero columns and remove rows empty in that subset."""
    if (
        isinstance(number_of_columns, (bool, np.bool_))
        or not isinstance(number_of_columns, (int, np.integer))
        or number_of_columns < 1
    ):
        raise ValueError("number_of_columns must be a positive integer.")
    if not issparse(A_sparse) or A_sparse.ndim != 2:
        raise ValueError("A_sparse must be a two-dimensional sparse matrix.")

    # Canonicalize a copy so cancelling duplicate entries and explicit stored
    # zeros do not count as nonzeros, without modifying the caller's matrix.
    A_sparse = csc_matrix(A_sparse, copy=True)
    if np.iscomplexobj(A_sparse) or not np.isfinite(A_sparse.data).all():
        raise ValueError("A_sparse must contain finite real values.")
    A_sparse.sum_duplicates()
    A_sparse.eliminate_zeros()

    # Find columns containing at least one nonzero value.
    # count_nonzero(axis=...) requires SciPy 1.15; this also supports the
    # project's declared SciPy >= 1.10 dependency.
    nonzero_column_mask = np.asarray(
        (A_sparse != 0).getnnz(axis=0)
    ).ravel() > 0

    original_nonzero_columns = np.flatnonzero(
        nonzero_column_mask
    )
    if original_nonzero_columns.size == 0:
        raise ValueError("A_sparse must contain at least one nonzero column.")

    A_nonzero = A_sparse[:, nonzero_column_mask]

    # Reproducibly sample columns from the original sparse matrix.
    rng = np.random.default_rng(random_state)

    sample_size = min(
        number_of_columns,
        A_nonzero.shape[1],
    )

    sampled_local_columns = np.sort(
        rng.choice(
            A_nonzero.shape[1],
            size=sample_size,
            replace=False,
        )
    )

    A_subset = A_nonzero[:, sampled_local_columns]

    # Remove rows that are completely zero in this particular subset.
    nonzero_row_mask = np.asarray(
        (A_subset != 0).getnnz(axis=1)
    ).ravel() > 0

    A_subset = A_subset[nonzero_row_mask, :]

    # This sampled matrix is small enough for the current dense pipeline.
    A_dense = A_subset.toarray().astype(float)

    # Map local subset indices back to columns in the original matrix.
    original_column_indices = original_nonzero_columns[
        sampled_local_columns
    ]

    return A_dense, original_column_indices


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--columns", type=int, default=400,
                        help="number of nonzero columns sampled once")
    parser.add_argument("--data-seed", type=int, default=42,
                        help="fixed subset seed, independent of algorithm trials")
    args = validate_arguments(parser, parser.parse_args(argv), default_k=20)
    if args.columns < 1 or args.data_seed < 0:
        parser.error("columns must be positive and data-seed nonnegative")
    return args


def main() -> None:
    args = parse_args()
    A_sparse = load_large_matrix()
    A, original_column_indices = make_reproducible_submatrix(
        A_sparse, number_of_columns=args.columns, random_state=args.data_seed,
    )
    print(f"Meszaros/large: original {A_sparse.shape}, dense subset {A.shape}")
    run_real_data(
        A, args, "meszaros", DATASET_PATH, original_indices=original_column_indices,
        preprocessing=dict(
            original_shape=list(A_sparse.shape), original_nnz=int(A_sparse.nnz),
            requested_columns=args.columns, data_seed=args.data_seed,
            operations="sample nonzero columns once; remove zero rows; convert subset to dense; no normalization",
        ),
    )


if __name__ == "__main__":
    main()
