"""Run four CSSP pipelines on a subset of Meszaros/large 
(Linear-programming problem, Shape: 4,282 x 8,617, sparse matrix) """

import argparse
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.sparse import csc_matrix, issparse

from cssp import run_four_pipelines, run_six_pipelines


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipelines",
        type=int,
        choices=(4, 6),
        default=4,
        help="run the original four pipelines or include two Rejection ARP pipelines",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    A_sparse = load_large_matrix()

    print("Original Meszaros/large matrix")
    print(f"Shape: {A_sparse.shape}")
    print(f"Nonzero values: {A_sparse.nnz}")
    print()

    A, original_column_indices = make_reproducible_submatrix(
        A_sparse,
        number_of_columns=400,
        random_state=42,
    )

    print("Experimental submatrix")
    print(f"Shape: {A.shape}")
    print(f"Finite values: {np.isfinite(A).all()}")
    print()

    k = 20

    run_pipelines = run_six_pipelines if args.pipelines == 6 else run_four_pipelines
    results = run_pipelines(
        A,
        k=k,
        random_state=42,
        oversampling=10,
        power_iterations=1,
    )

    header = (
        f"{'Pipeline':<27}"
        f"{'Relative error':>18}"
        f"{'Optimality':>15}"
        f"{'Algorithm time':>18}"
    )

    print(header)
    print("-" * len(header))

    for name, result in results.items():
        metrics = result["metrics"]
        timings = result["timings"]

        # Convert subset indices back to original large.mat indices.
        selected_original_columns = original_column_indices[
            result["indices"]
        ]

        print(
            f"{name:<27}"
            f"{metrics['relative_frobenius_error']:>18.6e}"
            f"{metrics['frobenius_optimality_ratio']:>15.6f}"
            f"{timings['algorithm_seconds']:>18.6f}"
        )

        print(
            "  Submatrix indices:",
            result["indices"].tolist(),
        )

        print(
            "  Original matrix indices:",
            selected_original_columns.tolist(),
        )


if __name__ == "__main__":
    main()
