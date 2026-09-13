"""Run four or six CSSP pipelines on the GSE10072 dataset."""

import argparse
from pathlib import Path

import numpy as np

from cssp import run_four_pipelines, run_six_pipelines


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "DNA_matrix.txt"
)


def load_gse10072(path: Path) -> tuple[np.ndarray, list[str]]:
    """Load the numeric expression matrix and sample identifiers."""
    with path.open("r", encoding="utf-8") as file:
        # Skip the GEO metadata before the matrix.
        for line in file:
            if line.startswith("!series_matrix_table_begin"):
                break
        else:
            raise ValueError("Matrix table was not found in the dataset.")

        # The next line contains ID_REF followed by sample identifiers.
        header = next(file).rstrip().split("\t")
        sample_ids = [
            value.strip('"')
            for value in header[1:]
        ]

        # Stop before the GEO table-end marker.
        numeric_lines = (
            line
            for line in file
            if not line.startswith("!series_matrix_table_end")
        )

        # Ignore the first column because it contains probe identifiers.
        A = np.loadtxt(
            numeric_lines,
            delimiter="\t",
            usecols=range(1, len(header)),
        )

    return A, sample_ids


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
    A, sample_ids = load_gse10072(DATASET_PATH)

    print(f"Dataset: GSE10072")
    print(f"Matrix shape: {A.shape}")
    print(f"Finite values: {np.isfinite(A).all()}")
    print(f"Number of tissue samples: {len(sample_ids)}")
    print()

    run_pipelines = run_six_pipelines if args.pipelines == 6 else run_four_pipelines
    results = run_pipelines(
        A,
        k=10,
        random_state=42,
        oversampling=10,
        power_iterations=1,
    )

    header = (
        f"{'Pipeline':<27}"
        f"{'Relative error':>18}"
        f"{'Optimality':>15}"
        f"{'Basis time':>15}"
        f"{'Selection time':>18}"
    )
    print(header)
    print("-" * len(header))

    for name, result in results.items():
        metrics = result["metrics"]
        timings = result["timings"]

        print(
            f"{name:<27}"
            f"{metrics['relative_frobenius_error']:>18.6e}"
            f"{metrics['frobenius_optimality_ratio']:>15.6f}"
            f"{timings['basis_seconds']:>15.6f}"
            f"{timings['selection_seconds']:>18.6f}"
        )

        selected_samples = [
            sample_ids[index]
            for index in result["indices"]
        ]

        print(f"  Indices: {result['indices'].tolist()}")
        print(f"  Samples: {selected_samples}")


if __name__ == "__main__":
    main()
