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
    """Load a finite, nonempty GEO matrix, stopping at its table-end marker."""
    with path.open("r", encoding="utf-8") as file:
        # Skip the GEO metadata before the matrix.
        for line in file:
            if line.startswith("!series_matrix_table_begin"):
                break
        else:
            raise ValueError("Matrix table was not found in the dataset.")

        # The next line contains ID_REF followed by sample identifiers.
        header_line = next(file, None)
        if header_line is None:
            raise ValueError("Matrix table header was not found in the dataset.")
        header = [value.strip().strip('"') for value in header_line.split("\t")]
        if len(header) < 2 or header[0] != "ID_REF" or not all(header[1:]):
            raise ValueError("Matrix table header must contain ID_REF and sample identifiers.")
        sample_ids = header[1:]

        def numeric_lines():
            # Returning stops iteration; filtering just the marker would also
            # feed any trailing GEO metadata into NumPy's numeric parser.
            found_row = False
            for line in file:
                if line.startswith("!series_matrix_table_end"):
                    if not found_row:
                        raise ValueError("Matrix table must contain at least one data row.")
                    return
                if not line.strip():
                    continue
                if len(line.rstrip("\r\n").split("\t")) != len(header):
                    raise ValueError("Matrix data row does not match the table header.")
                found_row = True
                yield line
            raise ValueError("Matrix table-end marker was not found in the dataset.")

        # Ignore the first column because it contains probe identifiers.
        A = np.loadtxt(
            numeric_lines(),
            delimiter="\t",
            usecols=range(1, len(header)),
            comments=None,
            ndmin=2,
        )

    if not np.isfinite(A).all():
        raise ValueError("Matrix table must contain only finite numeric values.")

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
