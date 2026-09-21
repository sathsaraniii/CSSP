# E7 — Repeated real-world CSSP evaluation with saved outputs.
"""Run four or six CSSP pipelines on the GSE10072 dataset.
    (DNA data, Shape: 22,283 x 107) """

import argparse
from pathlib import Path
import sys

import numpy as np



PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _real_data_experiment import add_arguments, validate_arguments, run_real_data  # noqa: E402
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


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return validate_arguments(parser, parser.parse_args(argv), default_k=10)


def main() -> None:
    args = parse_args()
    A, sample_ids = load_gse10072(DATASET_PATH)
    print(f"GSE10072: {A.shape}, {len(sample_ids)} tissue samples")
    run_real_data(A, args, "dna", DATASET_PATH, sample_ids=sample_ids,
                  preprocessing="Full GEO numeric table; no normalization or subsampling")


if __name__ == "__main__":
    main()
