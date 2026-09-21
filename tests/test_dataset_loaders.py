"""Small, local fixtures covering GEO parsing and sparse subset selection.

These check that experiment data is interpreted correctly:
- GEO gene-expression table parsing.
- Single-row and single-column inputs.
- Malformed headers, missing markers, and invalid values.
- Sparse matrices with explicitly stored zeros or duplicate entries.
- Removing zero columns correctly.
- Reproducible column sampling.
- Mapping sampled columns back to their original indices.
- Compatibility with an older SciPy sparse API."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _load_experiment(name):
    path = Path(__file__).resolve().parents[1] / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


load_gse10072 = _load_experiment("run_dna_experiment").load_gse10072


def _write_geo(tmp_path, header, rows, *, end=True, trailer=""):
    path = tmp_path / "matrix.txt"
    text = "!Series_title\tfixture\n!series_matrix_table_begin\n"
    text += header + "\n" + "\n".join(rows) + "\n"
    if end:
        text += "!series_matrix_table_end\n"
    path.write_text(text + trailer, encoding="utf-8")
    return path


@pytest.mark.parametrize("trailer", ["", "!Additional_metadata\tnot numeric\n"])
def test_geo_reads_matrix_and_stops_at_end_marker(tmp_path, trailer):
    path = _write_geo(
        tmp_path,
        '"ID_REF"\t"sample_1"\t"sample_2"',
        ['"probe_1"\t1\t2', '"probe_2"\t3\t4'],
        trailer=trailer,
    )

    matrix, sample_ids = load_gse10072(path)

    np.testing.assert_array_equal(matrix, [[1, 2], [3, 4]])
    assert sample_ids == ["sample_1", "sample_2"]


@pytest.mark.parametrize(
    ("header", "rows", "expected"),
    [
        ("ID_REF\ts1\ts2", ["p1\t1\t2"], [[1, 2]]),
        ("ID_REF\ts1", ["p1\t1", "p2\t2"], [[1], [2]]),
        ("ID_REF\ts1", ["p1\t1"], [[1]]),
    ],
)
def test_geo_preserves_two_dimensions(tmp_path, header, rows, expected):
    matrix, _ = load_gse10072(_write_geo(tmp_path, header, rows))

    assert matrix.ndim == 2
    np.testing.assert_array_equal(matrix, expected)


def test_geo_requires_end_marker(tmp_path):
    path = _write_geo(tmp_path, "ID_REF\ts1", ["p1\t1"], end=False)

    with pytest.raises(ValueError, match="table-end marker"):
        load_gse10072(path)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("metadata only\n", "Matrix table was not found"),
        ("!series_matrix_table_begin\n", "header was not found"),
    ],
)
def test_geo_rejects_missing_table_or_header(tmp_path, contents, message):
    path = tmp_path / "matrix.txt"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_gse10072(path)


@pytest.mark.parametrize("header", ["ID_REF", "WRONG\ts1", '"ID_REF"\t""'])
def test_geo_rejects_unusable_headers(tmp_path, header):
    path = _write_geo(tmp_path, header, [])

    with pytest.raises(ValueError, match="header must contain"):
        load_gse10072(path)


def test_geo_rejects_empty_matrix(tmp_path):
    path = _write_geo(tmp_path, "ID_REF\ts1", [])

    with pytest.raises(ValueError, match="at least one data row"):
        load_gse10072(path)


@pytest.mark.parametrize("row", ["p1\t1\t2", "p1"])
def test_geo_rejects_row_width_mismatch(tmp_path, row):
    path = _write_geo(tmp_path, "ID_REF\ts1", [row])

    with pytest.raises(ValueError, match="does not match"):
        load_gse10072(path)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_geo_rejects_nonfinite_matrix(tmp_path, value):
    path = _write_geo(tmp_path, "ID_REF\ts1", [f"p1\t{value}"])

    with pytest.raises(ValueError, match="finite"):
        load_gse10072(path)


@pytest.fixture
def sparse_loader():
    sparse = pytest.importorskip("scipy.sparse")
    make_subset = _load_experiment("run_large_experiment").make_reproducible_submatrix
    return sparse, make_subset


def test_sparse_subset_filters_true_nonzeros_without_modifying_input(sparse_loader):
    sparse, make_subset = sparse_loader
    # Column 0 contains only an explicit zero; column 1's duplicates cancel.
    # Row 1 contains only a stored zero in the retained column 2.
    matrix = sparse.csc_matrix(
        (
            np.array([0.0, 2.0, -2.0, 3.0, 0.0, 4.0, 5.0]),
            np.array([0, 0, 0, 0, 1, 2, 3]),
            np.array([0, 1, 3, 6, 7, 7]),
        ),
        shape=(4, 5),
    )
    before = (matrix.data.copy(), matrix.indices.copy(), matrix.indptr.copy())

    subset, original_columns = make_subset(matrix, number_of_columns=10)

    np.testing.assert_array_equal(original_columns, [2, 3])
    np.testing.assert_array_equal(subset, [[3, 0], [4, 0], [0, 5]])
    for actual, expected in zip((matrix.data, matrix.indices, matrix.indptr), before):
        np.testing.assert_array_equal(actual, expected)


def test_sparse_subset_reproducibility_and_column_mapping(sparse_loader):
    sparse, make_subset = sparse_loader
    dense = np.diag([1, 0, 2, 3, 4, 5])
    matrix = sparse.csc_matrix(dense)

    first, columns = make_subset(matrix, number_of_columns=3, random_state=9)
    second, columns_again = make_subset(matrix, number_of_columns=3, random_state=9)

    expected = dense[:, columns]
    expected = expected[np.any(expected != 0, axis=1)]
    assert len(columns) == 3
    assert np.all(np.diff(columns) > 0)
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(columns, columns_again)


def test_sparse_subset_does_not_require_new_count_nonzero_axis_api(
    sparse_loader, monkeypatch
):
    sparse, make_subset = sparse_loader
    original_count_nonzero = sparse.csc_matrix.count_nonzero

    # Before SciPy 1.15 this method did not accept an axis argument.
    def legacy_count_nonzero(self):
        return original_count_nonzero(self)

    monkeypatch.setattr(sparse.csc_matrix, "count_nonzero", legacy_count_nonzero)

    subset, columns = make_subset(sparse.csc_matrix([[0, 2], [0, 0]]))

    np.testing.assert_array_equal(subset, [[2]])
    np.testing.assert_array_equal(columns, [1])


@pytest.mark.parametrize("count", [0, -1, 1.5, True, np.bool_(True)])
def test_sparse_subset_rejects_invalid_sample_size(sparse_loader, count):
    sparse, make_subset = sparse_loader

    with pytest.raises(ValueError, match="positive integer"):
        make_subset(sparse.eye(3, format="csc"), number_of_columns=count)


def test_sparse_subset_rejects_all_zero_matrix(sparse_loader):
    sparse, make_subset = sparse_loader

    with pytest.raises(ValueError, match="nonzero column"):
        make_subset(sparse.csc_matrix((3, 4)))


@pytest.mark.parametrize("value", [np.nan, np.inf, 1j])
def test_sparse_subset_rejects_nonfinite_or_complex_input(sparse_loader, value):
    sparse, make_subset = sparse_loader

    with pytest.raises(ValueError, match="finite real"):
        make_subset(sparse.csc_matrix([[value]]))
