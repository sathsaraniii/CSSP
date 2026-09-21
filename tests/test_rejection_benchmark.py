"""Reference-data generation, explicit column orientation, and sparse safety."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

sparse = pytest.importorskip("scipy.sparse")
spec = importlib.util.spec_from_file_location(
    "rejection_benchmark", Path(__file__).resolve().parents[1] / "experiments/run_rejection_benchmark.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


def test_sparse_generator_matches_reference_distribution():
    A = bench.sparse_gaussian(80, 12, 30, 42)
    assert sparse.isspmatrix_csc(A)
    assert A.nnz == 360
    np.testing.assert_array_equal(np.diff(A.indptr), 30)
    for j in range(12):
        assert len(set(A.indices[A.indptr[j]:A.indptr[j+1]])) == 30
    rng = np.random.default_rng(42)
    rows = np.concatenate([rng.choice(80, 30, replace=False) for _ in range(12)])
    vals = rng.standard_normal(360)
    expected = sparse.coo_matrix((vals / (rows + 1.)**2, (rows, np.repeat(np.arange(12), 30))), shape=(80, 12))
    np.testing.assert_array_equal(A.toarray(), expected.toarray())


@pytest.mark.parametrize("kind", ["dense", "sparse"])
def test_cache_and_column_orientation(tmp_path, kind):
    A, meta = bench.get_dataset(kind, (80, 12), 42, tmp_path)
    assert A.shape == (12, 80)
    assert meta["reference_shape"] == [80, 12]
    assert meta["cssp_shape"] == [12, 80]
    cached, repeated = bench.get_dataset(kind, (80, 12), 42, tmp_path)
    assert repeated == meta
    if kind == "dense":
        expected = np.random.default_rng(42).standard_normal((80, 12)) / np.arange(1, 81)[:, None]**2
        np.testing.assert_array_equal(A, expected.T)
        np.testing.assert_array_equal(A, cached)
    else:
        assert sparse.issparse(A)
        assert (A != cached).nnz == 0
    Path(meta["path"]).with_suffix(".json").write_text(json.dumps(dict(meta, status="generating")))
    with pytest.raises(RuntimeError, match="incomplete"):
        bench.get_dataset(kind, (80, 12), 42, tmp_path)


@pytest.mark.parametrize("k", [2, 4, 7, 10])
def test_sparse_stack_blocks(k):
    S = bench.sparse_stack(20, k, 13)
    zeta = min(4, k)
    assert S.shape == (20, k)
    np.testing.assert_array_equal(np.diff(S.indptr), zeta)
    np.testing.assert_allclose(np.asarray(S.multiply(S).sum(axis=1)), 1.)
    lengths = [k // zeta + (j < k % zeta) for j in range(zeta)]
    starts = np.r_[0, np.cumsum(lengths)]
    for row in range(20):
        indices = S.indices[S.indptr[row]:S.indptr[row+1]]
        assert all(starts[j] <= indices[j] < starts[j+1] for j in range(zeta))


def test_sparse_sketch_only_densifies_narrow_result(monkeypatch):
    A = bench.sparse_gaussian(80, 12, 30, 42).T
    k = 4
    dense = A.toarray()
    original = sparse.csr_matrix.toarray
    original_csc = sparse.csc_matrix.toarray
    def guarded(self, *args, **kwargs):
        assert self.shape == (80, k), "Attempted to densify original matrix"
        return original(self, *args, **kwargs)
    def guarded_csc(self, *args, **kwargs):
        assert self.shape == (80, k)
        return original_csc(self, *args, **kwargs)
    monkeypatch.setattr(sparse.csr_matrix, "toarray", guarded)
    monkeypatch.setattr(sparse.csc_matrix, "toarray", guarded_csc)
    V = bench.construct_basis(A, k, 4)
    direct, _ = np.linalg.qr((bench.sparse_stack(12, k, 4).T @ dense).T, mode="reduced")
    np.testing.assert_allclose(V @ V.T, direct @ direct.T, atol=1e-12)
    np.testing.assert_allclose(V.T @ V, np.eye(k), atol=1e-12)
    for select in bench.SELECTORS.values():
        J = select(V, random_state=3)
        X = bench.interpolation(V, J)
        assert X.shape == (k, 80)
        np.testing.assert_allclose(X[:, J], np.eye(k), atol=1e-12)
        np.testing.assert_allclose(X, np.linalg.solve(V[J].T, V.T), atol=1e-12)
        approx = dense[:, J] @ X
        row_W = np.linalg.solve(V[J].T, V.T).T
        np.testing.assert_allclose(approx.T, row_W @ dense.T[J, :], atol=1e-12)


def test_timing_and_factor_dimensions():
    A = bench.sparse_gaussian(80, 12, 30, 42).T
    for method in bench.SELECTORS:
        result = bench.timed_run(A, method, 4, 4, 3)
        assert json.loads(result["factor_shape"]) == [4, 80]
        assert len(set(json.loads(result["indices"]))) == 4
        assert result["algorithm_seconds"] == pytest.approx(sum(result[f"{stage}_seconds"] for stage in ("basis", "selection", "interpolation")))


def test_memory_check_and_overwrite_safety(tmp_path, monkeypatch):
    args = bench.parse_args(["--profile", "reference", "--memory-gib", "0.001", "--run-name", "skip", "--no-plots"])
    monkeypatch.setattr(bench, "ROOT", tmp_path)
    monkeypatch.setattr(bench, "parse_args", lambda: args)
    monkeypatch.setattr(bench, "get_dataset", lambda *args: pytest.fail("Must skip before generating"))
    from contextlib import contextmanager
    @contextmanager
    def control(*args):
        yield lambda: []
    monkeypatch.setattr(bench, "controlled_blas", control)
    bench.main()
    config = json.loads((tmp_path / "results/e4/rejection/skip/config.json").read_text())
    assert config["status"] == "complete_with_skips"
    assert len(config["skipped"]) == 2
    with pytest.raises(SystemExit, match="Run exists"):
        bench.main()
