"""E6: same numerical data, equivalent dense/CSR workflow, and honest timings."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

sparse=pytest.importorskip("scipy.sparse")
spec=importlib.util.spec_from_file_location("e6",Path(__file__).resolve().parents[1]/"experiments/run_sparsity.py")
e6=importlib.util.module_from_spec(spec)
spec.loader.exec_module(e6)


def test_nested_masks_and_same_values():
    values,order=e6.matrix_source(40,42,0)
    repeated=e6.matrix_source(40,42,0)
    np.testing.assert_array_equal(values,repeated[0]); np.testing.assert_array_equal(order,repeated[1])
    support=set(range(1600))
    for density in (1.,.5,.1,.01):
        A,csr=e6.matrix_at_density(values,order,density)
        assert csr.nnz==round(density*A.size)
        current=set(np.flatnonzero(A)); assert current<=support; support=current
        np.testing.assert_array_equal(A,csr.toarray())
        np.testing.assert_allclose(np.linalg.norm(A),1.)
        assert A.shape==(40,40)
        assert e6.storage_bytes(A)==12800
        assert e6.storage_bytes(csr)==csr.data.nbytes+csr.indices.nbytes+csr.indptr.nbytes
    assert e6.storage_bytes(csr)<e6.storage_bytes(A)
    _,full=e6.matrix_at_density(values,order,1.)
    assert e6.storage_bytes(full)>values.nbytes


@pytest.mark.parametrize("method",["arp","rejection_arp"])
def test_equivalent_dense_sparse_factors_and_errors(method):
    A,csr=e6.matrix_at_density(*e6.matrix_source(40,42,0),.1)
    k=3; optimal=float(np.linalg.norm(np.linalg.svd(A,compute_uv=False)[k:]))
    dense=e6.measure_case(A,method,k,13,29,optimal)
    sparse_result=e6.measure_case(csr,method,k,13,29,optimal)
    np.testing.assert_allclose(dense["relative_error"],sparse_result["relative_error"],rtol=1e-10)
    V=e6.construct_basis(A,k,13); Vs=e6.construct_basis(csr,k,13)
    np.testing.assert_allclose(V@V.T,Vs@Vs.T,atol=1e-12)
    J=e6.SELECTORS[method](V,random_state=29); X=e6.interpolation(V,J)
    expected=np.linalg.norm(A-A[:,J]@X,"fro")
    np.testing.assert_allclose(e6.factor_error(csr,csr[:,J],X,block_columns=7),expected,rtol=1e-12)
    np.testing.assert_allclose(dense["frobenius_error"],expected)
    assert dense["indices"]==sparse_result["indices"]
    assert json.loads(dense["indices"])==J.tolist()
    for row in (dense,sparse_result):
        assert row["algorithm_seconds"]==pytest.approx(sum(row[x] for x in e6.TIMINGS[:3]))


def test_error_evaluation_outside_timer(monkeypatch):
    clock=[0.]
    monkeypatch.setattr(e6,"perf_counter",lambda:clock[0])
    def basis(*args): clock[0]+=2; return np.eye(4)[:,:2]
    def select(*args,**kwargs): clock[0]+=3; return np.array([0,1])
    def interpolate(*args): clock[0]+=5; return np.eye(4)[:2]
    def error(*args): clock[0]+=100; return 1.
    monkeypatch.setattr(e6,"construct_basis",basis)
    monkeypatch.setattr(e6,"SELECTORS",{"arp":select})
    monkeypatch.setattr(e6,"interpolation",interpolate)
    monkeypatch.setattr(e6,"factor_error",error)
    row=e6.measure_case(np.eye(4),"arp",2,0,0,1.)
    assert [row[k] for k in e6.TIMINGS]==[2,3,5,10]
    assert clock[0]==110


def test_pairing_counts_warmups_and_seed_reuse(monkeypatch):
    args=e6.parse_args(["--size","40","--k","3","--instances","1","--trials","2","--no-plots"])
    A,csr=e6.matrix_at_density(*e6.matrix_source(40,42,0),.5)
    calls=[]
    def measure(A,method,k,bs,ss,optimal,evaluate=True):
        calls.append((method,bs,ss,evaluate))
        return dict(relative_error=1.,svd_optimality_ratio=1.,input_bytes=e6.storage_bytes(A),factor_bytes=100,
                    indices="[0, 1, 2]",**{metric:1. for metric in e6.TIMINGS})
    monkeypatch.setattr(e6,"measure_case",measure)
    rows=list(e6.trial_rows(A,csr,1.,args,0,.5))
    assert len(rows)==8 and len(calls)==12
    assert sum(not c[3] for c in calls)==4
    for trial in range(2):
        for method in e6.SELECTORS:
            group=[r for r in rows if r["trial"]==trial and r["method"]==method]
            assert len(group)==2
            assert group[0]["basis_seed"]==group[1]["basis_seed"]
            assert group[0]["selector_seed"]==group[1]["selector_seed"]
    pairs=e6.paired_rows(rows)
    assert len(pairs)==4 and all(r["errors_close"] for r in pairs)
    assert all(r["dense_over_sparse_time"]==1 for r in pairs)
    within,summary=e6.summaries(rows)
    assert len(summary)==4*len(e6.METRICS)
    assert all(r["instances"]==1 and np.isnan(r["between_instance_std"]) for r in summary)


@pytest.mark.parametrize("options",[["--densities","0"],["--densities","nan"],["--densities","1","1"],
                                    ["--size","5","--k","3"],["--trials","1"],["--run-name","../bad"]])
def test_invalid_cli(options):
    with pytest.raises(SystemExit): e6.parse_args(options)


def test_full_output_contract_and_no_overwrite(tmp_path,monkeypatch):
    from contextlib import contextmanager
    @contextmanager
    def control(*args): yield lambda:[]
    args=e6.parse_args(["--size","40","--k","3","--densities",".5",".1","--instances","1","--trials","2","--no-plots","--run-name","pilot"])
    monkeypatch.setattr(e6,"ROOT",tmp_path); monkeypatch.setattr(e6,"parse_args",lambda:args)
    monkeypatch.setattr(e6,"controlled_blas",control)
    e6.main(); out=tmp_path/"results/e6/pilot"
    cfg=json.loads((out/"config.json").read_text())
    assert cfg["status"]=="complete" and len(cfg["matrices"])==2
    with (out/"raw_results.csv").open() as stream: assert len(list(e6.csv.DictReader(stream)))==16
    assert (out/"paired_results.csv").exists()
    with pytest.raises(SystemExit,match="Run exists"): e6.main()
