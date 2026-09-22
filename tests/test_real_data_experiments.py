"""E7 extends the existing entry points without changing dataset loaders."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load(name):
    path=Path(__file__).resolve().parents[1]/"experiments"/f"{name}.py"
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


dna=load("run_dna_experiment")
common=__import__("_real_data_experiment")


@pytest.mark.parametrize("pipelines",[4,6])
def test_saved_runs_mappings_seeds_and_baseline(tmp_path,monkeypatch,pipelines):
    args=dna.parse_args(["--pipelines",str(pipelines),"--ks","2","3","--trials","2","--run-name","pilot","--no-plots"])
    monkeypatch.setattr(common,"ROOT",tmp_path)
    A=np.random.default_rng(4).normal(size=(9,7))
    mapping=np.arange(7)*3+10; labels=[f"sample_{i}" for i in range(7)]
    out=common.run_real_data(A,args,"fixture","fixture.txt",mapping,labels)
    cfg=json.loads((out/"config.json").read_text())
    assert cfg["status"]=="complete" and cfg["original_column_indices"]==mapping.tolist()
    with (out/"raw_results.csv").open() as stream: rows=list(common.csv.DictReader(stream))
    assert len(rows)==pipelines*4
    assert {int(r["seed"]) for r in rows}=={42,43}
    s=np.linalg.svd(A,compute_uv=False)
    for r in rows:
        J=json.loads(r["indices"]); k=int(r["k"])
        assert json.loads(r["original_indices"])==mapping[J].tolist()
        assert json.loads(r["selected_samples"])==[labels[j] for j in J]
        np.testing.assert_allclose(float(r["optimal_rank_k_error"]),np.linalg.norm(s[k:]))
        np.testing.assert_allclose(float(r["algorithm_seconds"]),sum(float(r[x]) for x in common.TIMINGS[:3]))
    with pytest.raises(SystemExit,match="already exists"):
        common.run_real_data(A,args,"fixture","fixture.txt",mapping,labels)


def test_defaults_and_large_cli():
    pytest.importorskip("scipy")
    large=load("run_large_experiment")
    assert dna.parse_args([]).ks==[10]
    args=large.parse_args([])
    assert args.ks==[20] and args.columns==400 and args.trials==10
    assert large.parse_args(["--columns","60","--k","5","--data-seed","7"]).data_seed==7
    with pytest.raises(SystemExit): large.parse_args(["--columns","0"])


@pytest.mark.parametrize("argv",[["--trials","0"],["--ks","2","2"],["--k","0"],
                                 ["--seed","-1"],["--k","2","--ks","3"],["--run-name","../bad"]])
def test_invalid_cli(argv):
    with pytest.raises(SystemExit): dna.parse_args(argv)


def test_zero_baseline_and_single_trial(tmp_path,monkeypatch):
    monkeypatch.setattr(common,"ROOT",tmp_path)
    args=dna.parse_args(["--k","2","--trials","1","--run-name","zero","--no-plots"])
    out=common.run_real_data(np.eye(2),args,"fixture","fixture")
    with (out/"summary.csv").open() as stream: rows=list(common.csv.DictReader(stream))
    assert all(np.isnan(float(r["frobenius_optimality_ratio_mean"])) for r in rows)
    assert all(np.isnan(float(r["relative_frobenius_error_std"])) for r in rows)


def test_failure_keeps_partial_output(tmp_path,monkeypatch):
    monkeypatch.setattr(common,"ROOT",tmp_path)
    real=common.run_four_pipelines
    calls=[]
    def failing(*args,**kwargs):
        calls.append(1)
        if len(calls)>1: raise RuntimeError("test interruption")
        return real(*args,**kwargs)
    monkeypatch.setattr(common,"run_four_pipelines",failing)
    args=dna.parse_args(["--k","2","--trials","2","--run-name","partial","--no-plots"])
    with pytest.raises(RuntimeError,match="test interruption"):
        common.run_real_data(np.eye(3),args,"fixture","fixture")
    out=tmp_path/"results/e7/fixture/partial"
    assert json.loads((out/"config.json").read_text())["status"]=="incomplete"
    with (out/"raw_results.csv").open() as stream: assert len(list(common.csv.DictReader(stream)))==4


def test_multi_subset_run(tmp_path):
    sparse = pytest.importorskip("scipy.sparse")
    large = load("run_large_experiment")
    multi = __import__("_meszaros_subsets")
    args = large.parse_args(["--pipelines", "6", "--data-seeds", "42", "43", "--columns", "6",
                             "--k", "2", "--trials", "2", "--no-plots", "--run-name", "multi"])
    matrix = sparse.csc_matrix(np.random.default_rng(8).normal(size=(9, 10)))
    out = multi.run_subsets(matrix, args, tmp_path, "fixture", large.make_reproducible_submatrix)
    config = json.loads((out / "config.json").read_text())
    assert config["status"] == "complete" and config["completed_subsets"] == [42, 43]
    with (out / "raw_results.csv").open() as stream: rows = list(common.csv.DictReader(stream))
    assert len(rows) == 24
    with (out / "selection_agreement.csv").open() as stream: pairs = list(common.csv.DictReader(stream))
    assert len(pairs) == 60
    with (out / "subset_summary.csv").open() as stream: within = list(common.csv.DictReader(stream))
    with (out / "summary.csv").open() as stream: overall = list(common.csv.DictReader(stream))
    r = next(r for r in overall if r["method"] == "deterministic_osinsky" and r["metric"] == "relative_frobenius_error")
    values = [float(r["relative_frobenius_error_mean"]) for r in within if r["method"] == "deterministic_osinsky"]
    np.testing.assert_allclose(float(r["mean_of_subset_centers"]), np.mean(values))
    np.testing.assert_allclose(float(r["between_subset_std"]), np.std(values, ddof=1))
    with pytest.raises(SystemExit, match="Run exists"):
        multi.run_subsets(matrix, args, tmp_path, "fixture", large.make_reproducible_submatrix)


def test_agreement_ignores_order():
    multi = __import__("_meszaros_subsets")
    rows = [dict(subset_seed=42,k=2,trial=0,method=m,original_indices=json.dumps(j))
            for m,j in [("a",[2,7]),("b",[7,2]),("c",[2,8])]]
    _, pairs, summary = multi.aggregate([], rows)
    assert [r["same_column_set"] for r in pairs] == [True, False, False]
    assert summary[0]["same_set_fraction"] == 1


@pytest.mark.parametrize("argv", [["--data-seeds", "42", "42"], ["--data-seeds", "-1"],
                                  ["--data-seed", "42", "--data-seeds", "43"]])
def test_subset_seed_validation(argv):
    large = load("run_large_experiment")
    with pytest.raises(SystemExit): large.parse_args(argv)


def test_multi_subset_failure(tmp_path):
    large = load("run_large_experiment")
    multi = __import__("_meszaros_subsets")
    args = large.parse_args(["--data-seeds", "42", "43", "--run-name", "failed", "--no-plots"])
    def fail(*args, **kwargs): raise RuntimeError("bad subset")
    with pytest.raises(RuntimeError, match="bad subset"):
        multi.run_subsets(None, args, tmp_path, "fixture", fail)
    assert json.loads((tmp_path / "results/e7/meszaros/failed/config.json").read_text())["status"] == "incomplete"
