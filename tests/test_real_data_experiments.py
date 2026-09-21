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
