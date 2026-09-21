"""E5 controlled matrix structure, trial design, and two-level summaries."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "e5", Path(__file__).resolve().parents[1] / "experiments/run_matrix_structure.py"
)
e5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e5)


def args():
    return e5.parse_args(["--size", "30", "--k", "3", "--signal-rank", "6", "--instances", "1", "--trials", "2", "--repeats", "2", "--no-plots"])


def test_controlled_structures():
    components = e5.matrix_components(30, 6, 42, 0)
    for first, repeated in zip(components, e5.matrix_components(30, 6, 42, 0)):
        np.testing.assert_array_equal(first, repeated)
    for family in e5.FAMILIES:
        A = e5.make_family(family, components)
        assert A.shape == (30, 30)
        assert np.isfinite(A).all()
        np.testing.assert_allclose(np.linalg.norm(A), 1.)
    assert np.linalg.matrix_rank(e5.make_family("exact_low_rank", components)) == 6
    for family, spectrum in (("rapid_decay", np.geomspace(1., 1e-8, 30)), ("slow_decay", 1 / np.arange(1,31))):
        np.testing.assert_allclose(np.linalg.svd(e5.make_family(family, components), compute_uv=False), spectrum / np.linalg.norm(spectrum), atol=1e-14)
    signal, noise = components[2:4]
    np.testing.assert_allclose(np.linalg.norm(noise) / np.linalg.norm(signal), 1.)
    for family, fraction in (("small_noise", .01), ("large_noise", .5)):
        expected = signal + fraction * noise
        np.testing.assert_allclose(e5.make_family(family, components), expected / np.linalg.norm(expected))


def test_accuracy_trials_and_projection():
    options = args()
    A = e5.make_family("small_noise", e5.matrix_components(30,6,42,0))
    _, s, Vt = np.linalg.svd(A, full_matrices=False)
    rows = list(e5.accuracy_rows(A,s,Vt,options,0,"small_noise"))
    assert len(rows) == 11
    assert rows == list(e5.accuracy_rows(A,s,Vt,options,0,"small_noise"))
    assert len([r for r in rows if r["method"] == "deterministic_osinsky"]) == 1
    for r in rows:
        J = e5.json.loads(r["indices"])
        Q, _ = np.linalg.qr(A[:,J], mode="reduced")
        np.testing.assert_allclose(r["relative_error"],np.linalg.norm(A-Q@Q.T@A)/np.linalg.norm(A))
        assert r["svd_optimality_ratio"] >= 1 - 1e-12
    for trial in range(2):
        group = [r for r in rows if r["method"].startswith("randomized") and r["trial"] == trial]
        assert len({r["basis_seed"] for r in group}) == 1


def test_runtime_warmups_and_counts(monkeypatch):
    calls = []
    def fake(A, method, options, bs, ss):
        calls.append((method,bs,ss))
        return dict(method=method,**{metric:1. for metric in e5.TIMINGS})
    monkeypatch.setattr(e5,"measure",fake)
    rows = list(e5.runtime_rows(np.eye(30),args(),0,"gaussian"))
    assert len(rows)==12 and len(calls)==18
    for trial in range(2):
        group = [r for r in rows if r["trial"]==trial]
        assert {r["method"] for r in group} == set(e5.SIX_PIPELINE_CONFIGS)
        assert [r["execution_position"] for r in group] == list(range(6))


def test_instances_are_equally_weighted():
    rows = []
    for instance, values in ((0,[1.,3.]), (1,[10.])):
        for value in values:
            rows.append(dict(family="gaussian",instance=instance,mode="accuracy",method="arp",**{metric:value for metric in e5.ACCURACY}))
    within, summary = e5.summarize(rows)
    assert len(within)==6
    assert all(r["mean_of_instance_centers"]==6. for r in summary)
    assert all(r["instances"]==2 and r["total_runs"]==3 for r in summary)
    np.testing.assert_allclose(summary[0]["between_instance_std"],np.std([2,10],ddof=1))
    _, single = e5.summarize(rows[:2])
    assert np.isnan(single[0]["between_instance_std"])


@pytest.mark.parametrize("options", [["--k","25"],["--instances","0"],["--trials","1"],["--run-name","../bad"],["--families","gaussian","gaussian"]])
def test_invalid_cli(options):
    with pytest.raises(SystemExit):
        e5.parse_args(options)


def test_output_and_no_overwrite(tmp_path,monkeypatch):
    from contextlib import contextmanager
    @contextmanager
    def threads(*args):
        yield lambda: []
    options = args()
    options.families = ["gaussian"]
    options.run_name = "pilot"
    monkeypatch.setattr(e5,"ROOT",tmp_path)
    monkeypatch.setattr(e5,"parse_args",lambda:options)
    monkeypatch.setattr(e5,"controlled_blas",threads)
    e5.main()
    out = tmp_path / "results/e5/pilot"
    assert e5.json.loads((out/"config.json").read_text())["status"]=="complete"
    with (out/"raw_results.csv").open() as stream:
        assert len(list(e5.csv.DictReader(stream)))==23
    assert (out/"instance_summary.csv").exists()
    with pytest.raises(SystemExit, match="Run exists"):
        e5.main()
