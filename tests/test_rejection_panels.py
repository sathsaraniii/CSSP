import csv
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("panels", Path(__file__).resolve().parents[1] / "experiments/plot_rejection_panels.py")
panels = importlib.util.module_from_spec(spec)
spec.loader.exec_module(panels)


def fixture_run(tmp_path, ks):
    run = tmp_path / "run"
    run.mkdir()
    config = dict(experiment="E4 rejection", status="complete", ks=ks, repeats=3,
                  datasets_metadata={d: {"cssp_shape": [20, 100]} for d in ("dense", "sparse")})
    (run / "config.json").write_text(json.dumps(config))
    rows = [dict(dataset=d, mode=mode, method=method, k=k, repeats=3,
                 algorithm_seconds_p25=0.01, algorithm_seconds_median=0.02,
                 algorithm_seconds_p75=0.03)
            for d in ("dense", "sparse") for mode in panels.MODES for method in panels.METHODS for k in ks]
    with (run / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return run


@pytest.mark.parametrize("ks", [[10], [5, 10]])
def test_generate_and_preserve_sources(tmp_path, ks):
    pytest.importorskip("matplotlib")
    run = fixture_run(tmp_path, ks)
    before = (run / "summary.csv").read_bytes()
    out = panels.generate(run)
    assert len(list(out.glob("*.png"))) == 2
    assert before == (run / "summary.csv").read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        panels.generate(run)


def test_reject_missing_rows(tmp_path):
    run = fixture_run(tmp_path, [10])
    path = run / "summary.csv"
    path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
    with pytest.raises(ValueError, match="Missing"):
        panels.load_results(run)
