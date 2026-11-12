"""End-to-end run on real data; skipped unless MIRI_QPM_TEST_CONFIG points at a run config (TOML).

    MIRI_QPM_TEST_CONFIG=/path/qpm_s5365.toml MIRI_QPM_TEST_MODE=external pytest tests/test_integration.py

The output goes to pytest's tmp_path (the config's out_dir is ignored); only the first 2 exposures are used."""
import json
import os
try:
    import tomllib
except ImportError:
    import tomli as tomllib

import pytest

CFG = os.environ.get("MIRI_QPM_TEST_CONFIG")


@pytest.mark.skipif(not CFG, reason="set MIRI_QPM_TEST_CONFIG to a run config")
def test_pipeline_end_to_end(tmp_path):
    from jwst_gc_pipeline.miri_qpm.pipeline import run

    with open(CFG, "rb") as f:
        cfg = tomllib.load(f)
    cfg.update(out_dir=str(tmp_path), limit=2, grid="footprint", n_proc=2, quicklook=False)
    rep = run(cfg, os.environ.get("MIRI_QPM_TEST_MODE", "external"))
    assert rep["science_identity"]["bit_identical"]
    assert rep["n_exposures"] == 2 and rep["anchoring"]["mode"] == "off"
    assert all(v == 1.0 for v in rep["dataset_gain"]["gain_applied"].values())
    for p, row in rep["overlap"]["range_30_450"].items():
        if row["bias"] is not None:
            assert abs(row["bias"]) < 0.05, (p, row)
    saved = json.load(open(tmp_path / f"{cfg.get('name', 'qpm')}_qpm_report.json"))
    assert saved["recipe"]["id"] == "R1"
