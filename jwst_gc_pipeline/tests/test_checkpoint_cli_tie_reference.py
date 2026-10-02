"""The CLI entry point must not bypass `tie_through_reference_filter` (PR #988
review, blocking item 2).

`scripts/reduction/run_astrometry_checkpoint.py` is a SECOND production caller
of `run_visit_checkpoint`, independent of
`cataloging._run_astrometry_stage_checkpoint`. Before this file,
it called `run_visit_checkpoint(..., refcat=refcat, ...)` with the raw VIRAC2
refcat unconditionally -- so running it on an opted-in field's non-reference
filter (gc-treasury/10678 F480M) tied straight to VIRAC2 and, with --apply,
could write that bulk to the offsets table with no error, exactly the
failure `alignment_config.tie_through_reference_filter` exists to prevent.

These tests drive the real CLI module (loaded from its file path, same
pattern as ``test_checkpoint_cli_seed_flags.py``) and pin that it now routes
through ``astrometry_checkpoint.resolve_tie_reference``: an opted-in field
whose reference filter has not settled REFUSES (raises
``ReferenceFilterNotSettledError``) rather than silently tying to VIRAC2, and
a settled one hands ``run_visit_checkpoint`` the resolved (substituted)
reference, not the raw one.
"""
import importlib.util
import json
import pathlib

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry import astrometry_checkpoint as ac
from jwst_gc_pipeline.reduction import alignment_config as alignment_config_module
from jwst_gc_pipeline.reduction.alignment_config import (
    TABLE_CONSENSUS, VIRAC2, FieldAlignment)

_CLI = (pathlib.Path(__file__).parents[2] / 'scripts' / 'reduction'
        / 'run_astrometry_checkpoint.py')


def _load():
    spec = importlib.util.spec_from_file_location('_ckpt_cli_tie', _CLI)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def cli():
    return _load()


def _opted_in(monkeypatch, reference_filter="F212N"):
    cfg = FieldAlignment(
        proposal="TEST", fields=None, reference_frame=VIRAC2,
        source=TABLE_CONSENSUS, reference_filter=reference_filter,
        tie_through_reference_filter=True)
    monkeypatch.setattr(alignment_config_module, "resolve",
                        lambda proposal_id, field: cfg)


def _write_refcat(path):
    n = 20
    rng = np.random.default_rng(0)
    ra = 266.0 + rng.normal(scale=0.01, size=n)
    dec = -28.9 + rng.normal(scale=0.01, size=n)
    t = Table()
    t["skycoord"] = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    t["source"] = ["VIRAC2"] * (n - 2) + ["GaiaDR3"] * 2
    t.write(str(path))


def _write_perframe_catalog(path):
    t = Table({"x": [1.0, 2.0]})
    t.write(str(path))


def _write_consensus_fits(path, coords):
    t = Table()
    t["skycoord"] = coords
    t.meta["NVISITS"] = 2
    t.write(str(path))


def _write_m2_record(record_dir, filtername, consensus_catalog, passed=True):
    rec = dict(consensus_catalog=consensus_catalog, passed=passed,
              corrections=[], gate_override=None, date=ac._utcnow_iso(),
              visits=[])
    with open(record_dir / f"checkpoint_m2_{filtername}_latest.json", "w") as fh:
        json.dump(rec, fh)


# ---------------------------------------------------------------------------
# opted-in, NOT settled -> refuse, never tie to VIRAC2 silently
# ---------------------------------------------------------------------------

def test_opted_in_unsettled_reference_refuses(cli, tmp_path, monkeypatch):
    _opted_in(monkeypatch)
    refcat_path = tmp_path / "refcat.fits"
    _write_refcat(refcat_path)
    cat_path = tmp_path / "f480m_nrcblong_visit001_vgroup02101_exp00001_m2_daophot_basic.fits"
    _write_perframe_catalog(cat_path)
    # no checkpoint_m2_F212N_latest.json anywhere under basepath -> not settled

    with pytest.raises(ac.ReferenceFilterNotSettledError):
        cli.main(["--stage", "m2", "--filter", "F480M",
                  "--catalog-glob", str(cat_path),
                  "--refcat", str(refcat_path),
                  "--basepath", str(tmp_path),
                  "--proposal-id", "TEST", "--field", "001",
                  "--no-satstars"])


# ---------------------------------------------------------------------------
# opted-in, settled -> run_visit_checkpoint gets the RESOLVED reference
# ---------------------------------------------------------------------------

def test_opted_in_settled_reference_is_threaded_through(cli, tmp_path, monkeypatch):
    _opted_in(monkeypatch)
    refcat_path = tmp_path / "refcat.fits"
    _write_refcat(refcat_path)
    cat_path = tmp_path / "f480m_nrcblong_visit001_vgroup02101_exp00001_m2_daophot_basic.fits"
    _write_perframe_catalog(cat_path)

    record_dir = tmp_path / "astrometry_checkpoints"
    record_dir.mkdir()
    cons_path = tmp_path / "f212n_consensus.fits"
    n = 10
    coords = SkyCoord(ra=[266.0] * n * u.deg, dec=[-28.9] * n * u.deg,
                      frame="icrs")
    _write_consensus_fits(cons_path, coords)
    _write_m2_record(record_dir, "F212N", str(cons_path), passed=True)

    seen = {}

    def _fake_run_visit_checkpoint(tables, stage, refcat=None, **kw):
        seen["refcat"] = refcat
        return dict(passed=True, corrections=[], failures=[])

    monkeypatch.setattr(cli, "run_visit_checkpoint", _fake_run_visit_checkpoint)

    rc = cli.main(["--stage", "m2", "--filter", "F480M",
                  "--catalog-glob", str(cat_path),
                  "--refcat", str(refcat_path),
                  "--basepath", str(tmp_path),
                  "--proposal-id", "TEST", "--field", "001",
                  "--no-satstars"])
    assert rc == 0
    assert "refcat" in seen, "run_visit_checkpoint was never called"
    got = seen["refcat"]
    assert got is not None
    assert got["reference_kind"] == ac.REFERENCE_KIND_JWST_CONSENSUS
    assert got["reference_filter"] == "F212N"
    assert got["reference_path"] == str(cons_path)
    # Not the raw VIRAC2 refcat: its `all` is the F212N consensus, not the
    # refcat.fits coordinates written above.
    assert len(got["all"]) == n


def test_opted_out_field_still_gets_the_raw_refcat(cli, tmp_path, monkeypatch):
    """Backward compatibility: a field with no FieldAlignment entry (every
    pre-existing CLI invocation) must be unaffected."""
    monkeypatch.setattr(alignment_config_module, "resolve",
                        lambda proposal_id, field: None)
    refcat_path = tmp_path / "refcat.fits"
    _write_refcat(refcat_path)
    cat_path = tmp_path / "f212n_nrcblong_visit001_vgroup02101_exp00001_m2_daophot_basic.fits"
    _write_perframe_catalog(cat_path)

    seen = {}

    def _fake_run_visit_checkpoint(tables, stage, refcat=None, **kw):
        seen["refcat"] = refcat
        return dict(passed=True, corrections=[], failures=[])

    monkeypatch.setattr(cli, "run_visit_checkpoint", _fake_run_visit_checkpoint)

    rc = cli.main(["--stage", "m2", "--filter", "F212N",
                  "--catalog-glob", str(cat_path),
                  "--refcat", str(refcat_path),
                  "--basepath", str(tmp_path),
                  "--proposal-id", "9999", "--obsid", "001",
                  "--no-satstars"])
    assert rc == 0
    assert "reference_kind" not in seen["refcat"], (
        "an opted-out field must receive the RAW refcat dict unchanged, "
        "with no substitution provenance")
    assert len(seen["refcat"]["all"]) == 20, (
        "the raw refcat's 20 stars must pass through unchanged")
