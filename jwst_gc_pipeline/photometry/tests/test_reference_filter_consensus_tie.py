"""Tests for tying a non-reference filter through its field's reference
filter's own JWST consensus catalog instead of straight to VIRAC2
(``alignment_config.tie_through_reference_filter``, ``astrometry_checkpoint.
resolve_tie_reference`` / ``reference_filter_tie_settled``).

Motivating failure (gc-treasury 10678, o063): F212N and F480M each tied to
VIRAC2 independently.  F212N's tie was refused (no coherent dense peak);
F480M's own tie was applied.  The two bands landed 226 mas apart on the same
sky.  Tying F480M through F212N's consensus instead means the two bands agree
whether or not F212N's own VIRAC2 tie was itself applied.
"""
import json
import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry import astrometry_checkpoint as ac
from jwst_gc_pipeline.reduction import alignment_config as alignment_config_module
from jwst_gc_pipeline.reduction.alignment_config import (
    TABLE_CONSENSUS, VIRAC2, FieldAlignment)

from .test_visit_consensus import COSD, DEC0, RA0, _exposure_table, _field, _reference_sets

REC_NAME = "checkpoint_m2_F212N_latest.json"


def _opted_in(monkeypatch, reference_filter="F212N", tie_through=True):
    cfg = FieldAlignment(
        proposal="TEST", fields=None, reference_frame=VIRAC2,
        source=TABLE_CONSENSUS, reference_filter=reference_filter,
        tie_through_reference_filter=tie_through)
    monkeypatch.setattr(alignment_config_module, "resolve",
                        lambda proposal_id, field: cfg)
    return cfg


def _opted_out(monkeypatch):
    monkeypatch.setattr(alignment_config_module, "resolve",
                        lambda proposal_id, field: None)


def _write_consensus_fits(path, coords):
    tbl = Table()
    tbl["skycoord"] = coords
    tbl.meta["NVISITS"] = 2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tbl.write(path, format="fits", overwrite=True)


def _write_m2_record(record_dir, filtername, consensus_catalog, passed=True,
                     corrections=None, gate_override=None, date=None):
    rec = dict(consensus_catalog=consensus_catalog,
              passed=passed, corrections=corrections or [],
              gate_override=gate_override,
              date=date or ac._utcnow_iso(), visits=[])
    with open(os.path.join(record_dir, f"checkpoint_m2_{filtername}_latest.json"),
              "w") as fh:
        json.dump(rec, fh)


# ---------------------------------------------------------------------------
# resolve_tie_reference -- passthrough cases
# ---------------------------------------------------------------------------

def test_passthrough_refcat_none(monkeypatch, tmp_path):
    _opted_in(monkeypatch)
    assert ac.resolve_tie_reference(
        None, str(tmp_path), "TEST", None, "F480M") is None


def test_passthrough_when_not_opted_in(monkeypatch, tmp_path):
    _opted_out(monkeypatch)
    refcat = dict(all="sentinel-all", sparse="sentinel-sparse", mag=None)
    out = ac.resolve_tie_reference(refcat, str(tmp_path), "TEST", None, "F480M", record_dir=str(tmp_path))
    assert out is refcat


def test_passthrough_when_flag_false(monkeypatch, tmp_path):
    _opted_in(monkeypatch, tie_through=False)
    refcat = dict(all="sentinel-all", sparse="sentinel-sparse", mag=None)
    out = ac.resolve_tie_reference(refcat, str(tmp_path), "TEST", None, "F480M", record_dir=str(tmp_path))
    assert out is refcat


def test_passthrough_for_the_reference_filter_itself(monkeypatch, tmp_path):
    _opted_in(monkeypatch, reference_filter="F212N")
    refcat = dict(all="sentinel-all", sparse="sentinel-sparse", mag=None)
    out = ac.resolve_tie_reference(refcat, str(tmp_path), "TEST", None, "F212N", record_dir=str(tmp_path))
    assert out is refcat
    # case-insensitive
    out = ac.resolve_tie_reference(refcat, str(tmp_path), "TEST", None, "f212n", record_dir=str(tmp_path))
    assert out is refcat


# ---------------------------------------------------------------------------
# resolve_tie_reference / reference_filter_tie_settled -- blocking cases
# ---------------------------------------------------------------------------

def test_no_m2_record_blocks(monkeypatch, tmp_path):
    _opted_in(monkeypatch)
    refcat = dict(all=None, sparse=None, mag=None)
    with pytest.raises(ac.ReferenceFilterNotSettledError, match="no m2 record"):
        ac.resolve_tie_reference(refcat, str(tmp_path), "TEST", None, "F480M", record_dir=str(tmp_path))


def test_unpassed_record_with_no_override_blocks(monkeypatch, tmp_path):
    _opted_in(monkeypatch)
    cons_path = str(tmp_path / "f212n_consensus.fits")
    _write_consensus_fits(cons_path, SkyCoord(ra=[RA0] * u.deg, dec=[DEC0] * u.deg,
                                              frame="icrs"))
    _write_m2_record(str(tmp_path), "F212N", cons_path, passed=False)
    refcat = dict(all=None, sparse=None, mag=None)
    with pytest.raises(ac.ReferenceFilterNotSettledError, match="did not pass"):
        ac.resolve_tie_reference(refcat, str(tmp_path), "TEST", None, "F480M", record_dir=str(tmp_path))


def test_unpassed_record_with_used_override_settles(monkeypatch, tmp_path):
    _opted_in(monkeypatch)
    ra, dec = _field(n=50)
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    cons_path = str(tmp_path / "f212n_consensus.fits")
    _write_consensus_fits(cons_path, coords)
    _write_m2_record(str(tmp_path), "F212N", cons_path, passed=False,
                     gate_override=dict(env="ASTROM_CHECKPOINT_WARN_ONLY",
                                        used=True, reason="operator waiver"))
    refcat = dict(all=None, sparse=SkyCoord(ra=[RA0] * u.deg, dec=[DEC0] * u.deg,
                                            frame="icrs"), mag=None)
    out = ac.resolve_tie_reference(refcat, str(tmp_path), "TEST", None, "F480M", record_dir=str(tmp_path))
    assert out["reference_kind"] == ac.REFERENCE_KIND_JWST_CONSENSUS


def test_unused_override_does_not_settle(monkeypatch, tmp_path):
    _opted_in(monkeypatch)
    cons_path = str(tmp_path / "f212n_consensus.fits")
    _write_consensus_fits(cons_path, SkyCoord(ra=[RA0] * u.deg, dec=[DEC0] * u.deg,
                                              frame="icrs"))
    _write_m2_record(str(tmp_path), "F212N", cons_path, passed=False,
                     gate_override=dict(env="ASTROM_CHECKPOINT_WARN_ONLY",
                                        used=False, reason=""))
    refcat = dict(all=None, sparse=None, mag=None)
    with pytest.raises(ac.ReferenceFilterNotSettledError, match="did not pass"):
        ac.resolve_tie_reference(refcat, str(tmp_path), "TEST", None, "F480M", record_dir=str(tmp_path))


def test_applied_reference_correction_this_pass_blocks(monkeypatch, tmp_path):
    """A record whose OWN pass applied a consensus->reference bulk correction
    means the reference filter's frame just moved -- its consensus catalog may
    still be the pre-correction one until the frame is regenerated."""
    _opted_in(monkeypatch)
    ra, dec = _field(n=50)
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    cons_path = str(tmp_path / "f212n_consensus.fits")
    _write_consensus_fits(cons_path, coords)
    _write_m2_record(
        str(tmp_path), "F212N", cons_path, passed=True,
        corrections=[dict(visit="001", exposure=None, module=None,
                          filtername="F212N", dra_onsky_mas=12.0,
                          ddec_onsky_mas=-4.0,
                          source=f"m2 {ac.REFERENCE_TIE_SOURCE_SUFFIX}")])
    refcat = dict(all=None, sparse=None, mag=None)
    with pytest.raises(ac.ReferenceFilterNotSettledError, match="APPLIED"):
        ac.resolve_tie_reference(refcat, str(tmp_path), "TEST", None, "F480M", record_dir=str(tmp_path))


def test_missing_consensus_file_blocks(monkeypatch, tmp_path):
    _opted_in(monkeypatch)
    _write_m2_record(str(tmp_path), "F212N",
                     str(tmp_path / "does_not_exist.fits"), passed=True)
    refcat = dict(all=None, sparse=None, mag=None)
    with pytest.raises(ac.ReferenceFilterNotSettledError, match="no usable"):
        ac.resolve_tie_reference(refcat, str(tmp_path), "TEST", None, "F480M", record_dir=str(tmp_path))


def test_stale_consensus_file_blocks(monkeypatch, tmp_path):
    """The record's own `date` is far NEWER than the consensus file's mtime --
    the file predates this pass and was not written by it."""
    _opted_in(monkeypatch)
    ra, dec = _field(n=50)
    coords = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    cons_path = str(tmp_path / "f212n_consensus.fits")
    _write_consensus_fits(cons_path, coords)
    future = datetime.now(timezone.utc) + timedelta(hours=2)
    _write_m2_record(str(tmp_path), "F212N", cons_path, passed=True,
                     date=future.isoformat())
    refcat = dict(all=None, sparse=None, mag=None)
    with pytest.raises(ac.ReferenceFilterNotSettledError, match="OLDER"):
        ac.resolve_tie_reference(refcat, str(tmp_path), "TEST", None, "F480M", record_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# resolve_tie_reference -- the settled, substituting case
# ---------------------------------------------------------------------------

def test_settled_reference_returns_consensus_dict(monkeypatch, tmp_path):
    ra, dec = _field(n=300)
    ref_all, ref_sparse = _reference_sets(ra, dec)
    cons_path = str(tmp_path / "f212n_consensus.fits")
    _write_consensus_fits(cons_path, ref_all)
    _write_m2_record(str(tmp_path), "F212N", cons_path, passed=True)
    _opted_in(monkeypatch)
    virac2_refcat = dict(all=SkyCoord(ra=[0.0] * u.deg, dec=[0.0] * u.deg,
                                      frame="icrs"),
                         sparse=ref_sparse, mag=np.array([15.0]))
    out = ac.resolve_tie_reference(virac2_refcat, str(tmp_path), "TEST", None,
                                   "F480M", record_dir=str(tmp_path))
    assert out["reference_kind"] == ac.REFERENCE_KIND_JWST_CONSENSUS
    assert out["reference_filter"] == "F212N"
    assert out["reference_path"] == cons_path
    assert out["dense"] is True
    assert out["mag"] is None                 # check E disabled
    assert out["sparse"] is virac2_refcat["sparse"]   # gross cross-check kept
    assert len(out["all"]) == len(ref_all)


# ---------------------------------------------------------------------------
# synthetic end-to-end: F480M's own m2 checkpoint, ties through F212N
# ---------------------------------------------------------------------------

def test_f480m_recovers_a_known_offset_from_f212n_consensus(monkeypatch, tmp_path):
    """F212N's consensus IS the reference; F480M's own consensus sits a known
    (dra, ddec) away from it.  The m2 checkpoint must recover that offset,
    stamp `reference_kind='jwst_consensus'`, and tag the correction's
    provenance with the reference filter -- without ever touching VIRAC2."""
    DRA_MAS, DDEC_MAS = 15.0, -9.0
    rng = np.random.default_rng(99)
    ra, dec = _field(n=400, rng=rng)
    ref_all, ref_sparse = _reference_sets(ra, dec, rng=rng)

    cons_path = str(tmp_path / "f212n_consensus.fits")
    _write_consensus_fits(cons_path, ref_all)
    _write_m2_record(str(tmp_path), "F212N", cons_path, passed=True)
    _opted_in(monkeypatch)

    virac2_refcat = dict(all=SkyCoord(ra=[0.0] * u.deg, dec=[0.0] * u.deg,
                                      frame="icrs"),
                         sparse=ref_sparse, mag=None)
    tie_refcat = ac.resolve_tie_reference(
        virac2_refcat, str(tmp_path), "TEST", None, "F480M",
        record_dir=str(tmp_path))
    assert tie_refcat["reference_kind"] == ac.REFERENCE_KIND_JWST_CONSENSUS

    ra_f480m = ra + DRA_MAS / 3.6e6 / COSD
    dec_f480m = dec + DDEC_MAS / 3.6e6
    tables = [_exposure_table(ra_f480m, dec_f480m, exposure=e, filtername="F480M",
                              rng=np.random.default_rng(500 + e))
             for e in range(1, 5)]

    rec = ac.run_visit_checkpoint(tables, "m2", refcat=tie_refcat,
                                  filtername="F480M", record_dir=str(tmp_path),
                                  context="test-f480m-via-f212n")

    ref_tie = rec["visits"][0]["reference_tie"]
    assert ref_tie["reference_kind"] == ac.REFERENCE_KIND_JWST_CONSENSUS
    assert ref_tie["reference_filter"] == "F212N"
    assert ref_tie["reference_path"] == cons_path

    assert len(rec["corrections"]) == 1
    corr = rec["corrections"][0]
    assert corr["filtername"] == "F480M"
    assert corr["exposure"] is None and corr["module"] is None   # bulk row
    assert corr["source"].startswith(f"m2 {ac.REFERENCE_TIE_SOURCE_SUFFIX}")
    assert "(via F212N consensus)" in corr["source"]
    # sign convention: measure_reference_tie's dra_mas is (reference - consensus);
    # F480M's consensus sits at truth + (DRA, DDEC), F212N's reference at truth.
    assert corr["dra_onsky_mas"] == pytest.approx(-DRA_MAS, abs=3.0)
    assert corr["ddec_onsky_mas"] == pytest.approx(-DDEC_MAS, abs=3.0)


# ---------------------------------------------------------------------------
# alignment_config wiring: only 10678 is opted in
# ---------------------------------------------------------------------------

def test_only_10678_is_opted_in():
    from jwst_gc_pipeline.reduction.alignment_config import ALIGNMENT_CONFIG
    opted_in = [c for c in ALIGNMENT_CONFIG if c.tie_through_reference_filter]
    assert len(opted_in) == 1
    assert opted_in[0].proposal == "10678"
    assert opted_in[0].reference_filter == "F212N"


def test_field_alignment_defaults_to_opted_out():
    cfg = FieldAlignment(proposal="X", fields=None, reference_frame=VIRAC2,
                         source=TABLE_CONSENSUS)
    assert cfg.tie_through_reference_filter is False
