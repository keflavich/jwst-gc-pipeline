"""Merge-time registration of per-frame catalogs onto the m2 visit consensus.

The end-to-end test runs the real m2 checkpoint on synthetic exposures that
carry SUB-TOLERANCE per-exposure offsets -- the class m2 records and never
corrects -- and checks that the registration built from its record removes
them.  That pins the sign convention against the writer rather than against a
hand-made record.
"""
import json

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry.astrometry_checkpoint import run_visit_checkpoint
from jwst_gc_pipeline.photometry.m2_registration import (
    M2_REGISTRATION_ENV, M2_REGISTRATION_MAX_ENV, apply_m2_registration,
    load_m2_registration, registration_from_record)

RA0, DEC0 = 266.5, -28.7
COSD = np.cos(np.radians(DEC0))

# per-exposure (dra, ddec) on-sky mas: every one under the 2 mas consensus
# tolerance, so m2 corrects none of them.
SUBTOL = {1: (+1.5, -0.6), 2: (-1.2, +0.9), 3: (+0.8, +1.1),
          4: (-0.5, -1.3), 5: (+1.0, +0.2), 6: (-1.6, +0.4)}


def _truth(n=500, extent_arcsec=90.0, seed=7):
    rng = np.random.default_rng(seed)
    ra = RA0 + rng.uniform(0, extent_arcsec, n) / 3600.0 / COSD
    dec = DEC0 + rng.uniform(0, extent_arcsec, n) / 3600.0
    return ra, dec


def _exposure(ra, dec, exposure, dra_mas, ddec_mas, noise_mas=0.4,
              raoffset=0.1, deoffset=-0.05, module="nrcb1", filtername="F212N"):
    rng = np.random.default_rng(100 + exposure)
    n = len(ra)
    tbl = Table()
    tbl["skycoord_centroid"] = SkyCoord(
        ra=(ra + (dra_mas + rng.normal(0, noise_mas, n)) / 3.6e6 / COSD) * u.deg,
        dec=(dec + (ddec_mas + rng.normal(0, noise_mas, n)) / 3.6e6) * u.deg,
        frame="icrs")
    tbl["flux_fit"] = rng.uniform(1e3, 1e5, n)
    tbl["flux_err"] = tbl["flux_fit"] / 100.0
    tbl["qfit"] = rng.uniform(0.01, 0.05, n)
    tbl.meta.update(VISIT="001", EXPOSURE=f"{exposure:05d}", MODULE=module,
                    FILTER=filtername, RAOFFSET=raoffset, DEOFFSET=deoffset,
                    VGROUP="02101")
    return tbl


def _offsets_vs_truth(tables, ra, dec):
    """Per-exposure mean (dra, ddec) on-sky mas of each catalog from truth."""
    out = []
    for t in tables:
        sc = t["skycoord_centroid"]
        out.append(((sc.ra.deg - ra).mean() * COSD * 3.6e6,
                    (sc.dec.deg - dec).mean() * 3.6e6))
    return np.array(out)


def _visit():
    ra, dec = _truth()
    return ra, dec, [_exposure(ra, dec, e, *SUBTOL[e]) for e in sorted(SUBTOL)]


def test_registration_from_the_real_m2_record_removes_subtolerance_offsets(tmp_path):
    ra, dec, tables = _visit()
    record = run_visit_checkpoint(tables, "m2", filtername="F212N",
                                  record_dir=str(tmp_path), context="test")
    # The premise: m2 measured these and acted on none of them.
    assert record["corrections"] == []
    exps = record["visits"][0]["exposures"]
    assert len(exps) == len(SUBTOL)
    assert not any(e["misaligned"] for e in exps)

    before = _offsets_vs_truth(tables, ra, dec)
    reg = load_m2_registration(str(tmp_path), "F212N", env={})
    assert len(reg.entries) == len(SUBTOL)
    summary = apply_m2_registration(tables, reg)
    assert summary["counts"]["applied"] == len(SUBTOL)
    after = _offsets_vs_truth(tables, ra, dec)

    # Relative registration: the spread across exposures is what goes away.
    # The common part (the consensus gauge) is median-re-centred and stays.
    spread_before = before.std(axis=0)
    spread_after = after.std(axis=0)
    assert spread_before.min() > 0.7
    assert spread_after.max() < 0.15, (spread_before, spread_after)
    # The gauge stays at the median of the injected offsets, not elsewhere.
    inj = np.array([SUBTOL[e] for e in sorted(SUBTOL)])
    assert np.abs(after.mean(axis=0) - np.median(inj, axis=0)).max() < 0.5
    for t in tables:
        assert t.meta["M2REGST"] == "applied"
        assert t.meta["M2REGREC"] == "checkpoint_m2_F212N_latest.json"


def _record(entries, floor=4.0, stage="m2", filtername="F212N"):
    return dict(stage=stage, date="2026-10-08T00:00:00Z", passed=True,
                tolerances=dict(correction_floor_mas=floor),
                visits=[dict(visit="001", filtername=filtername,
                             exposures=entries)])


def _entry(exposure, dra, ddec, module="nrcb1", filtername="F212N",
           raoffset=0.1, deoffset=-0.05, **kw):
    e = dict(key=["001", exposure, module, filtername, "02101"],
             dra=dra, ddec=ddec, ok=True, unverified=False, alias_suspect=False,
             internal_tie=True, misaligned=False,
             raoffset_meta=raoffset, deoffset_meta=deoffset)
    e.update(kw)
    return e


def _one(exposure=1, **kw):
    ra, dec = _truth(n=50)
    return ra, dec, _exposure(ra, dec, exposure, 0.0, 0.0, noise_mas=0.0, **kw)


def _positions(t):
    sc = t["skycoord_centroid"]
    return sc.ra.deg.copy(), sc.dec.deg.copy()


def test_applied_shift_is_the_on_sky_offset():
    _, _, t = _one()
    ra0, dec0 = _positions(t)
    reg = registration_from_record(_record([_entry(1, 3.0, -2.0)]), env={})
    apply_m2_registration([t], reg)
    ra1, dec1 = _positions(t)
    np.testing.assert_allclose((ra1 - ra0) * np.cos(np.radians(dec0)) * 3.6e6,
                               3.0, atol=1e-4)
    np.testing.assert_allclose((dec1 - dec0) * 3.6e6, -2.0, atol=1e-4)
    assert t.meta["M2REGRA"] == 3.0 and t.meta["M2REGDE"] == -2.0


def test_a_regenerated_frame_is_not_shifted():
    """The baked offset changed after m2 measured it: the record describes a
    different WCS, so nothing is applied."""
    _, _, t = _one(raoffset=0.1 + 2e-6)
    ra0, dec0 = _positions(t)
    reg = registration_from_record(_record([_entry(1, 1.0, 1.0)]), env={})
    summary = apply_m2_registration([t], reg)
    assert summary["counts"]["stale"] == 1
    assert t.meta["M2REGST"] == "stale"
    np.testing.assert_array_equal(_positions(t)[0], ra0)
    np.testing.assert_array_equal(_positions(t)[1], dec0)


@pytest.mark.parametrize("flag", [dict(ok=False), dict(unverified=True),
                                  dict(alias_suspect=True),
                                  dict(alias_rejected=True),
                                  dict(internal_tie=False)])
def test_an_entry_m2_did_not_certify_is_not_applied(flag):
    _, _, t = _one()
    ra0, _ = _positions(t)
    reg = registration_from_record(_record([_entry(1, 1.0, 1.0, **flag)]),
                                   env={})
    summary = apply_m2_registration([t], reg)
    assert summary["counts"]["refused"] == 1, flag
    np.testing.assert_array_equal(_positions(t)[0], ra0)


def _row(dets, offsets, floor, misaligned=True, exposure=1):
    """One (exposure, module family) row of misaligned entries."""
    return _record([_entry(exposure, dra, ddec, module=d, misaligned=misaligned)
                    for d, (dra, ddec) in zip(dets, offsets)], floor=floor)


def _status(reg, module, exposure=1):
    _, _, t = _one(exposure=exposure, module=module)
    apply_m2_registration([t], reg)
    return t.meta["M2REGST"]


def test_a_detector_over_the_floor_in_a_row_m2_passed_is_registered():
    """sgrb2 F212N exposure 18: nrcb3 is 4.08 mas per detector, the misaligned
    nrcb row pools to 2.97 mas under a 4 mas floor, and m2 passed.  m2 left it
    in place, so the merge registers it."""
    dets = ("nrcb1", "nrcb3", "nrcb4")
    reg = registration_from_record(
        _row(dets, [(-2.25, 1.95), (-1.56, 3.77), (-1.04, 1.74)], floor=4.0),
        env={})
    assert reg.actionable == set()
    assert _status(reg, "nrcb3") == "applied"


def test_a_row_that_pools_to_the_floor_is_left_for_the_offsets_table():
    dets = ("nrcb1", "nrcb2")
    reg = registration_from_record(
        _row(dets, [(3.0, 3.0), (2.8, 2.6)], floor=4.0), env={})
    assert _status(reg, "nrcb1") == "m2_actionable"
    assert _status(reg, "nrcb2") == "m2_actionable"
    # The same row under an 8 mas floor is one m2 passed.
    reg8 = registration_from_record(
        _row(dets, [(3.0, 3.0), (2.8, 2.6)], floor=8.0), env={})
    assert _status(reg8, "nrcb1") == "applied"


def test_rows_pool_within_a_module_family_only():
    """nrca and nrcb are separate offsets-table rows: a big nrca entry does not
    drag nrcb over the floor."""
    rec = _record([_entry(1, 6.0, 0.0, module="nrca1", misaligned=True),
                   _entry(1, 2.5, 0.0, module="nrcb1", misaligned=True)],
                  floor=4.0)
    reg = registration_from_record(rec, env={})
    assert _status(reg, "nrca1") == "m2_actionable"
    assert _status(reg, "nrcb1") == "applied"


def test_an_exposure_m2_did_not_flag_is_never_actionable():
    reg = registration_from_record(
        _row(("nrcb1",), [(1.5, 1.0)], floor=0.0, misaligned=False), env={})
    assert _status(reg, "nrcb1") == "applied"


def test_a_record_without_a_floor_was_decided_at_zero():
    """Pre-floor m2 acted on every misaligned exposure."""
    rec = _row(("nrcb1",), [(2.5, 0.0)], floor=None)
    del rec["tolerances"]["correction_floor_mas"]
    reg = registration_from_record(rec, env={})
    assert reg.floor_mas == 0.0
    assert _status(reg, "nrcb1") == "m2_actionable"


def test_a_record_without_misaligned_flags_falls_back_to_the_tolerance():
    rec = _record([_entry(1, 2.5, 0.0, module="nrcb1"),
                   _entry(2, 1.5, 0.0, module="nrcb1")], floor=0.0)
    for e in rec["visits"][0]["exposures"]:
        del e["misaligned"]
    reg = registration_from_record(rec, env={})
    assert _status(reg, "nrcb1", exposure=1) == "m2_actionable"
    assert _status(reg, "nrcb1", exposure=2) == "applied"


def test_operator_ceiling():
    rec = _record([_entry(1, 1.2, 1.2)], floor=4.0)
    assert registration_from_record(rec, env={}).ceiling_mas is None
    reg = registration_from_record(rec, env={M2_REGISTRATION_MAX_ENV: "1.5"})
    assert _status(reg, "nrcb1") == "over_cap"


def test_absent_and_other_filter_entries_are_not_applied():
    _, _, t = _one(exposure=2)
    rec = _record([_entry(1, 1.0, 1.0),
                   _entry(2, 1.0, 1.0, filtername="F480M")])
    reg = registration_from_record(rec, filtername="F212N", env={})
    assert set(reg.entries) == {("001", 1, "nrcb1", "F212N", "02101")}
    assert apply_m2_registration([t], reg)["counts"]["absent"] == 1


def test_duplicate_key_with_different_values_is_ambiguous():
    _, _, t = _one()
    rec = _record([_entry(1, 1.0, 1.0), _entry(1, -1.0, 0.5)])
    reg = registration_from_record(rec, env={})
    assert apply_m2_registration([t], reg)["counts"]["ambiguous"] == 1


def test_duplicate_key_certified_and_refused_is_ambiguous():
    rec = _record([_entry(1, 1.0, 1.0), _entry(1, 1.0, 1.0, ok=False)])
    reg = registration_from_record(rec, env={})
    key = ("001", 1, "nrcb1", "F212N", "02101")
    assert key in reg.ambiguous and key not in reg.entries


def test_an_ambiguous_entry_is_left_out_of_its_rows_pool():
    """Its value is unknown, so it cannot decide its siblings' row."""
    rec = _record([_entry(1, 10.0, 0.0, module="nrcb1", misaligned=True),
                   _entry(1, 12.0, 0.0, module="nrcb1", misaligned=True),
                   _entry(1, 2.5, 0.0, module="nrcb2", misaligned=True)],
                  floor=4.0)
    reg = registration_from_record(rec, env={})
    assert _status(reg, "nrcb1") == "ambiguous"
    assert _status(reg, "nrcb2") == "applied"


def test_only_an_m2_record_is_accepted():
    with pytest.raises(ValueError, match="not m2"):
        registration_from_record(_record([], stage="m4"), env={})


def test_disabled_by_env_and_missing_record(tmp_path):
    (tmp_path / "checkpoint_m2_F212N_latest.json").write_text(
        json.dumps(_record([_entry(1, 1.0, 1.0)])))
    assert load_m2_registration(str(tmp_path), "F212N",
                                env={M2_REGISTRATION_ENV: "0"}) is None
    assert load_m2_registration(str(tmp_path), "F480M", env={}) is None
    assert load_m2_registration(str(tmp_path), "F212N", env={}) is not None


def test_merge_refuses_registration_together_with_an_offsets_table():
    from jwst_gc_pipeline.photometry.merge_catalogs import merge_individual_frames
    reg = registration_from_record(_record([]), env={})
    with pytest.raises(ValueError, match="both given"):
        merge_individual_frames(offsets_table=Table(), m2_registration=reg)
