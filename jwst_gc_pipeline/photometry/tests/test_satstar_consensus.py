"""Satstars in the visit consensus (issue #957)."""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry.satstar_consensus import (
    augment_consensus_with_satstars, consensus_with_satstars,
    satstar_catalog_path, satstar_consensus, satstar_stage_label)

RA0, DEC0 = 266.39, -29.15


def _at(dx_mas, dy_mas):
    """Sky positions offset (on-sky mas) from a fixed field centre."""
    dx = np.atleast_1d(np.asarray(dx_mas, float))
    dy = np.atleast_1d(np.asarray(dy_mas, float))
    return SkyCoord(RA0 + dx / 3.6e6 / np.cos(np.radians(DEC0)),
                    DEC0 + dy / 3.6e6, unit="deg")


def _sep_mas(a, b):
    return a.separation(b).to_value(u.mas)


def test_repeatable_satstar_is_kept_at_its_mean():
    sat = {("1", 1, "nrca1", "F212N", "02101"): _at([0.0], [0.0]),
           ("1", 2, "nrca1", "F212N", "02101"): _at([2.0], [0.0])}
    out = satstar_consensus(sat)
    assert len(out["coords"]) == 1
    assert out["nexp"][0] == 2
    assert out["rms_mas"][0] == pytest.approx(1.0, abs=1e-3)
    assert _sep_mas(out["coords"][0], _at([1.0], [0.0])[0]) < 0.01


def test_scatter_over_5mas_rms_is_rejected():
    sat = {("1", 1, "a", "F", "g"): _at([0.0], [0.0]),
           ("1", 2, "a", "F", "g"): _at([12.0], [0.0])}   # rms 6 mas
    out = satstar_consensus(sat)
    assert len(out["coords"]) == 0
    assert out["n_groups"] == 1 and out["n_rejected_rms"] == 1


def test_single_exposure_satstar_is_not_kept():
    sat = {("1", 1, "a", "F", "g"): _at([0.0, 5000.0], [0.0, 0.0]),
           ("1", 2, "a", "F", "g"): _at([5001.0], [0.0])}
    out = satstar_consensus(sat)
    assert len(out["coords"]) == 1
    assert _sep_mas(out["coords"][0], _at([5000.5], [0.0])[0]) < 0.01


def test_fits_farther_than_0p1_arcsec_are_not_linked():
    sat = {("1", 1, "a", "F", "g"): _at([0.0], [0.0]),
           ("1", 2, "a", "F", "g"): _at([150.0], [0.0])}
    assert len(satstar_consensus(sat)["coords"]) == 0


def test_exposure_offset_is_removed_before_grouping():
    # exposure 2 sits 30 mas east of exposure 1; its vs_consensus offset
    # (consensus minus exposure) carries it back.
    k1, k2 = ("1", 1, "a", "F", "g"), ("1", 2, "a", "F", "g")
    sat = {k1: _at([0.0], [0.0]), k2: _at([30.0], [0.0])}
    assert len(satstar_consensus(sat)["coords"]) == 0      # 15 mas rms
    out = satstar_consensus(sat, exposure_offsets={k2: (-30.0, 0.0)})
    assert len(out["coords"]) == 1
    assert out["rms_mas"][0] < 0.01


def test_second_fit_in_one_exposure_does_not_count_twice():
    # a deblend fragment 40 mas away in exposure 1 must not inflate the rms
    sat = {("1", 1, "a", "F", "g"): _at([0.0, 40.0], [0.0, 0.0]),
           ("1", 2, "a", "F", "g"): _at([1.0], [0.0])}
    out = satstar_consensus(sat)
    assert len(out["coords"]) == 1
    assert out["nexp"][0] == 2
    assert out["rms_mas"][0] < 1.0


def test_satstar_near_a_daophot_star_is_not_added():
    cons = dict(coords=_at([0.0, 10000.0], [0.0, 0.0]), mag=np.array([15.0, 16.0]))
    satc = dict(coords=_at([150.0, 3000.0], [0.0, 0.0]))
    coords, mag, n_added = augment_consensus_with_satstars(cons, satc)
    assert n_added == 1
    assert len(coords) == 3
    assert np.isnan(mag[-1]) and np.all(np.isfinite(mag[:2]))
    assert _sep_mas(coords[-1], _at([3000.0], [0.0])[0]) < 0.01


def test_visit_helper_uses_only_its_own_exposures():
    def tbl(visit, exp):
        t = Table()
        t.meta.update(VISIT=visit, EXPOSURE=f"_exp{exp:05d}", MODULE="nrca1",
                      FILTER="F212N", VGROUP="02101")
        return t
    tables = [tbl(1, 1), tbl(1, 2)]
    k1 = ("1", 1, "nrca1", "F212N", "02101")
    k2 = ("1", 2, "nrca1", "F212N", "02101")
    other = ("2", 1, "nrca1", "F212N", "02101")
    sat = {k1: _at([3000.0], [0.0]), k2: _at([3001.0], [0.0]),
           other: _at([3000.5, 9000.0], [0.0, 0.0])}
    cons = dict(coords=_at([0.0], [0.0]), mag=np.array([15.0]),
                exposures=[dict(key=k1, vs_consensus=dict(ok=True, dra=0.0, ddec=0.0)),
                           dict(key=k2, vs_consensus=dict(ok=True, dra=0.0, ddec=0.0))])
    coords, mag, summary = consensus_with_satstars(cons, tables, sat)
    assert summary["n_exposures_with_satstars"] == 2
    assert summary["n_added"] == 1
    assert len(coords) == 2


def test_stage_label_and_catalog_path(tmp_path):
    assert satstar_stage_label("m2") == "m12"
    assert satstar_stage_label("m1") == "m12"
    assert satstar_stage_label("m3") == "m3"
    assert satstar_stage_label("resbgsub_m5") == "resbgsub_m5"
    frame = tmp_path / "jw10678040001_02101_00001_nrca1_destreak_o040_crf.fits"
    t = Table()
    t.meta["FILENAME"] = str(frame)
    assert satstar_catalog_path(t, "m2") is None            # nothing on disk
    sat = tmp_path / "jw10678040001_02101_00001_nrca1_destreak_o040_crf_m12_satstar_catalog.fits"
    sat.write_bytes(b"")
    assert satstar_catalog_path(t, "m2") == str(sat)
    assert satstar_catalog_path(Table(), "m2") is None      # no FILENAME


# ---- reference blends ------------------------------------------------------

from jwst_gc_pipeline.photometry.reference_blends import blended_reference_mask  # noqa: E402


def test_equal_binary_at_0p5_arcsec_is_excluded():
    # VIRAC2 sits between two equal JWST stars 0.5" apart
    ref = _at([0.0, 10000.0], [0.0, 0.0])
    jw = _at([-250.0, 250.0, 10000.0], [0.0, 0.0, 0.0])
    flux = np.array([100.0, 100.0, 100.0])
    mask, info = blended_reference_mask(ref, [(jw, flux)])
    assert mask.tolist() == [True, False]
    assert info["n_seen"] == 2 and info["n_excluded"] == 1


def test_faint_neighbours_summing_over_25pct_is_a_group():
    ref = _at([0.0], [0.0])
    jw = _at([0.0, 150.0, -150.0, 0.0], [0.0, 0.0, 0.0, 200.0])
    flux = np.array([100.0, 10.0, 10.0, 10.0])      # 30% within 0.3"
    mask, _ = blended_reference_mask(ref, [(jw, flux)])
    assert mask[0]
    flux2 = np.array([100.0, 10.0, 10.0, 1.0])      # 21%
    mask2, _ = blended_reference_mask(ref, [(jw, flux2)])
    assert not mask2[0]


def test_blend_needs_a_majority_of_exposures():
    ref = _at([0.0], [0.0])
    single = (_at([0.0], [0.0]), np.array([100.0]))
    pair = (_at([0.0, 400.0], [0.0, 0.0]), np.array([100.0, 80.0]))
    assert not blended_reference_mask(ref, [single, single, pair])[0][0]
    assert blended_reference_mask(ref, [single, pair, pair])[0][0]


def test_blend_test_removes_the_tie_first():
    # reference 400 mas east of the JWST frame: without the tie the 0.3"
    # primary search misses the star entirely
    ref = _at([400.0], [0.0])
    jw = _at([0.0, -500.0], [0.0, 0.0])
    flux = np.array([100.0, 90.0])
    assert blended_reference_mask(ref, [(jw, flux)])[1]["n_seen"] == 0
    mask, info = blended_reference_mask(ref, [(jw, flux)], dra_mas=400.0)
    assert info["n_seen"] == 1 and mask[0]


def test_outside_fov_seed_rows_are_dropped():
    # a seed-derived position repeats trivially; it must not enter the consensus
    from jwst_gc_pipeline.photometry.satstar_consensus import _satstar_coords
    t = Table()
    t["skycoord_fit"] = _at([0.0, 1000.0], [0.0, 0.0])
    t["outside_fov_seed"] = [False, True]
    out = _satstar_coords(t)
    assert len(out) == 1
    assert _sep_mas(out[0], _at([0.0], [0.0])[0]) < 0.01
