"""Safeguards against the brick-1182 F200W seam failure (2026-07-12): a locally
misregistered overlap that bulk / coarse-grid / vs-reference QC all passed.

Covers (1) the per-tile offset-MAGNITUDE gate in ``measure_offset_grid`` (a
self-consistent tile offset by ~90 mas must FAIL, not pass on contrast alone) and
(2) the reference-free inter-frame overlap check.
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.astrometry_offsets import measure_offset_grid
from jwst_gc_pipeline.photometry.interframe_overlap import (
    pairwise_overlap_offsets, overlap_offset_grid, assert_overlaps_registered,
    OverlapMisregistrationError)


def _field(n=4000, ra0=266.54, dec0=-28.70, span=0.02, seed=0):
    """A random star field (SkyCoord) around (ra0, dec0), span in degrees."""
    rng = np.random.default_rng(seed)
    ra = ra0 + (rng.random(n) - 0.5) * span
    dec = dec0 + (rng.random(n) - 0.5) * span
    return ra, dec


def _shift(ra, dec, dra_mas, ddec_mas):
    """Apply an on-sky shift (mas) as a coordinate delta (dRA is on-sky here)."""
    cosd = np.cos(np.radians(dec))
    ra2 = ra + (dra_mas / 1000.0 / 3600.0) / cosd
    dec2 = dec + (ddec_mas / 1000.0 / 3600.0)
    return ra2, dec2


# ---------------------------------------------------------------------------
# (1) measure_offset_grid offset-magnitude gate
# ---------------------------------------------------------------------------

def test_grid_offset_magnitude_gate_fails_a_misregistered_but_coherent_tile():
    """A tile offset by ~90 mas with a razor-sharp peak (perfectly self-consistent)
    is the brick-1182 seam failure. Contrast alone passes it; the magnitude gate
    must FAIL it."""
    ra, dec = _field(seed=1)
    a = SkyCoord(ra * u.deg, dec * u.deg)
    # b = a everywhere EXCEPT the top dec-band, which is rigidly shifted 90 mas.
    top = dec > -28.70
    rb, db = ra.copy(), dec.copy()
    rb[top], db[top] = _shift(ra[top], dec[top], 90.0, 0.0)
    b = SkyCoord(rb * u.deg, db * u.deg)

    # contrast-only (legacy) view: the shifted tile still has a sharp peak -> clean
    legacy = measure_offset_grid(a, b, nx=4, ny=4, maxsep=1 * u.arcsec,
                                 max_off_mas=None)
    assert legacy["clean"] is True  # this is exactly why the seam slipped through

    # with the magnitude gate the offset tiles must fail
    gated = measure_offset_grid(a, b, nx=4, ny=4, maxsep=1 * u.arcsec,
                                max_off_mas=50.0)
    assert gated["clean"] is False
    assert gated["worst_off_mas"] > 50.0
    assert gated["worst_off_cell"] is not None
    bad = [c for c in gated["cells"] if not c["off_ok"]]
    assert bad, "expected at least one tile flagged by the magnitude gate"
    # the flagged tiles are the coherent-but-offset ones
    assert all(c["contrast_ok"] for c in bad)


def test_grid_all_zero_offset_passes_the_gate():
    ra, dec = _field(seed=2)
    a = SkyCoord(ra * u.deg, dec * u.deg)
    b = SkyCoord(ra * u.deg, dec * u.deg)  # identical -> zero offset
    g = measure_offset_grid(a, b, nx=4, ny=4, maxsep=1 * u.arcsec, max_off_mas=50.0)
    assert g["clean"] is True
    assert g["worst_off_mas"] < 50.0


# ---------------------------------------------------------------------------
# (2) reference-free inter-frame overlap check
# ---------------------------------------------------------------------------

def test_overlap_misregistration_is_flagged_and_raises():
    """Two overlapping groups offset by 45 mas vs each other (the v001-v002 seam
    value) must be flagged and raise."""
    ra, dec = _field(seed=3)
    g1 = SkyCoord(ra * u.deg, dec * u.deg)
    r2, d2 = _shift(ra, dec, 45.0, 0.0)
    g2 = SkyCoord(r2 * u.deg, d2 * u.deg)
    groups = {"v001": g1, "v002": g2}

    res = pairwise_overlap_offsets(groups, tol_mas=30.0, maxsep=1 * u.arcsec)
    pair = [r for r in res if r["overlap"]][0]
    assert pair["off_mas"] == pytest.approx(45.0, abs=8.0)
    assert pair["ok"] is False

    with pytest.raises(OverlapMisregistrationError):
        assert_overlaps_registered(groups, tol_mas=30.0, maxsep=1 * u.arcsec)


def test_well_registered_overlap_passes():
    ra, dec = _field(seed=4)
    g1 = SkyCoord(ra * u.deg, dec * u.deg)
    r2, d2 = _shift(ra, dec, 5.0, -3.0)  # 6 mas, well within tol
    g2 = SkyCoord(r2 * u.deg, d2 * u.deg)
    groups = {"a": g1, "b": g2}
    res = assert_overlaps_registered(groups, tol_mas=30.0, maxsep=1 * u.arcsec)
    assert all(r["ok"] for r in res)


def test_nonoverlapping_groups_are_not_a_failure():
    """Disjoint pointings have nothing to check -> ok=True, overlap=False."""
    ra1, dec1 = _field(seed=5, ra0=266.54, dec0=-28.70)
    ra2, dec2 = _field(seed=6, ra0=266.60, dec0=-28.60)  # far away
    groups = {"p1": SkyCoord(ra1 * u.deg, dec1 * u.deg),
              "p2": SkyCoord(ra2 * u.deg, dec2 * u.deg)}
    res = pairwise_overlap_offsets(groups, tol_mas=30.0, maxsep=1 * u.arcsec)
    assert len(res) == 1
    assert res[0]["overlap"] is False
    assert res[0]["ok"] is True
    # and no raise
    assert_overlaps_registered(groups, tol_mas=30.0, maxsep=1 * u.arcsec)


def test_result_schema_is_uniform_including_nonoverlap():
    """Every result dict must carry the SAME keys: the non-overlap early-exit
    used to omit n_peak/measurable, so consumers indexing r["measurable"]
    KeyErrored on any non-overlapping pair."""
    ra1, dec1 = _field(seed=5, ra0=266.54, dec0=-28.70)
    ra2, dec2 = _field(seed=6, ra0=266.60, dec0=-28.60)  # disjoint
    ra3, dec3 = _field(seed=7, ra0=266.54, dec0=-28.70)  # overlaps p1
    groups = {"p1": SkyCoord(ra1 * u.deg, dec1 * u.deg),
              "p2": SkyCoord(ra2 * u.deg, dec2 * u.deg),
              "p3": SkyCoord(ra3 * u.deg, dec3 * u.deg)}
    res = pairwise_overlap_offsets(groups, tol_mas=30.0, maxsep=1 * u.arcsec)
    assert len(res) == 3
    keysets = [set(r) for r in res]
    assert all(k == keysets[0] for k in keysets), keysets
    for r in res:
        assert "measurable" in r and "n_peak" in r
        if not r["overlap"]:
            assert r["measurable"] is False
            assert r["n_peak"] == 0


def test_per_tile_catches_a_local_seam_that_field_pooling_hides():
    """The brick-1182 seam exactly: two visits overlap over the whole field but a
    THIN dec-band of one is offset ~90 mas. The field-pooled single offset averages
    it away (< tol); the per-tile grid must FAIL."""
    ra, dec = _field(n=8000, seed=8)
    g1 = SkyCoord(ra * u.deg, dec * u.deg)
    rb, db = ra.copy(), dec.copy()
    band = np.abs(dec - (-28.70)) < 0.001   # a ~7" dec strip through the middle
    rb[band], db[band] = _shift(ra[band], dec[band], 90.0, 0.0)
    g2 = SkyCoord(rb * u.deg, db * u.deg)
    groups = {"v001": g1, "v002": g2}

    # field-pooled: the offset is dominated by the (matched) unshifted majority -> passes
    pooled = pairwise_overlap_offsets(groups, tol_mas=30.0, maxsep=1 * u.arcsec)
    assert pooled[0]["off_mas"] < 30.0  # field average hides the seam (the trap)

    # per-tile: the shifted band's tiles exceed tol -> FAIL
    grid = overlap_offset_grid(groups, tol_mas=30.0, nx=12, ny=12, maxsep=1 * u.arcsec)
    pair = [r for r in grid if r["overlap"]][0]
    assert pair["ok"] is False
    assert pair["worst_off_mas"] > 60.0
    with pytest.raises(OverlapMisregistrationError):
        assert_overlaps_registered(groups, tol_mas=30.0, per_tile=True,
                                   grid=(12, 12), maxsep=1 * u.arcsec)


def test_gross_overlap_offset_is_swept_and_flagged():
    """A >window overlap offset (the case registration_failsafes' +-2.5" window
    cannot see) must be recovered by the sweep and flagged, not missed."""
    ra, dec = _field(seed=7)
    g1 = SkyCoord(ra * u.deg, dec * u.deg)
    r2, d2 = _shift(ra, dec, 6000.0, 0.0)  # 6" -- beyond a narrow window
    g2 = SkyCoord(r2 * u.deg, d2 * u.deg)
    groups = {"a": g1, "b": g2}
    res = pairwise_overlap_offsets(groups, tol_mas=30.0, maxsep=2.5 * u.arcsec)
    pair = [r for r in res if r["overlap"]][0]
    assert pair["ok"] is False
    assert pair["off_mas"] == pytest.approx(6000.0, abs=200.0)
    assert pair["swept"] is True  # normalized to a python bool at the module boundary


# ---------------------------------------------------------------------------
# geometric overlap gate (2026-07-12): disjoint-but-adjacent groups must be
# SKIPPED, not false-FAILed at their structural cross-field separation
# ---------------------------------------------------------------------------

def test_adjacent_disjoint_fields_are_not_overlapping():
    """The NIRCam module-gap case that false-FAILed on real brick F405N data:
    two star fields ~45" apart (edge to edge DISJOINT).  The old proximity
    gate called them overlapping and measure_offset returned the structural
    ~gap-scale separation as a 'coherent' offset -> guaranteed false FAIL.
    The geometric footprint gate must skip them."""
    ra1, dec1 = _field(seed=31, ra0=266.54, dec0=-28.70, span=0.01)
    # adjacent field: shifted by ~0.0125 deg (45") in dec -> zero footprint overlap
    ra2, dec2 = _field(seed=32, ra0=266.54, dec0=-28.6875, span=0.01)
    groups = {"nrca": SkyCoord(ra1 * u.deg, dec1 * u.deg),
              "nrcb": SkyCoord(ra2 * u.deg, dec2 * u.deg)}
    res = pairwise_overlap_offsets(groups, tol_mas=30.0, maxsep=1 * u.arcsec)
    assert res[0]["overlap"] is False
    assert res[0]["ok"] is True
    grid = overlap_offset_grid(groups, tol_mas=30.0, nx=8, ny=8,
                               maxsep=1 * u.arcsec)
    assert grid[0]["overlap"] is False
    assert grid[0]["ok"] is True
    # no raise in either mode
    assert_overlaps_registered(groups, tol_mas=30.0, maxsep=1 * u.arcsec)
    assert_overlaps_registered(groups, tol_mas=30.0, per_tile=True,
                               grid=(8, 8), maxsep=1 * u.arcsec)


def test_gross_offset_on_truly_overlapping_fields_still_fails():
    """The footprint gate must NOT kill the brick-1182 detection: a 6" rigid
    offset barely moves an arcminute footprint, so the intersection survives
    and the swept histogram still finds the gross offset."""
    ra, dec = _field(seed=33, span=0.02)
    g1 = SkyCoord(ra * u.deg, dec * u.deg)
    r2, d2 = _shift(ra, dec, 6000.0, 0.0)
    g2 = SkyCoord(r2 * u.deg, d2 * u.deg)
    res = pairwise_overlap_offsets({"a": g1, "b": g2}, tol_mas=30.0,
                                   maxsep=2.5 * u.arcsec)
    pair = [r for r in res if r["overlap"]][0]
    assert pair["ok"] is False
    assert pair["off_mas"] == pytest.approx(6000.0, abs=250.0)
    assert pair["swept"] is True


def test_stripey_interleave_is_could_not_verify_not_fail():
    """Interleaved two-module mosaic: footprint bboxes overlap but NO tile has
    both groups' stars (brick F405N nrca|nrcb false-FAIL case).  Must report
    could_not_verify (owned by the visit-consensus + reference layers), never
    a structural-offset FAIL."""
    rng = np.random.default_rng(41)
    n = 3000
    ra = 266.54 + (rng.random(n) - 0.5) * 0.02
    # A occupies dec stripes [0-20"], [40-60"]; B occupies [20-40"], [60-80"]
    def _stripes(offsets):
        dec = []
        for lo in offsets:
            dec.append(-28.70 + (lo + rng.random(n // len(offsets)) * 20.0) / 3600.0)
        return np.concatenate(dec)
    dec_a = _stripes([0.0, 40.0])
    dec_b = _stripes([20.0, 60.0])
    ga = SkyCoord(ra[:len(dec_a)] * u.deg, dec_a * u.deg)
    gb = SkyCoord(ra[:len(dec_b)] * u.deg, dec_b * u.deg)
    grid = overlap_offset_grid({"nrca": ga, "nrcb": gb}, tol_mas=30.0,
                               nx=8, ny=8, maxsep=1 * u.arcsec,
                               min_overlap_pairs=20)
    pair = grid[0]
    assert pair["overlap"] is True          # bboxes DO intersect
    assert pair["could_not_verify"] is True  # but no mutual-coverage tile
    assert pair["ok"] is True                # never a structural false FAIL
    assert pair["n_total"] == 0
    # and no raise in the assert wrapper
    assert_overlaps_registered({"nrca": ga, "nrcb": gb}, tol_mas=30.0,
                               per_tile=True, grid=(8, 8),
                               maxsep=1 * u.arcsec, min_overlap_pairs=20)


def test_unrelated_populations_in_shared_footprint_are_unmeasurable():
    """Null test of the n_peak floor: two UNRELATED star lists on the same
    footprint (the partial-coverage-cell regime) produce only low-N noise
    peaks near the window edge -- those must be UNMEASURABLE cells, never
    structural FAILs (3rd real-data round: fails piled at ~2.9" = 0.95*maxsep
    with contrast 5-10)."""
    ra1, dec1 = _field(seed=51, span=0.01)
    ra2, dec2 = _field(seed=52, span=0.01)  # same footprint, different stars
    groups = {"a": SkyCoord(ra1 * u.deg, dec1 * u.deg),
              "b": SkyCoord(ra2 * u.deg, dec2 * u.deg)}
    grid = overlap_offset_grid(groups, tol_mas=30.0, nx=4, ny=4,
                               maxsep=3 * u.arcsec, min_overlap_pairs=40)
    pair = grid[0]
    assert pair["overlap"] is True
    # every cell is either unmeasurable or (rarely) a genuine-looking clean 0;
    # what is FORBIDDEN is a >tol FAIL fabricated from noise
    bad = [] if pair["clean"] or pair["could_not_verify"] else [pair]
    assert pair["ok"] is True, bad
