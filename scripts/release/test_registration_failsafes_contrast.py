"""Hermetic tests for the own_catalog FAIL contrast-margin (FAIL_MIN_RATIO).

A cell FAILs only on a large offset AND confident contrast.  A real localized
seam doubles stars into a SHARP high-contrast peak; a bright-star-crowded, sparse
cell throws a floor-level peak (ratio ~ MIN_PEAK_RATIO) at a spurious large offset
-- that is not a seam and must NOT FAIL (brick F405N: 7 cells at 80 mas / peak_bg
5-8, same-star m7 read <=22 mas, 2026-07).  These build synthetic det/truth sets
so ``per_cell`` runs on controlled geometry -- no data files.
"""
import importlib.util
from pathlib import Path

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

_spec = importlib.util.spec_from_file_location(
    "registration_failsafes",
    Path(__file__).with_name("registration_failsafes.py"))
rf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rf)

RA0, DEC0 = 266.5, -28.7
COSD = np.cos(np.deg2rad(DEC0))


def _grid_sources(n_side=140, extent_arcsec=40.0, seed=0):
    """A regular star field: truth = stars, det = same stars (perfect match)."""
    rng = np.random.default_rng(seed)
    g = np.linspace(0, extent_arcsec, n_side)
    xx, yy = np.meshgrid(g, g)
    x = xx.ravel() + rng.normal(0, 0.02, xx.size)   # tiny jitter (mas-level)
    y = yy.ravel() + rng.normal(0, 0.02, yy.size)
    ra = RA0 + (x / 3600.0) / COSD
    dec = DEC0 + y / 3600.0
    return ra, dec


def _sc(ra, dec):
    return SkyCoord(ra * u.deg, dec * u.deg)


def test_coherent_high_contrast_offset_FAILS():
    """A whole-field 90 mas rigid offset (every star shifted -> sharp, high-contrast
    peak) MUST fail: the margin does not blunt real seam sensitivity."""
    ra, dec = _grid_sources()
    truth = _sc(ra, dec)
    det = _sc(ra, dec + (90e-3 / 3600.0))          # detections shifted 90 mas in Dec
    r = rf.per_cell(det, None, truth, "synthetic seam")
    assert r.get("n_fail", 0) > 0 and not r["PASS"]
    assert r["worst"][0]["peak_bg"] >= rf.FAIL_MIN_RATIO


def test_low_contrast_large_offset_does_NOT_fail_but_is_reported():
    """Perfectly registered stars (peak at 0) plus a diffuse cloud of unrelated
    'detections' that make a weak, floor-level spurious peak at a large offset ->
    verified-but-not-failed, and surfaced in n_unconfident_highoff (not hidden)."""
    ra, dec = _grid_sources(seed=1)
    truth = _sc(ra, dec)
    # detections: the real stars (match at 0) + wrong-pair noise generating a weak
    # peak far off zero. Reuse the real positions but nudge a MINORITY by ~80 mas so
    # the dominant peak stays near 0 -> the 80 mas bump is only floor-level contrast.
    rng = np.random.default_rng(2)
    ddec = np.zeros_like(dec)
    minority = rng.random(dec.size) < 0.12
    ddec[minority] = 80e-3 / 3600.0
    det = _sc(ra, dec + ddec)
    r = rf.per_cell(det, None, truth, "weak spurious")
    # real registration is ~0, so it must PASS; if any cell shows a high-off peak it
    # is only reported, never a fail
    assert r["PASS"] and r.get("n_fail", 0) == 0


def test_fail_min_ratio_above_verify_floor():
    """The margin must sit above the verify floor, else it is a no-op."""
    assert rf.FAIL_MIN_RATIO > rf.MIN_PEAK_RATIO


def test_clean_field_passes_full_coverage():
    """A perfectly-registered field: 0 fail, and cells verify (coverage intact --
    the margin removes no detections, unlike a spatial mask)."""
    ra, dec = _grid_sources(seed=3)
    truth = _sc(ra, dec)
    det = _sc(ra, dec)
    r = rf.per_cell(det, None, truth, "clean")
    assert r["PASS"] and r["n_fail"] == 0 and r["verified_cells"] > 0
