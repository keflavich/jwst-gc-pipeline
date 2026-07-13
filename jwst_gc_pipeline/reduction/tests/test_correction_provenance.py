"""Correction-provenance safety (2026-07-13): every astrometric correction
records its BASE coordinate, offset (labelled convention), and TARGET, and is
verifiable at any later time.  Covers the astrometry_utils helpers and the
convention identities the fix_alignment apply-proof relies on."""
import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

from jwst_gc_pipeline.astrometry_utils import (
    GENERATION_KEYS, base_mismatch_mas, dra, dra_coordinate,
    generation_stamp, wcs_fiducial,
)


def _fake_header(crval1=266.54, crval2=-28.70, raoffset=None, deoffset=None,
                 **extra):
    h = fits.Header()
    h["CTYPE1"], h["CTYPE2"] = "RA---TAN", "DEC--TAN"
    h["CRVAL1"], h["CRVAL2"] = crval1, crval2
    h["CRPIX1"], h["CRPIX2"] = 1024.0, 1024.0
    h["CD1_1"], h["CD2_2"] = -0.031 / 3600.0, 0.031 / 3600.0
    h["CD1_2"] = h["CD2_1"] = 0.0
    if raoffset is not None:
        h["RAOFFSET"] = raoffset
        h["DEOFFSET"] = deoffset
    for k, v in extra.items():
        h[k] = v
    return h


def test_dra_conventions_are_distinct_and_consistent():
    ra1, ra2, dec = 266.55, 266.54, -28.70
    assert dra_coordinate(ra1, ra2) == pytest.approx(0.01)
    assert dra(ra1, ra2, dec) == pytest.approx(0.01 * np.cos(np.radians(dec)))
    # the identity every apply/verify step relies on
    assert dra(ra1, ra2, dec) == pytest.approx(
        dra_coordinate(ra1, ra2) * np.cos(np.radians(dec)))


def test_wcs_fiducial_removes_baked_offset():
    h0 = _fake_header()
    ra_a, dec_a = wcs_fiducial(h0)
    # same WCS but with a baked RAOFFSET: SIAF fiducial must be invariant
    h1 = _fake_header(crval1=266.54 + 0.5 / 3600.0, crval2=-28.70 - 0.2 / 3600.0,
                      raoffset=0.5, deoffset=-0.2)
    ra_b, dec_b = wcs_fiducial(h1)
    assert base_mismatch_mas(ra_a, dec_a, ra_b, dec_b) < 0.01


def test_base_mismatch_detects_generation_drift():
    h0 = _fake_header()
    ra_a, dec_a = wcs_fiducial(h0)
    # a 45 mas generation drift (the measured Brick class)
    h1 = _fake_header(crval1=266.54 + 45.0 / 3.6e6 / np.cos(np.radians(-28.70)))
    ra_b, dec_b = wcs_fiducial(h1)
    assert base_mismatch_mas(ra_a, dec_a, ra_b, dec_b) == pytest.approx(45.0, abs=1.0)


def test_generation_stamp_keys():
    h = _fake_header(CAL_VER="1.18.0", CRDS_CTX="jwst_1364.pmap", DVACORR=True)
    s = generation_stamp(h)
    assert s == {"cal_ver": "1.18.0", "crds_ctx": "jwst_1364.pmap",
                 "dvacorr": "True"}
    # absent keys stamp as empty string (comparable, never KeyError)
    s2 = generation_stamp(_fake_header())
    assert set(s2) == {k.lower() for k in GENERATION_KEYS}
    assert all(v == "" for v in s2.values())


def test_apply_proof_identity():
    """The fix_alignment apply-proof: target fiducial == base fiducial +
    COORDINATE shift.  Simulate with two WCS differing by a pure CRVAL shift
    (what adjust_wcs effects to first order)."""
    shift_coord_deg = -0.227 / 3600.0   # the measured brick dra_coordinate
    h0 = _fake_header()
    h1 = _fake_header(crval1=266.54 + shift_coord_deg)
    b = WCS(h0).pixel_to_world(1024, 1024)
    t = WCS(h1).pixel_to_world(1024, 1024)
    resid_mas = np.hypot(
        (t.ra.deg - (b.ra.deg + shift_coord_deg)) * np.cos(b.dec.rad),
        t.dec.deg - b.dec.deg) * 3.6e6
    assert resid_mas < 0.01
    # and the WRONG convention (on-sky value fed as coordinate) fails the proof
    onsky = shift_coord_deg * np.cos(np.radians(-28.70))
    resid_wrong = abs((t.ra.deg - (b.ra.deg + onsky)) * np.cos(b.dec.rad)) * 3.6e6
    assert resid_wrong > 20.0   # ~28 mas at the GC -- the caught bug class
