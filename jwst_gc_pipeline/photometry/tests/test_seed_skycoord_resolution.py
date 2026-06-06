"""Regression tests for seed catalog sky-coordinate resolution.

Background (2026-06-02 Star B bug):
    iter2/iter3 photometry combines iter1 daophot catalogs (carrying
    ``skycoord_centroid``) with satstar catalogs (carrying ``skycoord_fit``)
    and DAO postprocess detections (carrying ``skycoord``).  vstacking
    tables with mismatched mixin columns yields a combined SkyCoord
    column whose rows from the "other" table are MASKED.

    The previous ``_resolve_seed_skycoords`` walked candidate columns by
    name and called ``col.unmasked`` on the first one that existed.  The
    underlying fill for the masked rows is (ra=0, dec=0), which silently
    became "valid" sky coordinates of (0, 0).  Downstream,
    ``ww.world_to_pixel(SkyCoord(0,0))`` returns NaN x/y, and
    ``SeededFinder``'s finite filter dropped those rows without logging.

    On sickle F480M nrcb each-exposure, this silently lost ~1447 of 1476
    iter1 seeds at iter2.  Star B (a 51 kMJy/sr unsaturated star near a
    saturated neighbour) was one of them.  The bug presented as iter2
    catalog regression vs iter1 (918 vs 1447 sources, Star B residual
    442 -> 1700 MJy/sr in the iter3 mosaic).

These tests pin the fixes so the regression can't recur silently.
"""
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.wcs import WCS
import astropy.units as u

from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
    _resolve_seed_skycoords,
    _augment_seed_catalog_with_detections_sky,
    _combine_seed_and_satstars,
    SeededFinder,
)


def _make_wcs():
    """A trivial tangent-plane WCS for testing pixel<->sky round trips."""
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [50.5, 50.5]
    w.wcs.crval = [266.55, -28.80]
    w.wcs.cdelt = [-1.0 / 3600, 1.0 / 3600]
    w.wcs.cunit = ['deg', 'deg']
    return w


def _make_iter1_like(n=5):
    """iter1 daophot catalogs carry ``skycoord_centroid`` but no skycoord_fit."""
    rng = np.random.default_rng(0)
    t = Table()
    t['id'] = np.arange(1, n + 1)
    t['x_fit'] = 50.0 + rng.standard_normal(n)
    t['y_fit'] = 50.0 + rng.standard_normal(n)
    t['flux_fit'] = 100.0 + rng.uniform(0, 1000, n)
    t['skycoord_centroid'] = SkyCoord(
        ra=(266.55 + rng.standard_normal(n) * 1e-4) * u.deg,
        dec=(-28.80 + rng.standard_normal(n) * 1e-4) * u.deg,
        frame='icrs')
    return t


def _make_satstar_like(n=3):
    """satstar catalogs carry ``skycoord_fit`` but no skycoord_centroid."""
    rng = np.random.default_rng(1)
    t = Table()
    t['id'] = np.arange(1, n + 1)
    t['x_fit'] = 40.0 + rng.standard_normal(n)
    t['y_fit'] = 40.0 + rng.standard_normal(n)
    t['flux_fit'] = 5e4 + rng.uniform(0, 1e4, n)
    t['skycoord_fit'] = SkyCoord(
        ra=(266.56 + rng.standard_normal(n) * 1e-4) * u.deg,
        dec=(-28.79 + rng.standard_normal(n) * 1e-4) * u.deg,
        frame='icrs')
    return t


def test_resolve_skycoords_handles_masked_rows_after_vstack():
    """When iter1 (skycoord_centroid) vstacks with satstar (skycoord_fit),
    each candidate column is masked on the OTHER table's rows.  Resolution
    must produce a single skycoord column with NO row mapped to (0,0)."""
    iter1 = _make_iter1_like(n=5)
    sat = _make_satstar_like(n=3)
    combined = _combine_seed_and_satstars(iter1, sat)
    assert len(combined) == 8

    resolved = _resolve_seed_skycoords(Table(combined, copy=True))
    sk = resolved['skycoord']
    ra = np.asarray(sk.ra.deg, dtype=float)
    dec = np.asarray(sk.dec.deg, dtype=float)

    # No NaN, no (0,0) sentinel fill rows.
    assert np.all(np.isfinite(ra)), f"NaN in resolved RA: {ra}"
    assert np.all(np.isfinite(dec)), f"NaN in resolved Dec: {dec}"
    assert not np.any((ra == 0) & (dec == 0)), \
        "Resolved skycoord contains (0,0) sentinel rows"

    # iter1 rows (0..4) must keep their skycoord_centroid value.
    expected_iter1_ra = np.asarray(iter1['skycoord_centroid'].ra.deg)
    np.testing.assert_allclose(ra[:5], expected_iter1_ra, rtol=0, atol=1e-12)

    # sat rows (5..7) must keep their skycoord_fit value.
    expected_sat_ra = np.asarray(sat['skycoord_fit'].ra.deg)
    np.testing.assert_allclose(ra[5:], expected_sat_ra, rtol=0, atol=1e-12)


def test_augment_then_world_to_pixel_keeps_all_rows():
    """Round-trip: iter1+sat combined, augmented with DAO detections that
    have their own skycoord column, then projected to pixel coords.  No
    row should produce a NaN pixel, and no row should land at the same
    pixel as the (0,0) sky fallback would."""
    ww = _make_wcs()
    iter1 = _make_iter1_like(n=10)
    sat = _make_satstar_like(n=4)
    combined = _combine_seed_and_satstars(iter1, sat)

    # DAO postprocess: only x_centroid / y_centroid (no skycoord yet).
    det = Table()
    det['x_centroid'] = np.array([20.0, 30.0, 40.0])
    det['y_centroid'] = np.array([20.0, 30.0, 40.0])

    result, _ = _augment_seed_catalog_with_detections_sky(
        combined, det, ww=ww, match_radius_pix=1.0, return_stats=True)

    sk = result['skycoord']
    ra = np.asarray(sk.ra.deg, dtype=float)
    dec = np.asarray(sk.dec.deg, dtype=float)
    assert np.all(np.isfinite(ra)), "augment produced NaN RA"
    assert np.all(np.isfinite(dec)), "augment produced NaN Dec"
    assert not np.any((ra == 0) & (dec == 0)), \
        "augment produced (0,0) sentinel rows"

    # All rows project to finite pixel coords.
    xpix, ypix = ww.world_to_pixel(sk)
    xpix = np.asarray(xpix, dtype=float)
    ypix = np.asarray(ypix, dtype=float)
    assert np.all(np.isfinite(xpix)), \
        f"world_to_pixel produced NaN x; bad rows={np.where(~np.isfinite(xpix))[0]}"
    assert np.all(np.isfinite(ypix)), \
        f"world_to_pixel produced NaN y; bad rows={np.where(~np.isfinite(ypix))[0]}"


def test_seeded_finder_logs_silent_finite_drops(capsys):
    """SeededFinder must LOG when it drops rows whose sky->pixel mapped
    to NaN (e.g. from a previously-masked sentinel).  Without this log,
    catastrophic silent losses pass without warning."""
    ww = _make_wcs()
    # Build a seed table with two valid rows + one explicit NaN-sky row.
    t = Table()
    t['flux_fit'] = np.array([100.0, 200.0, 300.0])
    t['skycoord'] = SkyCoord(
        ra=[266.55, 266.555, np.nan] * u.deg,
        dec=[-28.80, -28.795, np.nan] * u.deg,
        frame='icrs')
    data = np.zeros((100, 100))
    out = SeededFinder(t, ww=ww)(data)
    captured = capsys.readouterr().out
    assert "non-finite sky->pixel" in captured, \
        f"SeededFinder did not log silent drop: stdout={captured!r}"
    # Two valid rows kept.
    assert len(out) == 2


def test_unmasked_skycoord_column_passes_through():
    """If the seed table already has a clean (no-mask) skycoord column,
    _resolve_seed_skycoords should return it untouched (fast path)."""
    t = Table()
    t['flux_fit'] = np.array([1.0, 2.0])
    sc = SkyCoord(ra=[266.55, 266.56] * u.deg,
                  dec=[-28.80, -28.79] * u.deg, frame='icrs')
    t['skycoord'] = sc
    out = _resolve_seed_skycoords(Table(t, copy=True))
    assert 'skycoord' in out.colnames
    out_ra = np.asarray(out['skycoord'].ra.deg)
    np.testing.assert_allclose(out_ra, sc.ra.deg, rtol=0, atol=1e-12)


def test_ra_dec_columns_build_skycoord():
    """Union seed catalogs from build_union_seed_catalog.py carry plain
    ra/dec columns.  Must still resolve cleanly."""
    t = Table()
    t['ra'] = np.array([266.55, 266.56])
    t['dec'] = np.array([-28.80, -28.79])
    t['flux_fit'] = np.array([1.0, 2.0])
    out = _resolve_seed_skycoords(Table(t, copy=True))
    sk = out['skycoord']
    assert isinstance(sk, SkyCoord)
    np.testing.assert_allclose(np.asarray(sk.ra.deg), [266.55, 266.56])
    np.testing.assert_allclose(np.asarray(sk.dec.deg), [-28.80, -28.79])


def test_xy_only_fallback_uses_wcs():
    """When no skycoord/ra/dec exist but (x, y) + ww are available, sky
    should be derived via pixel_to_world."""
    ww = _make_wcs()
    t = Table()
    t['x_init'] = np.array([50.0, 60.0])
    t['y_init'] = np.array([50.0, 60.0])
    t['flux_fit'] = np.array([1.0, 2.0])
    out = _resolve_seed_skycoords(Table(t, copy=True), ww=ww)
    sk = out['skycoord']
    assert isinstance(sk, SkyCoord)
    expected = ww.pixel_to_world(t['x_init'], t['y_init'])
    np.testing.assert_allclose(np.asarray(sk.ra.deg),
                               np.asarray(expected.ra.deg), rtol=0, atol=1e-12)
