"""A cached per-exposure satstar catalog stores the sky position the star had
when it was FIT.  The cache is deliberately not invalidated when the frame's
WCS changes (``_satstar_recovery_signature`` keys it on the recovery/deblend
config only), so after an offsets-table correction + frame regeneration the
stored ``skycoord_fit`` no longer matches its own pixel centroid, and the
consolidated satstar catalog publishes the OLD frame's astrometry.

Measured on brick F200W (issue #193): stored minus current-GWCS = +56.8/+88.7
mas over 3279 saturated stars in 40 exposures, against a +58.7/+88.2 mas
saturated-versus-unsaturated position excess in the m6 catalog built from them.

The pixel centroids are fine; only the projection is stale.  So every read
re-projects them through the frame's current GWCS.
"""
import os

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy import units as u

from jwst_gc_pipeline.photometry.satstar_wcs_refresh import (
    MissingSatstarFrameWarning, frame_path_for_satstar_catalog,
    refresh_satstar_skycoords)

PIXSCALE = 31.2 / 3600.0 / 1000.0  # NIRCam SW, deg/px


def _wcs(crval1=266.5, crval2=-28.7):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [64.5, 64.5]
    w.wcs.crval = [crval1, crval2]
    w.wcs.cdelt = [-PIXSCALE, PIXSCALE]
    return w


def _satstar_table(wcs_at_fit_time, x, y):
    """A per-exposure satstar catalog as the fitter writes it: pixel centroids
    plus the sky position they projected to under ``wcs_at_fit_time``."""
    sky = wcs_at_fit_time.pixel_to_world(x, y)
    tbl = Table({'xcentroid': np.asarray(x, dtype=float),
                 'ycentroid': np.asarray(y, dtype=float),
                 'flux_fit': np.ones(len(x))})
    tbl['skycoord_fit'] = sky
    tbl.meta.update(wcs_at_fit_time.to_header(relax=True))
    return tbl


def _write_frame(path, wcs_now, shape=(128, 128)):
    hdu0 = fits.PrimaryHDU()
    hdu1 = fits.ImageHDU(np.zeros(shape, dtype='float32'), name='SCI')
    hdu1.header.update(wcs_now.to_header(relax=True))
    fits.HDUList([hdu0, hdu1]).writeto(path, overwrite=True)


def _seps(tbl, reference_sky):
    got = SkyCoord(tbl['skycoord_fit'])
    return got.separation(reference_sky).to(u.mas).value


def test_stale_sky_is_reprojected_onto_the_moved_frame(tmp_path):
    """The frame moved 100 mas after the fit; the refresh must move the stored
    sky by exactly that, NOT leave it at the fit-time value."""
    fit_wcs = _wcs()
    now_wcs = _wcs(crval1=266.5 + 100e-3 / 3600.0 / np.cos(np.radians(-28.7)),
                   crval2=-28.7 + 30e-3 / 3600.0)
    x = np.array([10.0, 64.0, 120.0])
    y = np.array([20.0, 64.0, 100.0])
    tbl = _satstar_table(fit_wcs, x, y)
    stale = SkyCoord(tbl['skycoord_fit'])

    frame = str(tmp_path / 'exp_crf.fits')
    _write_frame(frame, now_wcs)
    cat = str(tmp_path / 'exp_crf_m3_satstar_catalog.fits')

    out, shift = refresh_satstar_skycoords(tbl, frame_path=frame,
                                           catalog_path=cat)

    # positions now agree with the CURRENT frame, to sub-mas
    assert np.all(_seps(out, now_wcs.pixel_to_world(x, y)) < 0.01)
    # and they are no longer what the cache stored
    assert np.all(_seps(out, stale) > 50)
    assert np.hypot(*shift) == pytest.approx(np.hypot(100.0, 30.0), rel=0.02)


def test_pixel_centroids_are_untouched(tmp_path):
    """Re-projection is the whole fix: the FIT is good, so the stored pixel
    centroid must survive the refresh byte for byte."""
    fit_wcs = _wcs()
    now_wcs = _wcs(crval1=266.6)
    x = np.array([10.0, 64.0, 120.0])
    y = np.array([20.0, 64.0, 100.0])
    tbl = _satstar_table(fit_wcs, x, y)
    frame = str(tmp_path / 'exp_crf.fits')
    _write_frame(frame, now_wcs)

    out, _ = refresh_satstar_skycoords(tbl, frame_path=frame)
    np.testing.assert_array_equal(np.asarray(out['xcentroid']), x)
    np.testing.assert_array_equal(np.asarray(out['ycentroid']), y)


def test_unmoved_frame_is_a_no_op(tmp_path):
    """A field whose frames have not moved since the fit must not be perturbed
    -- the refresh has to be safe to run on every read."""
    fit_wcs = _wcs()
    x = np.array([10.0, 64.0, 120.0])
    y = np.array([20.0, 64.0, 100.0])
    tbl = _satstar_table(fit_wcs, x, y)
    before = SkyCoord(tbl['skycoord_fit'])
    frame = str(tmp_path / 'exp_crf.fits')
    _write_frame(frame, fit_wcs)

    out, shift = refresh_satstar_skycoords(tbl, frame_path=frame)
    assert np.all(_seps(out, before) < 0.01)
    assert np.hypot(*shift) < 0.01


def test_component_anchor_follows_the_frame(tmp_path):
    """``sat_com_ra``/``sat_com_dec`` is the dedup's stable per-component sky
    anchor and is stored as sky only.  If it is left at the fit-time frame while
    ``skycoord_fit`` moves, one physical star splits into two rows in the
    consolidated catalog."""
    fit_wcs = _wcs()
    now_wcs = _wcs(crval1=266.5 + 200e-3 / 3600.0 / np.cos(np.radians(-28.7)))
    x = np.array([30.0, 90.0])
    y = np.array([40.0, 100.0])
    tbl = _satstar_table(fit_wcs, x, y)
    anchor_sky = fit_wcs.pixel_to_world(x + 1.0, y - 1.0)
    tbl['sat_com_ra'] = anchor_sky.ra.deg
    tbl['sat_com_dec'] = anchor_sky.dec.deg

    frame = str(tmp_path / 'exp_crf.fits')
    _write_frame(frame, now_wcs)
    out, _ = refresh_satstar_skycoords(tbl, frame_path=frame)

    expected = now_wcs.pixel_to_world(x + 1.0, y - 1.0)
    got = SkyCoord(np.asarray(out['sat_com_ra']) * u.deg,
                   np.asarray(out['sat_com_dec']) * u.deg)
    assert np.all(got.separation(expected).to(u.mas).value < 0.5)


def test_missing_frame_warns_rather_than_silently_serving_stale_sky(tmp_path):
    fit_wcs = _wcs()
    tbl = _satstar_table(fit_wcs, np.array([10.0]), np.array([20.0]))
    cat = str(tmp_path / 'gone_crf_m3_satstar_catalog.fits')
    with pytest.warns(MissingSatstarFrameWarning):
        out, shift = refresh_satstar_skycoords(tbl, catalog_path=cat)
    assert not np.isfinite(shift[0])
    assert len(out) == 1


@pytest.mark.parametrize('suffix', ['_m3', '_m12', '_iter2', '_resbgsub_m4'])
def test_frame_path_recovers_every_file_suffix(tmp_path, suffix):
    """The writer names the cache ``frame.replace('.fits', suffix +
    '_satstar_catalog.fits')`` for whatever the run's file suffix is, so the
    frame lookup has to strip all of them."""
    frame = tmp_path / 'jw01182004001_04101_00001_nrca1_destreak_o004_crf.fits'
    frame.write_text('')
    cat = str(frame).replace('.fits', f'{suffix}_satstar_catalog.fits')
    assert frame_path_for_satstar_catalog(cat) == str(frame)


def test_frame_path_returns_none_for_a_non_satstar_name(tmp_path):
    assert frame_path_for_satstar_catalog(str(tmp_path / 'x_crf.fits')) is None


def test_consolidated_build_reads_caches_on_the_current_frame(tmp_path):
    """``merge_catalogs.load_satstar_catalog`` builds the consolidated
    per-filter satstar catalog whose positions reach the merged photometry.  It
    must read each per-exposure cache THROUGH the frame's current WCS; reading
    it raw is how a June fit published June's astrometry from an August frame.
    """
    from jwst_gc_pipeline.photometry import merge_catalogs as MC

    fit_wcs = _wcs()
    now_wcs = _wcs(crval1=266.5 + 90e-3 / 3600.0 / np.cos(np.radians(-28.7)))
    x = np.array([25.0, 75.0])
    y = np.array([35.0, 85.0])
    frame = str(tmp_path / 'exp_crf.fits')
    _write_frame(frame, now_wcs)
    cat = str(tmp_path / 'exp_crf_m3_satstar_catalog.fits')
    _satstar_table(fit_wcs, x, y).write(cat, overwrite=True)

    got = MC._read_satstar_catalog_on_current_frame(cat, {})
    assert np.all(_seps(got, now_wcs.pixel_to_world(x, y)) < 0.01)
    assert np.all(_seps(got, fit_wcs.pixel_to_world(x, y)) > 50)


def test_consolidated_cache_generation_forces_a_rebuild():
    """The consolidated satstar cache on disk was built from raw (stale-sky)
    reads, and it is keyed by file count + dedup radius + algorithm token --
    none of which changed here.  The algorithm token must move so those caches
    rebuild instead of serving the old positions."""
    from jwst_gc_pipeline.photometry import merge_catalogs as MC
    assert MC._SATSTAR_DEDUP_ALG != 'fp3'
