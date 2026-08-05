"""Unit tests for aperture_photometry (synthetic mosaics, no CRDS/network)."""
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from jwst_gc_pipeline.photometry import aperture_photometry as ap


def _make_i2d(tmp_path, positions_xy, flux_mjysr_pix, nx=200, ny=200,
              pixscale_arcsec=0.06, nan_core=None, bkg=0.0):
    """A tiny MJy/sr mosaic with delta-ish Gaussian stars; returns path+wcs.

    flux_mjysr_pix is the desired SUMMED SCI (MJy/sr) per star (i.e. before the
    pixel-area conversion), placed as a narrow Gaussian so a small aperture
    captures ~all of it.
    """
    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2, ny / 2]
    w.wcs.cdelt = [-pixscale_arcsec / 3600, pixscale_arcsec / 3600]
    w.wcs.crval = [266.0, -28.0]
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    sci = np.full((ny, nx), bkg, float)
    yy, xx = np.mgrid[0:ny, 0:nx]
    sig = 1.2
    for (x0, y0), f in zip(positions_xy, flux_mjysr_pix):
        g = np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sig ** 2))
        g *= f / g.sum()
        sci += g
    if nan_core:
        for (x0, y0), rad in nan_core:
            m = (xx - x0) ** 2 + (yy - y0) ** 2 <= rad ** 2
            sci[m] = np.nan
    err = np.full((ny, nx), 0.01, float)
    hdr = w.to_header()
    hdr['BUNIT'] = 'MJy/sr'
    ph = fits.PrimaryHDU()
    hs = fits.ImageHDU(sci, hdr, name='SCI')
    he = fits.ImageHDU(err, hdr, name='ERR')
    path = str(tmp_path / 'jw09999-o001_t001_nircam_clear-f200w-merged_i2d.fits')
    fits.HDUList([ph, hs, he]).writeto(path, overwrite=True)
    return path, w


def _catalog(w, positions_xy):
    sky = w.pixel_to_world([p[0] for p in positions_xy],
                           [p[1] for p in positions_xy])
    t = Table()
    t['skycoord_fit'] = sky
    return t


def test_recovers_known_flux(tmp_path):
    pos = [(100, 100), (60, 140)]
    fsum = [500.0, 200.0]           # MJy/sr summed
    path, w = _make_i2d(tmp_path, pos, fsum)
    cat = _catalog(w, pos)
    out = ap.measure_aperture_photometry(cat, [path], filtername=None)
    assert ap.has_aperture_columns(out)
    pa = w.proj_plane_pixel_area()
    expect_jy = [(f * u.MJy / u.sr * pa).to(u.Jy).value for f in fsum]
    got = np.asarray(out['aper_flux_jy'], float)
    # 0.3" aperture on a sigma=1.2px (~0.07") gaussian captures ~all flux
    np.testing.assert_allclose(got, expect_jy, rtol=0.03)
    assert np.all(np.asarray(out['aper_area_frac'], float) > 0.99)
    assert not np.any(out['aper_core_saturated'])


def test_background_subtracted(tmp_path):
    pos = [(100, 100)]
    path, w = _make_i2d(tmp_path, pos, [300.0], bkg=2.0)
    cat = _catalog(w, pos)
    out = ap.measure_aperture_photometry(cat, [path])
    pa = w.proj_plane_pixel_area()
    expect = (300.0 * u.MJy / u.sr * pa).to(u.Jy).value
    got = float(out['aper_flux_jy'][0])
    # background (2.0/pix over the aperture) must be removed
    assert abs(got - expect) / expect < 0.05
    assert abs(float(out['aper_bkg'][0]) - 2.0) < 0.1


def test_nan_core_flagged_and_coverage(tmp_path):
    pos = [(100, 100)]
    path, w = _make_i2d(tmp_path, pos, [400.0], nan_core=[((100, 100), 2.5)])
    cat = _catalog(w, pos)
    out = ap.measure_aperture_photometry(cat, [path])
    assert bool(out['aper_core_saturated'][0])
    assert float(out['aper_area_frac'][0]) < 1.0     # NaN core reduces coverage
    assert np.isfinite(float(out['aper_flux_jy'][0]))  # still measurable (NaN-aware)


def test_offframe_positions_nan(tmp_path):
    pos = [(100, 100)]
    path, w = _make_i2d(tmp_path, pos, [400.0])
    # a catalog position far outside the frame
    off = SkyCoord(10.0 * u.deg, 10.0 * u.deg)
    cat = Table()
    cat['skycoord_fit'] = SkyCoord([off.ra, w.pixel_to_world(100, 100).ra],
                                   [off.dec, w.pixel_to_world(100, 100).dec])
    out = ap.measure_aperture_photometry(cat, [path])
    assert not np.isfinite(float(out['aper_flux_jy'][0]))   # off-frame
    assert np.isfinite(float(out['aper_flux_jy'][1]))       # on-frame


def test_curve_of_growth_monotonic(tmp_path):
    pos = [(100, 100)]
    path, w = _make_i2d(tmp_path, pos, [500.0])
    cat = _catalog(w, pos)
    out = ap.measure_aperture_photometry(cat, [path])
    cog = [float(out[f'aper_flux_jy_r{ap._rtag(r)}'][0]) for r in ap.RADII_ARCSEC]
    # enclosed flux must be non-decreasing with radius (within noise)
    assert all(b >= a - 1e-3 * abs(a) for a, b in zip(cog, cog[1:]))


def test_apcorr_table_shape(tmp_path):
    rng = np.random.default_rng(0)
    xs = rng.uniform(30, 170, 40)
    ys = rng.uniform(30, 170, 40)
    pos = list(zip(xs, ys))
    path, w = _make_i2d(tmp_path, pos, [400.0] * len(pos), nx=200, ny=200)
    cat = _catalog(w, pos)
    out = ap.measure_aperture_photometry(cat, [path])
    tbl = ap.build_aperture_correction_table(out, filtername='f200w',
                                             min_snr=0, isolation_arcsec=0)
    assert len(tbl) == len(ap.RADII_ARCSEC)
    assert set(('radius_arcsec', 'flux_ratio_to_total',
                'apcorr_mag', 'ratio_mad', 'n_stars')) <= set(tbl.colnames)
    # ratio at the reference (largest) radius is 1.0 by construction
    assert abs(float(tbl['flux_ratio_to_total'][-1]) - 1.0) < 1e-6
