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
    path, w = _make_i2d(tmp_path, pos, [400.0], nx=200, ny=200)
    # a position just BEYOND the array edge: world_to_pixel returns a FINITE but
    # out-of-range pixel, so this exercises the r_out_pix array-bounds guard
    # (not just the isfinite check).
    off = w.pixel_to_world(230, 230)
    on = w.pixel_to_world(100, 100)
    cat = Table()
    cat['skycoord_fit'] = SkyCoord([off.ra, on.ra], [off.dec, on.dec])
    # confirm the off point really is finite-but-out-of-bounds in pixel space
    ox, oy = w.world_to_pixel(off)
    assert np.isfinite(ox) and ox > 200
    out = ap.measure_aperture_photometry(cat, [path])
    assert not np.isfinite(float(out['aper_flux_jy'][0]))   # off-frame (bounds)
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


def test_aperture_photometry_enabled_toggle(monkeypatch):
    monkeypatch.delenv('SATSTAR_APERTURE_PHOT', raising=False)
    assert ap.aperture_photometry_enabled()
    for off in ('0', 'false', 'FALSE', 'no', 'off', 'Off'):
        monkeypatch.setenv('SATSTAR_APERTURE_PHOT', off)
        assert not ap.aperture_photometry_enabled()
    for on in ('1', 'true', 'yes', 'anything'):
        monkeypatch.setenv('SATSTAR_APERTURE_PHOT', on)
        assert ap.aperture_photometry_enabled()


def _touch_i2d(pipe, name):
    import os
    os.makedirs(pipe, exist_ok=True)
    fits.HDUList([fits.PrimaryHDU()]).writeto(f'{pipe}/{name}', overwrite=True)


def test_find_i2d_mosaics_rejects_and_prunes(tmp_path):
    base = str(tmp_path / 'faketarget')
    pipe = f'{base}/F200W/pipeline'
    keep_merged = 'jw09999-o001_t001_nircam_clear-f200w-merged_i2d.fits'
    keep_nrcb2 = 'jw09999-o002_t001_nircam_clear-f200w-nrcb_i2d.fits'
    for n in (keep_merged, keep_nrcb2,
              'jw09999-o001_t001_nircam_clear-f200w-merged_residual_i2d.fits',
              'jw09999-o001_t001_nircam_clear-f200w-merged-reproject_i2d.fits',
              'jw09999-o001_t001_nircam_clear-f200w-merged_m3_daophot_basic_mergedcat_residual_i2d.fits',
              'jw09999-o001_t001_nircam_clear-f200w-nrca_i2d.fits'):  # pruned: o001 has merged
        _touch_i2d(pipe, n)
    got = {__import__('os').path.basename(p)
           for p in ap.find_i2d_mosaics('f200w', 'faketarget', base)}
    assert got == {keep_merged, keep_nrcb2}


def test_add_aperture_photometry_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv('SATSTAR_APERTURE_PHOT', 'false')
    t = Table({'skycoord_fit': SkyCoord([1.0], [2.0], unit='deg')})
    out = ap.add_aperture_photometry(t, 'f200w', 'faketarget', str(tmp_path))
    assert not ap.has_aperture_columns(out)


def test_add_aperture_photometry_bad_mosaic_is_nonfatal(tmp_path):
    # a mosaic with no SCI extension must be caught, not crash the merge
    base = str(tmp_path / 'faketarget')
    pipe = f'{base}/F200W/pipeline'
    _touch_i2d(pipe, 'jw09999-o001_t001_nircam_clear-f200w-merged_i2d.fits')
    t = Table({'skycoord_fit': SkyCoord([266.0], [-28.0], unit='deg')})
    out = ap.add_aperture_photometry(t, 'f200w', 'faketarget', base)
    assert not ap.has_aperture_columns(out)   # graceful: KeyError('SCI') caught


def test_select_reference_stars_cuts():
    n = 50
    rng = np.random.default_rng(1)
    t = Table()
    t['skycoord'] = SkyCoord(266.0 + rng.uniform(0, 0.1, n),
                             -28.0 + rng.uniform(0, 0.1, n), unit='deg')
    t['flux'] = np.full(n, 1000.0)
    t['flux_err'] = np.full(n, 1.0)          # SNR 1000
    t['group_size'] = np.ones(n, int)
    t['nmatch'] = np.full(n, 3)
    t['qfit'] = np.zeros(n)
    t['is_saturated'] = np.zeros(n, bool)
    t['is_saturated'][:10] = True            # 10 saturated -> excluded
    keep = ap.select_reference_stars(t, snr_min=100, isolation_arcsec=0.0)
    assert keep.sum() == 40
    t['flux_err'][10:20] = 1000.0            # SNR 1 -> excluded
    keep = ap.select_reference_stars(t, snr_min=100, isolation_arcsec=0.0)
    assert keep.sum() == 30


def test_build_reference_apcorr_synthetic(tmp_path, monkeypatch):
    monkeypatch.setattr(ap, '_vega_zeropoint_jy', lambda f: 3631.0 * u.Jy)
    # 60 isolated bright unsaturated stars on a synthetic mosaic
    rng = np.random.default_rng(3)
    xs = rng.uniform(40, 160, 60); ys = rng.uniform(40, 160, 60)
    pos = list(zip(xs, ys))
    path, w = _make_i2d(tmp_path, pos, [800.0] * len(pos), nx=200, ny=200)
    sky = w.pixel_to_world(xs, ys)
    cat = Table({'skycoord': sky, 'flux': np.full(60, 1e4),
                 'flux_err': np.full(60, 10.0), 'group_size': np.ones(60, int),
                 'nmatch': np.full(60, 3), 'qfit': np.zeros(60),
                 'is_saturated': np.zeros(60, bool)})
    tbl = ap.build_reference_apcorr(cat, [path], 'f200w', snr_min=50,
                                    isolation_ladder=(0.0,), min_ref_stars=10,
                                    radii_arcsec=(0.05, 0.10, 0.20, 0.30))
    assert tbl.meta['n_reference_stars'] == 60
    assert 'i2d_mosaics' in tbl.meta and tbl.meta['recentered'] is True
    # enclosed flux increases with radius, =1 at the reference radius
    assert abs(float(tbl['flux_ratio_to_total'][-1]) - 1.0) < 1e-6


def test_aper_flux_valid_flags_masked_core(tmp_path):
    # clean star -> valid True; NaN-core star -> valid False (lower limit)
    path, w = _make_i2d(tmp_path, [(60, 60), (140, 140)], [500.0, 500.0],
                        nan_core=[((140, 140), 2.5)])
    cat = _catalog(w, [(60, 60), (140, 140)])
    out = ap.measure_aperture_photometry(cat, [path])
    assert 'aper_flux_valid' in out.colnames
    assert bool(out['aper_flux_valid'][0]) is True      # clean
    assert bool(out['aper_flux_valid'][1]) is False     # masked core
    assert bool(out['aper_core_saturated'][1]) is True


def test_apcorr_table_provenance_and_path():
    # diagnostic (satstar-catalog) vs refstars naming + provenance
    assert ap.apcorr_table_path('f200w', 'brick', '/b', kind='refstars') \
        .endswith('f200w_satstar_apcorr_refstars.ecsv')
    assert ap.apcorr_table_path('f200w', 'brick', '/b', kind='diagnostic') \
        .endswith('f200w_satstar_apcorr_diagnostic.ecsv')


def test_is_science_mosaic_positive_matcher():
    f, inst = 'f200w', 'nircam'
    good = ['jw01182-o004_t001_nircam_clear-f200w-merged_i2d.fits',
            'jw02211-o050_t001_nircam_clear-f200w-nrcb_i2d.fits',
            'jw02221-o001_t001_nircam_f405n-f444w_i2d.fits']  # dual-filter (f405n)
    bad = ['jw01182-o004_t001_nircam_clear-f200w-merged_residual_i2d.fits',
           'jw01182-o004_t001_nircam_clear-f200w-merged-reproject_i2d.fits',
           'jw01182-o004_t001_nircam_clear-f200w-merged_m3_daophot_basic_mergedcat_residual_i2d.fits',
           'jw01182-o004_t001_nircam_clear-f200w-merged_data_i2d.fits']
    assert ap._is_science_mosaic(good[0], f, inst)
    assert ap._is_science_mosaic(good[1], f, inst)
    assert ap._is_science_mosaic(good[2], 'f405n', inst)
    for b in bad:
        assert not ap._is_science_mosaic(b, f, inst), b


def test_zeropoint_failure_keeps_flux(tmp_path, monkeypatch):
    # an SVO outage must cost only mag_vega, not the aperture flux
    from astroquery.exceptions import InvalidQueryError

    def _boom(filt):
        raise InvalidQueryError("SVO down")
    monkeypatch.setattr(ap, '_vega_zeropoint_jy', _boom)
    path, w = _make_i2d(tmp_path, [(100, 100)], [500.0])
    cat = _catalog(w, [(100, 100)])
    out = ap.measure_aperture_photometry(cat, [path], filtername='f200w')
    assert np.isfinite(float(out['aper_flux_jy'][0]))        # flux kept
    assert not np.isfinite(float(out['aper_mag_vega'][0]))   # mag_vega NaN
    assert np.isfinite(float(out['aper_mag_ab'][0]))         # AB needs no zeropoint


def test_all_nan_drops_aperture_columns(tmp_path):
    # a valid mosaic but off-footprint positions -> all-NaN -> columns DROPPED so
    # nothing (cache-hit OR rebuild path) persists a false success
    from jwst_gc_pipeline.photometry import merge_catalogs as mc
    base = str(tmp_path / 'faketarget')
    pipe = f'{base}/F200W/pipeline'
    import os
    os.makedirs(pipe, exist_ok=True)
    path, w = _make_i2d(tmp_path, [(100, 100)], [500.0])
    os.replace(path, f'{pipe}/jw09999-o001_t001_nircam_clear-f200w-merged_i2d.fits')
    off = w.pixel_to_world(5000, 5000)          # far outside the 200x200 mosaic
    cat = Table({'skycoord_fit': SkyCoord([off.ra], [off.dec])})
    out = mc._ensure_satstar_aperture_photometry(cat, 'f200w', 'faketarget',
                                                 base, cache_path=None)
    assert not ap.has_aperture_columns(out)      # dropped, not a false success


def test_partial_coverage_below_threshold_is_invalid(tmp_path):
    # a small ring of masked pixels near (not on) the center -> core finite but
    # coverage just under 1.0 -> aper_flux_valid must be False (N4: the 0.999 cut)
    import numpy as _np
    path, w = _make_i2d(tmp_path, [(100, 100)], [500.0])
    with fits.open(path) as h:
        sci = h['SCI'].data
        yy, xx = _np.mgrid[0:sci.shape[0], 0:sci.shape[1]]
        r2 = (xx - 100) ** 2 + (yy - 100) ** 2
        sci[(r2 >= 9) & (r2 <= 16)] = _np.nan   # annulus of NaN inside the aperture
        h.writeto(path, overwrite=True)
    cat = _catalog(w, [(100, 100)])
    out = ap.measure_aperture_photometry(cat, [path])
    cov = float(out['aper_area_frac'][0])
    assert cov < 0.999                       # partial coverage (masked pixels)
    assert not bool(out['aper_core_saturated'][0])  # core itself finite
    assert not bool(out['aper_flux_valid'][0])      # still invalid via coverage


def test_dual_filter_mosaic_not_claimed_by_wide_partner():
    # f405n-f444w is F405N's science mosaic; must NOT be accepted for f444w (N5)
    b = 'jw02221-o001_t001_nircam_f405n-f444w_i2d.fits'
    assert ap._is_science_mosaic(b, 'f405n', 'nircam')       # narrow owner: yes
    assert not ap._is_science_mosaic(b, 'f444w', 'nircam')   # wide partner: no
    # the wide band still resolves its own merged mosaic
    assert ap._is_science_mosaic(
        'jw01182-o004_t001_nircam_clear-f444w-merged_i2d.fits', 'f444w', 'nircam')
