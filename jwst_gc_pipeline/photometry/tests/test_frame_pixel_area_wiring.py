"""The call sites of the per-frame pixel area (#1238).

``test_frame_pixel_area.py`` pins the arithmetic of ``frame_pixel_area``.  These
tests pin that the products use it: the consolidated satstar cache, the
gate-rejected satstar channel the gray-zone clip correction adopts from,
``replace_saturated``'s conversion, ``merge_crowdsource`` and the m8 forced fill
(default argument and forwarding).  A refactor that dropped one of those calls
would leave the arithmetic tests green and put a detector's pixel-area offset
(up to 34 mmag on NIRCam short-wave) back into the catalog.

Synthetic and CPU-only.
"""
import types

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.photometry import forced_fill as ff
from jwst_gc_pipeline.photometry import frame_pixel_area as FPA
from jwst_gc_pipeline.photometry import merge_catalogs as MC
from jwst_gc_pipeline.photometry.tests.test_forced_fill import _build
from jwst_gc_pipeline.photometry.tests.test_forced_fill_procs import (
    FRAMES, _frames_prepare)
from jwst_gc_pipeline.photometry.tests.test_satstar_cache_sky_refresh import (
    _satstar_table, _wcs, _write_frame)

TRUE_JY = 2.0e-3
#: PIXAR_SR of two SW detectors (NRCB1, NRCA3)
AREA_A, AREA_B = 2.2217e-14, 2.3121e-14
DEG2_PER_SR = FPA.DEG2_PER_SR

def _close(expected, rel=1e-9):
    """approx with no absolute slack: pytest's default abs=1e-12 exceeds a pixel
    area in sr (~2e-14), which would make a comparison of areas vacuous."""
    return pytest.approx(expected, rel=rel, abs=0.0)



def _set_area(path, area_sr, instrument='NIRCAM'):
    with fits.open(path, mode='update') as hdul:
        hdul[0].header['INSTRUME'] = instrument
        hdul['SCI'].header['PIXAR_SR'] = area_sr


def _pipeline_dir(tmp_path, filt='F212N'):
    pdir = tmp_path / filt / 'pipeline'
    pdir.mkdir(parents=True)
    (tmp_path / 'catalogs').mkdir()
    return pdir


def _frame_with_area(pdir, detector, area_sr, wcs, instrument='NIRCAM'):
    frame = pdir / f'jw02221001001_02101_00001_{detector}_o001_crf.fits'
    _write_frame(str(frame), wcs)
    _set_area(str(frame), area_sr, instrument)
    return frame


class _FakeSvo:
    @staticmethod
    def get_filter_list(_):
        t = Table({'filterID': ['JWST/NIRCam.F212N', 'JWST/NIRCam.F182M'],
                   'ZeroPoint': [850.0, 850.0]})
        t.add_index('filterID')
        return t


# --- load_satstar_catalog ---------------------------------------------------

def test_load_satstar_catalog_puts_the_exposures_on_one_area_and_stamps_it(tmp_path):
    """One star seen on two detectors reads the same true Jy; the cache must carry
    the reference area it is expressed in, in deg^2 (not sr)."""
    pdir = _pipeline_dir(tmp_path)
    wcs = _wcs()
    x, y = np.array([40.0]), np.array([60.0])
    for det, area in (('nrcb1', AREA_A), ('nrca3', AREA_B)):
        frame = _frame_with_area(pdir, det, area, wcs)
        t = _satstar_table(wcs, x, y)
        t['flux_fit'] = TRUE_JY / (area * 1e6)
        t['flux_err'] = 0.01 * TRUE_JY / (area * 1e6)
        t['flux_fit_raw'] = 1.1 * TRUE_JY / (area * 1e6)
        t.write(str(frame).replace('.fits', '_m6_satstar_catalog.fits'))

    out = MC.load_satstar_catalog('f212n', target='brick', basepath=str(tmp_path) + '/')

    assert len(out) == 1
    area_ref_sr = 0.031 ** 2 * FPA.SR_PER_ARCSEC2
    assert out.meta['PIXAREF'] == _close(area_ref_sr * DEG2_PER_SR)
    jy = float(out['flux_fit'][0]) * area_ref_sr * 1e6
    assert jy == pytest.approx(TRUE_JY, rel=1e-6)
    assert float(out['flux_err'][0]) * area_ref_sr * 1e6 == pytest.approx(
        0.01 * TRUE_JY, rel=1e-6)
    # the pre-calibration provenance column stays on the same scale as flux_fit
    assert float(out['flux_fit_raw'][0]) * area_ref_sr * 1e6 == pytest.approx(
        1.1 * TRUE_JY, rel=1e-6)


# --- the gate-rejected channel ---------------------------------------------

def _rejected_table(wcs, x, y, flux):
    t = _satstar_table(wcs, x, y)
    t['flux_fit'] = np.asarray(flux, dtype=float)
    t['flux_err'] = 0.05 * np.asarray(flux, dtype=float)
    t['qfit'] = 0.3
    t['reject_reason'] = 'implied_peak_gate'
    return t


def _rejected_tree(tmp_path, area_sr, flux):
    pdir = _pipeline_dir(tmp_path)
    wcs = _wcs()
    x, y = np.array([40.0]), np.array([60.0])
    frame = _frame_with_area(pdir, 'nrcb1', area_sr, wcs)
    base = str(frame).replace('.fits', '_m6_satstar_')
    _satstar_table(wcs, x, y).write(base + 'catalog.fits')
    _rejected_table(wcs, x, y, [flux]).write(base + 'rejected.fits')
    return wcs, x, y


def test_rejected_satstar_fluxes_are_rescaled_like_the_accepted_ones(tmp_path):
    area_ref = 0.031 ** 2 * FPA.SR_PER_ARCSEC2
    _rejected_tree(tmp_path, 1.04 * area_ref, 1000.0)
    rej = MC.load_rejected_satstar_catalog('f212n', target='brick',
                                           basepath=str(tmp_path) + '/')
    assert rej is not None and len(rej) == 1
    assert float(rej['flux_fit'][0]) == pytest.approx(1040.0, rel=1e-6)
    assert float(rej['flux_err'][0]) == pytest.approx(52.0, rel=1e-6)


def test_gray_zone_clip_correction_adopts_the_rescaled_flux(tmp_path, monkeypatch):
    """The 0.02-0.5 mag window of the gray-zone correction compares the rejected
    flux with the catalog's, which is on the reference area.  A row at +0.0207
    mag only qualifies, and is only adopted, once the rejected flux is rescaled."""
    area_ref = 0.031 ** 2 * FPA.SR_PER_ARCSEC2
    wcs, x, y = _rejected_tree(tmp_path, 1.04 * area_ref, 1000.0)
    monkeypatch.setattr(MC, 'load_satstar_catalog', lambda *a, **k: _far_satstar())
    monkeypatch.setattr(MC, 'SvoFps', _FakeSvo)
    (tmp_path / 'reduction').mkdir()
    Table({'Filter': ['F212N'], 'PSF FWHM (arcsec)': [0.06]}).write(
        tmp_path / 'reduction' / 'fwhm_table.ecsv', format='ascii.ecsv')

    star = wcs.pixel_to_world(x, y)
    cat = Table({'skycoord': star, 'flux': [1020.0], 'dflux': [5.0],
                 'x': [40.0], 'y': [60.0], 'dx': [0.01], 'dy': [0.01]})
    cat.meta['pixelscale_deg2'] = area_ref * DEG2_PER_SR * u.deg ** 2
    cat.meta[FPA.PIXAREF_KEY] = area_ref * DEG2_PER_SR
    MC.replace_saturated(cat, 'f212n', basepath=str(tmp_path) + '/',
                         fwhm_basepath=str(tmp_path), target='brick')

    assert bool(cat['satclip_corrected'][0])
    assert float(cat['flux'][0]) == pytest.approx(1040.0, rel=1e-6)
    assert float(cat['satclip_corr_mag'][0]) == pytest.approx(
        2.5 * np.log10(1040.0 / 1020.0), rel=1e-6)


def _far_satstar():
    """A satstar table with one star 1 degree from the test field."""
    return Table({
        'skycoord_fit': SkyCoord([267.5] * u.deg, [-28.7] * u.deg),
        'flux_fit': [5000.0], 'flux_err': [10.0],
        'x_fit': [81.0], 'y_fit': [81.0], 'x_err': [0.01], 'y_err': [0.01],
        'xcentroid': [900.0], 'ycentroid': [900.0]})


# --- replace_saturated's conversion ----------------------------------------

def _replace(tmp_path, monkeypatch, satstar, cat_area_deg2):
    monkeypatch.setattr(MC, 'load_satstar_catalog', lambda *a, **k: satstar)
    monkeypatch.setattr(MC, 'SvoFps', _FakeSvo)
    (tmp_path / 'reduction').mkdir(exist_ok=True)
    Table({'Filter': ['F212N'], 'PSF FWHM (arcsec)': [0.06]}).write(
        tmp_path / 'reduction' / 'fwhm_table.ecsv', format='ascii.ecsv')
    cat = Table({'skycoord': SkyCoord([266.5] * u.deg, [-28.7] * u.deg),
                 'flux': [1000.0], 'dflux': [5.0],
                 'mag_ab': [20.0] * u.mag, 'mag_vega': [18.0] * u.mag,
                 'emag_ab': [0.1] * u.mag,
                 'x': [512.0], 'y': [300.0], 'dx': [0.01], 'dy': [0.01]})
    cat.meta['pixelscale_deg2'] = cat_area_deg2 * u.deg ** 2
    MC.replace_saturated(cat, 'f212n', basepath=str(tmp_path) + '/',
                         fwhm_basepath=str(tmp_path), target='brick')
    return cat


def _satstar_with_area(area_deg2):
    t = _far_satstar()
    t['skycoord_fit'] = SkyCoord([266.5] * u.deg, [-28.7] * u.deg)
    t['flux_fit'] = [5000.0]
    if area_deg2 is not None:
        t.meta[FPA.PIXAREF_KEY] = area_deg2
    return t


def test_replace_saturated_converts_satstar_rows_with_the_area_they_carry(
        tmp_path, monkeypatch, capsys):
    area_sat = 0.031 ** 2 * FPA.SR_PER_ARCSEC2 * DEG2_PER_SR
    # the catalog was converted with a 1.5 % larger area
    cat = _replace(tmp_path, monkeypatch, _satstar_with_area(area_sat),
                   area_sat * 1.015)
    out = capsys.readouterr().out
    replaced = cat[0]
    flux_jy = 5000.0 * u.MJy / u.sr * (area_sat * u.deg ** 2)
    expect_ab = -2.5 * np.log10(flux_jy.to_value(u.Jy)) + MC.ABMAG_OFFSET
    assert float(replaced['mag_ab']) == pytest.approx(expect_ab, abs=1e-6)
    # ... and says that the rows it substitutes are off by the ratio
    assert 'the satstar fluxes are on pixel area' in out


def test_replace_saturated_is_quiet_when_the_areas_agree(tmp_path, monkeypatch, capsys):
    area_sat = 0.031 ** 2 * FPA.SR_PER_ARCSEC2 * DEG2_PER_SR
    _replace(tmp_path, monkeypatch, _satstar_with_area(area_sat), area_sat)
    assert 'the satstar fluxes are on pixel area' not in capsys.readouterr().out


# --- merge_crowdsource ------------------------------------------------------

class _Stop(Exception):
    pass


def _crowdsource_run(tmp_path, monkeypatch, meta_area_deg2, i2d_side_arcsec=0.030):
    filt, module = 'f212n', 'nrca1'
    (tmp_path / filt.upper() / 'pipeline').mkdir(parents=True)
    (tmp_path / filt.upper() / 'pipeline' /
     'jw02221-o001_t001_nircam_clear-f212n-nrca1_i2d.fits').write_bytes(b'')
    (tmp_path / 'catalogs').mkdir()
    t = Table({'x': [1.0, 2.0], 'y': [1.0, 2.0], 'qf': [1.0, 1.0],
               'flux': [1000.0, 2000.0], 'dflux': [10.0, 20.0], 'fwhm': [2.0, 2.0]})
    if meta_area_deg2 is not None:
        t.meta['PIXAREF'] = meta_area_deg2
    t.write(tmp_path / 'catalogs' / f'{filt}_{module}_o001_indivexp_merged_crowdsource.fits')

    class _Wcs:
        def proj_plane_pixel_area(self):
            return (i2d_side_arcsec * u.arcsec).to(u.deg) ** 2

        def pixel_to_world(self, x, y):
            return SkyCoord(266.5 * u.deg + np.asarray(x) * 1e-5 * u.deg,
                            -28.7 * u.deg + np.asarray(y) * 1e-5 * u.deg)

    monkeypatch.setattr(MC, 'frame_wcs', lambda fn: _Wcs())
    monkeypatch.setattr(MC, '_obs_filters_for', lambda target: {'2221': [filt]})
    monkeypatch.setattr(MC, '_obsid_for_glob', lambda *a, **k: '001')
    monkeypatch.setattr(MC, 'sanity_check_individual_table', lambda t: None)
    monkeypatch.setattr(MC, 'SvoFps', _FakeSvo)
    seen = []

    def _capture(tbls, **kw):
        seen.extend(tbls)
        raise _Stop

    monkeypatch.setattr(MC, 'merge_catalogs', _capture)
    with pytest.raises(_Stop):
        MC.merge_crowdsource(module=module, target='brick', indivexp=True,
                             basepath=str(tmp_path), field='001')
    return seen[0]


def test_merge_crowdsource_converts_with_pixaref_not_the_i2d_area(tmp_path, monkeypatch):
    stamp_deg2 = ((0.031 * u.arcsec).to(u.deg) ** 2).value
    tbl = _crowdsource_run(tmp_path, monkeypatch, stamp_deg2)
    assert tbl.meta['pixelscale_deg2'].to_value(u.deg ** 2) == _close(stamp_deg2)
    assert tbl.meta['pixelscale_arcsec'].to_value(u.arcsec) == pytest.approx(0.031)
    np.testing.assert_allclose(
        np.asarray(tbl['flux_jy'], dtype=float),
        (np.array([1000.0, 2000.0]) * u.MJy / u.sr * stamp_deg2 * u.deg ** 2).to_value(u.Jy))


def test_merge_crowdsource_keeps_the_i2d_area_without_a_stamp(tmp_path, monkeypatch):
    tbl = _crowdsource_run(tmp_path, monkeypatch, None)
    assert tbl.meta['pixelscale_arcsec'].to_value(u.arcsec) == pytest.approx(0.030)


# --- m8 forced fill ---------------------------------------------------------

def _crf_names(tmp_path, area_sr):
    names = []
    for fn in FRAMES:
        path = tmp_path / fn
        fits.HDUList([fits.PrimaryHDU(),
                      fits.ImageHDU(np.zeros((2, 2)), name='SCI')]).writeto(path)
        _set_area(str(path), area_sr)
        names.append(str(path))
    return names


def _filled_flux_jy(tbl, ndet):
    return np.asarray(tbl['flux_jy_f405n'][ndet:], dtype=float)


def test_forced_fill_band_forwards_the_frame_area(tmp_path):
    area = 2.3e-14
    names = _crf_names(tmp_path, area)
    prep = _frames_prepare()
    raw, fwd = (_build()[0] for _ in range(2))
    ndet = len(raw) - 2
    kw = dict(prepare_frame=lambda name: prep(FRAMES[names.index(name)]),
              frame_arg_builder=lambda fn: {'name': fn}, nsigma=3.0, fit_shape=(5, 5),
              verbose=False)
    ff.forced_fill_band(raw, 'f405n', names, **kw)
    ff.forced_fill_band(fwd, 'f405n', names, frame_area_sr=FPA.frame_area_sr, **kw)
    conv_jy = 3.0e-8        # _build's calibration
    np.testing.assert_allclose(_filled_flux_jy(fwd, ndet) / _filled_flux_jy(raw, ndet),
                               area * 1e6 / conv_jy, rtol=1e-6)


def test_run_forced_crossband_fill_uses_each_frames_pixar_sr_by_default(tmp_path):
    area = 2.3e-14
    names = _crf_names(tmp_path, area)
    prep = _frames_prepare()
    tbl, *_ = _build()
    ndet = len(tbl) - 2
    merged = str(tmp_path / 'm7.fits')
    tbl.write(merged)
    outs = {}
    for label, extra in (('default', {}), ('raw', {'frame_area_sr': None})):
        out = str(tmp_path / f'm8_{label}.fits')
        ff.run_forced_crossband_fill(
            merged, out, filternames=['F405N'], module='merged',
            frames_for=lambda filt: names,
            prepare_frame=lambda name: prep(FRAMES[names.index(name)]),
            frame_arg_builder_for=lambda filt: (lambda fn: {'name': fn}),
            verbose=False, **extra)
        outs[label] = Table.read(out)
    np.testing.assert_allclose(
        _filled_flux_jy(outs['default'], ndet) / _filled_flux_jy(outs['raw'], ndet),
        area * 1e6 / 3.0e-8, rtol=1e-6)


def test_forced_fill_warns_when_the_m7_header_lacks_pixaref(capsys):
    def run(meta):
        tbl, prepare_frame, *_ = _build()
        tbl.meta.update(meta)
        ff.forced_fill_band(tbl, 'f405n', ['frame1'], prepare_frame=prepare_frame,
                            frame_arg_builder=lambda fn: {}, fit_shape=(5, 5),
                            verbose=False, frame_area_sr=lambda fn: 2.3e-14)
        return capsys.readouterr().out
    assert 'has no 405PIXA' in run({})
    assert 'has no 405PIXA' not in run({'405PIXA': 7.4e-11})
