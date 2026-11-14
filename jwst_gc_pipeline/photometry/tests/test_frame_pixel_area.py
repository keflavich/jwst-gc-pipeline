"""Per-frame pixel area in the single-frame merges (#1238).

A per-frame ``flux_fit`` is a sum of MJy/sr over that detector's native pixels,
so its Jy flux needs that detector's ``PIXAR_SR``.  The eight NIRCam SW
detectors span 2.222-2.312e-14 sr (4 %).  ``merge_individual_frames`` used to
average the frames in image units and convert once.  These tests build frames
of one star with the same true flux in Jy but different detector areas and
check that the flux columns are brought to one reference area before the
average, and that every later conversion honours that reference.

Synthetic and CPU-only: the "crf" files carry just a SCI header.
"""
from types import SimpleNamespace

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.photometry import frame_pixel_area as FPA
from jwst_gc_pipeline.photometry import merge_catalogs as MC

RA0, DEC0 = 266.5, -28.8
COSD = np.cos(np.deg2rad(DEC0))
TRUE_JY = 2.0e-3
#: PIXAR_SR of three SW detectors (NRCB1, NRCA4, NRCA3)
AREAS_SR = [2.2217e-14, 2.2455e-14, 2.3121e-14]


def _crf(tmp_path, name, area_sr, instrument='NIRCAM'):
    path = tmp_path / f'{name}.fits'
    prim = fits.PrimaryHDU()
    prim.header['INSTRUME'] = instrument
    sci = fits.ImageHDU(data=np.zeros((4, 4)), name='SCI')
    if area_sr is not None:
        sci.header['PIXAR_SR'] = area_sr
    fits.HDUList([prim, sci]).writeto(path, overwrite=True)
    return str(path)


def _frame(tmp_path, k, area_sr, pixscale=None, with_file=True, header_area=True):
    """One daophot exposure catalog: the star at the origin and a reference
    star 20" east, both of ``TRUE_JY``."""
    flux = np.full(2, TRUE_JY / (area_sr * 1e6), dtype='float32')
    ra = RA0 + np.array([0.0, 20.0]) / 3600.0 / COSD
    dec = np.full(2, DEC0)
    t = Table()
    t['id'] = [1, 2]
    t['skycoord_centroid'] = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)
    t['flux_fit'] = flux
    t['flux_err'] = (0.01 * flux).astype('float32')
    t['flux_init'] = flux.copy()
    t['qfit'] = np.full(2, 0.1, dtype='float32')
    t['cfit'] = np.zeros(2, dtype='float32')
    t['flags'] = np.zeros(2, dtype='int32')
    t['local_bkg'] = np.array([3.5, 4.5], dtype='float32')
    t.meta.update(exposure=k + 1, MODULE='nrca', filter='f212n',
                  ra_offset=0.0 * u.arcsec, dec_offset=0.0 * u.arcsec)
    if with_file:
        t.meta['FILENAME'] = _crf(tmp_path, f'frame{k}', area_sr if header_area else None)
    if pixscale is not None:
        t.meta['pixscale'] = pixscale
    return t


def _frames(tmp_path):
    return [_frame(tmp_path, k, a) for k, a in enumerate(AREAS_SR)]


def _jy(tbl, area_sr, col='flux_fit'):
    return np.asarray(tbl[col], dtype=float) * area_sr * 1e6


def test_frames_disagree_in_jy_without_the_rescale_and_agree_with_it(tmp_path):
    tbls = _frames(tmp_path)
    # the bug: the one conversion area is the first frame's, so the frames read
    # different Jy (about 4 % between the extreme detectors)
    before = np.array([_jy(t, AREAS_SR[0])[0] for t in tbls])
    assert (before.max() / before.min() - 1) > 0.03

    area_ref = FPA.apply_frame_pixel_area(tbls, 'f212n')

    after = np.array([_jy(t, area_ref)[0] for t in tbls])
    np.testing.assert_allclose(after, TRUE_JY, rtol=1e-6)
    for t in tbls:
        np.testing.assert_allclose(_jy(t, area_ref, 'flux_err'), 0.01 * TRUE_JY,
                                   rtol=1e-6)
        np.testing.assert_allclose(_jy(t, area_ref, 'flux_init'), TRUE_JY, rtol=1e-6)


def test_reference_is_a_constant_of_the_filter_not_of_the_frames(tmp_path):
    ref_a = FPA.apply_frame_pixel_area(_frames(tmp_path), 'f212n')
    ref_b = FPA.apply_frame_pixel_area([_frame(tmp_path, 9, AREAS_SR[2])], 'f212n')
    assert ref_a == ref_b == pytest.approx(0.031 ** 2 * FPA.SR_PER_ARCSEC2)
    assert FPA.reference_pixel_area_sr('f480m') == pytest.approx(
        0.063 ** 2 * FPA.SR_PER_ARCSEC2)
    assert FPA.reference_pixel_area_sr('f770w') == pytest.approx(
        0.11 ** 2 * FPA.SR_PER_ARCSEC2)
    # the instrument keyword decides where the filter name is ambiguous
    assert FPA.reference_pixel_area_sr('f480m', 'NIRISS') == pytest.approx(
        0.0656 ** 2 * FPA.SR_PER_ARCSEC2)


def test_surface_brightness_columns_and_dtypes_are_untouched(tmp_path):
    tbls = _frames(tmp_path)
    FPA.apply_frame_pixel_area(tbls, 'f212n')
    for t in tbls:
        np.testing.assert_array_equal(t['local_bkg'], np.array([3.5, 4.5], dtype='float32'))
        assert t['flux_fit'].dtype == np.float32
        assert t['flux_err'].dtype == np.float32


def test_stamp_is_a_plain_float_in_deg2_and_pixscale_follows(tmp_path):
    tbls = [_frame(tmp_path, 0, AREAS_SR[0], pixscale=0.0307)]
    area_ref = FPA.apply_frame_pixel_area(tbls, 'f212n')
    meta = tbls[0].meta
    assert type(meta['PIXAREF']) is float
    assert meta['PIXAREF'] == pytest.approx(area_ref * (180 / np.pi) ** 2)
    assert meta['pixscale'] == pytest.approx(0.031)


def test_second_call_changes_nothing(tmp_path):
    tbls = _frames(tmp_path)
    FPA.apply_frame_pixel_area(tbls, 'f212n')
    once = [np.array(t['flux_fit']) for t in tbls]
    assert FPA.apply_frame_pixel_area(tbls, 'f212n') is None
    for t, ref in zip(tbls, once):
        np.testing.assert_array_equal(t['flux_fit'], ref)


def test_combined_flux_converts_to_the_true_flux_with_the_stamped_area(tmp_path):
    tbls = _frames(tmp_path)
    area_ref = FPA.apply_frame_pixel_area(tbls, 'f212n')
    out = MC.combine_singleframe(tbls, nanaverage=MC.nanaverage_numpy,
                                 verbose=False, filtername='f212n')
    assert 'PIXAREF' in out.meta
    area_sr = out.meta['PIXAREF'] / (180 / np.pi) ** 2
    assert area_sr == pytest.approx(area_ref)
    np.testing.assert_allclose(_jy(out, area_sr, 'flux_fit_avg'), TRUE_JY, rtol=1e-6)


def test_pixscale_stands_in_when_pixar_sr_is_absent(tmp_path):
    side = np.sqrt(AREAS_SR[1] / FPA.SR_PER_ARCSEC2)
    t = _frame(tmp_path, 0, AREAS_SR[1], pixscale=side, header_area=False)
    area, _inst, source = FPA.frame_pixel_area(t.meta['FILENAME'], t.meta)
    assert source == 'pixscale'
    assert area == pytest.approx(AREAS_SR[1])


def test_missing_area_raises(tmp_path):
    t = _frame(tmp_path, 0, AREAS_SR[0], header_area=False)
    with pytest.raises(FPA.FramePixelAreaError):
        FPA.apply_frame_pixel_area([t], 'f212n')
    t2 = _frame(tmp_path, 1, AREAS_SR[0], with_file=False)
    with pytest.raises(FPA.FramePixelAreaError):
        FPA.apply_frame_pixel_area([t2], 'f212n')


def test_crowdsource_columns_are_rescaled(tmp_path):
    t = Table({'flux': np.array([1.0, 2.0], dtype='float32'),
               'dflux': np.array([0.1, 0.2], dtype='float32'),
               'fluxiso': np.array([1.0, 2.0], dtype='float32'),
               'qf': np.ones(2, dtype='float32'),
               'sky': np.array([5.0, 6.0], dtype='float32')})
    t.meta['FILENAME'] = _crf(tmp_path, 'cs', AREAS_SR[2])
    area_ref = FPA.apply_frame_pixel_area([t], 'f212n')
    k = AREAS_SR[2] / area_ref
    np.testing.assert_allclose(t['flux'], [1.0 * k, 2.0 * k], rtol=1e-6)
    np.testing.assert_allclose(t['dflux'], [0.1 * k, 0.2 * k], rtol=1e-6)
    np.testing.assert_allclose(t['fluxiso'], [1.0 * k, 2.0 * k], rtol=1e-6)
    np.testing.assert_array_equal(t['sky'], np.array([5.0, 6.0], dtype='float32'))


# --- merge_daophot's conversion area ------------------------------------

class _FakeI2dWcs:
    """Stands in for an i2d WCS whose pixel is 0.030" (area 9e-4 arcsec^2)."""
    def proj_plane_pixel_area(self):
        return (0.030 * u.arcsec).to(u.deg) ** 2


def test_stamped_area_has_precedence_over_the_i2d_area():
    stamp = (0.031 * u.arcsec).to(u.deg) ** 2
    meta = {'PIXAREF': float(stamp.value), 'filter': 'f212n'}
    area, side = MC._conversion_pixel_area(meta, _FakeI2dWcs())
    assert area.to_value(u.deg ** 2) == pytest.approx(stamp.value)
    assert side.to_value(u.arcsec) == pytest.approx(0.031)


def test_unstamped_table_keeps_the_legacy_order_and_warns(capsys):
    meta = {'filter': 'f212n', 'PIXSCALE': 0.0312}
    area, _ = MC._conversion_pixel_area(meta, _FakeI2dWcs())
    assert area.to_value(u.deg ** 2) == pytest.approx(((0.030 * u.arcsec).to(u.deg) ** 2).value)
    assert 'PIXAREF' in capsys.readouterr().out
    area, side = MC._conversion_pixel_area(meta, None)
    assert side.to_value(u.arcsec) == pytest.approx(0.0312)


# --- satstar ------------------------------------------------------------

def _satstar(flux_fit):
    t = Table({'flux_fit': np.array([flux_fit], dtype=float),
               'flux_err': np.array([0.01 * flux_fit], dtype=float)})
    return t


def test_satstar_rows_from_two_detectors_reduce_to_the_same_jy(tmp_path):
    frames = [_crf(tmp_path, 'sa', AREAS_SR[0]), _crf(tmp_path, 'sb', AREAS_SR[2])]
    tables = [_satstar(TRUE_JY / (a * 1e6)) for a in (AREAS_SR[0], AREAS_SR[2])]
    area_ref = FPA.apply_satstar_frame_pixel_area(tables, frames, 'f212n')
    for t in tables:
        np.testing.assert_allclose(_jy(t, area_ref), TRUE_JY, rtol=1e-6)
        np.testing.assert_allclose(_jy(t, area_ref, 'flux_err'), 0.01 * TRUE_JY, rtol=1e-6)


def test_satstar_scaling_is_all_or_nothing(tmp_path, capsys):
    frames = [_crf(tmp_path, 'sa', AREAS_SR[0]), None]
    tables = [_satstar(1.0), _satstar(1.0)]
    assert FPA.apply_satstar_frame_pixel_area(tables, frames, 'f212n') is None
    assert all(float(t['flux_fit'][0]) == 1.0 for t in tables)
    assert 'no frame on disk' in capsys.readouterr().out


def test_satstar_cache_of_the_previous_algorithm_is_rejected():
    assert MC._SATSTAR_DEDUP_ALG != 'fp6'
    assert MC._satstar_dedup_alg_tag().startswith(MC._SATSTAR_DEDUP_ALG)
    assert MC._satstar_dedup_alg_tag() != 'fp6'


# --- m8 forced fill -----------------------------------------------------

def test_forced_fill_frames_of_two_areas_calibrate_to_equal_flux(monkeypatch):
    from jwst_gc_pipeline.photometry import forced_fill as ff

    conv_jy = 3.0e-8                   # the m7 conversion, whatever area it used
    areas = {'a': AREAS_SR[0], 'b': AREAS_SR[2]}
    ra = np.array([RA0])
    dec = np.array([DEC0])
    ny = nx = 20

    class _WW:
        def all_world2pix(self, r, d, origin, quiet=True):
            return np.full(len(r), 10.0), np.full(len(r), 10.0)

    def prepare_frame(name):
        # the flux a frame measures is set by its own detector area
        return SimpleNamespace(
            nan_replaced_data=np.zeros((ny, nx), dtype='float32'),
            err=np.ones((ny, nx), dtype='float32'),
            mask=np.zeros((ny, nx), dtype=bool), ww=_WW(),
            dao_psf_model=None, flux=TRUE_JY / (areas[name] * 1e6))

    def fake_forced_psf_photometry(data, psf, seed, **kw):
        flux = current['ctx'].flux
        return Table({'flux_fit': np.full(len(seed), flux),
                      'flux_err': np.full(len(seed), 0.1 * flux)})

    current = {}

    def prep(name):
        current['ctx'] = prepare_frame(name)
        return current['ctx']

    monkeypatch.setattr(ff, 'forced_psf_photometry', fake_forced_psf_photometry)
    got = {}
    for name in ('a', 'b'):
        _loc, f, var = ff._forced_fit_frame(
            name, 'f212n', ra, dec, prepare_frame=prep,
            frame_arg_builder=lambda fn: {'name': fn}, fit_shape=(5, 5),
            conv_jy=conv_jy, frame_area_sr=lambda fn: areas[fn])
        got[name] = (f[0] * conv_jy, np.sqrt(var[0]) * conv_jy)
    np.testing.assert_allclose(got['a'][0], got['b'][0], rtol=1e-6)
    np.testing.assert_allclose(got['a'][0], TRUE_JY, rtol=1e-6)
    np.testing.assert_allclose(got['a'][1], 0.1 * TRUE_JY, rtol=1e-6)

    # without frame_area_sr the frames keep image units and disagree by the areas
    raw = []
    for name in ('a', 'b'):
        _loc, f, _var = ff._forced_fit_frame(
            name, 'f212n', ra, dec, prepare_frame=prep,
            frame_arg_builder=lambda fn: {'name': fn}, fit_shape=(5, 5))
        raw.append(f[0] * conv_jy)
    assert raw[0] / raw[1] - 1 > 0.03


def test_merge_individual_frames_rescales_the_frames_before_the_average(tmp_path):
    """The wiring: the per-filter merge reads the frames from disk, so its product
    must carry PIXAREF and fluxes whose Jy agree across the detectors."""
    d = tmp_path / 'F212N'
    d.mkdir()
    (tmp_path / 'catalogs').mkdir()
    for k, area in enumerate(AREAS_SR, start=1):
        t = _frame(tmp_path, k, area)
        t['x_fit'] = np.zeros(2)
        t['y_fit'] = np.zeros(2)
        t['x_err'] = np.zeros(2)
        t['y_err'] = np.zeros(2)
        t['iter_detected'] = np.ones(2, dtype='i8')
        t['group_size'] = np.ones(2, dtype='i8')
        t['dra'] = np.zeros(2, dtype='f4')
        t['ddec'] = np.zeros(2, dtype='f4')
        t.meta['pixscale'] = 0.031
        t.write(d / f'f212n_nrca1_o003_visit001_vgroup02101_exp{k:05d}'
                    f'_m2_daophot_basic.fits')
    out = MC.merge_individual_frames(
        module='nrca1', filtername='f212n', progid='3571', field='003',
        method='dao', suffix='_basic', target='gc3571', basepath=str(tmp_path),
        iteration_label='m2', do_replace_saturated=False)
    assert 'PIXAREF' in out.meta
    area_sr = out.meta['PIXAREF'] / (180 / np.pi) ** 2
    np.testing.assert_allclose(_jy(out, area_sr, 'flux_fit'), TRUE_JY, rtol=1e-5)
