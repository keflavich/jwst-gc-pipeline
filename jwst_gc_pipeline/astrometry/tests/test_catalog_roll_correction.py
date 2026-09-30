"""Tests for the catalog-level roll correction (data-qa#346)."""
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.astrometry import catalog_roll_correction as crc
from jwst_gc_pipeline.reduction.roll_correction import _tangent, _untangent

RA0, DEC0 = 266.60, -28.35
AS = 1 / 3600.0


def _grid(n=21, half_arcsec=150.0, ra0=RA0, dec0=DEC0):
    xs = np.linspace(-half_arcsec, half_arcsec, n) / 206264.806
    xx, yy = np.meshgrid(xs, xs)
    return _untangent(xx.ravel(), yy.ravel(), ra0, dec0)


def _square(ra0, dec0, half_arcsec):
    h = half_arcsec / 206264.806
    x = np.array([-h, h, h, -h])
    y = np.array([-h, -h, h, h])
    ra, dec = _untangent(x, y, ra0, dec0)
    return np.column_stack([ra, dec])


def _visit(roll, ra0=RA0, dec0=DEC0, half=160.0, visit='001', prog='10678', obs='135'):
    return crc.VisitRoll(prog, obs, visit, roll, ra0, dec0, 'test', [_square(ra0, dec0, half)])


def test_single_visit_rotation_sign_and_size():
    model = crc.PointingModel([_visit(20.0)])
    # a star 100" due North of the pivot
    ra, dec = _untangent(np.array([0.0]), np.array([100 / 206264.806]), RA0, DEC0)
    r2, d2 = model.apply(ra, dec)
    x, y = _tangent(r2, d2, RA0, DEC0)
    # +roll rotates North toward East: the star moves East by r*theta
    assert x[0] * 206264.806 * 1e3 == pytest.approx(100 * 20 / 206264.806 * 1e3, rel=1e-6)
    assert y[0] * 206264.806 == pytest.approx(100.0, abs=1e-6)


def test_fit_rotation_recovers_roll_and_zero_bulk():
    ra, dec = _grid()
    model = crc.PointingModel([_visit(18.0)])
    r2, d2 = model.apply(ra, dec)
    fit = crc.fit_rotation(ra, dec, r2, d2, (RA0, DEC0))
    assert fit['roll_arcsec'] == pytest.approx(18.0, abs=1e-3)
    assert abs(fit['tx_mas']) < 1e-3 and abs(fit['ty_mas']) < 1e-3


def test_pivot_offset_adds_expected_rigid_shift():
    """Fitting about another point: the translation at the fit origin is d*theta."""
    ra, dec = _grid()
    pra, pde = _untangent(np.array([30 / 206264.806]), np.array([0.0]), RA0, DEC0)
    model = crc.PointingModel([_visit(20.0, ra0=pra[0], dec0=pde[0])])
    r2, d2 = model.apply(ra, dec)
    fit = crc.fit_rotation(ra, dec, r2, d2, (RA0, DEC0))
    # origin sits 30" West of the pivot (x=-30"): dy = -theta*x = +2.9 mas (North)
    assert fit['roll_arcsec'] == pytest.approx(20.0, abs=1e-3)
    assert fit["ty_mas"] == pytest.approx(30 * 20 / 206264.806 * 1e3, abs=1e-3)


def test_two_visit_overlap_takes_mean_displacement():
    ra_e, _ = _untangent(np.array([60 / 206264.806]), np.array([0.0]), RA0, DEC0)
    ra_w, _ = _untangent(np.array([-60 / 206264.806]), np.array([0.0]), RA0, DEC0)
    v1 = _visit(10.0, ra0=ra_e[0], half=80.0, visit='001')
    v2 = _visit(4.0, ra0=ra_w[0], half=80.0, visit='002')
    model = crc.PointingModel([v1, v2])
    # one star only in v1 (far east), one in the overlap (x=0), 50" north
    x = np.array([120.0, 0.0]) / 206264.806
    y = np.array([50.0, 50.0]) / 206264.806
    ra, dec = _untangent(x, y, RA0, DEC0)
    r2, d2 = model.apply(ra, dec)
    e1 = crc.PointingModel([v1]).apply(ra, dec)
    e2 = crc.PointingModel([v2]).apply(ra, dec)
    assert r2[0] == pytest.approx(e1[0][0], abs=1e-12)
    assert d2[0] == pytest.approx(e1[1][0], abs=1e-12)
    assert r2[1] == pytest.approx(0.5 * (e1[0][1] + e2[0][1]), abs=1e-10)
    assert d2[1] == pytest.approx(0.5 * (e1[1][1] + e2[1][1]), abs=1e-10)


def test_nan_rows_pass_through():
    model = crc.PointingModel([_visit(20.0)])
    r2, d2 = model.apply(np.array([np.nan, RA0]), np.array([np.nan, DEC0]))
    assert np.isnan(r2[0]) and np.isnan(d2[0])
    assert r2[1] == pytest.approx(RA0) and d2[1] == pytest.approx(DEC0)


@pytest.mark.parametrize('name,expect', [
    ('f212n_merged_o135_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits', 'perband'),
    ('f212n_nrca1_o040_indivexp_nrca1_m3_dao_basic.fits', 'perband'),
    ('basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_o135.fits', 'crossband'),
    ('basic_merged_indivexp_photometry_tables_merged_resbgsub_m7_o001_qualcuts_oksep2221.fits',
     'crossband'),
    ('f212n_merged_o135_indivexp_merged_m3_dao_basic_i2dseed.fits', 'excluded'),
    ('f212n_o135_consensus.fits', 'excluded'),
    ('gaia_virac2_refcat_epoch2026.70_o135.fits', 'excluded'),
    ('crossband_seed_manual_o135.fits', 'excluded'),
    ('f212n_o135_consolidated_satstar_catalog.fits', 'excluded'),
    ('f2550w_mirimage_indivexp_merged_m2_dao_basic_vetted_carta.fits', 'derived'),
    ('f212n_merged_indivexp_merged_crowdsource_nsky0.fits', 'legacy'),
    ('basic_merged_photometry_tables_merged_bgsub.ecsv', 'legacy'),
    ('f212n_merged_o135_indivexp_merged_m3_dao_basic.fits.prov.json', 'excluded'),
])
def test_classify(name, expect):
    assert crc.classify_catalog(name) == expect


def test_find_position_pairs_ignores_offsets_and_scatter():
    cols = ['skycoord.ra', 'skycoord.dec', 'dra', 'ddec', 'std_ra', 'std_dec', 'skycoord_ref.ra',
            'skycoord_ref.dec', 'skycoord_f212n.ra', 'skycoord_f212n.dec', 'f480m_ra', 'f480m_dec',
            'ra', 'dec', 'dra_f212n', 'sep_f212n', 'x_fit', 'y_fit']
    pairs = crc.find_position_pairs(cols)
    got = {(a, b, c) for a, b, c in pairs}
    assert ('skycoord.ra', 'skycoord.dec', None) in got
    assert ('skycoord_ref.ra', 'skycoord_ref.dec', 'ref') in got
    assert ('skycoord_f212n.ra', 'skycoord_f212n.dec', 'f212n') in got
    assert ('f480m_ra', 'f480m_dec', 'f480m') in got
    assert ('ra', 'dec', None) in got
    assert len(pairs) == 5


def _write_perband(path, ra, dec, extra_hdr=None):
    t = Table()
    t['skycoord.ra'] = ra
    t['skycoord.ra'].unit = 'deg'
    t['skycoord.dec'] = dec
    t['skycoord.dec'].unit = 'deg'
    t['flux'] = np.linspace(1, 2, len(ra))
    t['x_fit'] = np.arange(len(ra), dtype=float)
    hdu = fits.BinTableHDU(t)
    for k, v in (extra_hdr or {}).items():
        hdu.header[k] = v
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)


@pytest.fixture
def treasury_models():
    return {('10678', '135'): [_visit(20.0)]}


def test_correct_catalog_end_to_end(tmp_path, treasury_models):
    ra, dec = _grid()
    src = tmp_path / 'f212n_merged_o135_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits'
    _write_perband(src, ra, dec)
    before = fits.getdata(src).copy()
    out = tmp_path / 'out' / src.name
    res = crc.correct_catalog(str(src), str(out), 'gc-treasury', treasury_models)
    # input untouched
    assert np.array_equal(fits.getdata(src)['skycoord.ra'], before['skycoord.ra'])
    d = fits.getdata(out)
    h = fits.getheader(out, 1)
    assert h[crc.MARKER] is True
    assert h['ROLV00AS'] == pytest.approx(20.0)
    assert np.array_equal(d['x_fit'], before['x_fit'])        # pixel columns untouched
    fit = crc.fit_rotation(before['skycoord.ra'], before['skycoord.dec'],
                           d['skycoord.ra'], d['skycoord.dec'], (RA0, DEC0))
    assert fit['roll_arcsec'] == pytest.approx(20.0, abs=1e-3)
    # max displacement = r_max * theta at the grid corner
    rmax = 150 * np.sqrt(2)
    assert res['columns'][0]['max_disp_mas'] == pytest.approx(rmax * 20 / 206264.806 * 1e3, rel=1e-4)
    assert res['columns'][0]['roundtrip_mas'] < 0.01


def test_refusals(tmp_path, treasury_models):
    ra, dec = _grid(5)
    src = tmp_path / 'f212n_merged_o135_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits'
    _write_perband(src, ra, dec)
    with pytest.raises(crc.RollCatalogError, match='in place'):
        crc.correct_catalog(str(src), str(src), 'gc-treasury', treasury_models)
    out = tmp_path / 'o' / src.name
    crc.correct_catalog(str(src), str(out), 'gc-treasury', treasury_models)
    with pytest.raises(crc.RollCatalogError, match='output exists'):
        crc.correct_catalog(str(src), str(out), 'gc-treasury', treasury_models)
    # idempotency: the corrected output is refused as an input
    with pytest.raises(crc.RollCatalogError, match='already carries'):
        crc.correct_catalog(str(out), str(tmp_path / 'o2' / src.name), 'gc-treasury',
                            treasury_models)


def test_refuses_catalog_with_own_rollcorr_stamp(tmp_path, treasury_models):
    ra, dec = _grid(5)
    src = tmp_path / 'f212n_merged_o135_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits'
    _write_perband(src, ra, dec, extra_hdr={'ROLLCORR': True})
    with pytest.raises(crc.RollCatalogError, match='image-level'):
        crc.correct_catalog(str(src), str(tmp_path / 'o' / src.name), 'gc-treasury',
                            treasury_models)


@pytest.mark.parametrize('catalog_after_rotation', [True, False])
def test_born_rotated_by_frame_state_and_mtime(tmp_path, treasury_models,
                                               catalog_after_rotation):
    """Frames rotated at the image level: a catalog written AFTER the rotation
    is refused (born rotated); one written BEFORE it is corrected."""
    import copy
    ra, dec = _grid(5)
    src = tmp_path / 'f212n_merged_o135_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits'
    _write_perband(src, ra, dec)
    mt = os.path.getmtime(src)
    models = copy.deepcopy(treasury_models)
    for vrs in models.values():
        for v in vrs:
            v.frame_roll = dict(n_checked=8, n_rolled=8, modes=['posthoc-visit-pivot'],
                                first_epoch=mt - 60 if catalog_after_rotation else mt + 60)
    out = tmp_path / 'o' / src.name
    if catalog_after_rotation:
        with pytest.raises(crc.RollCatalogError, match='written after the frames'):
            crc.correct_catalog(str(src), str(out), 'gc-treasury', models)
    else:
        crc.correct_catalog(str(src), str(out), 'gc-treasury', models)
        assert out.exists()


def test_frame_roll_state_reads_sampled_crf(tmp_path):
    pdir = tmp_path / 'F212N' / 'pipeline'
    pdir.mkdir(parents=True)
    for i in range(3):
        sci = fits.ImageHDU(np.zeros((2, 2)), name='SCI')
        if i < 2:
            sci.header['ROLLCORR'] = True
            sci.header['ROLLMODE'] = 'posthoc-visit-pivot'
            sci.header['ROLLDATE'] = f'2026-10-0{i + 1}T00:00:00Z'
        fits.HDUList([fits.PrimaryHDU(), sci]).writeto(
            pdir / f'jw10678135001_02101_0000{i + 1}_nrca1_destreak_o135_crf.fits')
    st = crc.frame_roll_state(str(tmp_path), '10678', '135')
    assert st['n_checked'] == 3 and st['n_rolled'] == 2
    assert st['first_epoch'] == 1790812800      # 2026-10-01T00:00:00Z
    assert st['modes'] == ['posthoc-visit-pivot']
    assert crc.frame_roll_state(str(tmp_path), '10678', '999')['n_rolled'] == 0


def test_miri_skipped_by_default(tmp_path, treasury_models):
    ra, dec = _grid(5)
    src = tmp_path / 'f770w_mirimage_o135_indivexp_mirimage_m7_dao_basic_vetted.fits'
    _write_perband(src, ra, dec)
    plan = crc.plan_catalog(str(src), 'gc-treasury', treasury_models)
    assert plan['status'] == 'skipped-miri'


def test_crossband_per_row_reference_band(tmp_path):
    """skycoord_ref takes the rotation of the band named per row; with two
    observations of different roll the per-band columns rotate separately and
    sep_<band> is recomputed."""
    ra, dec = _grid(7)
    n = len(ra)
    models = {('1182', '004'): [_visit(10.0, prog='1182', obs='004')],
              ('2221', '001'): [_visit(-5.0, prog='2221', obs='001')]}
    t = Table()
    for c, v in (('skycoord_ref.ra', ra), ('skycoord_ref.dec', dec),
                 ('skycoord_f200w.ra', ra), ('skycoord_f200w.dec', dec),
                 ('skycoord_f212n.ra', ra), ('skycoord_f212n.dec', dec)):
        t[c] = v
        t[c].unit = 'deg'
    t['skycoord_ref_filtername'] = np.where(np.arange(n) % 2 == 0, 'f200w', 'f212n')
    t['sep_f200w'] = np.zeros(n)
    t['sep_f200w'].unit = 'deg'
    src = tmp_path / 'basic_merged_indivexp_photometry_tables_merged_resbgsub_m7.fits'
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(t)]).writeto(src)
    out = tmp_path / 'o' / src.name
    res = crc.correct_catalog(str(src), str(out), 'brick', models)
    d = fits.getdata(out)
    e200 = crc.PointingModel(models[('1182', '004')]).apply(ra, dec)
    e212 = crc.PointingModel(models[('2221', '001')]).apply(ra, dec)
    even = np.arange(n) % 2 == 0
    np.testing.assert_allclose(d['skycoord_ref.ra'][even], e200[0][even], atol=1e-12)
    np.testing.assert_allclose(d['skycoord_ref.ra'][~even], e212[0][~even], atol=1e-12)
    np.testing.assert_allclose(d['skycoord_f212n.dec'], e212[1], atol=1e-12)
    # f200w rows whose ref is f212n now have a non-zero separation
    assert 'sep_f200w' in res['sep_recomputed']
    assert np.all(d['sep_f200w'][even] < 1e-9)
    assert np.nanmax(d['sep_f200w'][~even]) > 0


def test_generic_radec_mixing_observations_needs_rebuild(tmp_path):
    ra, dec = _grid(5)
    models = {('1182', '004'): [_visit(10.0, prog='1182', obs='004')],
              ('2221', '001'): [_visit(-5.0, prog='2221', obs='001')]}
    t = Table({'ra': ra, 'dec': dec, 'f212n_ra': ra, 'f212n_dec': dec,
               'f200w_ra': ra, 'f200w_dec': dec})
    src = tmp_path / 'brick_crossband_m7_merge.fits'
    fits.HDUList([fits.PrimaryHDU(), fits.BinTableHDU(t)]).writeto(src)
    assert crc.plan_catalog(str(src), 'brick', models)['status'] == 'needs-rebuild'


def test_ecsv_roundtrip(tmp_path, treasury_models):
    ra, dec = _grid(5)
    t = Table({'skycoord.ra': ra, 'skycoord.dec': dec, 'flux_f212n': np.ones(len(ra)),
               'flux_f480m': np.ones(len(ra))})
    t['skycoord.ra'].unit = 'deg'
    t['skycoord.dec'].unit = 'deg'
    src = tmp_path / 'basic_merged_indivexp_photometry_tables_merged_resbgsub_m7_o135.ecsv'
    t.write(src, format='ascii.ecsv')
    out = tmp_path / 'o' / src.name
    crc.correct_catalog(str(src), str(out), 'gc-treasury', treasury_models)
    t2 = Table.read(out, format='ascii.ecsv')
    assert t2.meta[crc.MARKER] is True
    fit = crc.fit_rotation(ra, dec, np.asarray(t2['skycoord.ra']), np.asarray(t2['skycoord.dec']),
                           (RA0, DEC0))
    assert fit['roll_arcsec'] == pytest.approx(20.0, abs=1e-3)
    assert crc.plan_catalog(str(out), 'gc-treasury', treasury_models)['status'] == 'refused'


def test_ecsv_skycoord_mixin_roundtrip(tmp_path, treasury_models):
    """Pipeline ECSVs store SkyCoord mixins; they come back as SkyCoord, rotated."""
    from astropy.coordinates import SkyCoord
    ra, dec = _grid(5)
    t = Table({'skycoord_ref': SkyCoord(ra, dec, unit='deg', frame='icrs'),
               'skycoord_ref_filtername': np.full(len(ra), 'f212n'),
               'flux_f212n': np.ones(len(ra))})
    src = tmp_path / 'basic_merged_indivexp_photometry_tables_merged_resbgsub_m7_o135.ecsv'
    t.write(src, format='ascii.ecsv')
    out = tmp_path / 'o' / src.name
    crc.correct_catalog(str(src), str(out), 'gc-treasury', treasury_models)
    t2 = Table.read(out, format='ascii.ecsv')
    assert isinstance(t2['skycoord_ref'], SkyCoord)
    fit = crc.fit_rotation(ra, dec, t2['skycoord_ref'].ra.deg, t2['skycoord_ref'].dec.deg,
                           (RA0, DEC0))
    assert fit['roll_arcsec'] == pytest.approx(20.0, abs=1e-3)


def test_ecsv_verify_before_uses_unrotated_positions(tmp_path, treasury_models, monkeypatch):
    """Native-endian ECSV columns must be copied before the in-place rotation."""
    ra, dec = _grid(5)
    t = Table({'skycoord.ra': ra, 'skycoord.dec': dec, 'flux_f212n': np.ones(len(ra))})
    src = tmp_path / 'basic_merged_indivexp_photometry_tables_merged_resbgsub_m7_o135.ecsv'
    t.write(src, format='ascii.ecsv')
    seen = []
    monkeypatch.setattr(crc, 'reference_tie',
                        lambda r, d, *a, **k: seen.append(np.array(r)) or {'status': 'ok'})
    crc.correct_catalog(str(src), str(tmp_path / 'o' / src.name), 'gc-treasury',
                        treasury_models, verify_reference='dummy.fits')
    np.testing.assert_array_equal(seen[0], ra)
    assert np.max(np.abs(seen[1] - ra)) * 3.6e6 > 5     # moved by mas
