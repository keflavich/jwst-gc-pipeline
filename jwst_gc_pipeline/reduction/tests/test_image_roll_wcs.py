"""Tests for the post-alignment image WCS roll correction (image_roll_wcs.py)."""
import os

import numpy as np
import pytest
from astropy.io import fits

from jwst_gc_pipeline.reduction import image_roll_wcs as irw
from jwst_gc_pipeline.reduction import roll_correction as rc

pytest.importorskip('jwst')

from jwst_gc_pipeline.reduction.tests.test_roll_correction import _synthetic_frame  # noqa: E402


def _grid_sky(fn, n=64, step=8):
    from jwst.datamodels import ImageModel
    yy, xx = np.mgrid[0:n:step, 0:n:step]
    with ImageModel(fn) as m:
        return m.meta.wcs(xx.ravel().astype(float), yy.ravel().astype(float))


def _dev_mas(ra1, dec1, ra2, dec2):
    return np.hypot(((ra1 - ra2 + 180) % 360 - 180) * np.cos(np.radians(dec2)),
                    dec1 - dec2) * 3.6e6


def _shift_frame(src, dst, raoff_arcsec, deoff_arcsec):
    """dst = src shifted by the coordinate offset, as fix_alignment does it."""
    from astropy import units as u
    from jwst.datamodels import ImageModel
    from jwst.tweakreg.utils import adjust_wcs
    with ImageModel(src) as m:
        m.meta.wcs = adjust_wcs(m.meta.wcs, delta_ra=raoff_arcsec * u.arcsec,
                                delta_dec=deoff_arcsec * u.arcsec)
        m.save(dst)
    with fits.open(dst, mode='update') as h:
        h['SCI'].header['RAOFFSET'] = raoff_arcsec
        h['SCI'].header['DEOFFSET'] = deoff_arcsec
    return dst


PIVOT = (266.85 + 30 / 3600., -28.34 - 20 / 3600.)


def test_classify_image():
    c = irw.classify_image
    assert c('jw01182004001_02101_00001_nrca1_cal.fits') == 'cal'
    assert c('jw01182004001_02101_00001_nrca1_destreak.fits') == 'frame'
    assert c('jw01182004001_02101_00001_nrca1_destreak_o004_crf.fits') == 'frame'
    assert c('jw01182004001_02101_00001_nrca1_i2d.fits') == 'skip'
    assert c('jw01182004001_02101_00001_nrca1_destreak_satstar_catalog.fits') == 'skip'
    # stale stemless NIRCam crf (brick 2221 / cloudc, pre-destreak lineage)
    assert c('jw02221001001_07101_00001_nrca1_o001_crf.fits') == 'skip'
    assert c('jw10678-o040_t001_nircam_clear-f212n-merged_i2d.fits') == 'i2d-primary'
    assert c('jw10678-o040_t001_nircam_clear-f212n-nrca_data_i2d.fits') == 'i2d-primary'
    assert c('jw10678-o040_t001_nircam_clear-f212n-merged_m7_daophot_basic_mergedcat_'
             'residual_i2d.fits') == 'i2d-derived'
    assert c('jw10678-o040_t001_nircam_clear-f212n-merged_im0_badastrom_i2d.fits') == 'skip'


def test_classify_image_miri():
    c = irw.classify_image
    assert c('jw10678113001_02201_00001_mirimage_cal.fits') == 'cal'
    assert c('jw10678113001_02201_00001_mirimage_align.fits') == 'frame'
    assert c('jw10678113001_02201_00001_mirimage_o113_crf.fits') == 'frame'
    assert c('jw02221002001_03201_00001_mirimage_align_o002_crf.fits') == 'frame'
    assert c('jw10678113001_02201_00001_mirimage_i2d.fits') == 'skip'
    assert c('jw10678113001_02201_00001_mirimage_ramp.fits') == 'skip'
    assert c('jw10678-o113_t001_miri_f770w_i2d.fits') == 'i2d-primary'
    # image3 per-member intermediates are not products
    assert c('jw10678-o113_t001_miri_f770w_0_o113_crf.fits') == 'skip'
    assert c('jw10678-o113_t001_miri_f770w_segm.fits') == 'skip'
    # same (program, observation, visit) as the NIRCam frames of the visit
    assert irw._prog_obs_from_name('jw10678113001_02201_00001_mirimage_o113_crf.fits') == \
        ('10678', '113', '001')


def test_rotate_frame_matches_rotation_and_is_idempotent(tmp_path):
    fn = _synthetic_frame(tmp_path / 'x_destreak.fits', raoffset=1.5)
    ra0, dec0 = _grid_sky(fn)
    r = irw.rotate_frame(fn, 20.0, PIVOT, visit_key='10678-135-001', table_sha='abc')
    assert r['status'] == 'rotated' and r['verify_mas'] < 0.05
    ra1, dec1 = _grid_sky(fn)
    rp, dp = rc.rotation_about_pivot(ra0, dec0, *PIVOT, 20.0)
    assert _dev_mas(ra1, dec1, rp, dp).max() < 0.05
    assert _dev_mas(ra1, dec1, ra0, dec0).min() > 1.0
    h = fits.getheader(fn, ('SCI', 1))
    assert h['ROLLCORR'] is True and h['ROLLMODE'] == irw.MODE
    assert h['RAOFFSET'] == 1.5                  # alignment keywords untouched
    assert h['SIPGWMAX'] < 0.5
    # SIP header follows the rotated GWCS
    from astropy.wcs import WCS
    s = WCS(h, relax=True).pixel_to_world_values(np.array([5.0, 50.0]), np.array([7.0, 40.0]))
    from jwst.datamodels import ImageModel
    with ImageModel(fn) as m:
        g = m.meta.wcs(np.array([5.0, 50.0]), np.array([7.0, 40.0]))
    assert _dev_mas(s[0], s[1], g[0], g[1]).max() < 0.5
    # idempotent: second call is a no-op
    assert irw.rotate_frame(fn, 20.0, PIVOT)['status'] == 'already'
    ra2, dec2 = _grid_sky(fn)
    assert np.array_equal(ra2, ra1)
    # PR #1004's in-pipeline hook also honours the marker
    assert rc.apply_roll_correction(fn, 20.0, verbose=False) is None


def test_restore_roundtrip(tmp_path):
    fn = _synthetic_frame(tmp_path / 'x_destreak.fits', raoffset=1.5)
    h0 = fits.getheader(fn, ('SCI', 1))
    ra0, dec0 = _grid_sky(fn)
    irw.rotate_frame(fn, -12.0, PIVOT)
    assert irw.restore_product(fn)['status'] == 'restored'
    ra1, dec1 = _grid_sky(fn)
    assert _dev_mas(ra1, dec1, ra0, dec0).max() < 1e-6
    h1 = fits.getheader(fn, ('SCI', 1))
    assert 'ROLLCORR' not in h1 and 'ROLLMODE' not in h1
    for k in ('CRVAL1', 'CRVAL2', 'CRPIX1', 'CTYPE1', 'A_ORDER', 'SIPGWMAX'):
        assert h1.get(k) == h0.get(k)
    with fits.open(fn) as hl:
        assert irw.BACKUP_EXT not in [x.name for x in hl]
    # rotating again after a restore works (clean state)
    assert irw.rotate_frame(fn, -12.0, PIVOT)['status'] == 'rotated'


def test_dry_run_writes_nothing(tmp_path):
    fn = _synthetic_frame(tmp_path / 'x_destreak.fits', raoffset=0.0)
    before = open(fn, 'rb').read()
    r = irw.rotate_frame(fn, 20.0, PIVOT, dry_run=True)
    assert r['status'] == 'dry-run'
    assert open(fn, 'rb').read() == before
    assert not any(p.name.endswith('rolltmp.fits') for p in tmp_path.iterdir())


def test_refuses_symlink_and_pending(tmp_path):
    fn = _synthetic_frame(tmp_path / 'x_destreak.fits', raoffset=0.0)
    ln = tmp_path / 'y_destreak.fits'
    os.symlink(fn, ln)
    with pytest.raises(irw.ImageRollError, match='symlink'):
        irw.rotate_frame(str(ln), 20.0, PIVOT)
    with fits.open(fn, mode='update') as h:
        h['SCI'].header[rc.PENDING] = True
    with pytest.raises(irw.ImageRollError, match=rc.PENDING):
        irw.rotate_frame(fn, 20.0, PIVOT)


def test_cal_rotation_commutes_with_regeneration_shift(tmp_path):
    """Rotating _cal about (P - t) then shifting by t (a regeneration) must
    reproduce the aligned frame rotated about P."""
    t_ra, t_de = -4.4845, -19.914      # arcsec, coordinate (o040 F212N lineage)
    cal = _synthetic_frame(tmp_path / 'x_cal.fits')
    ds = _shift_frame(cal, str(tmp_path / 'x_destreak.fits'), t_ra, t_de)
    roll = 22.0
    r_ds = irw.rotate_frame(ds, roll, PIVOT)
    r_cal = irw.rotate_frame(cal, roll, PIVOT)
    assert r_cal['lineage_offset'] == (t_ra, t_de)
    assert r_cal['pivot'][1] == pytest.approx(PIVOT[1] - t_de / 3600.)
    assert r_ds['pivot'] == pytest.approx(PIVOT)
    regen = _shift_frame(cal, str(tmp_path / 'x_regen.fits'), t_ra, t_de)
    a = _grid_sky(ds)
    b = _grid_sky(regen)
    assert _dev_mas(*a, *b).max() < 0.1
    # rotating the cal about P itself would leave |t| * theta
    cal2 = _synthetic_frame(tmp_path / 'z_cal.fits')
    irw.rotate_frame(cal2, roll, PIVOT)   # no lineage -> pivot P
    wrong = _shift_frame(cal2, str(tmp_path / 'z_regen.fits'), t_ra, t_de)
    err = _dev_mas(*_grid_sky(wrong), *a).max()
    expect = np.hypot(t_ra * np.cos(np.radians(PIVOT[1])), t_de) * 1e3 * np.radians(roll / 3600.)
    assert err == pytest.approx(expect, rel=0.1)


def test_regenerated_destreak_inherits_marker_but_not_restorable(tmp_path):
    """destreak copies every HDU of the cal; the inherited backup must refuse."""
    import shutil
    cal = _synthetic_frame(tmp_path / 'x_cal.fits')
    irw.rotate_frame(cal, 10.0, PIVOT)
    ds = tmp_path / 'x_destreak.fits'
    shutil.copy(cal, ds)                  # what destreak.py's hdu.writeto does
    assert irw.rotate_frame(str(ds), 10.0, PIVOT)['status'] == 'already'
    with pytest.raises(irw.ImageRollError, match='inherited'):
        irw.restore_product(str(ds))


# ----------------------------------------------------------------------------
# i2d
# ----------------------------------------------------------------------------

def _synthetic_i2d(path, shape=(300, 500), crval=(266.39150399, -29.15138395),
                   pa_deg=89.68):
    from astropy import coordinates as coord
    from astropy import units as u
    from astropy.modeling.models import Pix2Sky_TAN
    from gwcs import coordinate_frames as cf
    from gwcs import wcs as gw
    from gwcs.fitswcs import FITSImagingWCSTransform
    from jwst.datamodels import ImageModel
    th = np.radians(pa_deg)
    pc = np.array([[-np.cos(th), np.sin(th)], [np.sin(th), np.cos(th)]])
    cdelt = 8.67e-6 * 3.6       # coarser pixel so a small array spans ~1'
    crpix = (shape[1] / 2 - 0.5, shape[0] / 2 - 0.5)
    tr = FITSImagingWCSTransform(Pix2Sky_TAN(), crpix=list(crpix), crval=list(crval),
                                 cdelt=[cdelt, cdelt], pc=pc)
    det = cf.Frame2D(name='detector', axes_order=(0, 1), unit=(u.pix, u.pix))
    world = cf.CelestialFrame(reference_frame=coord.ICRS(), name='world')
    m = ImageModel(shape)
    m.meta.wcs = gw.WCS([(det, tr), (world, None)])
    wi = m.meta.wcsinfo
    wi.ctype1, wi.ctype2 = 'RA---TAN', 'DEC--TAN'
    wi.cunit1 = wi.cunit2 = 'deg'
    wi.crpix1, wi.crpix2 = crpix[0] + 1, crpix[1] + 1
    wi.crval1, wi.crval2 = crval
    wi.cdelt1 = wi.cdelt2 = cdelt
    wi.pc1_1, wi.pc1_2, wi.pc2_1, wi.pc2_2 = pc.ravel()
    m.save(str(path))
    return str(path)


def _i2d_sky(fn, header=False):
    from jwst.datamodels import ImageModel
    with ImageModel(fn) as m:
        ny, nx = m.data.shape
        yy, xx = np.mgrid[0:ny:20, 0:nx:20]
        xx, yy = xx.ravel().astype(float), yy.ravel().astype(float)
        if header:
            from astropy.wcs import WCS
            return WCS(fits.getheader(fn, ('SCI', 1)), relax=True).pixel_to_world_values(xx, yy)
        return m.meta.wcs(xx, yy)


def _model(visits):
    from jwst_gc_pipeline.astrometry.catalog_roll_correction import PointingModel, VisitRoll
    return PointingModel([VisitRoll('10678', '040', v, r, pra, pde, 'test', fp)
                          for v, r, pra, pde, fp in visits])


def test_i2d_single_roll_exact_header_and_gwcs(tmp_path):
    fn = _synthetic_i2d(tmp_path / 'jw10678-o040_t001_nircam_clear-f212n-merged_i2d.fits')
    h0 = fits.getheader(fn, ('SCI', 1))
    assert h0['CRVAL1'] == pytest.approx(266.39150399)
    piv = (266.39150399 + 0.004, -29.15138395 - 0.003)
    model = _model([('001', 21.97, piv[0], piv[1], [])])
    ra0, dec0 = _i2d_sky(fn)
    r = irw.rotate_i2d(fn, model, visit_keys=['10678-040-001'])
    assert r['status'] == 'rotated' and r['max_mas'] < 0.01
    rt, dt = model.apply(ra0, dec0)
    assert _dev_mas(*_i2d_sky(fn), rt, dt).max() < 0.01              # GWCS
    assert _dev_mas(*_i2d_sky(fn, header=True), rt, dt).max() < 0.01  # FITS header
    h1 = fits.getheader(fn, ('SCI', 1))
    assert h1['ROLLCORR'] is True and h1['ROLLARC'] == 21.97
    assert irw.rotate_i2d(fn, model)['status'] == 'already'
    assert irw.restore_product(fn)['status'] == 'restored'
    assert _dev_mas(*_i2d_sky(fn), ra0, dec0).max() < 1e-6
    assert _dev_mas(*_i2d_sky(fn, header=True), ra0, dec0).max() < 1e-6


def test_i2d_multi_visit_reports_residual_and_needs_opt_in(tmp_path):
    fn = _synthetic_i2d(tmp_path / 'jw01182-o004_t001_nircam_clear-f200w-merged_i2d.fits')
    c = (266.39150399, -29.15138395)
    d = 0.012
    left = np.array([[c[0] + 2 * d, c[1] - d], [c[0], c[1] - d], [c[0], c[1] + d],
                     [c[0] + 2 * d, c[1] + d]])
    right = left.copy()
    right[:, 0] -= 2 * d
    model = _model([('001', 7.24, c[0] + d, c[1], [left]),
                    ('002', 10.73, c[0] - d, c[1], [right])])
    r = irw.rotate_i2d(fn, model, dry_run=True)
    assert r['status'] == 'dry-run' and not r['single']
    assert 0.01 < r['max_mas'] < 3.0
    with pytest.raises(irw.ImageRollError, match='allow_approx'):
        irw.rotate_i2d(fn, model)
    r2 = irw.rotate_i2d(fn, model, allow_approx=True)
    assert r2['status'] == 'rotated'
    assert fits.getheader(fn, ('SCI', 1))['ROLLIRES'] == pytest.approx(r['max_mas'], rel=1e-6)


def test_fit_tan_similarity_recovers_pure_rotation():
    h = fits.Header()
    h['CTYPE1'], h['CTYPE2'] = 'RA---TAN', 'DEC--TAN'
    h['CRVAL1'], h['CRVAL2'] = 266.4, -29.1
    h['CRPIX1'], h['CRPIX2'] = 250.5, 150.5
    h['CDELT1'] = h['CDELT2'] = 3e-5
    h['PC1_1'], h['PC1_2'], h['PC2_1'], h['PC2_2'] = -0.0056, 0.99998, 0.99998, 0.0056
    h['NAXIS1'], h['NAXIS2'] = 500, 300
    piv = (266.41, -29.09)
    fit = irw.fit_tan_similarity(h, lambda r, d: rc.rotation_about_pivot(r, d, *piv, 15.0),
                                 (300, 500))
    assert fit['max_mas'] < 0.01
    assert abs(fit['roll_arcsec']) == pytest.approx(15.0, rel=1e-3)


def test_in_flight_guard_parses_job_names(monkeypatch):
    import subprocess

    class R:
        stdout = ('101 gc-treasury10678-o063-m7\n102_3 brick2221-o001-reduce-F182M\n'
                  '103 1pass_extract_gc-treasury_F212N_o066\n104 gc-monitor\n')
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: R())
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.delenv('SLURM_ARRAY_JOB_ID', raising=False)
    busy, names = irw.in_flight_observations(user='x')
    assert ('10678', '063') in busy and ('2221', '001') in busy and (None, '066') in busy
    assert irw.is_busy('10678', '063', 'gc-treasury', busy, names)
    assert irw.is_busy('10678', '066', 'gc-treasury', busy, names)
    assert not irw.is_busy('10678', '040', 'gc-treasury', busy, names)
    # a field-level chain blocks every observation of that (non-treasury) field
    assert irw.is_busy('1182', '004', 'brick', busy, names)
    assert not irw.is_busy('1939', '001', 'sgra', busy, names)
    # a tokenless 10678 pipeline-stage job blocks every 10678 observation;
    # the hourly HiPS/cron jobs carry no stage token and block nothing
    assert not irw.is_busy('10678', '040', 'gc-treasury', busy,
                           names + ['treasuryhips10678-cron-build'])
    assert irw.is_busy('10678', '040', 'gc-treasury', busy,
                       names + ['gc-treasury10678-regen-F480M'])
    assert irw.is_busy('10678', '040', 'gc-treasury', busy, names + ['gctreasury10678-m7'])


def test_in_flight_guard_skips_its_own_job(monkeypatch):
    """The apply job's own name carries the field; it must not block itself."""
    import subprocess

    class R:
        stdout = '44440289 quintuplet-rollwcs-apply-pilot\n'
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: R())
    monkeypatch.setenv('SLURM_JOB_ID', '44440289')
    monkeypatch.delenv('SLURM_ARRAY_JOB_ID', raising=False)
    busy, names = irw.in_flight_observations(user='x')
    assert names == []
    assert not irw.is_busy('2045', '003', 'quintuplet', busy, names)
    # an array task of this job is also this job
    R.stdout = '500_2 brick-rollwcs-apply\n'
    monkeypatch.setenv('SLURM_JOB_ID', '502')
    monkeypatch.setenv('SLURM_ARRAY_JOB_ID', '500')
    assert irw.in_flight_observations(user='x')[1] == []
    # a different job with the field name still blocks
    monkeypatch.setenv('SLURM_JOB_ID', '1')
    monkeypatch.delenv('SLURM_ARRAY_JOB_ID')
    busy, names = irw.in_flight_observations(user='x')
    assert irw.is_busy('1182', '004', 'brick', busy, names)


def test_refuses_release_symlink_target(tmp_path):
    live = tmp_path / 'live'
    live.mkdir()
    fn = _synthetic_frame(live / 'x_destreak.fits', raoffset=0.0)
    other = _synthetic_frame(live / 'z_destreak.fits', raoffset=0.0)
    rel = tmp_path / 'releases' / 'v1' / 'frames'
    rel.mkdir(parents=True)
    os.symlink(fn, rel / 'x_destreak.fits')
    roots = (str(tmp_path / 'releases'),)
    with pytest.raises(irw.ImageRollError, match='release symlink'):
        irw._check_writable(str(fn), release_roots=roots)
    irw._check_writable(str(other), release_roots=roots)


def test_refuses_blue_field_path(tmp_path, monkeypatch):
    from jwst_gc_pipeline.astrometry import catalog_roll_correction as crc
    fn = _synthetic_frame(tmp_path / 'x_destreak.fits', raoffset=0.0)
    monkeypatch.setattr(crc, 'BLUE_JWST', str(tmp_path) + '/')
    with pytest.raises(irw.ImageRollError, match='#937'):
        irw._check_writable(str(fn), release_roots=())


def test_no_queue_check_refused_under_field_roots(tmp_path):
    from jwst_gc_pipeline.astrometry import catalog_roll_correction as crc
    for p in (crc.ORANGE_JWST + 'brick/F200W/pipeline/x_i2d.fits',
              crc.BLUE_JWST + 'brick/F200W/pipeline/x_i2d.fits'):
        with pytest.raises(SystemExit):
            irw.main(['--field', 'brick', '--apply', '--no-queue-check', '--file', p])
