"""SATSTAR_ZF_FLAT: the ZEROFRAME rim rewrite carries the pixel flat.

The crf is rate x PHOTMJSR / flat, while group 0 is raw DN, so cal / group 0
carries each pixel's 1 / flat and R x group 0 does not.  On wd2 the rim
pixels' flat spans 0.97-1.02 (16-84%).  Pinned here:

1. with a flat, a header-rate rim is R_header x group 0 / flat;
2. with a flat, the measured R(g0) curve is calibrated on cal x flat /
   group 0, so calibration pixels and rim pixels at different flat values
   agree;
3. a flat of the wrong shape, or none, leaves the earlier behaviour;
4. ``zeroframe_pixel_flat`` reads R_FLAT from the CRDS cache, cuts a
   subarray, and sets bad flat values to 1 as the jwst flat_field step does;
5. the switch is off by default, and on it adds 'p' to the cache key.
"""
import numpy as np
import pytest
from astropy.io import fits
from jwst.datamodels import dqflags

from jwst_gc_pipeline.reduction.saturated_star_finding import (
    satstar_fit_switches, zeroframe_fit_anchor, zeroframe_pixel_flat,
    zeroframe_recover_saturated)

SATBIT = 2
R_TRUE = 0.08
G0_RIM = 25000.0
_ENV = ('SATSTAR_ZF_FLAT', 'SATSTAR_ZF_R_HEADER',
        'NIRCAM_SATSTAR_RECOVERED_CAP', 'SATSTAR_ZF_KEEP_FINITE',
        'SATSTAR_ZF_G0_GROUPDQ', 'SATSTAR_ZF_FIRST_FRAME',
        'SATSTAR_ZF_RCURVE_GUARD', 'SATSTAR_ZF_RCURVE_MAXSTEP',
        'SATSTAR_ZF_RCURVE_SATCHECK', 'CRDS_PATH')


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)


def _scene(flat_cal=0.9, flat_rim=1.1):
    """Dark frame with four saturated stars (core at the group-0 rail, SAT
    rim at G0_RIM) and 180 calibration pixels at group 0 = 2100-5000 DN.
    Every unsaturated pixel reads R_TRUE x group 0 / flat, with the flat
    ``flat_cal`` at the calibration pixels and ``flat_rim`` at the stars."""
    ny, nx = 200, 200
    rng = np.random.default_rng(11)
    g0 = rng.normal(10, 3, (ny, nx))
    flat = np.ones((ny, nx))
    dq = np.zeros((ny, nx), dtype=np.uint32)
    yy, xx = np.mgrid[0:ny, 0:nx]
    core = np.zeros((ny, nx), dtype=bool)
    rim = np.zeros((ny, nx), dtype=bool)
    for y0, x0 in ((60, 60), (60, 140), (140, 60), (140, 140)):
        rr = np.hypot(xx - x0, yy - y0)
        core |= rr < 1.5
        rim |= (rr >= 1.5) & (rr < 4)
        flat[rr < 8] = flat_rim
    cols = np.arange(180) % 180 + 10
    rows = np.where(np.arange(180) < 90, 100, 102)
    g0[rows, cols] = rng.uniform(2100, 5000, 180)
    flat[rows, cols] = flat_cal
    data = R_TRUE * g0 / flat
    g0[core] = 48000.0
    g0[rim] = G0_RIM
    dq[core | rim] = SATBIT
    data[core] = 0.0
    data[rim] = G0_RIM * R_TRUE * 1.15
    return data, dq, g0, flat, core, rim


def test_header_rim_is_divided_by_the_flat(monkeypatch, capsys):
    monkeypatch.setenv('SATSTAR_ZF_R_HEADER', '1')
    data, dq, g0, flat, core, rim = _scene()
    rec, rim_mask, deep, R = zeroframe_recover_saturated(
        data, dq, g0, R_header=R_TRUE, flat=flat)
    out = capsys.readouterr().out
    assert R == R_TRUE
    assert rim_mask[rim].all()
    assert np.allclose(rec[rim], R_TRUE * G0_RIM / 1.1)
    assert deep[core].all()
    assert '[zeroframe flat] rim divided by the pixel flat, median 1.1' in out
    rec0, _, _, _ = zeroframe_recover_saturated(data, dq, g0,
                                                R_header=R_TRUE)
    assert np.allclose(rec0[rim], R_TRUE * G0_RIM)


def test_cal_offset_is_subtracted_after_the_flat(monkeypatch):
    monkeypatch.setenv('SATSTAR_ZF_R_HEADER', '1')
    data, dq, g0, flat, core, rim = _scene()
    off = np.full(data.shape, 3.0)
    rec, _, _, _ = zeroframe_recover_saturated(
        data, dq, g0, R_header=R_TRUE, flat=flat, cal_offset=off)
    assert np.allclose(rec[rim], R_TRUE * G0_RIM / 1.1 - 3.0)


def test_measured_curve_is_flat_free():
    """Calibration pixels at flat 0.9, stars at 1.1: without the flat the
    curve reads R_TRUE / 0.9 and the rim lands 22% above R_TRUE x group 0 /
    1.1; with it the curve is R_TRUE and the rim is exact."""
    data, dq, g0, flat, core, rim = _scene()
    rec0, _, _, R0 = zeroframe_recover_saturated(data, dq, g0)
    assert R0 == pytest.approx(R_TRUE / 0.9, rel=1e-6)
    assert np.allclose(rec0[rim], R_TRUE / 0.9 * G0_RIM)
    rec, rim_mask, _, R = zeroframe_recover_saturated(data, dq, g0,
                                                      flat=flat)
    assert R == pytest.approx(R_TRUE, rel=1e-6)
    assert rim_mask[rim].all()
    assert np.allclose(rec[rim], R_TRUE * G0_RIM / 1.1)


def test_fit_anchor_passes_the_flat(monkeypatch):
    monkeypatch.setenv('SATSTAR_ZF_R_HEADER', '1')
    data, dq, g0, flat, core, rim = _scene()
    rec, deep, rim_mask, _ = zeroframe_fit_anchor(data, dq, g0,
                                                  R_header=R_TRUE, flat=flat)
    assert np.allclose(rec[rim], R_TRUE * G0_RIM / 1.1)


def test_flat_of_the_wrong_shape_is_ignored(monkeypatch, capsys):
    monkeypatch.setenv('SATSTAR_ZF_R_HEADER', '1')
    data, dq, g0, flat, core, rim = _scene()
    rec, _, _, _ = zeroframe_recover_saturated(
        data, dq, g0, R_header=R_TRUE, flat=flat[:-1])
    assert np.allclose(rec[rim], R_TRUE * G0_RIM)
    assert 'WARNING [zeroframe flat] pixel flat has shape' in \
        capsys.readouterr().out


def _write_flat(tmp_path, name='jwst_nircam_flat_9999.fits', n=64):
    rng = np.random.default_rng(3)
    sci = rng.uniform(0.95, 1.05, (n, n)).astype(np.float32)
    dqa = np.zeros((n, n), dtype=np.uint32)
    sci[1, 1] = np.nan
    sci[1, 2] = 0.0
    sci[1, 3] = -0.5
    dqa[1, 4] = dqflags.pixel['DO_NOT_USE']
    dqa[1, 5] = dqflags.pixel['NO_FLAT_FIELD']
    d = tmp_path / 'references' / 'jwst' / 'nircam'
    d.mkdir(parents=True)
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(sci, name='SCI'),
                  fits.ImageHDU(dqa, name='DQ')]).writeto(d / name)
    return sci.astype(float)


def _hdr(**kw):
    h = fits.Header()
    h['INSTRUME'] = 'NIRCAM'
    h['R_FLAT'] = 'crds://jwst_nircam_flat_9999.fits'
    for k, v in kw.items():
        h[k] = v
    return h


def test_pixel_flat_reads_the_cache_and_resets_bad_values(tmp_path,
                                                          monkeypatch):
    sci = _write_flat(tmp_path)
    monkeypatch.setenv('CRDS_PATH', str(tmp_path))
    flat = zeroframe_pixel_flat(_hdr(), (64, 64))
    assert flat.shape == (64, 64)
    assert np.all(flat[1, 1:6] == 1.0)
    keep = np.ones((64, 64), dtype=bool)
    keep[1, 1:6] = False
    assert np.allclose(flat[keep], sci[keep])


def test_pixel_flat_cuts_the_subarray(tmp_path):
    sci = _write_flat(tmp_path)
    flat = zeroframe_pixel_flat(_hdr(SUBSTRT1=11, SUBSTRT2=21), (16, 32),
                                crds_dir=str(tmp_path))
    assert np.allclose(flat, sci[20:36, 10:42])
    assert zeroframe_pixel_flat(_hdr(SUBSTRT1=40, SUBSTRT2=1), (16, 32),
                                crds_dir=str(tmp_path)) is None


def test_pixel_flat_missing_pieces_return_none(tmp_path, monkeypatch,
                                               capsys):
    _write_flat(tmp_path)
    assert zeroframe_pixel_flat(_hdr(), (64, 64)) is None   # no CRDS_PATH
    monkeypatch.setenv('CRDS_PATH', str(tmp_path))
    h = _hdr()
    del h['R_FLAT']
    assert zeroframe_pixel_flat(h, (64, 64)) is None
    assert zeroframe_pixel_flat(_hdr(R_FLAT='N/A'), (64, 64)) is None
    assert zeroframe_pixel_flat(
        _hdr(R_FLAT='crds://jwst_nircam_flat_0001.fits'), (64, 64)) is None
    out = capsys.readouterr().out
    assert 'CRDS_PATH is not set' in out
    assert 'no R_FLAT in the header' in out
    assert 'jwst_nircam_flat_0001.fits not in' in out


def test_switch_is_off_by_default(monkeypatch):
    assert satstar_fit_switches()['zf_flat'] is False
    monkeypatch.setenv('NIRCAM_SATSTAR_RECOVERED_CAP', '1')
    assert satstar_fit_switches()['zf_flat'] is False
    monkeypatch.setenv('SATSTAR_ZF_FLAT', 'on')
    assert satstar_fit_switches()['zf_flat'] is True
    monkeypatch.setenv('SATSTAR_ZF_FLAT', 'ture')
    with pytest.raises(ValueError, match='SATSTAR_ZF_FLAT'):
        satstar_fit_switches()
