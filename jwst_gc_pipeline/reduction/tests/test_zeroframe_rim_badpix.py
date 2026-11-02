"""Dead, hot and reference pixels stay out of the ZEROFRAME rim rewrite
(``SATSTAR_ZF_RIM_BADPIX``, default ON).

On wd2 F150W nrcb1 (#1071), clumps flagged DO_NOT_USE|SATURATED|DEAD|HOT
(DQ 1313795) have a group-0 of ~0.001-1.2 DN: positive and below the ceiling,
so "clean".  They were rewritten to R*group0 ~ 1e-4-0.1 MJy/sr with a 5%
error, 400-5e8x the weight of a normal pixel, and four 16-18 mag stars next to
them came out with negative flux.  Their crf SCI and ERR are NaN, so left alone they carry zero
weight.  A rewritten rim pixel's error is also floored at the frame's median
ERR (``zeroframe_rim_error``).
"""
import numpy as np
import pytest
from astropy.io import fits
from jwst.datamodels import dqflags

import jwst_gc_pipeline.reduction.saturated_star_finding as SSF
from jwst_gc_pipeline.reduction.saturated_star_finding import (
    zeroframe_recover_saturated, zeroframe_rim_error)

SAT = dqflags.pixel['SATURATED']
DNU = dqflags.pixel['DO_NOT_USE']
DEAD = dqflags.pixel['DEAD']
HOT = dqflags.pixel['HOT']
REFPIX = dqflags.pixel['REFERENCE_PIXEL']
WD2_BADPIX = 1313795      # DO_NOT_USE|SATURATED|DEAD|HOT|NO_FLAT_FIELD|NO_LIN_CORR


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv('SATSTAR_ZF_RIM_BADPIX', raising=False)


def _scene(dq_dtype=np.uint32):
    """Dark frame, one saturated star (deep core at the rail, recoverable
    rim), a dead/hot clump with a ~0 group-0, and a saturated reference-pixel
    column."""
    ny = nx = 120
    r_true = 0.17
    g0 = np.random.default_rng(7).normal(10, 3, (ny, nx))
    data = g0 * r_true + np.random.default_rng(8).normal(0, 0.05, (ny, nx))
    dq = np.zeros((ny, nx), dtype=np.uint32)
    vals = np.random.default_rng(9).uniform(21000, 40000, (8, 8))
    g0[100:108, 8:16] = vals
    data[100:108, 8:16] = vals * r_true
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - 60, yy - 60)
    core = rr < 1.5
    rim = (rr >= 1.5) & (rr < 4)
    g0[core] = 48000.0
    g0[rim] = 20000.0
    dq[core | rim] = SAT
    data[core] = np.nan
    data[rim] = 20000.0 * r_true * 1.15
    bad = np.zeros((ny, nx), bool)
    bad[30:32, 70:73] = True
    dq[bad] = WD2_BADPIX
    g0[bad] = np.linspace(0.001, 0.06, bad.sum())
    data[bad] = 0.0            # NaN in the crf, zeroed by get_saturated_stars
    ref = np.zeros((ny, nx), bool)
    ref[40:44, 2] = True
    dq[ref] = SAT | REFPIX
    g0[ref] = 50.0
    # a signed DQ holds REFERENCE_PIXEL (bit 31) as a negative number
    return data, dq.astype(dq_dtype), g0, core, rim, bad, ref


@pytest.mark.parametrize('dq_dtype', [np.uint32, np.int32])
def test_bad_pixels_are_not_rewritten(dq_dtype):
    data, dq, g0, core, rim, bad, ref = _scene(dq_dtype)
    rec, rim_mask, deep, R = zeroframe_recover_saturated(data, dq, g0)
    assert np.isfinite(R)
    assert not rim_mask[bad].any() and not rim_mask[ref].any()
    assert np.array_equal(rec[bad], data[bad])
    assert np.array_equal(rec[ref], data[ref])
    # the star is recovered as before
    assert rim_mask[rim].all()
    assert np.allclose(rec[rim], R * 20000.0, rtol=0.05)
    assert deep[core].all()
    # a dead pixel is not deep core either: the recovered-core cap must not
    # read it as the star's core
    assert not deep[bad].any()


def test_switch_off_restores_the_rewrite(monkeypatch):
    data, dq, g0, core, rim, bad, ref = _scene()
    monkeypatch.setenv('SATSTAR_ZF_RIM_BADPIX', '0')
    rec, rim_mask, deep, R = zeroframe_recover_saturated(data, dq, g0)
    assert rim_mask[bad].all() and rim_mask[ref].all()
    assert np.all(rec[bad] < 0.02)              # R * group0 ~ 0.17 * 0.06


def test_only_the_bad_pixels_change():
    data, dq, g0, core, rim, bad, ref = _scene()
    rec1, rim1, deep1, R1 = zeroframe_recover_saturated(data, dq, g0)
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv('SATSTAR_ZF_RIM_BADPIX', '0')
        rec0, rim0, deep0, R0 = zeroframe_recover_saturated(data, dq, g0)
    assert R1 == R0
    keep = ~(bad | ref)
    assert np.array_equal(rim0[keep], rim1[keep])
    assert np.array_equal(deep0, deep1)
    assert np.array_equal(rec0[keep], rec1[keep], equal_nan=True)


# --------------------------------------------------------------------------
# rim error
# --------------------------------------------------------------------------

def _err_scene():
    data = np.array([[1e-4, 2000.0, 5.0, 3.0]])
    err = np.array([[np.nan, np.nan, 0.3, np.nan]])
    rim = np.array([[True, True, True, False]])
    return data, err, rim


def test_rim_error_is_floored_at_the_median_err():
    data, err, rim = _err_scene()
    out = zeroframe_rim_error(data, err, rim)
    assert out[0, 0] == pytest.approx(0.3)       # 5e-6 floored to median ERR
    assert out[0, 1] == pytest.approx(100.0)     # 5% of a real rim value
    assert out[0, 2] == 0.3                      # finite ERR kept
    assert np.isnan(out[0, 3])                   # not rim: left for 1e10
    assert np.isnan(err[0, 0])                   # input untouched


def test_rim_error_without_floor_is_five_percent():
    data, err, rim = _err_scene()
    out = zeroframe_rim_error(data, err, rim, floor=False)
    assert out[0, 0] == pytest.approx(5e-6)
    assert out[0, 1] == pytest.approx(100.0)


def test_rim_error_with_no_valid_err_keeps_five_percent():
    data, err, rim = _err_scene()
    err[0, 2] = np.nan
    out = zeroframe_rim_error(data, err, rim)
    assert out[0, 0] == pytest.approx(5e-6)


# --------------------------------------------------------------------------
# wiring in get_saturated_stars
# --------------------------------------------------------------------------

class _ErrReached(Exception):
    pass


@pytest.mark.parametrize('value, floor', [(None, True), ('1', True),
                                          ('0', False)])
def test_get_saturated_stars_floors_the_rim_error(monkeypatch, value, floor):
    """The anchor's rim reaches ``zeroframe_rim_error`` with the switch as
    ``floor``.  The anchor is stubbed and the error helper stops the fit."""
    if value is not None:
        monkeypatch.setenv('SATSTAR_ZF_RIM_BADPIX', value)
    n = 200
    sci = np.ones((n, n))
    dq = np.zeros((n, n), dtype=np.uint32)
    dq[98:103, 98:103] = SAT
    sci[dq != 0] = np.nan
    wcs_hdr = fits.Header({'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
                           'CRPIX1': 100, 'CRPIX2': 100, 'CRVAL1': 150.0,
                           'CRVAL2': 2.0, 'CDELT1': -1.7e-5, 'CDELT2': 1.7e-5,
                           'BUNIT': 'MJy/sr'})
    fh = fits.HDUList([
        fits.PrimaryHDU(header=fits.Header({
            'INSTRUME': 'NIRCAM', 'FILTER': 'F150W', 'PUPIL': 'CLEAR',
            'DETECTOR': 'NRCB1', 'MODULE': 'B', 'CHANNEL': 'SHORT'})),
        fits.ImageHDU(sci, header=wcs_hdr, name='SCI'),
        fits.ImageHDU(np.where(dq != 0, np.nan, 0.1), name='ERR'),
        fits.ImageHDU(dq, name='DQ'),
        fits.ImageHDU(np.where(dq != 0, np.nan, 0.01), name='VAR_POISSON')])
    rim = dq != 0
    deep = np.zeros((n, n), bool)
    seen = {}

    def _anchor(data, dq, zeroframe, **kw):
        return data, deep, rim, None

    def _err(data, err, rim_arg, *, floor=True):
        seen.update(rim=rim_arg, floor=floor, err=err)
        raise _ErrReached
    monkeypatch.setattr(SSF, 'zeroframe_fit_anchor', _anchor)
    monkeypatch.setattr(SSF, 'zeroframe_rim_error', _err)
    with pytest.raises(_ErrReached):
        SSF.get_saturated_stars(fh, zeroframe=np.ones((n, n)), plot=False)
    assert seen['rim'] is rim
    assert seen['floor'] is floor
    assert np.isnan(seen['err'][100, 100])
