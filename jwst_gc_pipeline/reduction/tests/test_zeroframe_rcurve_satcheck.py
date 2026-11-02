"""R(g0)-curve check against the measured SATURATED pixels
(SATSTAR_ZF_RCURVE_SATCHECK).

wd2 F164N / F187N / F212N (SHALLOW4), 2026-10: on a sparse frame no star pixel
stays unsaturated above the R_g0_min = 2000 DN calibration floor, so every
calibration bin holds hot and offset pixels.  The step guard of #972 sees a
smooth junk curve and keeps it: F187N nrca4 used R = 0.0165 where the star
pixels read cal/group0 = 1.44-1.51 (PHOTMJSR / t(group 0) = 1.46), and the
anchor rewrote every saturated core to ~1% of its rate.  SATURATED pixels with
a finite, positive ramp-fit rate, no DO_NOT_USE and a clean group 0 carry the
star-pixel R; the check rebuilds the curve from them when the two disagree by
more than a factor 2.  Pinned here:

1. all-junk calibration: the default rebuilds the curve from the SATURATED
   pixels and the rim is rewritten at the true rate; SATCHECK=0 keeps the
   collapsed curve;
2. a healthy curve is left exactly as it was (no log line);
3. fewer than _RCURVE_SATCHECK_MIN_PX measured SATURATED pixels: no change;
4. DO_NOT_USE, DEAD/HOT and group-0-saturated SATURATED pixels do not enter
   the check;
5. a precomputed R is not checked;
6. the switch and its cache-signature letter;
7. with fewer than _RCURVE_SATCHECK_MIN_PX measured SATURATED pixels, a
   curve more than a factor 2 from the header PHOTMJSR / t(group 0) is
   replaced by it (w51 F187N nrca1: 13 such pixels, R = 0.0026 against 2.66);
   the header value is read by get_saturated_stars and handed to the anchor.
"""
import numpy as np
import pytest
from astropy.io import fits

import jwst_gc_pipeline.reduction.saturated_star_finding as SSF
from jwst_gc_pipeline.reduction.saturated_star_finding import (
    _RCURVE_SATCHECK_MIN_PX, satstar_fit_switches, zeroframe_header_R,
    zeroframe_recover_saturated)

SATBIT, DNUBIT, HOTBIT, JUMPBIT = 2, 1, 2048, 4
R_TRUE = 1.46
R_JUNK = 0.012


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ('SATSTAR_ZF_RCURVE_SATCHECK', 'SATSTAR_ZF_RCURVE_GUARD',
              'SATSTAR_ZF_RCURVE_MAXSTEP', 'SATSTAR_ZF_KEEP_FINITE',
              'SATSTAR_ZF_RIM_BADPIX'):
        monkeypatch.delenv(k, raising=False)


def _scene(calib='junk', n_sat_meas=120, n_cal=40, sat_meas_dq=0):
    """Dark frame with one saturated star.

    * ``calib='junk'``: the unsaturated calibration pixels above 2000 DN are
      hot/offset pixels (cal/group0 = R_JUNK, DQ 0, as on wd2 F187N where the
      crf DQ of those pixels is clean); ``'real'``: star pixels at R_TRUE.
    * ``n_sat_meas`` SATURATED pixels with a ramp-fit rate (cal = R_TRUE x g0,
      g0 log-uniform in 2500-30000 DN, DQ = SATURATED | ``sat_meas_dq``).
    * a deep core at the group-0 rail (cal zeroed, as get_saturated_stars
      does for NaN-VAR_POISSON pixels).
    """
    ny, nx = 200, 200
    rng = np.random.default_rng(5)
    g0 = rng.normal(20, 4, (ny, nx))
    data = g0 * R_TRUE
    dq = np.zeros((ny, nx), dtype=np.uint32)
    vals = np.geomspace(2100, 30000, 4 * n_cal)
    rows, cols = 150 + np.arange(len(vals)) // 40, 10 + np.arange(len(vals)) % 40
    g0[rows, cols] = vals
    data[rows, cols] = vals * (R_JUNK if calib == 'junk' else R_TRUE)
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - 70, yy - 70)
    core = rr < 2.5
    g0[core] = 50000.0
    data[core] = 0.0
    dq[core] = SATBIT
    # measured SATURATED pixels: a block next to the star
    k = np.arange(n_sat_meas)
    my, mx = 90 + k // 30, 40 + k % 30
    gs = np.exp(rng.uniform(np.log(2500), np.log(30000), n_sat_meas))
    g0[my, mx] = gs
    data[my, mx] = gs * R_TRUE
    dq[my, mx] = SATBIT | sat_meas_dq
    meas = np.zeros((ny, nx), dtype=bool)
    meas[my, mx] = True
    return data, dq, g0, core, meas


def test_all_junk_curve_is_rebuilt_from_measured_saturated_pixels(capsys):
    data, dq, g0, core, meas = _scene()
    rec, rim, deep, R = zeroframe_recover_saturated(data, dq, g0)
    out = capsys.readouterr().out
    assert abs(R / R_TRUE - 1) < 0.03
    assert '[zeroframe R-curve check]' in out
    assert 'curve rebuilt' in out
    # the measured SATURATED pixels are rim pixels (group 0 clean): their
    # rewrite is now at the true rate
    assert rim[meas].all()
    assert np.allclose(rec[meas], data[meas], rtol=0.05)


def test_satcheck_off_keeps_the_collapsed_curve(monkeypatch, capsys):
    monkeypatch.setenv('SATSTAR_ZF_RCURVE_SATCHECK', '0')
    data, dq, g0, core, meas = _scene()
    rec, rim, deep, R = zeroframe_recover_saturated(data, dq, g0)
    out = capsys.readouterr().out
    assert R < 0.02 * R_TRUE
    assert '[zeroframe R-curve check]' not in out
    assert np.all(rec[meas] < 0.02 * data[meas])


def test_healthy_curve_is_unchanged(monkeypatch, capsys):
    data, dq, g0, core, meas = _scene(calib='real')
    rec1, rim1, deep1, R1 = zeroframe_recover_saturated(data, dq, g0)
    out = capsys.readouterr().out
    monkeypatch.setenv('SATSTAR_ZF_RCURVE_SATCHECK', '0')
    rec0, rim0, deep0, R0 = zeroframe_recover_saturated(data, dq, g0)
    assert R1 == R0
    assert np.array_equal(rec1, rec0, equal_nan=True)
    assert np.array_equal(rim1, rim0) and np.array_equal(deep1, deep0)
    assert '[zeroframe R-curve check]' not in out


def test_too_few_measured_saturated_pixels_no_change(monkeypatch, capsys):
    data, dq, g0, core, meas = _scene(n_sat_meas=_RCURVE_SATCHECK_MIN_PX - 1)
    rec1, rim1, deep1, R1 = zeroframe_recover_saturated(data, dq, g0)
    out = capsys.readouterr().out
    monkeypatch.setenv('SATSTAR_ZF_RCURVE_SATCHECK', '0')
    rec0, rim0, deep0, R0 = zeroframe_recover_saturated(data, dq, g0)
    assert R1 == R0 and R1 < 0.02 * R_TRUE
    assert '[zeroframe R-curve check]' not in out


@pytest.mark.parametrize('bad_dq', [DNUBIT, HOTBIT])
def test_dnu_and_hot_saturated_pixels_do_not_enter_the_check(bad_dq, capsys):
    data, dq, g0, core, meas = _scene(sat_meas_dq=bad_dq)
    rec, rim, deep, R = zeroframe_recover_saturated(data, dq, g0)
    assert R < 0.02 * R_TRUE
    assert '[zeroframe R-curve check]' not in capsys.readouterr().out


def test_group0_saturated_pixels_do_not_enter_the_check(capsys):
    data, dq, g0, core, meas = _scene()
    rec, rim, deep, R = zeroframe_recover_saturated(
        data, dq, g0, group0_saturated=meas | core)
    assert R < 0.02 * R_TRUE
    assert '[zeroframe R-curve check]' not in capsys.readouterr().out


def test_precomputed_R_is_not_checked(capsys):
    data, dq, g0, core, meas = _scene()
    rec, rim, deep, R = zeroframe_recover_saturated(data, dq, g0, R=R_JUNK)
    assert R == R_JUNK
    assert '[zeroframe R-curve check]' not in capsys.readouterr().out


def test_keep_finite_still_gets_the_rebuilt_curve(monkeypatch, capsys):
    """Under KEEP_FINITE the measured SATURATED pixels are kept, and the
    pixels still rewritten (dilation-buffer rim) use the rebuilt curve."""
    monkeypatch.setenv('SATSTAR_ZF_KEEP_FINITE', '1')
    data, dq, g0, core, meas = _scene()
    rec, rim, deep, R = zeroframe_recover_saturated(data, dq, g0)
    assert abs(R / R_TRUE - 1) < 0.03
    assert not rim[meas].any()
    assert np.array_equal(rec[meas], data[meas])


@pytest.mark.parametrize('value, on', [
    ('0', False), ('off', False), ('false', False), ('1', True), ('on', True),
    ('', True)])
def test_switch_spellings(monkeypatch, value, on):
    monkeypatch.setenv('SATSTAR_ZF_RCURVE_SATCHECK', value)
    assert satstar_fit_switches()['rcurve_satcheck'] is on


def test_switch_default_on():
    assert satstar_fit_switches()['rcurve_satcheck'] is True


# --------------------------------------------------------------------------
# header fallback (too few measured SATURATED pixels)
# --------------------------------------------------------------------------

FEW = _RCURVE_SATCHECK_MIN_PX - 1


def test_too_few_measured_pixels_use_the_header_rate(capsys):
    data, dq, g0, core, meas = _scene(n_sat_meas=FEW)
    rec, rim, deep, R = zeroframe_recover_saturated(data, dq, g0,
                                                    R_header=R_TRUE)
    out = capsys.readouterr().out
    assert R == R_TRUE
    assert 'the header value is used' in out
    assert rim[meas].all()
    assert np.allclose(rec[meas], data[meas], rtol=1e-9)


def test_header_rate_within_factor_two_leaves_the_curve(capsys):
    data, dq, g0, core, meas = _scene(calib='real', n_sat_meas=FEW)
    rec0, rim0, deep0, R0 = zeroframe_recover_saturated(data, dq, g0)
    rec1, rim1, deep1, R1 = zeroframe_recover_saturated(
        data, dq, g0, R_header=1.9 * R_TRUE)
    assert R1 == R0
    assert np.array_equal(rec1, rec0, equal_nan=True)
    assert np.array_equal(rim1, rim0) and np.array_equal(deep1, deep0)
    assert '[zeroframe R-curve check]' not in capsys.readouterr().out


def test_header_rate_is_not_consulted_when_the_pixel_check_ran(capsys):
    data, dq, g0, core, meas = _scene()
    rec, rim, deep, R = zeroframe_recover_saturated(data, dq, g0,
                                                    R_header=10 * R_TRUE)
    out = capsys.readouterr().out
    assert abs(R / R_TRUE - 1) < 0.03
    assert 'curve rebuilt' in out
    assert 'header' not in out


def test_header_rate_is_not_consulted_on_a_healthy_checked_curve(capsys):
    data, dq, g0, core, meas = _scene(calib='real')
    rec0, rim0, deep0, R0 = zeroframe_recover_saturated(data, dq, g0)
    rec1, rim1, deep1, R1 = zeroframe_recover_saturated(
        data, dq, g0, R_header=10 * R_TRUE)
    assert R1 == R0
    assert np.array_equal(rec1, rec0, equal_nan=True)
    assert '[zeroframe R-curve check]' not in capsys.readouterr().out


def test_header_rate_off_with_the_check(monkeypatch, capsys):
    monkeypatch.setenv('SATSTAR_ZF_RCURVE_SATCHECK', '0')
    data, dq, g0, core, meas = _scene(n_sat_meas=FEW)
    rec, rim, deep, R = zeroframe_recover_saturated(data, dq, g0,
                                                    R_header=R_TRUE)
    assert R < 0.02 * R_TRUE
    assert '[zeroframe R-curve check]' not in capsys.readouterr().out


def test_header_rate_does_not_override_a_precomputed_R(capsys):
    data, dq, g0, core, meas = _scene(n_sat_meas=FEW)
    rec, rim, deep, R = zeroframe_recover_saturated(data, dq, g0, R=R_JUNK,
                                                    R_header=R_TRUE)
    assert R == R_JUNK
    assert '[zeroframe R-curve check]' not in capsys.readouterr().out


@pytest.mark.parametrize('bad', [None, np.nan, 0.0, -1.0])
def test_unusable_header_rate_is_ignored(bad, capsys):
    data, dq, g0, core, meas = _scene(n_sat_meas=FEW)
    rec, rim, deep, R = zeroframe_recover_saturated(data, dq, g0, R_header=bad)
    assert R < 0.02 * R_TRUE
    assert '[zeroframe R-curve check]' not in capsys.readouterr().out


def _nircam_header(**kw):
    h = fits.Header({'INSTRUME': 'NIRCAM', 'NFRAMES': 4, 'TFRAME': 10.73677})
    h.update(kw)
    return h


def test_zeroframe_header_R_nircam():
    # wd2 F187N nrca4: SHALLOW4, PHOTMJSR 39.23 -> 1.462 (cal/group0 of the
    # star pixels there: 1.44-1.51)
    assert zeroframe_header_R(_nircam_header(), 39.23) == pytest.approx(
        39.23 / (10.73677 * 2.5))
    assert zeroframe_header_R(_nircam_header(NFRAMES=1), 2.0) == pytest.approx(
        2.0 / 10.73677)


@pytest.mark.parametrize('header, photmjsr', [
    (None, 1.0),
    (fits.Header({'INSTRUME': 'MIRI', 'NFRAMES': 1, 'TFRAME': 2.775}), 1.0),
    (fits.Header({'INSTRUME': 'NIRCAM', 'NFRAMES': 4}), 1.0),
    (fits.Header({'INSTRUME': 'NIRCAM', 'TFRAME': 10.7}), 1.0),
    (fits.Header({'INSTRUME': 'NIRCAM', 'NFRAMES': 0, 'TFRAME': 10.7}), 1.0),
    (fits.Header({'INSTRUME': 'NIRCAM', 'NFRAMES': 4, 'TFRAME': 10.7}), None),
    (fits.Header({'INSTRUME': 'NIRCAM', 'NFRAMES': 4, 'TFRAME': 10.7}), 0.0),
    (fits.Header({'INSTRUME': 'NIRCAM', 'NFRAMES': 4, 'TFRAME': 10.7}),
     np.nan)])
def test_zeroframe_header_R_unusable(header, photmjsr):
    assert zeroframe_header_R(header, photmjsr) is None


def test_fit_anchor_hands_the_header_rate_on(monkeypatch):
    seen = {}

    def _recover(data, dq, group0, **kw):
        seen.update(kw)
        shp = np.shape(data)
        return data, np.zeros(shp, bool), np.zeros(shp, bool), np.nan
    monkeypatch.setattr(SSF, 'zeroframe_recover_saturated', _recover)
    data, dq, g0, core, meas = _scene()
    SSF.zeroframe_fit_anchor(data, dq, g0, R_header=1.25)
    assert seen['R_header'] == 1.25


class _AnchorReached(Exception):
    pass


def test_get_saturated_stars_hands_the_header_rate_to_the_anchor(monkeypatch):
    """get_saturated_stars reads PHOTMJSR / t(group 0) from the frame headers
    and passes it to ``zeroframe_fit_anchor``.  The anchor is stubbed to stop
    the fit there."""
    n = 200
    sci = np.ones((n, n))
    dq = np.zeros((n, n), dtype=np.uint32)
    dq[98:103, 98:103] = SATBIT
    sci[dq != 0] = np.nan
    wcs_hdr = fits.Header({'CTYPE1': 'RA---TAN', 'CTYPE2': 'DEC--TAN',
                           'CRPIX1': 100, 'CRPIX2': 100, 'CRVAL1': 150.0,
                           'CRVAL2': 2.0, 'CDELT1': -1.7e-5, 'CDELT2': 1.7e-5,
                           'BUNIT': 'MJy/sr', 'PHOTMJSR': 39.23})
    fh = fits.HDUList([
        fits.PrimaryHDU(header=fits.Header({
            'INSTRUME': 'NIRCAM', 'FILTER': 'F187N', 'PUPIL': 'CLEAR',
            'DETECTOR': 'NRCA4', 'MODULE': 'A', 'CHANNEL': 'SHORT',
            'NFRAMES': 4, 'TFRAME': 10.73677, 'NGROUPS': 7, 'GROUPGAP': 1})),
        fits.ImageHDU(sci, header=wcs_hdr, name='SCI'),
        fits.ImageHDU(np.full((n, n), 0.1), name='ERR'),
        fits.ImageHDU(dq, name='DQ'),
        fits.ImageHDU(np.where(dq != 0, np.nan, 0.01), name='VAR_POISSON')])
    seen = {}

    def _anchor(data, dq, zeroframe, **kw):
        seen.update(kw)
        raise _AnchorReached
    monkeypatch.setattr(SSF, 'zeroframe_fit_anchor', _anchor)
    with pytest.raises(_AnchorReached):
        SSF.get_saturated_stars(fh, zeroframe=np.ones((n, n)), plot=False)
    assert seen['R_header'] == pytest.approx(39.23 / (10.73677 * 2.5))
