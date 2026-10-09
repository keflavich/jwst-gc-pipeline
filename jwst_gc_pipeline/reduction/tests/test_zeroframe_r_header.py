"""SATSTAR_ZF_R_HEADER: the ZEROFRAME rim rewrite uses PHOTMJSR / t(group 0).

On a crowded frame most unsaturated pixels above R_g0_min = 2000 DN are the
wings of saturated stars.  Those wings carry charge that migrated out of the
core over the ramp, so the measured R(g0) curve reads high where saturated
stars are dense: wd2 F150W nrcb3 (SHALLOW4) sat 7.5% above the header rate
and nrcb1 1.5%, and the rim and the recovered-core cap inherited the excess.
Pinned here:

1. with the switch off, inflated wing pixels set R (the earlier behaviour);
2. with it on, the rim is rewritten with R_header and the log says so;
3. no R_header, an explicit R, or a frame where no R could be measured
   leave the earlier behaviour in place;
4. the default follows NIRCAM_SATSTAR_RECOVERED_CAP; an export wins, and a
   typo raises.
"""
import numpy as np
import pytest
from scipy import ndimage

from jwst_gc_pipeline.reduction.saturated_star_finding import (
    satstar_fit_switches, zeroframe_fit_anchor, zeroframe_recover_saturated)

SATBIT = 2
R_TRUE = 0.08
WING_INFL = 1.08
_ENV = ('SATSTAR_ZF_R_HEADER', 'NIRCAM_SATSTAR_RECOVERED_CAP',
        'SATSTAR_ZF_KEEP_FINITE', 'SATSTAR_ZF_G0_GROUPDQ',
        'SATSTAR_ZF_FIRST_FRAME', 'SATSTAR_ZF_RCURVE_GUARD',
        'SATSTAR_ZF_RCURVE_MAXSTEP', 'SATSTAR_ZF_RCURVE_SATCHECK')


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)


def _scene(n_far=90, stars=((60, 60), (60, 140), (140, 60), (140, 140))):
    """Dark frame with saturated stars (core at the group-0 rail, SAT rim at
    g0 = 25000) whose unsaturated wings, out to 4 px from the SATURATED edge,
    read WING_INFL x R_TRUE x group0 at group0 = 2100-5000 DN, plus ``n_far``
    calibration pixels at R_TRUE in a far row."""
    ny, nx = 200, 200
    rng = np.random.default_rng(7)
    g0 = rng.normal(10, 3, (ny, nx))
    data = g0 * R_TRUE
    dq = np.zeros((ny, nx), dtype=np.uint32)
    yy, xx = np.mgrid[0:ny, 0:nx]
    core = np.zeros((ny, nx), dtype=bool)
    rim = np.zeros((ny, nx), dtype=bool)
    for y0, x0 in stars:
        rr = np.hypot(xx - x0, yy - y0)
        core |= rr < 1.5
        rim |= (rr >= 1.5) & (rr < 4)
    sat = core | rim
    wing = (~sat) & (ndimage.distance_transform_edt(~sat) <= 4)
    g0[wing] = rng.uniform(2100, 5000, int(wing.sum()))
    data[wing] = g0[wing] * R_TRUE * WING_INFL
    cols = np.arange(n_far) % 180 + 10
    rows = np.where(np.arange(n_far) < 90, 100, 102)
    g0[rows, cols] = rng.uniform(2100, 5000, n_far)
    data[rows, cols] = g0[rows, cols] * R_TRUE
    g0[core] = 48000.0
    g0[rim] = 25000.0
    dq[sat] = SATBIT
    data[core] = 0.0
    data[rim] = 25000.0 * R_TRUE * 1.15
    return data, dq, g0, core, rim, wing


def test_inflated_wings_set_r_with_the_switch_off(capsys):
    data, dq, g0, core, rim, wing = _scene()
    assert int(wing.sum()) > 90          # wings outnumber the far pixels
    _, _, _, R = zeroframe_recover_saturated(data, dq, g0, R_header=R_TRUE)
    assert R > R_TRUE * (1 + 0.5 * (WING_INFL - 1))
    assert '[zeroframe R header]' not in capsys.readouterr().out


def test_header_rate_rewrites_the_rim(monkeypatch, capsys):
    monkeypatch.setenv('SATSTAR_ZF_R_HEADER', '1')
    data, dq, g0, core, rim, wing = _scene()
    rec, rim_mask, deep, R = zeroframe_recover_saturated(
        data, dq, g0, R_header=R_TRUE)
    out = capsys.readouterr().out
    assert R == R_TRUE
    assert rim_mask[rim].all()
    assert np.allclose(rec[rim], R_TRUE * 25000.0)
    assert deep[core].all()
    assert '[zeroframe R header] rim rewritten with R_header=0.08' in out
    assert 'measured bright-end R=' in out


def test_fit_anchor_passes_the_header_rate(monkeypatch):
    monkeypatch.setenv('SATSTAR_ZF_R_HEADER', '1')
    data, dq, g0, core, rim, wing = _scene()
    rec, deep, rim_mask, _ = zeroframe_fit_anchor(data, dq, g0,
                                                  R_header=R_TRUE)
    assert np.allclose(rec[rim], R_TRUE * 25000.0)


def test_no_header_rate_keeps_the_measured_curve(monkeypatch, capsys):
    data, dq, g0, core, rim, wing = _scene()
    _, _, _, R_off = zeroframe_recover_saturated(data, dq, g0)
    monkeypatch.setenv('SATSTAR_ZF_R_HEADER', '1')
    _, _, _, R_on = zeroframe_recover_saturated(data, dq, g0)
    _, _, _, R_bad = zeroframe_recover_saturated(data, dq, g0,
                                                 R_header=float('nan'))
    assert R_on == R_off == R_bad
    assert '[zeroframe R header]' not in capsys.readouterr().out


def test_explicit_r_is_not_replaced(monkeypatch):
    monkeypatch.setenv('SATSTAR_ZF_R_HEADER', '1')
    data, dq, g0, core, rim, wing = _scene()
    rec, _, _, R = zeroframe_recover_saturated(data, dq, g0, R=0.05,
                                               R_header=R_TRUE)
    assert R == 0.05
    assert np.allclose(rec[rim], 0.05 * 25000.0)


def test_frame_without_a_measured_r_is_left_alone(monkeypatch, capsys):
    """Fewer than 50 calibration pixels: no R, no rewrite, with or without
    the header rate."""
    monkeypatch.setenv('SATSTAR_ZF_R_HEADER', '1')
    data, dq, g0, core, rim, wing = _scene(n_far=10, stars=((60, 60),))
    data[wing] = np.nan                  # leave 10 calibration pixels
    rec, rim_mask, deep, R = zeroframe_recover_saturated(
        data, dq, g0, R_header=R_TRUE)
    assert not np.isfinite(R)
    assert not rim_mask.any()
    assert '[zeroframe R header]' not in capsys.readouterr().out


def test_default_follows_the_recovered_cap(monkeypatch):
    assert satstar_fit_switches()['r_header'] is False
    monkeypatch.setenv('NIRCAM_SATSTAR_RECOVERED_CAP', '1')
    assert satstar_fit_switches()['r_header'] is True
    monkeypatch.setenv('SATSTAR_ZF_R_HEADER', '0')
    assert satstar_fit_switches()['r_header'] is False
    monkeypatch.setenv('SATSTAR_ZF_R_HEADER', '')
    assert satstar_fit_switches()['r_header'] is True
    monkeypatch.setenv('NIRCAM_SATSTAR_RECOVERED_CAP', '0')
    monkeypatch.setenv('SATSTAR_ZF_R_HEADER', 'on')
    assert satstar_fit_switches()['r_header'] is True
    monkeypatch.setenv('SATSTAR_ZF_R_HEADER', 'ture')
    with pytest.raises(ValueError, match='SATSTAR_ZF_R_HEADER'):
        satstar_fit_switches()
