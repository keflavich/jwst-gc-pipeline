"""R(g0)-curve guard and the other satstar fit switches of issue #972.

gc-treasury F480M (BRIGHT2, NGROUPS=4), 2026-09-25: real star pixels only reach
group-0 ~11-13.5k DN, so the top three calibration bins are filled by 36-58
DQ-flagged junk pixels (JUMP_DET / hot) with cal/group0 ~0.0005 against a real
R ~0.08.  With no consistency check the curve's bright end collapses and every
weakly saturated core is rewritten to ~3% of its rate.  Pinned here:

1. default: the bin-to-bin step guard is ON and the bright-end R is the last
   trustworthy bin; SATSTAR_ZF_RCURVE_GUARD=0 restores the collapsed curve and
   the per-frame log prints a WARNING for it;
2. a single surviving bin is used as-is (the median over all calibration
   pixels would re-admit the rejected junk);
3. a healthy curve is unchanged by the guard;
4. SATSTAR_ZF_KEEP_FINITE=1 keeps finite, non-DO_NOT_USE SATURATED pixels
   (neither rewritten nor masked), including when that leaves no rim pixel;
5. SATSTAR_OBS_PK_FROM_CRF=1 reads the observed peak from the crf values;
6. SATSTAR_QFIT_LOCAL_* compute a local qfit and, with the gate on, judge
   NIRCam in-FOV fits on it.
"""
import numpy as np
import pytest

from jwst_gc_pipeline.reduction.saturated_star_finding import (
    satstar_fit_switches, satstar_local_qfit, satstar_observed_peak,
    satstar_qfit_for_gate, zeroframe_fit_anchor, zeroframe_recover_saturated)

SATBIT, DNUBIT, JUMPBIT = 2, 1, 4
R_TRUE = 0.08
R_JUNK = 0.0005
_ENV = ('SATSTAR_ZF_RCURVE_GUARD', 'SATSTAR_ZF_RCURVE_DQ0',
        'SATSTAR_ZF_RCURVE_MAXSTEP', 'SATSTAR_ZF_KEEP_FINITE',
        'SATSTAR_OBS_PK_FROM_CRF', 'SATSTAR_QFIT_LOCAL_GATE',
        'SATSTAR_QFIT_LOCAL_R', 'SATSTAR_QFIT_LOCAL_MAX')


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in _ENV:
        monkeypatch.delenv(k, raising=False)


def _edges(ceiling=0.9 * 48000.0):
    return np.geomspace(2000.0, ceiling, 9)


def _scene(real_bins=range(5), junk_bins=(5, 6, 7), junk_dq=JUMPBIT,
           n_per_bin=30):
    """Dark frame, one saturated star (core at the group-0 rail, SAT rim at
    g0=25000 inside bin 6), n_per_bin real calibration pixels (R_TRUE) in each
    of ``real_bins`` and n_per_bin junk pixels (R_JUNK, DQ=junk_dq) in each of
    ``junk_bins``."""
    ny, nx = 160, 160
    rng = np.random.default_rng(3)
    g0 = rng.normal(10, 3, (ny, nx))
    data = g0 * R_TRUE
    dq = np.zeros((ny, nx), dtype=np.uint32)
    e = _edges()
    for k in range(8):
        lo, hi = e[k], e[k + 1]
        vals = rng.uniform(lo * 1.05, hi * 0.95, n_per_bin)
        row = 110 + k
        cols = slice(10, 10 + n_per_bin)
        if k in real_bins:
            g0[row, cols] = vals
            data[row, cols] = vals * R_TRUE
        elif k in junk_bins:
            g0[row, cols] = vals
            data[row, cols] = vals * R_JUNK
            dq[row, cols] = junk_dq
    yy, xx = np.mgrid[0:ny, 0:nx]
    rr = np.hypot(xx - 60, yy - 60)
    core = rr < 1.5
    rim = (rr >= 1.5) & (rr < 4)
    g0[core] = 48000.0
    g0[rim] = 25000.0
    dq[core | rim] = SATBIT
    data[core] = 0.0            # get_saturated_stars zeroes NaN-VAR_POISSON px
    data[rim] = 25000.0 * R_TRUE * 1.15
    return data, dq, g0, core, rim


# --------------------------------------------------------------------------
# R(g0) step guard
# --------------------------------------------------------------------------

def test_guard_is_on_by_default_and_restores_bright_end(capsys):
    data, dq, g0, core, rim = _scene()
    rec, rim_mask, deep, R = zeroframe_recover_saturated(data, dq, g0)
    out = capsys.readouterr().out
    assert abs(R / R_TRUE - 1) < 0.02
    assert np.allclose(rec[rim], R_TRUE * 25000.0, rtol=0.02)
    assert 'truncated R(g0) curve' in out
    assert '[zeroframe R-curve]' in out and 'guard=on' in out
    assert 'WARNING' not in out


def test_guard_off_keeps_the_collapse_and_warns(monkeypatch, capsys):
    """SATSTAR_ZF_RCURVE_GUARD=0 is the pre-#972 behaviour: the collapsed
    bright end rewrites the rim to a few % of its rate, and the always-on
    per-frame log line says so."""
    monkeypatch.setenv('SATSTAR_ZF_RCURVE_GUARD', '0')
    data, dq, g0, core, rim = _scene()
    rec, rim_mask, deep, R = zeroframe_recover_saturated(data, dq, g0)
    out = capsys.readouterr().out
    assert R < 0.05 * R_TRUE
    assert rim_mask[rim].all()
    assert np.all(rec[rim] < 0.1 * R_TRUE * 25000.0)
    assert '[zeroframe R-curve]' in out and 'R_bright/R_faint' in out
    assert 'guard=off' in out
    assert 'WARNING [zeroframe R-curve]' in out
    assert 'SATSTAR_ZF_RCURVE_GUARD is off' in out


@pytest.mark.parametrize('value, on', [
    ('0', False), ('false', False), ('OFF', False), ('no', False), ('', False),
    ('1', True), ('true', True), ('yes', True)])
def test_guard_switch_spellings(monkeypatch, value, on):
    monkeypatch.setenv('SATSTAR_ZF_RCURVE_GUARD', value)
    assert satstar_fit_switches()['rcurve_guard'] is on


def test_rcurve_dq0_option_is_not_read(monkeypatch):
    """SATSTAR_ZF_RCURVE_DQ0 (DQ==0-only calibration pixels) was measured
    harmful on F480M (bright-end R 0.076 -> 0.086-0.089: real bright star
    pixels are mostly DQ-flagged) and is not shipped; setting it changes
    nothing."""
    data, dq, g0, core, rim = _scene()
    _, _, _, R_default = zeroframe_recover_saturated(data, dq, g0)
    monkeypatch.setenv('SATSTAR_ZF_RCURVE_DQ0', '1')
    _, _, _, R_dq0 = zeroframe_recover_saturated(data, dq, g0)
    assert R_dq0 == R_default


def test_step_guard_catches_unflagged_junk():
    """Junk with DQ==0 is caught by the step, not by a DQ filter."""
    data, dq, g0, core, rim = _scene(junk_dq=0)
    _, _, _, R = zeroframe_recover_saturated(data, dq, g0)
    assert abs(R / R_TRUE - 1) < 0.02


def test_maxstep_is_configurable(monkeypatch):
    monkeypatch.setenv('SATSTAR_ZF_RCURVE_MAXSTEP', '1e9')
    data, dq, g0, core, rim = _scene()
    _, _, _, R = zeroframe_recover_saturated(data, dq, g0)
    assert R < 0.05 * R_TRUE      # a step that large is never "too large"


def test_single_surviving_bin_used_not_global_median():
    """Only bin 0 is real, bin 1 is unflagged junk: the guard keeps bin 0 and
    must use its median, not the median of all 60 calibration pixels."""
    data, dq, g0, core, rim = _scene(real_bins=(0,), junk_bins=(1,), junk_dq=0)
    _, _, _, R = zeroframe_recover_saturated(data, dq, g0)
    assert abs(R / R_TRUE - 1) < 0.02


def test_guard_noop_on_healthy_curve(monkeypatch):
    data, dq, g0, core, rim = _scene(real_bins=range(8), junk_bins=())
    monkeypatch.setenv('SATSTAR_ZF_RCURVE_GUARD', '0')
    rec0, rim0, deep0, R0 = zeroframe_recover_saturated(data, dq, g0)
    monkeypatch.delenv('SATSTAR_ZF_RCURVE_GUARD')
    rec1, rim1, deep1, R1 = zeroframe_recover_saturated(data, dq, g0)
    assert R1 == R0
    assert np.array_equal(rim0, rim1) and np.array_equal(deep0, deep1)
    assert np.array_equal(rec0, rec1, equal_nan=True)


# --------------------------------------------------------------------------
# SATSTAR_ZF_KEEP_FINITE
# --------------------------------------------------------------------------

def test_keep_finite_saturated_pixels(monkeypatch):
    """Finite, nonzero SAT pixels without DO_NOT_USE keep their ramp-fit rate
    and stay unmasked; SAT&DO_NOT_USE pixels (zeroed) are still rewritten."""
    data, dq, g0, core, rim = _scene()
    lost = rim & (np.arange(rim.size).reshape(rim.shape) % 2 == 0)
    dq[lost] |= DNUBIT
    data[lost] = 0.0
    kept = rim & ~lost
    # half of the kept pixels have group-0 above the pile-up ceiling: deep core
    # (masked) by default, data under KEEP_FINITE
    xx = np.mgrid[0:rim.shape[0], 0:rim.shape[1]][1]
    kept_hi = kept & (xx < 60)
    g0[kept_hi] = 46000.0
    rec_a, rim_a, deep_a, _ = zeroframe_recover_saturated(data, dq, g0)
    assert rim_a[kept & ~kept_hi].all()
    assert deep_a[kept_hi].all()
    monkeypatch.setenv('SATSTAR_ZF_KEEP_FINITE', '1')
    rec_b, rim_b, deep_b, _ = zeroframe_recover_saturated(data, dq, g0)
    assert not rim_b[kept].any()
    assert np.array_equal(rec_b[kept], data[kept])
    assert not deep_b[kept].any()
    assert rim_b[lost].all()
    assert np.allclose(rec_b[lost], R_TRUE * 25000.0, rtol=0.02)


def test_default_anchor_rewrites_rim_and_masks_deep_core():
    """The fit-anchor wrapper get_saturated_stars calls: rewritten data, the
    deep core as the fit mask, and no rewrite delta unless asked for."""
    data, dq, g0, core, rim = _scene()
    out, zf_deep, rim_mask, delta = zeroframe_fit_anchor(data, dq, g0)
    assert rim_mask[rim].all()
    assert np.allclose(out[rim], R_TRUE * 25000.0, rtol=0.02)
    assert zf_deep is not None and zf_deep[core].all() and not zf_deep[rim].any()
    assert delta is None


def test_keep_finite_with_no_rim_left_masks_only_the_deep_core(monkeypatch,
                                                              capsys):
    """Every rim pixel is finite and kept, so nothing is rewritten.  The fit
    mask must still be the deep core: returning None would make
    get_saturated_stars fall back to the whole any-group SATURATED blob and
    mask the pixels KEEP_FINITE keeps."""
    monkeypatch.setenv('SATSTAR_ZF_KEEP_FINITE', '1')
    data, dq, g0, core, rim = _scene()
    before = data.copy()
    out, zf_deep, rim_mask, delta = zeroframe_fit_anchor(data, dq, g0)
    assert not rim_mask.any()
    assert zf_deep is not None
    assert zf_deep[core].all()
    assert not zf_deep[rim].any()
    assert int(zf_deep.sum()) < int((core | rim).sum())
    assert np.array_equal(out, before)
    assert 'no rim pixel to rewrite' in capsys.readouterr().out


def test_empty_rim_without_keep_finite_keeps_the_blob_mask():
    """Unchanged pre-#972 path: nothing recovered -> no anchor mask."""
    data, dq, g0, core, rim = _scene()
    g0[rim] = 48000.0                      # the whole star is deep core
    out, zf_deep, rim_mask, delta = zeroframe_fit_anchor(data, dq, g0)
    assert not rim_mask.any()
    assert zf_deep is None
    assert out is data


def test_no_usable_R_leaves_the_frame_alone(monkeypatch):
    monkeypatch.setenv('SATSTAR_ZF_KEEP_FINITE', '1')
    data, dq, g0, core, rim = _scene()
    out, zf_deep, rim_mask, delta = zeroframe_fit_anchor(data, dq, g0[:10])
    assert out is data and zf_deep is None and delta is None


# --------------------------------------------------------------------------
# SATSTAR_OBS_PK_FROM_CRF
# --------------------------------------------------------------------------

def test_rewrite_delta_only_with_obs_pk_from_crf(monkeypatch):
    monkeypatch.setenv('SATSTAR_ZF_RCURVE_GUARD', '0')   # collapsed rewrite
    data, dq, g0, core, rim = _scene()
    _, _, _, delta_off = zeroframe_fit_anchor(data, dq, g0)
    assert delta_off is None
    monkeypatch.setenv('SATSTAR_OBS_PK_FROM_CRF', '1')
    out, _, rim_mask, delta = zeroframe_fit_anchor(data, dq, g0)
    assert delta is not None
    assert np.all(delta[~rim_mask] == 0)
    assert np.allclose(out - delta, data)


def test_observed_peak_reads_the_crf_not_the_rewrite(monkeypatch):
    """With the collapsed curve the rewrite drives the rim to ~3% of the crf;
    the observed peak read through the delta is the crf rim value."""
    monkeypatch.setenv('SATSTAR_ZF_RCURVE_GUARD', '0')
    monkeypatch.setenv('SATSTAR_OBS_PK_FROM_CRF', '1')
    data, dq, g0, core, rim = _scene()
    out, zf_deep, _, delta = zeroframe_fit_anchor(data, dq, g0)
    sl = (slice(40, 81), slice(40, 81))
    mask = zf_deep[sl]
    crf_peak = float(np.max(np.where(mask, -np.inf, data[sl])))
    pk_rewrite = satstar_observed_peak(out[sl], mask)
    pk_crf = satstar_observed_peak(out[sl], mask, delta[sl])
    assert pk_crf == pytest.approx(crf_peak)
    assert pk_crf == pytest.approx(25000.0 * R_TRUE * 1.15)
    assert pk_rewrite < 0.1 * pk_crf


def test_observed_peak_ignores_masked_pixels_and_handles_empty():
    cut = np.array([[1.0, 9.0], [3.0, np.nan]])
    mask = np.array([[False, True], [False, False]])
    assert satstar_observed_peak(cut, mask) == 3.0
    assert np.isnan(satstar_observed_peak(cut, np.ones_like(mask)))
    assert np.isnan(satstar_observed_peak(np.zeros((0, 0)),
                                          np.zeros((0, 0), dtype=bool)))


# --------------------------------------------------------------------------
# SATSTAR_QFIT_LOCAL_*
# --------------------------------------------------------------------------

def _qfit_scene():
    ny = nx = 81
    yy, xx = np.mgrid[0:ny, 0:nx]
    r2 = (xx - 40.0) ** 2 + (yy - 40.0) ** 2
    model = 1000.0 * np.exp(-r2 / (2 * 2.0 ** 2))
    cutout = model.copy()
    mask = np.zeros((ny, nx), dtype=bool)
    return cutout, model, mask, r2


def test_local_qfit_ignores_a_distant_neighbour_residual():
    cutout, model, mask, r2 = _qfit_scene()
    flux = float(model.sum())
    nb = (r2 > 30 ** 2) & (r2 < 33 ** 2)        # a neighbour's leftover ring
    cutout[nb] += 50.0
    box_q = float(np.abs(cutout - model).sum() / flux)
    loc_q = satstar_local_qfit(cutout, model, mask, r2, 10.0, 0.0, flux)
    assert box_q > 1.0
    assert loc_q == pytest.approx(0.0, abs=1e-12)
    cutout[(r2 < 3 ** 2)] += 20.0               # a residual AT the star
    assert satstar_local_qfit(cutout, model, mask, r2, 10.0, 0.0, flux) > 0


def test_local_qfit_is_nan_without_a_usable_fit():
    cutout, model, mask, r2 = _qfit_scene()
    assert np.isnan(satstar_local_qfit(cutout, model, mask, r2, 10.0, 0.0, 0.0))
    assert np.isnan(satstar_local_qfit(cutout, model, mask, r2, 10.0, 0.0,
                                       np.nan))
    assert np.isnan(satstar_local_qfit(cutout, model, mask, r2, 0.0, 0.0, 1.0))
    assert np.isnan(satstar_local_qfit(cutout, model, np.ones_like(mask), r2,
                                       10.0, 0.0, 1.0))


def test_local_qfit_subtracts_the_local_background():
    cutout, model, mask, r2 = _qfit_scene()
    flux = float(model.sum())
    assert satstar_local_qfit(cutout + 5.0, model, mask, r2, 10.0, 5.0,
                              flux) == pytest.approx(0.0, abs=1e-12)


def test_qfit_local_switch_defaults(monkeypatch):
    sw = satstar_fit_switches()
    assert sw['qfit_local_r'] == 0 and not sw['qfit_local_gate']
    monkeypatch.setenv('SATSTAR_QFIT_LOCAL_GATE', '1')
    sw = satstar_fit_switches()
    assert sw['qfit_local_gate'] and sw['qfit_local_r'] == 10.0
    assert sw['qfit_local_max'] == 1.0
    monkeypatch.setenv('SATSTAR_QFIT_LOCAL_R', '0')
    assert not satstar_fit_switches()['qfit_local_gate']
    monkeypatch.delenv('SATSTAR_QFIT_LOCAL_GATE')
    monkeypatch.setenv('SATSTAR_QFIT_LOCAL_R', '12')
    sw = satstar_fit_switches()          # column only, no gate
    assert sw['qfit_local_r'] == 12.0 and not sw['qfit_local_gate']


def test_gate_uses_box_qfit_unless_the_local_gate_is_on(monkeypatch):
    kw = dict(is_miri=False, forced_source=False)
    assert satstar_qfit_for_gate(7.0, 0.2, 5.0, **kw) == (7.0, 5.0)
    monkeypatch.setenv('SATSTAR_QFIT_LOCAL_GATE', '1')
    monkeypatch.setenv('SATSTAR_QFIT_LOCAL_MAX', '0.8')
    assert satstar_qfit_for_gate(7.0, 0.2, 5.0, **kw) == (0.2, 0.8)
    # MIRI, forced sources and a NaN local qfit keep the box qfit
    assert satstar_qfit_for_gate(7.0, 0.2, 5.0, is_miri=True,
                                 forced_source=False) == (7.0, 5.0)
    assert satstar_qfit_for_gate(7.0, 0.2, 5.0, is_miri=False,
                                 forced_source=True) == (7.0, 5.0)
    assert satstar_qfit_for_gate(7.0, np.nan, 5.0, **kw) == (7.0, 5.0)
