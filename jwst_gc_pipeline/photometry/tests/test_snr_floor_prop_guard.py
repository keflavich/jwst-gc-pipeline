"""Guard on the rows only the propagated S/N floor admits (issue #1025).

Rows with per-frame S/N below the floor and merged-flux S/N above it are the
propagated-only rows.  The guard drops three subsets of them (flags-only
route, finite low prominence, low qfit x S/N), each behind its own switch.
"""
import ast
import os

import numpy as np
import pytest
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry.cataloging import _filter_extended_emission
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS
from jwst_gc_pipeline.photometry.prop_floor_guard import prop_floor_guard_drop
from jwst_gc_pipeline.photometry.tests.test_snr_floor_propagated import _mk

ALL_ON = dict(prop_floor_prom_min=4.0, prop_floor_drop_flags=True,
              prop_floor_qfit_snr_min=3.0)


def _kept(t, on=True, **kw):
    out = _filter_extended_emission(t, local_snr_min=5.0, snr_floor_propagated=on,
                                    sky_clean_keep=False, **kw)
    return set(np.asarray(out['id']).tolist())


def test_flags_only_row_dropped_only_when_propagated_only():
    # merged S/N 16 for all; per-frame S/N 4, 4, 6 (the last clears the
    # per-frame floor too)
    t = _mk([2.5, 2.5, 2.5])
    t['flux_err'] = [10.0, 10.0, 40.0 / 6.0]
    assert _kept(t) == {0, 1, 2}
    assert _kept(t, prop_floor_drop_flags=True) == {2}


def test_flags_rule_leaves_rows_with_another_route():
    # flags = 1 rows that the peak-SB branch also admits are not flags-only
    t, data, w = _i2d_case([(7.0, 100, 150), (7.0, 200, 150)])
    t['flags'] = 1
    kept, _ = _kept_i2d(t, data, w, prop_floor_drop_flags=True)
    assert kept == {0, 1}
    # without the image the same rows have only the flags route
    assert _kept(_mk([2.5, 2.5, 2.5]), prop_floor_drop_flags=True) == set()


def test_flags_rule_leaves_recovered_rows():
    # qfit 0.5 lies in the recover tier (qfit_max 0.4 < 0.5 <= 0.6); the gate
    # is off, so the recover tier is a second star-like route besides flags
    t = _mk([2.5, 2.5, 2.5])
    kw = dict(qfit_recover_max=0.6, recover_prom_gate=False)
    assert _kept(t, **kw) == {0, 1, 2}
    assert _kept(t, prop_floor_drop_flags=True, **kw) == {0, 1, 2}
    assert _kept(t, prop_floor_drop_flags=True) == set()


def test_guard_inert_without_propagated_floor():
    t = _mk([2.5, 5.0, 10.0])
    t['flux_err'] = [10.0, 5.0, 4.0]
    off = _kept(t, on=False)
    assert _kept(t, on=False, **ALL_ON) == off
    # a catalogue without flux_err_prop has no propagated-only rows either
    t2 = _mk(None)
    assert _kept(t2, **ALL_ON) == _kept(t2)


def _six_row_table():
    t = _mk([2.5, 5.0, 10.0, 2.5, 2.5, 2.5], n=6)
    t['flux_err'] = [10.0, 5.0, 10.0, 4.0, 10.0, 10.0]
    t['qfit'] = [0.5, 0.5, 0.5, 0.5, 0.1, 0.5]
    return t


def test_guard_subset_ordering():
    t = _six_row_table()
    per_frame = _kept(t, on=False)
    guarded = _kept(t, **ALL_ON)
    assert per_frame <= guarded
    assert guarded <= _kept(t)


def test_guard_with_rules_maxed_reproduces_per_frame_floor():
    # all three rules at very large values and no re-admitting tier: the
    # propagated floor plus the guard keeps exactly the per-frame-floor rows
    t = _six_row_table()
    unguarded = _kept(t)
    per_frame = _kept(t, on=False)
    assert per_frame < unguarded               # the guard has rows to remove
    maxed = dict(prop_floor_prom_min=1e9, prop_floor_drop_flags=True,
                 prop_floor_qfit_snr_min=1e9)
    assert _kept(t, **maxed) == per_frame
    # each rule alone covers these flags-only qfit-0.5 rows except the
    # prominence rule, which keeps the NaN prominence of an image-less table
    assert _kept(t, prop_floor_drop_flags=True) == per_frame
    assert _kept(t, prop_floor_qfit_snr_min=1e9) == per_frame
    assert _kept(t, prop_floor_prom_min=1e9) == unguarded


def test_later_tier_readmits_a_guard_dropped_row_and_the_log_says_so(capsys):
    t = _mk([2.5, 2.5, 2.5])
    t['nmatch'] = [5, 5, 1]
    t['std_ra'] = 0.0
    t['std_dec'] = 0.0
    kw = dict(prop_floor_drop_flags=True)
    assert _kept(t, **kw) == set()
    out = capsys.readouterr().out
    assert '0 of the 3 dropped row(s) are back in the final catalogue' in out
    assert _kept(t, nmatch_confirm=3, nmatch_confirm_qfit_max=0.6, **kw) == {0, 1}
    out = capsys.readouterr().out
    assert '3 of 3 propagated-only row(s) dropped' in out
    assert '2 of the 3 dropped row(s) are back in the final catalogue' in out


def test_guard_log_silent_when_all_rules_off(capsys):
    _kept(_mk([2.5, 2.5, 2.5]))
    assert 'propagated-floor guard' not in capsys.readouterr().out


def test_guard_not_applied_on_miri_path_and_logs_it(capsys):
    t = _mk([2.5, 2.5, 2.5])
    out = _filter_extended_emission(t, min_prominence=5.0, local_snr_min=5.0,
                                    snr_floor_propagated=True,
                                    prop_floor_drop_flags=True)
    log = capsys.readouterr().out
    assert 'not applied on the MIRI prominence path' in log
    assert 'propagated-floor guard: 0 of' not in log
    # the MIRI keep is prominence alone, so the switch changes nothing
    off = _filter_extended_emission(t, min_prominence=5.0, local_snr_min=5.0,
                                    snr_floor_propagated=True)
    assert list(out['id']) == list(off['id'])
    assert 'not applied on the MIRI' not in capsys.readouterr().out


def _i2d_case(amps_xy, qfit=0.5):
    """Merged-only (per-frame S/N 4, merged 16) peak-SB rows on a noise image."""
    rng = np.random.default_rng(5)
    ny = nx = 300
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [nx / 2, ny / 2]
    w.wcs.crval = [266.5, -28.7]
    w.wcs.cdelt = [-0.03 / 3600, 0.03 / 3600]
    data = rng.normal(0.0, 1.0, (ny, nx))
    yy, xx = np.mgrid[0:ny, 0:nx]
    for amp, x, y in amps_xy:
        data += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.0 ** 2))
    n = len(amps_xy)
    t = Table({
        'id': np.arange(n),
        'skycoord': w.pixel_to_world(np.array([a[1] for a in amps_xy], float),
                                     np.array([a[2] for a in amps_xy], float)),
        'qfit': np.full(n, qfit), 'flags': np.zeros(n, int),
        'local_bkg': np.full(n, 0.1), 'group_size': np.ones(n, int),
        'is_saturated': np.zeros(n, bool),
        'flux': np.full(n, 40.0), 'flux_err': np.full(n, 10.0),
        'flux_err_prop': np.full(n, 2.5),
    })
    return t, data, w


def _kept_i2d(t, data, w, **kw):
    out = _filter_extended_emission(t, data_i2d_image=data, ww_i2d=w, label='test',
                                    local_snr_min=5.0, snr_floor_propagated=True,
                                    sky_clean_keep=False, star_prom_peak_min=0.0,
                                    **kw)
    return set(np.asarray(out['id']).tolist()), out


def test_prominence_rule_peak_branch_and_nan_prominence():
    # row 0 prominence ~3, row 1 ~7, row 2 near the edge (prominence NaN)
    t, data, w = _i2d_case([(3.0, 100, 150), (7.0, 200, 150), (7.0, 3, 150)])
    kept, out = _kept_i2d(t, data, w)
    assert kept == {0, 1, 2}
    prom = np.asarray(out['prominence'])
    assert 2.0 < prom[0] < 4.0 and prom[1] > 5.0 and np.isnan(prom[2])
    kept, _ = _kept_i2d(t, data, w, prop_floor_prom_min=4.0)
    assert kept == {1, 2}


def test_qfit_snr_rule_pins_both_sides_of_threshold():
    # Rule 1 removes qfit * S/N < k; the #1017 noise bound removes the other
    # side (qfit above sqrt(qfit_max^2 + (k/S/N)^2)).  Per-frame S/N 4: qfit
    # 0.5 -> 2 (dropped at k = 3), 1.0 -> 4 (kept).  Row 1 sits above the noise
    # bound too, so both rules act on the same table and neither changes the
    # other's rows.
    t, data, w = _i2d_case([(7.0, 100, 150), (7.0, 200, 150)])
    t['qfit'] = [0.5, 1.0]
    kept, _ = _kept_i2d(t, data, w)
    assert kept == {0, 1}
    kept, _ = _kept_i2d(t, data, w, prop_floor_qfit_snr_min=3.0)
    assert kept == {1}
    kept, _ = _kept_i2d(t, data, w, prop_floor_qfit_snr_min=1.5)
    assert kept == {0, 1}
    kept, _ = _kept_i2d(t, data, w, star_prom_min=7.0, qfit_snr_k=5.0)
    assert kept == {0, 1}      # the peak-SB branch is not under the noise bound


def test_prominence_rule_does_not_touch_per_frame_rows():
    t, data, w = _i2d_case([(3.0, 100, 150), (7.0, 200, 150)])
    t['flux_err'] = 40.0 / 6.0           # per-frame S/N 6 clears the floor
    kept, _ = _kept_i2d(t, data, w, **ALL_ON)
    assert kept == {0, 1}


def test_qfit_confident_row_never_dropped():
    t, data, w = _i2d_case([(3.0, 100, 150), (7.0, 200, 150)], qfit=0.1)
    kept, _ = _kept_i2d(t, data, w, **ALL_ON)
    assert kept == {0, 1}


def test_helper_counts_and_all_off():
    prop_only = np.array([True, True, True, False])
    kw = dict(flags_only=np.array([True, False, False, True]),
              prominence=np.array([2.0, 2.0, np.nan, 2.0]),
              qf=np.array([0.5, 0.5, 0.5, 0.5]),
              snr_per_frame=np.array([4.0, 4.0, 4.0, 4.0]))
    drop, c = prop_floor_guard_drop(prop_only, **kw)
    assert not drop.any() and c['dropped'] == 0 and c['prop_only'] == 3
    drop, c = prop_floor_guard_drop(prop_only, prom_min=4.0, **kw)
    assert drop.tolist() == [True, True, False, False]
    drop, c = prop_floor_guard_drop(prop_only, prom_min=4.0, drop_flags=True, **kw)
    assert c['flags'] == 1 and c['prominence'] == 2 and c['prominence_only'] == 1
    drop, c = prop_floor_guard_drop(prop_only, qfit_snr_min=3.0, **kw)
    assert drop.tolist() == [True, True, True, False]      # 0.5 * 4 = 2 < 3


def test_helper_counts_overlap_and_union():
    # flags rule rows 0, 1; prominence rule rows 1, 2; qfit*S/N rows 1, 2, 3
    prop_only = np.ones(5, dtype=bool)
    drop, c = prop_floor_guard_drop(
        prop_only, flags_only=np.array([1, 1, 0, 0, 0], bool),
        prominence=np.array([9.0, 2.0, 2.0, 9.0, 9.0]),
        qf=np.array([0.9, 0.5, 0.5, 0.5, 0.9]),
        snr_per_frame=np.full(5, 4.0), prom_min=4.0, drop_flags=True,
        qfit_snr_min=3.0)
    assert drop.tolist() == [True, True, True, True, False]
    assert (c['flags'], c['prominence'], c['qfit_snr']) == (2, 2, 3)
    assert (c['prominence_only'], c['qfit_snr_only'], c['dropped']) == (1, 1, 4)
    assert c['flags'] + c['prominence'] + c['qfit_snr'] > c['dropped']


def test_helper_thresholds_are_strict_and_infinite_prominence_is_kept():
    prop_only = np.ones(4, dtype=bool)
    kw = dict(flags_only=np.zeros(4, bool), snr_per_frame=np.full(4, 6.0))
    # prominence exactly at the value is kept; only a finite value below drops
    drop, _ = prop_floor_guard_drop(
        prop_only, prominence=np.array([4.0, 3.999, -np.inf, np.inf]),
        qf=np.full(4, 0.9), prom_min=4.0, **kw)
    assert drop.tolist() == [False, True, False, False]
    # qfit * S/N exactly at the value is kept: 0.5 * 6 = 3.0
    drop, _ = prop_floor_guard_drop(
        prop_only, prominence=np.full(4, 9.0),
        qf=np.array([0.5, 0.4999, np.nan, 0.1]), qfit_snr_min=3.0, **kw)
    assert drop.tolist() == [False, True, False, True]


def _run_manual_pipeline_filter_call():
    src = os.path.join(os.path.dirname(__file__), '..', 'cataloging.py')
    tree = ast.parse(open(src).read())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == 'run_manual_pipeline')
    calls = [n for n in ast.walk(fn) if isinstance(n, ast.Call)
             and getattr(n.func, 'id', None) == '_filter_extended_emission']
    assert len(calls) == 1
    return calls[0]


@pytest.mark.parametrize('kwarg, conv', [
    ('prop_floor_prom_min', 'float'), ('prop_floor_drop_flags', 'bool'),
    ('prop_floor_qfit_snr_min', 'float')])
def test_run_manual_pipeline_passes_each_switch_from_its_option(kwarg, conv):
    # the one production call site must read the matching option through mopt;
    # a hard-coded value or a misspelt key would otherwise only fail at run time
    call = _run_manual_pipeline_filter_call()
    kw = {k.arg: k.value for k in call.keywords}
    assert kwarg in kw, f'{kwarg} not passed by run_manual_pipeline'
    v = kw[kwarg]
    assert isinstance(v, ast.Call) and v.func.id == conv, ast.dump(v)
    inner = v.args[0]
    assert isinstance(inner, ast.Call) and inner.func.id == 'mopt', ast.dump(v)
    assert inner.args[0].id == 'opts_phase'
    assert inner.args[1].value == 'manual_ext_' + kwarg
    assert inner.args[1].value in MANUAL_DEFAULTS


def test_defaults_off():
    assert MANUAL_DEFAULTS['manual_ext_prop_floor_prom_min'] == 0.0
    assert MANUAL_DEFAULTS['manual_ext_prop_floor_drop_flags'] is False
    assert MANUAL_DEFAULTS['manual_ext_prop_floor_qfit_snr_min'] == 0.0
