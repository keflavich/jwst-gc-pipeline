"""Guard on the rows only the propagated S/N floor admits (issue #1025).

Rows with per-frame S/N below the floor and merged-flux S/N above it are the
propagated-only rows.  The guard drops three subsets of them (flags-only
route, finite low prominence, low qfit x S/N), each behind its own switch.
"""
import numpy as np
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


def test_guard_inert_without_propagated_floor():
    t = _mk([2.5, 5.0, 10.0])
    t['flux_err'] = [10.0, 5.0, 4.0]
    off = _kept(t, on=False)
    assert _kept(t, on=False, **ALL_ON) == off
    # a catalogue without flux_err_prop has no propagated-only rows either
    t2 = _mk(None)
    assert _kept(t2, **ALL_ON) == _kept(t2)


def test_guard_reproduces_per_frame_floor_for_all_rows_it_removes():
    t = _mk([2.5, 5.0, 10.0, 2.5, 2.5, 2.5], n=6)
    t['flux_err'] = [10.0, 5.0, 10.0, 4.0, 10.0, 10.0]
    t['qfit'] = [0.5, 0.5, 0.5, 0.5, 0.1, 0.5]
    per_frame = _kept(t, on=False)
    guarded = _kept(t, **ALL_ON)
    assert per_frame <= guarded
    assert guarded <= _kept(t)


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


def test_defaults_off():
    assert MANUAL_DEFAULTS['manual_ext_prop_floor_prom_min'] == 0.0
    assert MANUAL_DEFAULTS['manual_ext_prop_floor_drop_flags'] is False
    assert MANUAL_DEFAULTS['manual_ext_prop_floor_qfit_snr_min'] == 0.0
