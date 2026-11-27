"""Continuous distance in the near-saturation filter (env ``NEARSAT_CONTINUOUS_DIST``,
default off).

``_filter_near_saturation`` reads the distance transform of the SATURATED mask at
``np.rint(x_fit), np.rint(y_fit)`` and drops a fit at 1.0 px or less.  A fit 0.001 px
across a half-pixel boundary is measured from a different pixel, so the decision flips:
o105 F480M nrcalong exposure 1 keeps the row at x_fit 1834.500295 (rounds to 1835, 1.414 px
from the saturated pixel (1834, 1632)) and drops the row at x_fit 1834.499118 (rounds to
1834, 1.000 px), although both fits lie 1.26 / 1.27 px from that pixel's centre.  The
refit with the ePSF core moves the fit by that 0.001 px.  The o105 F480M nrcblong exposure 6
neighbour of a handed-off star is dropped the same way (x_fit 1264.542203, 1.000 px at the
rounded pixel, 1.52 px from the saturated pixel (1266, 634)).

``continuous_dist=True`` measures the Euclidean distance from the unrounded fit to the
nearest SATURATED pixel centre and drops at ``<= max_sat_dist_pix``.  These tests pin that

* off (the default), the rule is the rounded-pixel rule, row for row;
* on, the decision does not depend on which side of a half-pixel boundary the fit lies, the
  two F4 rows and the F5 neighbour row are kept, and the rule drops at exactly the distance;
* the per-iteration snapshots, the seeded-companion exemption and the all-restored case
  follow the same distance, and the switch reaches the filter from ``_manual_phot_pass``.
"""
import numpy as np
import pytest
from astropy.table import Table
from photutils.psf import SourceGrouper
from scipy import ndimage

from jwst_gc_pipeline.photometry import cataloging as C
from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as L
from jwst_gc_pipeline.photometry.tests.test_gate_reject_handoff import (
    FWHM, SHAPE as _SMALL, _gaussian_grid_psf, _options)

SAT = L.dqflags.pixel['SATURATED']
SWITCH = 'NEARSAT_CONTINUOUS_DIST'

# o105 F480M, nrcalong exposure 1 (F4): the saturated pixels around the star
F4_SAT = [(1830, 1632), (1831, 1630), (1831, 1631), (1831, 1632), (1832, 1629),
          (1832, 1630), (1832, 1631), (1832, 1632), (1833, 1629), (1833, 1630),
          (1833, 1631), (1833, 1632), (1834, 1629), (1834, 1630), (1834, 1631),
          (1834, 1632), (1835, 1629), (1835, 1630), (1835, 1631), (1836, 1629),
          (1836, 1630)]
F4_P = (1834.500295, 1633.160070)      # production fit: kept
F4_C = (1834.499118, 1633.171163)      # fit with the ePSF core: dropped
# o105 F480M, nrcblong exposure 6 (F5): the satstar core next to the neighbour
F5_SAT = [(1266, 631), (1266, 632), (1266, 633), (1266, 634), (1266, 635), (1267, 630),
          (1267, 631), (1267, 632), (1267, 633), (1267, 634), (1267, 635), (1267, 636),
          (1268, 630), (1268, 631), (1268, 632), (1268, 633), (1268, 634), (1268, 635),
          (1268, 636), (1269, 631), (1269, 632), (1269, 633), (1269, 634), (1269, 635),
          (1269, 636), (1270, 632), (1270, 633), (1270, 634), (1270, 635), (1270, 636)]
F5_P = (1264.4819, 634.4405)           # production fit: kept
F5_C = (1264.542203, 634.433635)       # after the satstar row is rejected: dropped


class _Snap:
    def __init__(self, results):
        self.results = results
        self.init_params = None


class _FakePhot:
    def __init__(self, results, with_snapshot=False):
        self.results = results
        if with_snapshot:
            self.fit_results = [_Snap(Table(results, copy=True))]


def _dq(shape, pixels):
    dq = np.zeros(shape, dtype=np.uint32)
    for x, y in pixels:
        dq[y, x] |= SAT
    return dq


def _fits(xy, flux=1e4, qfit=0.05):
    res = Table()
    res['id'] = np.arange(1, len(xy) + 1)
    res['x_fit'] = [p[0] for p in xy]
    res['y_fit'] = [p[1] for p in xy]
    res['flux_fit'] = np.full(len(xy), float(flux))
    res['qfit'] = np.full(len(xy), qfit)
    return res


def _kept_ids(res, dq, continuous=None, thr=1.0, **kw):
    phot = _FakePhot(res)
    kwargs = dict(kw)
    if continuous is not None:
        kwargs['continuous_dist'] = continuous
    L._filter_near_saturation(phot, dq, max_sat_dist_pix=thr, label='t', **kwargs)
    return [int(i) for i in phot.results['id']]


def _rounded_rule_drops(x, y, dq, thr=1.0):
    """The rounded-pixel rule written out independently of the code under test."""
    edt = ndimage.distance_transform_edt(~((dq & SAT) > 0))
    x, y = np.asarray(x, float), np.asarray(y, float)
    with np.errstate(invalid='ignore'):
        ix, iy = np.rint(x), np.rint(y)
    ny, nx = dq.shape
    ok = np.isfinite(x) & np.isfinite(y) & (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    dist = np.full(len(x), np.inf)
    dist[ok] = edt[iy[ok].astype(int), ix[ok].astype(int)]
    return dist <= thr


def test_default_is_the_rounded_pixel_rule_row_for_row():
    rng = np.random.default_rng(11)
    shape = (60, 70)
    dq = np.zeros(shape, dtype=np.uint32)
    for cy, cx, r in ((15, 20, 2), (40, 50, 3), (30, 8, 1), (5, 60, 2)):
        yy, xx = np.ogrid[:shape[0], :shape[1]]
        dq[(yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2] |= SAT
    x = rng.uniform(-3, 73, 600)
    y = rng.uniform(-3, 63, 600)
    # exact half-pixels, where np.rint rounds to even, NaN and out-of-frame rows
    x[:6] = [20.5, 21.5, 50.5, 49.5, np.nan, 20.0]
    y[:6] = [15.5, 16.5, 40.5, 41.5, 15.0, np.nan]
    res = _fits(list(zip(x, y)))
    want = [int(i) for i in res['id'][~_rounded_rule_drops(x, y, dq)]]
    assert _kept_ids(res, dq) == want
    assert _kept_ids(res, dq, continuous=False) == want
    # the snapshot loop of the default rule is the same rule
    phot = _FakePhot(res, with_snapshot=True)
    L._filter_near_saturation(phot, dq, max_sat_dist_pix=1.0, label='t')
    assert [int(i) for i in phot.fit_results[0].results['id']] == want


def test_f4_rows_the_rounded_rule_splits_and_the_continuous_rule_keeps():
    dq = _dq((1700, 1900), F4_SAT)
    res = _fits([F4_P, F4_C])
    assert _kept_ids(res, dq) == [1]                       # production: P kept, C dropped
    assert _kept_ids(res, dq, continuous=True) == [1, 2]   # both are 1.26 / 1.27 px away
    tree = L._mask_pixel_tree((dq & SAT) > 0)
    x, y = res['x_fit'].data, res['y_fit'].data
    d = L._continuous_sat_distance(x, y, np.ones(2, bool), tree)
    assert d == pytest.approx([1.2634, 1.2731], abs=1e-4)
    # the rounded distances the production rule used: 1.414 px (kept) and 1.000 px (dropped)
    edt = ndimage.distance_transform_edt(~((dq & SAT) > 0))
    assert edt[1633, 1835] == pytest.approx(2 ** 0.5)
    assert edt[1633, 1834] == 1.0
    assert np.rint(F4_P[0]) == 1835 and np.rint(F4_C[0]) == 1834


def test_f5_neighbour_row():
    dq = _dq((2048, 2048), F5_SAT)
    res = _fits([F5_P, F5_C])
    assert _kept_ids(res, dq) == [1]                       # production: P kept, C dropped
    assert _kept_ids(res, dq, continuous=True) == [1, 2]
    tree = L._mask_pixel_tree((dq & SAT) > 0)
    d = L._continuous_sat_distance(res['x_fit'].data, res['y_fit'].data, np.ones(2, bool), tree)
    assert d == pytest.approx([1.5807, 1.5209], abs=1e-4)


def test_threshold_is_the_distance_to_the_pixel_centre():
    dq = _dq((100, 100), [(50, 50)])
    xy = [(50.0, 50.0),        # on the pixel: dropped by both rules
          (50.0, 50.99),       # 0.99 px: dropped
          (50.0, 51.0),        # exactly 1.0: dropped (<=), as for the rounded pixel next to it
          (50.0, 51.01),       # 1.01 px on axis: kept by the continuous rule, rounds to 51 -> dropped
          (50.0, 51.4),        # 1.4 px on axis: kept by the continuous rule, rounds to 51 -> dropped
          (50.0, 51.6),        # 1.6 px on axis: rounds to 52, kept by both
          (50.6, 50.6)]        # 0.85 px on the diagonal: rounds to (51, 51), 1.414 -> kept; dropped by the disc
    res = _fits(xy)
    assert _kept_ids(res, dq) == [6, 7]
    assert _kept_ids(res, dq, continuous=True) == [4, 5, 6]
    # the threshold is the distance itself
    assert _kept_ids(res, dq, continuous=True, thr=1.5) == [6]
    assert _kept_ids(res, dq, continuous=True, thr=0.5) == [2, 3, 4, 5, 6, 7]


@pytest.mark.parametrize('y', [51.2, 52.7])
def test_decision_does_not_flip_at_a_half_pixel_boundary(y):
    """Fits 0.0025 px apart across x = 50.5: the rounded rule changes its answer there at
    y = 51.2 (rounds to (50, 51), 1.0 px, then to (51, 51), 1.414 px); the continuous rule
    does not."""
    dq = _dq((100, 100), [(50, 50)])
    xs = 50.5 + np.arange(-8, 9) * 0.0025
    res = _fits([(x, y) for x in xs])
    rounded = ~_rounded_rule_drops(xs, np.full(len(xs), y), dq)
    assert _kept_ids(res, dq) == [int(i) for i in res['id'][rounded]]
    cont = _kept_ids(res, dq, continuous=True)
    assert cont == list(res['id'])                        # every one is > 1.0 px from (50, 50)
    if y == 51.2:
        assert len(set(rounded)) == 2                      # the rounded rule does flip


def test_snapshots_follow_the_continuous_rule():
    dq = _dq((1700, 1900), F4_SAT)
    res = _fits([F4_C, (1832.0, 1631.0)])                  # second row sits on the saturated core
    phot = _FakePhot(res, with_snapshot=True)
    L._filter_near_saturation(phot, dq, max_sat_dist_pix=1.0, label='t', continuous_dist=True)
    assert [int(i) for i in phot.results['id']] == [1]
    assert [int(i) for i in phot.fit_results[0].results['id']] == [1]
    phot = _FakePhot(res, with_snapshot=True)
    L._filter_near_saturation(phot, dq, max_sat_dist_pix=1.0, label='t')
    assert len(phot.results) == 0 and len(phot.fit_results[0].results) == 0


def test_nan_and_off_frame_rows_are_never_dropped():
    dq = _dq((40, 40), [(0, 0), (39, 39)])
    xy = [(np.nan, 5.0), (5.0, np.nan), (-0.6, 0.0), (39.6, 39.0), (0.3, 0.2)]
    res = _fits(xy)
    for cont in (False, True):
        assert _kept_ids(res, dq, continuous=cont) == [1, 2, 3, 4]


def _companion_scene():
    """Restored footprint pixel R=(30, 30); lost SATURATED pixel L=(10, 10) far away."""
    dq = _dq((60, 60), [(30, 30), (10, 10)])
    restored = np.zeros((60, 60), bool)
    restored[30, 30] = True
    return dq, restored


def test_seeded_companion_next_to_a_restored_pixel_is_kept_by_both_rules():
    dq, restored = _companion_scene()
    res = _fits([(30.3, 30.5)])                            # 0.58 px from R: within the drop distance
    seed = np.array([[30.3, 30.5]])
    kw = dict(companion_seed_xy=seed, companion_seed_radius_pix=0.5,
              companion_restored_mask=restored)
    for cont in (False, True):
        assert _kept_ids(res, dq, continuous=cont, **kw) == [1]
    # without the exemption the fit is dropped
    assert _kept_ids(res, dq, continuous=True) == []


def test_seeded_companion_uses_the_continuous_distance_to_the_lost_pixel():
    dq, restored = _companion_scene()
    res = _fits([(10.6, 10.6)])                            # 0.85 px from the lost pixel L
    seed = np.array([[10.6, 10.6]])
    kw = dict(companion_seed_xy=seed, companion_seed_radius_pix=0.5,
              companion_restored_mask=restored)
    # rounded rule: rounds to (11, 11), 1.414 px, never dropped; continuous: inside the disc,
    # and L is still lost, so the exemption does not apply
    assert _kept_ids(res, dq, continuous=False, **kw) == [1]
    assert _kept_ids(res, dq, continuous=True, **kw) == []


def test_all_saturated_pixels_restored():
    dq = _dq((60, 60), [(30, 30), (31, 30)])
    restored = (dq & SAT) > 0
    res = _fits([(30.4, 30.4)])
    kw = dict(companion_seed_xy=np.array([[30.4, 30.4]]), companion_seed_radius_pix=0.5,
              companion_restored_mask=restored)
    assert _kept_ids(res, dq, continuous=True, **kw) == [1]
    assert _kept_ids(res, dq, continuous=True) == []


def test_helpers_on_an_empty_mask_and_empty_rows():
    assert L._mask_pixel_tree(np.zeros((5, 5), bool)) is None
    d = L._continuous_sat_distance(np.array([1.0, 2.0]), np.array([1.0, 2.0]),
                                   np.ones(2, bool), None)
    assert np.all(np.isinf(d))
    tree = L._mask_pixel_tree(_dq((5, 5), [(3, 1)]) > 0)
    d = L._continuous_sat_distance(np.array([3.0, 0.0, 9.0]), np.array([1.0, 1.0, 9.0]),
                                   np.array([True, True, False]), tree)
    assert d[0] == 0.0 and d[1] == 3.0 and np.isinf(d[2])


# ---------------------------------------------------------------------------
# the switch
# ---------------------------------------------------------------------------

def test_switch_is_off_unless_set(monkeypatch):
    monkeypatch.delenv(SWITCH, raising=False)
    assert C._near_sat_continuous_enabled(False) is False
    for val, want in (('1', True), ('true', True), ('ON', True), ('0', False),
                      ('off', False), ('', False), ('  ', False)):
        monkeypatch.setenv(SWITCH, val)
        assert C._near_sat_continuous_enabled(False) is want, val


def test_switch_is_nircam_only_and_rejects_typos(monkeypatch, capsys):
    monkeypatch.setenv(SWITCH, '1')
    assert C._near_sat_continuous_enabled(True) is False   # MIRI: 1.5 FWHM threshold, unchanged
    assert C._near_sat_continuous_enabled(False) is True
    assert 'NEARSAT_CONTINUOUS_DIST' in capsys.readouterr().out
    monkeypatch.setenv(SWITCH, 'maybe')
    with pytest.raises(ValueError, match=SWITCH):
        C._near_sat_continuous_enabled(False)


def _synthetic_pass(img, dq, *, continuous):
    err = np.ones(_SMALL)
    seed = Table()
    seed['x_init'] = [40.3]
    seed['y_init'] = [44.2]
    seed['flux_init'] = [2.0e4]
    res, _, _ = C._manual_phot_pass(
        data=img, mask=(dq & SAT) != 0, err=err, bad=np.zeros(_SMALL, bool),
        dao_psf_model=_gaussian_grid_psf(), init_params=seed, aperture_radius_pix=2 * FWHM,
        localbkg_inner=6, localbkg_outer=10, grouper=SourceGrouper(2 * FWHM),
        options=_options(), dq=dq, satstar_model_subtracted=np.zeros(_SMALL),
        label='t', near_sat_dist_pix=1.0, near_sat_continuous=continuous)
    return res


def test_manual_phot_pass_passes_the_switch_to_the_filter(monkeypatch):
    """A star 1.24 px from a saturated pixel, whose fit rounds next to it."""
    yy, xx = np.mgrid[:_SMALL[0], :_SMALL[1]]
    sig = FWHM / 2.3548
    img = 10.0 + 3000.0 * np.exp(-((xx - 40.3) ** 2 + (yy - 44.2) ** 2) / (2 * sig ** 2))
    img += np.random.default_rng(5).normal(0, 1.0, _SMALL)
    dq = _dq(_SMALL, [(40, 43)])
    seen = []
    orig = C._filter_near_saturation

    def spy(phot, dq_, **kw):
        seen.append(kw.get('continuous_dist'))
        return orig(phot, dq_, **kw)

    monkeypatch.setattr(C, '_filter_near_saturation', spy)
    res = _synthetic_pass(img, dq, continuous=False)
    assert seen == [False]
    assert res is None or len(res) == 0                    # rounds to (40, 44): 1.0 px, dropped
    seen.clear()
    res = _synthetic_pass(img, dq, continuous=True)
    assert seen == [True]
    assert res is not None and len(res) == 1
    assert res['x_fit'][0] == pytest.approx(40.3, abs=0.1)
    assert np.hypot(res['x_fit'][0] - 40, res['y_fit'][0] - 43) > 1.0
