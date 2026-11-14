"""Fit order of the seeds inside one deblended saturated component (#1094).

``build_deblended_source_records`` stamps every seed of a component with the
component's ``sat_area``, so the ``-sat_area`` sort in ``get_saturated_stars``
leaves them in the order ``deblend_blob_zeroframe`` emits: ZF-saturated cores in
raster order of their centroids, then the unsaturated peaks.  A fainter core
that precedes a brighter neighbour absorbs the neighbour's wings.

``order_key`` (the core radius, 0 for a peak) lets ``SATSTAR_DEBLEND_ORDER=1``
fit the larger cores first.  The switch is off by default and leaves ``sat_area``,
the record count and the order of the default path unchanged.
"""
import numpy as np
import pytest
from astropy.io import fits
from scipy import ndimage

from jwst_gc_pipeline.reduction import saturated_star_finding as SSF
from jwst_gc_pipeline.reduction.satstar_deblend import (
    _fof_merge, build_deblended_source_records, deblend_blob_zeroframe)

SHAPE = (120, 120)
FWHM = 1.61
# (x, y, core radius): raster order (top to bottom) is smallest core first
CORES = [(60, 25, 3.0), (60, 60, 5.0), (60, 95, 8.0)]


def _disk(x, y, r):
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2


def _scene():
    """Three ZF-saturated cores of unequal radius inside one DQ-saturated
    column, plus one compact unsaturated star beside the largest."""
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    sat = np.zeros(SHAPE, bool)
    sat[15:105, 45:76] = True
    zf = np.full(SHAPE, 100.0)
    for (x, y, r) in CORES:
        zf += 1e5 * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.0 ** 2))
    zf += 4000.0 * np.exp(-((xx - 60) ** 2 + (yy - 75) ** 2) / (2 * 0.8 ** 2))
    for (x, y, r) in CORES:
        zf[_disk(x, y, r)] = 0.0
    return zf, sat


def _records(**kw):
    zf, sat = _scene()
    sources, n = ndimage.label(sat)
    assert n == 1
    coms = ndimage.center_of_mass(sat, sources, [1])
    sizes = ndimage.sum_labels(sat, sources, [1])
    return build_deblended_source_records(
        sat, sources, coms, sizes, zf, np.zeros(SHAPE), FWHM, **kw)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv('SATSTAR_DEBLEND_ORDER', raising=False)


def test_records_carry_core_radius_and_zero_for_peaks():
    recs = _records()
    keys = sorted(r['order_key'] for r in recs)
    assert len(recs) >= 4, recs
    assert keys[0] == 0.0
    cores = [k for k in keys if k > 0]
    assert len(cores) == 3, keys
    for k, (_, _, r) in zip(cores, sorted(CORES, key=lambda c: c[2])):
        assert abs(k - r) < 1.0, (k, r)


def test_emitted_order_is_raster_not_brightness():
    """The default order inside the component: cores by raster order, so the
    smallest core (top of the frame) comes first.  This pins the behaviour the
    switch changes."""
    recs = _records()
    core_keys = [r['order_key'] for r in recs if r['order_key'] > 0]
    assert core_keys == sorted(core_keys)


def test_switch_off_leaves_the_order_untouched():
    recs = _records()
    before = [dict(r) for r in recs]
    SSF.sort_source_records(recs, deblend_order=False)
    assert recs == before


def test_switch_on_fits_larger_cores_first_then_peaks():
    recs = _records()
    SSF.sort_source_records(recs, deblend_order=True)
    keys = [r['order_key'] for r in recs]
    assert keys == sorted(keys, reverse=True), keys
    assert keys[0] > keys[1] > keys[2] > 0
    assert all(k == 0.0 for k in keys[3:])


def test_peaks_keep_their_emitted_order():
    recs = _records()
    peaks = [r['com'] for r in recs if r['order_key'] == 0.0]
    SSF.sort_source_records(recs, deblend_order=True)
    assert [r['com'] for r in recs if r['order_key'] == 0.0] == peaks


def test_switch_leaves_sat_area_and_record_count_unchanged():
    recs = _records()
    areas = sorted(r['sat_area'] for r in recs)
    n = len(recs)
    SSF.sort_source_records(recs, deblend_order=True)
    assert len(recs) == n
    assert sorted(r['sat_area'] for r in recs) == areas
    assert len({r['sat_area'] for r in recs}) == 1


def test_component_order_still_follows_sat_area():
    """The new key only breaks ties inside one sat_area."""
    recs = [{'sat_area': 10, 'order_key': 9.0}, {'sat_area': 50, 'order_key': 1.0},
            {'sat_area': 50, 'order_key': 4.0}, {'sat_area': 10, 'order_key': 0.0},
            {'sat_area': None, 'order_key': 7.0}]
    SSF.sort_source_records(recs, deblend_order=True)
    assert [(r['sat_area'], r['order_key']) for r in recs] == [
        (50, 4.0), (50, 1.0), (10, 9.0), (10, 0.0), (None, 7.0)]


def test_records_without_order_key_sort_as_before():
    recs = [{'sat_area': 5}, {'sat_area': 9}, {'sat_area': 5, 'order_key': 2.0}]
    SSF.sort_source_records(recs, deblend_order=True)
    assert [r['sat_area'] for r in recs] == [9, 5, 5]
    assert recs[1].get('order_key') == 2.0


def test_info_order_key_is_aligned_with_centres():
    zf, sat = _scene()
    sources, _ = ndimage.label(sat)
    sl = ndimage.find_objects(sources)[0]
    centers, info = deblend_blob_zeroframe(
        zf, np.zeros(SHAPE), sources, 1, sl, FWHM, sat_ceiling=np.inf)
    assert len(info['order_key']) == len(centers)
    for (cy, cx), k in zip(centers, info['order_key']):
        for (x, y, r) in CORES:
            if np.hypot(cx - x, cy - y) < 1.5:
                assert abs(k - r) < 1.0


def test_fof_merge_returns_the_largest_radius_of_each_group():
    centers = [(10.0, 10.0), (10.0, 12.0), (50.0, 50.0), (80.0, 80.0)]
    means, rad = _fof_merge(centers, link=3.0, radii=[2.0, 5.0, 0.0, 4.0],
                            return_radii=True)
    assert len(means) == len(rad) == 3
    assert rad == [5.0, 0.0, 4.0]
    assert _fof_merge(centers, link=3.0, radii=[2.0, 5.0, 0.0, 4.0]) == means
    assert _fof_merge([(1.0, 2.0)], link=3.0, radii=[7.0],
                      return_radii=True) == ([(1.0, 2.0)], [7.0])
    assert _fof_merge([], link=3.0, return_radii=True) == ([], [])


def test_switch_default_off_and_parsed():
    assert SSF.satstar_fit_switches(env={})['deblend_order'] is False
    assert SSF.satstar_fit_switches(
        env={'SATSTAR_DEBLEND_ORDER': '1'})['deblend_order'] is True
    assert SSF.satstar_fit_switches(
        env={'SATSTAR_DEBLEND_ORDER': 'off'})['deblend_order'] is False


def _frame(tmp_path):
    fn = tmp_path / 'f_crf.fits'
    fits.HDUList([fits.PrimaryHDU(header=fits.Header(
        {'INSTRUME': 'NIRCAM', 'FILTER': 'F277W'}))]).writeto(fn)
    return str(fn)


def test_switch_changes_the_cache_signature_on_deblended_frames(tmp_path):
    fn = _frame(tmp_path)
    on = {'SATSTAR_DEBLEND_ORDER': '1'}
    base = SSF.satstar_fit_switch_signature(fn, deblend_with_zeroframe=True,
                                            env={})
    sig = SSF.satstar_fit_switch_signature(fn, deblend_with_zeroframe=True,
                                           env=on)
    assert sig != base
    assert 'do' in sig.split('_')
    # off: key identical to a run that never knew the switch
    assert SSF.satstar_fit_switch_signature(
        fn, deblend_with_zeroframe=True,
        env={'SATSTAR_DEBLEND_ORDER': '0'}) == base
    # no deblend: the switch cannot act, so the key does not move
    assert SSF.satstar_fit_switch_signature(fn, env=on) == \
        SSF.satstar_fit_switch_signature(fn, env={})


def test_switch_is_scoped_by_optin_filters():
    assert 'SATSTAR_DEBLEND_ORDER' in SSF.SATSTAR_OPTIN_ENV
