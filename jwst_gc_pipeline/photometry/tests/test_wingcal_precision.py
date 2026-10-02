"""Wing self-calibration precision gate and C(0) = 1 anchor (#1041).

Measured across /orange: pooled C(r) buckets of 2-30 stars reached 12-20 at
large r_mask and shifted catalog stars by up to 3 mag; per-frame buckets on
ngc6334 F335M/F356W applied ratios of 16-27.  Separately, a star whose fit
masked one pixel (r_mask = 0.564) was corrected with C(3 px), which on wd2
moved those rows 0.03-0.07 mag away from dolphot.
"""
import builtins
import os

import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry import merge_catalogs as MC
from jwst_gc_pipeline.photometry.wingcal import (
    MEDIAN_SE_FACTOR, bucket_se, interp_wingcal_ratio, passes_se_gate,
    pool_bucket, relative_scatter_floor)

R1PX = float(np.sqrt(1 / np.pi))     # r_mask of a one-pixel core


# --- interpolation ----------------------------------------------------------

def test_interp_is_anchored_at_one_below_the_smallest_bucket():
    rs, vs = [3, 5], [1.30, 1.50]
    got = interp_wingcal_ratio([R1PX, 1.5, 3.0, 4.0, 5.0, 40.0, np.nan], rs, vs)
    want = [1 + 0.30 * R1PX / 3, 1.15, 1.30, 1.40, 1.50, 1.50, 1.0]
    np.testing.assert_allclose(got, want)


def test_interp_ignores_non_positive_buckets():
    np.testing.assert_allclose(
        interp_wingcal_ratio([10.0], [3, 10], [1.05, -0.01]), [1.05])
    np.testing.assert_allclose(
        interp_wingcal_ratio([3.0], [3], [0.0]), [1.0])


def test_interp_without_buckets_is_one():
    np.testing.assert_array_equal(interp_wingcal_ratio([3.0, 9.0], [], []),
                                  [1.0, 1.0])


# --- standard error and gate ------------------------------------------------

def test_bucket_se_is_the_se_of_a_median():
    np.testing.assert_allclose(bucket_se([0.2, 0.2], [4, 16]),
                               [MEDIAN_SE_FACTOR * 0.1, MEDIAN_SE_FACTOR * 0.05])
    assert np.isnan(bucket_se([np.nan], [10])[0])
    assert np.isnan(bucket_se([0.1], [0])[0])


def test_bucket_se_floors_the_scatter_relative_to_the_ratio():
    # ngc6334 F444W r=18: 3 stars, C = 3.22, madstd 0.017 (se 0.012).
    raw = bucket_se([0.017], [3])[0]
    floored = bucket_se([0.017], [3], ratio=[3.22], rel_floor=0.13)[0]
    assert raw < 0.05 < floored
    assert floored == pytest.approx(MEDIAN_SE_FACTOR * 0.13 * 3.22 / np.sqrt(3))
    # a well-measured bucket above the floor is unchanged
    assert bucket_se([0.05], [20], ratio=[1.05], rel_floor=0.04)[0] == \
        pytest.approx(bucket_se([0.05], [20])[0])


def test_scatter_floor_uses_the_smallest_bucket_with_enough_stars():
    # r=3 from 2 stars (madstd ~0) does not count; r=4 sets the floor.
    rs, v, s, n = [3, 4, 4, 16], [1.0, 1.0, 1.1, 10.0], \
        [1e-4, 0.04, 0.066, 0.01], [2, 10, 12, 6]
    assert relative_scatter_floor(rs, v, s, n) == pytest.approx(0.05)
    assert relative_scatter_floor([3], [1.0], [np.nan], [10]) == 0.0


def test_gate_rejects_nan_and_noisy_buckets_and_can_be_disabled(monkeypatch):
    se = np.array([0.01, 0.05, 0.06, np.nan])
    assert passes_se_gate(se).tolist() == [True, True, False, False]
    monkeypatch.setenv('SATSTAR_WINGCAL_MAX_SE', '0.1')
    assert passes_se_gate(se).tolist() == [True, True, True, False]
    monkeypatch.setenv('SATSTAR_WINGCAL_MAX_SE', '0')
    assert passes_se_gate(se).tolist() == [True, True, True, True]


def test_gate_rejects_a_near_zero_ratio_with_a_small_se():
    # Masked fits that are pure noise: the pooled median sits near zero with
    # an SE that passes the 0.05 gate on its own.
    ratio, se = pool_bucket([0.02, -0.03, 0.01], [30] * 3, [0.01] * 3)
    assert se < 0.05
    assert passes_se_gate([se]).tolist() == [True]
    assert passes_se_gate([se], ratio=[ratio]).tolist() == [False]
    assert passes_se_gate([0.01, 0.01], ratio=[0.49, 0.5]).tolist() == \
        [False, True]


# --- pooling ----------------------------------------------------------------

def test_pool_equal_frames_is_the_plain_mean():
    ratio, se = pool_bucket([1.05, 1.07], [4, 4], [0.02, 0.02])
    assert ratio == pytest.approx(1.06)
    stat = MEDIAN_SE_FACTOR * 0.02 / np.sqrt(8)
    chi2 = 2 * 0.01 ** 2 / (MEDIAN_SE_FACTOR * 0.02) ** 2 * 4
    assert se == pytest.approx(stat * np.sqrt(chi2))


def test_pool_downweights_a_noisy_outlier_frame():
    # The shape of the survey's large-r buckets: two well-measured frames
    # near 1.1 and one 2-star frame at 15.  The n-weighted mean was 2.27.
    v, n, s = [1.10, 1.12, 15.0], [10, 12, 2], [0.05, 0.06, 6.0]
    assert np.average(v, weights=n) > 2.2
    ratio, _ = pool_bucket(v, n, s)
    good, _ = pool_bucket(v[:2], n[:2], s[:2])
    assert ratio == pytest.approx(good, abs=0.005)


def test_pool_floors_a_near_zero_madstd():
    # Two stars that happen to agree must not dominate the bucket.
    ratio, _ = pool_bucket([1.50, 1.10, 1.12], [2, 10, 10], [1e-6, 0.05, 0.05])
    assert ratio < 1.20


def test_pool_without_madstd_has_no_se():
    ratio, se = pool_bucket([1.1, 1.3], [3, 1], [np.nan, np.nan])
    assert ratio == pytest.approx(1.15)
    assert np.isnan(se)


# --- pooled table: build, load, apply --------------------------------------

def _calib(pdir, name, rows, mtime=None):
    fn = pdir / f'{name}_m7_wingcal_calibrators.fits'
    Table(rows=rows, names=['rmask_px', 'ratio_median', 'n_stars',
                            'ratio_madstd']).write(fn, overwrite=True)
    if mtime is not None:
        os.utime(fn, (mtime, mtime))
    return fn


@pytest.fixture
def tree(tmp_path):
    pdir = tmp_path / 'F410M' / 'pipeline'
    pdir.mkdir(parents=True)
    (tmp_path / 'catalogs').mkdir()
    # r=3: well measured.  r=16: two frames of 2-3 stars at 12-15.
    _calib(pdir, 'a', [(3, 1.05, 20, 0.05), (16, 15.0, 2, 4.0)], mtime=1_000)
    _calib(pdir, 'b', [(3, 1.07, 20, 0.05), (16, 12.0, 3, 5.0)], mtime=1_000)
    return tmp_path, pdir


def test_pooled_table_carries_ratio_se(tree):
    base, _ = tree
    pooled = MC.build_pooled_wingcal('f410m', basepath=str(base), phase='m7')
    assert 'ratio_se' in pooled.colnames
    se = dict(zip(pooled['rmask_px'], pooled['ratio_se']))
    assert se[3] < 0.05
    assert se[16] > 0.05


def test_apply_pooled_skips_gated_buckets_and_anchors_small_cores(tree):
    base, _ = tree
    cat = Table({'flux_fit': [1000.0] * 4, 'flux_err': [10.0] * 4,
                 'wingcal_ratio': [1.0] * 4,
                 'wingcal_rmask': [R1PX, 3.0, 16.0, np.nan]})
    out = MC.apply_pooled_wingcal(cat, 'f410m', basepath=str(base), phase='m7')
    r3 = 1.06       # the scatter floor shifts it by 1e-4
    want = [1 + (r3 - 1) * R1PX / 3, r3, r3, 1.0]
    np.testing.assert_allclose(out['wingcal_ratio'], want, rtol=2e-4)
    np.testing.assert_allclose(out['flux_fit'] * out['wingcal_ratio'], 1000.0)
    assert out['wingcal_pooled'].tolist() == [True, True, True, False]


def test_apply_pooled_with_gate_disabled_uses_every_bucket(tree, monkeypatch):
    base, _ = tree
    monkeypatch.setenv('SATSTAR_WINGCAL_MAX_SE', '0')
    cat = Table({'flux_fit': [1000.0], 'flux_err': [10.0],
                 'wingcal_ratio': [1.0], 'wingcal_rmask': [16.0]})
    out = MC.apply_pooled_wingcal(cat, 'f410m', basepath=str(base), phase='m7')
    assert out['wingcal_ratio'][0] > 10


def test_pooled_single_frame_bucket_with_tight_scatter_is_gated(tmp_path):
    # ngc6334 F444W: r=3 in many frames at a fractional scatter of 0.13;
    # r=18 in one frame, 3 stars, C = 3.22, madstd 0.017.
    pdir = tmp_path / 'F444W' / 'pipeline'
    pdir.mkdir(parents=True)
    (tmp_path / 'catalogs').mkdir()
    for i in range(6):
        rows = [(3, 1.05, 8, 0.14)]
        if i == 0:
            rows.append((18, 3.22, 3, 0.017))
        _calib(pdir, f'f{i}', rows, mtime=1_000)
    pooled = MC.build_pooled_wingcal('f444w', basepath=str(tmp_path),
                                     phase='m7')
    se = dict(zip(pooled['rmask_px'], pooled['ratio_se']))
    assert se[3] < 0.05
    assert se[18] > 0.05
    cat = Table({'flux_fit': [1000.0], 'flux_err': [10.0],
                 'wingcal_ratio': [1.0], 'wingcal_rmask': [18.0]})
    out = MC.apply_pooled_wingcal(cat, 'f444w', basepath=str(tmp_path),
                                  phase='m7')
    assert out['wingcal_ratio'][0] == pytest.approx(1.05)


def test_pooled_table_without_ratio_se_is_rebuilt(tree):
    base, _ = tree
    pooled = MC.build_pooled_wingcal('f410m', basepath=str(base), phase='m7')
    fn = MC.pooled_wingcal_path(str(base), 'f410m', 'm7')
    pooled.remove_column('ratio_se')
    pooled.write(fn, format='ascii.ecsv', overwrite=True)
    os.utime(fn, (2_000, 2_000))       # newer than the inputs, same WCALSEL
    again = MC.load_pooled_wingcal('f410m', str(base), phase='m7')
    assert 'ratio_se' in again.colnames


# --- per-frame application --------------------------------------------------

S = pytest.importorskip('jwst_gc_pipeline.reduction.saturated_star_finding')


def _apply(monkeypatch, cal, rmask):
    monkeypatch.setattr(S, '_wing_selfcal', lambda *a, **k: cal)
    tab = Table({'flux_fit': [1000.0] * len(rmask),
                 'flux_err': [10.0] * len(rmask),
                 'wingcal_rmask': rmask})
    return S.apply_wing_selfcal(tab, None, None, None, None,
                                severity_floor=1000.0)


def test_per_frame_skips_noisy_buckets_and_anchors(monkeypatch):
    # r=16: 8 stars, madstd 5 -> se 2.2.  r=3: se 0.014.
    cal = {3: (1.05, 20, 0.05), 16: (15.0, 8, 5.0)}
    out = _apply(monkeypatch, cal, [R1PX, 3.0, 16.0, np.nan])
    want = [1 + 0.05 * R1PX / 3, 1.05, 1.05, 1.0]
    np.testing.assert_allclose(out['wingcal_ratio'], want)
    np.testing.assert_allclose(out['flux_fit'], 1000.0 / np.array(want))
    np.testing.assert_allclose(out['flux_fit_raw'], 1000.0)
    # every bucket is still persisted for the pool, which gates on its own
    assert sorted(builtins.satstar_wingcal_measurements['rmask_px']) == [3, 16]


def test_per_frame_small_bucket_with_tight_scatter_is_gated(monkeypatch):
    cal = {3: (1.05, 20, 0.14), 18: (3.22, 3, 0.017)}
    out = _apply(monkeypatch, cal, [18.0])
    assert out['wingcal_ratio'][0] == pytest.approx(1.05)


def test_per_frame_skips_a_near_zero_bucket(monkeypatch):
    cal = {3: (1.05, 20, 0.05), 10: (0.01, 30, 0.01)}
    out = _apply(monkeypatch, cal, [10.0])
    assert out['wingcal_ratio'][0] == pytest.approx(1.05)


def test_per_frame_with_no_passing_bucket_is_left_to_the_pool(monkeypatch):
    cal = {3: (1.40, 8, 0.5), 16: (15.0, 8, 5.0)}
    out = _apply(monkeypatch, cal, [3.0, 16.0])
    assert out['wingcal_ratio'].tolist() == [1.0, 1.0]
    assert out['flux_fit'].tolist() == [1000.0, 1000.0]


def test_per_frame_gate_can_be_disabled(monkeypatch):
    monkeypatch.setenv('SATSTAR_WINGCAL_MAX_SE', '0')
    cal = {3: (1.05, 20, 0.05), 16: (15.0, 8, 5.0)}
    out = _apply(monkeypatch, cal, [16.0])
    assert out['wingcal_ratio'][0] == pytest.approx(15.0)


def test_masked_fits_at_or_below_zero_stay_in_the_bucket(monkeypatch):
    """A non-positive masked fit is a noisy measurement of faint wings.
    Dropping it kept only the upper tail; with these nine stars the old
    median was 0.15 from 7 stars, the unbiased one is 0.12 from 9."""
    from photutils.psf import CircularGaussianPRF

    rng = np.random.default_rng(1)
    ny = nx = 420
    yy, xx = np.mgrid[0:ny, 0:nx]
    data = rng.normal(0, 1.0, (ny, nx))
    for xc in (130, 210, 290):
        for yc in (130, 210, 290):
            data += CircularGaussianPRF(fwhm=2.0, flux=2000.0, x_0=xc,
                                        y_0=yc)(xx, yy)
    masked = iter([-50.0, -20.0, 30.0, 80.0, 120.0, 150.0, 900.0, 1000.0,
                   1100.0])

    class _Phot:
        def __init__(self, *a, **k):
            pass

        def __call__(self, data, error=None, mask=None, init_params=None):
            x = int(round(float(init_params['x_init'][0])))
            y = int(round(float(init_params['y_init'][0])))
            core_masked = mask is not None and bool(mask[y, x])
            return Table({'flux_fit': [next(masked) if core_masked else 1000.0]})

    monkeypatch.setattr(S, 'PSFPhotometry', _Phot)
    out = S._wing_selfcal(data, np.ones_like(data), None,
                          CircularGaussianPRF(fwhm=2.0), radii=[3],
                          fwhm_pix=2.0, severity_floor=1000.0)
    ratio, n, _ = out[3]
    assert n == 9
    assert ratio == pytest.approx(0.12)
