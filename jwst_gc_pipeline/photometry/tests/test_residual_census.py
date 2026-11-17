"""Unit tests for the residual-image unsubtracted-fraction estimator (#972)."""
import numpy as np
import pytest
from astropy.table import MaskedColumn, Table

from jwst_gc_pipeline.photometry.residual_census import (
    DEFAULT_BOX_HALF, bin_edges, box_peak, column_as_float, format_table,
    residual_fraction, unsubtracted_fraction_by_bin)


def _gauss(shape, x0, y0, amp, sig=1.5):
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    return amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sig ** 2))


def _synthetic_frame():
    """Four stars: clean, half-left, untouched and over-subtracted."""
    shape = (40, 80)
    xs = np.array([10., 30., 50., 70.])
    ys = np.full(4, 20.)
    amps = np.array([100., 100., 100., 100.])
    left = np.array([0.02, 0.6, 1.0, -0.4])
    pre = sum(_gauss(shape, x, y, a) for x, y, a in zip(xs, ys, amps))
    resid = sum(_gauss(shape, x, y, a * f)
                for x, y, a, f in zip(xs, ys, amps, left))
    return pre, resid, xs, ys, left


def test_residual_fraction_recovers_planted_fraction():
    pre, resid, xs, ys, left = _synthetic_frame()
    frac = residual_fraction(pre, resid, xs, ys)
    np.testing.assert_allclose(frac, left, atol=0.02)


def test_unsubtracted_fraction_by_bin_counts():
    mag = np.array([12.1, 12.2, 12.3, 12.4, 13.1, 13.2, np.nan, 14.0])
    frac = np.array([0.0, 0.6, 0.9, 0.1, 0.0, -0.5, 0.9, np.nan])
    rows = unsubtracted_fraction_by_bin(mag, frac, np.array([12.0, 12.5, 13.5]))
    assert rows[0]['n'] == 4 and rows[0]['n_unsub'] == 2
    assert rows[0]['frac_unsub'] == 0.5 and rows[0]['n_over'] == 0
    assert rows[1]['n'] == 2 and rows[1]['n_unsub'] == 0
    assert rows[1]['n_over'] == 1


def test_empty_bin_is_nan_and_formats():
    rows = unsubtracted_fraction_by_bin(np.array([12.1]), np.array([0.0]),
                                        np.array([14.0, 14.5]))
    assert rows[0]['n'] == 0 and np.isnan(rows[0]['frac_unsub'])
    assert '14.0' in format_table(rows)


def test_box_peak_edges_and_nan():
    img = np.zeros((10, 10))
    img[0, 0] = 5.
    img[5, 5] = np.nan
    out = box_peak(img, [0, 5, -3, np.nan], [0, 5, 4, 2])
    assert out[0] == 5.
    assert out[1] == 0.0  # the NaN pixel is skipped, finite neighbours remain
    assert np.isnan(out[2]) and np.isnan(out[3])


def test_signed_extreme_keeps_negative_core():
    img = np.zeros((9, 9))
    img[4, 4] = -3.
    img[4, 5] = 1.
    assert box_peak(img, [4], [4])[0] == 1.
    assert box_peak(img, [4], [4], signed_extreme=True)[0] == -3.


def test_nonpositive_pre_peak_gives_nan():
    pre = np.zeros((10, 10))
    resid = np.ones((10, 10))
    assert np.isnan(residual_fraction(pre, resid, [5], [5])[0])


def test_bin_edges_keep_the_partial_last_bin():
    # The default F480M range 11.5-14.7 in 0.5 mag steps must not lose 14.5-14.7.
    np.testing.assert_allclose(bin_edges(11.5, 14.7, 0.5),
                               [11.5, 12., 12.5, 13., 13.5, 14., 14.5, 14.7])
    np.testing.assert_allclose(bin_edges(11.5, 14.5, 0.5),
                               [11.5, 12., 12.5, 13., 13.5, 14., 14.5])


def test_masked_column_becomes_nan_not_zero(tmp_path):
    # A masked x keeps 0.0 under the mask; read through, it would be measured
    # at image column 0.  It must instead be unmeasurable.
    col = MaskedColumn([10., 20., 30.], mask=[False, True, False])
    out = column_as_float(col)
    assert out[0] == 10. and out[2] == 30. and np.isnan(out[1])
    path = tmp_path / 'stars.ecsv'
    Table({'x': col}).write(path)
    back = column_as_float(Table.read(path)['x'])
    assert np.isnan(back[1]) and back[0] == 10.
    # a plain column passes through unchanged
    np.testing.assert_array_equal(column_as_float(Table({'x': [1, 2]})['x']), [1., 2.])
    # and the NaN position gives an unmeasurable star, not a column-0 reading
    pre = np.zeros((20, 40))
    pre[10, 0] = 100.
    pre[10, 10] = 100.
    resid = pre.copy()
    frac = residual_fraction(pre, resid, column_as_float(col), [10., 10., 10.])
    assert np.isnan(frac[1]) and frac[0] == pytest.approx(1.0)


def test_unmeasured_stars_are_counted_per_bin():
    mag = np.array([12.1, 12.2, 12.3, 13.0, np.nan])
    frac = np.array([0.0, np.nan, 0.9, np.nan, 0.9])
    rows = unsubtracted_fraction_by_bin(mag, frac, np.array([12.0, 12.5, 13.5]))
    assert (rows[0]['n'], rows[0]['n_unmeasured'], rows[0]['n_unsub']) == (2, 1, 1)
    assert (rows[1]['n'], rows[1]['n_unmeasured']) == (0, 1)
    assert 'unmeas' in format_table(rows)


def test_all_nan_box_and_edge_clipping():
    img = np.full((10, 10), np.nan)
    img[0, 0] = 2.
    out = box_peak(img, [5, 0], [5, 0], signed_extreme=True)
    assert np.isnan(out[0]) and out[1] == 2.


@pytest.mark.parametrize('left,expect_unsub,expect_over', [
    (0.5, 0, 0),        # exactly at UNSUB_THRESHOLD: not counted
    (0.5000001, 1, 0),
    (-0.2, 0, 0),       # exactly at OVERSUB_THRESHOLD: not counted
    (-0.2000001, 0, 1),
])
def test_threshold_comparisons_are_strict(left, expect_unsub, expect_over):
    rows = unsubtracted_fraction_by_bin(np.array([12.2]), np.array([left]),
                                        np.array([12.0, 12.5]))
    assert rows[0]['n_unsub'] == expect_unsub
    assert rows[0]['n_over'] == expect_over


def test_bin_membership_is_lower_inclusive_upper_exclusive():
    mag = np.array([12.0, 12.5, 13.0])
    rows = unsubtracted_fraction_by_bin(mag, np.zeros(3),
                                        np.array([12.0, 12.5, 13.0]))
    assert [r['n'] for r in rows] == [1, 1]


def test_subpixel_position_rounds_to_nearest_pixel():
    img = np.zeros((20, 20))
    img[10, 13] = 7.
    # x = 15.4 rounds to 15 (box 13..17 holds the pixel); x = 15.6 rounds to 16
    # (box 14..18 does not), whereas truncation would give 15 for both.
    assert box_peak(img, [15.4], [10.])[0] == 7.
    assert box_peak(img, [15.6], [10.])[0] == 0.
    # y analogue
    img2 = np.zeros((20, 20))
    img2[13, 10] = 7.
    assert box_peak(img2, [10.], [15.4])[0] == 7.
    assert box_peak(img2, [10.], [15.6])[0] == 0.


def test_default_box_is_5x5():
    assert DEFAULT_BOX_HALF == 2
    img = np.zeros((20, 20))
    img[10, 12] = 4.   # 2 px away: inside a 5x5 box
    img[10, 13] = 9.   # 3 px away: outside
    assert box_peak(img, [10], [10])[0] == 4.
