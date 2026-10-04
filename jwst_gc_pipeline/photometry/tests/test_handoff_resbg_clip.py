"""Tests for the resbg-hole clip under handed-off saturated stars.

The fitted data are ``crf - bg``.  A negative hole in ``bg`` at a handed-off
core puts a positive pedestal in the data; the clip must take it out, giving
``crf - max(bg, floor)`` with ``floor`` the bg median in the annulus just
outside the clip radius.
"""
import numpy as np
import pytest

from jwst_gc_pipeline.photometry.cataloging import (
    _apply_handoff_resbg_clip, _clip_resbg_hole_at_handoff,
    _HANDOFF_RESBG_CLIP_ANNULUS_PIX, _HANDOFF_RESBG_CLIP_RADIUS_PIX)

R = _HANDOFF_RESBG_CLIP_RADIUS_PIX
N = 80
YY, XX = np.mgrid[:N, :N]


def _star(x, y=40.0):
    return np.array([[x, y]]), np.hypot(XX - x, YY - y)


def _frame(bg):
    crf = np.full((N, N), 100.0)
    return crf, crf - bg


def test_defaults():
    assert R == 10.0
    assert _HANDOFF_RESBG_CLIP_ANNULUS_PIX == 5.0


def test_pedestal_removed_inside_radius_only():
    star, rr = _star(40.0)
    bg = np.full((N, N), 4.0)
    bg[rr < 3] = -20.0                    # the hole at the core
    bg[40, 46] = 3.0                      # mild dip, r = 6
    bg[33, 33] = -5.0                     # r = 9.9, inside R
    bg[47, 48] = -5.0                     # r = 10.6, annulus (outside R)
    bg[5:9, 5:9] = -30.0                  # a hole far from the star
    crf, data = _frame(bg)
    out, n, npix, removed = _clip_resbg_hole_at_handoff(data, bg, star)
    assert n == 1
    inside = rr < R
    low = inside & (bg < 4.0)
    np.testing.assert_allclose(out[inside],
                               crf[inside] - np.maximum(bg[inside], 4.0))
    assert np.all(out[low] < data[low]), 'the clip must lower the data'
    assert out[40, 40] == 96.0
    np.testing.assert_allclose(removed, (4.0 - bg)[low])
    assert np.all(removed > 0)
    assert npix == int(low.sum())
    np.testing.assert_array_equal(out[~inside], data[~inside])
    assert not np.shares_memory(out, data)


@pytest.mark.parametrize('xc', [20.0, 60.0])
def test_floor_follows_a_gradient(xc):
    """bg = x - 40 across the frame; the frame median is ~0.  At x = 60 the
    local level is +20 (a frame-median floor would leave most of the
    pedestal); at x = 20 it is -20 (a frame-median floor would over-subtract).
    The annulus floor removes the hole depth in both cases."""
    star, rr = _star(xc)
    level = xc - 40.0
    bg = XX - 40.0
    bg[rr < 3] = level - 25.0
    crf, data = _frame(bg)
    out, n, _, _ = _clip_resbg_hole_at_handoff(data, bg, star)
    assert n == 1
    inside = rr < R
    np.testing.assert_allclose(out[inside],
                               crf[inside] - np.maximum(bg[inside], level))
    np.testing.assert_allclose(out[rr < 3], crf[rr < 3] - level)


def test_zero_bg_is_a_reproject_miss():
    star, rr = _star(40.0)
    bg = np.full((N, N), 4.0)
    bg[rr < 3] = -20.0
    ann = (rr >= R) & (rr < R + 5)
    zero = ann & (XX < 46)
    assert zero.sum() > 0.5 * ann.sum()
    bg[zero] = 0.0
    bg[40, 42] = 0.0                      # a miss inside the hole
    crf, data = _frame(bg)
    out, _, _, _ = _clip_resbg_hole_at_handoff(data, bg, star)
    core = (rr < 3) & (bg != 0)
    # floor from the valid annulus pixels only (4.0), not pulled toward 0
    np.testing.assert_allclose(out[core], crf[core] - 4.0)
    assert out[40, 42] == data[40, 42]


def test_gap_nan_and_missing_floor_left_alone():
    star, rr = _star(40.0)
    bg = np.full((N, N), 4.0)
    bg[rr < 3] = -20.0
    bg[40, 41] = np.nan
    crf, data = _frame(bg)
    data[40, 41] = 100.0
    data[40, 39] = 0.0                    # detector gap
    data[39, 40] = np.nan
    out, n, _, _ = _clip_resbg_hole_at_handoff(data, bg, star)
    assert n == 1
    assert out[40, 41] == 100.0
    assert out[40, 39] == 0.0
    assert np.isnan(out[39, 40])
    assert out[41, 40] == 96.0
    # no valid bg in the annulus: the star is skipped
    b2 = np.where(rr >= R, 0.0, bg)
    out2, n2, npix2, _ = _clip_resbg_hole_at_handoff(data, b2, star)
    assert (n2, npix2) == (0, 0)
    np.testing.assert_array_equal(out2, data)


@pytest.mark.parametrize('order', [[0, 1], [1, 0]])
def test_overlap_takes_the_higher_floor(order):
    bg = np.where(XX < 40, 2.0, 6.0)
    bg[40, 40] = -20.0
    crf, data = _frame(bg)
    stars = np.array([[34.0, 40.0], [46.0, 40.0]])
    floors = []
    for xc, yc in stars:
        r = np.hypot(XX - xc, YY - yc)
        floors.append(np.median(bg[(r >= R) & (r < R + 5)]))
    assert floors[0] < floors[1]
    out, n, _, _ = _clip_resbg_hole_at_handoff(data, bg, stars[order])
    assert n == 2
    assert out[40, 40] == pytest.approx(crf[40, 40] - floors[1])


@pytest.mark.parametrize('xy', [None, np.zeros((0, 2)),
                                np.array([[np.nan, 5.0]]),
                                np.array([[-50.0, -50.0]])])
def test_no_usable_position_is_a_no_op(xy):
    _, rr = _star(40.0)
    bg = np.full((N, N), 4.0)
    bg[rr < 3] = -20.0
    _, data = _frame(bg)
    out, n, npix, removed = _clip_resbg_hole_at_handoff(data, bg, xy)
    assert (n, npix, removed.size) == (0, 0, 0)
    np.testing.assert_array_equal(out, data)


def test_wiring_runs_only_with_handoff_and_resbg():
    star, rr = _star(40.0)
    bg = np.full((N, N), 4.0)
    bg[rr < 3] = -20.0
    crf, data = _frame(bg)
    for args in ((bg, None, 'resbg.fits'), (bg, star, None), (bg, star, ''),
                 (None, star, 'resbg.fits')):
        out, msg = _apply_handoff_resbg_clip(data, *args)
        assert out is data and msg is None
    out, msg = _apply_handoff_resbg_clip(data, bg, star, 'resbg.fits')
    np.testing.assert_allclose(out[rr < 3], crf[rr < 3] - 4.0)
    assert msg == (f"[manual] hand-off resbg clip: 1 star(s), "
                   f"{int((rr < 3).sum())} pixel(s) lowered, median 24 MJy/sr")
