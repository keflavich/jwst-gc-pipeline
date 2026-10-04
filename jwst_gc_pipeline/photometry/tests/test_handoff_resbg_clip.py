"""Tests for the resbg-hole clip under handed-off saturated stars."""
import numpy as np

from jwst_gc_pipeline.photometry.cataloging import (
    _clip_resbg_hole_at_handoff, _HANDOFF_RESBG_CLIP_RADIUS_PIX)

R = 10.0
FM = 3.5


def _setup():
    bg = np.full((60, 60), 4.0)
    bg[28:33, 28:33] = -20.0      # hole at the core
    bg[30, 36] = 3.0              # mild dip below the field median, inside R
    bg[39, 39] = -5.0             # inside the bounding box, outside the circle
    bg[45:50, 45:50] = -30.0      # hole far from the hand-off star
    data = np.full((60, 60), 100.0)
    return data, bg


def test_hole_raised_inside_radius_only():
    data, bg = _setup()
    out, n, npix, added = _clip_resbg_hole_at_handoff(
        data, bg, np.array([[30.0, 30.0]]), R, FM)
    assert n == 1
    yy, xx = np.mgrid[:60, :60]
    inside = (xx - 30.0) ** 2 + (yy - 30.0) ** 2 <= R ** 2
    low = bg < FM
    sel = inside & low
    np.testing.assert_allclose(out[sel], data[sel] + (FM - bg[sel]))
    assert npix == sel.sum() and added.size == npix
    # bg >= field median inside R untouched
    assert np.all(out[inside & ~low] == data[inside & ~low])
    # outside R untouched, including the far hole
    assert np.all(out[~inside] == data[~inside])
    assert out[47, 47] == data[47, 47]
    assert out[30, 30] == 100.0 + FM + 20.0
    assert not np.shares_memory(out, data)


def test_no_handoff_positions_is_noop():
    data, bg = _setup()
    for xy in (None, np.zeros((0, 2))):
        out, n, npix, added = _clip_resbg_hole_at_handoff(data, bg, xy, R, FM)
        assert n == 0 and npix == 0 and added.size == 0
        np.testing.assert_array_equal(out, data)


def test_nan_bg_left_alone():
    data, bg = _setup()
    bg[29, 29] = np.nan
    data[31, 31] = np.nan
    out, n, npix, added = _clip_resbg_hole_at_handoff(
        data, bg, np.array([[30.0, 30.0]]), R, FM)
    assert out[29, 29] == 100.0
    assert np.isnan(out[31, 31])
    assert np.all(np.isfinite(added))
    assert np.isfinite(out[~np.isnan(data)]).all()


def test_edge_position_and_default_radius():
    data, bg = _setup()
    out, n, npix, _ = _clip_resbg_hole_at_handoff(
        data, bg, np.array([[0.0, 0.0], [np.nan, 5.0]]), R, FM)
    assert n == 1
    assert _HANDOFF_RESBG_CLIP_RADIUS_PIX == 10.0
