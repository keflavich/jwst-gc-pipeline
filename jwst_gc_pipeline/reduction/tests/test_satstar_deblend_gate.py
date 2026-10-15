"""The ZEROFRAME deblender reliability gate (``satstar_deblend_gate``).

data-qa measured (10678, 2026-10-15) that most RAW deblend splits proposed at
the seed stage are PSF-sidelobe/diffraction-spike artifacts, not real second
stars: only ~0.5-1.3% of proposed secondary centres are confirmed by an
independent group-0-GROUPDQ-saturated core or a VIRAC2 crossmatch. This gate
requires >=2 independent group-0-saturated cores before accepting a proposed
>=2-centre split; otherwise it collapses back to the single blob centroid
(the behaviour with ``zeroframe_deblend=False``).

These tests cover the gate in isolation (``count_group0_cores``,
``gate_split_centers``, the env switch) and wired into
``build_deblended_source_records``.
"""
import numpy as np
import pytest
from scipy import ndimage

from jwst_gc_pipeline.reduction.satstar_deblend import (
    build_deblended_source_records)
from jwst_gc_pipeline.reduction.satstar_deblend_gate import (
    count_group0_cores, deblend_gate_enabled, gate_split_centers)

SHAPE = (60, 60)


def _disk(x, y, r=2):
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    return (xx - x) ** 2 + (yy - y) ** 2 <= r ** 2


def _blob_and_slice(x0=10, y0=30, x1=16, y1=30, r=4):
    """A single connected saturated blob spanning two touching cores (centres
    6 px apart, radius 4 each -> overlapping), for the slice/bbox machinery
    the gate reads."""
    blob = _disk(x0, y0, r) | _disk(x1, y1, r)
    sources, n = ndimage.label(blob)
    assert n == 1
    sl = ndimage.find_objects(sources)[0]
    return blob, sl


# ---- count_group0_cores -----------------------------------------------------

def test_count_group0_cores_none_signal_returns_none():
    blob, sl = _blob_and_slice()
    assert count_group0_cores(None, blob, sl) is None


def test_count_group0_cores_none_slice_returns_none():
    blob, _ = _blob_and_slice()
    g0 = np.zeros(SHAPE, bool)
    assert count_group0_cores(g0, blob, None) is None


def test_count_group0_cores_no_overlap_is_zero():
    blob, sl = _blob_and_slice()
    g0 = np.zeros(SHAPE, bool)   # nothing saturated in group 0 anywhere
    assert count_group0_cores(g0, blob, sl) == 0


def test_count_group0_cores_two_distinct_cores():
    blob, sl = _blob_and_slice()
    g0 = _disk(10, 30, r=1) | _disk(16, 30, r=1)   # one small core per star
    assert count_group0_cores(g0, blob, sl) == 2


def test_count_group0_cores_one_core_only():
    blob, sl = _blob_and_slice()
    g0 = _disk(10, 30, r=1)   # only ONE star independently saturates in g0
    assert count_group0_cores(g0, blob, sl) == 1


def test_count_group0_cores_dilation_merges_adjacent_cores():
    """Two group-0 cores within ``dilate_px`` of each other merge into one
    cluster -- the gate counts STARS, not raw blob fragments, so two tiny
    adjacent hot patches of the SAME star must not inflate the count."""
    blob, sl = _blob_and_slice()
    g0 = _disk(10, 30, r=1) | _disk(13, 30, r=1)   # 3 px apart, same star
    assert count_group0_cores(g0, blob, sl, dilate_px=2) == 1


def test_count_group0_cores_empty_blob_is_zero():
    g0 = np.ones(SHAPE, bool)
    empty_blob = np.zeros(SHAPE, bool)
    sl = (slice(0, 10), slice(0, 10))
    assert count_group0_cores(g0, empty_blob, sl) == 0


# ---- gate_split_centers ------------------------------------------------------

def test_single_center_passes_through_unchanged():
    blob, sl = _blob_and_slice()
    centers, n = gate_split_centers([(30.0, 15.0)], (30.0, 15.0), blob, sl, None)
    assert centers == [(30.0, 15.0)]
    assert n is None


def test_two_centers_rejected_when_signal_unavailable():
    """Fail closed: no independent group-0 signal -> collapse to the single
    blob centroid, same as a frame with no zeroframe at all."""
    blob, sl = _blob_and_slice()
    com = (30.0, 15.0)
    centers, n = gate_split_centers(
        [(30.0, 10.0), (30.0, 20.0)], com, blob, sl, None)
    assert centers == [com]
    assert n is None


def test_two_centers_rejected_with_only_one_confirmed_core():
    blob, sl = _blob_and_slice()
    com = (30.0, 15.0)
    g0 = _disk(10, 30, r=1)
    centers, n = gate_split_centers(
        [(30.0, 10.0), (30.0, 16.0)], com, blob, sl, g0)
    assert centers == [com]
    assert n == 1


def test_two_centers_accepted_with_two_confirmed_cores():
    blob, sl = _blob_and_slice()
    com = (30.0, 15.0)
    proposed = [(30.0, 10.0), (30.0, 16.0)]
    g0 = _disk(10, 30, r=1) | _disk(16, 30, r=1)
    centers, n = gate_split_centers(proposed, com, blob, sl, g0)
    assert centers == proposed
    assert n == 2


def test_env_switch_off_disables_the_gate(monkeypatch):
    monkeypatch.setenv('SATSTAR_DEBLEND_GATE', '0')
    blob, sl = _blob_and_slice()
    com = (30.0, 15.0)
    proposed = [(30.0, 10.0), (30.0, 20.0)]
    # no independent signal at all, but the gate itself is off -> keep split
    centers, n = gate_split_centers(proposed, com, blob, sl, None)
    assert centers == proposed
    assert n is None


def test_bad_env_value_raises():
    with pytest.raises(ValueError):
        deblend_gate_enabled(env={'SATSTAR_DEBLEND_GATE': 'maybe'})


def test_env_switch_default_is_on():
    assert deblend_gate_enabled(env={}) is True


# ---- wired into build_deblended_source_records ------------------------------

def _gauss(x, y, amp, sig=0.8):
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    return amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))


def _merged_pair_scene():
    a, b = (25, 30), (33, 30)
    sat = _disk(*a, r=4) | _disk(*b, r=4)
    sources, n = ndimage.label(sat)
    assert n == 1
    zf = 100.0 + _gauss(*a, 5000.0) + _gauss(*b, 3000.0)
    coms = ndimage.center_of_mass(sat, sources, [1])
    sizes = ndimage.sum_labels(sat, sources, [1])
    return sat, sources, coms, sizes, zf


def test_build_records_collapses_unconfirmed_split():
    sat, sources, coms, sizes, zf = _merged_pair_scene()
    records = build_deblended_source_records(
        sat, sources, coms, sizes, zf, np.zeros(SHAPE), 2.0,
        group0_saturated=None)
    assert len(records) == 1, records


def test_build_records_keeps_confirmed_split():
    sat, sources, coms, sizes, zf = _merged_pair_scene()
    g0 = _disk(25, 30, r=1) | _disk(33, 30, r=1)
    records = build_deblended_source_records(
        sat, sources, coms, sizes, zf, np.zeros(SHAPE), 2.0,
        group0_saturated=g0)
    assert len(records) == 2, records


def test_build_records_gate_off_keeps_unconfirmed_split(monkeypatch):
    monkeypatch.setenv('SATSTAR_DEBLEND_GATE', '0')
    sat, sources, coms, sizes, zf = _merged_pair_scene()
    records = build_deblended_source_records(
        sat, sources, coms, sizes, zf, np.zeros(SHAPE), 2.0,
        group0_saturated=None)
    assert len(records) == 2, records


if __name__ == '__main__':
    pytest.main([__file__])
