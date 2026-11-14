"""The ZEROFRAME deblender reliability gate (``satstar_deblend_gate``).

data-qa measured (10678, 2026-10) that most RAW deblend splits proposed at
the seed stage are PSF-sidelobe/diffraction-spike artifacts, not real second
stars. This gate requires each proposed centre of a >=2-centre split to have
its OWN distinct, defect-filtered group-0-GROUPDQ-saturated core (1-to-1
nearest match); unmatched centres are dropped, and the whole split collapses
to the single blob centroid when fewer than 2 centres survive -- the
behaviour with ``zeroframe_deblend=False``.

These tests cover the gate in isolation (``group0_core_positions``,
``count_group0_cores``, ``gate_split_centers``, the defect registry, the env
switch) and wired into ``build_deblended_source_records`` (including its
``gate_diagnostics`` counters).
"""
import numpy as np
import pytest
from scipy import ndimage

from jwst_gc_pipeline.reduction.satstar_deblend import (
    build_deblended_source_records)
from jwst_gc_pipeline.reduction.satstar_deblend_gate import (
    count_group0_cores, deblend_gate_enabled, gate_split_centers,
    group0_core_positions, load_defect_registry)

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


def _bridged_blob_and_slice(xs, y=30, r=4):
    """A single connected saturated blob spanning several x positions (a
    bright-star diffraction spike can bridge cores much farther apart than
    two touching disks), built from a disk at each x plus a 1-px-tall strip
    connecting them, so ``ndimage.label`` always sees ONE component no matter
    how far apart the proposed centres under test are."""
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    blob = (yy == y) & (xx >= min(xs)) & (xx <= max(xs))
    for x in xs:
        blob = blob | _disk(x, y, r)
    sources, n = ndimage.label(blob)
    assert n == 1
    sl = ndimage.find_objects(sources)[0]
    return blob, sl


# ---- group0_core_positions / count_group0_cores ------------------------------

def test_count_group0_cores_none_signal_returns_none():
    blob, sl = _blob_and_slice()
    assert count_group0_cores(None, blob, sl) is None


def test_group0_core_positions_none_signal_returns_none():
    blob, sl = _blob_and_slice()
    assert group0_core_positions(None, blob, sl) is None


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


def test_group0_core_positions_returns_full_frame_coords():
    blob, sl = _blob_and_slice()
    g0 = _disk(10, 30, r=1) | _disk(16, 30, r=1)
    positions = group0_core_positions(g0, blob, sl)
    assert len(positions) == 2
    xs = sorted(round(p[1]) for p in positions)
    assert xs == [10, 16]


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


def test_count_group0_cores_small_cluster_dropped_by_min_area():
    """A 1-2 px group-0 speck is noise, not a core (matches data-qa's scan
    convention: a cluster must have area >= MIN_CORE_AREA)."""
    blob, sl = _blob_and_slice()
    yy, xx = np.mgrid[0:SHAPE[0], 0:SHAPE[1]]
    g0 = (_disk(10, 30, r=1)
          | ((xx == 16) & (yy == 30)))   # single-pixel speck at the 2nd core
    assert count_group0_cores(g0, blob, sl) == 1


# ---- defect-registry filtering ------------------------------------------------

def test_defect_registered_core_is_excluded():
    """A group-0 core whose (detector, 4px-bin) recurs as a registered
    fixed-pattern defect is dropped from the count/positions, even though the
    raw clustering sees it as a perfectly good independent core."""
    blob, sl = _blob_and_slice()
    g0 = _disk(10, 30, r=1) | _disk(16, 30, r=1)
    # core 2 sits at (y=30, x=16) -> bin (30//4, 16//4) = (7, 4)
    registry = {'nrcb1': {(7, 4)}}
    positions = group0_core_positions(g0, blob, sl, detector='nrcb1',
                                      registry=registry)
    assert len(positions) == 1
    assert round(positions[0][1]) == 10   # only the un-flagged core survives


def test_defect_registry_is_per_detector():
    """The same bin on a DIFFERENT detector is not flagged."""
    blob, sl = _blob_and_slice()
    g0 = _disk(10, 30, r=1) | _disk(16, 30, r=1)
    registry = {'nrcb1': {(7, 4)}}
    positions = group0_core_positions(g0, blob, sl, detector='nrca1',
                                      registry=registry)
    assert len(positions) == 2


def test_default_registry_loads_and_covers_known_detectors():
    """Smoke test for the packaged registry (built from data-qa's full 4040-
    frame 10678 scan): it loads without error and has an entry for every
    10678 NIRCam detector it was built from."""
    reg = load_defect_registry()
    assert reg, "default registry should not be empty"
    for det in ('nrca1', 'nrcb1', 'nrcalong', 'nrcblong'):
        assert det in reg
        assert len(reg[det]) > 0


def test_missing_registry_file_fails_open(tmp_path):
    """A registry path that does not exist must not suppress real splits --
    fail OPEN (no filtering), never closed."""
    missing = str(tmp_path / 'does_not_exist.json')
    reg = load_defect_registry(path=missing)
    assert reg == {}


def test_no_detector_disables_defect_filtering():
    """``detector=None`` (e.g. a caller that has not threaded it through) must
    not filter anything, even against a non-empty registry -- unknown
    detector is not the same as 'apply no filter', so skip filtering
    entirely rather than guessing."""
    blob, sl = _blob_and_slice()
    g0 = _disk(10, 30, r=1) | _disk(16, 30, r=1)
    registry = {'nrcb1': {(7, 4)}}
    positions = group0_core_positions(g0, blob, sl, detector=None,
                                      registry=registry)
    assert len(positions) == 2


# ---- gate_split_centers: per-centre matching ---------------------------------

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
        [(30.0, 10.0), (30.0, 16.0)], com, blob, sl, g0,
        match_radius_px=1.5)
    assert centers == [com]
    assert n == 1


def test_two_centers_accepted_with_two_confirmed_cores():
    blob, sl = _blob_and_slice()
    com = (30.0, 15.0)
    proposed = [(30.0, 10.0), (30.0, 16.0)]
    g0 = _disk(10, 30, r=1) | _disk(16, 30, r=1)
    centers, n = gate_split_centers(proposed, com, blob, sl, g0,
                                    match_radius_px=1.5)
    assert centers == proposed
    assert n == 2


def test_per_centre_match_keeps_only_confirmed_of_several_proposed():
    """2 confirmed cores, 4 PROPOSED centres (the old bulk ``n_g0 >= 2`` check
    would have kept all 4 once the raw count cleared the floor): only the 2
    centres with their own matching core survive."""
    blob, sl = _bridged_blob_and_slice([10, 13, 19, 22])
    com = (30.0, 16.0)
    proposed = [(30.0, 10.0), (30.0, 13.0), (30.0, 19.0), (30.0, 22.0)]
    g0 = _disk(10, 30, r=1) | _disk(22, 30, r=1)
    centers, n = gate_split_centers(proposed, com, blob, sl, g0,
                                    match_radius_px=1.5)
    assert sorted(centers) == [(30.0, 10.0), (30.0, 22.0)]
    assert n == 2


def test_per_centre_match_is_one_to_one():
    """A single group-0 core must not confirm two different proposed centres
    (``linear_sum_assignment``, not independent nearest-neighbour per
    centre)."""
    blob, sl = _bridged_blob_and_slice([9, 11])
    com = (30.0, 14.0)
    # two centres both close to the SAME single core at x=10; only one core
    # exists at all, so the split must still collapse (< 2 confirmed centres)
    proposed = [(30.0, 9.0), (30.0, 11.0)]
    g0 = _disk(10, 30, r=1)
    centers, n = gate_split_centers(proposed, com, blob, sl, g0,
                                    match_radius_px=1.5)
    assert centers == [com]
    assert n == 1


def test_three_to_five_centres_collapse_to_single_when_defect_removes_a_core():
    """A defect cluster posing as the 2nd core: 2 raw group-0 clusters, but
    one is a registered fixed-pattern defect, leaving only 1 real confirmed
    core -- the whole split (5 proposed centres) collapses to the blob
    centroid, not just to the 1 confirmed centre."""
    blob, sl = _bridged_blob_and_slice([10, 15, 20, 25, 30])
    com = (30.0, 20.0)
    proposed = [(30.0, 10.0), (30.0, 15.0), (30.0, 20.0), (30.0, 25.0),
                (30.0, 30.0)]
    g0 = _disk(10, 30, r=1) | _disk(30, 30, r=1)
    # core at (30, 30) -> bin (7, 7) flagged as a fixed-pattern defect
    registry = {'nrca1': {(7, 7)}}
    centers, n = gate_split_centers(proposed, com, blob, sl, g0,
                                    detector='nrca1', registry=registry,
                                    match_radius_px=1.5)
    assert centers == [com]
    assert n == 1   # only the non-defect core counts


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


def test_build_records_diagnostics_count_unavailable():
    sat, sources, coms, sizes, zf = _merged_pair_scene()
    diag = {}
    build_deblended_source_records(
        sat, sources, coms, sizes, zf, np.zeros(SHAPE), 2.0,
        group0_saturated=None, gate_diagnostics=diag)
    assert diag['n_multi_centre_proposed'] == 1
    assert diag['n_gate_unavailable'] == 1
    assert diag['n_gate_rejected'] == 0
    assert diag['n_gate_accepted'] == 0


def test_build_records_diagnostics_count_rejected():
    sat, sources, coms, sizes, zf = _merged_pair_scene()
    diag = {}
    build_deblended_source_records(
        sat, sources, coms, sizes, zf, np.zeros(SHAPE), 2.0,
        group0_saturated=np.zeros(SHAPE, bool), gate_diagnostics=diag)
    assert diag['n_gate_unavailable'] == 0
    assert diag['n_gate_rejected'] == 1
    assert diag['n_gate_accepted'] == 0


def test_build_records_diagnostics_count_accepted():
    sat, sources, coms, sizes, zf = _merged_pair_scene()
    g0 = _disk(25, 30, r=1) | _disk(33, 30, r=1)
    diag = {}
    build_deblended_source_records(
        sat, sources, coms, sizes, zf, np.zeros(SHAPE), 2.0,
        group0_saturated=g0, gate_diagnostics=diag)
    assert diag['n_gate_unavailable'] == 0
    assert diag['n_gate_rejected'] == 0
    assert diag['n_gate_accepted'] == 1


if __name__ == '__main__':
    pytest.main([__file__])
