"""Per-visit consistency of RECORDED_BULK fields (#857).

m4 (1979) and m92 (1334) record one bulk per (visit, filter).  The m7
cross-filter gate demands one frame per visit, so a filter-to-filter
difference in the recorded bulks reappears at the gate as a residual of the
same size.  These tests pin that identity and guard the invariant for any
entry that later opts in to ``tie_through_reference_filter``.
"""
import numpy as np
import pytest

from jwst_gc_pipeline.reduction import alignment_config as ac
from jwst_gc_pipeline.reduction.alignment_config import (
    ANY, GAIA, RECORDED_BULK, BulkEntry, FieldAlignment)
from jwst_gc_pipeline.reduction.recorded_bulk_audit import (
    applied_bulk_onsky_mas, bulk_spread_by_visit)

DEC = 43.139
COSD = np.cos(np.radians(DEC))
VISIT = 'jw09999001001'


def _cfg(recorded_bulk):
    return FieldAlignment(
        proposal='9999', fields=None, reference_frame=GAIA,
        source=RECORDED_BULK, consensus_jitter=True, visit_key='full',
        dec_ref_deg=DEC, recorded_bulk=recorded_bulk)


def _raw_frame_mas():
    """One star's position in the raw assign_wcs frame, identical in both bands
    (the raw frames of the two bands agree, as measured on m4 o002)."""
    return np.array([123.0, -45.0])


def _after_fix_alignment_mas(cfg, filtername):
    """Position after ``fix_alignment`` applies the recorded bulk, on-sky mas."""
    dra_coord, ddec, _ = ac.lookup_recorded_bulk(cfg, VISIT, filtername)
    return _raw_frame_mas() + np.array([dra_coord * 1000.0 * COSD, ddec * 1000.0])


def test_crossfilter_differential_equals_the_recorded_bulk_difference():
    cfg = _cfg({(VISIT, 'F150W'): BulkEntry(-1853.5, -710.6, onsky_mas=True),
                (VISIT, 'F090W'): BulkEntry(-1832.1, -708.2, onsky_mas=True)})
    differential = (_after_fix_alignment_mas(cfg, 'F090W')
                    - _after_fix_alignment_mas(cfg, 'F150W'))
    assert differential == pytest.approx([21.4, 2.4], abs=1e-6)
    row = bulk_spread_by_visit(cfg)[VISIT]
    assert row['differential_mas']['F090W'] == pytest.approx(
        tuple(differential), abs=1e-6)
    # anchor = the band nearest VIRAC2 Ks, the m7 gate's anchor
    assert row['anchor'] == 'F150W'
    assert row['max_differential_mas'] == pytest.approx(np.hypot(21.4, 2.4))


def test_one_visit_bulk_leaves_no_crossfilter_differential():
    cfg = _cfg({(VISIT, ANY): BulkEntry(-1853.5, -710.6, onsky_mas=True)})
    assert (_after_fix_alignment_mas(cfg, 'F090W')
            == pytest.approx(_after_fix_alignment_mas(cfg, 'F150W'), abs=1e-9))
    # an ANY key names no filter, so the audit has no per-filter row to split
    assert bulk_spread_by_visit(cfg) == {}


def test_applied_bulk_is_nan_where_no_key_matches():
    cfg = _cfg({(VISIT, 'F150W'): BulkEntry(1.0, 2.0, onsky_mas=True)})
    dra, ddec = applied_bulk_onsky_mas(cfg, VISIT, 'F277W')
    assert np.isnan(dra) and np.isnan(ddec)


def _entry(proposal):
    (cfg,) = [c for c in ac.ALIGNMENT_CONFIG if c.proposal == proposal]
    return cfg


def test_m4_o002_differential_matches_the_measured_gate_residual():
    """m4 o002 F322W2 minus F150W2 in the vetted catalogs is (-547.5, +92.7)
    mas; the recorded bulks differ by (-547.6, +92.4) mas.  The m7 gate reports
    anchor minus filter, (+548.3, -92.5) mas.  Re-keying the 1979 bulks to one
    per visit changes this pin."""
    row = bulk_spread_by_visit(_entry('1979'))['jw01979002001']
    assert row['anchor'] == 'F150W2'
    assert row['differential_mas']['F322W2'] == pytest.approx(
        (-547.6, 92.4), abs=0.05)


def test_m92_f090w_differential_matches_the_recorded_bulks():
    row = bulk_spread_by_visit(_entry('1334'))['jw01334001001']
    assert row['anchor'] == 'F277W'
    assert row['differential_mas']['F090W'] == pytest.approx(
        (20.0, 3.5), abs=0.05)


@pytest.mark.parametrize(
    'cfg', [c for c in ac.ALIGNMENT_CONFIG
            if c.source == RECORDED_BULK and c.tie_through_reference_filter],
    ids=lambda c: c.proposal)
def test_opted_in_recorded_bulk_entry_has_one_bulk_per_visit(cfg):
    """An entry that ties its other bands through ``reference_filter`` must
    resolve every filter of a visit to the same bulk, and name the band the m7
    gate anchors on.  Vacuous while no RECORDED_BULK entry is opted in."""
    from jwst_gc_pipeline.photometry.visit_consensus import (
        pick_reference_anchor_filter)
    # consensus_jitter routes the per-exposure term to the consensus table
    assert cfg.consensus_jitter
    for visit, row in bulk_spread_by_visit(cfg).items():
        assert row['max_differential_mas'] == pytest.approx(0.0, abs=1e-9), visit
        assert cfg.reference_filter == pick_reference_anchor_filter(
            list(row['filters'])), visit
