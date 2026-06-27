"""Regression tests for the adaptive MIRI east-edge glow trim
(``_adaptive_east_trim`` in PipelineMIRI).

Background: a blind fixed east-edge DQ trim (MIRI_TRIM_EAST=40) discarded clean
edge data in brick F2550W visit-001 frames _08/10/12101, which cover a
detector-gap notch at their far-east edge with FLAT (un-glowing) data -- the
eastern-neighbour tile carries the coronagraph defect there, so trimming the
clean edge reopened the notch as NaN.  The adaptive trim must:
  * return 0 for a frame whose east edge is flat (no glow) -> preserve real data
  * return a positive run for a frame whose east edge is elevated (glow)
"""
import numpy as np
import pytest

pytest.importorskip("jwst")  # PipelineMIRI imports the jwst pipeline at module load
from jwst_gc_pipeline.reduction.PipelineMIRI import _adaptive_east_trim


def _frame(nx=1024, ny=1024, baseline=1000.0, edge_gain=None, edge_width=60):
    """Build (dq, sci) for a fully-illuminated frame (science cols 0..nx-1).
    edge_gain: peak multiplicative elevation at the far-east column, ramping
    linearly over the last ``edge_width`` columns (None -> flat edge)."""
    dq = np.zeros((ny, nx), dtype=np.uint32)
    sci = np.full((ny, nx), baseline, dtype=np.float32)
    if edge_gain is not None:
        x = np.arange(nx)
        ramp = np.clip((x - (nx - edge_width)) / edge_width, 0, 1)
        sci *= (1.0 + (edge_gain - 1.0) * ramp)[None, :]
    return dq, sci


def test_flat_edge_returns_zero():
    """Flat east edge -> trim 0 (real data preserved)."""
    dq, sci = _frame(edge_gain=None)
    assert _adaptive_east_trim(dq, sci, 0, 1023) == 0


def test_glow_edge_returns_positive_run():
    """A +40% east-edge glow ramp over 60 cols is flagged (>= ~half the ramp)."""
    dq, sci = _frame(edge_gain=1.40, edge_width=60)
    run = _adaptive_east_trim(dq, sci, 0, 1023)
    assert run > 20, run


def test_mild_subthreshold_glow_not_trimmed():
    """A glow below the 8% threshold must not be trimmed (avoid eating real data)."""
    dq, sci = _frame(edge_gain=1.03, edge_width=60)
    assert _adaptive_east_trim(dq, sci, 0, 1023) == 0


def test_run_capped_at_east_max():
    """A pathological frame elevated everywhere cannot trim past east_max."""
    dq, sci = _frame(edge_gain=1.40, edge_width=1024)
    run = _adaptive_east_trim(dq, sci, 0, 1023, east_max=96)
    assert run <= 96
