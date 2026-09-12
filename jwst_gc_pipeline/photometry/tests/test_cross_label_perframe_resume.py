"""A per-frame fit serves every merge label, so the resume must cross labels.

Requesting the standard ``nrca,nrcb,merged`` fitted every NIRCam frame TWICE:
the ``merged`` pass runs over the same files as the per-module passes, and the
product a fit writes carries no merge token (``file_module = file_detector``),
so the second fit recomputed the first's file byte for byte.

Measured 2026-09-11 (issue #840): crowded_l3 carried 560 markers for 280 frames
(280 ``merged`` + 140 ``nrca`` + 140 ``nrcb``), and g028's m12 fan-out logged
560 ``Completed basic photometry`` lines against 280 frames and 280 per-frame
catalogs.

``PHASE`` here is ``m12`` throughout, and every call passes ``cross_label=True``
explicitly: that equivalence holds where the fit takes no label-scoped input,
which is the first phase and no other.  ``test_label_scoped_phase_resume.py``
covers m3..m7, where it does not.
"""
import os

import pytest

from jwst_gc_pipeline.photometry.cataloging import (
    PERFRAME_MERGE_LABELS, perframe_marker_path, select_resumable_frames)

FRAME = 'jw09438006001_0210b_00001_nrca1_destreak_o006_crf.fits'
FILT = 'F210M'
PHASE = 'm12'


def _frame(tmp_path, name=FRAME):
    p = tmp_path / name
    p.write_text('stand-in for a crf; only its mtime is read')
    return str(p)


def _args(fn):
    return [{'filename': fn}]


def _mark(marker_dir, fn, merge, kind='ok', detector='nrca1'):
    p = perframe_marker_path(str(marker_dir), fn, detector, FILT, PHASE, kind,
                             merge=merge)
    open(p, 'w').close()
    # markers must be at least as new as the frame
    os.utime(p, (os.path.getmtime(fn) + 10,) * 2)
    return p


def test_the_merged_pass_resumes_a_frame_the_nrca_pass_fitted(tmp_path):
    """The whole point: one fit, not one per label."""
    md = tmp_path / 'markers'; md.mkdir()
    fn = _frame(tmp_path)
    _mark(md, fn, 'nrca')
    todo, ok, nov, stale = select_resumable_frames(
        _args(fn), str(md), FILT, PHASE, 'merged', cross_label=True)
    assert ok == [fn], "the merged pass refitted a frame nrca had already done"
    assert todo == [] and stale == []


def test_and_the_reverse_direction(tmp_path):
    md = tmp_path / 'markers'; md.mkdir()
    fn = _frame(tmp_path)
    _mark(md, fn, 'merged')
    todo, ok, _nov, _stale = select_resumable_frames(
        _args(fn), str(md), FILT, PHASE, 'nrca', cross_label=True)
    assert ok == [fn] and todo == []


@pytest.mark.parametrize("wrote,asks", [(w, a) for w in PERFRAME_MERGE_LABELS
                                        for a in PERFRAME_MERGE_LABELS])
def test_any_label_receipt_satisfies_any_other(tmp_path, wrote, asks):
    md = tmp_path / 'markers'; md.mkdir()
    fn = _frame(tmp_path)
    _mark(md, fn, wrote)
    todo, ok, _nov, _stale = select_resumable_frames(
        _args(fn), str(md), FILT, PHASE, asks, cross_label=True)
    assert ok == [fn], f"{asks} refitted a frame {wrote} had fitted"
    assert todo == []


def test_a_frame_with_NO_marker_is_still_refitted(tmp_path):
    """The exemption is about which pass wrote the receipt, not about whether
    one exists."""
    md = tmp_path / 'markers'; md.mkdir()
    fn = _frame(tmp_path)
    todo, ok, _nov, _stale = select_resumable_frames(
        _args(fn), str(md), FILT, PHASE, 'merged', cross_label=True)
    assert todo and ok == []


def test_an_UNSCOPED_marker_still_does_not_resume(tmp_path):
    """Refused for the reason the merge-scoped rule was written: an unscoped
    marker cannot say whether any pass wrote it."""
    md = tmp_path / 'markers'; md.mkdir()
    fn = _frame(tmp_path)
    p = perframe_marker_path(str(md), fn, 'nrca1', FILT, PHASE, 'ok', merge=None)
    open(p, 'w').close()
    os.utime(p, (os.path.getmtime(fn) + 10,) * 2)
    todo, ok, _nov, _stale = select_resumable_frames(
        _args(fn), str(md), FILT, PHASE, 'merged', cross_label=True)
    assert todo and ok == [], "an unscoped marker must not resume"


def test_a_STALE_cross_label_marker_does_not_resume(tmp_path):
    """The staleness gate is unchanged: a receipt older than the frame it names
    is a receipt for a frame that no longer exists (#570)."""
    md = tmp_path / 'markers'; md.mkdir()
    fn = _frame(tmp_path)
    p = _mark(md, fn, 'nrca')
    os.utime(p, (os.path.getmtime(fn) - 3600,) * 2)   # older than the frame
    todo, ok, _nov, stale = select_resumable_frames(
        _args(fn), str(md), FILT, PHASE, 'merged', cross_label=True)
    assert stale == [fn] and todo and ok == []


def test_a_cross_label_nooverlap_marker_carries_over(tmp_path):
    md = tmp_path / 'markers'; md.mkdir()
    fn = _frame(tmp_path)
    _mark(md, fn, 'nrcb', kind='nooverlap')
    todo, ok, nov, _stale = select_resumable_frames(
        _args(fn), str(md), FILT, PHASE, 'merged', cross_label=True)
    assert [f for f, _r in nov] == [fn] and todo == [] and ok == []


def test_seed_inputs_still_gate_a_cross_label_marker(tmp_path):
    """m3..m7 fit against the previous phase's finalize products; a marker older
    than those is stale however it is labelled."""
    md = tmp_path / 'markers'; md.mkdir()
    fn = _frame(tmp_path)
    p = _mark(md, fn, 'nrca')
    seed = tmp_path / 'seed.fits'
    seed.write_text('previous phase product')
    os.utime(str(seed), (os.path.getmtime(p) + 60,) * 2)   # seed newer than marker
    todo, ok, _nov, stale = select_resumable_frames(
        _args(fn), str(md), FILT, PHASE, 'merged', seed_inputs=(str(seed),),
        cross_label=True)
    assert stale == [fn] and todo and ok == []
