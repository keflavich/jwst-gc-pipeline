"""A frame resumed across merge labels must still leave a receipt under THIS
label, or the finalize reads it as a dropped exposure.

#841 let the `merged` pass resume from an `nrca`/`nrcb` marker instead of
refitting. It did not stamp a `merged` marker for what it skipped, and the
finalize's completeness check is per label, so it aborted:

    RuntimeError: manual [m3] F150W/merged: --manual-finalize-only found 95
    candidate frame(s) with NO completion marker -- the per-frame fan-out did
    not finish them (a dropped exposure corrupts the catalog).

Measured on wd1 2026-09-12: F150W (m3 ran after #841) had 1 merged marker where
96 were expected, against 48 nrca + 48 nrcb; F115W (m3 ran before #841) had the
full 96.
"""
import inspect
import os

from jwst_gc_pipeline.photometry import cataloging as _cat
from jwst_gc_pipeline.photometry.cataloging import (
    perframe_detector_token, perframe_marker_path, select_resumable_frames,
    stamp_resumed_markers)

FRAME = 'jw03523005001_10101_00001_nrcb3_align_o005_crf.fits'


def _setup(tmp_path):
    """A real frame and an empty marker directory."""
    d = tmp_path / 'catalogs' / '_perframe_markers'
    d.mkdir(parents=True)
    frame = tmp_path / 'pipeline' / FRAME
    frame.parent.mkdir()
    frame.touch()
    frame = str(frame)
    return d, frame


def _marker(d, frame, kind, merge):
    return perframe_marker_path(str(d), frame, perframe_detector_token(frame),
                                'f150w', 'm3', kind, merge=merge)


def test_the_resume_stamps_this_labels_ok_marker(tmp_path):
    """The #841 case end to end: `merged` resumes on the `nrca` receipt, and the
    stamp leaves the `merged` receipt the per-label completeness check asks
    for."""
    d, frame = _setup(tmp_path)
    open(_marker(d, frame, 'ok', 'nrca'), 'w').close()
    todo, ok, nov, _stale = select_resumable_frames(
        [{'filename': frame}], str(d), 'f150w', 'm3', 'merged',
        cross_label=True)
    assert (todo, ok) == ([], [frame])
    assert not os.path.exists(_marker(d, frame, 'ok', 'merged'))
    stamp_resumed_markers(str(d), ok, nov, 'f150w', 'm3', 'merged')
    assert os.path.exists(_marker(d, frame, 'ok', 'merged'))


def test_the_resume_stamps_this_labels_nooverlap_marker(tmp_path):
    """A legitimate no-overlap miss also needs this label's receipt."""
    d, frame = _setup(tmp_path)
    open(_marker(d, frame, 'nooverlap', 'nrca'), 'w').close()
    todo, ok, nov, _stale = select_resumable_frames(
        [{'filename': frame}], str(d), 'f150w', 'm3', 'merged',
        cross_label=True)
    assert (todo, ok, [f for f, _ in nov]) == ([], [], [frame])
    stamp_resumed_markers(str(d), ok, nov, 'f150w', 'm3', 'merged')
    assert os.path.exists(_marker(d, frame, 'nooverlap', 'merged'))
    assert not os.path.exists(_marker(d, frame, 'ok', 'merged'))


def test_it_does_not_clobber_an_existing_marker(tmp_path):
    """The stamp is for what this pass SKIPPED; a marker the fit itself wrote
    must keep its own mtime, which the staleness gate reads."""
    d, frame = _setup(tmp_path)
    p = _marker(d, frame, 'ok', 'merged')
    open(p, 'w').close()
    os.utime(p, (1e9, 1e9))
    stamp_resumed_markers(str(d), [frame], [], 'f150w', 'm3', 'merged')
    assert os.path.getmtime(p) == 1e9


def test_the_stamp_sits_on_the_resume_path_not_the_refit_path():
    """It must be inside the `else:` that runs when skip_if_done is ON -- the
    opt-out branch only prints an offer and refits everything, and stamping
    there would record fits that never happened."""
    src = inspect.getsource(_cat.run_manual_pipeline)
    i_offer = src.index('the resume is opt-in')
    i_stamp = src.index('stamp_resumed_markers(')
    i_extend = src.index('overlapping_now.extend(_ok)')
    assert i_offer < i_stamp < i_extend, (
        'the marker stamp must be on the resume branch, after the opt-in notice'
    )
    assert src.count('stamp_resumed_markers(') == 1
