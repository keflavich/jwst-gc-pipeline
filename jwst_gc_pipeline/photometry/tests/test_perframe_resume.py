"""A timed-out fan-out must resume from its completion markers (#333).

`--skip-if-done` predicted output FILENAMES, and the manual per-frame pipeline
never consulted it at all -- `run_manual_pipeline` has no reference to
`skip_if_done`, and the legacy call site that does check it is unreachable on a
manual run (`crowdsource_catalogs_long.py:4849` returns into
`run_manual_pipeline` first).  So a fan-out that hit its wall redid every frame.

The resume key is the per-frame completion marker the fan-out already writes:

    <basepath>/catalogs/_perframe_markers/<frame>.<filt>.<module>.<phase>.ok

written by `_on_result` on the worker that SUCCEEDED.  Unlike a predicted
filename it cannot drift from what the writer does.
"""
import os

import pytest

from jwst_gc_pipeline.photometry.cataloging import (
    perframe_marker_path, select_resumable_frames)


def _marker_dir(tmp_path):
    d = tmp_path / 'catalogs' / '_perframe_markers'
    d.mkdir(parents=True)
    return d


def _marker_name(frame, filt, detector, phase, kind='ok', merge=None):
    """The REAL builder, not a copy of it.  A second format here is how the
    previous round shipped two call sites disagreeing about the same frames."""
    return os.path.basename(
        perframe_marker_path('', frame, detector, filt, phase, kind, merge))


#: The shape the live sgrb2 tree carries (1440 of these on 2026-08-07).
FRAMES = [f'jw05365001001_03101_0000{i}_nrca{d}_destreak_o001_crf.fits'
          for i in (1, 2) for d in (1, 2)]


def _select(frame_args, marker_dir, filt, phase, resume, merge='nrca'):
    """`run_manual_pipeline`'s own selection, imported rather than restated.

    `resume` mirrors the caller's `skip_if_done and (skip_finalize or
    finalize_only)` gate, which is the only part that stays here.  Everything
    below it -- the marker format, the ok/nooverlap/todo split, the merge
    scoping -- comes from cataloging.py, so a change there that breaks the
    resume breaks these tests.
    """
    if not resume:
        return list(frame_args), [], []
    todo, ok, nov = select_resumable_frames(frame_args, marker_dir, filt,
                                            phase, merge)
    return todo, ok, [f for f, _ in nov]


def test_marked_frames_are_not_refitted(tmp_path):
    """The whole point: a wall-clock casualty must not redo finished work."""
    d = _marker_dir(tmp_path)
    args = [{'filename': f} for f in FRAMES]
    for f in FRAMES[:3]:
        (d / _marker_name(f, 'f212n', f.split('_')[3], 'm12', merge='nrca')).touch()
    todo, done, _ = _select(args, str(d), 'f212n', 'm12', resume=True)
    assert len(done) == 3
    assert [a['filename'] for a in todo] == [FRAMES[3]]


def test_resumed_frames_still_count_as_present(tmp_path):
    """They must land in `overlapping_now`, or the finalize completeness check
    would call a frame this run legitimately skipped a DROPPED exposure and
    hard-crash -- the guard that exists to catch real drops."""
    d = _marker_dir(tmp_path)
    args = [{'filename': f} for f in FRAMES]
    for f in FRAMES:
        (d / _marker_name(f, 'f212n', f.split('_')[3], 'm12', merge='nrca')).touch()
    todo, done, _ = _select(args, str(d), 'f212n', 'm12', resume=True)
    assert todo == []
    assert set(done) == set(FRAMES)


def test_a_nooverlap_marker_is_also_a_resume(tmp_path):
    """A frame that legitimately missed the cutout is finished business too."""
    d = _marker_dir(tmp_path)
    args = [{'filename': f} for f in FRAMES]
    (d / _marker_name(FRAMES[0], 'f212n', FRAMES[0].split('_')[3], 'm12',
                      'nooverlap', merge='nrca')).touch()
    todo, done, nov = _select(args, str(d), 'f212n', 'm12', resume=True)
    assert nov == [FRAMES[0]] and done == []
    assert len(todo) == 3


def test_without_the_flag_nothing_is_skipped(tmp_path):
    """Default behaviour is unchanged: a normal run refits regardless of what
    a previous run left behind."""
    d = _marker_dir(tmp_path)
    args = [{'filename': f} for f in FRAMES]
    for f in FRAMES:
        (d / _marker_name(f, 'f212n', f.split('_')[3], 'm12', merge='nrca')).touch()
    todo, done, _ = _select(args, str(d), 'f212n', 'm12', resume=False)
    assert len(todo) == len(FRAMES) and done == []


def test_a_marker_for_a_DIFFERENT_phase_does_not_resume(tmp_path):
    """m12's marker must not let m3 skip the frame."""
    d = _marker_dir(tmp_path)
    args = [{'filename': f} for f in FRAMES]
    for f in FRAMES:
        (d / _marker_name(f, 'f212n', f.split('_')[3], 'm12', merge='nrca')).touch()
    todo, done, _ = _select(args, str(d), 'f212n', 'm3', resume=True)
    assert len(todo) == len(FRAMES) and done == []


def test_a_marker_for_a_DIFFERENT_filter_does_not_resume(tmp_path):
    d = _marker_dir(tmp_path)
    args = [{'filename': f} for f in FRAMES]
    for f in FRAMES:
        (d / _marker_name(f, 'f212n', f.split('_')[3], 'm12', merge='nrca')).touch()
    todo, done, _ = _select(args, str(d), 'f480m', 'm12', resume=True)
    assert len(todo) == len(FRAMES) and done == []


def test_the_marker_is_keyed_by_DETECTOR_not_module(tmp_path):
    """`module='merged'` spans detectors, so a per-module key would collide 8
    frames onto one marker and resume seven that were never fitted."""
    d = _marker_dir(tmp_path)
    args = [{'filename': f} for f in FRAMES]
    # one marker written under the MODULE name rather than the detector
    (d / _marker_name(FRAMES[0], 'f212n', 'merged', 'm12', merge='nrca')).touch()
    todo, done, _ = _select(args, str(d), 'f212n', 'm12', resume=True)
    assert done == [], 'a module-keyed marker must not satisfy a detector key'
    assert len(todo) == len(FRAMES)


def test_the_legacy_unscoped_name_is_still_expressible():
    """The completeness check keeps reading pre-existing markers, so the
    unscoped form must still be constructible. Verbatim from the live tree."""
    live = ('jw05365001001_03101_00001_nrca1_destreak_o001_crf.fits'
            '.f212n.nrca1.m12.ok')
    frame = 'jw05365001001_03101_00001_nrca1_destreak_o001_crf.fits'
    assert _marker_name(frame, 'f212n', frame.split('_')[3], 'm12') == live


def test_the_merged_pass_does_not_resume_on_a_PER_MODULE_marker():
    """THE defect this scoping exists for.  `merged` fits the SAME files as
    nrca/nrcb with the same detector token, so a detector-keyed marker was
    written three times under one name -- 1440 on the live sgrb2 tree,
    saturated at 10 detectors x 144 and NOT growing while the merged pass ran.
    Resuming on that would skip every merged frame whose nrca pass finished and
    silently produce no merged catalog."""
    import tempfile
    d = tempfile.mkdtemp()
    args = [{'filename': f} for f in FRAMES]
    for f in FRAMES:
        (open(os.path.join(d, _marker_name(f, 'f212n', f.split('_')[3], 'm12',
                                           merge='nrca')), 'w').close())
    todo, done, _ = _select(args, d, 'f212n', 'm12', resume=True, merge='merged')
    assert done == [], 'a per-module marker must not resume the merged pass'
    assert len(todo) == len(FRAMES)


def test_an_UNSCOPED_legacy_marker_does_not_resume_either():
    """A marker with no merge label cannot say which pass wrote it.  The
    completeness check still honours those; the RESUME deliberately does not --
    re-fitting is cheap, a silently absent merged catalog is not."""
    import tempfile
    d = tempfile.mkdtemp()
    args = [{'filename': f} for f in FRAMES]
    for f in FRAMES:
        open(os.path.join(d, _marker_name(f, 'f212n', f.split('_')[3], 'm12')),
             'w').close()
    todo, done, _ = _select(args, d, 'f212n', 'm12', resume=True, merge='nrca')
    assert done == []
    assert len(todo) == len(FRAMES)


# ---------------------------------------------------------------------------
# CALL-SITE guards.  The previous round's lesson one level up: extracting
# `select_resumable_frames` fixed the format copy, but the tests still restated
# the WIRING, and four production lines each mutated alone left 38 passed --
#
#   delete  overlapping_now.extend(_ok)
#   delete  frame_args = _todo                  <- resume computes, then refits
#   resume  select_resumable_frames(..., None)  <- merged resumes on nrca markers
#   writer  _marker_path(..., 'ok') without merge=module
#
# A behavioural test needs the whole fan-out; the surrounding tests use a source
# guard for exactly that reason (test_perframe_helpers.py:101).
# ---------------------------------------------------------------------------

def _run_manual_src():
    import inspect
    from jwst_gc_pipeline.photometry import cataloging
    return inspect.getsource(cataloging.run_manual_pipeline)


def test_the_resume_is_called_with_the_MERGE_LABEL_not_None():
    """Passing None resumes the merged pass on markers nrca wrote -- the defect
    the merge scoping exists to prevent, restored verbatim."""
    import re
    src = _run_manual_src()
    calls = re.findall(r'select_resumable_frames\((?:[^()]|\([^()]*\))*\)', src)
    assert calls, 'run_manual_pipeline no longer calls select_resumable_frames'
    for c in calls:
        assert 'module' in c, f'resume must be scoped to the merge label: {c}'
        assert 'None' not in c, f'resume must not be unscoped: {c}'


def test_the_resumed_frames_reach_overlapping_now():
    """Without this the finalize's completeness check calls a legitimately
    skipped frame a DROPPED exposure -- the guard that exists to catch real
    drops -- so the property must be pinned at the CALLER, not just asserted of
    the helper's return value."""
    src = _run_manual_src()
    assert 'overlapping_now.extend(' in src


def test_the_todo_list_actually_replaces_frame_args():
    """The one that matters most: without it the resume computes its three
    lists and then fits every frame anyway -- #333 unfixed, the wall hit again,
    suite green."""
    src = _run_manual_src()
    assert 'frame_args = _todo' in src


def test_both_marker_WRITES_carry_the_merge_label():
    """A writer that drops `merge=module` re-creates the 1440-marker collision:
    three passes over one name, and the merged pass resumes on markers it did
    not write."""
    import re
    src = _run_manual_src()
    # WRITE sites only -- `open(_marker_path(...), 'w')`.  The completeness
    # check READS the unscoped name too, on purpose, so an in-flight finalize
    # still honours the markers already on disk.
    writes = re.findall(r'open\(_marker_path\((?:[^()]|\([^()]*\))*\)', src)
    assert len(writes) >= 2, writes
    for w in writes:
        assert 'merge=' in w, f'marker write without a merge label: {w}'
