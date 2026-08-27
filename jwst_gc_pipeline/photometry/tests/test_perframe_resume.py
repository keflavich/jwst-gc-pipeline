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
    perframe_detector_token, perframe_legacy_detector_token,
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


def _args(tmp_path, frames):
    """frame_args with the frames REALLY on disk, as production has them.

    The resume compares each marker against its frame's mtime (#570), so a test
    whose "frames" are bare strings with nothing behind them cannot exercise it.
    Full paths, too: production passes paths, and a fixture that passes bare
    names is the same gap that hid the #562 token bug.
    """
    d = tmp_path / 'pipeline'
    d.mkdir(exist_ok=True)
    out = []
    for f in frames:
        p = d / f
        p.touch()
        out.append(str(p))
    return [{'filename': f} for f in out]


def _bn(x):
    """Basenames, for comparing results against the bare FRAMES names."""
    if isinstance(x, str):
        return os.path.basename(x)
    return [os.path.basename(i) for i in x]


def _age_frame(path, seconds):
    """Make a frame LOOK regenerated: push its mtime `seconds` into the future
    relative to now, so an already-written marker is older than it."""
    t = os.path.getmtime(path) + seconds
    os.utime(path, (t, t))


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
    todo, ok, nov, _stale = select_resumable_frames(frame_args, marker_dir,
                                                   filt, phase, merge)
    return todo, ok, [f for f, _ in nov]


def test_marked_frames_are_not_refitted(tmp_path):
    """The whole point: a wall-clock casualty must not redo finished work."""
    d = _marker_dir(tmp_path)
    args = _args(tmp_path, FRAMES)
    for f in FRAMES[:3]:
        (d / _marker_name(f, 'f212n', perframe_detector_token(f), 'm12', merge='nrca')).touch()
    todo, done, _ = _select(args, str(d), 'f212n', 'm12', resume=True)
    assert len(done) == 3
    assert _bn([a['filename'] for a in todo]) == [FRAMES[3]]


def test_resumed_frames_still_count_as_present(tmp_path):
    """They must land in `overlapping_now`, or the finalize completeness check
    would call a frame this run legitimately skipped a DROPPED exposure and
    hard-crash -- the guard that exists to catch real drops."""
    d = _marker_dir(tmp_path)
    args = _args(tmp_path, FRAMES)
    for f in FRAMES:
        (d / _marker_name(f, 'f212n', perframe_detector_token(f), 'm12', merge='nrca')).touch()
    todo, done, _ = _select(args, str(d), 'f212n', 'm12', resume=True)
    assert todo == []
    assert set(_bn(done)) == set(FRAMES)


def test_a_nooverlap_marker_is_also_a_resume(tmp_path):
    """A frame that legitimately missed the cutout is finished business too."""
    d = _marker_dir(tmp_path)
    args = _args(tmp_path, FRAMES)
    (d / _marker_name(FRAMES[0], 'f212n', perframe_detector_token(FRAMES[0]), 'm12',
                      'nooverlap', merge='nrca')).touch()
    todo, done, nov = _select(args, str(d), 'f212n', 'm12', resume=True)
    assert _bn(nov) == [FRAMES[0]] and done == []
    assert len(todo) == 3


def test_without_the_flag_nothing_is_skipped(tmp_path):
    """Default behaviour is unchanged: a normal run refits regardless of what
    a previous run left behind."""
    d = _marker_dir(tmp_path)
    args = _args(tmp_path, FRAMES)
    for f in FRAMES:
        (d / _marker_name(f, 'f212n', perframe_detector_token(f), 'm12', merge='nrca')).touch()
    todo, done, _ = _select(args, str(d), 'f212n', 'm12', resume=False)
    assert len(todo) == len(FRAMES) and done == []


def test_a_marker_for_a_DIFFERENT_phase_does_not_resume(tmp_path):
    """m12's marker must not let m3 skip the frame."""
    d = _marker_dir(tmp_path)
    args = _args(tmp_path, FRAMES)
    for f in FRAMES:
        (d / _marker_name(f, 'f212n', perframe_detector_token(f), 'm12', merge='nrca')).touch()
    todo, done, _ = _select(args, str(d), 'f212n', 'm3', resume=True)
    assert len(todo) == len(FRAMES) and done == []


def test_a_marker_for_a_DIFFERENT_filter_does_not_resume(tmp_path):
    d = _marker_dir(tmp_path)
    args = _args(tmp_path, FRAMES)
    for f in FRAMES:
        (d / _marker_name(f, 'f212n', perframe_detector_token(f), 'm12', merge='nrca')).touch()
    todo, done, _ = _select(args, str(d), 'f480m', 'm12', resume=True)
    assert len(todo) == len(FRAMES) and done == []


def test_the_marker_is_keyed_by_DETECTOR_not_module(tmp_path):
    """`module='merged'` spans detectors, so a per-module key would collide 8
    frames onto one marker and resume seven that were never fitted."""
    d = _marker_dir(tmp_path)
    args = _args(tmp_path, FRAMES)
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


def test_the_merged_pass_does_not_resume_on_a_PER_MODULE_marker(tmp_path):
    """THE defect this scoping exists for.  `merged` fits the SAME files as
    nrca/nrcb with the same detector token, so a detector-keyed marker was
    written three times under one name -- 1440 on the live sgrb2 tree,
    saturated at 10 detectors x 144 and NOT growing while the merged pass ran.
    Resuming on that would skip every merged frame whose nrca pass finished and
    silently produce no merged catalog."""
    import tempfile
    d = tempfile.mkdtemp()
    args = _args(tmp_path, FRAMES)
    for f in FRAMES:
        (open(os.path.join(d, _marker_name(f, 'f212n', perframe_detector_token(f), 'm12',
                                           merge='nrca')), 'w').close())
    todo, done, _ = _select(args, d, 'f212n', 'm12', resume=True, merge='merged')
    assert done == [], 'a per-module marker must not resume the merged pass'
    assert len(todo) == len(FRAMES)


def test_an_UNSCOPED_legacy_marker_does_not_resume_either(tmp_path):
    """A marker with no merge label cannot say which pass wrote it.  The
    completeness check still honours those; the RESUME deliberately does not --
    re-fitting is cheap, a silently absent merged catalog is not."""
    import tempfile
    d = tempfile.mkdtemp()
    args = _args(tmp_path, FRAMES)
    for f in FRAMES:
        open(os.path.join(d, _marker_name(f, 'f212n', perframe_detector_token(f), 'm12')),
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


# ---------------------------------------------------------------------------
# #562 -- the detector token was taken from the RAW value the caller held, and
# callers hold full paths.  A field directory whose own name contains an
# underscore then contributes one to the split and shifts every index by one, so
# the token became the EXPOSURE number.  Both spellings are on the gc2211_o049
# tree, 16 markers under each.
# ---------------------------------------------------------------------------

#: A real affected path: `gc2211_o049` carries the extra underscore.
AFFECTED = ('/orange/adamginsburg/jwst/gc2211_o049/F277W/pipeline/'
            'jw02211049001_02201_00001_nrcalong_destreak_o049_crf.fits')

#: The same shape under a field directory with no underscore, where the bug is
#: invisible -- which is why brick / w51 / sgrb2 never showed it.
UNAFFECTED = ('/orange/adamginsburg/jwst/brick/F410M/pipeline/'
              'jw02221001001_02201_00001_nrcalong_destreak_o001_crf.fits')


def test_the_detector_token_is_the_DETECTOR_not_the_exposure():
    """The defect itself: '00001' is an exposure number, not a detector."""
    from jwst_gc_pipeline.photometry.cataloging import perframe_detector_token
    assert perframe_detector_token(AFFECTED) == 'nrcalong'
    assert perframe_detector_token(UNAFFECTED) == 'nrcalong'


def test_a_full_path_and_its_basename_agree():
    """Callers pass either; the marker must not depend on which."""
    from jwst_gc_pipeline.photometry.cataloging import perframe_detector_token
    assert (perframe_detector_token(AFFECTED)
            == perframe_detector_token(os.path.basename(AFFECTED)))


def test_the_marker_name_no_longer_depends_on_the_field_directory():
    """Two trees, same frame, same marker."""
    from jwst_gc_pipeline.photometry.cataloging import perframe_detector_token
    a = _marker_name(AFFECTED, 'f277w',
                     perframe_detector_token(AFFECTED),
                     'm12', merge='nrca')
    b = _marker_name(os.path.basename(AFFECTED), 'f277w', 'nrcalong', 'm12',
                     merge='nrca')
    assert a == b == (os.path.basename(AFFECTED)
                      + '.f277w.nrca-nrcalong.m12.ok')


def test_the_legacy_spelling_is_only_generated_where_it_differed():
    """On an unaffected tree there is one token to try, not two -- otherwise
    every reader does double the stat() calls for nothing."""
    from jwst_gc_pipeline.photometry.cataloging import (
        _perframe_detector_tokens)
    assert _perframe_detector_tokens(UNAFFECTED) == ('nrcalong',)
    assert _perframe_detector_tokens(AFFECTED) == ('nrcalong', '00001')


def _affected_frame(tmp_path):
    """A REAL frame under a directory whose name carries an underscore.

    The resume compares marker mtime against the frame's, so the frame has to
    exist.  It must also be a path a caller would really hold: the constant
    `AFFECTED` is an absolute /orange/... path that happens to exist on the
    reduction host and does not exist in CI, so a test resuming on it passed
    locally by depending on survey data being mounted.  Build it under tmp_path
    instead.

    The legacy token is whatever this particular tmp_path makes it -- pytest's
    directory names contain underscores of their own -- so tests derive it with
    the real function rather than hardcoding a value.
    """
    d = tmp_path / 'gc2211_o049' / 'pipeline'
    d.mkdir(parents=True, exist_ok=True)
    p = d / os.path.basename(AFFECTED)
    p.touch()
    return str(p)


def test_a_PRE_562_marker_still_resumes(tmp_path):
    """The migration guarantee.  Markers already on disk were written with the
    old token; refusing them would refit every frame on the affected trees --
    and, in the finalize, report them as dropped exposures and hard-crash."""
    d = _marker_dir(tmp_path)
    frame = _affected_frame(tmp_path)
    legacy = perframe_legacy_detector_token(frame)
    if legacy == perframe_detector_token(frame):
        pytest.skip('this tmp_path does not shift the split; nothing to migrate')
    (d / _marker_name(frame, 'f277w', legacy, 'm12', merge='nrca')).write_text('')
    todo, ok, _ = _select([{'filename': frame}], str(d), 'f277w', 'm12',
                          resume=True, merge='nrca')
    assert ok == [frame]
    assert todo == []


def test_a_post_562_marker_resumes(tmp_path):
    """The new spelling, same frame."""
    d = _marker_dir(tmp_path)
    frame = _affected_frame(tmp_path)
    (d / _marker_name(frame, 'f277w', perframe_detector_token(frame), 'm12',
                      merge='nrca')).write_text('')
    todo, ok, _ = _select([{'filename': frame}], str(d), 'f277w', 'm12',
                          resume=True, merge='nrca')
    assert ok == [frame]
    assert todo == []


def test_the_legacy_token_stays_MERGE_SCOPED(tmp_path):
    """Accepting the old detector spelling must not also re-admit the unscoped
    name: a marker that cannot say which pass wrote it would let the merged pass
    resume on the nrca pass's work (the 1440-marker collision)."""
    d = _marker_dir(tmp_path)
    frame = _affected_frame(tmp_path)
    legacy = perframe_legacy_detector_token(frame)
    if legacy == perframe_detector_token(frame):
        pytest.skip('this tmp_path does not shift the split; no legacy path here')
    (d / _marker_name(frame, 'f277w', legacy, 'm12', merge='nrca')).write_text('')
    # The frame is REAL and the marker is NEWER than it, so the mtime gate would
    # resume -- it is merge scoping alone that must send this to todo.  With a
    # nonexistent frame this test passed no matter what scoping did.
    todo, ok, _ = _select([{'filename': frame}], str(d), 'f277w', 'm12',
                          resume=True, merge='nrca')
    assert ok == [frame], 'precondition: the marker resumes for its OWN pass'
    todo, ok, _ = _select([{'filename': frame}], str(d), 'f277w', 'm12',
                          resume=True, merge='merged')
    assert ok == []
    assert todo == [{'filename': frame}]


def test_the_WRITER_never_emits_the_legacy_token():
    """Readers accept it; writers must not, or the divergence keeps growing."""
    src = _run_manual_src()
    import re
    writes = re.findall(r'open\(_marker_path\((?:[^()]|\([^()]*\))*\)', src)
    assert len(writes) >= 2, writes
    for w in writes:
        assert 'perframe_detector_token(' in w, (
            f'marker write not using the basename-derived token: {w}')
        assert "split('_')[3]" not in w, w


# ---------------------------------------------------------------------------
# #570 -- a marker records that a frame was fitted once, not WHICH frame.
# Regenerating from _cal rewrites every _crf and leaves every marker in place,
# so an existence test cannot tell a finished frame from one whose only recorded
# fit predates it.  Measured on gc2211_o046 (2026-08-26): 480 markers matching
# the frames on disk sat beside ~720 from 2026-08-11, while every _crf dated
# 2026-08-25.  brick carried 977 of 2400 in the same state.
# ---------------------------------------------------------------------------


def _mark_all(d, args, filt='f212n', phase='m12', merge='nrca'):
    for a in args:
        f = a['filename']
        (d / _marker_name(f, filt, perframe_detector_token(f), phase,
                          merge=merge)).touch()


def test_a_marker_OLDER_than_its_frame_does_not_resume(tmp_path):
    """The regeneration case: the frame was rewritten after it was fitted."""
    d = _marker_dir(tmp_path)
    args = _args(tmp_path, FRAMES)
    _mark_all(d, args)
    _age_frame(args[0]['filename'], 3600)      # frame regenerated an hour later
    todo, done, _ = _select(args, str(d), 'f212n', 'm12', resume=True)
    assert _bn([a['filename'] for a in todo]) == [FRAMES[0]]
    assert FRAMES[0] not in _bn(done)
    assert len(done) == 3


def test_the_stale_frames_are_REPORTED_not_just_refit(tmp_path):
    """A silent refit is indistinguishable from a silent skip of the wrong
    frames.  The count is what tells an operator a regeneration happened."""
    d = _marker_dir(tmp_path)
    args = _args(tmp_path, FRAMES)
    _mark_all(d, args)
    for a in args[:2]:
        _age_frame(a['filename'], 3600)
    todo, ok, nov, stale = select_resumable_frames(args, str(d), 'f212n', 'm12',
                                                   'nrca')
    assert set(_bn(stale)) == set(FRAMES[:2])
    assert set(_bn(a['filename'] for a in todo)) == set(FRAMES[:2])
    assert len(ok) == 2


def test_a_marker_NEWER_than_its_frame_still_resumes(tmp_path):
    """The ordinary case must not regress: fitted after the frame was written."""
    d = _marker_dir(tmp_path)
    args = _args(tmp_path, FRAMES)
    _mark_all(d, args)
    todo, done, _ = _select(args, str(d), 'f212n', 'm12', resume=True)
    assert todo == []
    assert len(done) == len(FRAMES)


def test_a_stale_NOOVERLAP_marker_also_refits(tmp_path):
    """no-overlap is a claim about a specific frame's footprint; a regenerated
    frame may overlap where its predecessor did not."""
    d = _marker_dir(tmp_path)
    args = _args(tmp_path, FRAMES)
    f = args[0]['filename']
    (d / _marker_name(f, 'f212n', perframe_detector_token(f), 'm12',
                      kind='nooverlap', merge='nrca')).touch()
    _age_frame(f, 3600)
    todo, ok, nov, stale = select_resumable_frames(args, str(d), 'f212n', 'm12',
                                                   'nrca')
    assert _bn(stale) == [FRAMES[0]]
    assert nov == []
    assert FRAMES[0] in _bn(a['filename'] for a in todo)


def test_a_LEGACY_spelled_marker_is_gated_too(tmp_path):
    """PR #563 made readers accept the pre-#562 spelling, which WIDENS what the
    resume can match -- including stale markers it previously could not see.
    The gate has to cover that path or #563 makes #570 worse."""
    d = _marker_dir(tmp_path)
    args = _args(tmp_path, FRAMES)
    f = args[0]['filename']
    legacy = perframe_legacy_detector_token(f)
    if legacy == perframe_detector_token(f):
        pytest.skip('this tmp_path does not shift the split; no legacy path here')
    (d / _marker_name(f, 'f212n', legacy, 'm12', merge='nrca')).touch()
    todo, ok, nov, stale = select_resumable_frames(args, str(d), 'f212n', 'm12',
                                                   'nrca')
    assert _bn(ok) == [FRAMES[0]]              # fresh legacy marker resumes
    _age_frame(f, 3600)
    todo, ok, nov, stale = select_resumable_frames(args, str(d), 'f212n', 'm12',
                                                   'nrca')
    assert _bn(stale) == [FRAMES[0]]           # stale legacy marker does NOT
    assert ok == []


def test_a_missing_frame_is_not_treated_as_current(tmp_path):
    """Cannot compare against a frame that is not there; send it to todo and let
    the existing dropped-exposure guard own the case rather than inventing a
    second answer for it."""
    d = _marker_dir(tmp_path)
    args = _args(tmp_path, FRAMES)
    _mark_all(d, args)
    os.remove(args[0]['filename'])
    todo, ok, nov, stale = select_resumable_frames(args, str(d), 'f212n', 'm12',
                                                   'nrca')
    assert FRAMES[0] in _bn(a['filename'] for a in todo)
    assert FRAMES[0] not in _bn(ok)
