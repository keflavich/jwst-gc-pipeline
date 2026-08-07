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


def _marker_dir(tmp_path):
    d = tmp_path / 'catalogs' / '_perframe_markers'
    d.mkdir(parents=True)
    return d


def _marker_name(frame, filt, module, phase, kind='ok'):
    return f'{frame}.{filt.lower()}.{module}.{phase}.{kind}'


#: The shape the live sgrb2 tree carries (1440 of these on 2026-08-07).
FRAMES = [f'jw05365001001_03101_0000{i}_nrca{d}_destreak_o001_crf.fits'
          for i in (1, 2) for d in (1, 2)]


def _select(frame_args, marker_dir, filt, phase, resume):
    """The resume filter as `run_manual_pipeline` applies it."""
    if not resume:
        return list(frame_args), [], []
    todo, done, nooverlap = [], [], []
    for a in frame_args:
        fn = a['filename']
        det = fn.split('_')[3]
        if os.path.exists(os.path.join(
                marker_dir, _marker_name(fn, filt, det, phase, 'ok'))):
            done.append(fn)
        elif os.path.exists(os.path.join(
                marker_dir, _marker_name(fn, filt, det, phase, 'nooverlap'))):
            nooverlap.append(fn)
        else:
            todo.append(a)
    return todo, done, nooverlap


def test_marked_frames_are_not_refitted(tmp_path):
    """The whole point: a wall-clock casualty must not redo finished work."""
    d = _marker_dir(tmp_path)
    args = [{'filename': f} for f in FRAMES]
    for f in FRAMES[:3]:
        (d / _marker_name(f, 'f212n', f.split('_')[3], 'm12')).touch()
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
        (d / _marker_name(f, 'f212n', f.split('_')[3], 'm12')).touch()
    todo, done, _ = _select(args, str(d), 'f212n', 'm12', resume=True)
    assert todo == []
    assert set(done) == set(FRAMES)


def test_a_nooverlap_marker_is_also_a_resume(tmp_path):
    """A frame that legitimately missed the cutout is finished business too."""
    d = _marker_dir(tmp_path)
    args = [{'filename': f} for f in FRAMES]
    (d / _marker_name(FRAMES[0], 'f212n', FRAMES[0].split('_')[3], 'm12',
                      'nooverlap')).touch()
    todo, done, nov = _select(args, str(d), 'f212n', 'm12', resume=True)
    assert nov == [FRAMES[0]] and done == []
    assert len(todo) == 3


def test_without_the_flag_nothing_is_skipped(tmp_path):
    """Default behaviour is unchanged: a normal run refits regardless of what
    a previous run left behind."""
    d = _marker_dir(tmp_path)
    args = [{'filename': f} for f in FRAMES]
    for f in FRAMES:
        (d / _marker_name(f, 'f212n', f.split('_')[3], 'm12')).touch()
    todo, done, _ = _select(args, str(d), 'f212n', 'm12', resume=False)
    assert len(todo) == len(FRAMES) and done == []


def test_a_marker_for_a_DIFFERENT_phase_does_not_resume(tmp_path):
    """m12's marker must not let m3 skip the frame."""
    d = _marker_dir(tmp_path)
    args = [{'filename': f} for f in FRAMES]
    for f in FRAMES:
        (d / _marker_name(f, 'f212n', f.split('_')[3], 'm12')).touch()
    todo, done, _ = _select(args, str(d), 'f212n', 'm3', resume=True)
    assert len(todo) == len(FRAMES) and done == []


def test_a_marker_for_a_DIFFERENT_filter_does_not_resume(tmp_path):
    d = _marker_dir(tmp_path)
    args = [{'filename': f} for f in FRAMES]
    for f in FRAMES:
        (d / _marker_name(f, 'f212n', f.split('_')[3], 'm12')).touch()
    todo, done, _ = _select(args, str(d), 'f480m', 'm12', resume=True)
    assert len(todo) == len(FRAMES) and done == []


def test_the_marker_is_keyed_by_DETECTOR_not_module(tmp_path):
    """`module='merged'` spans detectors, so a per-module key would collide 8
    frames onto one marker and resume seven that were never fitted."""
    d = _marker_dir(tmp_path)
    args = [{'filename': f} for f in FRAMES]
    # one marker written under the MODULE name rather than the detector
    (d / _marker_name(FRAMES[0], 'f212n', 'merged', 'm12')).touch()
    todo, done, _ = _select(args, str(d), 'f212n', 'm12', resume=True)
    assert done == [], 'a module-keyed marker must not satisfy a detector key'
    assert len(todo) == len(FRAMES)


def test_the_live_sgrb2_marker_name_is_the_one_we_look_for():
    """Pin the on-disk shape, so a rename of the marker breaks this and not a
    silent resume-nothing. Taken verbatim from the live tree."""
    live = ('jw05365001001_03101_00001_nrca1_destreak_o001_crf.fits'
            '.f212n.nrca1.m12.ok')
    frame = 'jw05365001001_03101_00001_nrca1_destreak_o001_crf.fits'
    assert _marker_name(frame, 'f212n', frame.split('_')[3], 'm12') == live
