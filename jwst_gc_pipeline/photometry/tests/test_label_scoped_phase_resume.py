"""m3..m7 fit a frame differently under each merge label, so the resume stops there.

The cross-label resume (#840/#841) rests on a frame's fit being the same
computation whichever label asked for it: the product is keyed by the DETECTOR
and carries no merge token, so the ``merged`` pass recomputes the ``nrca``
pass's file.

That holds at m12 and at no later phase.  ``frame_args`` carries four values
keyed on the REQUESTED module -- ``prev_seed_catalog``, ``resbg_path``,
``satstar_flux_overrides``, ``satstar_flux_drops`` -- all ``None`` at m12 and
all populated from ``{module}``-named products from m3 on.  Measured on brick
F115W (1182/004, read 2026-09-11):

    phase          nrca     nrcb    merged
    m2 vetted    111126   196255   321234     (merged exceeds nrca + nrcb)
    m3 i2dseed   517025   440116   943427
    m4 i2dseed   226917   335089   557144

plus a per-label ``...-f115w-{nrca,nrcb,merged}_{phase}_..._smoothed_bg_i2d``
from m5 on.  The ``merged`` pass fits each frame against a seed roughly twice
the size of the ``nrca`` pass's, on a different background map.  Resuming that
from the other pass's marker does not save a recomputation, it replaces one fit
with a different one -- and the substitution is silent, because both write the
same filename.
"""
import os

import pytest

from jwst_gc_pipeline.photometry import cataloging
from jwst_gc_pipeline.photometry.cataloging import (
    perframe_cross_label_ok, perframe_marker_path, select_resumable_frames)

FRAME = 'jw01182004001_0210b_00001_nrca1_destreak_o004_crf.fits'
FILT = 'F115W'


def _frame(tmp_path):
    p = tmp_path / FRAME
    p.write_text('stand-in for a crf; only its mtime is read')
    return str(p)


def _mark(marker_dir, fn, merge, phase, kind='ok'):
    p = perframe_marker_path(str(marker_dir), fn, 'nrca1', FILT, phase, kind,
                             merge=merge)
    open(p, 'w').close()
    os.utime(p, (os.path.getmtime(fn) + 10,) * 2)
    return p


# --- the predicate the caller decides with --------------------------------

def test_m12_takes_no_label_scoped_input():
    """All four are None at the first phase -- the passes are one computation."""
    assert perframe_cross_label_ok(None, None, None, None) is True


@pytest.mark.parametrize("slot", range(4))
def test_any_ONE_label_scoped_input_stops_the_cross_label_resume(slot):
    """seed catalog / background map / satstar overrides / satstar drops.

    Each is looked up per requested module, so any one of them present means
    this pass's fit is not the other pass's fit."""
    args = [None, None, None, None]
    args[slot] = 'a {module}-keyed product'
    assert perframe_cross_label_ok(*args) is False


# --- what that does to the selection --------------------------------------

def test_the_merged_pass_does_NOT_resume_an_nrca_marker_at_m3(tmp_path):
    """The defect this guards: at m3 the merged pass's fit uses a seed ~2x the
    nrca pass's, so nrca's receipt is not a receipt for it."""
    md = tmp_path / 'markers'; md.mkdir()
    fn = _frame(tmp_path)
    _mark(md, fn, 'nrca', 'm3')
    todo, ok, _nov, stale = select_resumable_frames(
        [{'filename': fn}], str(md), FILT, 'm3', 'merged', cross_label=False)
    assert ok == [], ("the merged pass resumed a fit the nrca pass made against "
                      "a different seed and background")
    assert [a['filename'] for a in todo] == [fn]
    assert stale == [], 'refused for its label, not for being out of date'


def test_its_OWN_marker_still_resumes_at_m3(tmp_path):
    """The wall-clock resume the flag exists for is untouched at every phase."""
    md = tmp_path / 'markers'; md.mkdir()
    fn = _frame(tmp_path)
    _mark(md, fn, 'merged', 'm3')
    todo, ok, _nov, _stale = select_resumable_frames(
        [{'filename': fn}], str(md), FILT, 'm3', 'merged', cross_label=False)
    assert ok == [fn] and todo == []


def test_a_cross_label_nooverlap_marker_is_refused_too(tmp_path):
    """A cutout miss is a property of the frame, but accepting it here would
    make the rule depend on which marker kind the other pass happened to
    write."""
    md = tmp_path / 'markers'; md.mkdir()
    fn = _frame(tmp_path)
    _mark(md, fn, 'nrcb', 'm5', kind='nooverlap')
    todo, ok, nov, _stale = select_resumable_frames(
        [{'filename': fn}], str(md), FILT, 'm5', 'merged', cross_label=False)
    assert nov == [] and ok == [] and [a['filename'] for a in todo] == [fn]


def test_cross_label_is_OFF_by_default(tmp_path):
    """A caller that has not thought about it gets the conservative rule."""
    md = tmp_path / 'markers'; md.mkdir()
    fn = _frame(tmp_path)
    _mark(md, fn, 'nrca', 'm3')
    todo, ok, _nov, _stale = select_resumable_frames(
        [{'filename': fn}], str(md), FILT, 'm3', 'merged')
    assert ok == [] and [a['filename'] for a in todo] == [fn]


# --- and that the production caller actually asks the predicate ------------

def test_the_fan_out_passes_the_PREDICATE_not_a_constant():
    """A helper-only test would pass with the call site hard-coding True.

    The fan-out is inside `run_manual_pipeline`, which cannot be called from a
    unit test, so pin the call site itself: it must hand `select_resumable_frames`
    the predicate's result, computed from the four values it is about to put in
    `frame_args`.
    """
    src = open(cataloging.__file__).read()
    call = src.split('_todo, _ok, _nov, _stale = select_resumable_frames(')[1]
    call = call.split(')\n')[0]
    assert 'cross_label=_cross_label' in call, (
        'the per-frame fan-out no longer passes a computed cross_label')
    assert 'cross_label=True' not in src, (
        'a hard-coded cross_label=True would resume m3..m7 across labels again')
    decide = src.split('_cross_label = ')[1].split('\n\n')[0]
    assert decide.startswith('perframe_cross_label_ok('), (
        'the fan-out decides cross-label resume some other way now'
    )
    for name in ('prev_seed', 'resbg_path', 'satstar_overrides',
                 'satstar_drops'):
        assert name in decide, f'{name} no longer reaches the decision'
