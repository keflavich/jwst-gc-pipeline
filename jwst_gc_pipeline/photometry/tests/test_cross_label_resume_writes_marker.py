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
import re

from jwst_gc_pipeline.photometry import cataloging as _cat


def _resume_branch():
    """The source of the skip-if-done branch that consumes `_ok`/`_nov`."""
    src = inspect.getsource(_cat.run_manual_pipeline)
    i = src.index('overlapping_now.extend(_ok)')
    return src[max(0, i - 2500):i + 200]


def test_the_resume_stamps_this_labels_ok_marker():
    branch = _resume_branch()
    assert re.search(r"for _rfn in _ok:", branch), (
        'resumed frames must be stamped with a marker under the CURRENT merge '
        'label, or the per-label completeness check reads them as dropped')
    assert re.search(r"perframe_marker_path\([^)]*merge=module", branch, re.S)
    assert "'ok'" in branch


def test_the_resume_stamps_this_labels_nooverlap_marker():
    branch = _resume_branch()
    assert re.search(r"for _rfn, _rreason in _nov:", branch), (
        'a legitimate no-overlap miss also needs this label\'s receipt'
    )
    assert "'nooverlap'" in branch


def test_it_does_not_clobber_an_existing_marker():
    """The stamp is for what this pass SKIPPED; a marker the fit itself wrote
    must keep its own mtime, which the staleness gate reads."""
    branch = _resume_branch()
    assert branch.count('if not os.path.exists(_rp):') >= 2


def test_the_stamp_sits_on_the_resume_path_not_the_refit_path():
    """It must be inside the `else:` that runs when skip_if_done is ON -- the
    opt-out branch only prints an offer and refits everything, and stamping
    there would record fits that never happened."""
    src = inspect.getsource(_cat.run_manual_pipeline)
    i_offer = src.index('the resume is opt-in')
    i_stamp = src.index('for _rfn in _ok:')
    assert i_stamp > i_offer, (
        'the marker stamp must be on the resume branch, after the opt-in notice'
    )
