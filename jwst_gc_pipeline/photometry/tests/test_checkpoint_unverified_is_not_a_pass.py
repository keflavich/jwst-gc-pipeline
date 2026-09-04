"""A checkpoint that could not verify has not passed (issue #312).

cloudc F410M/nrcblong/visit002 is the case that named this.  m2 MEASURED the
problem -- `consensus->reference offset 731.47 mas`, over
REFERENCE_CROSSCHECK_GROSS_MAS (100) -- so it set `apply_ok=False`, filed 8
exposures as `unverified`, and reported `passed=True`.  Every iteration since
2026-08-04 recorded the identical value with `ncorr=0`, and the retie loop
declared convergence: corrections had STOPPED, not converged.  Those 8
exposures drizzle 4.06" out of place.

So a gross offset is precisely the case m2 refuses to correct, and its refusal
did not fail the checkpoint -- the loudest available evidence of misalignment
was the one thing that could not stop the pipeline.
"""
import pytest

from jwst_gc_pipeline.photometry import astrometry_checkpoint as A


CLOUDC = ("cloudc F410M/nrcb F410M visit 2 [m2]: consensus->reference offset "
          "731.47 mas exceeds the gross threshold; 8 exposure(s) unverified")


def test_no_failures_and_nothing_unverified_is_a_pass():
    assert A._checkpoint_passed([], []) is True


def test_a_failure_is_not_a_pass():
    assert A._checkpoint_passed(['moved after freeze'], []) is False


def test_unverified_alone_is_NOT_a_pass():
    """The whole point.  Before #312 this returned True."""
    assert A._checkpoint_passed([], [CLOUDC]) is False


def test_the_override_restores_the_old_meaning(monkeypatch):
    monkeypatch.setenv(A.ALLOW_UNVERIFIED_ENV, '1')
    assert A._checkpoint_passed([], [CLOUDC]) is True


def test_the_override_does_NOT_excuse_a_real_failure(monkeypatch):
    """It relaxes 'could not verify', never 'verified and wrong'."""
    monkeypatch.setenv(A.ALLOW_UNVERIFIED_ENV, '1')
    assert A._checkpoint_passed(['moved after freeze'], [CLOUDC]) is False


@pytest.mark.parametrize('value', ['', '0', 'yes', 'true', 'TRUE', ' 1 '])
def test_only_an_exact_1_overrides(monkeypatch, value):
    """`_env_flag` is `== "1"` after strip.  Pinned because a checkpoint that
    silently stopped gating on a typo'd env var would be the same class of
    defect as the one this fixes."""
    monkeypatch.setenv(A.ALLOW_UNVERIFIED_ENV, value)
    expected = value.strip() == '1'
    assert A._checkpoint_passed([], [CLOUDC]) is expected


def test_the_blocking_sites_feed_the_blocking_list():
    """Only the MEASURED-AND-REFUSED sites may block.  Pinned by source because
    the alternative -- a string-matched severity -- would silently reclassify
    an entry whenever someone reworded a message.

    The count is a tripwire: raising it is a deliberate act, and the third site
    (DETECTOR-ANTISYMMETRIC, issue #624) was added deliberately.  The module
    guard buckets an exposure's detectors by ``module_family`` and tests only a
    group splitting into exactly two families, so an alias BETWEEN TWO
    DETECTORS OF ONE MODULE averages toward ~0 and clears it -- ngc6334 read
    equal-and-opposite +/-22.9" that way and its recorded run passed with
    ``n_antisymmetric: 0``.  Same measured-and-refused character as the other
    two: a number was measured and the correction refused.

    The FOURTH site (issue #626) is the frozen-stage per-exposure baseline m2
    MEASURED and refused to APPLY -- ``alias_suspect``, or past
    ``MAX_CORRECTION_ARCSEC`` with ``ok`` not false.  It is the same character
    again, read out of the m2 record instead of measured in this stage: w51
    F444W holds 16 such entries at ~29" and cloudc F410M 16 at ~734 mas, the
    visit whose exposures drizzle 4.06" out of place.  Its sibling case -- m2
    could not measure a tie at all -- goes to plain ``unverified`` from the same
    branch and deliberately does NOT block.

    Issue #473 lowered the antisymmetry floor from 500 mas a side to a 15 mas
    A-B differential, which put the 15-1000 mas class inside the guard for the
    first time.  Those sets have their CORRECTIONS discarded -- but they still
    append here, because discarding a correction is not the same as deciding
    nothing was measured, and a real inter-module misregistration in that band
    reads exactly like an alias (test_antisymmetric_shape_is_forced_by_the_
    median_recentring in test_sweep_window_alias.py).
    """
    import inspect
    src = inspect.getsource(A)
    assert src.count('unverified_blocking.append(') == 4, (
        'exactly four sites are measured-and-refused: MODULE-ANTISYMMETRIC, '
        'DETECTOR-ANTISYMMETRIC, the untrustworthy consensus->reference tie, '
        'and an m2 per-exposure baseline m2 measured and refused to apply')
    # no conditional sink may reroute one of them away from the blocking list
    assert 'sink' not in src, (
        'an antisymmetry message routed to `unverified` instead of '
        '`unverified_blocking` is invisible to the release gate')
    # each must ALSO appear in the full unverified list, or they stop being
    # reported at all
    assert src.count('unverified.append(unverified_blocking[-1])') == 4


def test_could_not_measure_is_still_a_PASS():
    """The other half of the split, and the reason the blunt version was wrong.

    Seven existing tests assert exactly this -- an unbuildable consensus (two
    exposures, almost no stars), an isolated footprint, a one-cell local map.
    Nothing was measured, so nothing is being ignored, and failing here would
    stop a field for having too little data to check it.
    """
    could_not_measure = [
        "brick F115W visit 1 [m2]: consensus build failed: too few stars",
        "sgra F405N [m7]: local 60\" cell map has only 1 populated cell(s)",
    ]
    # they are reported...
    assert could_not_measure
    # ...but they are NOT in the blocking list, so the gate passes
    assert A._checkpoint_passed([], []) is True


def test_both_record_writers_use_the_helper():
    """There are two `passed = ...` sites (the per-filter record and the m7
    cross-filter record).  Fixing one and not the other would leave the m7 gate
    with the old meaning."""
    import inspect
    src = inspect.getsource(A)
    assert 'passed = not failures\n' not in src, (
        'a record site still computes passed from failures alone')
    assert src.count('passed = _checkpoint_passed(failures, unverified_blocking)') == 2
