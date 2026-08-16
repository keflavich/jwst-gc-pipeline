"""Stage-1/2 member scoping + idempotence (#417).

A fresh SKIP=0 NIRCam reduction ran Detector1+Image2 three times per exposure:
the stage-1/2 loop filtered asn members only on ``'_nrc'`` while ``__main__``
runs ``main()`` once per module (nrca, nrcb, merged) with the same
``skip_step1and2``.  These tests pin both halves of the fix:

* the pure predicates (``member_in_stage12_pass``, ``stage12_products_fresh``)
  -- a selection table over a fake asn member list, and a freshness truth
  table over a tmpdir;
* the driver wiring -- source inspection of ``PipelineRerunNIRCAM-LONG.py``
  (the repo idiom for this driver, as in ``test_crf_source_branch_order.py``:
  driving ``main()`` needs MAST, CRDS and real ramps), asserting the stage-1/2
  loop consults both predicates before ``Detector1Pipeline.call``.

Reverting the module scoping or the freshness skip in the driver fails a
wiring test; reverting either predicate's semantics fails a table test.
"""
import ast
import os
import pathlib

import pytest

from jwst_gc_pipeline.reduction.stage12_selection import (
    member_in_stage12_pass, stage12_products_fresh)

SRC = (pathlib.Path(__file__).resolve().parents[1]
       / "PipelineRerunNIRCAM-LONG.py")

# A 10678-shaped member list: 8 SW detectors + 2 LW detectors + a NIRISS
# member (the loop's existing '_nrc' guard exists because same-obs NIRISS
# asns were seen; any non-NIRCam expname exercises it).
SW_A = [f'jw10678001001_02101_00001_nrca{i}_cal.fits' for i in (1, 2, 3, 4)]
SW_B = [f'jw10678001001_02101_00001_nrcb{i}_cal.fits' for i in (1, 2, 3, 4)]
LW_A = ['jw10678001001_02101_00001_nrcalong_cal.fits']
LW_B = ['jw10678001001_02101_00001_nrcblong_cal.fits']
NON_NIRCAM = ['jw10678001001_02101_00001_nis_cal.fits']
ALL_NIRCAM = SW_A + LW_A + SW_B + LW_B


# ---------------------------------------------------------------------------
# (b) member selection
# ---------------------------------------------------------------------------

def test_nrca_pass_claims_exactly_the_a_module_members():
    selected = [e for e in ALL_NIRCAM + NON_NIRCAM
                if member_in_stage12_pass(e, 'nrca')]
    assert selected == SW_A + LW_A


def test_nrcb_pass_claims_exactly_the_b_module_members():
    selected = [e for e in ALL_NIRCAM + NON_NIRCAM
                if member_in_stage12_pass(e, 'nrcb')]
    assert selected == SW_B + LW_B


def test_nrcalong_belongs_to_the_nrca_pass():
    """The LW A detector is 'nrcalong'; substring containment puts it in the
    nrca pass, matching the later-stage asn member trim."""
    assert member_in_stage12_pass(LW_A[0], 'nrca')
    assert not member_in_stage12_pass(LW_A[0], 'nrcb')
    assert member_in_stage12_pass(LW_B[0], 'nrcb')
    assert not member_in_stage12_pass(LW_B[0], 'nrca')


def test_module_passes_partition_the_nircam_members():
    """Every NIRCam member is claimed by exactly one of the nrca/nrcb passes,
    so the union across module passes covers all members exactly once."""
    for expname in ALL_NIRCAM:
        claims = [m for m in ('nrca', 'nrcb')
                  if member_in_stage12_pass(expname, m)]
        assert len(claims) == 1, (expname, claims)


def test_selection_matches_the_later_stage_member_trim():
    """The tweakreg block trims members with ``f'{module}' in expname``; the
    stage-1/2 scoping must reproduce that expression exactly."""
    for module in ('nrca', 'nrcb'):
        later_trim = [e for e in ALL_NIRCAM if f'{module}' in e]
        stage12 = [e for e in ALL_NIRCAM
                   if member_in_stage12_pass(e, module)]
        assert stage12 == later_trim


def test_merged_pass_keeps_every_nircam_member():
    """A merged-only (or single-module) run must be able to produce every
    _cal; freshness, tested below, is what makes the default three-pass
    sequence cheap."""
    selected = [e for e in ALL_NIRCAM + NON_NIRCAM
                if member_in_stage12_pass(e, 'merged')]
    assert selected == ALL_NIRCAM


def test_non_nircam_members_are_claimed_by_no_pass():
    for module in ('nrca', 'nrcb', 'merged'):
        assert not member_in_stage12_pass(NON_NIRCAM[0], module)


# ---------------------------------------------------------------------------
# (a) freshness truth table
# ---------------------------------------------------------------------------

UNCAL_T = 1_000_000.0


def _touch(path, mtime):
    path.write_bytes(b'x')
    os.utime(path, (mtime, mtime))


def _member(tmp_path, uncal=UNCAL_T, cal=UNCAL_T + 60, ramp=UNCAL_T + 30):
    """Lay down an uncal/cal/ramp triple; None omits that file."""
    stem = 'jw10678001001_02101_00001_nrcalong'
    paths = {suffix: tmp_path / f'{stem}_{suffix}.fits'
             for suffix in ('uncal', 'cal', 'ramp')}
    for suffix, mtime in (('uncal', uncal), ('cal', cal), ('ramp', ramp)):
        if mtime is not None:
            _touch(paths[suffix], mtime)
    return str(paths['uncal'])


def test_fresh_products_skip(tmp_path):
    assert stage12_products_fresh(_member(tmp_path)) is True


def test_missing_cal_reprocesses(tmp_path):
    assert stage12_products_fresh(_member(tmp_path, cal=None)) is False


def test_missing_ramp_reprocesses(tmp_path):
    """Detector1 succeeded (or its ramp was cleaned) with no cal counterpart
    -- and vice versa half-states -- must rerun both stages."""
    assert stage12_products_fresh(_member(tmp_path, ramp=None)) is False


def test_stale_cal_reprocesses(tmp_path):
    """A re-downloaded uncal newer than the cal invalidates it."""
    assert stage12_products_fresh(
        _member(tmp_path, cal=UNCAL_T - 60)) is False


def test_stale_ramp_reprocesses(tmp_path):
    assert stage12_products_fresh(
        _member(tmp_path, ramp=UNCAL_T - 60)) is False


def test_equal_mtime_reprocesses(tmp_path):
    """'Newer' is strict: an mtime tie reprocesses (wasted compute beats a
    silently kept maybe-stale product)."""
    assert stage12_products_fresh(_member(tmp_path, cal=UNCAL_T)) is False


def test_missing_uncal_with_both_products_skips(tmp_path):
    """Without the uncal, reprocessing is impossible; two present products
    count as current instead of crashing Detector1 on the absent input."""
    assert stage12_products_fresh(_member(tmp_path, uncal=None)) is True


def test_missing_uncal_and_missing_product_reprocesses(tmp_path):
    """The member proceeds to Detector1, which fails loudly on the missing
    uncal -- the pre-#417 behavior for this broken state."""
    assert stage12_products_fresh(
        _member(tmp_path, uncal=None, ramp=None)) is False


def test_wrong_suffix_raises():
    with pytest.raises(ValueError, match='_uncal'):
        stage12_products_fresh('jw10678001001_02101_00001_nrcalong_cal.fits')


# ---------------------------------------------------------------------------
# (c) driver wiring
# ---------------------------------------------------------------------------

def _stage12_loop():
    """The member loop inside the driver's ``if not skip_step1and2:`` block."""
    tree = ast.parse(SRC.read_text())
    for node in ast.walk(tree):
        if (isinstance(node, ast.If)
                and isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not)
                and isinstance(node.test.operand, ast.Name)
                and node.test.operand.id == 'skip_step1and2'):
            fors = [n for n in ast.walk(node) if isinstance(n, ast.For)]
            if fors:
                return fors[0]
    raise AssertionError(
        "could not find the stage-1/2 member loop under 'if not skip_step1and2:'")


def _calls_named(node, name):
    """Calls to ``name(...)`` (plain) inside ``node``."""
    return [n for n in ast.walk(node)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name) and n.func.id == name]


def _detector1_call(node):
    calls = [n for n in ast.walk(node)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == 'call'
             and isinstance(n.func.value, ast.Name)
             and n.func.value.id == 'Detector1Pipeline']
    assert len(calls) == 1, "expected exactly one Detector1Pipeline.call in the loop"
    return calls[0]


def test_stage12_loop_scopes_members_to_the_module_pass():
    """Reverting the module scoping restores the 3x Detector1 reruns."""
    loop = _stage12_loop()
    scoping = _calls_named(loop, 'member_in_stage12_pass')
    assert scoping, (
        "the stage-1/2 loop does not consult member_in_stage12_pass: every "
        "module pass will ramp-fit every member again (#417)")
    assert min(c.lineno for c in scoping) < _detector1_call(loop).lineno


def test_stage12_loop_skips_members_with_fresh_products():
    """Reverting the idempotence check makes retries and the merged pass
    re-run Detector1+Image2 on already-produced members."""
    loop = _stage12_loop()
    freshness = _calls_named(loop, 'stage12_products_fresh')
    assert freshness, (
        "the stage-1/2 loop does not consult stage12_products_fresh: a retry "
        "after a partial failure reprocesses every member (#417)")
    assert min(c.lineno for c in freshness) < _detector1_call(loop).lineno


def test_each_skip_is_logged():
    """Both skip paths announce themselves at the loop's print verbosity."""
    src = SRC.read_text()
    assert "pass does not claim it" in src
    assert "Skipping stage 1+2 for" in src
