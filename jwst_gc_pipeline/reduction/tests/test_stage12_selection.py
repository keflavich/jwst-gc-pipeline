"""Stage-1/2 member scoping + intra-run idempotence (#417).

A fresh SKIP=0 NIRCam reduction ran Detector1+Image2 three times per exposure:
the stage-1/2 loop filtered asn members only on ``'_nrc'`` while ``__main__``
runs ``main()`` once per module (nrca, nrcb, merged) with the same
``skip_step1and2``.  These tests pin the fix:

* the pure predicates (``member_in_stage12_pass``, the in-process memo,
  ``stage12_products_fresh``, ``stage12_skip_reason``) -- a selection table
  over a fake asn member list, a memo table, and a freshness truth table over
  a tmpdir, including that a bare ``SKIP=0`` run in a new process reprocesses
  members whose products are already on disk (only ``STAGE12_RESUME=1`` keeps
  them);
* the driver wiring -- source inspection of ``PipelineRerunNIRCAM-LONG.py``
  (the repo idiom for this driver, as in ``test_crf_source_branch_order.py``:
  driving ``main()`` needs MAST, CRDS and real ramps), asserting the stage-1/2
  loop scopes on the pass's own ``module``, consults the skip reason
  positively before ``Detector1Pipeline.call``, and records each processed
  member afterwards.

Reverting the module scoping or the skip check in the driver fails a wiring
test, as does hardcoding the scoping module or inverting the skip test;
reverting either predicate's semantics fails a table test.
"""
import ast
import os
import pathlib

import pytest

from jwst_gc_pipeline.reduction.stage12_selection import (
    STAGE12_RESUME_ENV, member_in_stage12_pass, note_stage12_processed,
    reset_stage12_processed, stage12_already_processed, stage12_products_fresh,
    stage12_resume_enabled, stage12_skip_reason)

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


@pytest.fixture(autouse=True)
def _clean_memo():
    """Every test starts and ends with an empty in-process memo."""
    reset_stage12_processed()
    yield
    reset_stage12_processed()


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
    _cal; the memo, tested below, is what makes the default three-pass
    sequence cheap."""
    selected = [e for e in ALL_NIRCAM + NON_NIRCAM
                if member_in_stage12_pass(e, 'merged')]
    assert selected == ALL_NIRCAM


def test_non_nircam_members_are_claimed_by_no_pass():
    for module in ('nrca', 'nrcb', 'merged'):
        assert not member_in_stage12_pass(NON_NIRCAM[0], module)


def test_unknown_module_claims_every_member():
    """The nrca/nrcb branch is an allow-list; any other module name (today
    only 'merged') keeps every NIRCam member, so an unrecognised module
    over-processes instead of silently dropping exposures."""
    for expname in ALL_NIRCAM:
        assert member_in_stage12_pass(expname, 'nrcab')
        assert member_in_stage12_pass(expname, '')


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


def test_missing_ramp_is_ignored_when_ramps_are_not_saved(tmp_path):
    """#421 may make ramp retention optional; with ``require_ramp=False`` the
    absent ramp no longer forces every member to be reprocessed forever."""
    assert stage12_products_fresh(_member(tmp_path, ramp=None),
                                  require_ramp=False) is True


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
# (d) in-process memo + what SKIP=0 still means
# ---------------------------------------------------------------------------

def test_memo_is_empty_in_a_new_process(tmp_path):
    assert stage12_already_processed(_member(tmp_path)) is False


def test_memo_records_only_the_noted_member(tmp_path):
    uncal_a = _member(tmp_path)
    uncal_b = str(tmp_path / 'jw10678001001_02101_00001_nrcb1_uncal.fits')
    note_stage12_processed(uncal_a)
    assert stage12_already_processed(uncal_a) is True
    assert stage12_already_processed(uncal_b) is False


def test_memo_is_path_normalised(tmp_path, monkeypatch):
    """The driver chdirs to the per-filter output_dir and the asn expnames are
    bare basenames, so the memo has to survive basename-vs-absolute."""
    uncal = _member(tmp_path)
    note_stage12_processed(uncal)
    monkeypatch.chdir(tmp_path)
    assert stage12_already_processed(os.path.basename(uncal)) is True


def test_memoized_member_is_skipped_in_a_later_module_pass(tmp_path,
                                                           monkeypatch):
    """The merged pass must not redo what the nrca pass just calibrated: this
    is the remaining third of the #417 3x."""
    monkeypatch.delenv(STAGE12_RESUME_ENV, raising=False)
    uncal = _member(tmp_path)
    assert stage12_skip_reason(uncal) is None
    note_stage12_processed(uncal)
    reason = stage12_skip_reason(uncal)
    assert reason and 'earlier module pass' in reason


def test_products_on_disk_alone_do_not_skip(tmp_path, monkeypatch):
    """A NEW process with SKIP=0 reprocesses a member whose _cal and _ramp are
    already on disk and newer than the uncal.  Products are current only with
    respect to the uncal's mtime, so a CRDS repin / jwst bump / parameter
    change leaves them present, newer, and wrong -- exactly what SKIP=0 is
    documented to fix."""
    monkeypatch.delenv(STAGE12_RESUME_ENV, raising=False)
    uncal = _member(tmp_path)
    assert stage12_products_fresh(uncal) is True
    assert stage12_skip_reason(uncal) is None


def test_resume_opt_in_skips_fresh_products(tmp_path, monkeypatch):
    """With the operator asking for a resume, the on-disk check comes back."""
    monkeypatch.setenv(STAGE12_RESUME_ENV, '1')
    uncal = _member(tmp_path)
    reason = stage12_skip_reason(uncal)
    assert reason and STAGE12_RESUME_ENV in reason


def test_resume_opt_in_still_reprocesses_stale_products(tmp_path, monkeypatch):
    monkeypatch.setenv(STAGE12_RESUME_ENV, '1')
    assert stage12_skip_reason(_member(tmp_path, cal=UNCAL_T - 60)) is None


def test_resume_env_parsing(monkeypatch):
    for value in ('1', 'true', 'TRUE', 'yes', 'on'):
        assert stage12_resume_enabled({STAGE12_RESUME_ENV: value}) is True
    for value in ('0', '', 'false', 'no'):
        assert stage12_resume_enabled({STAGE12_RESUME_ENV: value}) is False
    assert stage12_resume_enabled({}) is False


def test_resume_argument_overrides_the_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(STAGE12_RESUME_ENV, '1')
    assert stage12_skip_reason(_member(tmp_path), resume=False) is None


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


def _guard_ifs(loop, call_name):
    """``if [not] call_name(...): ... continue`` guards inside ``loop``."""
    guards = []
    for node in ast.walk(loop):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        negated = isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
        inner = test.operand if negated else test
        if (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
                and inner.func.id == call_name
                and any(isinstance(n, ast.Continue) for n in ast.walk(node))):
            guards.append((node, negated, inner))
    return guards


def test_stage12_loop_scopes_members_to_the_module_pass():
    """Reverting the module scoping restores the 3x Detector1 reruns."""
    loop = _stage12_loop()
    scoping = _calls_named(loop, 'member_in_stage12_pass')
    assert scoping, (
        "the stage-1/2 loop does not consult member_in_stage12_pass: every "
        "module pass will ramp-fit every member again (#417)")
    assert min(c.lineno for c in scoping) < _detector1_call(loop).lineno


def test_scoping_uses_this_pass_s_module():
    """Hardcoding the module (e.g. always 'merged') would keep every member in
    every pass and restore the full 3x while the call still existed."""
    loop = _stage12_loop()
    guards = _guard_ifs(loop, 'member_in_stage12_pass')
    assert len(guards) == 1, "expected one 'if not member_in_stage12_pass(...): continue' guard"
    node, negated, call = guards[0]
    assert negated, "the scoping guard must skip members the pass does NOT claim"
    assert len(call.args) == 2, "member_in_stage12_pass(expname, module)"
    assert isinstance(call.args[1], ast.Name) and call.args[1].id == 'module', (
        "the scoping module must be this pass's 'module' variable, not a constant")


def test_stage12_loop_skips_members_it_already_processed():
    """Reverting the skip check makes the merged pass re-run Detector1+Image2
    on the members the module passes just produced."""
    loop = _stage12_loop()
    skip_calls = _calls_named(loop, 'stage12_skip_reason')
    assert skip_calls, (
        "the stage-1/2 loop does not consult stage12_skip_reason: the merged "
        "pass reprocesses every member the module passes produced (#417)")
    assert min(c.lineno for c in skip_calls) < _detector1_call(loop).lineno


def test_skip_reason_is_used_positively_and_on_the_uncal():
    """An inverted test (``if not stage12_skip_reason(...)``) would skip the
    members that still need processing and process the rest."""
    loop = _stage12_loop()
    calls = _calls_named(loop, 'stage12_skip_reason')
    assert len(calls) == 1
    call = calls[0]
    assert isinstance(call.args[0], ast.Name) and call.args[0].id == 'uncal_fn'
    assigns = [n for n in ast.walk(loop)
               if isinstance(n, ast.Assign) and n.value is call]
    assert len(assigns) == 1, "stage12_skip_reason(...) must be assigned to a name"
    target = assigns[0].targets[0]
    assert isinstance(target, ast.Name)
    positive_guards = [
        n for n in ast.walk(loop)
        if isinstance(n, ast.If) and isinstance(n.test, ast.Name)
        and n.test.id == target.id
        and any(isinstance(c, ast.Continue) for c in ast.walk(n))]
    assert positive_guards, (
        f"expected 'if {target.id}: ... continue' -- a truthy reason means skip, "
        "so the guard must not be negated")


def test_skip_reason_is_told_whether_ramps_are_saved():
    """If #421 stops saving ramps, the resume check must stop requiring one
    rather than returning False forever."""
    loop = _stage12_loop()
    call = _calls_named(loop, 'stage12_skip_reason')[0]
    kwargs = {kw.arg: kw.value for kw in call.keywords}
    assert 'require_ramp' in kwargs, "pass require_ramp so the ramp clause tracks the driver"
    assert isinstance(kwargs['require_ramp'], ast.Name)
    assert kwargs['require_ramp'].id == 'SAVE_CALIBRATED_RAMP'

    d1_kwargs = {kw.arg: kw.value for kw in _detector1_call(loop).keywords}
    assert isinstance(d1_kwargs['save_calibrated_ramp'], ast.Name), (
        "Detector1Pipeline.call's save_calibrated_ramp must be the same flag "
        "the skip check is told about")
    assert d1_kwargs['save_calibrated_ramp'].id == 'SAVE_CALIBRATED_RAMP'


def test_processed_members_are_recorded_after_both_stages():
    """Without the memo write, the later module passes see nothing skipped."""
    loop = _stage12_loop()
    notes = _calls_named(loop, 'note_stage12_processed')
    assert len(notes) == 1, (
        "the stage-1/2 loop must record each processed member so a later "
        "module pass in this same interpreter skips it (#417)")
    assert isinstance(notes[0].args[0], ast.Name)
    assert notes[0].args[0].id == 'uncal_fn'
    assert notes[0].lineno > _detector1_call(loop).lineno, (
        "record the member after it has actually been processed")


def test_provenance_assert_runs_before_the_module_scoping():
    """The whole-asn `jw_prefix(proposal) + field` check must see every member of
    every pass; below the scoping `continue` a single-module-policy field
    (sickle nrcb-only) would never check a foreign, mis-globbed member."""
    loop = _stage12_loop()
    asserts = [n for n in ast.walk(loop) if isinstance(n, ast.Assert)]
    assert asserts, "the whole-asn provenance assert disappeared"
    scoping = _calls_named(loop, 'member_in_stage12_pass')
    assert min(a.lineno for a in asserts) < min(c.lineno for c in scoping)


def test_each_skip_is_logged():
    """Both skip paths announce themselves at the loop's print verbosity."""
    src = SRC.read_text()
    assert "pass does not claim it" in src
    assert "Skipping stage 1+2 for" in src
