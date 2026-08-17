"""Run the NIRCam stage-1/2 loop and count Detector1/Image2 calls (#417, #423).

#417: a fresh ``SKIP=0`` NIRCam reduction ran Detector1+Image2 three times per
exposure, because ``__main__`` calls ``main()`` once per module
(nrca, nrcb, merged) in ONE interpreter and the stage-1/2 block sits inside
every pass.  #423 fixed it with per-pass module scoping plus an in-process memo
of the uncals this interpreter has already calibrated.

``test_stage12_selection.py`` pins that fix by ``ast.parse`` inspection: the
calls exist, in the right order, with the right arguments.  Source assertions
cannot see whether the memo SURVIVES from one module pass to the next, which is
the whole mechanism -- inserting a ``reset_stage12_processed()`` at the top of
the ``if not skip_step1and2:`` block passes that suite while doubling the ramp
fits (192 -> 384 Detector1 calls on brick F212N).  These tests close that gap by
running the loop.

The loop needs neither MAST nor CRDS nor real ramps: ``ast.parse`` the driver,
slice the ``if not skip_step1and2:`` block by ``lineno``/``end_lineno``,
``textwrap.dedent`` it, and ``exec`` it against an in-memory asn with recording
``Detector1Pipeline``/``Image2Pipeline`` stubs.  The block reads only names the
harness supplies, so a driver edit that reaches for a name outside that set
raises ``NameError`` here and says which name it was.

Counts asserted below over a 10678-shaped member list (8 SW + 2 LW detectors),
matching what the same harness measures on the real brick F212N asn
(192 NIRCam members, ``/orange/adamginsburg/jwst/brick/F212N/pipeline/``):

===========================================  ==================  ============
run                                          brick F212N (192)   here (10)
===========================================  ==================  ============
pre-#423 driver, nrca,nrcb,merged            576                 30
shipped driver, nrca,nrcb,merged             192                 10
shipped driver with the memo neutered        384                 20
shipped driver, ``-m merged`` alone          192                 10
shipped driver, ``-m nrca`` alone             96                  5
===========================================  ==================  ============
"""
import ast
import os
import pathlib
import textwrap

import pytest

from jwst_gc_pipeline.reduction import stage12_selection
from jwst_gc_pipeline.reduction.stage12_selection import (
    STAGE12_RESUME_ENV, reset_stage12_processed)

SRC = (pathlib.Path(__file__).resolve().parents[1]
       / "PipelineRerunNIRCAM-LONG.py")

PROPOSAL_ID = '10678'
FIELD = '001'
STEM = f'jw0{PROPOSAL_ID}{FIELD}001_02101_00001'
SW_A = [f'{STEM}_nrca{i}_cal.fits' for i in (1, 2, 3, 4)]
SW_B = [f'{STEM}_nrcb{i}_cal.fits' for i in (1, 2, 3, 4)]
LW_A = [f'{STEM}_nrcalong_cal.fits']
LW_B = [f'{STEM}_nrcblong_cal.fits']
NON_NIRCAM = [f'{STEM}_nis_cal.fits']
A_MEMBERS = SW_A + LW_A
B_MEMBERS = SW_B + LW_B
ALL_NIRCAM = A_MEMBERS + B_MEMBERS


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------

def stage12_block_source():
    """The driver's whole ``if not skip_step1and2:`` block, dedented."""
    text = SRC.read_text()
    lines = text.splitlines()
    for node in ast.walk(ast.parse(text)):
        if (isinstance(node, ast.If)
                and isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not)
                and isinstance(node.test.operand, ast.Name)
                and node.test.operand.id == 'skip_step1and2'):
            block = '\n'.join(lines[node.lineno - 1:node.end_lineno])
            return textwrap.dedent(block)
    raise AssertionError(
        f"no 'if not skip_step1and2:' block in {SRC}; the stage-1/2 loop moved "
        "and this harness needs to follow it")


class LoopRun:
    """What one ``nrca,nrcb,merged`` sequence did."""

    def __init__(self):
        self.detector1 = []
        self.image2 = []
        self.printed = []

    @property
    def detector1_inputs(self):
        return [fn for fn, _ in self.detector1]

    @property
    def image2_inputs(self):
        return [fn for fn, _ in self.image2]


def run_stage12(members, modules=('nrca', 'nrcb', 'merged'), source=None,
                proposal_id=PROPOSAL_ID, field=FIELD, output_dir='.',
                save_calibrated_ramp=True):
    """Exec the driver's stage-1/2 block once per module, recording the calls.

    One ``LoopRun`` covers the whole module sequence, the way one interpreter
    covers it in ``__main__``: the in-process memo is reset once at the start
    and then left to the driver.
    """
    reset_stage12_processed()
    code = compile(stage12_block_source() if source is None else source,
                   str(SRC), 'exec')
    run = LoopRun()

    class Detector1Pipeline:
        @staticmethod
        def call(uncal_fn, **kwargs):
            run.detector1.append((uncal_fn, kwargs))

    class Image2Pipeline:
        @staticmethod
        def call(rate_fn, **kwargs):
            run.image2.append((rate_fn, kwargs))

    asn_data = {'products': [{'members': [{'expname': e} for e in members]}]}
    namespace = {name: getattr(stage12_selection, name)
                 for name in dir(stage12_selection) if not name.startswith('_')}
    namespace.update({
        'skip_step1and2': False,
        'asn_data': asn_data,
        'proposal_id': proposal_id,
        'field': field,
        'output_dir': output_dir,
        'SAVE_CALIBRATED_RAMP': save_calibrated_ramp,
        'Detector1Pipeline': Detector1Pipeline,
        'Image2Pipeline': Image2Pipeline,
        'print': lambda *args, **kwargs: run.printed.append(
            ' '.join(str(a) for a in args)),
    })
    for module in modules:
        namespace['module'] = module
        try:
            exec(code, namespace)
        except NameError as exc:
            raise AssertionError(
                f"the stage-1/2 block reads a name this harness does not "
                f"supply ({exc}); add it to run_stage12's namespace") from exc
    return run


def neutered_memo_source():
    """The block with the cross-pass memo defeated by a per-pass reset.

    The edit a maintainer could plausibly make ("clear stale state before the
    loop") and the one the source-inspection tests cannot see.
    """
    block = stage12_block_source()
    header, rest = block.split('\n', 1)
    assert header.strip() == 'if not skip_step1and2:', header
    return f"{header}\n    reset_stage12_processed()\n{rest}"


def _lay_down_products(tmp_path, members, uncal_mtime=1_000_000.0,
                       product_mtime=1_000_060.0):
    """An already-reduced tree: every member's uncal, cal and ramp on disk."""
    for expname in members:
        for suffix in ('uncal', 'cal', 'ramp'):
            path = tmp_path / expname.replace('_cal.fits', f'_{suffix}.fits')
            path.write_bytes(b'x')
            mtime = uncal_mtime if suffix == 'uncal' else product_mtime
            os.utime(path, (mtime, mtime))


@pytest.fixture(autouse=True)
def _clean_memo():
    reset_stage12_processed()
    yield
    reset_stage12_processed()


@pytest.fixture(autouse=True)
def _no_resume(monkeypatch):
    """Default runs are bare ``SKIP=0``; the resume tests set the variable."""
    monkeypatch.delenv(STAGE12_RESUME_ENV, raising=False)


# ---------------------------------------------------------------------------
# the harness itself reaches the driver
# ---------------------------------------------------------------------------

def test_block_source_is_the_shipped_loop():
    block = stage12_block_source()
    assert block.startswith('if not skip_step1and2:')
    assert 'Detector1Pipeline.call' in block
    assert 'Image2Pipeline.call' in block
    assert "for member in asn_data['products'][0]['members']" in block


# ---------------------------------------------------------------------------
# #417: one Detector1 per exposure per run
# ---------------------------------------------------------------------------

def test_each_exposure_is_ramp_fitted_once_per_run():
    """The headline #417 claim, measured: three module passes in one
    interpreter, one Detector1 call per exposure."""
    run = run_stage12(ALL_NIRCAM + NON_NIRCAM)
    assert len(run.detector1) == len(ALL_NIRCAM)
    assert sorted(run.detector1_inputs) == sorted(
        e.replace('_cal.fits', '_uncal.fits') for e in ALL_NIRCAM)


def test_each_exposure_is_image2_processed_once_per_run():
    run = run_stage12(ALL_NIRCAM + NON_NIRCAM)
    assert sorted(run.image2_inputs) == sorted(
        e.replace('_cal.fits', '_rate.fits') for e in ALL_NIRCAM)


def test_the_memo_survives_from_one_module_pass_to_the_next():
    """The nrca/nrcb passes claim their own members, so every Detector1 call
    in the merged pass is a member the memo failed to carry."""
    run = run_stage12(ALL_NIRCAM)
    first_two_passes = len(A_MEMBERS) + len(B_MEMBERS)
    assert len(run.detector1) == first_two_passes, (
        f"the merged pass re-ran Detector1 on "
        f"{len(run.detector1) - first_two_passes} member(s) the module passes "
        "already calibrated: the in-process memo is not surviving the pass "
        "boundary (#417)")


def test_neutering_the_memo_is_visible_here():
    """Harness sensitivity check: with a ``reset_stage12_processed()`` at the
    top of the block -- which passes ``test_stage12_selection.py`` 35/35 -- the
    merged pass redoes both module passes, and these counts say so."""
    run = run_stage12(ALL_NIRCAM, source=neutered_memo_source())
    assert len(run.detector1) == 2 * len(ALL_NIRCAM)
    assert len(set(run.detector1_inputs)) == len(ALL_NIRCAM)


def test_module_order_does_not_change_the_count():
    for modules in (('nrca', 'nrcb', 'merged'), ('merged', 'nrca', 'nrcb'),
                    ('nrcb', 'merged', 'nrca')):
        run = run_stage12(ALL_NIRCAM, modules=modules)
        assert len(run.detector1) == len(ALL_NIRCAM), modules


def test_non_nircam_members_are_never_processed():
    run = run_stage12(NON_NIRCAM)
    assert run.detector1 == []
    assert run.image2 == []


# ---------------------------------------------------------------------------
# hand-split module runs
# ---------------------------------------------------------------------------

def test_merged_alone_produces_every_cal():
    """A merged-only run must be able to calibrate every member; with a fresh
    process it does all of them, including any another run already did."""
    run = run_stage12(ALL_NIRCAM, modules=('merged',))
    assert sorted(run.detector1_inputs) == sorted(
        e.replace('_cal.fits', '_uncal.fits') for e in ALL_NIRCAM)


def test_nrca_alone_produces_only_its_own_module():
    """``-m nrca`` writes the A-module `_cal` files and no others, which is why
    ``docs/HIPERGATOR.md`` tells the operator to run both modules."""
    run = run_stage12(ALL_NIRCAM, modules=('nrca',))
    assert sorted(run.detector1_inputs) == sorted(
        e.replace('_cal.fits', '_uncal.fits') for e in A_MEMBERS)


def test_merged_after_nrca_in_a_new_process_redoes_the_a_module():
    """``-m nrca`` then ``-m merged`` as two SEPARATE processes: the memo is
    per-process, so the merged run re-fits the A-module ramps as well.  Only
    ``STAGE12_RESUME=1`` fills in just the missing ones (next test)."""
    run = run_stage12(ALL_NIRCAM, modules=('merged',))
    assert len(run.detector1) == len(ALL_NIRCAM)
    assert len(run.detector1) > len(B_MEMBERS)


def test_resume_fills_in_only_the_missing_members(tmp_path, monkeypatch):
    """``STAGE12_RESUME=1`` after an ``-m nrca`` run: the A-module products are
    on disk and newer than their uncals, so the merged run calibrates only the
    B module."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(STAGE12_RESUME_ENV, '1')
    _lay_down_products(tmp_path, ALL_NIRCAM)
    for expname in B_MEMBERS:
        for suffix in ('cal', 'ramp'):
            (tmp_path / expname.replace('_cal.fits', f'_{suffix}.fits')).unlink()
    run = run_stage12(ALL_NIRCAM, modules=('merged',))
    assert sorted(run.detector1_inputs) == sorted(
        e.replace('_cal.fits', '_uncal.fits') for e in B_MEMBERS)


# ---------------------------------------------------------------------------
# what SKIP=0 still means
# ---------------------------------------------------------------------------

def test_skip0_reprocesses_a_fully_reduced_tree(tmp_path, monkeypatch):
    """The round-1 blocker, measured through the driver: with products on disk
    and newer than every uncal, a bare ``SKIP=0`` run still re-fits all of
    them, so a CRDS repin or a ``jwst`` bump is forced through."""
    monkeypatch.chdir(tmp_path)
    _lay_down_products(tmp_path, ALL_NIRCAM)
    run = run_stage12(ALL_NIRCAM)
    assert len(run.detector1) == len(ALL_NIRCAM)


def test_resume_skips_a_fully_reduced_tree(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(STAGE12_RESUME_ENV, '1')
    _lay_down_products(tmp_path, ALL_NIRCAM)
    run = run_stage12(ALL_NIRCAM)
    assert run.detector1 == []
    assert run.image2 == []


def test_resume_reprocesses_a_member_whose_ramp_is_gone(tmp_path, monkeypatch):
    """Ramp retention is #421's subject; with ramps still being saved, a member
    missing its ramp is reprocessed even under resume."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(STAGE12_RESUME_ENV, '1')
    _lay_down_products(tmp_path, ALL_NIRCAM)
    (tmp_path / LW_A[0].replace('_cal.fits', '_ramp.fits')).unlink()
    run = run_stage12(ALL_NIRCAM)
    assert run.detector1_inputs == [LW_A[0].replace('_cal.fits', '_uncal.fits')]


def test_ramps_not_saved_stops_the_ramp_clause_forcing_a_reprocess(
        tmp_path, monkeypatch):
    """#421 state: with ``save_calibrated_ramp`` off there are no ramps, and
    the resume check must stop requiring one instead of reprocessing every
    member forever."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(STAGE12_RESUME_ENV, '1')
    _lay_down_products(tmp_path, ALL_NIRCAM)
    for expname in ALL_NIRCAM:
        (tmp_path / expname.replace('_cal.fits', '_ramp.fits')).unlink()
    assert run_stage12(ALL_NIRCAM, save_calibrated_ramp=False).detector1 == []
    assert len(run_stage12(ALL_NIRCAM,
                           save_calibrated_ramp=True).detector1) == len(ALL_NIRCAM)


def test_detector1_is_told_the_same_ramp_flag_the_skip_check_is():
    run = run_stage12(ALL_NIRCAM[:1], save_calibrated_ramp=False)
    assert run.detector1[0][1]['save_calibrated_ramp'] is False


# ---------------------------------------------------------------------------
# provenance + logging, at runtime
# ---------------------------------------------------------------------------

def test_a_foreign_member_trips_the_provenance_assert_on_every_pass():
    """A mis-globbed member from another proposal or field stops the run in
    the nrcb pass too, which is what a single-module-policy field (sickle,
    nrcb only) gets."""
    foreign = 'jw02221002001_02201_00002_nrcb1_cal.fits'
    with pytest.raises(AssertionError):
        run_stage12([foreign], modules=('nrcb',))


def test_every_skip_announces_itself():
    run = run_stage12(ALL_NIRCAM + NON_NIRCAM)
    printed = '\n'.join(run.printed)
    assert 'pass does not claim it' in printed
    assert 'Skipping stage 1+2 for' in printed
    assert 'Skipping non-NIRCam member' in printed
