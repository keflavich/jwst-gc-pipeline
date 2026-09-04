"""The per-frame fan-out must be ABLE to pass ``--skip-if-done``, and must not
pass it unasked.

A fan-out task writes a completion marker for every frame it finishes, and
``cataloging.select_resumable_frames`` reads them back -- but only through

    skip_if_done and (skip_finalize or finalize_only)

and the phase sbatch never passed the flag at all, so ``SKIP_IF_DONE`` in the
environment did nothing.  Every cancel-and-resubmit refit the whole shard and
met the same wall clock.  Measured on wd1, 2026-08-29, same field/phase/shard
count, full range over the 32 array tasks (issue #570):

    without --skip-if-done (job 40592842)   4:26:41 - 8:44:12  median 5:51:17
    with    --skip-if-done (job 40623830)   0:00:46 - 0:01:12  median 0:00:50

The flag is plumbed, and it is OPT-IN.  Nothing in a completion marker changes
when the photometry code does: re-running cataloging to apply a fit fix, on
frames whose mtimes never move, is the campaign's most common reason to launch
this, and a defaulted-on resume would skip every frame and report green with the
old photometry.  So:

  * fan-out mode passes ``--skip-if-done`` when ``SKIP_IF_DONE=1``, executed
    from the lines as shipped rather than read for a substring;
  * it arrives together with ``--manual-skip-finalize``, the other half of the
    guard above -- without it the flag is inert in this mode;
  * the DEFAULT is off, so an ordinary re-catalog refits;
  * finalize mode never grows it (that mode fits nothing and verifies markers
    through its own strict all-markers check);
  * ``MODE_ARGS`` reaches the python invocation, and the spelling is the option
    the parser defines -- a rename on either side fails here rather than
    silently dropping the resume again;
  * the driver exports the variable, so the documented one-liner reaches the
    array.

What the resume still refuses, with the flag ON, is pinned in
``photometry/tests/test_perframe_resume.py``: a marker older than its ``_crf``
(regenerated frame) or older than the phase's seed inputs (a re-run finalize
rewrote the previous phase's vetted catalog / residual i2d / smoothed bg).
"""
import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(REPO, 'scripts', 'reduction')
PHASE = os.path.join(SCRIPTS, 'submit_cataloging_perframe_phase.sbatch')
DRIVER = os.path.join(SCRIPTS, 'submit_cataloging_perframe.sh')
LONG = os.path.join(REPO, 'jwst_gc_pipeline', 'photometry',
                    'crowdsource_catalogs_long.py')
CATALOGING = os.path.join(REPO, 'jwst_gc_pipeline', 'photometry', 'cataloging.py')

FLAG = '--skip-if-done'
ENV_VAR = 'SKIP_IF_DONE'


def _text(path):
    with open(path) as fh:
        return fh.read()


def _mode_block():
    """The shipped lines that decide MODE_ARGS, plus the default they read."""
    src = _text(PHASE)
    default = re.search(r'^' + ENV_VAR + r'=\$\{' + ENV_VAR + r':-\S+?\}$',
                        src, re.MULTILINE)
    assert default, f'{ENV_VAR} has no default in the phase script'
    block = re.search(r'^if \[ "\$MODE" = "fanout" \]; then\n(?:.*\n)*?^fi$',
                      src, re.MULTILINE)
    assert block, 'the MODE dispatch is not where this test looks for it'
    return default.group(0) + '\n' + block.group(0)


def _mode_args(mode, env=None):
    """Run the shipped block and report the MODE_ARGS it produces."""
    full = dict(os.environ)
    full.pop(ENV_VAR, None)
    full.update({'MODE': mode, 'NSHARDS': '16', 'SLURM_ARRAY_TASK_ID': '3',
                 'TARGET': 'wd1', 'PHASE': 'm12'})
    full.update(env or {})
    # Stubs for the two side effects the block has on a compute node: the
    # runtime rename (scontrol) and the finalize's 180 s Lustre settle.
    stub = ('_pf_rename_wanted() { false; }\nsleep() { :; }\n')
    run = subprocess.run(['bash', '-c', stub + _mode_block() +
                          '\necho "MODE_ARGS=[$MODE_ARGS]"'],
                         capture_output=True, text=True, env=full, timeout=60)
    assert run.returncode == 0, run.stderr
    got = re.search(r'MODE_ARGS=\[(.*)\]', run.stdout)
    assert got, run.stdout
    return got.group(1)


def test_the_default_does_not_resume():
    """A plain re-run refits.  Nothing in a marker changes when the photometry
    code does, so a re-catalog to apply a fit fix must not be a near-no-op."""
    assert FLAG not in _mode_args('fanout').split(), (
        'the fan-out passes ' + FLAG + ' unasked, so a re-catalog on unchanged '
        'frames skips every frame and ships the old photometry')


def test_the_flag_is_plumbed_when_asked_for():
    """The defect #570 is about: SKIP_IF_DONE=1 used to do nothing at all."""
    assert FLAG in _mode_args('fanout', {ENV_VAR: '1'}).split(), (
        'SKIP_IF_DONE=1 does not reach the run, so a restart refits every '
        'frame it has already finished')


def test_the_flag_arrives_with_manual_skip_finalize():
    """Both halves of the guard, or the flag does nothing in this mode."""
    args = _mode_args('fanout', {ENV_VAR: '1'}).split()
    assert '--manual-skip-finalize' in args and FLAG in args


@pytest.mark.parametrize('off', ['0', 'no', ''])
def test_only_an_explicit_1_turns_it_on(off):
    assert FLAG not in _mode_args('fanout', {ENV_VAR: off}).split()


def test_finalize_mode_is_untouched():
    args = _mode_args('finalize', {ENV_VAR: '1'}).split()
    assert args == ['--manual-finalize-only'], args


def test_mode_args_reaches_the_command():
    """A built argument that is never interpolated is not passed."""
    assert '$MODE_ARGS' in _text(PHASE)


def test_the_driver_exports_the_variable():
    """The documented restart is `SKIP_IF_DONE=1 submit_cataloging_perframe.sh`;
    the array only sees it because the driver exports it and COMMON is ALL."""
    src = _text(DRIVER)
    assert re.search(r'^export ' + ENV_VAR + r'=', src, re.MULTILINE), src[:0]
    assert re.search(r'COMMON="ALL,', src), 'COMMON no longer inherits ALL'


def test_the_driver_documents_the_restart():
    assert ENV_VAR + '=1' in _text(DRIVER), (
        'the opt-in resume is undiscoverable if the driver does not name it')


def test_the_spelling_is_the_option_the_parser_defines():
    assert re.search(r"add_option\('" + re.escape(FLAG) + r"'", _text(LONG)), (
        FLAG + ' is not an option of the entry point the phase script runs')


def test_the_resume_still_gates_on_marker_age_and_seed_age():
    """The flag is only safe to offer while both staleness gates exist.

    That the CALL SITE actually feeds the second one is pinned next to the gate
    itself (test_perframe_resume.test_the_call_site_passes_the_phase_SEED_INPUTS);
    here we only check the signature the sbatch is documented against.
    """
    src = _text(CATALOGING)
    assert 'def _marker_is_current(marker_path, frame_path, seed_inputs=()):' in src
    assert re.search(r'def select_resumable_frames\([^)]*\n\s*seed_inputs=', src), (
        'the resume no longer takes the phase seed inputs; a re-run finalize '
        'would leave every marker looking current')


def test_the_run_offers_the_resume_when_it_is_off():
    """The restart case must not depend on the operator having read a doc."""
    src = _text(CATALOGING)
    assert ENV_VAR + '=1' in src, (
        'the fan-out never names the opt-in, so a wall-clocked restart has no '
        'way to learn about it from its own log')
