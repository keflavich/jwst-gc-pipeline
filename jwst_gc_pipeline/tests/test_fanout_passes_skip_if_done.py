"""The per-frame fan-out must pass ``--skip-if-done``.

A fan-out task writes a completion marker for every frame it finishes, and
``cataloging.select_resumable_frames`` reads them back -- but only through

    if getattr(options, 'skip_if_done', False) and (skip_finalize or finalize_only)

so with the flag off the markers are written, kept, and ignored.  Every
cancel-and-resubmit then refits the whole shard and meets the same wall clock.
Measured on wd1, 2026-08-29, same field/phase/shard count (issue #570):

    without --skip-if-done (job 40592842)   4:26:41 - 4:29:20 per shard
    with    --skip-if-done (job 40623830)   0:00:46 - 0:00:50 per shard

Pinned here:

  * fan-out mode builds ``--skip-if-done`` into ``MODE_ARGS`` by default --
    executed from the lines as shipped, not read for a substring;
  * it arrives together with ``--manual-skip-finalize``, which is the other half
    of the guard above: without it the flag is inert in this mode;
  * ``SKIP_IF_DONE=0`` takes it off again, so a run that wants the
    unconditional refit still has one;
  * finalize mode does not grow it (that mode fits nothing and verifies markers
    through its own strict all-markers check);
  * ``MODE_ARGS`` reaches the python invocation, and the spelling is the option
    the parser defines -- a rename on either side fails here rather than
    silently dropping the resume again.

The resume itself stays gated on marker mtime vs frame mtime
(``_marker_is_current``, #571): a regenerated frame is refitted and counted, and
that is what makes defaulting this on safe.  That gate has its own tests; this
file pins only the enablement.
"""
import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(REPO, 'scripts', 'reduction')
PHASE = os.path.join(SCRIPTS, 'submit_cataloging_perframe_phase.sbatch')
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


def test_fanout_passes_the_flag_by_default():
    """The one line this issue is about."""
    assert FLAG in _mode_args('fanout').split(), (
        'the fan-out does not pass ' + FLAG + ', so every resubmit refits '
        'every frame it has already finished')


def test_the_flag_arrives_with_manual_skip_finalize():
    """Both halves of the guard, or the flag does nothing in this mode."""
    args = _mode_args('fanout').split()
    assert '--manual-skip-finalize' in args and FLAG in args


@pytest.mark.parametrize('off', ['0', 'no'])
def test_the_default_can_be_turned_off(off):
    assert FLAG not in _mode_args('fanout', {ENV_VAR: off}).split()


def test_finalize_mode_is_untouched():
    args = _mode_args('finalize').split()
    assert args == ['--manual-finalize-only'], args


def test_mode_args_reaches_the_command():
    """A built argument that is never interpolated is not passed."""
    assert '$MODE_ARGS' in _text(PHASE)


def test_the_spelling_is_the_option_the_parser_defines():
    assert re.search(r"add_option\('" + re.escape(FLAG) + r"'", _text(LONG)), (
        FLAG + ' is not an option of the entry point the phase script runs')


def test_the_resume_still_gates_on_marker_age():
    """Defaulting the flag on is safe only while the mtime gate is in place."""
    src = _text(CATALOGING)
    assert 'def _marker_is_current(' in src
    assert re.search(r'if getattr\(options, .skip_if_done., False\) and \(',
                     src), 'the resume call site moved; re-check this default'
