"""The bounded-fixed-point branch must be REACHED, must raise the floor, and
must take another pass rather than leaving the loop.

`--accept-below-mas` exits 4 to say "this residual repeats, and it is small
enough to be the systematic a per-exposure offsets table cannot express".  The
loop then has to do three things, and doing any subset is worse than doing none:

1. raise `ASTROM_M2_CORRECTION_FLOOR_MAS` above the residual;
2. re-reduce ONCE MORE under that floor, so the frames are re-drizzled with the
   corrections the checkpoint just wrote into the offsets table;
3. leave by the normal converged exit.

Skipping (1) means m2 re-measures the same residual inside the final chain,
CORRECTS it, and the frozen m3+ stages raise on the shift -- so the field stops
one stage later, having spent a whole reduce and an m12 to get there.

Skipping (2) is subtler and worse.  Reaching this branch at all requires that
m2 just wrote new corrections into the offsets table and stale-tagged this
filter's `_i2d` mosaics.  The frames on disk still carry the PREVIOUS pass's
baked `RAOFFSET`.  Breaking out here would leave the table describing frames
that were never re-drizzled -- the frame/table disagreement behind brick-1182
v001 -- and the mosaics renamed `_im0_badastrom` permanently.

The same `set -e` trap that hid the reduce guard applies to the check itself:
its nonzero exit codes ARE its output, so a bare `fp_out=$(...)` would exit the
script one line before the branch that reads them.
"""
import pathlib
import re
import shlex
import subprocess

import pytest

LOOP = (pathlib.Path(__file__).parents[2] / 'scripts' / 'reduction'
        / 'run_field_retie_loop.sh')


def _src():
    return LOOP.read_text()


def _branch():
    """The `elif [ "$fp_rc" -eq 4 ]` arm, up to the next arm."""
    src = _src()
    start = src.index('elif [ "$fp_rc" -eq 4 ]')
    return src[start:src.index('elif [ "$fp_rc" -ne 0 ]', start)]


def _invocation():
    """The `fp_out=$(...)` statement alone.

    Scoped deliberately: the flag names also appear in the header comment, so
    an assertion over the whole file stays green after the flag is deleted from
    the call.
    """
    src = _src()
    start = src.index('fp_out=$(PYTHONPATH=')
    return src[start:src.index('echo "$fp_out"', start)]


def test_the_check_is_asked_for_a_ceiling():
    assert '--accept-below-mas' in _invocation()
    assert 'RETIE_ACCEPT_RESIDUAL_MAS' in _invocation()


def test_the_ceiling_defaults_to_accepting_nothing():
    """Unset means every fixed point stops, which is what it did before."""
    assert re.search(r'RETIE_ACCEPT_RESIDUAL_MAS:-0', _invocation())


def test_the_check_invocation_cannot_kill_the_loop():
    """rc 3 and rc 4 are the two verdicts; under `set -e` a bare assignment
    exits at the first of them."""
    assert '|| fp_rc=' in _invocation(), (
        'the fixed-point check exits nonzero to report its verdict; without '
        '`|| fp_rc=$?` set -e kills the loop before the branch reads it')


def test_a_bounded_fixed_point_does_NOT_leave_the_loop_here():
    """It re-reduces once more instead.

    m2 has just rewritten the offsets table and stale-tagged the mosaics; the
    frames still carry the previous pass's shift.  Breaking here ships that
    disagreement and leaves the mosaics quarantined.
    """
    branch = _branch()
    assert not re.search(r'^\s*break\s*$', branch, re.M), (
        'breaking out of the iteration loop here leaves the offsets table '
        'ahead of the frames and the _i2d mosaics stale-tagged for good')


def test_it_raises_the_correction_floor_before_taking_that_pass():
    branch = _branch()
    assert 'ASTROM_M2_CORRECTION_FLOOR_MAS="$fp_floor"' in branch
    assert 'export ASTROM_M2_CORRECTION_FLOOR_MAS' in branch


def test_accepting_on_the_LAST_iteration_stops_and_says_how_to_finish():
    """Falling through on the final pass would end the loop with the table and
    the frames still disagreeing -- the same defect, reached by running out of
    iterations instead of by breaking."""
    branch = _branch()
    assert '"$it" -eq "$MAXITER"' in branch
    assert 'MAXITER=$((MAXITER + 1))' in branch


def test_a_missing_floor_stops_rather_than_running_at_the_old_one():
    """If the check said BOUNDED but printed no floor, proceeding would run the
    frozen stages at the old floor -- the failure this branch exists to avoid,
    reached by the path that was supposed to avoid it."""
    branch = _branch()
    assert 'exit 3' in branch
    assert 'no floor' in branch


def test_the_floor_is_taken_from_the_checks_own_output():
    """Not recomputed by the shell: a second run of the check would judge a
    different set of records, and the number that reaches the chain has to be
    the one whose reasoning is in the log."""
    assert 'ASTROM_M2_CORRECTION_FLOOR_MAS=' in _branch()
    assert re.search(r"sed -n 's/\^ASTROM_M2_CORRECTION_FLOOR_MAS=//p'",
                     _branch())


def test_a_malformed_ceiling_refuses_instead_of_disabling_the_stop():
    """A typo makes argparse exit 2.  Treated as "some other failure" that
    would switch the fixed-point STOP off for the rest of the run, and the loop
    would grind to MAXITER at ~7 h a pass with a usage message in the log."""
    src = _src()
    assert 'RETIE_ACCEPT_RESIDUAL_MAS must' in src or 'is not a number' in src
    start = src.index('elif [ "$fp_rc" -ne 0 ]')
    other = src[start:start + 600]
    assert 'exit 3' in other, (
        'an exit code that is not 0/3/4 means the check never reported a '
        'verdict; continuing past it runs blind')


# ---------------------------------------------------------------------------
# ...and the branch actually EXECUTED, not just read
# ---------------------------------------------------------------------------

def _run_branch(fp_rc, fp_out, it=2, maxiter=3):
    """Execute the fp_rc dispatch out of the real script under bash.

    The text assertions above pin what the branch says; this pins what bash
    does with it -- the `sed` extraction, `set -e`, the variable capture and
    the exit codes, none of which a substring check can see.
    """
    src = _src()
    start = src.index('        if [ "$fp_rc" -eq 3 ]; then')
    end = src.index('        elif [ "$fp_rc" -ne 0 ]; then', start)
    end = src.index('        fi\n', end)
    body = src[start:end] + '        fi\n'
    script = (
        'set -euo pipefail\n'
        f'it={it}\nMAXITER={maxiter}\nfp_rc={fp_rc}\n'
        # shlex.quote, not repr: repr escapes the newlines, and the branch
        # extracts the floor with a line-anchored sed.
        f'fp_out={shlex.quote(fp_out)}\n'
        'ASTROM_M2_CORRECTION_FLOOR_MAS=${ASTROM_M2_CORRECTION_FLOOR_MAS:-}\n'
        + body +
        'echo "AFTER_BRANCH floor=${ASTROM_M2_CORRECTION_FLOOR_MAS}"\n')
    return subprocess.run(['bash', '-c', script], capture_output=True, text=True)


BOUNDED_OUT = ('F162M/o012: REPEATING over the last 3 pass(es)\n'
               'BOUNDED: the largest residual still measured is 5.02 mas\n'
               'ASTROM_M2_CORRECTION_FLOOR_MAS=5.5')


def test_executed_rc4_exports_the_floor_and_keeps_going():
    r = _run_branch(4, BOUNDED_OUT, it=2, maxiter=3)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'AFTER_BRANCH floor=5.5' in r.stdout, r.stdout


def test_executed_rc4_on_the_last_iteration_stops():
    r = _run_branch(4, BOUNDED_OUT, it=3, maxiter=3)
    assert r.returncode == 2, r.stdout + r.stderr
    assert 'MAXITER=4' in r.stdout, r.stdout


def test_executed_rc4_without_a_floor_stops():
    r = _run_branch(4, 'BOUNDED: ...but no floor line', it=2, maxiter=3)
    assert r.returncode == 3, r.stdout + r.stderr


def test_executed_rc3_stops():
    r = _run_branch(3, 'NOT BOUNDED: ...', it=2, maxiter=3)
    assert r.returncode == 3, r.stdout + r.stderr


def test_executed_rc2_stops_rather_than_running_blind():
    r = _run_branch(2, 'usage: retie_fixed_point ...', it=2, maxiter=3)
    assert r.returncode == 3, r.stdout + r.stderr


def test_executed_rc0_falls_through():
    r = _run_branch(0, 'F162M/o012: still moving', it=2, maxiter=3)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'AFTER_BRANCH floor=' in r.stdout


@pytest.mark.parametrize('bad', ['abc', '1.2.3', '15mas', '-5'])
def test_a_non_numeric_ceiling_is_refused_before_the_check_runs(bad):
    src = _src()
    start = src.index('        case "${RETIE_ACCEPT_RESIDUAL_MAS:-0}" in')
    end = src.index('        esac\n', start) + len('        esac\n')
    script = (f'RETIE_ACCEPT_RESIDUAL_MAS={bad!r}\n' + src[start:end]
              + 'echo ACCEPTED\n')
    r = subprocess.run(['bash', '-c', script], capture_output=True, text=True)
    assert r.returncode == 2, r.stdout
    assert 'ACCEPTED' not in r.stdout


@pytest.mark.parametrize('good', ['0', '15', '7.5', ''])
def test_a_numeric_ceiling_passes_the_guard(good):
    src = _src()
    start = src.index('        case "${RETIE_ACCEPT_RESIDUAL_MAS:-0}" in')
    end = src.index('        esac\n', start) + len('        esac\n')
    script = (f'RETIE_ACCEPT_RESIDUAL_MAS={good!r}\n' + src[start:end]
              + 'echo ACCEPTED\n')
    r = subprocess.run(['bash', '-c', script], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'ACCEPTED' in r.stdout
