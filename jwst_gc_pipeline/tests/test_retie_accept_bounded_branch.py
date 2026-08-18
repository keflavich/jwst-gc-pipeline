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
import os
import pathlib
import re
import shlex
import subprocess
import tempfile

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

    `src.index` here LOCATES the block; it is not the assertion.  A grep-shaped
    test that silently passes when the thing it greps for moves is the pattern
    this repo has been removing, and this is the other one: if the line moves,
    `index` raises `ValueError` and every test using it fails loudly.  What is
    asserted about the block is in the tests below, and the strongest of them
    executes it (`test_EXECUTED_the_check_receives_the_runs_own_filter_list`)
    rather than reading it.
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
    # max(existing, computed), so acceptance can never LOWER an operator floor
    assert "_effective_floor=$(awk" in branch
    assert 'ASTROM_M2_CORRECTION_FLOOR_MAS="$_effective_floor"' in branch
    assert 'export ASTROM_M2_CORRECTION_FLOOR_MAS' in branch


def test_both_messages_quote_the_floor_the_run_will_ACTUALLY_use():
    """They used to interpolate the computed value while the run continued at
    the max, so with a preset of 4.0 and a computed 0.6 the log said 0.6 -- and
    the last-iteration line then told a human to set 0.6, which is the exact
    lowering the max exists to prevent."""
    branch = _branch()
    assert '$fp_floor (was' not in branch, (
        'the announcement must quote the exported floor, not the computed one')
    said = [ln.strip() for ln in branch.splitlines()
            if ln.strip().startswith('echo ')
            and 'ASTROM_M2_CORRECTION_FLOOR_MAS=' in ln]
    assert said, 'the branch must say which floor it is continuing at'
    for line in said:
        assert '$_effective_floor' in line, line


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


# ---------------------------------------------------------------------------
# The acceptance path has to be able to COMPLETE
# ---------------------------------------------------------------------------

def test_acceptance_requires_more_iterations_than_a_fixed_point_needs():
    """At the loop's own defaults the accept path could never finish.

    A fixed point needs three passes before it can be judged, so the check first
    has an opinion at iteration 3 -- which at `MAXITER=3` is the last one.
    Accepting there has nowhere to re-reduce, so the run ends with the offsets
    table ahead of the frames and the mosaics stale-tagged: exactly the state
    this path exists to avoid, reached by running out of iterations instead of
    by breaking.
    """
    from jwst_gc_pipeline.photometry.retie_fixed_point import DEFAULT_REPEATS
    src = _src()
    assert 'MAXITER" -lt 4' in src, (
        'acceptance must require headroom beyond the pass count a fixed point '
        f'needs to be judged (DEFAULT_REPEATS={DEFAULT_REPEATS})')
    guard = src[src.index('RETIE_ACCEPT_RESIDUAL_MAS:-0}" != "0" ]'):][:600]
    assert 'exit 2' in guard
    assert 'MAXITER=4' in guard, 'the refusal must say what to re-run with'


@pytest.mark.parametrize('maxiter,expect', [('3', 2), ('4', 0)])
def test_executed_the_maxiter_guard_refuses_before_a_pass_is_spent(maxiter,
                                                                   expect):
    src = _src()
    start = src.index('if [ "${RETIE_ACCEPT_RESIDUAL_MAS:-0}" != "0" ] && [ "$MAXITER" -lt 4 ]')
    end = src.index('fi\n', start) + len('fi\n')
    script = (f'MAXITER={maxiter}\nRETIE_ACCEPT_RESIDUAL_MAS=15\n'
              + src[start:end] + 'echo PROCEEDED\n')
    r = subprocess.run(['bash', '-c', script], capture_output=True, text=True)
    assert r.returncode == expect, r.stdout + r.stderr
    assert ('PROCEEDED' in r.stdout) == (expect == 0)


def test_the_floor_is_never_LOWERED_by_acceptance():
    """Eight fields run at an operator floor of 4.0 mas today.  A computed 0.5
    would make the next checkpoint correct everything above 0.5 mas, so the loop
    would never converge -- acceptance would cause the failure it prevents."""
    branch = _branch()
    assert 'awk' in branch and 'a>b' in branch, (
        'the raised floor must be max(existing, computed), not an assignment')


@pytest.mark.parametrize('prev,computed,want', [
    ('4.0', '0.5', '4.0'), ('4.0', '8.0', '8.0'), ('', '5.5', '5.5')])
def test_executed_the_floor_takes_the_larger_of_the_two(prev, computed, want):
    script = (f'_prev_floor={prev or 0}\nfp_floor={computed}\n'
              "ASTROM_M2_CORRECTION_FLOOR_MAS=$(awk -v a=\"$_prev_floor\" "
              "-v b=\"$fp_floor\" 'BEGIN{print (a>b)?a:b}')\n"
              'echo "FLOOR=$ASTROM_M2_CORRECTION_FLOOR_MAS"\n')
    r = subprocess.run(['bash', '-c', script], capture_output=True, text=True)
    assert f'FLOOR={want}' in r.stdout, r.stdout


def test_the_check_is_told_which_filters_to_expect():
    """`--expect-filters` is the only thing standing between the acceptance
    branch and the hole it was added to close, and it appears exactly ONCE
    outside the module -- in the loop's invocation.

    Deleting that one line leaves the whole suite green (431 passed) and sends
    gc2211/o046 straight back to rc=4, accepting at 3.63 mas against an operator
    floor of 4.0.  A production wiring with no test is a wiring that gets
    refactored away.
    """
    assert '--expect-filters' in _invocation(), (
        'the loop must declare which filters it expects, or the coverage '
        'refusal can never fire')


def test_the_expected_filters_are_the_runs_own_list_verbatim():
    """It has to be the run's own filter list, verbatim.

    An earlier version of this test read a 60-character window after the flag
    and asserted a ``$`` appeared somewhere in it.  The flag is 16 characters,
    so the window ran past the end of the line into ``|| fp_rc=$?`` and any
    literal shorter than about eleven characters passed: ``"F200W"`` froze an
    under-declaration into production and ``""`` turned the coverage gate off
    outright, both with the suite green.  Assert the ARGUMENT, not the
    neighbourhood."""
    import re as _re
    inv = _invocation()
    m = _re.search(r'--expect-filters\s+("[^"]*"|\S+)', inv)
    assert m, f'--expect-filters is absent from the invocation: {inv}'
    assert m.group(1) == '"$FILTERS"', (
        f"--expect-filters must take the run's own filter list verbatim, "
        f"got {m.group(1)}")


def test_EXECUTED_the_check_receives_the_runs_own_filter_list():
    """Run the real invocation with a stub `python` on PATH and read the argv it
    receives.

    The source-text pin above cannot see two runtime-breaking mutants that leave
    every retie test green: reassigning `FILTERS="F999W"` just before the check,
    and `if false && [ "${RETIE_FIXED_POINT_CHECK:-1}" = "1" ]`, which makes the
    whole check unreachable.  Executing it is the only way to pin what the
    process is actually handed."""
    # From the ENCLOSING guard, not from `fp_rc=0`: a mutant that reassigns
    # FILTERS just above the call, or wraps the guard in `if false &&`, lives
    # outside the narrower slice and would go unseen.
    src = _src()
    start = src.index('    if [ "${RETIE_FIXED_POINT_CHECK:-1}" = "1" ]')
    end = src.index('        echo "$fp_out"\n', start)
    body = src[start:end] + '    fi\n'
    # `set -e` + the `case` refusal are part of what is being executed; give the
    # loop variables the branch reads.
    body = body.replace('exit 2 ;;', 'echo REFUSED >&2; exit 2 ;;')

    with tempfile.TemporaryDirectory() as d:
        argv_log = os.path.join(d, 'argv.txt')
        stub = os.path.join(d, 'python')
        with open(stub, 'w') as fh:
            fh.write('#!/bin/sh\nprintf "%s\\n" "$@" > ' + shlex.quote(argv_log)
                     + '\nexit 0\n')
        os.chmod(stub, 0o755)
        script = (
            'set -euo pipefail\n'
            f'PATH={shlex.quote(d)}:$PATH\n'
            # FIELD deliberately not 012: with the fixture and the assertion
            # agreeing on the same literal, an --obs-token hardcoded to
            # "o012" is indistinguishable from the interpolation.
            'BASE=/tmp/base\nFIELD=007\nFILTERS="F162M F480M"\nit=2\n'
            'RETIE_RUN_START=20260818T000000Z\n'
            'RETIE_ACCEPT_RESIDUAL_MAS=15\nPIPE_ROOT=/tmp/pipe\n'
            + body)
        r = subprocess.run(['bash', '-c', script], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        assert os.path.exists(argv_log), (
            'the fixed-point check never ran -- the guard around it is '
            'unreachable')
        with open(argv_log) as fh:
            argv = [ln.rstrip('\n') for ln in fh]

    assert '--expect-filters' in argv, argv
    assert argv[argv.index('--expect-filters') + 1] == 'F162M F480M', argv
    assert argv[argv.index('--obs-token') + 1] == 'o007', argv
    assert argv[argv.index('--accept-below-mas') + 1] == '15', argv


def test_the_acceptance_ceiling_cap_IS_the_gate_it_is_derived_from():
    """The cap was unpinned across a 600x range -- 15, 50 and 9999 all left the
    suite green, and only 1e5 and 1e9 were caught, by two hardcoded values.

    It is now derived: the quantity being waived is a PER-EXPOSURE residual, and
    the nearest downstream gate that sees one is the m7 local-cell gate.  A cap
    above it admits floors that spend a whole m3-m7 chain and then fail at m7 on
    the same number.  Pin the identity, not a literal, so the two move together.
    """
    from jwst_gc_pipeline.photometry.retie_fixed_point import (
        MAX_ACCEPT_CEILING_MAS)
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        LOCAL_CELL_TOL_MAS)
    assert MAX_ACCEPT_CEILING_MAS == float(LOCAL_CELL_TOL_MAS), (
        f'the acceptance cap ({MAX_ACCEPT_CEILING_MAS}) has drifted from the '
        f'm7 local-cell gate ({LOCAL_CELL_TOL_MAS}) it is derived from')


def test_EXECUTED_a_whitespace_only_FILTERS_stops_before_the_reduce():
    """`${FILTERS:?}` rejects "" and passes " ".  FILTERS is both the reduce
    list and the coverage declaration, so a whitespace-only value declares
    nothing while looking set -- and the Python side only sees it at `it >= 2`,
    which is one full ~7 h reduce pass later."""
    src = _src()
    start = src.index('FILTERS=${FILTERS:?set FILTERS}')
    end = src.index('MODULES=${MODULES:-nrcb}', start)
    body = src[start:end]
    r = subprocess.run(['bash', '-c', 'set -euo pipefail\nFILTERS="   "\n' + body],
                       capture_output=True, text=True)
    assert r.returncode != 0, 'a whitespace-only FILTERS must stop the run'
    assert 'whitespace only' in r.stderr, r.stderr
    ok = subprocess.run(
        ['bash', '-c', 'set -euo pipefail\nFILTERS="F162M F480M"\n' + body],
        capture_output=True, text=True)
    assert ok.returncode == 0, ok.stderr
