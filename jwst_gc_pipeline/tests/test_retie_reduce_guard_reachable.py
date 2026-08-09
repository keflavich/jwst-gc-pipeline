"""The reduce guard must be REACHED, not merely present.

`reduce_fully_succeeded` exists so a partially-failed reduce is never cataloged
-- its own comment cites sgrc iteration 3, which lost four LW filters to CRDS
504s and went on to catalog the mixture.

But the loop runs under `set -euo pipefail`, and

    red_out=$(sbatch --wait ...)

is a plain assignment: when the command fails, `set -e` exits the script THERE.
`sbatch --wait` returns nonzero as soon as any array task fails -- exactly the
case the guard diagnoses -- so the loop died before `red_rc=$?`, before
`echo "$red_out"`, and before the guard ran.

Observed on cloudef, 2026-08-09: F360M of four filters failed and the retie log
ends at

    [iter 1] reducing (fix_alignment applies consensus table if present: yes)

with no sbatch output, no job id, and no reason.  The loop stopped safely and
silently; sacct was the only way to learn anything had happened.
"""
import pathlib
import re
import subprocess

LOOP = (pathlib.Path(__file__).parents[2] / 'scripts' / 'reduction'
        / 'run_field_retie_loop.sh')


def _src():
    return LOOP.read_text()


def test_the_loop_still_runs_under_set_e():
    """The premise.  If `set -e` were dropped the assignment would be harmless
    -- and a dozen other failures would stop being fatal, which is worse."""
    assert re.search(r'^set -euo pipefail$', _src(), re.M)


def test_the_sbatch_assignment_cannot_kill_the_loop():
    """`|| red_rc=$?` is what keeps `set -e` from exiting at the assignment."""
    src = _src()
    start = src.index('red_out=$(sbatch --wait')
    # the statement ends where the command substitution closes, at the line
    # that submits the sbatch script
    end = src.index('submit_reduction.sbatch', start)
    stmt = src[start:src.index('\n', end)]
    assert '|| red_rc=' in stmt, (
        'a bare command substitution exits under set -e, so the guard below '
        f'never runs and a partial reduce stops the loop with no diagnosis; '
        f'statement was: {stmt!r}')


def test_red_rc_is_initialised_before_the_call():
    """With `|| red_rc=$?` the success path never assigns, so an uninitialised
    `red_rc` would trip `set -u` on the very next line."""
    src = _src()
    assert re.search(r'red_rc=0\s*\n\s*red_out=\$\(sbatch', src)


def test_the_guard_is_actually_called_with_the_return_code():
    assert re.search(r'reduce_fully_succeeded "\$red_jid" "\$NF" "\$red_rc"',
                     _src())


def test_the_guard_still_explains_itself():
    """Stopping is only half of it; the log has to say why."""
    src = _src()
    assert 'REDUCE DID NOT FULLY SUCCEED -- STOPPING before cataloging.' in src
    assert 'could not parse a job id from sbatch' in src


def test_shell_syntax_is_valid():
    assert subprocess.run(['bash', '-n', str(LOOP)]).returncode == 0


def test_a_failing_assignment_really_does_exit_under_set_e():
    """The mechanism, demonstrated rather than asserted -- this is the part that
    is easy to disbelieve."""
    bare = subprocess.run(
        ['bash', '-c', 'set -euo pipefail\nv=$(false)\necho REACHED'],
        capture_output=True, text=True)
    assert 'REACHED' not in bare.stdout

    guarded = subprocess.run(
        ['bash', '-c', 'set -euo pipefail\nrc=0\nv=$(false) || rc=$?\n'
                       'echo "REACHED rc=$rc"'],
        capture_output=True, text=True)
    assert 'REACHED rc=1' in guarded.stdout


# ---------------------------------------------------------------------------
# The SAME trap, three more times in the same eight lines (review of #366).
# `set -o pipefail` makes a non-matching grep fail the whole pipeline, so every
# `x=$(... | grep ...)` is a plain assignment that exits the script exactly when
# the pattern does not match -- which is the case each of them exists to handle.
# ---------------------------------------------------------------------------

def test_a_SUBMISSION_failure_still_reaches_the_guard():
    """An array-TASK failure prints a job id; a SUBMISSION failure (QOS limit,
    bad partition, malformed --export) prints an error with no leading number,
    so the job-id grep matched nothing and the loop died one line before the
    guard -- the same silent stop, for the other half of the failure space."""
    src = _src()
    line = next(ln for ln in src.splitlines() if 'red_jid=$(echo' in ln)
    assert line.rstrip().endswith('|| true'), line


def test_the_fin_jid_FALLBACK_can_actually_run():
    """`if [ -z "$fin_jid" ]` exists for 'the first pattern did not match', and
    that is precisely when the unguarded assignment above it exits."""
    src = _src()
    lines = [ln for ln in src.splitlines() if 'fin_jid=$(echo' in ln]
    assert len(lines) == 2, lines
    for ln in lines:
        assert ln.rstrip().endswith('|| true'), ln


def test_a_failed_cataloging_SUBMISSION_stops_with_a_reason():
    src = _src()
    assert 'chain_rc=0' in src
    assert 'bash "$HERE/submit_cataloging_perframe.sh") || chain_rc=$?' in src
    assert 'the cataloging submission FAILED' in src


def test_an_unparseable_finalize_id_stops_rather_than_waiting_on_nothing():
    """`wait_job ""` returns immediately, and the loop would read that as a
    finished job -- converged on a chain that was never queued."""
    assert 'could not parse a finalize job id' in _src()


def test_the_pipefail_trap_is_real_for_grep():
    """Demonstrated, like the assignment one: this is the part that makes the
    fallback dead code rather than merely redundant."""
    import subprocess
    bare = subprocess.run(
        ['bash', '-c', 'set -euo pipefail\n'
                       'out="sbatch: error: QOSMaxSubmitJobPerUserLimit"\n'
                       'jid=$(echo "$out" | grep -oE "^[0-9]+" | head -1)\n'
                       'echo "REACHED"'],
        capture_output=True, text=True)
    assert 'REACHED' not in bare.stdout

    guarded = subprocess.run(
        ['bash', '-c', 'set -euo pipefail\n'
                       'out="sbatch: error: QOSMaxSubmitJobPerUserLimit"\n'
                       'jid=$(echo "$out" | grep -oE "^[0-9]+" | head -1) || true\n'
                       'echo "REACHED jid=[$jid]"'],
        capture_output=True, text=True)
    assert 'REACHED jid=[]' in guarded.stdout
