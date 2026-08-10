"""A re-tie loop must say which code it is running.

`run_field_retie_loop.sh` drives a field's reduce -> catalog-to-m2 -> correct
cycle for as long as it takes to converge.  bash parses each command as it
reaches it and never re-reads what it has already run, so a loop is effectively
frozen at whatever its checkout contained when it started.  At
MAXITER=12 and ~7 h a pass that is a fortnight, and a safety guard merged on day
two never reaches it.

Observed (#364): sgrc's loop ran 2026-08-07 to 08-09 from a checkout that
predates `reduce_fully_succeeded` -- the guard that stops a partially-failed
reduce being cataloged -- so for two days it cataloged without checking its
reduce had succeeded.  Measured on that same checkout today:

    /blue/.../jwst-gc-pipeline-parscache @ facd471 (2026-08-07),
        107 commit(s) behind origin/main
    grep -c reduce_fully_succeeded  ->  0

Nothing in the loop's output said so, and it could not have: the code that would
report a missing guard is part of what is missing.

What is pinned here is REPORTING: the checkout, at launch and at every
iteration, so a log answers "which code produced this" on its own.

Deliberately NOT a refusal, and the two designs that were tried are recorded in
the script beside the code, because both were measured and both are wrong.  A
commit-distance gate refuses 395 of this repository's 400 worktrees, i.e. the
branch workflow CLAUDE.md mandates.  A named-guard gate -- #364's own
"minimum-version assertion" -- PASSES the actual checkout from #364, because the
guard missing there (`reduce_fully_succeeded`) lives in the loop script itself,
and a script cannot check itself for a guard whose absence also removes the
check.  That is #364's central observation and no self-check escapes it: the
code that would report a missing guard is part of what is missing.
"""
import pathlib
import re
import subprocess

LOOP = (pathlib.Path(__file__).parents[2] / 'scripts' / 'reduction'
        / 'run_field_retie_loop.sh')


def _src():
    return LOOP.read_text()


def _source_helpers(script, env=None):
    """Run `script` with the loop's helper functions in scope.

    `RETIE_LOOP_SOURCE_ONLY` makes the loop return before its first iteration,
    which is how the existing loop tests reach its functions without submitting
    anything.  Run under the loop's OWN `set -euo pipefail`, because several of
    the hazards being tested only exist under those options.
    """
    base = dict(RETIE_LOOP_SOURCE_ONLY='1', PROPOSAL='4147', FIELD='012',
                TARGET='sgrc', FILTERS='F115W', OFFSETS_TBL='/tmp/unused.csv',
                PATH='/usr/bin:/bin:/usr/local/bin', HOME='/tmp')
    base.update(env or {})
    body = (f'set -euo pipefail\nsource "{LOOP}" >/dev/null 2>&1\n' + script)
    return subprocess.run(['bash', '-c', body], capture_output=True, text=True,
                          env=base, cwd=str(LOOP.parents[2]))


def _repo(tmp_path, name, behind=0, subject_chars=8):
    """A throwaway git repo, optionally with `behind` commits on origin/main.

    `subject_chars` pads each commit subject.  It matters: the SIGPIPE hazard
    below needs the listing to exceed a 64 KiB pipe buffer, and short subjects
    never do.
    """
    origin, work = tmp_path / f'{name}-origin', tmp_path / name
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(origin)], check=True)
    env = {'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@t',
           'GIT_COMMITTER_NAME': 't', 'GIT_COMMITTER_EMAIL': 't@t',
           'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path)}
    for i in range(1 + behind):
        subprocess.run(['git', '-C', str(origin), 'commit', '-q', '--allow-empty',
                        '-m', f'c{i} ' + 'x' * subject_chars], check=True, env=env)
    subprocess.run(['git', 'clone', '-q', str(origin), str(work)], check=True,
                   env=env)
    if behind:
        subprocess.run(['git', '-C', str(work), 'reset', '-q', '--hard',
                        f'HEAD~{behind}'], check=True, env=env)
    return work


# --- the report -----------------------------------------------------------

def test_the_provenance_line_names_the_commit_and_the_distance(tmp_path):
    repo = _repo(tmp_path, 'current')
    out = _source_helpers(f'checkout_provenance "{repo}"')
    assert out.returncode == 0, out.stderr
    line = out.stdout.strip()
    assert str(repo) in line
    assert 'behind' in line
    assert re.search(r'@ [0-9a-f]{7,}', line), line


def test_a_non_repository_reports_unknown_rather_than_stopping(tmp_path):
    """A loop must not die because it was launched from outside a checkout."""
    plain = tmp_path / 'plain'
    plain.mkdir()
    out = _source_helpers(f'checkout_provenance "{plain}"')
    assert out.returncode == 0, out.stderr
    assert 'provenance unknown' in out.stdout


def test_a_missing_directory_reports_rather_than_stopping():
    out = _source_helpers('checkout_provenance /nonexistent-xyz')
    assert out.returncode == 0, out.stderr
    assert 'does not exist' in out.stdout


def test_every_iteration_reports_the_checkout_not_only_the_launch():
    """An iteration's log is what gets read when its results are questioned.

    Printing once at launch is not enough: a two-week log is read in pieces,
    and the piece someone reads is the iteration that produced the number they
    are asking about.
    """
    src = _src()
    body = src[src.index('for ((it=1; it<=MAXITER;'):]
    assert 'checkout_provenance' in body, (
        'the per-iteration banner does not report the checkout, so an '
        'iteration log cannot say which code produced it')


# --- the distance report, which is a REPORT ---------------------------------

def test_a_checkout_behind_its_upstream_is_reported_not_refused(tmp_path):
    """Reported, and the loop still starts.

    Refusing on distance was tried and measured against this repository's own
    400 worktrees: 395 behind, 156 deliberately both ahead and behind (a topic
    branch under test, which is how a re-tie is normally driven), and a detached
    HEAD at a release tag refused too.  A rule that refuses 99% of checkouts is
    a rule the operator overrides by habit.
    """
    repo = _repo(tmp_path, 'stale', behind=3)
    out = _source_helpers(f'warn_if_behind "{repo}"; echo STILL_RUNNING')
    assert out.returncode == 0, out.stdout + out.stderr
    assert '3 commit(s) behind' in out.stdout
    assert 'STILL_RUNNING' in out.stdout
    assert 'REFUS' not in out.stdout.upper()


def test_the_report_says_the_distance_is_against_a_local_ref(tmp_path):
    """"0 behind" is not an all-clear, and the log must not imply it is.

    `rev-list HEAD..origin/main` compares against the LOCAL copy of that ref,
    which is only as fresh as the last fetch, so a checkout that has not fetched
    for a month reads as current.
    """
    repo = _repo(tmp_path, 'stale', behind=2)
    out = _source_helpers(f'warn_if_behind "{repo}"')
    assert 'LOCAL' in out.stdout
    assert 'last fetch' in out.stdout


def test_the_zero_behind_line_carries_the_caveat_too(tmp_path):
    """The case the whole design argument turns on, and the one that had no caveat.

    `warn_if_behind` only speaks when the distance is non-zero.  But the
    dangerous reading is `0 commit(s) behind` on a checkout that has not fetched
    in a month -- so the qualifier has to be on the provenance line itself,
    which is what a reader actually sees.
    """
    repo = _repo(tmp_path, 'current')                 # 0 behind by construction
    out = _source_helpers(f'checkout_provenance "{repo}"')
    assert out.returncode == 0, out.stderr
    assert '0 commit(s) behind' in out.stdout
    assert 'local ref' in out.stdout
    assert 'last fetched' in out.stdout


def test_a_never_fetched_checkout_says_so(tmp_path):
    """"0 behind, last fetched never" is the shape that must not read as clean."""
    repo = _repo(tmp_path, 'nofetch')
    fetch_head = repo / '.git' / 'FETCH_HEAD'
    if fetch_head.exists():
        fetch_head.unlink()
    out = _source_helpers(f'checkout_provenance "{repo}"')
    assert 'last fetched never' in out.stdout


def test_the_launch_and_iteration_lines_name_the_same_path():
    """One checkout must not appear under two different paths in one log."""
    src = _src()
    banner = src[src.index('CHECKOUT PROVENANCE'):src.index('for ((it=1; it<=MAXITER;')]
    body = src[src.index('for ((it=1; it<=MAXITER;'):]
    assert 'checkout_provenance "$_here_top_or_here"' in banner
    assert 'checkout_provenance "$_here_top_or_here"' in body


def test_the_report_says_what_being_behind_costs(tmp_path):
    """The number alone means nothing to a reader; the consequence is the point."""
    repo = _repo(tmp_path, 'stale', behind=2)
    out = _source_helpers(f'warn_if_behind "{repo}"')
    assert 'NOT in this run' in out.stdout
    assert 'restart the loop' in out.stdout.lower()


def test_an_undeterminable_distance_is_reported_too(tmp_path):
    repo = _repo(tmp_path, 'noupstream')
    subprocess.run(['git', '-C', str(repo), 'remote', 'remove', 'origin'],
                   check=True, env={'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path)})
    out = _source_helpers(f'warn_if_behind "{repo}"')
    assert out.returncode == 0, out.stdout
    assert 'cannot tell how far' in out.stdout


def test_a_long_commit_list_cannot_kill_the_loop(tmp_path):
    """No pipeline into `head`, so nothing here can take SIGPIPE.

    An earlier version listed the missing commits with `git log | head -20`.
    That makes git take SIGPIPE once the reader stops; `pipefail` turns it into
    141 and `set -e` turns 141 into an exit -- the same shape as #366, where
    `set -e` killed this loop at the sbatch assignment.  Reproducing it needs
    both more than 20 commits and a listing over the pipe buffer, which is why a
    naive fixture misses it; 29 commits with 6 KiB subjects clears both.
    """
    repo = _repo(tmp_path, 'verystale', behind=29, subject_chars=6000)
    out = _source_helpers(f'warn_if_behind "{repo}"; echo STILL_RUNNING')
    assert out.returncode == 0, out.stdout
    assert 'STILL_RUNNING' in out.stdout
    assert '29 commit(s) behind' in out.stdout


def test_the_report_runs_before_any_reduce_is_submitted():
    src = _src()
    assert src.index('warn_if_behind "$_here_top_or_here"') < src.index(
        'for ((it=1; it<=MAXITER;')


def test_the_loop_never_refuses_on_provenance_alone():
    """The whole design decision, pinned.

    Both stronger designs were tried and measured, and both are wrong -- a
    distance gate refuses the mandated worktree workflow, and a named-guard gate
    PASSES the actual #364 checkout, because the guard missing there lives in
    this script and a script cannot check itself for a guard whose absence also
    removes the check.  If a future change adds a refusal here it needs a new
    argument, not this test deleted quietly.
    """
    src = _src()
    banner = src[src.index('CHECKOUT PROVENANCE'):src.index('for ((it=1; it<=MAXITER;')]
    # One exit is allowed here and it is not a refusal: RETIE_PROVENANCE_ONLY is
    # the operator asking to read the banner and stop.  Anything else exiting on
    # what the provenance SAYS is the design this PR rejected.
    exits = [l.strip() for l in banner.splitlines() if l.strip().startswith('exit ')]
    assert exits == ['exit 0'], (
        f'the provenance banner must not stop the loop on what it finds; '
        f'exits present: {exits}')
    assert 'RETIE_PROVENANCE_ONLY' in banner, (
        'the only permitted exit must be the opt-in inspect mode')


def test_the_reason_refusal_was_rejected_is_recorded():
    """A measured negative result is worth as much as the fix, if it is written down."""
    src = _src()
    src_new = src[src.index('# Report -- never refuse'):]
    # Assert the ARGUMENT, not a literal count.  The worktree population is
    # mutable -- it was 400 when this was written and 411 a day later -- so
    # pinning the number means re-measuring the comment breaks the test, which
    # is backwards: it pressures the comment to stay stale.
    assert 'worktree' in src_new
    assert re.search(r'\b\d{3}\b.*behind', src_new) or \
        re.search(r'behind.*\b\d{3}\b', src_new), (
            'the false-refusal measurement must stay next to the code it '
            'explains, with its counts')
    assert 'reduce_fully_succeeded' in src_new
    assert 'cannot check itself' in src_new


# --- the mid-run edit note ------------------------------------------------

def test_a_mid_run_edit_to_the_script_is_reported(tmp_path):
    """A mid-run edit makes the run unpredictable, and the note must say so.

    bash reads a script incrementally, by byte offset.  The iteration loop is
    parsed before it runs, so an edit cannot change the loop mid-flight -- but
    an edit that shifts offsets makes bash resume mid-token in the code AFTER
    the loop.  An earlier version of this note told the operator the running
    loop was simply "still the old version", which is reassurance at exactly the
    wrong moment.

    Reported rather than acted on: whether a loop may deliberately adopt code
    mid-run is a decision about its contract, which #364 leaves open.
    """
    out = _source_helpers(
        'SELF_SUM_AT_LAUNCH=deadbeef; warn_if_self_changed')
    assert out.returncode == 0, out.stderr
    assert 'has changed on disk' in out.stdout
    assert 'unpredictable' in out.stdout
    assert 'INCREMENTALLY' in out.stdout


def test_no_note_when_the_script_is_untouched():
    out = _source_helpers('warn_if_self_changed; echo QUIET')
    assert out.returncode == 0, out.stderr
    assert 'has changed on disk' not in out.stdout
    assert 'QUIET' in out.stdout


def test_maxiter_below_one_is_refused():
    """`MAXITER=0` is the obvious way to ask for a no-op, and it was not one.

    Values below 1 skip the iteration loop and fall straight through to the FULL
    m3-m7 cataloging submission at the bottom of the script.  On 2026-08-09 that
    submitted twelve jobs against sgrc 4147/012 (39044950, 39044953-39044961),
    ten of them RUNNING before they were cancelled, from someone trying to read
    the provenance banner.
    """
    out = subprocess.run(
        ['bash', str(LOOP)], capture_output=True, text=True,
        env=dict(MAXITER='0', PROPOSAL='4147', FIELD='012', TARGET='sgrc',
                 FILTERS='F115W', OFFSETS_TBL='/tmp/unused.csv',
                 PATH='/usr/bin:/bin:/usr/local/bin', HOME='/tmp'))
    assert out.returncode == 2, out.stdout + out.stderr
    assert 'REFUSING' in out.stdout
    assert 'do NOT make this a no-op' in out.stdout
    assert 'sbatch' not in out.stdout


def test_there_is_a_way_to_read_the_provenance_without_submitting():
    """The banner is the PR's whole product; it must be readable safely."""
    src = _src()
    assert 'RETIE_PROVENANCE_ONLY' in src
    # and it must exit before anything is submitted
    assert src.index('RETIE_PROVENANCE_ONLY') < src.index('for ((it=1; it<=MAXITER;')
