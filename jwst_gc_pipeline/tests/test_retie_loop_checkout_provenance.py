"""A re-tie loop must say which code it is running, and refuse to start on old code.

`run_field_retie_loop.sh` drives a field's reduce -> catalog-to-m2 -> correct
cycle for as long as it takes to converge.  bash reads a script ONCE, at launch,
so a loop is frozen at whatever its checkout contained when it started.  At
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

Two behaviours are pinned here.  Reporting the checkout each iteration, so a log
answers "which code produced this" on its own; and refusing to START from a
checkout behind its upstream, which is enforced at launch only -- there no work
is lost, whereas stopping a running loop the same way would discard hours of
reduce.
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


# --- the refusal ----------------------------------------------------------

def test_a_current_checkout_is_allowed(tmp_path):
    repo = _repo(tmp_path, 'current')
    out = _source_helpers(f'assert_checkouts_current "{repo}"')
    assert out.returncode == 0, out.stdout + out.stderr


def test_a_checkout_behind_its_upstream_is_refused(tmp_path):
    repo = _repo(tmp_path, 'stale', behind=3)
    out = _source_helpers(f'assert_checkouts_current "{repo}"')
    assert out.returncode != 0, out.stdout
    assert 'REFUSING to start' in out.stdout
    assert '3 commit(s) behind' in out.stdout


def test_the_refusal_survives_a_long_commit_list(tmp_path):
    """The listing must not kill the script before the verdict is printed.

    `git log ... | head -20` makes git take SIGPIPE once the reader stops;
    `pipefail` turns that into 141 and `set -e` turns 141 into an exit -- at the
    listing, before the refusal.  Same shape as #366, where `set -e` killed the
    loop at the sbatch assignment.

    Reproducing it needs BOTH conditions, which is why a naive fixture misses
    it: more than 20 commits (or `head` never closes the pipe) AND a listing
    over the ~64 KiB pipe buffer (or git finishes writing before it is closed).
    29 commits with 6 KiB subjects clears both; 120 commits with `c0`-style
    subjects is ~1.5 KiB and does not, and passed against the unfixed code.
    """
    repo = _repo(tmp_path, 'verystale', behind=29, subject_chars=6000)
    out = _source_helpers(f'assert_checkouts_current "{repo}"')
    assert out.returncode != 0, out.stdout
    assert 'REFUSING to start' in out.stdout, (
        'the refusal was never printed -- the listing exited first: '
        + out.stdout[-400:])
    assert '29 commit(s) behind' in out.stdout
    assert 'and 9 more' in out.stdout


def test_an_undeterminable_distance_warns_but_does_not_refuse(tmp_path):
    """No upstream ref to compare against is UNKNOWN, not stale.

    Refusing here would block every environment that cannot reach the remote,
    which is a larger population than the one this guard protects.
    """
    repo = _repo(tmp_path, 'noupstream')
    subprocess.run(['git', '-C', str(repo), 'remote', 'remove', 'origin'],
                   check=True, env={'PATH': '/usr/bin:/bin', 'HOME': str(tmp_path)})
    out = _source_helpers(f'assert_checkouts_current "{repo}"')
    assert out.returncode == 0, out.stdout
    assert 'cannot tell whether' in out.stdout
    assert 'REFUSING' not in out.stdout


def test_the_refusal_can_be_overridden_deliberately():
    """Running old code on purpose must stay possible, and must be visible."""
    src = _src()
    assert 'RETIE_ALLOW_STALE_CHECKOUT' in src
    assert re.search(r'RETIE_ALLOW_STALE_CHECKOUT.*?\n.*?staleness check disabled',
                     src), 'the override must announce itself in the log'


def test_the_check_runs_before_any_reduce_is_submitted():
    """At launch nothing is lost; mid-run it would discard hours of reduce."""
    src = _src()
    assert src.index('assert_checkouts_current "$HERE"') < src.index(
        'for ((it=1; it<=MAXITER;')


# --- the mid-run edit note ------------------------------------------------

def test_a_mid_run_edit_to_the_script_is_reported(tmp_path):
    """bash does not re-read the file, so an edit changes nothing that is running.

    Reported rather than acted on: whether a loop may adopt code mid-run is a
    decision about its contract, which #364 leaves open.
    """
    out = _source_helpers(
        'SELF_SUM_AT_LAUNCH=deadbeef; warn_if_self_changed')
    assert out.returncode == 0, out.stderr
    assert 'has changed on disk' in out.stdout
    assert 'still the old' in out.stdout


def test_no_note_when_the_script_is_untouched():
    out = _source_helpers('warn_if_self_changed; echo QUIET')
    assert out.returncode == 0, out.stderr
    assert 'has changed on disk' not in out.stdout
    assert 'QUIET' in out.stdout
