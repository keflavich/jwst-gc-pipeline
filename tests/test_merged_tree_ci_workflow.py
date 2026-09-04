"""CI must re-test the MERGED tree when ``main`` moves, and must not hide a red suite.

Issue #249.  ``tests.yml`` runs when a branch is pushed and never again, so a
green mark on a pull request describes that branch merged into the ``main`` of
that moment.  Two branches that are each green, each mergeable, and that touch
different lines can still produce a ``main`` that raises: #235 + #243 gave
``NameError: name 'pool' is not defined`` on every call to
``update_offsets_table``, and #426 + #435 gave 21 failures in
``test_stage12_loop_behavior.py``.  Both reached ``main`` and were found by
hand about an hour later.

``retest-open-prs.yml`` closes that by re-running the suite on the merge result
every time ``main`` moves.  The properties asserted here are the ones whose
loss would make it report green while the merged tree is broken:

* it must PERFORM the merge against an explicit base.  ``refs/pull/N/merge`` is
  recomputed asynchronously by GitHub after a push to ``main``, and
  ``gh run rerun`` reuses the original event's ``GITHUB_SHA`` -- both replay a
  stale merge and call it green, which is the defect, not the fix.
* the suite's exit status must reach the job.  PR #374 was an earlier attempt
  at this check; it passed on a tree that could not import, because it
  discarded pytest's exit code and searched the output for ``ERROR`` lines that
  ``-rf`` never prints.
* both workflows must run ONE definition of the suite.  A second copy of the
  install/pytest steps drifts, and then the merged-tree check tests something
  other than what the pull request check tested.
* a conflicting pull request must get NO job.  A job that finds the conflict
  itself, skips the suite step and concludes success puts a green mark on a
  tree nobody built -- ``PR #132 merged into main  pass  7s`` beside
  ``PR #544 merged into main  pass  29m58s`` -- which is the state being
  guarded against, produced by the guard.
* ``git merge --abort`` must be guarded.  It exits 128 when the merge left no
  MERGE_HEAD, and under ``set -euo pipefail`` that ends the loop before the
  combined tree is assembled.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip('yaml')

_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS = _ROOT / '.github' / 'workflows'
#: the single install+pytest definition both workflows must use
SUITE_ACTION = _ROOT / '.github' / 'actions' / 'pytest-suite' / 'action.yml'
MERGED_TREE = _WORKFLOWS / 'retest-open-prs.yml'
TESTS = _WORKFLOWS / 'tests.yml'

#: PyYAML resolves the bare key ``on:`` to the boolean True (YAML 1.1).
ON = True


def _load(path):
    assert path.is_file(), f'{path} is missing'
    return yaml.safe_load(path.read_text())


def _steps(job):
    return job.get('steps') or []


def _all_steps(workflow):
    for job in workflow['jobs'].values():
        yield from _steps(job)


def _suite_steps(job):
    """The steps of ``job`` that run the one shared install+pytest definition."""
    return [step for step in _steps(job)
            if step.get('uses') == './.github/actions/pytest-suite']


def test_merged_tree_workflow_runs_when_main_moves():
    """Without a ``push: main`` trigger nothing recomputes a stale green mark."""
    wf = _load(MERGED_TREE)
    push = wf[ON]['push']
    assert push['branches'] == ['main'], (
        'the merged-tree check must fire on every push to main; that is the '
        'event the stale mark is stale with respect to')


def test_merged_tree_workflow_merges_rather_than_replaying_a_ref():
    """``refs/pull/N/merge`` and ``gh run rerun`` both replay a stale merge."""
    body = MERGED_TREE.read_text()
    assert re.search(r'git\s+merge\b', body), (
        'the workflow must perform the merge itself against an explicit base '
        'SHA')
    # comments explain why these are wrong, so only look at executable lines
    code = '\n'.join(line for line in body.splitlines()
                     if not line.lstrip().startswith('#'))
    assert 'refs/pull/${{ matrix.pr }}/merge' not in code
    assert not re.search(r'refs/pull/\S*/merge', code), (
        "checking out GitHub's precomputed merge ref can test a merge with a "
        'main that has already moved on -- the defect this workflow exists to '
        'catch')
    assert 'run rerun' not in code, (
        'a re-run reuses the original event GITHUB_SHA, i.e. the merge commit '
        'computed when the run was first triggered')


def test_both_workflows_run_the_same_suite_definition():
    """One definition of install+pytest, used by both, so they cannot drift."""
    assert SUITE_ACTION.is_file(), f'{SUITE_ACTION} is missing'
    rel = './.github/actions/pytest-suite'
    for path in (TESTS, MERGED_TREE):
        wf = _load(path)
        uses = [step.get('uses') for step in _all_steps(wf)]
        assert rel in uses, (
            f'{path.name} must run the suite through {rel}, not through its '
            'own copy of the install and pytest steps')
    # and no workflow may invoke pytest directly any more
    for path in sorted(_WORKFLOWS.glob('*.yml')):
        for step in _all_steps(_load(path)):
            assert 'pytest ' not in (step.get('run') or ''), (
                f'{path.name} invokes pytest directly; the single definition '
                f'is {rel}')


def test_the_suite_exit_status_is_not_discarded():
    """PR #374 reported green on a tree that could not import."""
    action = _load(SUITE_ACTION)
    run_steps = [step for step in action['runs']['steps'] if 'run' in step]
    pytest_steps = [step for step in run_steps if 'pytest ' in step['run']]
    assert len(pytest_steps) == 1, 'expected exactly one pytest invocation'
    script = pytest_steps[0]['run']
    line = next(ln.strip() for ln in script.splitlines()
                if ln.strip().startswith('pytest '))
    for masker in ('|| true', '|| echo', '; true', '&& true', 'set +e', '|'):
        assert masker not in line, (
            f'{masker!r} in {line!r} discards the suite exit status')
    assert 'continue-on-error' not in SUITE_ACTION.read_text()

    for path in (TESTS, MERGED_TREE):
        wf = _load(path)
        for name, job in wf['jobs'].items():
            assert not job.get('continue-on-error'), (
                f'{path.name}:{name} is continue-on-error, so a red suite '
                'leaves the run green')
            for step in _steps(job):
                assert not step.get('continue-on-error'), (
                    f'{path.name}:{name} has a continue-on-error step')


def test_every_open_pr_is_accounted_for():
    """A merged-tree check that silently skips most PRs is not a check.

    ``gh pr list`` answers ``mergeable: UNKNOWN`` until GitHub computes it --
    4 of the 5 open pull requests on 2026-09-04 -- so selecting on
    ``mergeable == "MERGEABLE"`` drops most of the population without saying
    so.  Enumerate them all, decide mergeability by trying the merge, and put
    each one in exactly one of the two lists: tested, or recorded as
    conflicting.
    """
    body = MERGED_TREE.read_text()
    code = '\n'.join(line for line in body.splitlines()
                     if not line.lstrip().startswith('#'))
    assert 'MERGEABLE' not in code, (
        'do not filter the enumeration on mergeable; it reads UNKNOWN for '
        'pull requests GitHub has not evaluated yet')

    wf = _load(MERGED_TREE)
    trial = [step for step in _steps(wf['jobs']['enumerate'])
             if 'git merge' in (step.get('run') or '')]
    assert len(trial) == 1, (
        'the `enumerate` job must decide mergeability by TRYING the merge, '
        'once, over the whole open-pull-request list')
    assert 'outputs.prs' in yaml.dump(trial[0].get('env') or {}), (
        'the trial must run over the full enumerated list, not a subset')
    script = trial[0]['run']
    for key in ('mergeable', 'conflicts'):
        assert f'{key}=' in script, (
            f'the trial must record the {key} pull requests; a pull request '
            'that appears in neither list has gone unreported')


def test_the_combined_tree_is_tested_too():
    """The two recorded incidents are visible only with BOTH branches merged.

    The detector has to PIN that job.  A previous version of this test looked
    for a non-matrix job whose body mentioned ``jq``, which the ``enumerate``
    job also satisfies -- so deleting the whole ``all-together:`` job left all
    seven tests passing.  A job counts here only if it RUNS THE SUITE, and the
    combined one only if it merges the enumerated list rather than one branch.
    """
    wf = _load(MERGED_TREE)
    jobs = wf['jobs']
    runs_suite = {name for name, job in jobs.items() if _suite_steps(job)}
    per_pr = sorted(name for name in runs_suite
                    if 'matrix' in (jobs[name].get('strategy') or {}))
    assert per_pr, 'expected a per-pull-request matrix job that runs the suite'
    for name in per_pr:
        assert jobs[name]['strategy']['fail-fast'] is False, (
            'one broken pull request must not cancel the others')

    combined = sorted(runs_suite - set(per_pr))
    assert combined, (
        'expected a job that merges every open pull request together AND runs '
        'the suite on the result: it is the only arrangement that sees a '
        'two-branch interaction while both are still open (#235 + #243, '
        '#426 + #435)')
    for name in combined:
        merges = [step for step in _steps(jobs[name])
                  if 'git merge' in (step.get('run') or '')]
        assert merges, f'{name}: runs the suite but merges nothing into it'
        script = '\n'.join(step['run'] for step in merges)
        assert re.search(r'for\s+\w+\s+in\b', script), (
            f'{name}: must merge the pull requests in a loop, not one branch')
        env = yaml.dump([step.get('env') or {} for step in merges])
        assert 'needs.enumerate.outputs.' in env, (
            f'{name}: the loop must be fed the list from the `enumerate` job')


def test_a_conflicting_pr_gets_no_green_mark():
    """A job that finds a conflict and skips the suite still concludes success.

    Measured on this pull request's own run: ``PR #132 merged into main  pass
    7s`` and ``#140  pass  10s``, beside ``PR #544  pass  29m58s``.  The two
    seven-second jobs hit a conflict, skipped the pytest step and went green --
    a green mark describing a tree nobody built, which is the state #249 exists
    to catch.

    So mergeability is decided in ``enumerate``, before any job exists, and
    only the mergeable pull requests enter the matrix.  ``matrix`` is not among
    the contexts available to ``jobs.<job_id>.if`` (github, needs, vars,
    inputs), so a conflicting pull request cannot be rendered as a *skipped*
    job; it gets none, and is named by the ``conflicts`` job instead.
    """
    wf = _load(MERGED_TREE)
    jobs = wf['jobs']
    outputs = jobs['enumerate'].get('outputs') or {}
    for key in ('mergeable', 'conflicts'):
        assert key in outputs, (
            f'the `enumerate` job must publish `{key}`; the conflict decision '
            'belongs there, before a job is created for the pull request')

    matrix_jobs = [name for name, job in jobs.items()
                   if 'matrix' in (job.get('strategy') or {})]
    assert matrix_jobs, 'expected a per-pull-request matrix job'
    for name in matrix_jobs:
        src = yaml.dump(jobs[name]['strategy']['matrix'])
        assert 'outputs.mergeable' in src, (
            f'{name}: the matrix must be built from the MERGEABLE list, so '
            'that a conflicting pull request gets no job rather than a green '
            'one that ran nothing')

    for name, job in jobs.items():
        for step in _suite_steps(job):
            assert 'if' not in step, (
                f'{name}: the suite step is conditional (if: {step["if"]!r}); '
                'a skipped step leaves the job green, which is the same defect '
                'in a different place')

    assert 'outputs.conflicts' in MERGED_TREE.read_text(), (
        'the conflicting pull requests must still be named in the run')


def test_a_failed_merge_is_cleaned_up_without_ending_the_script():
    """``git merge --abort`` is not a safe cleanup on its own.

    It exits 128 when the merge failed WITHOUT leaving MERGE_HEAD -- e.g.
    ``fatal: refusing to merge unrelated histories``.  Both merge loops run
    under ``set -euo pipefail``, so an unguarded ``--abort`` ends the script:
    in ``enumerate`` no lists are published, and in ``all-together`` the
    combined tree -- the one arrangement that catches the recorded incidents --
    is never assembled.
    """
    aborts = [line.strip() for line in MERGED_TREE.read_text().splitlines()
              if 'git merge --abort' in line
              and not line.lstrip().startswith('#')]
    assert aborts, 'a failed merge must be cleaned up before the loop goes on'
    for line in aborts:
        assert re.search(r'git merge --abort\b[^|]*\|\|\s*git reset --hard',
                         line), (
            f'{line!r}: guard it with `|| git reset --hard <sha>`; --abort '
            'exits 128 when the merge left no MERGE_HEAD, and under '
            '`set -euo pipefail` that ends the loop')


def test_git_merge_abort_alone_exits_nonzero_after_a_refused_merge(tmp_path):
    """The measurement behind the assertion above, run rather than asserted."""
    if shutil.which('git') is None:
        pytest.skip('git is not installed')
    repo = tmp_path / 'repo'
    repo.mkdir()

    def git(*args):
        return subprocess.run(('git',) + args, cwd=repo, capture_output=True,
                              text=True, check=True)

    git('init', '-q')
    git('config', 'user.email', 'merged-tree-check@invalid')
    git('config', 'user.name', 'merged-tree check')
    (repo / 'a').write_text('a\n')
    git('add', 'a')
    git('commit', '-qm', 'base')
    base = git('rev-parse', 'HEAD').stdout.strip()
    # an unrelated history: `git merge` refuses it outright, so it fails
    # WITHOUT writing MERGE_HEAD -- the state `--abort` cannot undo
    git('checkout', '-q', '--orphan', 'other')
    (repo / 'b').write_text('b\n')
    git('add', 'b')
    git('commit', '-qm', 'other')
    other = git('rev-parse', 'HEAD').stdout.strip()
    git('checkout', '-q', '--detach', base)

    def run(cleanup):
        script = (f'set -euo pipefail\n'
                  f'before=$(git rev-parse HEAD)\n'
                  f'if git merge --no-edit --no-ff {other}; then\n'
                  f'  echo merged\n'
                  f'else\n'
                  f'  {cleanup}\n'
                  f'fi\n'
                  f'echo loop-continues\n')
        return subprocess.run(['bash', '-c', script], cwd=repo,
                              capture_output=True, text=True)

    plain = run('git merge --abort')
    assert plain.returncode != 0, (
        'expected the unguarded --abort to end the script')
    assert 'MERGE_HEAD missing' in plain.stderr
    assert 'loop-continues' not in plain.stdout, (
        'the loop must be shown to die, or this test proves nothing')

    guarded = run('git merge --abort || git reset --hard "$before"')
    assert guarded.returncode == 0, guarded.stderr
    assert 'loop-continues' in guarded.stdout
    assert git('rev-parse', 'HEAD').stdout.strip() == base, (
        'the fallback must leave the tree back at the base commit')


def test_the_suite_definition_is_not_taken_from_the_tree_under_test():
    """The merged tree must be built in a subdirectory, not over the workspace.

    The first run of this workflow failed on every mergeable pull request with
    ``Can't find 'action.yml' ... under .github/actions/pytest-suite``: the
    merge is built from ``main``, and ``main`` does not carry a definition that
    has not landed yet.  Checking the base out over the workspace root also
    hands the pull request under test the composite action that judges it.
    Both are fixed by checking the workflow's own ref out at the root and the
    merged tree into ``merged/``.
    """
    wf = _load(MERGED_TREE)
    for name, job in wf['jobs'].items():
        steps = _steps(job)
        checkouts = [s for s in steps
                     if str(s.get('uses', '')).startswith('actions/checkout')]
        suite = [s for s in steps
                 if s.get('uses') == './.github/actions/pytest-suite']
        if not suite:
            continue
        assert len(checkouts) == 2, (
            f'{name}: expected two checkouts -- this workflow ref at the root '
            'for the suite definition, and the base into a subdirectory')
        root, tree = checkouts
        assert 'path' not in (root.get('with') or {}), (
            f'{name}: the first checkout must land at the workspace root, '
            'where `uses: ./.github/actions/pytest-suite` resolves')
        assert 'ref' not in (root.get('with') or {}), (
            f'{name}: the first checkout must take the default ref, i.e. the '
            'commit this workflow is running from')
        path = (tree.get('with') or {}).get('path')
        assert path, (
            f'{name}: the base checkout must use `path:` so it does not '
            'overwrite the suite definition')
        assert (suite[0].get('with') or {}).get('path') == path, (
            f'{name}: the suite must be pointed at {path!r}, otherwise it '
            'tests the workflow ref instead of the merged tree')
        for step in steps:
            if 'git merge' in (step.get('run') or ''):
                assert step.get('working-directory') == path, (
                    f'{name}: the merge must happen inside {path!r}')
