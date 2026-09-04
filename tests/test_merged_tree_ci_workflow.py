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
"""
import re
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
    so.  Enumerate them all; report the conflicting ones as conflicting.
    """
    body = MERGED_TREE.read_text()
    code = '\n'.join(line for line in body.splitlines()
                     if not line.lstrip().startswith('#'))
    assert 'MERGEABLE' not in code, (
        'do not filter the enumeration on mergeable; it reads UNKNOWN for '
        'pull requests GitHub has not evaluated yet')
    assert 'git merge --abort' in code, (
        'a conflicting merge must be aborted and recorded, not left to fail '
        'the job as if the tree were broken')


def test_the_combined_tree_is_tested_too():
    """The two recorded incidents are visible only with BOTH branches merged."""
    wf = _load(MERGED_TREE)
    jobs = wf['jobs']
    per_pr = [name for name, job in jobs.items()
              if 'matrix' in (job.get('strategy') or {})]
    assert per_pr, 'expected a per-pull-request matrix job'
    for name in per_pr:
        assert jobs[name]['strategy']['fail-fast'] is False, (
            'one broken pull request must not cancel the others')
    combined = [name for name, job in jobs.items()
                if name not in per_pr and 'steps' in job
                and 'jq' in yaml.dump(job)]
    assert combined, (
        'expected a job that merges every open pull request together: it is '
        'the only arrangement that sees a two-branch interaction while both '
        'are still open (#235 + #243, #426 + #435)')
