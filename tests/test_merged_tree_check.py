"""The check that re-tests open pull requests against the current main.

The failure it exists for: two branches each pass their own tests, merge without
a git conflict, and produce a `main` that raises.  That happened when one branch
added a `pool` parameter to the function that writes measured astrometric
corrections into a field's offsets table, and another split the same function
into a locking wrapper plus a private body -- different lines, clean merge,
parameter no longer forwarded, 29 failures on a tree neither branch produced.
"""
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).resolve().parents[1]
           / 'scripts' / 'ci' / 'merged_tree_check.py')


def _load():
    spec = importlib.util.spec_from_file_location('merged_tree_check', _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


mtc = _load()


# ---------------------------------------------------------------------------
# Reading pytest's report
# ---------------------------------------------------------------------------

PYTEST_OUTPUT = """\
....F...E..
=========================== short test summary info ============================
FAILED jwst_gc_pipeline/photometry/tests/test_astrometry_checkpoint.py::test_pool - NameError: name 'pool' is not defined
ERROR tests/test_doc_code_references.py::test_alignment_config_table_matches_code
2 failed, 46 passed in 31.4s
"""


def test_the_failing_tests_are_read_by_identity_not_counted():
    """Two runs are compared by WHICH tests failed, so a merge that breaks a
    different test than main already breaks is still visible."""
    assert mtc.failing_test_ids(PYTEST_OUTPUT) == {
        'jwst_gc_pipeline/photometry/tests/test_astrometry_checkpoint.py::test_pool',
        'tests/test_doc_code_references.py::test_alignment_config_table_matches_code',
    }


def test_a_failure_message_containing_the_word_FAILED_is_not_a_test():
    """The summary line is anchored at the start, because an assertion message
    can quote the word."""
    output = ('FAILED a.py::real - AssertionError: expected FAILED b.py::fake\n'
              '  and continued FAILED c.py::alsofake\n')
    assert mtc.failing_test_ids(output) == {'a.py::real'}


def test_no_failures_reads_as_an_empty_set():
    assert mtc.failing_test_ids('1281 passed in 830s\n') == set()


# ---------------------------------------------------------------------------
# Subtracting the baseline
# ---------------------------------------------------------------------------

def test_a_test_already_failing_on_main_is_not_blamed_on_a_branch():
    """`main` is allowed to be red without that becoming every open branch's
    finding.  test_residual_model_policy.py has been red on main since
    2026-07-05 for reasons unrelated to any open pull request."""
    baseline = 'FAILED tests/test_residual_model_policy.py::test_residual\n'
    merged = 'FAILED tests/test_residual_model_policy.py::test_residual\n'
    assert mtc.introduced_failures(baseline, merged) == []


def test_a_test_the_merge_breaks_is_reported():
    baseline = 'FAILED tests/test_residual_model_policy.py::test_residual\n'
    merged = ('FAILED tests/test_residual_model_policy.py::test_residual\n'
              'FAILED jwst_gc_pipeline/photometry/tests/test_x.py::test_pool\n')
    assert mtc.introduced_failures(baseline, merged) == [
        'jwst_gc_pipeline/photometry/tests/test_x.py::test_pool']


def test_a_test_that_main_breaks_and_the_branch_FIXES_is_not_a_failure():
    """The subtraction is one-directional on purpose: this check reports what a
    merge breaks, not what it repairs."""
    baseline = 'FAILED a.py::x\nFAILED b.py::y\n'
    merged = 'FAILED a.py::x\n'
    assert mtc.introduced_failures(baseline, merged) == []


# ---------------------------------------------------------------------------
# Choosing which pull requests to test
# ---------------------------------------------------------------------------

def _pr(number, title='a change', draft=False, mergeable='MERGEABLE'):
    return dict(number=number, title=title, isDraft=draft, mergeable=mergeable)


def test_a_draft_is_not_tested():
    """Drafts are expected to be red; reporting them every time main moves is
    how a check gets muted."""
    assert not mtc.is_ready_for_merge_test(_pr(1, draft=True))


@pytest.mark.parametrize('title', [
    'WIP: multi-epoch proper-motion catalogs',
    '[WIP] something',
    'Draft: rework the merge',
    'do not merge until the survey lands',
])
def test_a_title_marked_not_ready_is_not_tested(title):
    """This repository has usually marked work-in-progress in the title rather
    than with GitHub's draft flag."""
    assert not mtc.is_ready_for_merge_test(_pr(1, title=title))


def test_a_conflicting_pull_request_is_not_tested():
    """A git conflict is already visible on the pull request itself, so
    re-reporting it here would be noise, not a finding."""
    assert not mtc.is_ready_for_merge_test(_pr(1, mergeable='CONFLICTING'))


def test_mergeability_not_yet_computed_is_still_tested():
    """GitHub returns UNKNOWN while it works out mergeability.  Skipping on
    UNKNOWN would silently drop whichever pull requests were asked about too
    soon -- the merge attempt itself is the reliable answer."""
    assert mtc.is_ready_for_merge_test(_pr(1, mergeable='UNKNOWN'))


def test_the_oldest_pull_requests_are_tested_first():
    """A long-open branch is the one most likely to be stale against main,
    which is the failure this check exists for."""
    selected = mtc.select_pull_requests([_pr(370), _pr(132), _pr(249)])
    assert [pr['number'] for pr in selected] == [132, 249, 370]


def test_a_cap_keeps_the_oldest():
    selected = mtc.select_pull_requests([_pr(370), _pr(132), _pr(249)], limit=2)
    assert [pr['number'] for pr in selected] == [132, 249]


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def test_the_report_says_which_tests_a_merge_introduced():
    report = mtc.render_report(
        [dict(number=243, title='split the writer', introduced=['a.py::test_pool'])],
        baseline_failures=set())
    assert '#243' in report
    assert 'a.py::test_pool' in report
    assert '1 new failure' in report


def test_the_report_states_the_baseline_rather_than_hiding_it():
    """A reader must be able to tell "main is red" from "this branch is"."""
    report = mtc.render_report(
        [dict(number=1, title='x', introduced=[])],
        baseline_failures={'tests/test_residual_model_policy.py::test_residual'})
    assert 'already fails 1 test' in report
    assert 'test_residual_model_policy' in report


def test_a_dropped_pull_request_is_named_not_silently_omitted():
    """A silent truncation reads as "everything was covered"."""
    report = mtc.render_report([], baseline_failures=set(), cap_dropped=3)
    assert '3 further open pull request(s) were not tested' in report


def test_a_skipped_draft_is_listed_so_the_report_is_not_read_as_full_coverage():
    report = mtc.render_report(
        [], baseline_failures=set(),
        skipped=[dict(number=140, title='WIP: proper motions')])
    assert '#140' in report
    assert 'Not tested' in report


def test_the_combined_report_names_every_branch_in_the_tree():
    """Without the list, a reader of a combined failure cannot tell which
    branches were in it -- and no single branch is at fault by itself."""
    report = mtc.render_all_together_report(
        merged_numbers=[235, 243], refused=[140],
        introduced=['a.py::test_pool'], baseline_failures=set())
    assert '#235' in report and '#243' in report
    assert '#140' in report          # said to be left out, not silently dropped
    assert 'a.py::test_pool' in report


def test_the_combined_report_says_so_when_nothing_broke():
    report = mtc.render_all_together_report(
        merged_numbers=[1], refused=[], introduced=[], baseline_failures=set())
    assert 'No test fails on the combined tree' in report


# ---------------------------------------------------------------------------
# Not destroying the checkout it runs in
# ---------------------------------------------------------------------------

def test_a_checkout_with_uncommitted_work_is_refused():
    """This script moves HEAD and merges branches into the tree it runs in.
    The repository's own convention is that the primary working tree IS the
    live reduction environment, so the default has to be refusal."""
    with pytest.raises(mtc.DirtyCheckoutError, match='2 uncommitted file'):
        mtc.assert_safe_to_check_out_other_commits(
            ' M jwst_gc_pipeline/photometry/cataloging.py\n?? scratch.py\n')


def test_a_clean_checkout_is_allowed():
    mtc.assert_safe_to_check_out_other_commits('')
    mtc.assert_safe_to_check_out_other_commits('   \n')


def test_the_refusal_is_overridable_for_a_scratch_clone():
    mtc.assert_safe_to_check_out_other_commits(' M a.py\n', allow_dirty=True)


# ---------------------------------------------------------------------------
# End to end: two branches that each pass, merge cleanly, and break
# ---------------------------------------------------------------------------

import subprocess
import textwrap


def _git(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args],
                          capture_output=True, text=True, check=True)


def _commit(repo, path, body, message):
    (repo / path).parent.mkdir(parents=True, exist_ok=True)
    (repo / path).write_text(textwrap.dedent(body))
    _git(repo, 'add', '-A')
    _git(repo, '-c', 'user.name=t', '-c', 'user.email=t@t', 'commit', '-m', message)


def _reproduce_the_merge_only_defect(repo):
    """The same class as the `pool` failure, in miniature.

    `main` defines a helper near the top of a file and a writer near the
    bottom, far enough apart that git diffs them as separate hunks.

      * Branch A makes the writer CALL the helper (an edit at the bottom).
      * Branch B RENAMES the helper and updates its only existing caller (edits
        at the top).

    Neither branch is wrong and both pass their own tests.  git merges them
    with no conflict because they touch different lines.  The merged tree calls
    a name that no longer exists -- which is what happened for real when one
    branch added a `pool` parameter to the offsets-table writer and another
    split that writer into a wrapper plus a private body that never received
    it.
    """
    # Indented to match the surrounding block so textwrap.dedent keeps it,
    # and long enough that git diffs the top and bottom as separate hunks.
    filler = '\n'.join(f'        # padding line {i}' for i in range(1, 41))

    _git(repo, 'init', '-q', '-b', 'main')
    _commit(repo, 'pkg/__init__.py', '', 'package marker')
    _commit(repo, 'pkg/writer.py', f"""
        def helper(rows):
            return len(rows)


        def count(rows):
            return helper(rows)


        {filler}


        def write(path, rows):
            return f'{{path}}:{{len(rows)}}'
        """, 'main')
    _commit(repo, 'pkg/test_writer.py', """
        from pkg.writer import count, write

        def test_count():
            assert count([1, 2]) == 2

        def test_write():
            assert write('t', [1, 2]) == 't:2'
        """, 'a test')

    # Branch A: the writer starts using the helper.  Bottom of the file.
    _git(repo, 'checkout', '-q', '-b', 'uses-the-helper')
    _commit(repo, 'pkg/writer.py', f"""
        def helper(rows):
            return len(rows)


        def count(rows):
            return helper(rows)


        {filler}


        def write(path, rows):
            return f'{{path}}:{{helper(rows)}}'
        """, 'write() now goes through helper()')

    # Branch B: the helper is renamed, with its only caller updated.  Top of
    # the file.  Nothing here knows write() exists.
    _git(repo, 'checkout', '-q', 'main')
    _git(repo, 'checkout', '-q', '-b', 'renames-the-helper')
    _commit(repo, 'pkg/writer.py', f"""
        def _helper(rows):
            return len(rows)


        def count(rows):
            return _helper(rows)


        {filler}


        def write(path, rows):
            return f'{{path}}:{{len(rows)}}'
        """, 'helper() is private')
    _git(repo, 'checkout', '-q', 'main')


def test_two_branches_that_each_pass_are_caught_when_merged_together(tmp_path, monkeypatch):
    repo = tmp_path / 'repo'
    repo.mkdir()
    _reproduce_the_merge_only_defect(repo)

    # each branch alone is green -- which is why per-branch CI cannot see this
    for branch in ('uses-the-helper', 'renames-the-helper'):
        _git(repo, 'checkout', '-q', branch)
        assert subprocess.run(['python', '-m', 'pytest', 'pkg', '-q'],
                              cwd=repo, capture_output=True).returncode == 0, branch
    _git(repo, 'checkout', '-q', 'main')

    # git merges them with no conflict at all
    _git(repo, 'checkout', '-q', '-b', 'both')
    _git(repo, '-c', 'user.name=t', '-c', 'user.email=t@t',
         'merge', '--no-edit', 'uses-the-helper')
    _git(repo, '-c', 'user.name=t', '-c', 'user.email=t@t',
         'merge', '--no-edit', 'renames-the-helper')
    merged = subprocess.run(['python', '-m', 'pytest', 'pkg', '-q', '-rf'],
                            cwd=repo, capture_output=True, text=True)
    assert merged.returncode != 0, 'the reproduction did not actually break'
    _git(repo, 'checkout', '-q', 'main')
    _git(repo, 'branch', '-D', 'both')

    # ...and the check reports exactly that, naming both branches
    _git(repo, 'remote', 'add', 'origin', str(repo))
    monkeypatch.chdir(repo)
    monkeypatch.setattr(mtc, 'open_pull_requests', lambda repo=None: [
        dict(number=1, title='use the helper', headRefName='uses-the-helper',
             isDraft=False, mergeable='MERGEABLE'),
        dict(number=2, title='rename the helper', headRefName='renames-the-helper',
             isDraft=False, mergeable='MERGEABLE'),
    ])
    out = tmp_path / 'report.md'
    code = mtc.main(['--all-together', '--test-path', 'pkg',
                     '--ref-pattern', 'refs/heads/{branch}',
                     '--output', str(out)])
    report = out.read_text()
    assert code == 1, report
    assert 'pkg/test_writer.py::test_write' in report
    assert '#1' in report and '#2' in report


def test_the_same_two_branches_are_green_one_at_a_time(tmp_path, monkeypatch):
    """The other half of the point: testing each branch against `main` on its
    own reports nothing, because neither is broken until the other lands."""
    repo = tmp_path / 'repo'
    repo.mkdir()
    _reproduce_the_merge_only_defect(repo)
    _git(repo, 'remote', 'add', 'origin', str(repo))
    monkeypatch.chdir(repo)
    monkeypatch.setattr(mtc, 'open_pull_requests', lambda repo=None: [
        dict(number=1, title='use the helper', headRefName='uses-the-helper',
             isDraft=False, mergeable='MERGEABLE'),
        dict(number=2, title='rename the helper', headRefName='renames-the-helper',
             isDraft=False, mergeable='MERGEABLE'),
    ])
    out = tmp_path / 'report.md'
    code = mtc.main(['--test-path', 'pkg', '--ref-pattern', 'refs/heads/{branch}',
                     '--output', str(out)])
    assert code == 0
    assert 'no new failures' in out.read_text()
