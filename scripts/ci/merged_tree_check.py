#!/usr/bin/env python
"""Test each open pull request against the CURRENT main, not the one it was opened on.

## What this is for

Two pull requests can each pass their own tests, merge without a git conflict,
and produce a ``main`` that raises on the normal path. It happened here: one
branch added a ``pool`` parameter to ``update_offsets_table`` (the function that
writes measured astrometric corrections into a field's offsets table), another
split that same function into a locking wrapper plus a private body. Different
lines, so git merged them silently, and the parameter stopped being forwarded to
the body that used it — 29 test failures on a tree neither branch produced.

GitHub's per-pull-request run already tests a MERGED tree: ``actions/checkout``
resolves ``refs/pull/<n>/merge``, which is the branch merged into ``main``. So
the merged tree is not the gap. **Staleness is.** That run happens when the
branch is pushed and does not repeat when ``main`` moves, so a green check can
describe a merge with a ``main`` that no longer exists. Measured 2026-08-10, the
two oldest open pull requests here carried green checks computed against a
``main`` **659 and 665 commits** behind, and the repository has no branch
protection requiring a branch to be current before it merges.

This script closes that by re-running the suite on each open pull request merged
into today's ``main``, and — separately — on every mergeable pull request merged
together, which is the only arrangement that sees an A-plus-B interaction before
either one lands.

## Reading the result

A pull request that is red on its own is its author's problem and not this
check's business. So failures are reported RELATIVE to a baseline run of
``main`` alone: the reported set is what the merge introduced, and an empty set
is a pass even when both trees have failures in common. `main` itself is
allowed to be red without that being blamed on every open branch.

This module holds the parts worth testing on their own — selecting the pull
requests, parsing pytest's output into test identifiers, and the subtraction —
so that the workflow calling it can stay a thin shell.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

#: A pytest short-summary line: ``FAILED path::test - message`` or ``ERROR path``.
#: Anchored at the line start because the message can itself contain the word.
_SUMMARY_LINE = re.compile(r'^(?:FAILED|ERROR)\s+(\S+)')

#: Title markers for a pull request its author has said is not ready. Draft
#: status is read from the API separately; this catches the ones marked only in
#: the title, which is how this repository has usually done it.
_NOT_READY = ('wip:', '[wip]', 'draft:', 'do not merge', "don't merge")


def failing_test_ids(pytest_output):
    """The set of test identifiers pytest reported as FAILED or ERROR.

    Reads the short summary rather than counting, so two runs can be compared
    by WHICH tests failed instead of how many. A test that fails for a
    different reason in the two runs is still the same identifier and so is not
    reported as introduced -- deliberately: this check is about tests that a
    merge breaks, and re-litigating why an already-failing test fails belongs
    to whoever owns that test.
    """
    ids = set()
    for line in pytest_output.splitlines():
        match = _SUMMARY_LINE.match(line.strip())
        if match:
            ids.add(match.group(1))
    return ids


def introduced_failures(baseline_output, merged_output):
    """Tests failing on the merged tree that were not already failing on main.

    Returns a sorted list so the report is stable between runs.
    """
    return sorted(failing_test_ids(merged_output)
                  - failing_test_ids(baseline_output))


def is_ready_for_merge_test(pull_request):
    """Whether an open pull request should be tested against current main.

    Skips drafts and titles marked work-in-progress: they are expected to be
    red, and reporting them every time main moves is how a check gets muted.
    Also skips one git cannot merge cleanly -- a conflict is already visible on
    the pull request itself and is not this check's finding.
    """
    if pull_request.get('isDraft'):
        return False
    title = str(pull_request.get('title', '')).lower()
    if any(marker in title for marker in _NOT_READY):
        return False
    # 'UNKNOWN' means GitHub has not finished computing mergeability; treat it
    # as testable rather than silently skipping, and let the merge itself say.
    return str(pull_request.get('mergeable', 'UNKNOWN')).upper() != 'CONFLICTING'


def select_pull_requests(pull_requests, limit=None):
    """The testable pull requests, oldest first, optionally capped.

    Oldest first because a long-open branch is the one most likely to be stale
    against main, which is the failure this check exists for. When a cap drops
    any, the caller is expected to say so in the report -- a silent truncation
    reads as "everything was covered".
    """
    ready = [pr for pr in pull_requests if is_ready_for_merge_test(pr)]
    ready.sort(key=lambda pr: pr.get('number', 0))
    return ready if limit is None else ready[:limit]


#: How to fetch a pull request's head.  ``{n}`` is the number and ``{branch}``
#: its branch name.  GitHub serves every pull request at ``pull/<n>/head``; a
#: local mirror or a test fixture has only ordinary branches, which is what the
#: override is for.
DEFAULT_REF_PATTERN = 'pull/{n}/head'


def fetch_ref(pull_request, ref_pattern=DEFAULT_REF_PATTERN):
    """The refspec source for one pull request's head."""
    return ref_pattern.format(n=pull_request['number'],
                              branch=pull_request.get('headRefName', ''))


def open_pull_requests(repo=None):
    """Open pull requests, from the GitHub CLI."""
    cmd = ['gh', 'pr', 'list', '--state', 'open', '--limit', '100',
           '--json', 'number,title,headRefName,headRefOid,isDraft,mergeable']
    if repo:
        cmd += ['--repo', repo]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def render_report(results, baseline_failures, skipped=(), cap_dropped=0):
    """A markdown report of what each merge introduced.

    ``results`` is a list of dicts with ``number``, ``title`` and either
    ``introduced`` (a list of test ids) or ``error`` (a string).
    """
    lines = ['# Open pull requests against current `main`', '']
    if baseline_failures:
        lines += [f'`main` alone already fails {len(baseline_failures)} test(s); '
                  'those are subtracted from every row below.', '',
                  '<details><summary>baseline failures on `main`</summary>', '']
        lines += [f'- `{t}`' for t in sorted(baseline_failures)]
        lines += ['', '</details>', '']
    else:
        lines += ['`main` alone is green, so anything below is introduced by the '
                  'merge.', '']

    lines += ['| PR | title | merged with today\'s `main` |', '|---|---|---|']
    for result in results:
        if result.get('error'):
            verdict = f"could not test: {result['error']}"
        elif result['introduced']:
            verdict = f"**{len(result['introduced'])} new failure(s)**"
        else:
            verdict = 'no new failures'
        lines.append(f"| #{result['number']} | {result['title']} | {verdict} |")

    for result in results:
        if result.get('introduced'):
            lines += ['', f"## #{result['number']} introduces:", '']
            lines += [f'- `{t}`' for t in result['introduced']]

    if skipped:
        lines += ['', '## Not tested', '']
        lines += [f"- #{pr['number']} {pr['title']} (draft or marked "
                  'work-in-progress)' for pr in skipped]
    if cap_dropped:
        lines += ['', f'**{cap_dropped} further open pull request(s) were not '
                  'tested because of the per-run cap.** They are not covered by '
                  'this report.']
    return '\n'.join(lines) + '\n'


class DirtyCheckoutError(RuntimeError):
    """The checkout has uncommitted work, which this script would destroy."""


def assert_safe_to_check_out_other_commits(status_porcelain, allow_dirty=False):
    """Refuse to run in a checkout holding uncommitted work.

    This script moves HEAD around and merges other branches into the checkout it
    is run from, which is harmless on a throwaway CI runner and destroys an
    afternoon of work in a live one.  The repository's own convention is that
    the primary working tree IS the live reduction environment, so the default
    has to be refusal rather than a warning.
    """
    if allow_dirty or not status_porcelain.strip():
        return
    n_files = len(status_porcelain.strip().splitlines())
    raise DirtyCheckoutError(
        f'refusing to run: this checkout has {n_files} uncommitted file(s), and '
        f'this script checks out other commits and merges branches into the '
        f'tree it runs in -- it would destroy them.  Run it in a scratch clone '
        f'(git clone . /tmp/somewhere), or pass --allow-dirty if you are sure '
        f'nothing here matters.')


def _run(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def _pytest(paths):
    return _run([sys.executable, '-m', 'pytest', *paths, '-q', '-rf',
                 '--color=no'])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', default=None, help='owner/name; default is the checkout\'s')
    parser.add_argument('--limit', type=int, default=None,
                        help='cap the number of pull requests tested; the report '
                             'says how many were dropped')
    parser.add_argument('--test-path', action='append', default=None,
                        help='pytest target (repeatable); default matches CI')
    parser.add_argument('--output', default=None, help='write the markdown report here')
    parser.add_argument('--list-only', action='store_true',
                        help='print the pull requests that would be tested, and stop')
    parser.add_argument('--ref-pattern', default=DEFAULT_REF_PATTERN,
                        help='refspec source for a pull request head; '
                             f'default {DEFAULT_REF_PATTERN!r}.  Accepts {{n}} '
                             'and {branch}, for a local mirror that has only '
                             'ordinary branches')
    parser.add_argument('--allow-dirty', action='store_true',
                        help='run even though the checkout has uncommitted work, '
                             'which this script will destroy')
    parser.add_argument('--all-together', action='store_true',
                        help='merge every testable pull request onto main at once '
                             'and run the suite on that single tree, instead of '
                             'testing them one at a time')
    args = parser.parse_args(argv)

    paths = args.test_path or ['jwst_gc_pipeline', 'tests']
    ref_pattern = args.ref_pattern
    if not args.list_only:
        try:
            assert_safe_to_check_out_other_commits(
                _run(['git', 'status', '--porcelain']).stdout,
                allow_dirty=args.allow_dirty)
        except DirtyCheckoutError as ex:
            print(str(ex), file=sys.stderr)
            return 2
    pulls = open_pull_requests(args.repo)
    selected = select_pull_requests(pulls)
    skipped = [pr for pr in pulls if not is_ready_for_merge_test(pr)]
    dropped = 0
    if args.limit is not None and len(selected) > args.limit:
        dropped = len(selected) - args.limit
        selected = selected[:args.limit]

    if args.list_only:
        for pr in selected:
            print(f"#{pr['number']}\t{pr['headRefName']}\t{pr['title']}")
        if dropped:
            print(f'({dropped} dropped by --limit)', file=sys.stderr)
        return 0

    main_sha = _run(['git', 'rev-parse', 'HEAD']).stdout.strip()
    # The branch name if there is one, so the checkout is left as it was found
    # rather than detached at a commit that happens to be the same.
    original_ref = _run(['git', 'symbolic-ref', '--quiet', '--short',
                         'HEAD']).stdout.strip() or main_sha
    print(f'baseline: main at {main_sha[:9]}', flush=True)
    baseline = failing_test_ids(_pytest(paths).stdout)

    if args.all_together:
        report, failed = _all_together(selected, main_sha, baseline, paths,
                                       skipped=skipped, cap_dropped=dropped,
                                       ref_pattern=args.ref_pattern)
        _restore(original_ref, [pr['number'] for pr in selected])
        print(report)
        if args.output:
            with open(args.output, 'w') as handle:
                handle.write(report)
        return 1 if failed else 0

    results = []
    for pr in selected:
        number = pr['number']
        print(f"--- #{number} {pr['title']}", flush=True)
        _run(['git', 'checkout', '--force', main_sha])
        fetch = _run(['git', 'fetch', 'origin',
                      f'{fetch_ref(pr, ref_pattern)}:pr-{number}'])
        if fetch.returncode != 0:
            results.append(dict(number=number, title=pr['title'],
                                introduced=[], error='could not fetch the branch'))
            continue
        merge = _run(['git', '-c', 'user.name=ci', '-c', 'user.email=ci@local',
                      'merge', '--no-edit', f'pr-{number}'])
        if merge.returncode != 0:
            _run(['git', 'merge', '--abort'])
            results.append(dict(number=number, title=pr['title'], introduced=[],
                                error='conflicts with current `main`'))
            continue
        merged_out = _pytest(paths).stdout
        results.append(dict(number=number, title=pr['title'],
                            introduced=introduced_failures_from(baseline, merged_out)))

    _restore(original_ref, [pr['number'] for pr in selected])
    report = render_report(results, baseline, skipped=skipped, cap_dropped=dropped)
    print(report)
    if args.output:
        with open(args.output, 'w') as handle:
            handle.write(report)
    return 1 if any(r['introduced'] or r.get('error') for r in results) else 0


def _restore(original_ref, pr_numbers):
    """Put the checkout back where it was found and drop the fetched branches.

    Without this the checkout is left detached on a merge commit that is on no
    branch, which reads as a broken repository to whoever looks next.
    """
    _run(['git', 'checkout', '--force', original_ref])
    for number in pr_numbers:
        _run(['git', 'branch', '-D', f'pr-{number}'])


def introduced_failures_from(baseline_ids, merged_output):
    """``introduced_failures`` when the baseline is already a set of ids."""
    return sorted(failing_test_ids(merged_output) - set(baseline_ids))


def render_all_together_report(merged_numbers, refused, introduced,
                               baseline_failures, skipped=(), cap_dropped=0):
    """Markdown for the one-tree-with-everything run.

    This is the arrangement that sees an A-plus-B interaction BEFORE either one
    lands, which is the case the per-pull-request runs cannot reach: neither
    branch is red until the other has already merged.
    """
    lines = ['# Every open pull request merged together', '',
             'This is the arrangement that sees an interaction between two '
             'branches before either one lands. Testing each branch against '
             '`main` on its own cannot: neither is red until the other has '
             'already merged.', '']
    if merged_numbers:
        lines += ['Merged onto `main`: '
                  + ', '.join(f'#{n}' for n in merged_numbers), '']
    else:
        lines += ['Nothing was merged, so there is no combined tree to report '
                  'on.', '']
    if refused:
        lines += ['Left out because they conflict with `main` or with an '
                  'earlier branch in this list (a conflict is already visible '
                  'on the pull request, and is not this check\'s finding): '
                  + ', '.join(f'#{n}' for n in refused), '']
    if baseline_failures:
        lines += [f'`main` alone already fails {len(baseline_failures)} '
                  'test(s); those are subtracted below.', '']
    if introduced:
        lines += [f'**{len(introduced)} test(s) fail on the combined tree that '
                  'do not fail on `main`:**', '']
        lines += [f'- `{t}`' for t in introduced]
        lines += ['', 'Each listed branch may well be green on its own. That is '
                  'the point: the defect exists only in the combination.']
    else:
        lines += ['No test fails on the combined tree that does not already '
                  'fail on `main`.']
    if skipped:
        lines += ['', '## Not included', '']
        lines += [f"- #{pr['number']} {pr['title']} (draft or marked "
                  'work-in-progress)' for pr in skipped]
    if cap_dropped:
        lines += ['', f'**{cap_dropped} further open pull request(s) were left '
                  'out because of the per-run cap.** They are not covered.']
    return '\n'.join(lines) + '\n'


def _all_together(selected, main_sha, baseline, paths, skipped=(), cap_dropped=0,
                  ref_pattern=DEFAULT_REF_PATTERN):
    """Merge every selected pull request onto main and run the suite once."""
    _run(['git', 'checkout', '--force', main_sha])
    merged, refused = [], []
    for pr in selected:
        number = pr['number']
        if _run(['git', 'fetch', 'origin',
                 f'{fetch_ref(pr, ref_pattern)}:pr-{number}']).returncode != 0:
            refused.append(number)
            continue
        merge = _run(['git', '-c', 'user.name=ci', '-c', 'user.email=ci@local',
                      'merge', '--no-edit', f'pr-{number}'])
        if merge.returncode != 0:
            _run(['git', 'merge', '--abort'])
            refused.append(number)
            continue
        merged.append(number)
        print(f'merged #{number}', flush=True)

    introduced = (introduced_failures_from(baseline, _pytest(paths).stdout)
                  if merged else [])
    _run(['git', 'checkout', '--force', main_sha])
    report = render_all_together_report(merged, refused, introduced, baseline,
                                        skipped=skipped, cap_dropped=cap_dropped)
    return report, bool(introduced)


if __name__ == '__main__':
    raise SystemExit(main())
