"""Pipeline tag resolution and the production-run guard.

Tag scheme
----------
* **Release tag** (created by the ``tag-on-merge`` GitHub Action on every merged
  PR): ``YYYY-MM-DD_PR<number>``, e.g. ``2026-07-16_PR104``.  The date is the
  merge date; ``<number>`` is the PR number.  Annotated tag on the merge commit.
* **Dev tag** (any run whose ``HEAD`` is not exactly a release tag, or whose tree
  is dirty): ``<release>_<shortcommit>[-dirty]``, e.g.
  ``2026-07-16_PR104_a1b2c3d`` or ``2026-07-16_PR104_a1b2c3d-dirty``.  The
  ``<release>`` prefix is the nearest reachable release tag, so a dev product is
  always traceable to the release lineage it descends from, and its own commit.

Enforcement
-----------
``assert_runnable_version(stage)`` is the run guard every *production* stage
entry must call.  On the live reduction tree it HARD-BLOCKS an untagged or dirty
run (raising :class:`UntaggedPipelineError`) unless the caller opts into a dev
run via ``allow_dev=True`` or ``GC_ALLOW_DEV=1``.  A dev run is permitted but
STAMPS the dev tag, so a dev product can never masquerade as a release product.
"""
import functools
import os
import re
import subprocess
import warnings

# YYYY-MM-DD_PR<n>
_RELEASE_RE = re.compile(r'^(?P<date>\d{4}-\d{2}-\d{2})_PR(?P<pr>\d+)$')
# YYYY-MM-DD_PR<n>_<hexcommit>[-dirty]
_DEV_RE = re.compile(
    r'^(?P<date>\d{4}-\d{2}-\d{2})_PR(?P<pr>\d+)_(?P<commit>[0-9a-f]{7,40})'
    r'(?P<dirty>-dirty)?$')

_ALLOW_DEV_ENV = 'GC_ALLOW_DEV'


class UntaggedPipelineError(RuntimeError):
    """Raised when a production stage is run on an untagged or dirty tree.

    Set ``GC_ALLOW_DEV=1`` (or pass ``allow_dev=True``) for a development run,
    or tag ``HEAD`` with a ``YYYY-MM-DD_PR<n>`` release tag.
    """


def is_release_tag(text):
    """True iff ``text`` is a well-formed release tag ``YYYY-MM-DD_PR<n>``."""
    return bool(_RELEASE_RE.match(text or ''))


def is_dev_tag(text):
    """True iff ``text`` is a well-formed dev tag ``..._<commit>[-dirty]``."""
    return bool(_DEV_RE.match(text or ''))


def parse_tag(text):
    """Parse a release or dev tag into a dict.

    Returns ``{'date','pr','commit'|None,'dirty':bool,'dev':bool,'tag':text}``
    or ``None`` if ``text`` matches neither pattern.
    """
    m = _RELEASE_RE.match(text or '')
    if m:
        return {'date': m['date'], 'pr': int(m['pr']), 'commit': None,
                'dirty': False, 'dev': False, 'tag': text}
    m = _DEV_RE.match(text or '')
    if m:
        return {'date': m['date'], 'pr': int(m['pr']), 'commit': m['commit'],
                'dirty': bool(m['dirty']), 'dev': True, 'tag': text}
    return None


def format_release_tag(date, pr):
    """``(date='2026-07-16', pr=104) -> '2026-07-16_PR104'``."""
    return f'{date}_PR{pr}'


def format_dev_tag(release, commit, dirty=False):
    """``('2026-07-16_PR104', 'a1b2c3d', dirty) -> '..._a1b2c3d[-dirty]'``."""
    return f'{release}_{commit}' + ('-dirty' if dirty else '')


def _repo_dir():
    """Directory of the repo containing this package (worktree-aware)."""
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.dirname(pkg_dir)


def _git(args, repo_dir):
    """Run ``git -C repo_dir <args>`` and return stripped stdout, or None."""
    try:
        return subprocess.check_output(
            ['git', '-C', repo_dir, *args],
            stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.SubprocessError, OSError):
        return None


def _is_dirty(repo_dir):
    """True iff the tree differs from HEAD in ANY way that could change output.

    ``git status --porcelain`` catches all three cases a ``git diff --quiet``
    misses: STAGED edits (index != HEAD), unstaged edits, AND untracked files.
    This matters because the run guard's whole promise is that a release tag
    implies unmodified code -- a staged edit to a stage source file on a tagged
    HEAD must NOT resolve to the release tag. (The fingerprint layer already
    hashes working-tree content via ``git hash-object``; this closes the same
    gap in the guard itself.)
    """
    out = _git(['status', '--porcelain'], repo_dir)
    if out is None:
        # git could not be run -> cannot prove the tree is clean; treat as dirty
        # (fail-closed for the run guard).
        return True
    return out.strip() != ''


def _exact_release_tag(repo_dir):
    """Return the release tag pointing exactly at HEAD, or None.

    ``git tag --points-at HEAD`` lists ALL tags on HEAD; we pick the one that
    matches the release pattern (there should be at most one per merged PR).
    """
    out = _git(['tag', '--points-at', 'HEAD'], repo_dir)
    if not out:
        return None
    for line in out.splitlines():
        if is_release_tag(line.strip()):
            return line.strip()
    return None


def _nearest_release(repo_dir):
    """Nearest reachable release tag (the dev-tag ``<release>`` prefix), or None.

    ``git describe`` only considers tags matching the release glob, so a stray
    non-release tag never becomes the prefix.
    """
    return _git(['describe', '--tags', '--abbrev=0',
                 '--match', '[0-9]*-[0-9]*-[0-9]*_PR[0-9]*'], repo_dir)


def _short_commit(repo_dir):
    return _git(['rev-parse', '--short=7', 'HEAD'], repo_dir)


@functools.lru_cache(maxsize=8)
def get_pipeline_tag(repo_dir=None):
    """Resolve the pipeline tag for the current checkout (cached per repo_dir).

    * clean tree whose HEAD is exactly a release tag -> that release tag.
    * otherwise -> the dev tag ``<nearest-release>_<shortcommit>[-dirty]``.
      If no release tag is reachable, ``<nearest-release>`` becomes
      ``0000-00-00_PR0`` (an explicit "pre-tagging" sentinel) so the string is
      still parseable and clearly non-release.
    """
    repo_dir = repo_dir or _repo_dir()
    dirty = _is_dirty(repo_dir)
    if not dirty:
        exact = _exact_release_tag(repo_dir)
        if exact:
            return exact
    nearest = _nearest_release(repo_dir) or '0000-00-00_PR0'
    commit = _short_commit(repo_dir) or 'unknown'
    return format_dev_tag(nearest, commit, dirty=dirty)


def is_production_tag(tag):
    """A tag is production-runnable iff it is an exact (non-dev) release tag."""
    return is_release_tag(tag)


def assert_runnable_version(stage, allow_dev=None, repo_dir=None):
    """Guard a production stage entry; return the resolved pipeline tag.

    Parameters
    ----------
    stage : str
        Stage name, for the error/warning message ('imaging', 'm12', ...).
    allow_dev : bool or None
        If None (default), read ``GC_ALLOW_DEV`` from the environment (``'1'``,
        ``'true'``, ``'yes'`` -> True).  A truthy value permits a dev run.
    repo_dir : str or None
        Repo to inspect; defaults to the repo containing this package.

    Returns
    -------
    str
        The resolved tag (a release tag for a production run, or the dev tag for
        a permitted dev run).

    Raises
    ------
    UntaggedPipelineError
        If the tree is untagged/dirty and dev runs are not permitted.
    """
    repo_dir = repo_dir or _repo_dir()
    if allow_dev is None:
        allow_dev = os.environ.get(_ALLOW_DEV_ENV, '').strip().lower() in (
            '1', 'true', 'yes', 'on')
    tag = get_pipeline_tag(repo_dir)
    if is_production_tag(tag):
        return tag
    # Non-release (dev/dirty) tree.
    if allow_dev:
        warnings.warn(
            f"[{stage}] DEV pipeline run on untagged/dirty tree; stamping dev "
            f"tag {tag!r}. This product is NOT a release product.",
            stacklevel=2)
        return tag
    raise UntaggedPipelineError(
        f"[{stage}] refusing to run a PRODUCTION stage on an untagged or dirty "
        f"tree (resolved tag {tag!r}). Either tag HEAD with a release tag "
        f"'YYYY-MM-DD_PR<n>' (normally done automatically by the tag-on-merge "
        f"GitHub Action), or opt into a development run with {_ALLOW_DEV_ENV}=1 "
        f"(or allow_dev=True).")
