"""``GCPIPEV`` must read ``-dirty`` for staged edits and untracked files.

The stamp's whole promise is that a product header names the code that made
it.  ``git diff --quiet`` compares the worktree to the INDEX, so a ``git
add``-ed edit, or a brand-new untracked module dropped into the package,
resolved to a CLEAN commit id -- a catalog stamped with a commit that does not
describe its own code.  ``versioning.tags._is_dirty`` already uses ``git status
--porcelain`` for ``GCTAG``; these tests pin the same rule for ``GCPIPEV``.

Reverting ``_tree_is_dirty`` to ``git diff --quiet`` fails
``test_staged_edit_reads_dirty`` and ``test_untracked_file_reads_dirty``.
"""
import subprocess

import pytest

from jwst_gc_pipeline.provenance import _tree_is_dirty


def _git(repo, *args):
    subprocess.check_call(['git', '-C', str(repo), *args],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


@pytest.fixture
def repo(tmp_path):
    """A one-commit git repo with a tracked file."""
    _git(tmp_path, 'init', '-q')
    _git(tmp_path, 'config', 'user.email', 'test@example.com')
    _git(tmp_path, 'config', 'user.name', 'test')
    (tmp_path / 'tracked.py').write_text('x = 1\n')
    _git(tmp_path, 'add', 'tracked.py')
    _git(tmp_path, 'commit', '-qm', 'init')
    return tmp_path


def test_clean_tree_reads_clean(repo):
    assert _tree_is_dirty(repo) is False


def test_unstaged_edit_reads_dirty(repo):
    (repo / 'tracked.py').write_text('x = 2\n')
    assert _tree_is_dirty(repo) is True


def test_staged_edit_reads_dirty(repo):
    # The gap: `git diff --quiet` compares worktree to INDEX, so once the edit
    # is staged it reports NO difference and the stamp reads clean.
    (repo / 'tracked.py').write_text('x = 3\n')
    _git(repo, 'add', 'tracked.py')
    assert subprocess.call(['git', '-C', str(repo), 'diff', '--quiet'],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) == 0, (
        'precondition: a staged edit is invisible to `git diff --quiet`')
    assert _tree_is_dirty(repo) is True


def test_untracked_file_reads_dirty(repo):
    # A brand-new module added to the package changes what the code does and is
    # invisible to `git diff` in either direction.
    (repo / 'brandnew.py').write_text('def f():\n    return 1\n')
    assert subprocess.call(['git', '-C', str(repo), 'diff', '--quiet'],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL) == 0, (
        'precondition: an untracked file is invisible to `git diff --quiet`')
    assert _tree_is_dirty(repo) is True


def test_ungittable_path_fails_closed(tmp_path):
    # No repo here at all: cleanliness cannot be established, so the stamp must
    # say -dirty rather than claim a clean tree it never checked.
    assert _tree_is_dirty(tmp_path / 'not-a-repo') is True


def test_stamp_carries_the_dirty_suffix(repo, monkeypatch):
    """End to end: the value written into ``GCPIPEV`` gains ``-dirty``."""
    import jwst_gc_pipeline.provenance as prov
    pkg = repo / 'jwst_gc_pipeline'
    pkg.mkdir()
    monkeypatch.setattr(prov.os.path, 'abspath',
                        lambda p: str(pkg / 'provenance.py'))
    (repo / 'brandnew.py').write_text('x = 1\n')  # untracked
    prov.get_pipeline_commit.cache_clear()
    try:
        assert prov.get_pipeline_commit().endswith('-dirty')
    finally:
        prov.get_pipeline_commit.cache_clear()
