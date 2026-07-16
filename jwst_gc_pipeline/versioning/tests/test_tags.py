"""Tests for pipeline-tag parsing and the production run guard."""
import os
import subprocess

import pytest

from jwst_gc_pipeline.versioning import tags


def test_release_tag_roundtrip():
    t = tags.format_release_tag('2026-07-16', 104)
    assert t == '2026-07-16_PR104'
    assert tags.is_release_tag(t)
    assert not tags.is_dev_tag(t)
    p = tags.parse_tag(t)
    assert p == {'date': '2026-07-16', 'pr': 104, 'commit': None,
                 'dirty': False, 'dev': False, 'tag': t}


def test_dev_tag_roundtrip():
    t = tags.format_dev_tag('2026-07-16_PR104', 'a1b2c3d', dirty=True)
    assert t == '2026-07-16_PR104_a1b2c3d-dirty'
    assert tags.is_dev_tag(t)
    assert not tags.is_release_tag(t)
    p = tags.parse_tag(t)
    assert p['dev'] and p['dirty'] and p['pr'] == 104 and p['commit'] == 'a1b2c3d'


@pytest.mark.parametrize('bad', ['', 'v1.0-2026.06', '2026-7-16_PR1',
                                 '2026-07-16', '2026-07-16_PR', 'PR104'])
def test_rejects_malformed(bad):
    assert tags.parse_tag(bad) is None
    assert not tags.is_release_tag(bad)


def _git(repo, *args):
    subprocess.check_call(['git', '-C', repo, *args],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _init_repo(tmp_path):
    repo = str(tmp_path)
    _git(repo, 'init', '-q')
    _git(repo, 'config', 'user.email', 't@t')
    _git(repo, 'config', 'user.name', 't')
    (tmp_path / 'f.txt').write_text('one\n')
    _git(repo, 'add', 'f.txt')
    _git(repo, 'commit', '-q', '-m', 'c1')
    return repo


def test_exact_release_tag_is_production(tmp_path):
    tags.get_pipeline_tag.cache_clear()
    repo = _init_repo(tmp_path)
    _git(repo, 'tag', '2026-07-16_PR104')
    assert tags.get_pipeline_tag(repo) == '2026-07-16_PR104'
    assert tags.assert_runnable_version('imaging', repo_dir=repo) == '2026-07-16_PR104'


def test_untagged_clean_is_dev_and_blocks(tmp_path):
    tags.get_pipeline_tag.cache_clear()
    repo = _init_repo(tmp_path)  # committed, but no tag on HEAD
    tag = tags.get_pipeline_tag(repo)
    assert tags.is_dev_tag(tag)
    with pytest.raises(tags.UntaggedPipelineError):
        tags.assert_runnable_version('m12', allow_dev=False, repo_dir=repo)
    # dev opt-in is permitted and returns the dev tag (with a warning)
    with pytest.warns(UserWarning):
        got = tags.assert_runnable_version('m12', allow_dev=True, repo_dir=repo)
    assert got == tag


def test_dirty_tree_is_never_production(tmp_path):
    tags.get_pipeline_tag.cache_clear()
    repo = _init_repo(tmp_path)
    _git(repo, 'tag', '2026-07-16_PR104')
    (tmp_path / 'f.txt').write_text('one\ntwo\n')  # dirty
    tags.get_pipeline_tag.cache_clear()
    tag = tags.get_pipeline_tag(repo)
    assert tag.endswith('-dirty') and tags.is_dev_tag(tag)
    with pytest.raises(tags.UntaggedPipelineError):
        tags.assert_runnable_version('imaging', allow_dev=False, repo_dir=repo)


def test_allow_dev_env(tmp_path, monkeypatch):
    tags.get_pipeline_tag.cache_clear()
    repo = _init_repo(tmp_path)
    monkeypatch.setenv('GC_ALLOW_DEV', '1')
    with pytest.warns(UserWarning):
        tag = tags.assert_runnable_version('m3', repo_dir=repo)
    assert tags.is_dev_tag(tag)
