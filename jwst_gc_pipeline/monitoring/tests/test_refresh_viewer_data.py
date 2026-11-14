"""`refresh_viewer_data.sh` must rebuild the sky viewer's two derived files, and
must not publish a bad one.

The focal-plane and guide-star JSONs had no scheduler: both were last built by
hand on 2026-09-18 and the viewer kept serving them, with no sign of their age.
Both builders were also pinned to release v1.8 after v1.9 had been delivered, so
a scheduler running them unchanged would have rebuilt the same stale as-flown set.

These run the real script with STUB builders, so they test the wiring and the
publish checks rather than the builders.
"""
import importlib.util
import json
import os
import stat
import subprocess
import sys

import pytest

_MON = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'scripts', 'monitoring'))
_SH = os.path.join(_MON, 'refresh_viewer_data.sh')
_REPO = os.path.abspath(os.path.join(_MON, '..', '..'))


def _module(name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_MON, f'{name}.py'))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, _MON)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(_MON)
    return module


def _focal(n=3, flown=2):
    return json.dumps({'pointings': [{'id': f'o{i:03d}'} for i in range(n)],
                       'n_as_flown': flown})


def _guide(n=2, unflown=1):
    return json.dumps({'n': n, 'sources': [{'guide star': str(i)}
                                           for i in range(n)],
                       'unflown': ['o001'] * unflown})


def _stub(tmp_path, name, body):
    p = tmp_path / name
    p.write_text('#!/bin/bash\n' + body + '\n')
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def _builders(focal='echo \'%s\' > "$out"' % _focal(),
              guide='echo \'%s\' > "$out"' % _guide(),
              focal_rc=0, guide_rc=0):
    """A stub `python` that plays either builder, picked by the script name."""
    return f'''
out=
prev=
for a in "$@"; do
  if [ "$prev" = "--out" ]; then out="$a"; fi
  prev="$a"
done
case "$1" in
  *build_focal_plane.py) {focal}; exit {focal_rc} ;;
  *build_guidestars.py) {guide}; exit {guide_rc} ;;
esac
exit 99
'''


def _run(tmp_path, body, docroot=None, rsync_rc=0, **env):
    docroot = docroot or tmp_path / 'docroot'
    docroot.mkdir(exist_ok=True)
    rsync_log = tmp_path / 'rsync.log'
    e = dict(os.environ, REPO=_REPO, DOCROOT=str(docroot),
             PYTHON=_stub(tmp_path, 'fakepython', body),
             CHECK_PYTHON=sys.executable,
             RSYNC=_stub(tmp_path, 'fakersync',
                         f'echo "$@" >> {rsync_log}; exit {rsync_rc}'),
             WEB_HOST='webhost', WEB_DIR='/web/avm_images',
             ALLOW_STALE_CHECKOUT='1', DEPLOY='1')
    e.update(env)
    done = subprocess.run(['bash', _SH], env=e, capture_output=True, text=True,
                          timeout=120)
    return done, docroot, rsync_log


def _seed(docroot, focal=None, guide=None):
    (docroot / 'jwst_gc_focal_plane.json').write_text(focal or _focal())
    (docroot / 'jwst_gc_cat_guidestars.json').write_text(guide or _guide())


# --- the script ---------------------------------------------------------------

def test_both_builders_succeed_and_both_files_land(tmp_path):
    done, docroot, rsync_log = _run(tmp_path, _builders())
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads((docroot / 'jwst_gc_focal_plane.json').read_text())['n_as_flown'] == 2
    assert json.loads((docroot / 'jwst_gc_cat_guidestars.json').read_text())['n'] == 2
    assert '3 pointings (2 as-flown), 2 guide stars, 1 unflown' in done.stdout
    assert 'deploy=0' in done.stdout


def test_the_files_are_copied_to_the_web_host(tmp_path):
    done, docroot, rsync_log = _run(tmp_path, _builders())
    line = rsync_log.read_text()
    assert 'jwst_gc_focal_plane.json' in line
    assert 'jwst_gc_cat_guidestars.json' in line
    assert 'webhost:/web/avm_images/' in line


def test_no_staging_file_is_left_behind(tmp_path):
    done, docroot, _ = _run(tmp_path, _builders())
    assert [p for p in os.listdir(docroot) if p.startswith('.')] == []


def test_a_failed_focal_plane_builder_replaces_neither_file(tmp_path):
    """All or nothing: the guide-star build is not even attempted."""
    docroot = tmp_path / 'docroot'
    docroot.mkdir()
    _seed(docroot, focal=_focal(5, 5), guide=_guide(7))
    done, _, rsync_log = _run(
        tmp_path, _builders(focal='printf \'{"point\' > "$out"', focal_rc=3),
        docroot=docroot)
    assert done.returncode == 3
    assert json.loads((docroot / 'jwst_gc_focal_plane.json').read_text())['n_as_flown'] == 5
    assert json.loads((docroot / 'jwst_gc_cat_guidestars.json').read_text())['n'] == 7
    assert [p for p in os.listdir(docroot) if p.startswith('.')] == []
    assert not rsync_log.exists()


def test_a_failed_guide_builder_leaves_the_good_focal_plane_unpublished(tmp_path):
    docroot = tmp_path / 'docroot'
    docroot.mkdir()
    _seed(docroot, focal=_focal(3, 1))
    done, _, rsync_log = _run(tmp_path, _builders(guide_rc=2), docroot=docroot)
    assert done.returncode == 2
    # the new focal plane (n_as_flown 2) was built but must not have been published
    assert json.loads((docroot / 'jwst_gc_focal_plane.json').read_text())['n_as_flown'] == 1
    assert not rsync_log.exists()


@pytest.mark.parametrize('body', ["printf '' > \"$out\"",
                                  "printf '{\"n\": 4, \"sour' > \"$out\""])
def test_an_empty_or_truncated_guide_file_keeps_the_published_copy(tmp_path, body):
    """The builder exits 0 here: a killed write, or a MAST outage that returned
    nothing, must not replace a good file."""
    docroot = tmp_path / 'docroot'
    docroot.mkdir()
    _seed(docroot)
    done, _, rsync_log = _run(tmp_path, _builders(guide=body), docroot=docroot)
    assert done.returncode != 0
    assert json.loads((docroot / 'jwst_gc_cat_guidestars.json').read_text()) == json.loads(_guide())
    assert json.loads((docroot / 'jwst_gc_focal_plane.json').read_text()) == json.loads(_focal())
    assert not rsync_log.exists()


def test_fewer_pointings_than_the_published_file_is_refused(tmp_path):
    docroot = tmp_path / 'docroot'
    docroot.mkdir()
    _seed(docroot, focal=_focal(5, 2))
    done, _, _ = _run(tmp_path, _builders(), docroot=docroot)       # 3 pointings
    assert done.returncode != 0
    assert 'pointings would shrink from 5 to 3' in done.stderr
    assert json.loads((docroot / 'jwst_gc_focal_plane.json').read_text())['pointings'].__len__() == 5


def test_a_lost_as_flown_set_is_refused_although_the_pointing_count_is_equal(tmp_path):
    """The failure the issue describes: the plan is complete, the delivered
    frames were not read."""
    docroot = tmp_path / 'docroot'
    docroot.mkdir()
    _seed(docroot, focal=_focal(3, 3))
    done, _, _ = _run(tmp_path, _builders(), docroot=docroot)       # n_as_flown 2
    assert done.returncode != 0
    assert 'n_as_flown would shrink from 3 to 2' in done.stderr


def test_allow_shrink_publishes_the_smaller_file(tmp_path):
    docroot = tmp_path / 'docroot'
    docroot.mkdir()
    _seed(docroot, focal=_focal(5, 5), guide=_guide(7))
    done, _, _ = _run(tmp_path, _builders(), docroot=docroot, ALLOW_SHRINK='1')
    assert done.returncode == 0
    assert len(json.loads((docroot / 'jwst_gc_focal_plane.json').read_text())['pointings']) == 3


def test_the_first_build_has_nothing_to_shrink_from(tmp_path):
    done, docroot, _ = _run(tmp_path, _builders())
    assert done.returncode == 0
    assert (docroot / 'jwst_gc_focal_plane.json').is_file()


def test_rsync_is_skipped_when_deploy_is_off(tmp_path):
    done, docroot, rsync_log = _run(tmp_path, _builders(), DEPLOY='0')
    assert done.returncode == 0
    assert (docroot / 'jwst_gc_focal_plane.json').is_file()
    assert not rsync_log.exists()


def test_a_failed_rsync_fails_the_job(tmp_path):
    done, docroot, _ = _run(tmp_path, _builders(), rsync_rc=23)
    assert done.returncode == 23
    assert 'rsync rc=23' in done.stderr
    assert (docroot / 'jwst_gc_focal_plane.json').is_file()


def test_a_dry_run_publishes_nothing_and_keeps_the_staged_files(tmp_path):
    docroot = tmp_path / 'docroot'
    docroot.mkdir()
    _seed(docroot, focal=_focal(3, 1))
    done, _, rsync_log = _run(tmp_path, _builders(), docroot=docroot, DRY_RUN='1')
    assert done.returncode == 0
    assert json.loads((docroot / 'jwst_gc_focal_plane.json').read_text())['n_as_flown'] == 1
    assert not rsync_log.exists()
    assert len([p for p in os.listdir(docroot) if p.startswith('.')]) == 2


def test_the_exposures_override_reaches_the_focal_plane_builder(tmp_path):
    seen = tmp_path / 'args.txt'
    body = f'echo "$@" >> {seen}\n' + _builders()
    _run(tmp_path, body, EXPOSURES='/some/release/exposures')
    first = seen.read_text().splitlines()[0]
    assert '--exposures /some/release/exposures' in first


def test_a_stale_checkout_publishes_nothing(tmp_path):
    up = tmp_path / 'up'
    up.mkdir()
    env = dict(os.environ, GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@x',
               GIT_COMMITTER_NAME='t', GIT_COMMITTER_EMAIL='t@x')
    sh = lambda *a, **k: subprocess.run(['git', *a], check=True, env=env, **k)
    sh('init', '-q', '-b', 'main', str(up))
    (up / 'f').write_text('0')
    sh('-C', str(up), 'add', '.')
    sh('-C', str(up), 'commit', '-qm', 'base')
    repo = tmp_path / 'repo'
    sh('clone', '-q', str(up), str(repo))
    (up / 'f').write_text('1')
    sh('-C', str(up), 'commit', '-qam', 'c1')
    (repo / 'f').write_text('local edit')       # blocks the fast-forward
    done, docroot, rsync_log = _run(tmp_path, _builders(), REPO=str(repo),
                                    ALLOW_STALE_CHECKOUT='0')
    assert done.returncode == 3
    assert 'behind' in done.stderr
    assert os.listdir(docroot) == []
    assert not rsync_log.exists()


# --- the helpers --------------------------------------------------------------

def test_the_newest_release_wins_and_orders_numerically(tmp_path):
    vd = _module('viewer_data')
    for name in ('v1.8-2026.09', 'v1.9-2026.10', 'v1.10-2026.11', 'v2.0-x'):
        (tmp_path / name / 'gc-treasury' / 'exposures').mkdir(parents=True)
    (tmp_path / 'v3.0-2027.01').mkdir()          # no Treasury exposures
    (tmp_path / 'site').mkdir()
    # v2.0 sorts above v1.10; v3.0 is passed over because it has no frames
    assert vd.newest_release_exposures(str(tmp_path)).endswith(
        os.path.join('v2.0-x', 'gc-treasury', 'exposures'))
    (tmp_path / 'v2.0-x' / 'gc-treasury' / 'exposures').rmdir()
    assert vd.newest_release_exposures(str(tmp_path)).endswith(
        os.path.join('v1.10-2026.11', 'gc-treasury', 'exposures'))


def test_resolve_exposures_precedence(tmp_path, monkeypatch):
    vd = _module('viewer_data')
    (tmp_path / 'v1.9-2026.10' / 'gc-treasury' / 'exposures').mkdir(parents=True)
    monkeypatch.delenv('EXPOSURES', raising=False)
    assert vd.resolve_exposures(None, str(tmp_path)).startswith(str(tmp_path))
    monkeypatch.setenv('EXPOSURES', '/from/env')
    assert vd.resolve_exposures(None, str(tmp_path)) == '/from/env'
    assert vd.resolve_exposures('/explicit', str(tmp_path)) == '/explicit'
    assert vd.resolve_exposures('', str(tmp_path)) == ''     # plan attitude kept
    monkeypatch.delenv('EXPOSURES')
    with pytest.raises(SystemExit):
        vd.resolve_exposures(None, str(tmp_path / 'empty'))


def test_release_name_is_read_from_the_path():
    vd = _module('viewer_data')
    assert vd.release_name('/o/releases/v1.9-2026.10/gc-treasury/exposures') == 'v1.9-2026.10'
    assert vd.release_name('/elsewhere/frames') is None
    assert vd.release_name('') is None


def test_the_builders_default_to_the_docroot_the_publisher_uses():
    """`htdocs_staging` does not exist, so a bare run wrote somewhere nothing
    serves from.  The docroot constant is repeated in `viewer_data` so a builder
    need not import the publisher; this keeps the two equal."""
    ph = _module('publish_hips_layers')
    vd = _module('viewer_data')
    gs = _module('build_guidestars')
    assert vd.DOCROOT == ph.DOCROOT
    assert os.path.dirname(gs.DEFAULT_OUT) == ph.DOCROOT
    assert os.path.basename(gs.DEFAULT_OUT) == 'jwst_gc_cat_guidestars.json'
    assert gs.DEFAULT_EXPOSURES is None


def test_the_focal_plane_builder_default_out_and_release_pin():
    pytest.importorskip('pysiaf')
    bfp = _module('build_focal_plane')
    ph = _module('publish_hips_layers')
    assert os.path.dirname(bfp.DEFAULT_OUT) == ph.DOCROOT
    assert os.path.basename(bfp.DEFAULT_OUT) == 'jwst_gc_focal_plane.json'
    assert bfp.DEFAULT_EXPOSURES is None


def test_web_target_comes_from_the_publisher():
    vd = _module('viewer_data')
    ph = _module('publish_hips_layers')
    assert vd.web_target() == (ph.WEB_HOST, ph.WEB_DIR)


# --- the documents say when and from what they were built -----------------------

def test_the_focal_plane_records_when_and_from_which_release(tmp_path):
    pytest.importorskip('pysiaf')
    import argparse
    bfp = _module('build_focal_plane')
    src = tmp_path / 'footprints.json'
    src.write_text(json.dumps({'pa_v3': 87.0, 'planned': [
        {'number': '1', 'ra': 266.5, 'dec': -28.9}]}))
    out = tmp_path / 'fp.json'
    exposures = str(tmp_path / 'releases' / 'v1.9-2026.10' / 'gc-treasury' / 'exposures')
    bfp.build(argparse.Namespace(footprints=str(src), out=str(out),
                                 anchor='NRCALL_FULL', programme='10678',
                                 exposures=exposures))
    doc = json.loads(out.read_text())
    assert doc['release'] == 'v1.9-2026.10'
    assert doc['generated'].endswith('Z') and 'T' in doc['generated']
    # a caller with no exposures still gets a stamp, and no release
    bfp.build(argparse.Namespace(footprints=str(src), out=str(out),
                                 anchor='NRCALL_FULL', programme='10678'))
    doc = json.loads(out.read_text())
    assert doc['release'] is None and doc['generated']


def test_the_guide_star_document_records_when_and_from_where(tmp_path, monkeypatch):
    gs = _module('build_guidestars')
    per_star = {'S1': {'ra': 1.0, 'dec': 2.0, 'gs_mag': 12.0, 'orders': {1},
                       'obs': {'o001'}, 'visits': {'v1'}, 'frames': 1}}
    per_visit = {'v1': {'observation': 'o001', 'stars': {'S1': 1}}}
    monkeypatch.setattr(gs, 'scan_mast', lambda *a, **k: (per_star, per_visit, []))
    out = tmp_path / 'gs.json'
    gs.main(['--out', str(out), '--footprints', str(tmp_path / 'none.json')])
    doc = json.loads(out.read_text())
    assert doc['source'] == 'mast' and 'release' not in doc
    assert doc['generated'].endswith('Z')
