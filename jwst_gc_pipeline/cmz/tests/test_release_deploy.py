"""Tests for scripts/release/deploy_site.sh.

The release site and the pipeline monitor are two independently generated trees
that share one docroot: `htdocs/jwst-gc/` is written by make_webpage.py, and
`htdocs/jwst-gc/monitor/` by the monitoring deploy. `releases/site/` contains no
`monitor/`, so a release sync carrying `--delete` sees the whole 194 MB monitor
tree as extraneous and removes it -- which is what took
https://starformation.astro.ufl.edu/jwst-gc/monitor/ offline on 2026-08-06.

`ssh` and `rsync` are stubbed, but not into recorders: `ssh` runs the command
locally and `rsync` is the real rsync with the `host:` prefix stripped, so the
sync actually happens against a temp-dir "docroot". The protect rule is
therefore tested by its effect, not by asserting that a flag was passed --
`test_dropping_the_protect_rule_is_what_kills_the_monitor` replays the recorded
argv without it and shows the monitor die.
"""
import os
import shutil
import subprocess

import pytest

# .../jwst_gc_pipeline/cmz/tests/test_release_deploy.py -> repo root (4 up)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
_DEPLOY = os.path.join(_REPO, 'scripts', 'release', 'deploy_site.sh')
_HOST = 'testhost'

_SSH = '''#!/bin/bash
# $1 is the host, $2 the command; run it here instead of over the network.
echo "$2" >> "$SSH_LOG"
shift
exec bash -c "$*"
'''

_RSYNC = '''#!/bin/bash
printf '%s\\n' "$@" > "$RSYNC_ARGS"
args=()
for a in "$@"; do args+=("${a#HOST:}"); done
exec "$REAL_RSYNC" "${args[@]}"
'''.replace('HOST:', _HOST + ':')

# A sync that removes the monitor even though it was told not to -- the failure
# mode the post-sync check exists to catch.
_RSYNC_SABOTEUR = '''#!/bin/bash
printf '%s\\n' "$@" > "$RSYNC_ARGS"
args=()
for a in "$@"; do args+=("${a#HOST:}"); done
"$REAL_RSYNC" "${args[@]}"
rm -rf "$SABOTAGE_TARGET"
'''.replace('HOST:', _HOST + ':')


def _real_rsync():
    path = shutil.which('rsync')
    if path is None:
        pytest.skip('rsync is not installed')
    return path


def _stub_bin(tmp_path, rsync_body=_RSYNC):
    binned = tmp_path / 'bin'
    binned.mkdir(exist_ok=True)
    for name, body in (('ssh', _SSH), ('rsync', rsync_body)):
        script = binned / name
        script.write_text(body)
        script.chmod(0o755)
    return binned


def _site(tmp_path):
    """A freshly generated release site: an index, a field page, an asset."""
    src = tmp_path / 'site'
    (src / 'assets').mkdir(parents=True)
    (src / 'index.html').write_text('<h1>release</h1>')
    (src / 'brick.html').write_text('<h1>brick</h1>')
    (src / 'assets' / 'brick.jpg').write_bytes(b'\xff\xd8jpeg')
    return src


def _docroot(tmp_path, with_monitor=True):
    """The docroot as the server holds it: last release + the monitor tree."""
    dest = tmp_path / 'htdocs' / 'jwst-gc'
    dest.mkdir(parents=True)
    (dest / 'index.html').write_text('<h1>old release</h1>')
    (dest / 'retired_field.html').write_text('a field no longer in the release')
    if with_monitor:
        (dest / 'monitor' / 'figures').mkdir(parents=True)
        (dest / 'monitor' / 'index.html').write_text('<h1>monitor</h1>')
        (dest / 'monitor' / 'figures' / 'brick.png').write_bytes(b'\x89PNG')
    return dest


def _run(tmp_path, src, dest, args=(), rsync_body=_RSYNC, extra_env=None):
    binned = _stub_bin(tmp_path, rsync_body)
    env = dict(os.environ,
               PATH=f'{binned}:{os.environ["PATH"]}',
               REAL_RSYNC=_real_rsync(),
               RSYNC_ARGS=str(tmp_path / 'rsync_argv'),
               SSH_LOG=str(tmp_path / 'ssh_log'),
               RELEASE_WEB_HOST=_HOST,
               RELEASE_WEB_DIR=str(dest))
    env.update(extra_env or {})
    proc = subprocess.run(['bash', _DEPLOY, *args, str(src)],
                          capture_output=True, text=True, env=env)
    argv_file = tmp_path / 'rsync_argv'
    argv = argv_file.read_text().splitlines() if argv_file.exists() else []
    return proc, argv


def test_the_release_sync_leaves_the_monitor_alone(tmp_path):
    src, dest = _site(tmp_path), _docroot(tmp_path)
    proc, argv = _run(tmp_path, src, dest)

    assert proc.returncode == 0, proc.stderr
    # the monitor tree survives, whole
    assert (dest / 'monitor' / 'index.html').read_text() == '<h1>monitor</h1>'
    assert (dest / 'monitor' / 'figures' / 'brick.png').exists()
    # ... and --delete is genuinely active, so this is not just a no-op sync
    assert not (dest / 'retired_field.html').exists()
    assert (dest / 'index.html').read_text() == '<h1>release</h1>'
    assert (dest / 'assets' / 'brick.jpg').exists()
    assert '--delete' in argv


def test_dropping_the_protect_rule_is_what_kills_the_monitor(tmp_path):
    """The mutation: same sync, protect/exclude removed, monitor gone."""
    src, dest = _site(tmp_path), _docroot(tmp_path)
    _, argv = _run(tmp_path, src, dest)
    assert (dest / 'monitor' / 'index.html').exists()

    unprotected = [a for a in argv
                   if not a.startswith(('--filter=', '--exclude='))]
    assert len(unprotected) == len(argv) - 2, argv
    unprotected = [a.replace(f'{_HOST}:', '') for a in unprotected]
    subprocess.run([_real_rsync(), *unprotected], check=True,
                   capture_output=True)

    assert not (dest / 'monitor').exists()


def test_it_refuses_a_destination_that_is_not_the_release_root(tmp_path):
    src = _site(tmp_path)
    for bad in ('htdocs/jwst-gc/monitor',      # would bury the monitor
                'htdocs',                      # would --delete every project
                'htdocs/jwst-gc-old'):
        dest = tmp_path / bad
        dest.mkdir(parents=True, exist_ok=True)
        proc, argv = _run(tmp_path, src, dest)
        assert proc.returncode == 2, (bad, proc.returncode, proc.stderr)
        assert argv == [], f'{bad}: rsync ran anyway'
        assert 'must end in /jwst-gc' in proc.stderr


def test_a_trailing_slash_on_the_destination_is_still_the_release_root(tmp_path):
    src, dest = _site(tmp_path), _docroot(tmp_path)
    proc, argv = _run(tmp_path, src, dest, extra_env={
        'RELEASE_WEB_DIR': str(dest) + '/'})
    assert proc.returncode == 0, proc.stderr
    assert (dest / 'monitor' / 'index.html').exists()


def test_it_refuses_a_source_that_was_never_generated(tmp_path):
    src = tmp_path / 'empty'
    src.mkdir()
    proc, argv = _run(tmp_path, src, _docroot(tmp_path))
    assert proc.returncode == 3
    assert 'make_webpage.py' in proc.stderr
    assert argv == []


def test_a_sync_that_loses_the_monitor_is_a_loud_failure(tmp_path):
    src, dest = _site(tmp_path), _docroot(tmp_path)
    proc, _ = _run(tmp_path, src, dest, rsync_body=_RSYNC_SABOTEUR,
                   extra_env={'SABOTAGE_TARGET': str(dest / 'monitor')})
    assert proc.returncode == 4
    assert 'deploy_monitor.sh' in proc.stderr        # says how to put it back
    assert 'deployed ->' not in proc.stdout


def test_a_docroot_that_never_had_a_monitor_is_not_a_failure(tmp_path):
    """First deployment of a new field group: nothing to lose, nothing to warn."""
    src, dest = _site(tmp_path), _docroot(tmp_path, with_monitor=False)
    proc, _ = _run(tmp_path, src, dest)
    assert proc.returncode == 0, proc.stderr
    assert 'deployed ->' in proc.stdout


def test_the_dry_run_touches_nothing(tmp_path):
    src, dest = _site(tmp_path), _docroot(tmp_path)
    proc, argv = _run(tmp_path, src, dest, args=('--dry-run',))
    assert proc.returncode == 0, proc.stderr
    assert '--dry-run' in argv
    assert (dest / 'retired_field.html').exists()
    assert (dest / 'index.html').read_text() == '<h1>old release</h1>'
    assert (dest / 'monitor' / 'index.html').exists()
    # a silent dry run reads as "no work to do"; it has to itemize
    assert '-i' in argv
