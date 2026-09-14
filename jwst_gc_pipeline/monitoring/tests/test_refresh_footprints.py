"""`refresh_monitor.sh` must rebuild the footprints it publishes.

It did not.  The pages regenerated hourly while `footprints.json` sat at
whatever date somebody last ran the builder by hand, and because the sky view
renders fine from stale data there was nothing to notice.  Measured 2026-09-12:
the served page said 3 executed while STScI said 12 -- 9 tiles in nine hours.

These run the real script with a STUB python, so they test the wiring rather
than the builder.
"""
import json
import os
import stat
import subprocess

import pytest

_SH = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'scripts', 'monitoring', 'refresh_monitor.sh'))


def _stub(tmp_path, body):
    """A fake `python` the script will call for every step."""
    p = tmp_path / 'fakepython'
    p.write_text('#!/bin/bash\n' + body + '\n')
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def _run(tmp_path, body, **env):
    outdir = tmp_path / 'out'
    outdir.mkdir(exist_ok=True)
    pub = tmp_path / 'pub'
    pub.mkdir(exist_ok=True)
    e = dict(os.environ, REPO=os.path.abspath(os.path.join(
        os.path.dirname(_SH), '..', '..')),
        OUTDIR=str(outdir), PUBDIR=str(pub),
        PYTHON=_stub(tmp_path, body), MONITOR_DEPLOY='0')
    e.update(env)
    done = subprocess.run(['bash', _SH], env=e, capture_output=True, text=True,
                          timeout=120)
    return done, outdir


#: A stub that writes a plausible footprints file when asked to build one, and
#: succeeds silently for the two generator invocations.
_GOOD = '''
for a in "$@"; do
  if [ "$a" = "--out" ]; then next=1; continue; fi
  if [ -n "${next:-}" ]; then echo '{"program":"10678","n_planned":1}' > "$a"; exit 0; fi
done
exit 0
'''


def test_the_refresh_rebuilds_footprints_json(tmp_path):
    done, outdir = _run(tmp_path, _GOOD)
    fp = outdir / 'footprints.json'
    assert fp.is_file(), done.stdout + done.stderr
    assert json.loads(fp.read_text())['program'] == '10678'
    assert 'footprints_rc=0' in done.stdout


def test_a_failed_rebuild_keeps_the_previous_file_and_does_not_fail_the_job(tmp_path):
    """The status comes from a third-party page.  That being down is not a
    reason to stop publishing the pipeline's own status, and a half-written
    file would be WORSE than a stale one -- the sky view parses it client-side,
    so a truncated file is a broken map."""
    outdir = tmp_path / 'out'
    outdir.mkdir()
    good = {"program": "10678", "n_planned": 139, "keepme": True}
    (outdir / 'footprints.json').write_text(json.dumps(good))
    # the stub half-writes the temp file, then fails
    done, _ = _run(tmp_path, '''
for a in "$@"; do
  if [ "$a" = "--out" ]; then next=1; continue; fi
  if [ -n "${next:-}" ]; then printf '{"program": "106' > "$a"; exit 3; fi
done
exit 0
''')
    assert json.loads((outdir / 'footprints.json').read_text()) == good
    assert done.returncode == 0
    assert 'footprints_rc=3' in done.stdout
    assert 'keeping the' in done.stderr


def test_a_partial_write_never_replaces_a_good_file(tmp_path):
    """rc 0 but an empty output -- a builder killed after opening the file."""
    outdir = tmp_path / 'out'
    outdir.mkdir()
    (outdir / 'footprints.json').write_text('{"keepme": true}')
    done, _ = _run(tmp_path, '''
for a in "$@"; do
  if [ "$a" = "--out" ]; then next=1; continue; fi
  if [ -n "${next:-}" ]; then : > "$a"; exit 0; fi
done
exit 0
''')
    assert json.loads((outdir / 'footprints.json').read_text()) == {'keepme': True}
    assert done.returncode == 0


def test_no_temp_file_is_left_behind(tmp_path):
    done, outdir = _run(tmp_path, _GOOD)
    leftovers = [p for p in os.listdir(outdir) if p.startswith('.footprints')]
    assert leftovers == [], leftovers


def test_the_rebuild_can_be_turned_off(tmp_path):
    """Offline runs, or a deliberately pinned footprints.json."""
    outdir = tmp_path / 'out'
    outdir.mkdir()
    (outdir / 'footprints.json').write_text('{"pinned": true}')
    done, _ = _run(tmp_path, _GOOD, FOOTPRINTS='0')
    assert json.loads((outdir / 'footprints.json').read_text()) == {'pinned': True}
    assert done.returncode == 0


def test_footprints_are_rebuilt_before_the_pages_that_embed_them(tmp_path):
    """A rebuild after the render publishes the OLD statuses for another hour."""
    order = tmp_path / 'order.txt'
    done, _ = _run(tmp_path, f'''
for a in "$@"; do
  if [ "$a" = "--out" ]; then next=1; continue; fi
  if [ -n "${{next:-}}" ]; then
    echo footprints >> {order}
    echo '{{"program":"10678"}}' > "$a"; exit 0; fi
done
echo pages >> {order}
exit 0
''')
    assert order.read_text().split() == ['footprints', 'pages', 'pages']


def test_the_program_is_configurable(tmp_path):
    seen = tmp_path / 'args.txt'
    done, _ = _run(tmp_path, f'''
echo "$@" >> {seen}
for a in "$@"; do
  if [ "$a" = "--out" ]; then next=1; continue; fi
  if [ -n "${{next:-}}" ]; then echo '{{}}' > "$a"; exit 0; fi
done
exit 0
''', FOOTPRINT_PROGRAM='99999')
    assert '99999' in seen.read_text()


def test_a_rebuild_that_lost_its_statuses_keeps_the_previous_file(tmp_path):
    """`fetch_visit_status` returns {} on an unresolvable host, a 404 or a
    page-shape change, and `build` then exits 0 with a perfectly well-formed
    file carrying `status_counts: {}`.  Without a guard an STScI hiccup replaces
    a good file hourly and the page loses every status and every link.
    (Review of #852.)"""
    outdir = tmp_path / 'out'
    outdir.mkdir()
    good = {"program": "10678", "status_counts": {"Executed": 12},
            "n_planned": 127}
    (outdir / 'footprints.json').write_text(json.dumps(good))
    done, _ = _run(tmp_path, '''
for a in "$@"; do
  if [ "$a" = "--out" ]; then next=1; continue; fi
  if [ -n "${next:-}" ]; then
    echo '{"program":"10678","status_counts":{},"n_planned":139}' > "$a"; exit 0; fi
done
exit 0
''')
    assert json.loads((outdir / 'footprints.json').read_text()) == good
    assert done.returncode == 0
    assert 'no visit statuses' in done.stderr


def test_the_first_ever_build_is_not_blocked_by_the_guard(tmp_path):
    """With no previous file there is nothing to protect, and a status-less
    build is still better than none -- otherwise a fresh checkout could never
    write one."""
    outdir = tmp_path / 'out'
    outdir.mkdir()
    done, _ = _run(tmp_path, '''
for a in "$@"; do
  if [ "$a" = "--out" ]; then next=1; continue; fi
  if [ -n "${next:-}" ]; then
    echo '{"program":"10678","status_counts":{}}' > "$a"; exit 0; fi
done
exit 0
''')
    assert (outdir / 'footprints.json').is_file()
    assert done.returncode == 0


def test_a_rebuild_that_gained_statuses_replaces_a_status_less_one(tmp_path):
    outdir = tmp_path / 'out'
    outdir.mkdir()
    (outdir / 'footprints.json').write_text('{"status_counts": {}}')
    done, _ = _run(tmp_path, '''
for a in "$@"; do
  if [ "$a" = "--out" ]; then next=1; continue; fi
  if [ -n "${next:-}" ]; then
    echo '{"status_counts":{"Executed":12}}' > "$a"; exit 0; fi
done
exit 0
''')
    got = json.loads((outdir / 'footprints.json').read_text())
    assert got['status_counts'] == {'Executed': 12}


# --- publishing from a stale checkout ---------------------------------------

def _repo(tmp_path, behind):
    """A git repo whose HEAD is `behind` commits behind its own origin/main."""
    import subprocess as sp
    up = tmp_path / 'upstream'
    up.mkdir()
    sp.run(['git', 'init', '-q', '-b', 'main', str(up)], check=True)
    env = dict(os.environ, GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@x',
               GIT_COMMITTER_NAME='t', GIT_COMMITTER_EMAIL='t@x')
    (up / 'f').write_text('0')
    sp.run(['git', '-C', str(up), 'add', '.'], check=True)
    sp.run(['git', '-C', str(up), 'commit', '-qm', 'base'], env=env, check=True)
    repo = tmp_path / 'repo'
    sp.run(['git', 'clone', '-q', str(up), str(repo)], check=True)
    for i in range(behind):
        (up / 'f').write_text(str(i + 1))
        sp.run(['git', '-C', str(up), 'commit', '-qam', f'c{i}'], env=env, check=True)
    sp.run(['git', '-C', str(repo), 'fetch', '-q', 'origin'], check=True)
    return repo


def test_a_stale_checkout_generates_but_does_not_publish(tmp_path):
    """The monitor reverted three times over two days because this cron ran
    from a checkout 599 commits behind main with MONITOR_DEPLOY=1: it
    regenerated a months-old page hourly and published it over the current
    one.  Generating from a stale tree is harmless; publishing from one is
    what overwrites good pages."""
    repo = _repo(tmp_path, behind=3)
    done, outdir = _run(tmp_path, _GOOD, REPO=str(repo), MONITOR_DEPLOY='1')
    assert 'behind origin/main' in done.stderr
    assert 'deploy SKIPPED' in done.stderr
    assert done.returncode == 0            # a stale tree is not a job failure
    assert (outdir / 'footprints.json').is_file()   # generation still happened


def test_a_current_checkout_publishes(tmp_path):
    repo = _repo(tmp_path, behind=0)
    done, _ = _run(tmp_path, _GOOD, REPO=str(repo), MONITOR_DEPLOY='0')
    assert 'behind origin/main' not in done.stderr
    assert 'deploy SKIPPED' not in done.stderr


def test_the_override_exists_for_a_deliberate_branch_run(tmp_path):
    repo = _repo(tmp_path, behind=3)
    done, _ = _run(tmp_path, _GOOD, REPO=str(repo), MONITOR_DEPLOY='0',
                   ALLOW_STALE_CHECKOUT='1')
    assert 'behind origin/main' not in done.stderr


def test_an_unfetchable_remote_is_not_treated_as_staleness(tmp_path):
    """A network failure must not block the hourly run; it says so instead."""
    import subprocess as sp
    repo = tmp_path / 'norem'
    repo.mkdir()
    sp.run(['git', 'init', '-q', '-b', 'main', str(repo)], check=True)
    done, _ = _run(tmp_path, _GOOD, REPO=str(repo), MONITOR_DEPLOY='0')
    assert 'staleness not checked' in done.stderr
    assert 'deploy SKIPPED' not in done.stderr
    assert done.returncode == 0
