"""A re-tie loop runs one commit from start to finish (#679).

`run_field_retie_loop.sh` is frozen by bash at launch, but each reduce and
cataloging job imports `jwst_gc_pipeline` from `PIPE_ROOT` when it starts, and
the loop reads `$HERE/submit_*.sh` and `$HERE/offsets_value_digest.py` afresh on
every iteration.  Pointed at a live checkout, a fortnight-long run therefore
spans every commit merged meanwhile, while the provenance banner names one.

The loop now snapshots its source tree into a detached, locked worktree and
re-executes from it, so `PIPE_ROOT` and `$HERE` name files nothing can rewrite.
The pin is not removed at exit (the m3-m7 chain it submits is still queued);
`prune_retie_pins.sh` removes pins whose jobs have left the queue.

Everything here runs in throwaway git repositories under `tmp_path`.  The
re-exec itself is exercised on a copy of the script cut off at the end of the pin
block, so no sbatch is reachable.
"""
import os
import pathlib
import re
import stat
import subprocess

import pytest

ROOT = pathlib.Path(__file__).parents[2]
LOOP = ROOT / 'scripts' / 'reduction' / 'run_field_retie_loop.sh'
PRUNE = ROOT / 'scripts' / 'reduction' / 'prune_retie_pins.sh'
REL = 'scripts/reduction/run_field_retie_loop.sh'
PIN_END = 'QOS=${QOS:-astronomy-dept-b}\n'

GIT_ENV = {'GIT_AUTHOR_NAME': 't', 'GIT_AUTHOR_EMAIL': 't@t',
           'GIT_COMMITTER_NAME': 't', 'GIT_COMMITTER_EMAIL': 't@t'}


def _env(tmp_path, **extra):
    env = dict(PROPOSAL='4147', FIELD='012', TARGET='sgrc', FILTERS='F115W', OFFSETS_TBL='/tmp/unused.csv',
               PATH='/usr/bin:/bin:/usr/local/bin', HOME=str(tmp_path),
               RETIE_PIN_DIR=str(tmp_path / 'pins'), **GIT_ENV)
    env.update(extra)
    return env


def _git(repo, *args, tmp_path=None):
    return subprocess.run(['git', '-C', str(repo), *args], check=True,
                          capture_output=True, text=True,
                          env={**GIT_ENV, 'PATH': '/usr/bin:/bin',
                               'HOME': str(tmp_path or repo)}).stdout.strip()


def _source_repo(tmp_path, cut=False):
    """A repo holding the loop script and a stub package.

    With `cut` the committed script ends where the pin block does and prints
    what the re-executed process sees, so a real exec can be run safely.
    """
    repo = tmp_path / 'src'
    (repo / 'scripts' / 'reduction').mkdir(parents=True)
    (repo / 'jwst_gc_pipeline').mkdir()
    (repo / 'jwst_gc_pipeline' / '__init__.py').write_text('')
    (repo / 'marker.txt').write_text('v1\n')
    text = LOOP.read_text()
    if cut:
        text = text[:text.index(PIN_END)] + (
            'echo "RAN HERE=$HERE PIPE_ROOT=$PIPE_ROOT PINNED=${RETIE_PINNED:-}'
            ' SRC=${RETIE_PIN_SRC:-}"\nexit 0\n')
    (repo / REL).write_text(text)
    subprocess.run(['git', 'init', '-q', '-b', 'main', str(repo)], check=True)
    _git(repo, 'add', '-A', tmp_path=tmp_path)
    _git(repo, 'commit', '-q', '-m', 'init', tmp_path=tmp_path)
    return repo


def _call(tmp_path, repo, body, **env):
    """Run `body` with the loop's functions in scope (source-only mode)."""
    script = (f'set -euo pipefail\nsource "{repo / REL}" >/dev/null 2>&1\n'
              + body)
    return subprocess.run(
        ['bash', '-c', script], capture_output=True, text=True,
        env=_env(tmp_path, RETIE_LOOP_SOURCE_ONLY='1', **env), cwd=str(repo))


def _worktrees(repo, tmp_path):
    return [line.split(' ', 1)[1] for line in
            _git(repo, 'worktree', 'list', '--porcelain',
                 tmp_path=tmp_path).splitlines() if line.startswith('worktree ')]


# --- pin_checkout ----------------------------------------------------------

def test_pin_checkout_makes_a_detached_locked_worktree_at_head(tmp_path):
    repo = _source_repo(tmp_path)
    sha = _git(repo, 'rev-parse', 'HEAD', tmp_path=tmp_path)
    r = _call(tmp_path, repo, 'pin_checkout', PIPE_ROOT=str(repo))
    assert r.returncode == 0, r.stderr
    dest = pathlib.Path(r.stdout.strip())
    assert dest.parent == tmp_path / 'pins'
    assert re.fullmatch(
        rf'retie-sgrc4147-o012-\d{{8}}T\d{{6}}Z-{sha[:7]}', dest.name), dest.name
    assert _git(dest, 'rev-parse', 'HEAD', tmp_path=tmp_path) == sha
    assert _git(dest, 'branch', '--show-current', tmp_path=tmp_path) == ''
    porcelain = _git(repo, 'worktree', 'list', '--porcelain', tmp_path=tmp_path)
    assert 'locked' in porcelain
    assert (dest / REL).is_file()


def test_pin_checkout_defaults_to_the_script_own_checkout(tmp_path):
    """PIPE_ROOT unset: pin the tree this script lives in (question 2)."""
    repo = _source_repo(tmp_path)
    r = _call(tmp_path, repo, 'pin_checkout')
    assert r.returncode == 0, r.stderr
    dest = pathlib.Path(r.stdout.strip())
    assert str(dest) in _worktrees(repo, tmp_path)


def test_a_commit_after_pinning_does_not_reach_the_pin(tmp_path):
    """The vintage test: the source moves on, the pin does not."""
    repo = _source_repo(tmp_path)
    sha = _git(repo, 'rev-parse', 'HEAD', tmp_path=tmp_path)
    dest = pathlib.Path(_call(tmp_path, repo, 'pin_checkout',
                              PIPE_ROOT=str(repo)).stdout.strip())
    (repo / 'marker.txt').write_text('v2\n')
    _git(repo, 'commit', '-q', '-am', 'later', tmp_path=tmp_path)
    assert _git(repo, 'rev-parse', 'HEAD', tmp_path=tmp_path) != sha
    assert _git(dest, 'rev-parse', 'HEAD', tmp_path=tmp_path) == sha
    assert (dest / 'marker.txt').read_text() == 'v1\n'


def test_a_dirty_source_is_refused(tmp_path):
    repo = _source_repo(tmp_path)
    (repo / 'marker.txt').write_text('edited\n')
    r = _call(tmp_path, repo, 'pin_checkout', PIPE_ROOT=str(repo))
    assert r.returncode == 2, r.stdout + r.stderr
    assert 'marker.txt' in r.stderr
    assert 'RETIE_PIN_ALLOW_DIRTY' in r.stderr
    assert not (tmp_path / 'pins').exists()


def test_untracked_files_count_as_dirty(tmp_path):
    repo = _source_repo(tmp_path)
    (repo / 'new_input.csv').write_text('x\n')
    r = _call(tmp_path, repo, 'pin_checkout', PIPE_ROOT=str(repo))
    assert r.returncode == 2, r.stdout + r.stderr


def test_allow_dirty_pins_head_and_says_what_it_left_out(tmp_path):
    repo = _source_repo(tmp_path)
    (repo / 'marker.txt').write_text('edited\n')
    r = _call(tmp_path, repo, 'pin_checkout', PIPE_ROOT=str(repo),
              RETIE_PIN_ALLOW_DIRTY='1')
    assert r.returncode == 0, r.stderr
    assert 'DIRTY SOURCE, uncommitted changes NOT in the run' in r.stderr
    dest = pathlib.Path(r.stdout.strip())
    assert (dest / 'marker.txt').read_text() == 'v1\n'


def test_a_non_git_source_is_refused(tmp_path):
    repo = _source_repo(tmp_path)
    plain = tmp_path / 'plain'
    plain.mkdir()
    r = _call(tmp_path, repo, 'pin_checkout', PIPE_ROOT=str(plain))
    assert r.returncode == 2, r.stdout + r.stderr
    assert 'not a git checkout' in r.stderr


def test_dry_run_creates_nothing_and_says_what_it_would_do(tmp_path):
    repo = _source_repo(tmp_path)
    before = _worktrees(repo, tmp_path)
    r = _call(tmp_path, repo, 'pin_checkout --dry-run', PIPE_ROOT=str(repo))
    assert r.returncode == 0, r.stderr
    assert r.stdout.startswith('would pin ')
    assert _worktrees(repo, tmp_path) == before
    assert not (tmp_path / 'pins').exists()


# --- the re-exec -----------------------------------------------------------

def _run_cut(tmp_path, repo, **env):
    return subprocess.run(['bash', str(repo / REL)], capture_output=True,
                          text=True, env=_env(tmp_path, **env), cwd=str(repo))


def test_the_loop_re_executes_from_the_pinned_tree(tmp_path):
    repo = _source_repo(tmp_path, cut=True)
    r = _run_cut(tmp_path, repo, PIPE_ROOT=str(repo))
    assert r.returncode == 0, r.stdout + r.stderr
    ran = re.search(r'RAN HERE=(\S+) PIPE_ROOT=(\S+) PINNED=(\S*) SRC=(\S*)',
                    r.stdout)
    assert ran, r.stdout
    here, pipe_root, pinned, src = ran.groups()
    pins = tmp_path / 'pins'
    assert pipe_root.startswith(str(pins) + '/retie-sgrc4147-o012-')
    assert here == f'{pipe_root}/scripts/reduction'
    assert pinned == '1'
    assert src == str(repo)
    assert len(_worktrees(repo, tmp_path)) == 2


def test_an_already_pinned_run_does_not_recurse(tmp_path):
    repo = _source_repo(tmp_path, cut=True)
    r = _run_cut(tmp_path, repo, PIPE_ROOT=str(repo), RETIE_PINNED='1')
    assert r.returncode == 0, r.stdout + r.stderr
    assert f'PIPE_ROOT={repo} PINNED=1' in r.stdout
    assert _worktrees(repo, tmp_path) == [str(repo)]


def test_pin_off_runs_in_place_and_warns(tmp_path):
    repo = _source_repo(tmp_path, cut=True)
    r = _run_cut(tmp_path, repo, PIPE_ROOT=str(repo), RETIE_PIN='0')
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'pin: NONE (RETIE_PIN=0)' in r.stderr
    assert f'PIPE_ROOT={repo} PINNED=' in r.stdout
    assert _worktrees(repo, tmp_path) == [str(repo)]


def test_a_dirty_source_stops_the_launch_with_rc_2(tmp_path):
    repo = _source_repo(tmp_path, cut=True)
    (repo / 'marker.txt').write_text('edited\n')
    r = _run_cut(tmp_path, repo, PIPE_ROOT=str(repo))
    assert r.returncode == 2, r.stdout + r.stderr
    assert 'RAN' not in r.stdout


def test_provenance_only_creates_no_worktree(tmp_path):
    """The real script, so the banner and the early exit are the real ones."""
    if not (ROOT / '.git').exists():
        pytest.skip('not a git checkout')
    table = tmp_path / 'offsets.csv'
    table.write_text('Visit,Exposure,Filter,Module,Vgroup,dra (arcsec),'
                     'ddec (arcsec)\njw03958001001,1,F770W,none,2101,0.1,-0.1\n')
    before = _worktrees(ROOT, tmp_path)
    env = _env(tmp_path, PIPE_ROOT=str(ROOT), PYTHONPATH=str(ROOT),
               RETIE_PROVENANCE_ONLY='1', RETIE_PIN_ALLOW_DIRTY='1',
               OFFSETS_TBL=str(table), MAXITER='3', TARGET='sickle',
               PROPOSAL='3958', FILTERS='F770W')
    r = subprocess.run(['bash', str(LOOP)], capture_output=True, text=True,
                       env=env, cwd=str(ROOT), timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert re.search(r'pin\s+: would pin \S+ @ [0-9a-f]{7} -> ', r.stdout), r.stdout
    assert _worktrees(ROOT, tmp_path) == before
    assert not (tmp_path / 'pins').exists()


# --- the contract in the script text ---------------------------------------

def test_the_pin_precedes_every_use_of_pipe_root_and_here():
    src = LOOP.read_text()
    call = src.index('_pin_dest=$(pin_checkout)')
    assert src.index('pin_checkout () {') < call
    # first Python probe and first helper read come after the re-exec point
    assert call < src.index('PER_FIELD_FLOOR=$(')
    assert call < src.index('"$HERE/offsets_value_digest.py"')
    assert call < src.index('RETIE_RUN_START=$(date')
    # the source-only return (which tests rely on) is after it, and the call is
    # guarded against it
    assert call < src.index('[ -n "${RETIE_LOOP_SOURCE_ONLY:-}" ] && return')
    assert '[ -n "${RETIE_LOOP_SOURCE_ONLY:-}" ]' in src[:call]


def test_every_job_gets_the_post_pin_pipe_root_and_helpers_come_from_here():
    src = LOOP.read_text()
    # PIPE_ROOT is only ever assigned from itself, so what is forwarded is what
    # the re-exec put there.
    assigns = re.findall(r'^\s*PIPE_ROOT=(\S+)', src, re.M)
    assert assigns[0] == '${PIPE_ROOT:-}'
    assert 'PIPE_ROOT=$PIPE_ROOT' in src            # m12 and full chain
    assert 'PIPE_ROOT=$PIPE_ROOT"' in src           # export_common
    # every helper the loop runs is addressed through $HERE, which after the
    # exec is inside the pin
    code = [ln for ln in src.splitlines() if not ln.lstrip().startswith('#')]
    for helper in ('submit_reduction.sbatch', 'submit_cataloging_perframe.sh',
                   'offsets_value_digest.py'):
        uses = [ln for ln in code if helper in ln and 'echo' not in ln]
        assert uses, helper
        for ln in uses:
            assert f'$HERE/{helper}' in ln, ln
    assert src.count('HERE="$(cd') == 1


# --- prune_retie_pins.sh ---------------------------------------------------

def _stub_squeue(tmp_path, names=(), fail=False):
    bindir = tmp_path / 'bin'
    bindir.mkdir(exist_ok=True)
    body = 'exit 1\n' if fail else ''.join(f'echo {n}\n' for n in names)
    stub = bindir / 'squeue'
    stub.write_text('#!/bin/bash\n' + body)
    stub.chmod(stub.stat().st_mode | stat.S_IXUSR)
    return bindir


def _prune(tmp_path, *args, names=(), fail=False):
    bindir = _stub_squeue(tmp_path, names, fail)
    env = _env(tmp_path, PATH=f'{bindir}:/usr/bin:/bin', USER='tester')
    return subprocess.run(['bash', str(PRUNE), *args], capture_output=True,
                          text=True, env=env)


def _two_pins(tmp_path):
    repo = _source_repo(tmp_path)
    pins = tmp_path / 'pins'
    pins.mkdir()
    a = pins / 'retie-sgrc4147-o012-20261101T000000Z-abcdef1'
    b = pins / 'retie-cloudc2221-o002-20261101T000000Z-abcdef1'
    for p in (a, b):
        _git(repo, 'worktree', 'add', '-q', '--detach', str(p), 'HEAD',
             tmp_path=tmp_path)
        _git(repo, 'worktree', 'lock', str(p), tmp_path=tmp_path)
    return repo, a, b


def test_prune_dry_run_lists_and_removes_nothing(tmp_path):
    repo, a, b = _two_pins(tmp_path)
    r = _prune(tmp_path)
    assert r.returncode == 0, r.stderr
    assert f'removable {a}' in r.stdout and f'removable {b}' in r.stdout
    assert a.exists() and b.exists()


def test_prune_keeps_a_pin_with_a_queued_job_including_a_fanout_name(tmp_path):
    """The m4 fan-out names are what #1172's ONLY_SHARDS reruns resubmit."""
    repo, a, b = _two_pins(tmp_path)
    r = _prune(tmp_path, '--apply', names=['sgrc4147-o012-m6-fanout'])
    assert r.returncode == 0, r.stderr
    assert f'in use   {a}' in r.stdout
    assert a.exists()
    assert not b.exists()
    assert str(b) not in _worktrees(repo, tmp_path)


def test_prune_matches_the_whole_prefix_not_a_substring(tmp_path):
    repo, a, b = _two_pins(tmp_path)
    r = _prune(tmp_path, '--apply', names=['xsgrc4147-o012-reduce-retie1',
                                           'sgrc4147-o0120-m3-finalize'])
    assert r.returncode == 0, r.stderr
    assert not a.exists() and not b.exists()


def test_prune_removes_nothing_when_squeue_fails(tmp_path):
    repo, a, b = _two_pins(tmp_path)
    r = _prune(tmp_path, '--apply', fail=True)
    assert r.returncode == 1
    assert a.exists() and b.exists()


def test_prune_leaves_unrelated_directories_alone(tmp_path):
    repo, a, b = _two_pins(tmp_path)
    other = tmp_path / 'pins' / 'retie-notes'
    other.mkdir()
    r = _prune(tmp_path, '--apply')
    assert r.returncode == 0, r.stderr
    assert other.is_dir()


# --- review additions ------------------------------------------------------

def test_an_uncreatable_pin_is_refused_cleanly_not_exec_failed(tmp_path):
    """A failing `worktree add` inside $( ) || must stop the launch with rc 2."""
    repo = _source_repo(tmp_path)
    pins = tmp_path / 'pins'
    pins.mkdir()
    pins.chmod(0o500)
    try:
        if os.access(pins, os.W_OK):
            pytest.skip('directory permissions are not enforced here')
        r = _call(tmp_path, repo, 'pin_checkout', PIPE_ROOT=str(repo))
    finally:
        pins.chmod(0o700)
    assert r.returncode == 2, (r.returncode, r.stderr)
    assert 'REFUSING to pin' in r.stderr


def _lock_with_reason(repo, pin, reason, tmp_path):
    _git(repo, 'worktree', 'unlock', str(pin), tmp_path=tmp_path)
    _git(repo, 'worktree', 'lock', '--reason', reason, str(pin), tmp_path=tmp_path)


def test_prune_keeps_a_pin_whose_loop_is_still_running(tmp_path):
    """No queued job does not mean no loop: it may be between iterations."""
    import socket
    repo, a, b = _two_pins(tmp_path)
    host = socket.gethostname()
    hostname = subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip()
    _lock_with_reason(repo, a, f'retie loop {hostname} pid {os.getpid()}', tmp_path)
    r = _prune(tmp_path, '--apply')
    assert r.returncode == 0, r.stderr
    assert a.exists(), (host, r.stdout)
    assert f'in use   {a} (its loop' in r.stdout
    assert not b.exists()


def test_prune_removes_a_pin_whose_loop_has_exited(tmp_path):
    repo, a, b = _two_pins(tmp_path)
    hostname = subprocess.run(['hostname'], capture_output=True, text=True).stdout.strip()
    dead = subprocess.Popen(['true'])
    dead.wait()
    _lock_with_reason(repo, a, f'retie loop {hostname} pid {dead.pid}', tmp_path)
    r = _prune(tmp_path, '--apply')
    assert r.returncode == 0, r.stderr
    assert not a.exists()


def test_prune_keeps_a_pin_locked_from_another_host_unless_told(tmp_path):
    repo, a, b = _two_pins(tmp_path)
    _lock_with_reason(repo, a, 'retie loop some-other-login pid 1', tmp_path)
    r = _prune(tmp_path, '--apply')
    assert a.exists() and 'in use?' in r.stdout
    bindir = tmp_path / 'bin'
    env = _env(tmp_path, PATH=f'{bindir}:/usr/bin:/bin', USER='tester',
               RETIE_PRUNE_ASSUME_LOOPS_DEAD='1')
    r = subprocess.run(['bash', str(PRUNE), '--apply'], capture_output=True,
                       text=True, env=env)
    assert r.returncode == 0, r.stderr
    assert not a.exists()
