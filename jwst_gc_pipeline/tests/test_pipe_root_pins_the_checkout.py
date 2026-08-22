"""PIPE_ROOT must actually pin the checkout, or refuse.

Exporting ``PYTHONPATH="$PIPE_ROOT:$PYTHONPATH"`` does not pin anything.  SLURM
starts a job in the SUBMIT directory and ``python -m`` puts the cwd at the FRONT
of ``sys.path``, ahead of everything PYTHONPATH contributes.  So submitting from
the main checkout with ``PIPE_ROOT=<worktree>`` runs MAIN's code while every log
line names the worktree -- silently, because both checkouts import fine.

Measured 2026-08-22: three fields (gc2211_o023, gc2211_o028,
cloudef_controlfield) were resubmitted with ``PIPE_ROOT=<worktree holding the
fix>`` and all 24 m12 shards failed against MAIN's unfixed code, with the
traceback naming ``/repos/jwst-gc-pipeline/...`` rather than the worktree.

``scripts/reduction/_pipe_root.sh`` cds first, then sets PYTHONPATH, then
verifies by importing and comparing paths.  The verification is the load-bearing
part: it turns a silent wrong-code run into a refusal in the log's first lines.
"""
import os
import subprocess
import sys

import pytest


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HELPER = os.path.join(REPO, 'scripts', 'reduction', '_pipe_root.sh')

#: every submitter that accepts PIPE_ROOT
SUBMITTERS = [
    'submit_cataloging.sbatch',
    'submit_cataloging_m7.sbatch',
    'submit_cataloging_m8_merge.sbatch',
    'submit_cataloging_m8_partial.sbatch',
    'submit_cataloging_niriss.sbatch',
    'submit_cataloging_perframe_phase.sbatch',
    'submit_merge.sbatch',
    'submit_reduction.sbatch',
    'submit_reduction_miri.sbatch',
    'submit_reduction_niriss.sbatch',
]


def _run(script, env):
    full = dict(os.environ)
    full.pop('PIPE_ROOT', None)
    full.update(env)
    full['PYTHON'] = sys.executable
    return subprocess.run(['bash', script], capture_output=True, text=True,
                          env=full, cwd=REPO, timeout=300)


@pytest.mark.parametrize('name', SUBMITTERS)
def test_every_submitter_routes_through_the_helper(name):
    """A submitter that hand-rolls the PYTHONPATH line reintroduces the bug in
    exactly one place, which is how it survived across ten scripts."""
    path = os.path.join(REPO, 'scripts', 'reduction', name)
    src = open(path).read()
    assert '_pipe_root.sh' in src, f'{name} does not source the helper'
    assert 'export PYTHONPATH="$PIPE_ROOT' not in src, (
        f'{name} still sets PYTHONPATH itself; that alone does not pin anything')


def test_helper_is_a_noop_without_pipe_root(tmp_path):
    """The production case: no PIPE_ROOT, no cd, no refusal -- it just reports
    where the package came from."""
    r = _run(HELPER, {})
    assert r.returncode == 0, r.stderr
    assert 'jwst_gc_pipeline resolves to' in r.stdout


def test_helper_refuses_a_pipe_root_without_the_package(tmp_path):
    r = _run(HELPER, {'PIPE_ROOT': str(tmp_path)})
    assert r.returncode == 2
    assert 'no jwst_gc_pipeline' in r.stderr


def test_helper_accepts_this_checkout():
    r = _run(HELPER, {'PIPE_ROOT': REPO})
    assert r.returncode == 0, r.stderr + r.stdout
    assert os.path.realpath(REPO) in r.stdout


def test_pythonpath_alone_does_not_pin_but_the_helper_does(tmp_path):
    """The bug and the fix, side by side, in the mechanism that caused it.

    A stand-in "worktree" holds a package that reports its own location.  The
    job is launched from a DIFFERENT directory that also holds a package --
    that directory is what SLURM's WorkDir is, and it is why the offsets fix
    sat unused: the submit dir was the main checkout.

    - PYTHONPATH alone: the cwd wins, the worktree loses, exit 0.  Silent.
    - the helper: cd first, then verify -> the worktree wins.

    Two package directories are used rather than the real checkout because the
    point is which of two candidates `python -m` picks, and that is decided by
    sys.path order alone.
    """
    for name in ('submitdir', 'worktree'):
        pkg = tmp_path / name / 'jwst_gc_pipeline'
        pkg.mkdir(parents=True)
        (pkg / '__init__.py').write_text('')
    submitdir, worktree = tmp_path / 'submitdir', tmp_path / 'worktree'

    probe = ('import os, jwst_gc_pipeline as p; '
             'print(os.path.dirname(os.path.dirname(os.path.abspath(p.__file__))))')

    def where(setup):
        r = subprocess.run(['bash', '-c', f'{setup}; "$PYTHON" -c "{probe}"'],
                           capture_output=True, text=True, cwd=str(submitdir),
                           env={**os.environ, 'PYTHON': sys.executable,
                                'PIPE_ROOT': str(worktree), 'PYTHONPATH': ''},
                           timeout=300)
        assert r.returncode == 0, r.stderr
        return os.path.realpath(r.stdout.strip())

    old = where('export PYTHONPATH="$PIPE_ROOT:$PYTHONPATH"')
    assert old == os.path.realpath(submitdir), (
        'expected the OLD behaviour to lose to the cwd; if this ever equals '
        'the worktree, python resolution changed and the helper may be moot')

    new = where(f'. {HELPER!r} >/dev/null')
    assert new == os.path.realpath(worktree), (
        f'helper failed to pin: imported {new}, wanted {worktree}')
    assert old != new


@pytest.mark.parametrize('name', SUBMITTERS)
def test_submitters_stay_syntactically_valid(name):
    path = os.path.join(REPO, 'scripts', 'reduction', name)
    r = subprocess.run(['bash', '-n', path], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
