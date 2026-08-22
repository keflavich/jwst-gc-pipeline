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
import re
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


def _locator_block(name):
    """The pipe-root block as literally written in a submitter.

    Extracted rather than executed: running a whole submitter would launch a
    real reduction/cataloging job.  The block is taken verbatim from the file,
    so it is still the shipped code under test.
    """
    path = os.path.join(REPO, 'scripts', 'reduction', name)
    lines = open(path).read().split('\n')
    i = next(k for k, ln in enumerate(lines) if ln.strip().startswith('_PR_H='))
    j = next(k for k in range(i, len(lines)) if lines[k].strip() == 'fi')
    return '\n'.join(ln.strip() for ln in lines[i:j + 1])


def test_the_locator_survives_sbatch_copying_the_script(tmp_path):
    """sbatch COPIES the batch script to a spool dir.  Verified on this cluster
    (job 39949056)::

        BASH_SOURCE0=[/var/spool/slurmd/job39949056/slurm_script]
        dirname=[/var/spool/slurmd/job39949056]

    So `dirname $BASH_SOURCE` is the SPOOL directory and cannot be the only way
    a job finds its own helper -- the first version of this change would have
    failed on every pinned job.  Here the block runs from a spool stand-in with
    only PIPE_ROOT set, which is what a hand-submitted job has, and must fall
    back to $PIPE_ROOT/scripts/reduction.
    """
    spool = tmp_path / 'var' / 'spool' / 'job1'
    spool.mkdir(parents=True)
    block = _locator_block('submit_cataloging.sbatch')
    r = subprocess.run(['bash', '-c', block], capture_output=True, text=True,
                       cwd=str(spool), timeout=300,
                       env={**{k: v for k, v in os.environ.items()
                               if k not in ('GC_SCRIPTS_DIR', 'SLURM_SUBMIT_DIR')},
                            'PIPE_ROOT': REPO, 'PYTHON': sys.executable})
    out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert 'jwst_gc_pipeline resolves to' in out, (
        'the copied script could not locate _pipe_root.sh:\n' + out)
    assert os.path.realpath(REPO) in out


def test_the_locator_refuses_when_the_helper_is_unreachable(tmp_path):
    """The other half: PIPE_ROOT set, helper genuinely unreachable.  Continuing
    unpinned is the failure this exists to stop, so a dry fallback chain must
    exit rather than skip."""
    spool = tmp_path / 'spool'
    spool.mkdir()
    fake_root = tmp_path / 'root'
    (fake_root / 'jwst_gc_pipeline').mkdir(parents=True)
    (fake_root / 'jwst_gc_pipeline' / '__init__.py').write_text('')
    # fake_root has the package but no scripts/reduction/_pipe_root.sh
    block = _locator_block('submit_cataloging.sbatch')
    r = subprocess.run(['bash', '-c', block], capture_output=True, text=True,
                       cwd=str(spool), timeout=300,
                       env={**{k: v for k, v in os.environ.items()
                               if k not in ('GC_SCRIPTS_DIR', 'SLURM_SUBMIT_DIR')},
                            'PIPE_ROOT': str(fake_root),
                            'PYTHON': sys.executable})
    assert r.returncode == 2, r.stdout + r.stderr
    assert 'not found -- refusing' in r.stdout + r.stderr


def test_an_unpinned_job_does_not_refuse_when_the_helper_is_missing(tmp_path):
    """No PIPE_ROOT means nothing to pin, so an unreachable helper is not an
    error -- production runs must not start failing because of this change."""
    spool = tmp_path / 'spool'
    spool.mkdir()
    block = _locator_block('submit_cataloging.sbatch')
    r = subprocess.run(['bash', '-c', block], capture_output=True, text=True,
                       cwd=str(spool), timeout=300,
                       env={**{k: v for k, v in os.environ.items()
                               if k not in ('GC_SCRIPTS_DIR', 'SLURM_SUBMIT_DIR',
                                            'PIPE_ROOT')},
                            'PYTHON': sys.executable})
    assert r.returncode == 0, r.stdout + r.stderr


@pytest.mark.parametrize('name', SUBMITTERS)
def test_every_submitter_has_the_locator_block(name):
    """All ten, or the one that lacks it silently keeps the old behaviour."""
    assert _locator_block(name).strip(), f'{name} has no locator block'


def test_the_driver_hands_down_the_scripts_dir():
    """GC_SCRIPTS_DIR is the first and most direct candidate; the chain-driver
    knows where it lives, so it should not make the job guess."""
    src = open(os.path.join(REPO, 'scripts', 'reduction',
                            'submit_cataloging_perframe.sh')).read()
    assert 'export GC_SCRIPTS_DIR=' in src


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


# --------------------------------------------------------------------------
# provenance: WHICH CODE, not just which tree
# --------------------------------------------------------------------------

def test_the_report_names_the_commit(tmp_path):
    """A path says which tree ran; it does not say which code.

    Two jobs that both printed `.../jwst-gc-pipeline-wt-excl` this week ran
    different code, because main was merged into that worktree between their
    submissions.  Answering "did sgrc job 39941339 have the per-field floor
    (#478)?" meant comparing its submit time against a merge time.  The commit
    makes it one line.
    """
    r = _run(HELPER, {'PIPE_ROOT': REPO})
    assert r.returncode == 0, r.stderr + r.stdout
    m = re.search(r'resolves to \S+ @ ([0-9a-f]{7,})', r.stdout)
    assert m, f'no commit in the report:\n{r.stdout}'
    head = subprocess.run(['git', '-C', REPO, 'rev-parse', '--short', 'HEAD'],
                          capture_output=True, text=True).stdout.strip()
    assert m.group(1) == head


def test_a_dirty_tree_says_so(tmp_path):
    """A dirty checkout is running code that is in no commit at all, so the
    sha alone would be misleading."""
    r = _run(HELPER, {'PIPE_ROOT': REPO})
    out = r.stdout
    dirty = bool(subprocess.run(['git', '-C', REPO, 'status', '--porcelain'],
                                capture_output=True, text=True).stdout.strip())
    assert ('(DIRTY)' in out) == dirty, out


def test_provenance_never_blocks_a_run(tmp_path):
    """Reporting is not a gate.  A PIPE_ROOT that holds the package but is not
    a git repo still runs -- refusing there would stop reductions for the sake
    of a log line."""
    root = tmp_path / 'notarepo'
    (root / 'jwst_gc_pipeline').mkdir(parents=True)
    (root / 'jwst_gc_pipeline' / '__init__.py').write_text('')
    r = _run(HELPER, {'PIPE_ROOT': str(root)})
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'resolves to' in r.stdout
    assert '@' not in r.stdout.split('resolves to')[1].split('\n')[0]
