"""The m8 merge runs once per module, each against that module's own m7 anchor."""
import os
import stat
import subprocess

import pytest

SBATCH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '..', '..', 'scripts', 'reduction', 'submit_cataloging_m8_merge.sbatch'))


def _run(tmp_path, **env):
    repo = tmp_path / 'repo'
    (repo / 'scripts' / 'reduction').mkdir(parents=True)
    (repo / 'scripts' / 'reduction' / 'm8_merge_partials.py').write_text('')
    log = tmp_path / 'calls.log'
    fake = tmp_path / 'python'
    # Answers the package-root probe with the fake repo; records every merge call.
    fake.write_text('#!/bin/bash\n'
                    f'if [ "$1" = "-c" ]; then echo {repo}; exit 0; fi\n'
                    f'echo "$@" >> {log}\n')
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    full = {'PATH': os.environ['PATH'], 'HOME': str(tmp_path),
            'PYTHON': str(fake), 'TARGET': 'w51', 'FIELD': '001',
            'FILTERS': 'F140M F480M', 'CATDIR': str(tmp_path / 'cat')}
    full.update(env)
    r = subprocess.run(['bash', SBATCH], env=full, capture_output=True, text=True)
    calls = log.read_text().splitlines() if log.exists() else []
    # Other helpers also invoke $PYTHON (e.g. `python -`); keep only the merges.
    calls = [c for c in calls if 'm8_merge_partials.py' in c]
    return r, calls


def test_comma_module_list_runs_one_merge_per_module(tmp_path):
    r, calls = _run(tmp_path, MODULES='nrca,nrcb,merged')
    assert r.returncode == 0, r.stderr
    assert [c.split('--module=')[1].split()[0] for c in calls] == ['nrca', 'nrcb', 'merged']
    assert all('--filters=F140M,F480M' in c for c in calls)


def test_single_module_unchanged(tmp_path):
    r, calls = _run(tmp_path, MODULES='nrcb')
    assert r.returncode == 0, r.stderr
    assert len(calls) == 1 and '--module=nrcb' in calls[0]


def test_explicit_m7_runs_once(tmp_path):
    r, calls = _run(tmp_path, MODULES='nrca,nrcb', M7='/x/anchor_m7.fits')
    assert r.returncode == 0, r.stderr
    assert len(calls) == 1 and '--m7=/x/anchor_m7.fits' in calls[0]
    assert '--module' not in calls[0]
