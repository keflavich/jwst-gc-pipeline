"""The re-tie loop refuses a convergence scope it can never apply.

`run_field_retie_loop.sh` decides "did this iteration re-tie anything?" by
digesting the offsets table before and after m2, scoped to its own observation
(#714).  A digest that FAILS is read fail-open -- `table_value_digest` emits a
fresh unique token, so the comparison reports movement and the loop keeps
going.  That is right for a transient (a table being rewritten under the read)
and wrong for a condition settled before the first iteration: then every call
fails the same way, `[ "$tbl_after" = "$tbl_before" ]` can never hold, the
"no SHIFT VALUE changed -> this is NOT a checkpoint re-tie -> STOPPING" branch
is dead for the whole run, and the loop re-reduces to MAXITER measuring the
same residual.  That is issue #272's behaviour, produced by the file written to
remove it.

Two such conditions exist and are operator-reachable today: a `FIELD` that does
not spell an observation, and an `OFFSETS_TBL` with no `Visit` column
(`brick/offsets/Offsets_JWST_Brick2221_average.csv` and its two siblings, 30
rows each, carry none).  Both are now refused at startup, beside the existing
`CONSENSUS_TBL` refusal, in two layers: a bash shape check on `FIELD` that
holds even where `python` does not resolve, and a probe of the digest itself,
which is the only thing that can see whether the TABLE can be attributed.

The loop is driven for real here, under `RETIE_PROVENANCE_ONLY=1`, which exits
before any `sbatch`; the preflight runs ahead of both.
"""
import os
import pathlib
import re
import subprocess
import sys

import pytest

astropy_table = pytest.importorskip("astropy.table")
Table = astropy_table.Table

_ROOT = pathlib.Path(__file__).parents[3]
_LOOP = _ROOT / 'scripts' / 'reduction' / 'run_field_retie_loop.sh'


def _table(visits=('jw03958001001', 'jw03958002001', 'jw03958007001')):
    n = len(visits)
    return Table({
        'Visit': list(visits),
        'Exposure': list(range(1, n + 1)),
        'Filter': ['F770W'] * n,
        'Module': ['none'] * n,
        'Vgroup': ['2101'] * n,
        'dra (arcsec)': [0.1] * n,
        'ddec (arcsec)': [-0.1] * n,
    })


def _run(field, table_path, **env_extra):
    env = {
        'PATH': os.path.dirname(sys.executable) + ':/usr/bin:/bin',
        'HOME': os.environ.get('HOME', '/tmp'),
        'PROPOSAL': '3958', 'TARGET': 'sickle', 'FILTERS': 'F770W',
        'FIELD': field,
        'OFFSETS_TBL': str(table_path),
        'PIPE_ROOT': str(_ROOT),
        'PYTHONPATH': str(_ROOT),
        'RETIE_PROVENANCE_ONLY': '1',
        'MAXITER': '3',
    }
    env.update(env_extra)
    proc = subprocess.run(['bash', str(_LOOP)], capture_output=True, text=True,
                          env=env, cwd=str(_ROOT), timeout=600)
    return proc.returncode, proc.stdout + proc.stderr


def test_the_joint_field_the_registry_registers_is_accepted(tmp_path):
    """THE regression.  `alignment_config` registers sickle's MIRI as
    ('001-002', '001', '002') and `offsets_table_path` hands '001-002' a real
    table, so this FIELD reaches the loop -- and used to make every digest exit
    2, disabling the stop condition for the run."""
    path = tmp_path / 'offsets.csv'
    _table().write(path, format='ascii.csv', overwrite=True)
    rc, out = _run('001-002', path)
    assert rc == 0, out
    assert 'convergence digest scoped to observation: 001-002' in out
    assert 'REFUSING' not in out


def test_a_plain_observation_is_still_accepted(tmp_path):
    path = tmp_path / 'offsets.csv'
    _table(('jw02092002001', 'jw02092005001')).write(
        path, format='ascii.csv', overwrite=True)
    rc, out = _run('002', path)
    assert rc == 0, out
    assert 'convergence digest scoped to observation: 002' in out


def test_an_absent_table_does_not_refuse(tmp_path):
    """A first iteration starts with no table at all; that digests to `none`
    with status 0 and must not be read as an unusable scope."""
    rc, out = _run('002', tmp_path / 'not-written-yet.csv')
    assert rc == 0, out
    assert 'REFUSING' not in out


def test_a_field_the_digest_cannot_parse_stops_the_run(tmp_path):
    """The gate.  Fail-open on a permanent condition is a run that can never
    stop, so this must be a refusal, not a warning."""
    path = tmp_path / 'offsets.csv'
    _table().write(path, format='ascii.csv', overwrite=True)
    rc, out = _run('o002', path)
    assert rc == 2, out
    assert 'REFUSING: FIELD=o002 does not spell an observation' in out
    assert 'issue #272' in out
    # It stopped BEFORE the provenance-only exit, i.e. before any submission
    # path was reached at all.
    assert 'submitted' not in out


def test_a_field_shape_refusal_needs_no_interpreter(tmp_path):
    """Layer 1 is bash: the shape check must hold in an environment where the
    digest probe cannot run at all."""
    path = tmp_path / 'offsets.csv'
    _table().write(path, format='ascii.csv', overwrite=True)
    rc, out = _run('001-', path, PATH='/usr/bin:/bin')
    assert rc == 2, out
    assert 'REFUSING: FIELD=001- does not spell an observation' in out


def test_a_table_with_no_visit_column_stops_the_run(tmp_path):
    """Reachable today through the documented OFFSETS_TBL override: the three
    `Offsets_JWST_Brick2221_*_average.csv` tables have no `Visit` column, so no
    row of them can be attributed to an observation."""
    path = tmp_path / 'average.csv'
    Table({'Filter': ['F410M'], 'Module': ['nrcb'],
           'dra (arcsec)': [0.1], 'ddec (arcsec)': [-0.1]}).write(
        path, format='ascii.csv', overwrite=True)
    rc, out = _run('002', path)
    assert rc == 2, out
    assert 'REFUSING: the convergence digest cannot be scoped' in out
    assert 'Visit' in out


def test_the_preflight_runs_before_any_sbatch():
    """Wiring, not just behaviour: the refusal has to sit ahead of every
    submission in the file.  Comment lines are dropped first -- the script
    quotes `sbatch` in its prose long before it runs one."""
    code = [line for line in _LOOP.read_text().splitlines()
            if not line.lstrip().startswith('#')]
    guard = next(i for i, line in enumerate(code)
                 if 'REFUSING: the convergence digest cannot be scoped' in line)
    submits = [i for i, line in enumerate(code)
               if re.match(r'\s*[A-Za-z_]*=?\$?\(?(sbatch|bash) ', line)]
    assert len(submits) >= 3, 'the loop should still submit something'
    assert guard < min(submits)


def test_an_unrunnable_probe_is_reported_and_not_refused(tmp_path):
    """Layer 2 refuses ONLY on the digest's own "cannot digest" status (2).

    Every other python call in this script already falls back silently, and the
    loop's existing tests source it with a PATH that has no `python` on it.  A
    probe that did not run is not evidence that the scope is unusable, so it is
    reported.
    """
    path = tmp_path / 'offsets.csv'
    _table().write(path, format='ascii.csv', overwrite=True)
    rc, out = _run('001-002', path, PATH='/usr/bin:/bin')
    assert rc == 0, out
    assert 'could not run the digest to check the scope' in out
    assert 'REFUSING' not in out


def test_the_existing_source_only_tests_still_reach_the_helpers():
    """The loop's own test helpers source it with `PATH=/usr/bin:/bin:...` and
    no `python`; the preflight sits ahead of that return, so an `exit` there
    would take the whole test process with it."""
    proc = subprocess.run(
        ['bash', '-c',
         f'source "{_LOOP}" >/dev/null 2>&1; type table_value_digest'],
        capture_output=True, text=True,
        env={'RETIE_LOOP_SOURCE_ONLY': '1', 'PROPOSAL': '4147', 'FIELD': '012',
             'TARGET': 'sgrc', 'FILTERS': 'F115W',
             'OFFSETS_TBL': '/tmp/unused.csv',
             'PATH': '/usr/bin:/bin:/usr/local/bin', 'HOME': '/tmp'})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert 'table_value_digest is a function' in proc.stdout
