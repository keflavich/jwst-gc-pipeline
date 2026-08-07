"""The revert must move the table AND leave the audit trail readable.

Driven through the script's own `revert()`, on a real CSV, rather than against
the flag helper -- the flag helper is tested in
`jwst_gc_pipeline/reduction/tests/test_broadcast_provenance.py`, and a repair
tool that is only tested through its detector can move the wrong column and
still pass.
"""
import importlib.util
import os

import numpy as np
import pytest
from astropy.table import Table

SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'reduction')


def _load():
    spec = importlib.util.spec_from_file_location(
        'revert_broadcast_provenance',
        os.path.join(SCRIPTS, 'revert_broadcast_provenance.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


#: The live gc2211 numbers: five visits, one shared correction, and as-built
#: values that reproduce an independent measurement of each region.
PROV = (-7031.7, 15009.7)
AS_BUILT = {
    'jw02211023001': (3.1262, -1.8234),
    'jw02211028001': (-10.2894, 20.5478),
    'jw02211046001': (0.0279, -0.1480),
    'jw02211049001': (0.0011, -0.0251),
    'jw02211050001': (-2.5338, 5.1329),
}


def _table(tmp_path, prov=PROV, name='Offsets_JWST_Brick2211_VIRAC2locked.csv'):
    rows = []
    for v, (dra, ddec) in AS_BUILT.items():
        for exp in (1, 2):
            rows.append({
                'Visit': v, 'Filter': 'F277W', 'Exposure': exp,
                'Module': 'nrcalong', 'dra': dra, 'ddec': ddec,
                'dra (arcsec)': dra + prov[0] / 1000.0,
                'ddec (arcsec)': ddec + prov[1] / 1000.0,
                'prov_stage': 'm2', 'prov_date': '2026-08-01T00:00:00Z',
                'prov_source': 'astrometry_checkpoint',
                'prov_dra_added_mas': prov[0], 'prov_ddec_added_mas': prov[1]})
    p = str(tmp_path / name)
    Table(rows).write(p, format='ascii.csv', overwrite=True)
    return p


def test_the_applied_pair_takes_the_AS_BUILT_value(tmp_path):
    """The whole point, and the direction that matters: the products were
    drizzled from `(arcsec)`, and `(arcsec)` is what has to change."""
    m = _load()
    p = _table(tmp_path)
    assert m.revert(p, apply=True) == 10
    t = Table.read(p, format='ascii.csv')
    for row in t:
        dra, ddec = AS_BUILT[str(row['Visit'])]
        assert row['dra (arcsec)'] == pytest.approx(dra, abs=1e-9)
        assert row['ddec (arcsec)'] == pytest.approx(ddec, abs=1e-9)
        # and the as-built pair itself is untouched -- it is the good copy
        assert row['dra'] == pytest.approx(dra, abs=1e-9)


def test_the_discarded_provenance_is_zeroed_and_NAMED(tmp_path):
    """Clearing `prov_*` without saying why would leave the table looking as
    though it had never been corrected."""
    m = _load()
    p = _table(tmp_path)
    m.revert(p, apply=True)
    t = Table.read(p, format='ascii.csv')
    assert np.allclose(np.asarray(t['prov_dra_added_mas'], float), 0.0)
    assert np.allclose(np.asarray(t['prov_ddec_added_mas'], float), 0.0)
    assert set(str(s) for s in t['prov_stage']) == {'revert'}
    assert all('revert_broadcast_provenance' in str(s) for s in t['prov_source'])


def test_a_dry_run_writes_nothing(tmp_path):
    m = _load()
    p = _table(tmp_path)
    before = open(p).read()
    assert m.revert(p, apply=False) == 10
    assert open(p).read() == before


def test_a_table_with_REAL_per_visit_corrections_is_refused(tmp_path):
    """The guard on the tool: it can only touch a table the detector flags, so
    it cannot be pointed at corrections that were genuinely measured."""
    m = _load()
    p = str(tmp_path / 'Offsets_JWST_Brick2211_VIRAC2locked.csv')
    rows = []
    for i, (v, (dra, ddec)) in enumerate(AS_BUILT.items()):
        prov = (-7000.0 + 500 * i, 15000.0 - 400 * i)      # all different
        rows.append({
            'Visit': v, 'Filter': 'F277W', 'Exposure': 1, 'Module': 'nrcalong',
            'dra': dra, 'ddec': ddec,
            'dra (arcsec)': dra + prov[0] / 1000.0,
            'ddec (arcsec)': ddec + prov[1] / 1000.0,
            'prov_stage': 'm2', 'prov_date': 'x', 'prov_source': 'y',
            'prov_dra_added_mas': prov[0], 'prov_ddec_added_mas': prov[1]})
    Table(rows).write(p, format='ascii.csv', overwrite=True)
    before = open(p).read()
    assert m.revert(p, apply=True) == 0
    assert open(p).read() == before


def test_a_backup_is_left_beside_the_table(tmp_path):
    m = _load()
    p = _table(tmp_path)
    original = open(p).read()
    m.revert(p, apply=True)
    backups = [f for f in os.listdir(tmp_path) if '.pre_provrevert_' in f]
    assert len(backups) == 1, backups
    assert open(os.path.join(tmp_path, backups[0])).read() == original


def test_running_it_twice_is_a_no_op(tmp_path):
    """After the revert the table is clean, so a second pass must not fire --
    otherwise a repeated run would keep rewriting a healthy table."""
    m = _load()
    p = _table(tmp_path)
    m.revert(p, apply=True)
    after = open(p).read()
    assert m.revert(p, apply=True) == 0
    assert open(p).read() == after


def test_a_single_pair_table_is_left_alone(tmp_path):
    m = _load()
    p = str(tmp_path / 'Offsets_JWST_Brick2211_consensus.csv')
    Table({'Visit': ['a', 'b'], 'Filter': ['F277W'] * 2, 'Exposure': [1, 1],
           'dra': [1.0, 2.0], 'ddec': [3.0, 4.0]}).write(
        p, format='ascii.csv', overwrite=True)
    assert m.revert(p, apply=True) == 0
