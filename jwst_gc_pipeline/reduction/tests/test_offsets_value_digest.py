"""The re-tie loop's "did this iteration re-tie anything?" test.

`run_field_retie_loop.sh` md5-summed the whole offsets table.  The m2 checkpoint
re-stamps `prov_date` on rows it did not move, so a round that wrote no
correction still changed the file, the loop read that as a re-tie, and it
re-reduced and re-measured an identical residual -- issue #272, three
consecutive rounds reporting 15 corrections against 0 changed `dra`/`ddec`
cells.  These pin that a provenance re-stamp digests the SAME and a shift
change digests DIFFERENT.
"""
import importlib.util
import pathlib
import re

import pytest

astropy_table = pytest.importorskip("astropy.table")
Table = astropy_table.Table

_SRC = (pathlib.Path(__file__).parents[3] / 'scripts' / 'reduction'
        / 'offsets_value_digest.py')
_LOOP = (pathlib.Path(__file__).parents[3] / 'scripts' / 'reduction'
         / 'run_field_retie_loop.sh')


def _load():
    spec = importlib.util.spec_from_file_location('_offsets_digest', _SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def mod():
    return _load()


def _table():
    return Table({
        'Visit': ['jw02092002001', 'jw02092002001', 'jw02092005001'],
        'Exposure': [7, 8, 7],
        'Filter': ['F360M', 'F360M', 'F360M'],
        'Module': ['nrcblong', 'nrcblong', 'nrcblong'],
        'Vgroup': ['2101', '2101', '2101'],
        'dra': [1e-4, 2e-4, 3e-4],
        'ddec': [-1e-4, -2e-4, -3e-4],
        'dra (arcsec)': [0.360, 0.720, 1.080],
        'ddec (arcsec)': [-0.360, -0.720, -1.080],
        'prov_stage': ['m2', 'm2', ''],
        'prov_date': ['2026-08-04T08:37:54Z', '2026-08-04T08:37:54Z', ''],
        'prov_dra_added_mas': [-1.22, -1.22, 0.0],
    })


def _write(tmp_path, table, name='offsets.csv'):
    path = tmp_path / name
    table.write(path, format='ascii.csv', overwrite=True)
    return str(path)


def test_provenance_restamp_does_not_change_the_digest(mod, tmp_path):
    """THE regression: the write that made the loop spin."""
    before = mod.digest(_write(tmp_path, _table()))
    t = _table()
    t['prov_date'] = ['2026-08-23T11:00:00Z', '2026-08-23T11:00:00Z', '']
    t['prov_stage'] = ['m2', 'm2', 'm2']
    after = mod.digest(_write(tmp_path, t))
    assert after == before


def test_a_changed_shift_changes_the_digest(mod, tmp_path):
    before = mod.digest(_write(tmp_path, _table()))
    t = _table()
    t['dra (arcsec)'][0] += 0.005          # ~5 mas, a real correction
    assert mod.digest(_write(tmp_path, t)) != before


def test_a_changed_plain_pair_changes_the_digest(mod, tmp_path):
    """Both column pairs are digested: `_heal_column_pairs` writes the plain
    pair, and a loop that watched only `(arcsec)` would miss it."""
    before = mod.digest(_write(tmp_path, _table()))
    t = _table()
    t['dra'][2] += 1e-5
    assert mod.digest(_write(tmp_path, t)) != before


def test_row_order_does_not_change_the_digest(mod, tmp_path):
    before = mod.digest(_write(tmp_path, _table()))
    t = _table()[[2, 0, 1]]
    assert mod.digest(_write(tmp_path, t)) == before


def test_added_and_removed_rows_change_the_digest(mod, tmp_path):
    before = mod.digest(_write(tmp_path, _table()))
    assert mod.digest(_write(tmp_path, _table()[:2])) != before


def test_float_reserialisation_below_the_rounding_does_not_count(mod, tmp_path):
    """A rewrite that moves the last float digit of an unchanged quantity is not
    a re-tie; reading it as one is the same false positive the md5sum had."""
    before = mod.digest(_write(tmp_path, _table()))
    t = _table()
    t['dra (arcsec)'][0] += 1e-12
    assert mod.digest(_write(tmp_path, t)) == before


def test_two_rows_that_differ_only_by_vgroup_are_distinguished(mod, tmp_path):
    """cloudc's table has two vgroups per exposure, so the four-column key is
    ambiguous for half its rows (issue #272).  Moving one of them must show."""
    t = _table()
    t['Visit'] = ['jw02092002001'] * 3
    t['Exposure'] = [7, 7, 7]
    t['Vgroup'] = ['02201', '08201', '00101']
    before = mod.digest(_write(tmp_path, t))
    t['dra (arcsec)'][1] += 0.005
    assert mod.digest(_write(tmp_path, t)) != before


def test_missing_table_digests_to_none(mod, tmp_path):
    assert mod.digest(str(tmp_path / 'nope.csv')) == "none"
    assert mod.digest("") == "none"


def test_a_table_with_no_value_columns_raises(mod, tmp_path):
    """Fail loudly rather than return one constant for every table, which would
    report every iteration as 'no re-tie' and stop the loop on iteration 1."""
    path = _write(tmp_path, Table({'Visit': ['a'], 'prov_date': ['x']}))
    with pytest.raises(ValueError, match="none of"):
        mod.digest(path)


def test_cli_prints_the_digest_and_exits_zero(mod, tmp_path, capsys):
    path = _write(tmp_path, _table())
    assert mod.main([path]) == 0
    assert capsys.readouterr().out.strip() == mod.digest(path)


def test_cli_exits_2_on_an_undigestible_table(mod, tmp_path, capsys):
    path = _write(tmp_path, Table({'Visit': ['a']}))
    assert mod.main([path]) == 2
    assert "cannot digest" in capsys.readouterr().err


def test_the_loop_uses_the_digest_and_not_an_md5_of_the_table():
    """The wiring, not just the helper: an md5sum of `$CONSENSUS_TBL` anywhere
    is the defect coming back."""
    text = _LOOP.read_text()
    assert 'offsets_value_digest.py' in text
    assert re.search(r'tbl_before=\$\(table_value_digest', text)
    assert re.search(r'tbl_after=\$\(table_value_digest', text)
    assert 'md5sum "$CONSENSUS_TBL"' not in text
