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


# --------------------------------------------------------------------------
# One table, more than one observation (issue #714).  10678 registers
# fields=None, so all 139 treasury tiles share one consensus table and an
# unscoped digest reads a neighbouring tile's correction as this tile's re-tie.
# `_table()` above already holds observations 002 and 005.
# --------------------------------------------------------------------------

def test_a_sibling_observations_move_is_not_this_observations_retie(mod, tmp_path):
    """THE regression: obs 005 re-ties, obs 002's loop must see no movement."""
    before = mod.digest(_write(tmp_path, _table()), observation='002')
    t = _table()
    t['dra (arcsec)'][2] += 0.005          # row 2 is jw02092005001
    t['ddec (arcsec)'][2] -= 0.005
    assert mod.digest(_write(tmp_path, t), observation='002') == before


def test_scoping_still_sees_this_observations_own_retie(mod, tmp_path):
    """The gate the scope must NOT weaken: a correction on obs 002's own rows
    still reads as movement, so the loop still re-reduces and re-measures."""
    before = mod.digest(_write(tmp_path, _table()), observation='002')
    t = _table()
    t['dra (arcsec)'][0] += 0.005          # row 0 is jw02092002001
    assert mod.digest(_write(tmp_path, t), observation='002') != before


def test_scoping_still_sees_a_row_added_or_removed(mod, tmp_path):
    """The first m2 seeds rows rather than moving them; a seeded row for this
    observation is a re-tie."""
    before = mod.digest(_write(tmp_path, _table()), observation='002')
    t = _table()[[0, 2]]                   # obs 002 loses exposure 8
    assert mod.digest(_write(tmp_path, t), observation='002') != before


def test_scoping_does_not_hide_a_provenance_restamp_as_movement(mod, tmp_path):
    """#272's false positive stays fixed under the scope."""
    before = mod.digest(_write(tmp_path, _table()), observation='002')
    t = _table()
    t['prov_date'] = ['2026-09-05T11:00:00Z', '2026-09-05T11:00:00Z', '']
    assert mod.digest(_write(tmp_path, t), observation='002') == before


def test_the_unscoped_digest_is_unchanged_by_the_option(mod, tmp_path):
    """Every single-observation field passes its own FIELD, and the eight fields
    running today must digest exactly as they did: no scope, every row."""
    path = _write(tmp_path, _table())
    assert mod.digest(path, observation=None) == mod.digest(path)
    before = mod.digest(path)
    t = _table()
    t['dra (arcsec)'][2] += 0.005          # obs 005, which the scope hides
    assert mod.digest(_write(tmp_path, t)) != before


def test_an_unattributable_visit_is_kept_in_a_scoped_digest(mod, tmp_path):
    """Scoping may hide a change that BELONGS to another observation, never one
    that cannot be placed: a row whose Visit does not parse still counts."""
    t = _table()
    t['Visit'] = ['jw02092002001', 'jw02092002001', 'hand-edited']
    before = mod.digest(_write(tmp_path, t), observation='002')
    t['dra (arcsec)'][2] += 0.005
    assert mod.digest(_write(tmp_path, t), observation='002') != before


def test_an_observation_number_is_zero_padded(mod, tmp_path):
    """FIELD=2 and FIELD=002 name the same observation; the table writes 002."""
    path = _write(tmp_path, _table())
    assert mod.digest(path, observation='2') == mod.digest(path, observation='002')


def test_a_prefix_of_an_observation_number_does_not_match(mod, tmp_path):
    """Observations 002 and 020 are different tiles; matching on a prefix would
    pool them."""
    t = _table()
    t['Visit'] = ['jw10678088001', 'jw10678088001', 'jw10678008001']
    before = mod.digest(_write(tmp_path, t), observation='008')
    t['dra (arcsec)'][0] += 0.005          # observation 088, not 008
    assert mod.digest(_write(tmp_path, t), observation='008') == before


def test_an_observation_with_no_rows_yet_digests_stably_and_is_not_none(mod, tmp_path):
    """A treasury tile's first iteration digests a table holding only its
    neighbours' rows; that must be a stable value distinguishable from an
    absent table, so seeding this tile's rows reads as the re-tie."""
    path = _write(tmp_path, _table())
    empty = mod.digest(path, observation='099')
    assert empty == mod.digest(path, observation='099')
    assert empty != "none"
    t = _table()
    t['Visit'] = ['jw02092002001', 'jw02092002001', 'jw02092099001']
    assert mod.digest(_write(tmp_path, t), observation='099') != empty


def test_scoping_a_table_with_no_visit_column_raises(mod, tmp_path):
    """Refuse rather than digest every row as if it were this observation's --
    that is the unscoped behaviour wearing the scope's name."""
    path = _write(tmp_path, Table({
        'Exposure': [7], 'Filter': ['F212N'], 'Module': ['nrcb1'],
        'Vgroup': ['2101'], 'dra (arcsec)': [0.1], 'ddec (arcsec)': [0.1]}))
    assert mod.digest(path) is not None            # unscoped is fine
    with pytest.raises(ValueError, match="Visit"):
        mod.digest(path, observation='002')


def test_a_malformed_observation_number_raises(mod, tmp_path):
    path = _write(tmp_path, _table())
    with pytest.raises(ValueError, match="observation"):
        mod.digest(path, observation='o002')


def test_cli_takes_the_observation(mod, tmp_path, capsys):
    path = _write(tmp_path, _table())
    assert mod.main([path, '--observation', '002']) == 0
    assert capsys.readouterr().out.strip() == mod.digest(path, observation='002')


def test_cli_exits_2_on_an_unscopeable_table(mod, tmp_path, capsys):
    """The loop reads a nonzero rc as 'the table changed' and keeps going, with
    the reason on stderr -- it must never read as 'unchanged'."""
    path = _write(tmp_path, Table({
        'Exposure': [7], 'dra (arcsec)': [0.1], 'ddec (arcsec)': [0.1]}))
    assert mod.main([path, '--observation', '002']) == 2
    assert "cannot digest" in capsys.readouterr().err


def test_the_loop_scopes_the_digest_to_its_own_observation():
    """The wiring: an unscoped digest of a shared table is the defect."""
    text = _LOOP.read_text()
    assert re.search(r'--observation "\$FIELD"', text)


def test_a_malformed_observation_raises_even_on_an_absent_table(mod, tmp_path):
    """Reporting it as `none` would run the loop on with the scope not applied."""
    with pytest.raises(ValueError, match="observation"):
        mod.digest(str(tmp_path / 'nope.csv'), observation='')
