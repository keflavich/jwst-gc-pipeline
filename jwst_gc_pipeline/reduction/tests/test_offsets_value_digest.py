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


# --------------------------------------------------------------------------
# Review of PR #763.
#
# B1  A JOINT field spelling -- `alignment_config` registers sickle's MIRI as
#     ('001-002', '001', '002') and sgrb2's as '002-998' -- made every digest
#     call exit 2, and the loop's fail-open turned each into a fresh unique
#     token.  `[ "$tbl_after" = "$tbl_before" ]` could then never hold, so the
#     "no SHIFT VALUE changed -> STOPPING" branch was dead for the whole run.
#
# B3  The visit parse was looser than the one the m2 checkpoint WRITES rows
#     with, and read a malformed id as a confident wrong observation.
# --------------------------------------------------------------------------

def _joint_table():
    """Two observations jointly registered as one field, plus a third."""
    t = _table()
    t['Visit'] = ['jw03958001001', 'jw03958002001', 'jw03958007001']
    return t


def test_a_joint_field_spelling_can_be_scoped(mod, tmp_path):
    """THE B1 regression: `--observation 001-002` is the spelling the registry
    holds and the loop passes, and it must digest rather than raise."""
    path = _write(tmp_path, _joint_table())
    assert re.fullmatch(r'[0-9a-f]{64}', mod.digest(path, observation='001-002'))


def test_a_joint_field_covers_every_observation_it_names(mod, tmp_path):
    """001-002 is BOTH observations' rows, not a literal that matches none."""
    path = _write(tmp_path, _joint_table())
    joint = mod.digest(path, observation='001-002')
    assert joint != mod.digest(path, observation='001')
    assert joint != mod.digest(path, observation='002')
    # ... and it is not merely "everything": obs 007 is still out of scope.
    assert joint != mod.digest(path)


def test_a_joint_field_still_sees_a_retie_on_either_of_its_observations(mod, tmp_path):
    """Gate preservation: the scope must not hide the joint field's OWN
    movement, on either half."""
    before = mod.digest(_write(tmp_path, _joint_table()), observation='001-002')
    for row in (0, 1):
        t = _joint_table()
        t['dra (arcsec)'][row] += 0.005
        assert mod.digest(_write(tmp_path, t), observation='001-002') != before


def test_a_joint_field_still_hides_a_foreign_observations_move(mod, tmp_path):
    """And it must still narrow: obs 007 sharing the table is not this run."""
    before = mod.digest(_write(tmp_path, _joint_table()), observation='001-002')
    t = _joint_table()
    t['dra (arcsec)'][2] += 0.005          # jw03958007001
    assert mod.digest(_write(tmp_path, t), observation='001-002') == before


def test_the_joint_spelling_agrees_with_naming_observation_field_token(mod):
    """One decomposition, two places.  The digest spells `str(f).split('-')`
    itself rather than importing the package (an ImportError here would exit 2
    on every call, which is exactly the B1 failure); this pins the two
    spellings together so they cannot drift apart."""
    naming = pytest.importorskip('jwst_gc_pipeline.photometry.naming')
    for field in ('1', '01', '001', '002-998', '001-002', '12', '1-2-3'):
        assert ('-'.join(mod.normalise_observation(field))
                == naming.observation_field_token(field)), field


def test_an_observation_wider_than_three_digits_still_raises(mod, tmp_path):
    """`observation_field_token` only pads, so it accepts '1234'.  Here that
    would scope to zero attributable rows and report every iteration as 'no
    re-tie' -- silent, and on the STOPPING side."""
    path = _write(tmp_path, _table())
    with pytest.raises(ValueError, match='1234'):
        mod.digest(path, observation='1234')


def test_a_trailing_or_leading_hyphen_still_raises(mod, tmp_path):
    path = _write(tmp_path, _table())
    for bad in ('001-', '-002', '001--002', 'o001-002'):
        with pytest.raises(ValueError, match='observation'):
            mod.digest(path, observation=bad)


def test_a_malformed_visit_is_unparseable_rather_than_mis_attributed(mod):
    """B3.  Unanchored, `^jw\\d{5}(\\d{3})` read a visit id one digit short as
    observation `230` and one digit long as `102` -- neither of which is any
    real observation, so the row fell out of EVERY scope and a re-tie on it was
    invisible.  Unparseable means KEPT, which cannot hide movement."""
    assert mod._observation_of('jw02211023001') == '023'      # well formed
    assert mod._observation_of('jw2211023001') is None        # one digit short
    assert mod._observation_of('jw002211023001') is None      # one digit long
    assert mod._observation_of('jw02211023001_02101') is None  # a stem, not an id
    assert mod._observation_of('hand-edited') is None


def test_a_malformed_visit_row_still_counts_as_this_observations_movement(mod, tmp_path):
    """The consequence of the above, at the digest: a shift on a row whose
    Visit does not parse still reads as a re-tie for every observation."""
    t = _table()
    t['Visit'] = ['jw02092002001', 'jw02092002001', 'jw2211023001']
    before = mod.digest(_write(tmp_path, t), observation='002')
    t['dra (arcsec)'][2] += 0.005
    assert mod.digest(_write(tmp_path, t), observation='002') != before


def test_the_visit_parse_agrees_with_the_m2_checkpoints_own(mod):
    """The rows are WRITTEN by `astrometry_checkpoint`, keyed on
    `visit_obs_key`.  Two parsers of one column that disagree put a row in one
    observation for the writer and another for this reader."""
    ckpt = pytest.importorskip('jwst_gc_pipeline.photometry.astrometry_checkpoint')
    for visit in ('jw02211023001', 'jw02211050001', 'jw10678088001',
                  'jw10678008001', 'jw02092002001', 'jw03958001001',
                  'jw2211023001', 'jw002211023001', 'jw02211023001 '):
        assert mod._observation_of(visit) == ckpt.visit_obs_key(visit)[0], visit


def test_every_registered_field_spelling_can_be_scoped(mod):
    """The loop passes `$FIELD` straight through, and `alignment_config` is
    where a field spelling is declared.  A registration the digest cannot scope
    disables the loop's stop condition for that field."""
    cfg = pytest.importorskip('jwst_gc_pipeline.reduction.alignment_config')
    seen = 0
    for entry in cfg.ALIGNMENT_CONFIG:
        for field in (entry.fields or ()):
            assert mod.normalise_observation(field), field
            seen += 1
    assert seen > 10, 'the registry should have supplied real field spellings'


# --------------------------------------------------------------------------
# Review item 4: the false positive that survives the window it matters in.
#
# The digest returned the literal "none" for an absent file and a hex for an
# empty scope, so on the treasury's first night a tile that wrote NOTHING read
# a re-tie as soon as a NEIGHBOUR's m2 created the shared table.  139 tiles
# share a table that does not exist yet on Sep 10-13, so that is the first
# iteration of most of them.
# --------------------------------------------------------------------------

def test_a_neighbour_creating_the_shared_table_is_not_this_tiles_retie(mod, tmp_path):
    """THE regression: tile 088 wrote nothing; tile 089's m2 created the file."""
    missing = str(tmp_path / 'not-written-yet.csv')
    before = mod.digest(missing, observation='088')
    t = _table()
    t['Visit'] = ['jw10678089001', 'jw10678089001', 'jw10678089002']
    assert mod.digest(_write(tmp_path, t), observation='088') == before


def test_seeding_this_observations_own_rows_is_still_a_retie(mod, tmp_path):
    """The gate this must NOT weaken: the FIRST m2 seeds rows rather than
    moving them, and that is the re-tie the loop has to see."""
    missing = str(tmp_path / 'not-written-yet.csv')
    before = mod.digest(missing, observation='088')
    t = _table()
    t['Visit'] = ['jw10678089001', 'jw10678088001', 'jw10678089002']
    assert mod.digest(_write(tmp_path, t), observation='088') != before


def test_losing_every_row_of_this_observation_is_still_movement(mod, tmp_path):
    """And the reverse: a table this observation's rows vanish from moved."""
    t = _table()
    t['Visit'] = ['jw10678089001', 'jw10678088001', 'jw10678089002']
    populated = mod.digest(_write(tmp_path, t), observation='088')
    t['Visit'] = ['jw10678089001', 'jw10678089001', 'jw10678089002']
    assert mod.digest(_write(tmp_path, t), observation='088') != populated


def test_an_absent_table_is_still_none_when_unscoped(mod, tmp_path):
    """Every single-observation field digested `none` for an absent table
    before `--observation` existed, and the unscoped answer is unchanged."""
    assert mod.digest(str(tmp_path / 'nope.csv')) == "none"


def test_the_empty_scope_value_does_not_depend_on_the_tables_columns(mod, tmp_path):
    """It has to be reachable from an ABSENT table, where the value columns a
    future table will carry are unknowable."""
    t = _table()
    t['Visit'] = ['jw10678089001'] * 3
    with_all = mod.digest(_write(tmp_path, t), observation='088')
    t.remove_columns(['dra', 'ddec'])
    fewer = mod.digest(_write(tmp_path, t, name='fewer.csv'), observation='088')
    assert with_all == fewer == mod.digest(str(tmp_path / 'nope.csv'),
                                           observation='088')


def test_two_observations_with_no_rows_do_not_share_a_digest(mod, tmp_path):
    """The empty value is still per-observation, so a digest taken for one tile
    is not comparable with another's."""
    missing = str(tmp_path / 'nope.csv')
    assert (mod.digest(missing, observation='088')
            != mod.digest(missing, observation='089'))
