"""Quarantine of superseded pre-obs-token checkpoint records (#827).

The tool moves an untokened ``checkpoint_*`` record into
``pre_obstoken_stale/`` only where ``current_records`` supersedes it, never
deletes, and records each move in a manifest so it can be reversed.
"""
import glob
import importlib.util
import json
import os

SCRIPTS = os.path.join(os.path.dirname(__file__), '..', 'scripts', 'reduction')


def _load():
    spec = importlib.util.spec_from_file_location(
        'quarantine_pre_obstoken_checkpoints',
        os.path.join(SCRIPTS, 'quarantine_pre_obstoken_checkpoints.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


UNT = 'checkpoint_m2_F444W_latest.json'
UNT_HIST = 'checkpoint_m2_F444W_20260801T120000Z.json'
TOK = 'checkpoint_m2_F444W_o001_latest.json'
TOK_HIST = 'checkpoint_m2_F444W_o001_20260816T143240Z.json'
ORPHAN = 'checkpoint_m2_F200W_latest.json'
MODLESS = 'checkpoint_m7_crossfilter_latest.json'
MODULE_TOK = 'checkpoint_m7_crossfilter_nrcb_o001_latest.json'


def _field(tmp_path, *names, field='w51'):
    d = tmp_path / field / 'astrometry_checkpoints'
    d.mkdir(parents=True)
    for n in names:
        (d / n).write_text('{}')
    return d


def _listing(d):
    return sorted(os.listdir(d))


def test_dry_run_moves_nothing(tmp_path, capsys):
    m = _load()
    d = _field(tmp_path, UNT, UNT_HIST, TOK, TOK_HIST)
    before = _listing(d)
    rc = m.main(['--base', str(tmp_path), '--field', 'w51'])
    assert rc == 0
    assert _listing(d) == before
    out = capsys.readouterr().out
    assert '2 to quarantine' in out and 'dry run' in out


def test_execute_moves_only_superseded_files(tmp_path):
    m = _load()
    d = _field(tmp_path, UNT, UNT_HIST, TOK, TOK_HIST, ORPHAN, MODLESS, MODULE_TOK)
    m.main(['--base', str(tmp_path), '--field', 'w51', '--execute'])
    assert _listing(d) == sorted([TOK, TOK_HIST, ORPHAN, MODLESS, MODULE_TOK,
                                  'pre_obstoken_stale'])
    assert _listing(d / 'pre_obstoken_stale') == sorted(
        [UNT, UNT_HIST, 'MANIFEST.json'])


def test_files_with_no_tokened_sibling_are_untouched(tmp_path):
    m = _load()
    d = _field(tmp_path, ORPHAN, TOK)
    moves, _counts = m.plan_for_dir(str(d))
    assert moves == []


def test_module_less_m7_with_only_module_siblings_is_untouched(tmp_path):
    m = _load()
    d = _field(tmp_path, MODLESS, MODULE_TOK)
    moves, _counts = m.plan_for_dir(str(d))
    assert moves == []


def test_module_less_m7_with_a_module_less_tokened_sibling_moves(tmp_path):
    m = _load()
    d = _field(tmp_path, MODLESS, 'checkpoint_m7_crossfilter_o001_latest.json')
    moves, _counts = m.plan_for_dir(str(d))
    assert [mv['name'] for mv in moves] == [MODLESS]


def test_manifest_lists_name_mtime_and_superseding_record(tmp_path):
    m = _load()
    d = _field(tmp_path, UNT, TOK)
    os.utime(d / UNT, (1234567.0, 1234567.0))
    m.main(['--base', str(tmp_path), '--field', 'w51', '--execute'])
    manifest = json.loads((d / 'pre_obstoken_stale' / 'MANIFEST.json').read_text())
    [ent] = manifest['files']
    assert ent['name'] == UNT
    assert ent['mtime'] == 1234567.0
    assert ent['superseded_by'] == [TOK]


def test_running_twice_is_a_no_op(tmp_path):
    m = _load()
    d = _field(tmp_path, UNT, UNT_HIST, TOK)
    m.main(['--base', str(tmp_path), '--field', 'w51', '--execute'])
    first = (d / 'pre_obstoken_stale' / 'MANIFEST.json').read_text()
    after = _listing(d)
    m.main(['--base', str(tmp_path), '--field', 'w51', '--execute'])
    assert _listing(d) == after
    assert (d / 'pre_obstoken_stale' / 'MANIFEST.json').read_text() == first


def test_moved_files_no_longer_match_the_latest_glob(tmp_path):
    m = _load()
    d = _field(tmp_path, UNT, TOK)
    m.main(['--base', str(tmp_path), '--field', 'w51', '--execute'])
    names = [os.path.basename(p)
             for p in glob.glob(str(d / 'checkpoint_*_latest.json'))]
    assert names == [TOK]


def test_restore_reverses_the_move(tmp_path):
    m = _load()
    d = _field(tmp_path, UNT, UNT_HIST, TOK)
    before = _listing(d)
    m.main(['--base', str(tmp_path), '--field', 'w51', '--execute'])
    m.main(['--base', str(tmp_path), '--field', 'w51', '--restore', '--execute'])
    assert sorted(n for n in _listing(d) if n != 'pre_obstoken_stale') == before


def test_a_collision_in_the_stale_directory_is_not_overwritten(tmp_path):
    m = _load()
    d = _field(tmp_path, UNT, TOK)
    stale = d / 'pre_obstoken_stale'
    stale.mkdir()
    (stale / UNT).write_text('earlier')
    m.main(['--base', str(tmp_path), '--field', 'w51', '--execute'])
    assert (stale / UNT).read_text() == 'earlier'
    assert (d / UNT).exists()


def test_each_base_directory_is_processed_on_its_own(tmp_path):
    m = _load()
    blue, orange = tmp_path / 'blue', tmp_path / 'orange'
    db = _field(blue, UNT, TOK, field='brick')
    do = _field(orange, UNT, field='brick')
    m.main(['--base', str(blue), '--base', str(orange), '--field', 'brick',
            '--execute'])
    assert UNT not in _listing(db)
    # The orange copy has no tokened sibling in ITS directory: untouched.
    assert _listing(do) == [UNT]


def test_all_walks_every_field_and_missing_base_is_reported(tmp_path, capsys):
    m = _load()
    _field(tmp_path, UNT, TOK, field='w51')
    _field(tmp_path, UNT, TOK, field='sickle')
    m.main(['--base', str(tmp_path), '--all'])
    out = capsys.readouterr().out
    assert 'w51' in out and 'sickle' in out
    assert m.main(['--base', str(tmp_path / 'absent'), '--all']) == 1


# --- review follow-ups on PR #1303 -------------------------------------------

def test_a_dangling_symlink_in_place_of_a_record_does_not_crash(tmp_path, capsys):
    """The plan read the mtime through the link: FileNotFoundError, dry run too."""
    m = _load()
    d = _field(tmp_path, TOK)
    os.symlink(str(tmp_path / 'nonexistent' / 'target.json'), str(d / UNT))
    assert m.main(['--base', str(tmp_path), '--field', 'w51']) == 0
    assert '1 to quarantine' in capsys.readouterr().out
    assert os.path.islink(d / UNT)
    assert m.main(['--base', str(tmp_path), '--field', 'w51', '--execute']) == 0
    stale = d / 'pre_obstoken_stale'
    assert os.path.islink(stale / UNT) and not os.path.lexists(d / UNT)
    m.main(['--base', str(tmp_path), '--field', 'w51', '--restore', '--execute'])
    assert os.path.islink(d / UNT)


def test_a_shared_filter_record_is_left_alone(tmp_path, capsys):
    """ngc6334 images F200W in two proposals: an untokened record there may be
    the only record of the proposal that has no tokened one, so it stays."""
    m = _load()
    names = ('checkpoint_m2_F200W_latest.json',
             'checkpoint_m2_F200W_j7213_latest.json')
    d = _field(tmp_path, *names, field='ngc6334')
    moves, counts = m.plan_for_dir(str(d))
    assert moves == []
    assert counts['ambiguous'] == [names[0]]
    m.main(['--base', str(tmp_path), '--field', 'ngc6334', '--execute'])
    assert _listing(d) == sorted(names)
    assert 'left alone (ambiguous' in capsys.readouterr().out


def test_a_filter_less_record_on_a_shared_filter_field_is_left_alone(tmp_path):
    m = _load()
    d = _field(tmp_path, MODLESS, 'checkpoint_m7_crossfilter_o001_latest.json',
               field='ngc6334')
    moves, counts = m.plan_for_dir(str(d))
    assert moves == [] and counts['ambiguous'] == [MODLESS]


def test_a_field_unknown_to_the_registry_is_ambiguous(tmp_path):
    m = _load()
    d = _field(tmp_path, MODLESS, 'checkpoint_m7_crossfilter_o001_latest.json',
               field='nosuchfield')
    moves, counts = m.plan_for_dir(str(d))
    assert moves == [] and counts['ambiguous'] == [MODLESS]


def test_restore_dry_run_leaves_the_stale_directory_alone(tmp_path):
    m = _load()
    d = _field(tmp_path, UNT, TOK)
    m.main(['--base', str(tmp_path), '--field', 'w51', '--execute'])
    stale = d / 'pre_obstoken_stale'
    before_stale, before_dir = _listing(stale), _listing(d)
    manifest = (stale / 'MANIFEST.json').read_text()
    assert m.main(['--base', str(tmp_path), '--field', 'w51', '--restore']) == 0
    assert _listing(stale) == before_stale and _listing(d) == before_dir
    assert (stale / 'MANIFEST.json').read_text() == manifest


def test_restore_never_overwrites_a_newer_record_under_the_original_name(
        tmp_path, capsys):
    m = _load()
    d = _field(tmp_path, UNT, TOK)
    m.main(['--base', str(tmp_path), '--field', 'w51', '--execute'])
    (d / UNT).write_text('newer')
    rc = m.main(['--base', str(tmp_path), '--field', 'w51', '--restore',
                 '--execute'])
    assert (d / UNT).read_text() == 'newer'
    assert (d / 'pre_obstoken_stale' / UNT).read_text() == '{}'
    assert rc == 2


def test_a_skipped_collision_exits_nonzero(tmp_path, capsys):
    m = _load()
    d = _field(tmp_path, UNT, TOK)
    stale = d / 'pre_obstoken_stale'
    stale.mkdir()
    (stale / UNT).write_text('earlier')
    rc = m.main(['--base', str(tmp_path), '--field', 'w51', '--execute'])
    assert rc == 2
    assert 'SKIPPED' in capsys.readouterr().out


def test_a_clean_run_exits_zero(tmp_path):
    m = _load()
    _field(tmp_path, UNT, TOK)
    assert m.main(['--base', str(tmp_path), '--field', 'w51', '--execute']) == 0


def test_a_mistyped_field_is_reported_even_when_another_matches(tmp_path, capsys):
    m = _load()
    _field(tmp_path, UNT, TOK)
    m.main(['--base', str(tmp_path), '--field', 'nosuch', '--field', 'w51'])
    err = capsys.readouterr().err
    assert '--field nosuch' in err and 'w51' not in err


def test_a_field_present_under_one_base_only_is_not_reported_missing(tmp_path,
                                                                    capsys):
    m = _load()
    _field(tmp_path / 'blue', UNT, TOK, field='brick')
    (tmp_path / 'orange').mkdir()
    m.main(['--base', str(tmp_path / 'blue'), '--base', str(tmp_path / 'orange'),
            '--field', 'brick'])
    assert capsys.readouterr().err == ''
