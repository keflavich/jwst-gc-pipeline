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
