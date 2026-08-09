"""Re-keying an offsets table must not change a single applied offset.

gc2211's table is keyed (Visit, Exposure, Filter) only, so the m2 checkpoint
cannot write a correction back to it -- a correction is per (visit, filter,
exposure, MODULE, VGROUP)::

    OffsetsTableUpdateError: 32 corrections spanning module families
    ['nrca', 'nrcb'] land on the same row(s) (19,)
    OffsetsTableUpdateError: module(s) ['nrcb1'..'nrcb4'] contribute MORE THAN
    ONE correction to the same row(s) (10,)

Both refusals are right: pooling module A against module B, or two visit groups
onto one row, averages away a real difference.

The migration replicates each row per (module, vgroup) present on disk, copying
every measured column verbatim.  The property that makes it safe is that the
value a frame RECEIVES is unchanged -- checked here against
`unified_alignment.locked_row_match`, the reader's own narrowing, rather than a
second copy of those rules.
"""
import importlib.util
import os

import pytest
from astropy.table import Table

from jwst_gc_pipeline.reduction.unified_alignment import locked_row_match

SCRIPTS = os.path.join(os.path.dirname(__file__), '..', '..',
                       'scripts', 'reduction')


def _load():
    spec = importlib.util.spec_from_file_location(
        'expand_offsets_granularity',
        os.path.join(SCRIPTS, 'expand_offsets_granularity.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _table():
    """Two gc2211 rows, with the field's real recorded o023 F200W offset."""
    return Table({
        'Visit': ['jw02211023001', 'jw02211023001'],
        'Exposure': [1, 2],
        'Filter': ['F200W', 'F200W'],
        'dra': [3.1123311174764146, 3.112159238530139],
        'ddec': [-1.8178707041954567, -1.8178090307003458],
        'dra (arcsec)': [3.1123311174764146, 3.112159238530139],
        'ddec (arcsec)': [-1.8178707041954567, -1.8178090307003458],
    })


def _frames(tmp_path, filt='F200W', dets=('nrca1', 'nrcb1'),
            vgroups=('02201', '04201'), exps=(1, 2)):
    d = tmp_path / 'gc2211' / filt / 'pipeline'
    d.mkdir(parents=True)
    for vg in vgroups:
        for e in exps:
            for det in dets:
                (d / f'jw02211023001_{vg}_{e:05d}_{det}_cal.fits').write_bytes(b'x')
    return d


def test_the_frames_on_disk_drive_the_expansion(tmp_path, monkeypatch):
    m = _load()
    monkeypatch.setattr(m, 'BASE', str(tmp_path))
    _frames(tmp_path)
    found = m.frames_for('gc2211', 'F200W')
    assert found[('jw02211023001', 1)] == {
        ('nrca', '02201'), ('nrca', '04201'),
        ('nrcb', '02201'), ('nrcb', '04201')}


def test_every_row_gains_module_and_vgroup(tmp_path, monkeypatch):
    m = _load()
    monkeypatch.setattr(m, 'BASE', str(tmp_path))
    _frames(tmp_path)
    out, _ = m.expand(_table(), 'gc2211')
    assert len(out) == 8            # 2 exposures x 2 modules x 2 vgroups
    assert set(out['Module']) == {'nrca', 'nrcb'}
    assert set(out['Vgroup']) == {'02201', '04201'}


def test_NO_offset_is_changed(tmp_path, monkeypatch):
    """The whole point: this re-keys rows, it never re-measures."""
    m = _load()
    monkeypatch.setattr(m, 'BASE', str(tmp_path))
    _frames(tmp_path)
    tbl = _table()
    out, _ = m.expand(tbl, 'gc2211')
    for col in ('dra', 'ddec', 'dra (arcsec)', 'ddec (arcsec)'):
        for exposure in (1, 2):
            want = float(tbl[tbl['Exposure'] == exposure][col][0])
            got = set(float(v) for v in out[out['Exposure'] == exposure][col])
            assert got == {want}, f'{col} exp{exposure}: {got} != {want}'


def test_every_frame_still_resolves_to_the_SAME_offset(tmp_path, monkeypatch):
    """Checked through the reader's own narrowing, not a copy of it."""
    m = _load()
    monkeypatch.setattr(m, 'BASE', str(tmp_path))
    _frames(tmp_path)
    tbl = _table()
    out, _ = m.expand(tbl, 'gc2211')
    checked, problems = m.verify(tbl, out, 'gc2211')
    assert problems == []
    assert checked == 8


def test_a_row_with_NO_frames_on_disk_is_kept_untouched(tmp_path, monkeypatch):
    """A row this tool cannot resolve is a row it must not touch: dropping it
    would delete a recorded solution for data that is merely not staged."""
    m = _load()
    monkeypatch.setattr(m, 'BASE', str(tmp_path))
    _frames(tmp_path, exps=(1,))            # nothing on disk for exposure 2
    out, notes = m.expand(_table(), 'gc2211')
    kept = out[out['Exposure'] == 2]
    assert len(kept) == 1
    assert float(kept['dra'][0]) == 3.112159238530139
    assert any('no frames on disk' in n[3] for n in notes)


def test_the_expanded_table_matches_exactly_one_row_per_frame(tmp_path,
                                                              monkeypatch):
    """The refusals this migration exists to lift are about MANY corrections
    landing on ONE row.  The converse must not appear: one frame matching many
    rows would make the reader raise instead."""
    m = _load()
    monkeypatch.setattr(m, 'BASE', str(tmp_path))
    _frames(tmp_path)
    out, _ = m.expand(_table(), 'gc2211')
    for module in ('nrca1', 'nrcb1'):
        for vgroup in ('02201', '04201'):
            for exposure in (1, 2):
                match = locked_row_match(out, visit='jw02211023001',
                                         exposure=exposure, filtername='F200W',
                                         module=module, vgroup=vgroup)
                assert match.sum() == 1, (module, vgroup, exposure, match.sum())


def test_an_already_expanded_table_is_a_no_op(tmp_path, monkeypatch, capsys):
    m = _load()
    monkeypatch.setattr(m, 'BASE', str(tmp_path))
    tbl = _table()
    tbl['Module'] = ['nrca', 'nrca']
    tbl['Vgroup'] = ['02201', '02201']
    p = tmp_path / 'Offsets_test.csv'
    tbl.write(str(p))
    assert m.main(['--field', 'gc2211', '--table', str(p)]) == 0
    assert 'already carries Module and Vgroup' in capsys.readouterr().out


def test_the_DRY_RUN_writes_nothing(tmp_path, monkeypatch):
    m = _load()
    monkeypatch.setattr(m, 'BASE', str(tmp_path))
    _frames(tmp_path)
    p = tmp_path / 'Offsets_test.csv'
    _table().write(str(p))
    before = p.read_bytes()
    assert m.main(['--field', 'gc2211', '--table', str(p)]) == 0
    assert p.read_bytes() == before


def test_EXECUTE_writes_and_leaves_a_backup(tmp_path, monkeypatch):
    m = _load()
    monkeypatch.setattr(m, 'BASE', str(tmp_path))
    _frames(tmp_path)
    p = tmp_path / 'Offsets_test.csv'
    _table().write(str(p))
    before = p.read_bytes()
    assert m.main(['--field', 'gc2211', '--table', str(p), '--execute']) == 0
    assert len(Table.read(str(p))) == 8
    backups = list(tmp_path.glob('Offsets_test.csv.pre_granularity_*'))
    assert len(backups) == 1
    assert backups[0].read_bytes() == before


def test_verify_REFUSES_rather_than_writing_when_a_frame_would_move(tmp_path,
                                                                    monkeypatch):
    """--verify is a gate, not a report: a table that would change any applied
    offset must exit nonzero and write nothing."""
    m = _load()
    monkeypatch.setattr(m, 'BASE', str(tmp_path))
    _frames(tmp_path)
    p = tmp_path / 'Offsets_test.csv'
    _table().write(str(p))

    real_expand = m.expand

    def sabotage(tbl, field):
        out, notes = real_expand(tbl, field)
        out['dra'][0] = 99.0        # one frame would receive a different shift
        return out, notes

    monkeypatch.setattr(m, 'expand', sabotage)
    rc = m.main(['--field', 'gc2211', '--table', str(p), '--verify',
                 '--execute'])
    assert rc == 1
    assert len(Table.read(str(p))) == 2, 'must not have written'


# ---------------------------------------------------------------------------
# The extraction itself: locked_row_match came out of _shift_from_locked, so
# its narrowing must still behave exactly as it did inline.
# ---------------------------------------------------------------------------

def test_a_module_less_table_still_matches_by_visit_exposure_filter():
    """The pre-expansion shape, which every other field still uses."""
    tbl = _table()
    match = locked_row_match(tbl, visit='jw02211023001', exposure=2,
                             filtername='F200W', module='nrca1',
                             vgroup='02201')
    assert match.sum() == 1
    assert float(tbl[match]['dra'][0]) == 3.112159238530139


def test_the_module_cell_matches_a_detector_by_family():
    """`nrca` in the table has to accept `nrca1`..`nrca4`."""
    tbl = _table()
    tbl['Module'] = ['nrca', 'nrcb']
    tbl['Exposure'] = [1, 1]
    for det in ('nrca1', 'nrca4'):
        m = locked_row_match(tbl, visit='jw02211023001', exposure=1,
                             filtername='F200W', module=det, vgroup='02201')
        assert m.sum() == 1
        assert str(tbl[m]['Module'][0]) == 'nrca'


@pytest.mark.parametrize('filt,expect', [('F200W', 1), ('F277W', 0)])
def test_the_filter_still_narrows(filt, expect):
    assert locked_row_match(_table(), visit='jw02211023001', exposure=1,
                            filtername=filt, module='nrca1',
                            vgroup='02201').sum() == expect
