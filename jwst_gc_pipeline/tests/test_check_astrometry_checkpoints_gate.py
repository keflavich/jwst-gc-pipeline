"""The release gate that makes deferring the frozen-stage checks safe.

Moving the stop out of the chain is only defensible if something refuses the
field before it ships.  These pin that it does, and that it fails CLOSED on
every way of not knowing.
"""
import importlib.util
import json
import os
import pathlib

import pytest

_GATE = (pathlib.Path(__file__).parents[2] / 'scripts' / 'release'
         / 'check_astrometry_checkpoints.py')


def _load():
    spec = importlib.util.spec_from_file_location('_ckpt_gate', _GATE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def gate(tmp_path, monkeypatch):
    monkeypatch.setenv('GC_BASEPATH_OVERRIDE', str(tmp_path))
    mod = _load()
    mod.BASE = str(tmp_path)
    return mod


def _write(tmp_path, name, passed, failures=(), unverified_blocking=()):
    d = tmp_path / 'fld' / 'astrometry_checkpoints'
    d.mkdir(parents=True, exist_ok=True)
    (d / name).write_text(json.dumps(dict(
        passed=passed, failures=list(failures),
        unverified_blocking=list(unverified_blocking),
        date='2026-08-18T00:00:00Z')))
    return d / name


def test_a_clean_field_passes(gate, tmp_path):
    _write(tmp_path, 'checkpoint_m3_F212N_o001_latest.json', True)
    _write(tmp_path, 'checkpoint_m7_crossfilter_o001_latest.json', True)
    assert gate.main(['--field', 'fld']) == 0


def test_a_failed_frozen_record_refuses(gate, tmp_path, capsys):
    _write(tmp_path, 'checkpoint_m3_F212N_o001_latest.json', False,
           failures=['F212N visit 1 [m3]: consensus->reference MOVED 30.0 mas'])
    assert gate.main(['--field', 'fld']) == 1
    both = capsys.readouterr()
    assert 'MOVED 30.0 mas' in both.out, both.out
    assert 'REFUSING TO STAGE' in both.err, both.err


def test_a_MEASURED_AND_REFUSED_item_also_refuses(gate, tmp_path, capsys):
    """`passed` is false for a measured-and-refused tie even with no `failures`
    (issue #312).  Reading only `failures` would ship the cloudc 731 mas case."""
    _write(tmp_path, 'checkpoint_m5_F410M_o002_latest.json', False,
           unverified_blocking=['consensus->reference offset 731.0 mas but the '
                                'tie is not trustworthy'])
    assert gate.main(['--field', 'fld']) == 1
    assert '731.0 mas' in capsys.readouterr().out


def test_the_m7_crossfilter_record_is_covered(gate, tmp_path):
    """It has no filter in its name, so a filter-keyed reader skips it -- and it
    is the gate that catches a field whose bands disagree with each other."""
    _write(tmp_path, 'checkpoint_m7_crossfilter_o001_latest.json', False,
           failures=['F212N vs F405N: 21.4 mas'])
    assert gate.main(['--field', 'fld']) == 1


def test_a_CORRECTING_stage_record_does_not_refuse(gate, tmp_path):
    """m2 raises in place, so an m2 record on disk with passed=false is an
    iteration that was stopped and re-run.  Refusing on it would block a field
    for a pass that has since been superseded."""
    _write(tmp_path, 'checkpoint_m2_F212N_o001_latest.json', False,
           failures=['an m2 iteration that was corrected and re-run'])
    _write(tmp_path, 'checkpoint_m3_F212N_o001_latest.json', True)
    assert gate.main(['--field', 'fld']) == 0


def test_no_records_at_all_refuses(gate, tmp_path):
    """A field that never ran the checkpoint is UNVERIFIED, not verified."""
    (tmp_path / 'fld' / 'astrometry_checkpoints').mkdir(parents=True)
    assert gate.main(['--field', 'fld']) == 3


def test_an_unreadable_record_refuses(gate, tmp_path):
    """Fail closed: a record that cannot be read is not a passing record."""
    p = _write(tmp_path, 'checkpoint_m3_F212N_o001_latest.json', True)
    p.write_text('{not json')
    assert gate.main(['--field', 'fld']) == 2


def test_observations_scope_the_scan(gate, tmp_path):
    _write(tmp_path, 'checkpoint_m3_F212N_o001_latest.json', True)
    _write(tmp_path, 'checkpoint_m3_F212N_o004_latest.json', False,
           failures=['o004 moved'])
    assert gate.main(['--field', 'fld', '--observations', '001']) == 0
    assert gate.main(['--field', 'fld', '--observations', '004']) == 1
    assert gate.main(['--field', 'fld']) == 1, 'unscoped must see both'


def test_a_TOKENISED_record_supersedes_its_untokened_sibling(gate, tmp_path):
    """Obs tokens arrived partway through, so a field carries both spellings and
    the untokened one is frozen at whatever it last wrote.  sickle's untokened
    m3/F187N is from 2026-08-05 and its tokenised one from 2026-08-17; reading
    both refuses the field forever no matter what the current run measured."""
    _write(tmp_path, 'checkpoint_m3_F187N_latest.json', False,
           failures=['a superseded 2026-08-05 pass'])
    _write(tmp_path, 'checkpoint_m3_F187N_o007_latest.json', True)
    assert gate.main(['--field', 'fld']) == 0


def test_an_untokened_record_with_no_sibling_is_still_read(gate, tmp_path):
    """The precedence rule must not become a way to hide a failure: a field with
    only untokened records is still fully checked."""
    _write(tmp_path, 'checkpoint_m3_F212N_latest.json', False,
           failures=['still the only record for this filter'])
    assert gate.main(['--field', 'fld']) == 1


def test_precedence_is_per_FILTER_not_per_field(gate, tmp_path):
    """One filter having a tokenised record must not silence another filter's
    untokened one."""
    _write(tmp_path, 'checkpoint_m3_F212N_o007_latest.json', True)
    _write(tmp_path, 'checkpoint_m3_F405N_latest.json', False,
           failures=['a different filter, no tokenised sibling'])
    assert gate.main(['--field', 'fld']) == 1
