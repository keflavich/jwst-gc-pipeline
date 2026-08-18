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


# ---------------------------------------------------------------------------
# a verdict on products that no longer exist is not a verdict on these ones
# ---------------------------------------------------------------------------

def _dated(tmp_path, name, passed, date, failures=()):
    p = _write(tmp_path, name, passed, failures=failures)
    rec = json.loads(p.read_text())
    rec['date'] = date
    p.write_text(json.dumps(rec))
    return p


def test_a_frozen_record_older_than_the_current_m2_is_SUPERSEDED(gate, tmp_path,
                                                                 capsys):
    """m2 rewrites the offsets table and the field is re-reduced from it, so a
    frozen verdict older than the newest m2 was measured on products that no
    longer exist.

    Live: arches's only frozen record is an m3/F212N from 2026-08-02 saying an
    exposure is 11.94 mas off consensus, while its m2 for the same filter last
    ran 2026-08-16 -- fourteen days and many re-reductions later.  Reading that
    as "this field FAILS" asserts a defect in products it never saw.
    """
    _dated(tmp_path, 'checkpoint_m2_F212N_o001_latest.json', True,
           '2026-08-16T08:38:49Z')
    _dated(tmp_path, 'checkpoint_m3_F212N_latest.json', False,
           '2026-08-02T07:00:25Z', failures=['11.94 mas off the visit consensus'])
    assert gate.main(['--field', 'fld']) == 3
    both = capsys.readouterr()
    assert 'SUPERSEDED' in both.out, both.out
    assert 'not a statement about what is on disk now' in both.out, both.out
    assert 'no verdict on the CURRENT products' in both.err, both.err


def test_a_superseded_record_does_not_become_a_PASS(gate, tmp_path):
    """The whole point of rc 3: superseded is not clean.  A field whose frozen
    stages have not run since its last m2 is unverified, and unverified does not
    ship."""
    _dated(tmp_path, 'checkpoint_m2_F212N_o001_latest.json', True,
           '2026-08-16T00:00:00Z')
    _dated(tmp_path, 'checkpoint_m3_F212N_latest.json', True,
           '2026-08-02T00:00:00Z')
    assert gate.main(['--field', 'fld']) == 3


def test_a_CURRENT_failure_outranks_a_superseded_one(gate, tmp_path):
    """rc 1 beats rc 3: a real failure on the current products is the more
    specific verdict, and burying it under "unverified" would understate it."""
    _dated(tmp_path, 'checkpoint_m2_F212N_o001_latest.json', True,
           '2026-08-10T00:00:00Z')
    _dated(tmp_path, 'checkpoint_m3_F212N_o001_latest.json', False,
           '2026-08-02T00:00:00Z', failures=['superseded'])
    _dated(tmp_path, 'checkpoint_m5_F405N_o001_latest.json', False,
           '2026-08-12T00:00:00Z', failures=['measured on the current products'])
    assert gate.main(['--field', 'fld']) == 1


def test_a_frozen_record_NEWER_than_its_m2_is_current(gate, tmp_path):
    """The ordinary case, and the one that must not be swept into 'superseded':
    brick's m5/F200W is a day newer than the newest m2 for that filter, so its
    2.3 mas is a live verdict and the field is refused on it."""
    _dated(tmp_path, 'checkpoint_m2_F200W_latest.json', True,
           '2026-07-22T13:16:17Z')
    _dated(tmp_path, 'checkpoint_m5_F200W_latest.json', False,
           '2026-07-23T20:14:42Z', failures=['MOVED 2.30 mas since the m2 freeze'])
    assert gate.main(['--field', 'fld']) == 1


def test_staleness_is_judged_per_FILTER_not_per_field(gate, tmp_path):
    """One filter's m2 re-running must not supersede another filter's frozen
    verdict -- the filters are reduced and cataloged independently."""
    _dated(tmp_path, 'checkpoint_m2_F115W_o004_latest.json', True,
           '2026-08-16T00:00:00Z')
    _dated(tmp_path, 'checkpoint_m2_F200W_latest.json', True,
           '2026-07-22T00:00:00Z')
    _dated(tmp_path, 'checkpoint_m5_F200W_latest.json', False,
           '2026-07-23T00:00:00Z', failures=['still current for F200W'])
    assert gate.main(['--field', 'fld']) == 1


def test_the_crossfilter_record_is_judged_against_the_fields_newest_m2(
        gate, tmp_path):
    """It has no filter of its own, so it has no per-filter baseline; the
    field's newest m2 is the only thing that can supersede it."""
    _dated(tmp_path, 'checkpoint_m2_F212N_o001_latest.json', True,
           '2026-08-16T00:00:00Z')
    _dated(tmp_path, 'checkpoint_m7_crossfilter_o001_latest.json', False,
           '2026-08-02T00:00:00Z', failures=['F212N vs F405N: 21.4 mas'])
    assert gate.main(['--field', 'fld']) == 3
