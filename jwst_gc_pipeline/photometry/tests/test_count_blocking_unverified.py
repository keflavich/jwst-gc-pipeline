"""The blast-radius counter must be reproducible (#341).

This number has been wrong twice, both times by counting the wrong thing:

  * 21 -- grepped ONE of the blocking site's three message spellings, so sgra
    dropped out;
  * 37 -- counted every ``_latest`` file, so a filter with both a tokened and
    an untokened ``_latest`` was counted twice.

Both are cheap to pin, so they are pinned here even though nothing else under
``scripts/analysis/`` has tests.
"""
import importlib.util
import json
import os

import pytest

_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                     'scripts', 'analysis', 'count_blocking_unverified.py')


def _mod():
    spec = importlib.util.spec_from_file_location('cbu', os.path.abspath(_PATH))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _write(d, name, **kw):
    rec = dict(stage='m2', passed=True, filtername=kw.pop('filt', 'F162M'),
               unverified=kw.pop('unverified', []), **kw)
    p = os.path.join(d, name)
    with open(p, 'w') as fh:
        json.dump(rec, fh)
    return p


def _tree(tmp_path, field='sgrc'):
    d = tmp_path / field / 'astrometry_checkpoints'
    d.mkdir(parents=True)
    return str(d)


TIE = ('sgrc F162M/nrca F162M visit 1 [m2]: consensus->reference offset 5.70 '
       'mas but the tie is not trustworthy ... -- NOT applying; investigate')
LEGACY_VIRAC = 'w51 F140M/merged [m2]: ... VIRAC tie is not trustworthy ...'
LEGACY_DISAGREE = 'sgra F212N/merged [m2]: ... independent checks DISAGREE ...'
BENIGN = 'cloudc F212N [m2]: consensus build failed: too few stars'


# --- the three spellings ----------------------------------------------------

@pytest.mark.parametrize('msg', [TIE, LEGACY_VIRAC, LEGACY_DISAGREE])
def test_every_historical_spelling_counts(msg):
    """Matching one spelling is how the first number came out 21."""
    m = _mod()
    entries, exact = m.blocking_entries(dict(unverified=[msg]))
    assert entries and exact is False


def test_a_could_not_measure_entry_does_not_count():
    m = _mod()
    entries, _ = m.blocking_entries(dict(unverified=[BENIGN]))
    assert entries == []


def test_the_persisted_field_wins_over_the_text():
    """Post-#341 records carry the classification, so no pattern is consulted --
    including when the message happens to contain a legacy spelling."""
    m = _mod()
    entries, exact = m.blocking_entries(
        dict(unverified=[TIE], unverified_blocking=[]))
    assert entries == [] and exact is True


# --- latest per FILTER, not per record NAME ---------------------------------

def test_a_tokened_and_an_untokened_latest_count_ONCE(tmp_path, capsys):
    """`_latest` records are tokened now and the untokened predecessor is not
    removed, so one filter can have two live files describing the same item
    (sgrc F162M at 5.70 and 5.72 mas).  That is how 37 happened."""
    m = _mod()
    d = _tree(tmp_path)
    older = _write(d, 'checkpoint_m2_F162M_latest.json', unverified=[TIE])
    newer = _write(d, 'checkpoint_m2_F162M_o012_latest.json', unverified=[TIE])
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    m.BASE = str(tmp_path)
    assert m.main([]) == 0
    out = capsys.readouterr().out
    assert 'scanned 2 record(s)' in out, out
    assert 'considered 1 record(s)' in out, out
    assert 'blocking-unverified entries (latest per filter): 1' in out, out
    # both name the same (field, filter)
    assert (m.record_filter_key(older, json.load(open(older)))
            == m.record_filter_key(newer, json.load(open(newer))))


def test_the_key_comes_from_the_record_not_the_filename(tmp_path):
    """The filename carries the observation token whose presence is exactly
    what makes two files one filter, so the key must not be parsed from it."""
    m = _mod()
    d = _tree(tmp_path)
    p = _write(d, 'checkpoint_m2_F162M_o012_latest.json', filt='F162M')
    assert m.record_filter_key(p, json.load(open(p))) == ('sgrc', 'F162M')


# --- the denominator --------------------------------------------------------

def test_an_empty_scan_is_not_a_clean_bill_of_health(tmp_path, capsys):
    """A mistyped base, an unmounted /orange or a glob that stopped matching
    must not print what a clean tree prints."""
    m = _mod()
    m.BASE = str(tmp_path / 'nope')
    rc = m.main([])
    out = capsys.readouterr().out
    assert rc == 1, out
    assert 'scanned 0 record(s)' in out
    assert 'not a clean bill of health' in out


def test_the_denominator_is_always_printed(tmp_path, capsys):
    m = _mod()
    d = _tree(tmp_path)
    _write(d, 'checkpoint_m2_F162M_latest.json', unverified=[TIE])
    _write(d, 'checkpoint_m2_F115W_latest.json', filt='F115W', unverified=[BENIGN])
    m.BASE = str(tmp_path)
    assert m.main([]) == 0
    out = capsys.readouterr().out
    assert 'scanned 2 record(s)' in out, out
    assert 'considered 2 record(s)' in out, out
    assert 'blocking-unverified entries (latest per filter): 1' in out, out


def test_the_method_tally_counts_every_record_considered(tmp_path, capsys):
    """`0 counted EXACTLY` must distinguish 'nothing written since #341' from
    'records written and none blocked'.  Counting only the records that
    CONTRIBUTED an entry cannot."""
    m = _mod()
    d = _tree(tmp_path)
    _write(d, 'checkpoint_m2_F162M_latest.json', filt='F162M',
           unverified=[], unverified_blocking=[])                # migrated, clean
    _write(d, 'checkpoint_m2_F115W_latest.json', filt='F115W',
           unverified=[TIE], unverified_blocking=[TIE])          # migrated, blocking
    m.BASE = str(tmp_path)
    assert m.main([]) == 0
    out = capsys.readouterr().out
    assert '2 record(s) counted EXACTLY' in out, out
    assert '0 record(s) counted from message text' in out, out
