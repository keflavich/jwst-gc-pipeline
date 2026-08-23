"""An m2 checkpoint that failed before it measured anything must be REPORTED.

Issue #407.  ngc6334 F090W's only m2 record -- the only one the field has ever
produced -- says ``passed: false`` with three visits refused for duplicate
exposure identity and zero exposures ingested.  Every per-exposure counter the
monitor derives is therefore 0, which reads exactly like a filter that has not
been cataloged, and ``check_astrometry`` returned an EMPTY verdict list for it.
A field with no astrometry at all was invisible on the page and surfaced only
through a hand survey of every field's records.

These tests pin the two halves: the scanner must carry the record's own verdict
and reasons, and the check must turn "ran, refused, measured nothing" into a
finding -- without firing on a healthy record, which is what would make it noise
instead of a signal.
"""
import json
import os

from jwst_gc_pipeline.monitoring import checks, scan


def _write(tmp_path, name, payload):
    ckdir = tmp_path / 'astrometry_checkpoints'
    os.makedirs(ckdir, exist_ok=True)
    (ckdir / name).write_text(json.dumps(payload))
    return scan.astrometry_checkpoints(str(tmp_path))


#: The shape ngc6334 F090W is in: visits present, consensus refused, no
#: exposures anywhere, and the reasons only in ``failures``.
REFUSED = {
    'stage': 'm2', 'date': '2026-07-29T00:51:17Z', 'passed': False,
    'failures': [
        'ngc6334 F090W/nrcb F090W visit 1 [m2]: duplicate exposure identity: '
        '20 exposure identity/ies ingested more than once',
        'ngc6334 F090W/nrcb F090W visit 2 [m2]: duplicate exposure identity: '
        '20 exposure identity/ies ingested more than once',
    ],
    'visits': [{'visit': '1', 'consensus': {}, 'exposures': []},
               {'visit': '2', 'consensus': {}, 'exposures': []}],
}

#: A filter that measured its exposures and agreed with consensus.
HEALTHY = {
    'stage': 'm2', 'date': '2026-08-01T00:00:00Z', 'passed': True,
    'failures': [],
    'visits': [{'visit': '1', 'consensus': {'consensus_ok': True},
                'exposures': [{'off': 1.0, 'contrast': 40.0, 'ok': True},
                              {'off': 1.5, 'contrast': 38.0, 'ok': True}]}],
}


def _run(astrom, **kw):
    run = {'target': 'ngc6334', 'proposal': '6778', 'obsid': '001',
           'astrometry': astrom, 'per_filter': {}}
    run.update(kw)
    return run


def _empty_verdicts(astrom, **kw):
    return [v for v in checks.check_astrometry(_run(astrom, **kw))
            if v['name'].startswith('astrometry-checkpoint-empty')]


def test_scanner_carries_the_records_own_verdict_and_reasons(tmp_path):
    """``passed``/``failures``/``n_visits`` reach the summary.

    Without them the check has nothing to distinguish a refused consensus from
    an uncataloged filter: both give n_exposures 0 and n_misaligned 0.
    """
    got = _write(tmp_path, 'checkpoint_m2_F090W_latest.json', REFUSED)
    rec = got['F090W']
    assert rec['passed'] is False
    assert rec['n_visits'] == 2
    assert len(rec['failures']) == 2
    assert 'duplicate exposure identity' in rec['failures'][0]
    assert rec['n_exposures'] == 0
    json.dumps(rec, allow_nan=False)      # must stay JSON-dumpable for --json


def test_a_refused_consensus_is_a_reported_failure(tmp_path):
    """The verdict exists, is a fail, and carries the recorded reasons."""
    got = _write(tmp_path, 'checkpoint_m2_F090W_latest.json', REFUSED)
    verdicts = _empty_verdicts(got)
    assert len(verdicts) == 1, 'a field with no m2 astrometry produced no finding'
    v = verdicts[0]
    assert v['severity'] == 'fail'
    assert '0 exposures' in v['summary']
    assert '2 visit(s)' in v['summary']
    rows = v['evidence']['rows']
    assert rows['total'] == 2
    assert 'duplicate exposure identity' in rows['data'][0][0]


def test_a_healthy_record_produces_no_such_finding(tmp_path):
    """A check that fires on a good field is noise, not a signal."""
    got = _write(tmp_path, 'checkpoint_m2_F212N_latest.json', HEALTHY)
    assert _empty_verdicts(got) == []


def test_newer_frames_do_not_suppress_it(tmp_path):
    """No supersede-suppression, unlike the misalignment branch.

    The ``_latest`` record is overwritten by the next m2 run, so a record still
    saying "zero exposures" is the newest thing m2 has managed to say about the
    filter.  Reduced frames written afterwards do not answer it.
    """
    got = _write(tmp_path, 'checkpoint_m2_F090W_latest.json', REFUSED)
    got['F090W']['mtime'] = 100
    run_kw = {'per_filter': {'F090W': {'reduced': {'mtime': 10_000}}}}
    verdicts = _empty_verdicts(got, **run_kw)
    assert len(verdicts) == 1
    assert verdicts[0]['severity'] == 'fail'


def test_an_unattributable_record_warns_rather_than_fails(tmp_path):
    """An untokened record on a shared filter cannot be pinned to one obs."""
    got = _write(tmp_path, 'checkpoint_m2_F090W_latest.json', REFUSED)
    got['F090W']['attributable'] = False
    verdicts = _empty_verdicts(got)
    assert len(verdicts) == 1
    assert verdicts[0]['severity'] == 'warn'
    assert 'unattributed' in verdicts[0]['summary']
