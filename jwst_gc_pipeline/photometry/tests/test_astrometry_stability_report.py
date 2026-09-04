"""The per-release astrometric stability report.

The behaviours pinned here are the ones an adversarial review found wrong in
the first cut: reading only ``failures`` (goes blind exactly when the tolerance
is generous), characterising only the exposures past the threshold (a selection
effect that manufactures significance), a sign-based agree/disagree test (wrong
in both directions), and a writer that could abort a release.
"""
import json
import os
import sys

import pytest

_RELEASE = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..', 'scripts', 'release'))
if _RELEASE not in sys.path:
    sys.path.insert(0, _RELEASE)

from astrometry_stability_report import (  # noqa: E402
    GROUP_AGREEMENT_MAS, collect, render)


def _record(exposures, tolerances=None, failures=None):
    """A checkpoint record shaped like the real ones."""
    rec = {'visits': [{'visit': '1', 'exposures': [
        {'key': list(k), 'dra': v[0], 'ddec': v[1]}
        for k, v in exposures.items()]}]}
    if tolerances is not None:
        rec['tolerances'] = tolerances
    if failures is not None:
        rec['failures'] = failures
    return rec


def _field(tmp_path, records):
    d = tmp_path / 'astrometry_checkpoints'
    d.mkdir(exist_ok=True)
    for name, rec in records.items():
        (d / f'checkpoint_{name}_latest.json').write_text(json.dumps(rec))
    return tmp_path


def _key(det, exp, filt='F200W', visit='1'):
    return (visit, exp, det, filt, '04101')


# --- the movement must come from the measurements, not the failure text ------

def test_movement_inside_tolerance_is_still_reported(tmp_path):
    # The regression that motivated the rewrite: a 1.2 mas movement is below a
    # 2.0 mas tolerance, so it never becomes a failure line.  A failure-scraping
    # reader would report "nothing moved" for a field that moved.
    base = {_key('nrcb2', i): (0.0, 0.0) for i in range(1, 5)}
    later = {_key('nrcb2', i): (1.2, 0.0) for i in range(1, 5)}
    d = _field(tmp_path, {
        'm2_F200W_o004': _record(base),
        'm3_F200W_o004': _record(later, tolerances={'stage_stability_tol_mas': 2.0},
                                 failures=[])})
    data = collect(str(d))
    assert len(data['per_stage']['m3']) == 4
    text = render('brick', data, 'vtest')
    assert 'No frozen-stage movement' not in text
    assert '+1.20' in text


def test_full_population_is_measured_not_just_the_tail(tmp_path):
    # 4 exposures move 2.3 mas and 8 move 0.1: the report must describe all 12,
    # because quoting the tight tail alone overstates the systematic.
    base = {_key('nrcb2', i): (0.0, 0.0) for i in range(1, 13)}
    later = {_key('nrcb2', i): (2.3 if i <= 4 else 0.1, 0.0)
             for i in range(1, 13)}
    d = _field(tmp_path, {'m2_F200W_o004': _record(base),
                          'm3_F200W_o004': _record(later)})
    rows = collect(str(d))['per_stage']['m3']
    assert len(rows) == 12
    text = render('brick', collect(str(d)), 'vtest')
    # mean over all twelve is 0.83, not the tail's 2.3
    assert '+0.83' in text


def test_exposure_absent_from_m2_is_not_a_movement(tmp_path):
    # No frozen value exists, so there is nothing it can have moved away from.
    d = _field(tmp_path, {
        'm2_F200W_o004': _record({_key('nrcb2', 1): (0.0, 0.0)}),
        'm3_F200W_o004': _record({_key('nrcb2', 1): (0.5, 0.0),
                                  _key('nrcb2', 2): (9.0, 0.0)})})
    rows = collect(str(d))['per_stage']['m3']
    assert [r['exposure'] for r in rows] == [1]


def test_stage_with_no_m2_partner_is_skipped(tmp_path):
    d = _field(tmp_path, {'m3_F200W_o004': _record({_key('nrcb2', 1): (5.0, 0.0)})})
    assert collect(str(d))['per_stage'] == {}


# --- group agreement is a magnitude test, not a sign test -------------------

def test_groups_straddling_zero_but_agreeing_are_not_called_opposed(tmp_path):
    # (+2.50,+0.01) and (+2.52,-0.01) differ by 0.02 mas.  A sign test calls
    # them opposed and then claims a pooled average "would be near zero", which
    # is false -- it is +2.51.
    base = {_key(d_, i): (0.0, 0.0) for d_ in ('nrca2', 'nrca3')
            for i in range(1, 5)}
    later = {}
    for i in range(1, 5):
        later[_key('nrca2', i)] = (2.50, +0.01)
        later[_key('nrca3', i)] = (2.52, -0.01)
    d = _field(tmp_path, {'m2_F200W_o004': _record(base),
                          'm3_F200W_o004': _record(later)})
    text = render('brick', collect(str(d)), 'vtest')
    assert 'do not share one offset' not in text
    assert 'agree to within' in text


def test_groups_far_apart_with_the_same_sign_are_called_differential(tmp_path):
    # (+1,0) and (+9,0) share a sign and are 8 mas apart; pooling them gives
    # +5, a number describing neither.
    base = {_key(d_, i): (0.0, 0.0) for d_ in ('nrca2', 'nrca3')
            for i in range(1, 5)}
    later = {}
    for i in range(1, 5):
        later[_key('nrca2', i)] = (1.0, 0.0)
        later[_key('nrca3', i)] = (9.0, 0.0)
    d = _field(tmp_path, {'m2_F200W_o004': _record(base),
                          'm3_F200W_o004': _record(later)})
    text = render('brick', collect(str(d)), 'vtest')
    assert 'do not share one offset' in text
    assert 'differential' in text
    assert 'Pooled mean shift' not in text


def test_agreement_threshold_is_the_documented_constant():
    assert 0 < GROUP_AGREEMENT_MAS <= 2.0


# --- observation scoping is exact -------------------------------------------

def test_obs_token_excludes_an_untokened_legacy_record(tmp_path):
    # An untokened record belongs to no tokened release; admitting it pools a
    # different reduction's numbers into this report.
    base = {_key('nrcb2', 1): (0.0, 0.0)}
    d = _field(tmp_path, {
        'm2_F200W_o004': _record(base),
        'm3_F200W_o004': _record({_key('nrcb2', 1): (1.0, 0.0)}),
        'm2_F200W': _record(base),
        'm3_F200W': _record({_key('nrcb2', 1): (7.0, 0.0)})})
    scoped = collect(str(d), 'o004')['per_stage']['m3']
    assert len(scoped) == 1
    assert scoped[0]['moved_mas'] == pytest.approx(1.0)
    assert len(collect(str(d))['per_stage']['m3']) == 2


# --- failure modes are reported, never silent, never fatal ------------------

def test_unreadable_record_is_disclosed_in_the_report(tmp_path):
    d = _field(tmp_path, {'m2_F200W_o004': _record({_key('nrcb2', 1): (0.0, 0.0)})})
    (d / 'astrometry_checkpoints'
     / 'checkpoint_m3_F200W_o004_latest.json').write_text('{ not json')
    data = collect(str(d))
    assert len(data['unreadable']) == 1
    assert 'Incomplete' in render('brick', data, 'vtest')


@pytest.mark.parametrize('payload', ['null', '[]', '"a string"', '3'])
def test_a_record_that_is_not_a_dict_does_not_raise(tmp_path, payload):
    d = _field(tmp_path, {'m2_F200W_o004': _record({_key('nrcb2', 1): (0.0, 0.0)})})
    (d / 'astrometry_checkpoints'
     / 'checkpoint_m3_F200W_o004_latest.json').write_text(payload)
    data = collect(str(d))          # must not raise
    render('brick', data, 'vtest')  # must not raise
    assert data['unreadable']


def test_no_records_says_unstated_not_stable(tmp_path):
    # "no movement measured" must never read as "the positions are good".
    text = render('arches', collect(str(_field(tmp_path, {}))), 'vtest')
    assert 'NOT a statement that the positions' in text


def test_report_states_photometry_is_unaffected(tmp_path):
    text = render('arches', collect(str(_field(tmp_path, {}))), 'vtest')
    assert 'does not affect photometry' in text.lower()


def test_malformed_exposure_entries_are_skipped_not_fatal(tmp_path):
    bad = {'visits': [{'visit': '1', 'exposures': [
        {'key': ['1', 1, 'nrcb2', 'F200W', '04101'], 'dra': None, 'ddec': 1.0},
        {'key': [], 'dra': 1.0, 'ddec': 1.0},
        {'key': ['1', 2, 'nrcb2', 'F200W', '04101'], 'dra': 'x', 'ddec': 1.0},
        {'key': ['1', 3, 'nrcb2', 'F200W', '04101'], 'dra': 1.0, 'ddec': 1.0},
    ]}]}
    d = _field(tmp_path, {
        'm2_F200W_o004': _record({_key('nrcb2', i): (0.0, 0.0)
                                  for i in range(1, 4)}),
        'm3_F200W_o004': bad})
    rows = collect(str(d))['per_stage']['m3']
    assert [r['exposure'] for r in rows] == [3]
