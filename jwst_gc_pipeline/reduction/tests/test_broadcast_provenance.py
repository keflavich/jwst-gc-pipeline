"""One correction written to every visit is not five measurements (#284).

gc2211's five observations are 0.3-17.6 arcmin apart and tie to VIRAC2 at
0.05", 0.15", 3.2", 5.6" and 22.4" -- five different states.  Every one carried
the same `prov_*` pair, to 0.0000 mas:

    F200W  o023 o046 o049 o050        prov (-2470.1,  +2825.4) mas
    F277W  o023 o028 o046 o049 o050   prov (-7031.7, +15009.7) mas

The reducer reads `dra (arcsec)` = as-built + `prov_*`, so that smear reached
every product while the as-built pair -- which reproduces an independent swept
offset-histogram measurement of each region to 21-56 mas in RA and 1-26 mas in
Dec -- stayed correct and unread.
"""
import warnings

import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.reduction.validate_offsets_table import (
    BroadcastProvenanceError, BROADCAST_PROV_MIN_MAS, assert_offsets_table_sane,
    flag_broadcast_provenance)


def _tbl(visits, filt='F277W', prov=(-7031.7, 15009.7), per_visit=None):
    """One row per visit.  `per_visit` overrides prov for a given visit."""
    rows = []
    for i, v in enumerate(visits):
        p = (per_visit or {}).get(v, prov)
        rows.append({
            'Visit': v, 'Filter': filt, 'Exposure': 1, 'Module': 'nrcalong',
            'dra': 1.0 + i, 'ddec': -2.0 - i,
            'dra (arcsec)': 1.0 + i + p[0] / 1000.0,
            'ddec (arcsec)': -2.0 - i + p[1] / 1000.0,
            'prov_dra_added_mas': p[0], 'prov_ddec_added_mas': p[1]})
    return Table(rows)


GC2211_F277W = ['jw02211023001', 'jw02211028001', 'jw02211046001',
                'jw02211049001', 'jw02211050001']


def test_the_gc2211_shape_is_flagged():
    bad = flag_broadcast_provenance(_tbl(GC2211_F277W))
    assert len(bad) == 1, bad
    assert bad[0]['n_visits'] == 5
    assert bad[0]['spread_mas'] == pytest.approx(0.0, abs=1e-9)
    assert bad[0]['prov_dra_mas'] == pytest.approx(-7031.7)


def test_genuinely_per_visit_corrections_are_NOT_flagged():
    """The normal case, and the one that must never trip: every other live
    table measures its own correction per visit and they differ."""
    t = _tbl(GC2211_F277W, per_visit={
        'jw02211023001': (-7031.7, 15009.7),
        'jw02211028001': (-6800.0, 14500.0),
        'jw02211046001': (-15.0, 30.0),
        'jw02211049001': (-40.0, -12.0),
        'jw02211050001': (-2200.0, 5100.0)})
    assert flag_broadcast_provenance(t) == []


def test_a_single_visit_filter_cannot_be_broadcast():
    """gc2211's F150W exists only in o028, so there is nothing to smear across
    and its identical-by-definition provenance is not a finding."""
    assert flag_broadcast_provenance(_tbl(['jw02211028001'], filt='F150W')) == []


def test_two_uncorrected_visits_agreeing_about_zero_is_not_a_finding():
    """Every uncorrected row carries prov 0.  Without the magnitude floor the
    check would fire on every clean table in the tree."""
    t = _tbl(GC2211_F277W, prov=(0.0, 0.0))
    assert flag_broadcast_provenance(t) == []


def test_the_floor_is_on_the_MAGNITUDE_not_the_spread():
    """A tiny-but-nonzero shared correction is still shared; what makes zero
    uninteresting is that it is zero, not that it agrees."""
    small = BROADCAST_PROV_MIN_MAS / 10.0
    assert flag_broadcast_provenance(_tbl(GC2211_F277W, prov=(small, 0.0))) == []
    big = BROADCAST_PROV_MIN_MAS * 2
    assert len(flag_broadcast_provenance(_tbl(GC2211_F277W, prov=(big, 0.0)))) == 1


def test_a_table_with_no_provenance_columns_is_skipped():
    t = _tbl(GC2211_F277W)
    t.remove_columns(['prov_dra_added_mas', 'prov_ddec_added_mas'])
    assert flag_broadcast_provenance(t) == []


def test_it_WARNS_through_assert_offsets_table_sane():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        issues = assert_offsets_table_sane(_tbl(GC2211_F277W), context='test')
    assert any('BROADCAST OFFSETS PROVENANCE' in str(c.message) for c in caught)
    assert any(i.get('kind') == 'broadcast_provenance' for i in issues)


def test_it_STOPS_on_the_collapse_switch():
    """This changes the shift the reducer applies -- unlike the audit-trail
    divergence -- so it rides the collapse switch, not the weaker one."""
    with pytest.raises(BroadcastProvenanceError, match='BROADCAST'):
        assert_offsets_table_sane(_tbl(GC2211_F277W), context='test',
                                  raise_on_issue=True)


def test_the_message_says_REVERT_not_reconcile():
    """Reconciling copies the applied pair onto the as-built one, which here
    would destroy the only good copy -- the sickle #270 mistake."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        assert_offsets_table_sane(_tbl(GC2211_F277W))
    msg = next(str(c.message) for c in caught
               if 'BROADCAST' in str(c.message))
    assert 'revert_broadcast_provenance' in msg
    assert 'Do NOT reconcile' in msg
