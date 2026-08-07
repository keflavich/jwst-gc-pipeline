"""Five observations that are all visit 001 are five different pointings (#284).

A JWST visit id is ``jw`` + proposal(5) + observation(3) + visit(3).  Every
narrowing site keyed on ``int(str(visit)[-3:])`` -- the visit number alone --
which is unique for a field whose observations each hold one visit.  gc2211 is
the exception:

    jw02211023001  jw02211028001  jw02211046001  jw02211049001  jw02211050001
             ^^^ observation                              ^^^ visit == 001

so all five keyed to 1, and a correction measured on ONE of them was added to
all five.  Its table ended up carrying a single ``prov_*`` pair across five
pointings 0.3-17.6 arcmin apart whose true ties to VIRAC2 are 0.05", 0.15",
3.2", 5.6" and 22.4".
"""
import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    AmbiguousVisitMatchError, _match_rows, visit_obs_key)

GC2211 = ['jw02211023001', 'jw02211028001', 'jw02211046001',
          'jw02211049001', 'jw02211050001']


def _tbl(visits=GC2211, filt='F277W'):
    rows = []
    for v in visits:
        for exp in (1, 2):
            rows.append({'Visit': v, 'Filter': filt, 'Exposure': exp,
                         'Module': 'nrcalong', 'dra': 0.0, 'ddec': 0.0,
                         'dra (arcsec)': 0.0, 'ddec (arcsec)': 0.0})
    return Table(rows)


def _corr(visit, exposure=None, filt='F277W'):
    c = {'visit': visit, 'filtername': filt, 'module': None}
    if exposure is not None:
        c['exposure'] = exposure
    return c


def test_the_visit_key_separates_observation_from_visit():
    assert visit_obs_key('jw02211023001') == ('023', 1)
    assert visit_obs_key('jw02211050001') == ('050', 1)
    # a bare number keeps the old meaning
    assert visit_obs_key('001') == (None, 1)
    assert visit_obs_key(2) == (None, 2)


def test_a_correction_lands_on_ONE_observation():
    """The defect, directly: before this, o023's correction matched all ten
    rows -- every exposure of every one of the five observations."""
    tbl = _tbl()
    idx = _match_rows(_corr('jw02211023001'), tbl)
    assert len(idx) == 2, idx                    # o023's two exposures only
    assert set(str(v) for v in tbl['Visit'][idx]) == {'jw02211023001'}


def test_each_of_the_five_gets_its_own_rows_and_they_do_not_overlap():
    tbl = _tbl()
    seen = set()
    for v in GC2211:
        idx = set(int(i) for i in _match_rows(_corr(v), tbl))
        assert len(idx) == 2, (v, idx)
        assert not (idx & seen), f'{v} overlaps an earlier observation'
        seen |= idx
    assert len(seen) == len(tbl)                 # and together they cover it


def test_a_bare_visit_number_across_several_observations_RAISES():
    """Refusing beats guessing: with no observation on the correction there is
    nothing to tell the five apart, and matching them all is the bug."""
    with pytest.raises(AmbiguousVisitMatchError, match='broadcast'):
        _match_rows(_corr('001'), _tbl())


def test_a_bare_visit_number_is_fine_when_there_is_one_observation():
    """Every other field, and the old behaviour, must be untouched."""
    tbl = _tbl(visits=['jw05365001001', 'jw05365001002'], filt='F212N')
    idx = _match_rows(_corr('001', filt='F212N'), tbl)
    assert len(idx) == 2
    assert set(str(v) for v in tbl['Visit'][idx]) == {'jw05365001001'}


def test_a_single_observation_field_matches_by_visit_as_before():
    """sgrb2-shaped: one observation, two visits, distinguished by the last
    three digits exactly as they always were."""
    tbl = _tbl(visits=['jw05365001001', 'jw05365001002'], filt='F212N')
    for v, want in (('jw05365001001', 'jw05365001001'),
                    ('jw05365001002', 'jw05365001002')):
        idx = _match_rows(_corr(v, filt='F212N'), tbl)
        assert set(str(x) for x in tbl['Visit'][idx]) == {want}


def test_exposure_narrowing_still_applies_within_the_observation():
    tbl = _tbl()
    idx = _match_rows(_corr('jw02211050001', exposure=2), tbl)
    assert len(idx) == 1
    assert str(tbl['Visit'][idx][0]) == 'jw02211050001'
    assert int(tbl['Exposure'][idx][0]) == 2


def test_a_correction_for_an_absent_observation_matches_nothing():
    """Not an error here -- callers decide what an empty match means -- but it
    must not fall back to matching some other observation."""
    assert len(_match_rows(_corr('jw02211099001'), _tbl())) == 0


def test_rows_with_a_bare_visit_stay_eligible_for_an_observation_correction():
    """A table that mixes full ids and bare numbers is matched as loosely as
    its least-specific rows allow, rather than silently narrowing to none."""
    tbl = _tbl(visits=['jw02211023001', '001'])
    idx = _match_rows(_corr('jw02211023001'), tbl)
    assert set(str(v) for v in tbl['Visit'][idx]) == {'jw02211023001', '001'}
