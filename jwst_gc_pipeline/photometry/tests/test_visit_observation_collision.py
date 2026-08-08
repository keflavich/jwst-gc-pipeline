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


def test_the_refusal_stays_inside_the_writers_error_contract(tmp_path):
    """`AmbiguousVisitMatchError` is a ValueError and NOT what callers catch.

    Every caller and `run_field_retie_loop.sh` are written around
    `OffsetsTableUpdateError`, so letting this escape as itself silently
    changes what `update_offsets_table` can raise -- the identical defect #331
    fixed in this same function three commits earlier.

    Reachable, not theoretical: `scripts/reduction/step0_bulk_offset.py:308`
    builds `visit=str(args.visit).zfill(3)`, a bare visit, which is exactly the
    input that cannot be told apart on a multi-observation table.
    """
    import numpy as np
    from astropy.table import Table
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        OffsetsTableUpdateError, update_offsets_table)

    rows = []
    for v in GC2211:
        rows.append({'Visit': v, 'Filter': 'F277W', 'Exposure': 1,
                     'Module': 'nrcalong', 'dra': 0.0, 'ddec': 0.0,
                     'dra (arcsec)': 0.0, 'ddec (arcsec)': 0.0})
    p = str(tmp_path / 'Offsets_JWST_Brick2211_VIRAC2locked.csv')
    Table(rows).write(p, format='ascii.csv', overwrite=True)

    bare = [dict(visit='001', exposure=1, module='nrcalong', filtername='F277W',
                 dra_onsky_mas=10.0, ddec_onsky_mas=-5.0, dec_deg=-28.9,
                 source='step0-shaped bare visit')]
    with pytest.raises(OffsetsTableUpdateError) as exc:
        update_offsets_table(p, bare, stage='m2')
    msg = str(exc.value)
    assert 'NOT writing' in msg
    assert 'broadcast' in msg


def test_the_refusal_names_the_escape_hatch():
    """zfill(3) passes a full jw... id through unchanged, so the caller can fix
    this without any other change -- the message should say so."""
    t = _tbl()
    with pytest.raises(AmbiguousVisitMatchError) as exc:
        _match_rows(_corr('001'), t)
    msg = str(exc.value)
    assert 'step0_bulk_offset' in msg
    assert 'zfill' in msg


def test_a_BULK_correction_with_a_bare_visit_also_stays_in_contract(tmp_path):
    """The shape the error message names, and the one the guards skip.

    `_assert_one_correction_per_row` and
    `pool_corrections_to_table_granularity` both `continue` on
    `_is_bulk_correction`, so a bulk correction is never narrowed by them.  It
    used to reach `_match_rows` for the first time in the heal loop, outside the
    try, and escape as a bare `AmbiguousVisitMatchError` -- while
    `step0_bulk_offset.py`, which emits exactly this (no exposure, no module,
    `str(--visit).zfill(3)`), is what the message tells operators to run.
    """
    import numpy as np
    from astropy.table import Table
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        OffsetsTableUpdateError, _is_bulk_correction, update_offsets_table)

    rows = []
    for v in GC2211:
        rows.append({'Visit': v, 'Filter': 'F277W', 'Exposure': 1,
                     'Module': 'nrcalong', 'dra': 0.0, 'ddec': 0.0,
                     'dra (arcsec)': 0.0, 'ddec (arcsec)': 0.0})
    p = str(tmp_path / 'Offsets_JWST_Brick2211_VIRAC2locked.csv')
    Table(rows).write(p, format='ascii.csv', overwrite=True)

    bulk = dict(visit='001', exposure=None, module=None, filtername='F277W',
                dra_onsky_mas=12.0, ddec_onsky_mas=-8.0, dec_deg=-28.9,
                source='step0 bulk offset')
    assert _is_bulk_correction(bulk), 'fixture must exercise the bulk path'

    before = open(p).read()
    with pytest.raises(OffsetsTableUpdateError) as exc:
        update_offsets_table(p, [bulk], stage='m2')
    assert 'NOT writing' in str(exc.value)
    assert 'broadcast' in str(exc.value)
    assert open(p).read() == before, 'a refusal must not half-write'


def test_the_narrowing_happens_once(tmp_path):
    """The heal loop reuses the guards' narrowing rather than repeating it, so
    there is no second site that can raise and no way for the two to disagree
    about which rows a correction touches."""
    import inspect
    from jwst_gc_pipeline.photometry import astrometry_checkpoint as ac
    src = inspect.getsource(ac.update_offsets_table)
    assert '_rows_for = [(corr, _match_rows(corr, tbl)) for corr in corrections]' in src
    heal = src[src.index('_touched = set()'):src.index('_heal_column_pairs')]
    assert '_match_rows' not in heal, (
        'the heal loop must reuse _rows_for, not narrow again outside the try')
