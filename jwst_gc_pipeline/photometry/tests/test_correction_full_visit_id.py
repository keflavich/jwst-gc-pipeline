"""A correction must name the OBSERVATION it was measured on, not just a visit.

Every gc2211 m12 finalize died here once the field's records stopped coming out
empty (#352 unblocked them)::

    AmbiguousVisitMatchError: correction names visit '1' with no observation,
    but the table spans observations ['023', '028', '046', '049', '050'].
    Matching on the visit number alone would add this correction to every one of
    them -- the gc2211 #284 broadcast.

The guard is right; the correction was under-specified.  A per-frame catalog's
``VISIT`` metadatum is the visit NUMBER, and all five gc2211 observations are
visit 001, so the number alone addresses five pointings 0.3-17.6 arcmin apart.
The observation is in the source frame's name, which the catalog carries as
``FILENAME``.

These check the resolution, and -- separately -- that a resolved correction
actually SURVIVES ``_match_rows`` against the real five-observation table
shape, which is the thing that was failing.
"""
import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    AmbiguousVisitMatchError, _match_rows, resolve_full_visit_id,
    visit_obs_key)

FRAME = ('/orange/adamginsburg/jwst/gc2211//F200W/pipeline/'
         'jw02211{obs}001_02201_0000{exp}_nrcb1_destreak_o{obs}_crf.fits')


def _tbl(obs='050', exp=1, filename=None, visit=1):
    t = Table({'x': [1.0]})
    t.meta['VISIT'] = visit
    if filename is not False:
        t.meta['FILENAME'] = filename or FRAME.format(obs=obs, exp=exp)
    return t


def test_the_observation_comes_from_the_frame_name():
    tables = [_tbl(exp=i) for i in (1, 2, 3)]
    assert resolve_full_visit_id(tables, 1) == 'jw02211050001'
    assert visit_obs_key(resolve_full_visit_id(tables, 1)) == ('050', 1)


@pytest.mark.parametrize('obs', ['023', '028', '046', '049', '050'])
def test_each_gc2211_observation_resolves_to_ITSELF(obs):
    """The five are all visit 001; only the observation tells them apart."""
    assert resolve_full_visit_id([_tbl(obs=obs)], 1) == f'jw02211{obs}001'


def test_a_group_with_NO_filename_keeps_the_bare_visit():
    """No name, no observation -- and inventing one would be worse than the
    error the caller already raises."""
    assert resolve_full_visit_id([_tbl(filename=False)], 1) == 1


def test_an_unparseable_filename_keeps_the_bare_visit():
    assert resolve_full_visit_id([_tbl(filename='some_mosaic_i2d.fits')], 1) == 1


def test_a_MIXED_group_keeps_the_bare_visit_and_stays_refused():
    """Two observations' exposures in one consensus is the contamination class
    #352 fixed.  Picking either one attaches the correction to a pointing it was
    not measured on, so this must NOT resolve -- the bare visit keeps
    `_match_rows` refusing, and the error names the real problem."""
    tables = [_tbl(obs='023'), _tbl(obs='050')]
    assert resolve_full_visit_id(tables, 1) == 1


def test_a_name_that_DISAGREES_with_the_metadatum_is_not_trusted():
    """FILENAME says visit 002, VISIT says 1: one of them describes a different
    frame, and neither can be trusted to address a table row."""
    t = _tbl(filename='/p/jw02211050002_02201_00001_nrcb1_crf.fits', visit=1)
    assert resolve_full_visit_id([t], 1) == 1


def test_an_already_full_id_is_unchanged():
    t = _tbl()
    assert resolve_full_visit_id([t], 'jw02211050001') == 'jw02211050001'


# ---------------------------------------------------------------------------
# The point of the exercise: the resolved id must survive the matcher that was
# rejecting the bare one, against gc2211's real table shape.
# ---------------------------------------------------------------------------

def _gc2211_table():
    obs = ['023', '028', '046', '049', '050']
    return Table({
        'Visit': [f'jw02211{o}001' for o in obs],
        'Filter': ['F200W'] * len(obs),
        'Module': ['nrcb'] * len(obs),
        'dra': np.zeros(len(obs)),
        'ddec': np.zeros(len(obs)),
    })


def test_the_BARE_visit_is_still_refused():
    """Unchanged behaviour -- this is the guard, not the bug."""
    tbl = _gc2211_table()
    with pytest.raises(AmbiguousVisitMatchError):
        _match_rows(dict(visit='1', filtername='F200W', module='nrcb'), tbl)


def test_the_RESOLVED_visit_matches_exactly_one_observation():
    tbl = _gc2211_table()
    for i, obs in enumerate(['023', '028', '046', '049', '050']):
        full = resolve_full_visit_id([_tbl(obs=obs)], 1)
        rows = _match_rows(dict(visit=full, filtername='F200W',
                                module='nrcb'), tbl)
        assert list(rows) == [i], f'{obs} matched {list(rows)}, wanted [{i}]'


def test_a_single_observation_field_is_UNAFFECTED():
    """The upgrade must be a no-op wherever the visit number was already
    unambiguous, which is every field except gc2211."""
    tbl = Table({'Visit': ['jw02221001001'], 'Filter': ['F182M'],
                 'Module': ['nrca'], 'dra': [0.0], 'ddec': [0.0]})
    bare = _match_rows(dict(visit='1', filtername='F182M', module='nrca'), tbl)
    t = _tbl(filename='/p/jw02221001001_02201_00001_nrca1_crf.fits')
    full = resolve_full_visit_id([t], 1)
    assert full == 'jw02221001001'
    resolved = _match_rows(dict(visit=full, filtername='F182M',
                                module='nrca'), tbl)
    assert list(bare) == list(resolved) == [0]


# ---------------------------------------------------------------------------
# The WIRING.  Every test above calls resolve_full_visit_id directly, so both
# mutants below left the suite green:
#
#     corr_visit = visit                          (upgrade never happens)
#     bulk site left as visit= not corr_visit=    (a separate edit, easy to miss)
#
# The second is realistic: the bulk correction is appended ten lines further
# down than the per-exposure one.
# ---------------------------------------------------------------------------

def test_BOTH_correction_sites_use_the_resolved_visit():
    import inspect
    import re as _re

    from jwst_gc_pipeline.photometry import astrometry_checkpoint as A
    src = inspect.getsource(A.run_visit_checkpoint)

    appends = [m.start() for m in _re.finditer(r'corrections\.append\(', src)]
    assert len(appends) == 2, (
        f'expected the per-exposure and bulk sites, found {len(appends)}; '
        f'a new one must be checked too')
    for start in appends:
        block = src[start:start + 400]
        assert 'visit=corr_visit' in block, (
            'a corrections.append() that does not carry the resolved visit id '
            'emits a correction no multi-observation table can address')


def test_the_resolved_visit_is_computed_BEFORE_the_corrections():
    """`corr_visit = ...` moved below the appends would leave a NameError or a
    stale value; ordering is part of the wiring."""
    import inspect

    from jwst_gc_pipeline.photometry import astrometry_checkpoint as A
    src = inspect.getsource(A.run_visit_checkpoint)
    assert 'corr_visit = resolve_full_visit_id' in src, (
        'corr_visit is assigned from something other than the resolver, so the '
        'upgrade never happens and every correction keeps the bare visit')
    assert (src.index('corr_visit = resolve_full_visit_id')
            < src.index('corrections.append('))
