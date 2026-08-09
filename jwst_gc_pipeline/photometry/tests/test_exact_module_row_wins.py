"""A table carrying BOTH module spellings must not match two rows.

cloudef's consensus table has, for F360M exposure 1 of visit jw02092002001::

    Module=nrcb      (the short-wave detectors, pooled)
    Module=nrcblong  (the long-wave detector)

`_module_variants('nrcblong')` is `{'nrcb', 'nrcblong'}` -- the de-`long`
variant exists so a table storing only the family spelling can still serve an
LW frame -- so the LW frame matched its own row AND the short-wave one::

    ValueError: consensus jitter match=2 for visit=jw02092002001 exp=1
    mod=nrcblong filt=F360M vgroup=02101; expected <=1 row

That failed the reduce (38998472_2) and stopped the field.  A row naming the
module exactly is never the wrong row, so it wins; the variant fallback keeps
every table carrying only one spelling working as before.
"""
import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    _module_variants, lookup_consensus_offset)


def _tbl(rows):
    return Table(rows=rows, names=('Filter', 'Module', 'Visit', 'Exposure',
                                   'Vgroup', 'dra (arcsec)', 'ddec (arcsec)'))


#: cloudef's real shape: both spellings for the same (visit, exposure, vgroup).
CLOUDEF = _tbl([
    ('F360M', 'nrcb', 'jw02092002001', 1, '02101', 0.01287, -0.01035),
    ('F360M', 'nrcblong', 'jw02092002001', 1, '02101', -0.00864, 0.00356),
])


def test_the_variant_set_is_what_made_it_ambiguous():
    """The premise, pinned: this is why the exact row has to win."""
    assert _module_variants('nrcblong') == {'nrcb', 'nrcblong'}
    assert 'nrcb' in _module_variants('nrcblong')


def _match(tbl, module, filtername='F360M', visit='jw02092002001',
           exposure=1, vgroup='02101'):
    """The jitter narrowing, as `_read_consensus` performs it."""
    vf = ((tbl['Visit'] == visit) & (tbl['Filter'] == filtername))
    rows = vf & (tbl['Exposure'] == int(exposure))
    exact = rows & np.array([str(m) == str(module) for m in tbl['Module']])
    if exact.sum() >= 1:
        return tbl[exact]
    variants = _module_variants(module)
    return tbl[rows & np.array([str(m) in variants for m in tbl['Module']])]


def test_the_LONG_WAVE_frame_gets_its_OWN_row():
    got = _match(CLOUDEF, 'nrcblong')
    assert len(got) == 1
    assert str(got['Module'][0]) == 'nrcblong'
    assert float(got['dra (arcsec)'][0]) == -0.00864


def test_the_SHORT_WAVE_frame_still_gets_the_family_row():
    got = _match(CLOUDEF, 'nrcb')
    assert len(got) == 1
    assert str(got['Module'][0]) == 'nrcb'
    assert float(got['dra (arcsec)'][0]) == 0.01287


def _real(tbl, module, filtername='F360M', visit='jw02092002001',
          exposure=1, vgroup='02101'):
    """Through the SHIPPING function.  The fallback tests used to run against
    the local `_match` restatement, so deleting the fallback from
    `lookup_consensus_offset` left the suite green -- the exact-wins mutant died
    only because one test drove the real thing."""
    t = tbl.copy()
    if 'prov_stage' not in t.colnames:
        t['prov_stage'] = ['m2'] * len(t)
    return lookup_consensus_offset(t, visit, exposure, module, filtername,
                                   vgroup=vgroup)[:2]


def test_a_table_with_ONLY_the_family_spelling_still_serves_an_LW_frame():
    """The fallback's whole purpose -- removing it would break every table that
    predates the per-channel rows.  Driven through the real function."""
    only_family = _tbl([('F360M', 'nrcb', 'jw02092002001', 1, '02101',
                         0.01287, -0.01035)])
    dra, ddec = _real(only_family, 'nrcblong')
    assert dra == pytest.approx(0.01287, abs=1e-6)


def test_a_DETECTOR_level_module_still_reaches_its_family_row():
    """`nrcb1` has no row of its own; it must still find `nrcb`."""
    only_family = _tbl([('F360M', 'nrcb', 'jw02092002001', 1, '02101',
                         0.01287, -0.01035)])
    dra, ddec = _real(only_family, 'nrcb1')
    assert dra == pytest.approx(0.01287, abs=1e-6)


def test_the_REAL_lookup_prefers_the_exact_row_over_the_family_one():
    """The other half, also end-to-end, so neither branch is measured on a
    copy of the narrowing."""
    dra, ddec = _real(CLOUDEF, 'nrcblong')
    assert dra == pytest.approx(-0.00864, abs=1e-6)


def test_the_two_spellings_need_not_co_exist_within_a_VGROUP():
    """Exactness is decided AFTER the vgroup narrowing.  Deciding it across all
    groups would take the exact row from the wrong group, lose it to the vgroup
    filter, and raise match=0 where the fallback would have found the right
    row.  Not reachable on any table today -- 0 such exposures across arches,
    cloudef and w51 -- but the ordering is a deliberate choice."""
    t = _tbl([
        ('F360M', 'nrcblong', 'jw02092002001', 1, '02101', 1.0, 1.0),
        ('F360M', 'nrcb', 'jw02092002001', 1, '02102', 2.0, 2.0),
    ])
    assert _real(t, 'nrcblong', vgroup='02101')[0] == pytest.approx(1.0)
    # group 02102 has only the family spelling; the fallback must find it
    assert _real(t, 'nrcblong', vgroup='02102')[0] == pytest.approx(2.0)


def test_a_detector_row_beats_the_family_row_when_both_exist():
    both = _tbl([
        ('F360M', 'nrcb', 'jw02092002001', 1, '02101', 0.01287, -0.01035),
        ('F360M', 'nrcb1', 'jw02092002001', 1, '02101', 0.00042, 0.00099),
    ])
    got = _match(both, 'nrcb1')
    assert len(got) == 1
    assert str(got['Module'][0]) == 'nrcb1'


def test_a_module_with_NO_row_at_all_matches_nothing():
    """Silence, not a wrong row: an absent module is the caller's problem to
    report, not something to satisfy with a neighbour's shift."""
    assert len(_match(CLOUDEF, 'nrca')) == 0


@pytest.mark.parametrize('mod', ['nrcalong', 'nrcblong'])
def test_neither_long_module_can_be_swallowed_by_its_family(mod):
    fam = mod.replace('long', '')
    t = _tbl([('F360M', fam, 'jw02092002001', 1, '02101', 1.0, 1.0),
              ('F360M', mod, 'jw02092002001', 1, '02101', 2.0, 2.0)])
    got = _match(t, mod)
    assert len(got) == 1
    assert float(got['dra (arcsec)'][0]) == 2.0


def test_the_reader_uses_exact_first():
    """Pinned by source: the fix is worthless if `lookup_consensus_offset` still
    builds its mask from the variant set alone."""
    import inspect
    src = inspect.getsource(lookup_consensus_offset)
    assert '_exact' in src
    assert src.index('_exact = ') < src.index('_module_variants(module)')
    # and the vgroup narrowing comes first, so exactness is decided within the
    # group being asked about
    assert src.index('vgroup_row_matches') < src.index('_exact = ')


def test_the_REAL_lookup_no_longer_raises_on_the_cloudef_shape():
    """End to end through the shipping function, not the local restatement."""
    t = CLOUDEF.copy()
    t['prov_stage'] = ['m2', 'm2']
    dra, ddec = lookup_consensus_offset(t, 'jw02092002001', 1, 'nrcblong',
                                        'F360M', vgroup='02101')[:2]
    assert dra == pytest.approx(-0.00864, abs=1e-6)
