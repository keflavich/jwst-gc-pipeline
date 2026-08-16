"""Obs scoping of the MAST association-product mask (issue #416).

All 139 gc-treasury tiles share ``FILTERS='F212N;F480M'``, so the reduction's
filter-only mask over the program obs table selects EVERY released observation
and each fresh tile reduction downloads the whole program's asn products.  The
mask must narrow the obs table to the observation under reduction while a
single-obs field passes trivially (unchanged behavior).
"""
import numpy as np

from jwst_gc_pipeline.reduction.mast_obs_scope import (
    obs_id_prefix, observation_scope_mask)


def test_prefix_is_five_digit_zero_padded():
    # jw + 5-digit program: 4-digit proposals pad (jw02221), 5-digit ones
    # spell as-is (jw10678) -- the #414 spelling, not jw0{proposal}.
    assert obs_id_prefix('2221', '001') == 'jw02221-o001'
    assert obs_id_prefix('10678', '042') == 'jw10678-o042'


def test_treasury_tile_keeps_only_its_own_observation():
    obs_ids = ['jw10678-o001_t001_nircam_clear-f212n',
               'jw10678-o002_t002_nircam_clear-f212n',
               'jw10678-o139_t139_nircam_clear-f480m']
    mask = observation_scope_mask(obs_ids, '10678', '001')
    assert mask.tolist() == [True, False, False]


def test_single_obs_field_passes_trivially():
    """Every row of a single-obs field already spells this observation, so the
    mask is all-True and current behavior is preserved."""
    obs_ids = ['jw02221-o001_t001_nircam_clear-f405n-nrcalong',
               'jw02221-o001_t001_nircam_clear-f410m-nrcalong']
    assert observation_scope_mask(obs_ids, '2221', '001').all()


def test_rows_without_an_observation_token_are_kept():
    """A candidate-association spelling (``jw{PPPPP}-c...``) attributes itself
    to no observation; dropping it would change what single-obs fields
    download today."""
    obs_ids = ['jw02221-c1001_t001_nircam', 'jw02221-o002_t002_nircam']
    mask = observation_scope_mask(obs_ids, '2221', '001')
    assert mask.tolist() == [True, False]


def test_mask_composes_with_the_filter_mask():
    """The reduction ANDs this onto its filter mask; the composed mask selects
    exactly the requested observation's rows of the requested filter."""
    obs_ids = np.array(['jw10678-o001_t001_nircam_clear-f212n',
                        'jw10678-o001_t001_nircam_clear-f480m',
                        'jw10678-o002_t002_nircam_clear-f212n'])
    filters = np.array(['F212N;F480M', 'F212N;F480M', 'F212N;F480M'])
    filter_msk = ((np.char.find(filters, 'F212N') >= 0) |
                  (np.char.find(obs_ids, 'f212n') >= 0))
    msk = filter_msk & observation_scope_mask(obs_ids, '10678', '001')
    assert msk.tolist() == [True, True, False]

def test_a_joint_obsid_keeps_both_halves():
    """A joint registration names a SET of observations, each spelled on its
    own in MAST.

    sgrb2's MIRI is registered ``002-998`` and sickle's ``001-002``; no MAST
    row is ever named ``jw05365-o002-998``, so a single-prefix test matched
    nothing and the reduction would have downloaded no association at all.
    """
    obs_ids = ['jw05365-o002_t001_miri_f770w',
               'jw05365-o998_t001_miri_f770w',
               'jw05365-o007_t001_miri_f770w']
    mask = observation_scope_mask(obs_ids, '5365', '002-998')
    assert mask.tolist() == [True, True, False]


def test_keeping_nothing_says_so(capsys):
    """An emptied obs table looks exactly like "nothing released yet"."""
    obs_ids = ['jw10678-o002_t002_nircam_clear-f212n']
    assert not observation_scope_mask(obs_ids, '10678', '001').any()
    out = capsys.readouterr().out
    assert 'kept 0 of 1' in out and 'jw10678-o001' in out, out


def test_the_reduction_ANDs_the_mask_onto_its_filter_mask():
    """Source guard on the one production wiring.

    Every test above calls the helper directly, so deleting the line that
    composes it into the reduction's mask leaves them all passing while the
    obs-blind download returns.  Astroquery's MAST call cannot be exercised
    here, so pin the composition and its position ahead of the product list.
    """
    import os
    import jwst_gc_pipeline.reduction as _red
    src = open(os.path.join(os.path.dirname(_red.__file__),
                            'PipelineRerunNIRCAM-LONG.py')).read()
    assert 'from jwst_gc_pipeline.reduction.mast_obs_scope import (' in src \
        or 'mast_obs_scope import observation_scope_mask' in src, \
        'the reduction no longer imports observation_scope_mask'
    i = src.find('msk &= observation_scope_mask(')
    assert i > 0, 'the obs-scope mask is no longer ANDed onto the filter mask'
    j = src.find('Observations.get_product_list(obs_table[msk])')
    assert j > i, 'the mask must be composed BEFORE the product list is built'
