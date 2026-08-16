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