"""Obs scoping of the MAST association-product mask (issue #416).

All 139 gc-treasury tiles share ``FILTERS='F212N;F480M'``, so the reduction's
filter-only mask over the program obs table selects EVERY released observation
and each fresh tile reduction downloads the whole program's asn products.  The
mask narrows the obs table to the observation under reduction.

The obs table is queried per PROPOSAL, so this narrows the two-field proposals
as well (2221 = brick + cloudc, 3958 = brick + sickle, 2045 = arches +
quintuplet): the reduce stops pulling the other field's association products
into this field's ``output_dir``.  What each field then reads is unchanged --
the consumer glob is already ``-o{field}``-scoped.

The reduction's own wiring of this mask is exercised in
``test_mast_download_obs_scope.py``, which drives ``main`` with an injected
``Observations`` and reads the table the download is handed.
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


def test_a_fields_own_rows_are_all_kept():
    """Every row spelling this observation survives, whatever else is in the
    proposal's table."""
    obs_ids = ['jw02221-o001_t001_nircam_clear-f405n-nrcalong',
               'jw02221-o001_t001_nircam_clear-f410m-nrcalong']
    assert observation_scope_mask(obs_ids, '2221', '001').all()


def test_a_proposal_spanning_two_fields_is_narrowed_to_this_one():
    """2221 covers brick (nircam o001) and cloudc (o002), and the obs table is
    queried per proposal, so a brick reduce previously downloaded cloudc's
    association products into brick's output_dir."""
    obs_ids = ['jw02221-o001_t001_nircam_clear-f182m',
               'jw02221-o002_t001_nircam_clear-f182m',
               'jw02221-o003_t001_nircam_clear-f182m']
    assert observation_scope_mask(obs_ids, '2221', '001').tolist() == [
        True, False, False]


def test_an_unpadded_field_selects_the_observation_it_names():
    """``startswith('jw10678-o1')`` accepts o100 and o139 while REJECTING tile
    1, and the mask is non-empty, so the "kept 0" report cannot catch it.  An
    unpadded --field is one slip across 139 array-job tiles."""
    obs_ids = ['jw10678-o001_t001_nircam_clear-f212n',
               'jw10678-o010_t010_nircam_clear-f212n',
               'jw10678-o100_t100_nircam_clear-f212n',
               'jw10678-o139_t139_nircam_clear-f212n']
    assert observation_scope_mask(obs_ids, '10678', '1').tolist() == [
        True, False, False, False]
    assert observation_scope_mask(obs_ids, '10678', '01').tolist() == [
        True, False, False, False]
    assert observation_scope_mask(obs_ids, '10678', '010').tolist() == [
        False, True, False, False]
    assert obs_id_prefix('10678', '1') == 'jw10678-o001'


def test_the_prefix_match_stops_at_the_observation_number():
    """Even correctly padded, a bare ``startswith`` accepts a longer number
    (o001 vs o0011); MAST always writes a separator after the token."""
    obs_ids = ['jw10678-o001_t001_nircam_clear-f212n',
               'jw10678-o0011_t011_nircam_clear-f212n']
    assert observation_scope_mask(obs_ids, '10678', '001').tolist() == [
        True, False]


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


# The production wiring is exercised behaviourally in
# test_mast_download_obs_scope.py: a source-text guard on the ``msk &= ...``
# line survives both regressions that matter (masking the wrong column, and
# swapping proposal_id/field), because neither changes that line's text.
