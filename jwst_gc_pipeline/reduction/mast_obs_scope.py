"""Observation scoping for the MAST association-product download.

Program 10678 (GC Treasury) puts 139 observations -- one mosaic tile each --
under the single gc-treasury field, and every observation shares
``FILTERS='F212N;F480M'``, so the reduction's filter-only mask over the
program obs table selects EVERY released observation.  Each fresh tile
reduction then feeds the whole program's observations to
``Observations.get_product_list`` and downloads every tile's association
products -- an O(released-tiles) cost repeated per visit through the campaign,
on the MAST call with the documented 22 h hang history (issue #416).

``observation_scope_mask`` narrows the obs table to the observation under
reduction.  Pure numpy-over-strings, so it is testable without astroquery.
"""
import numpy as np


def obs_id_prefix(proposal_id, field):
    """The MAST ``obs_id`` prefix of one observation: ``jw{PPPPP}-o{field}``.

    JWST programs are zero-padded to five digits in product names, so proposal
    2221/obs 001 -> ``jw02221-o001`` and proposal 10678/obs 001 ->
    ``jw10678-o001``.
    """
    return f'jw{int(proposal_id):05d}-o{field}'


def observation_scope_mask(obs_ids, proposal_id, field):
    """Boolean mask keeping obs-table rows of the requested observation.

    ``obs_ids`` is the ``obs_id`` column of a MAST observation table already
    restricted to one proposal (``query_criteria(proposal_id=...)``); rows
    whose ``obs_id`` starts with ``jw{proposal}-o{field}`` are this
    observation's.  Rows carrying no ``-o`` observation token at all (e.g. a
    candidate-association ``jw{PPPPP}-c...`` spelling) are kept as well: their
    name attributes them to no foreign observation, and dropping them would
    change what a single-obs field downloads today.  For a single-observation
    field every row spells this observation, so the mask is all-True and
    behavior is unchanged.

    A JOINT ``field`` names a SET of observations, each with its own MAST
    spelling: sgrb2's MIRI is registered ``002-998`` and sickle's ``001-002``,
    and no MAST row is ever named ``jw05365-o002-998``.  Tested as one prefix
    it matches nothing, so every attributed row would be dropped and the
    reduction would download no association at all -- the same shape as the m2
    joint-token blocker.
    Decompose and test MEMBERSHIP, keeping the joint spelling in the set as
    well in case a product ever IS named that way.

    Keeping NOTHING is reported: an empty obs table reaches
    ``get_product_list`` looking like "no products released yet", and a wrong
    ``--field`` looks identical.
    """
    obs_ids = np.asarray(obs_ids, dtype=str)
    prefixes = list(dict.fromkeys(
        [obs_id_prefix(proposal_id, field)]
        + [obs_id_prefix(proposal_id, p) for p in str(field).split('-') if p]))
    own = np.zeros(obs_ids.shape, dtype=bool)
    for prefix in prefixes:
        own |= np.char.startswith(obs_ids, prefix)
    unattributed = np.char.find(obs_ids, '-o') < 0
    mask = own | unattributed
    if len(obs_ids) and not mask.any():
        print(f"MAST obs scoping: kept 0 of {len(obs_ids)} obs-table row(s) -- "
              f"none of them spells {sorted(prefixes)}.  Either this "
              f"observation has no released products yet, or --field names an "
              f"observation this proposal does not have.", flush=True)
    return mask
