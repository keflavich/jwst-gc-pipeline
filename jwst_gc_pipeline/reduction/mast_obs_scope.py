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
    """
    obs_ids = np.asarray(obs_ids, dtype=str)
    own = np.char.startswith(obs_ids, obs_id_prefix(proposal_id, field))
    unattributed = np.char.find(obs_ids, '-o') < 0
    return own | unattributed
