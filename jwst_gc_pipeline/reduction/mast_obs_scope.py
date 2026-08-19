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
import re

import numpy as np

#: The five-digit proposal spelling MAST uses (``jw02221``, ``jw10678``).
#: ``mast_names.jw_prefix`` is the shared helper for it and arrives with PR
#: #426; until then this module carries the same rule, with the same validation,
#: so the two agree on every proposal and on what is not a proposal.  Once #426
#: is in, delete ``_jw_prefix`` and import ``jw_prefix`` unconditionally.
try:
    from jwst_gc_pipeline.mast_names import jw_prefix
except ImportError:
    #: One to five decimal digits, leading zeros allowed; the shapes ``int()``
    #: would have accepted and normalised silently (a sign, an underscore digit
    #: group, surrounding whitespace, a non-ASCII digit) are refused, and so is
    #: a number wider than the five digits every JWST product name assumes.
    _PROPOSAL_RE = re.compile(r'\A[0-9]{1,5}\Z')

    def jw_prefix(proposal_id):
        """``'jw02221'`` / ``'jw10678'`` -- a proposal's MAST filename prefix."""
        text = proposal_id if isinstance(proposal_id, str) else str(proposal_id)
        if _PROPOSAL_RE.match(text) is None or int(text) == 0:
            raise ValueError(
                f'proposal_id {proposal_id!r} is not a JWST proposal number: '
                f'expected one to five decimal digits')
        return 'jw' + format(int(text), '05d')


def observation_number(field):
    """``field`` as MAST spells an observation number: three digits.

    ``'1'``, ``'01'`` and ``'001'`` all name observation 001 and MAST writes
    ``jw10678-o001``, so an unpadded ``--field`` has to be padded before it can
    be compared.  A field that is not a plain number (a joint ``'002-998'``
    registration, which the caller also decomposes) is returned unchanged: the
    mask then matches nothing under that spelling and says so, which beats
    raising inside a reduction over a name MAST may yet use.
    """
    text = str(field).strip()
    return f'{int(text):03d}' if text.isdigit() else text


def obs_id_prefix(proposal_id, field):
    """The MAST ``obs_id`` prefix of one observation: ``jw{PPPPP}-o{NNN}``.

    JWST programs are zero-padded to five digits in product names and
    observations to three, so proposal 2221/obs 1 -> ``jw02221-o001`` and
    proposal 10678/obs 42 -> ``jw10678-o042``.

    The prefix comes from ``jw_prefix``, which REFUSES a value that is not a
    proposal number rather than formatting it: this string is what a MAST query
    is filtered on, so a proposal wider than five digits or carrying a sign
    would otherwise become a prefix that matches nothing and silently empties
    the obs table.
    """
    return f'{jw_prefix(proposal_id)}-o{observation_number(field)}'


def observation_scope_mask(obs_ids, proposal_id, field):
    """Boolean mask keeping obs-table rows of the requested observation.

    ``obs_ids`` is the ``obs_id`` column of a MAST observation table already
    restricted to one proposal (``query_criteria(proposal_id=...)``); rows
    whose ``obs_id`` starts with ``jw{proposal}-o{field}`` are this
    observation's.  Rows carrying no ``-o`` observation token at all (e.g. a
    candidate-association ``jw{PPPPP}-c...`` spelling) are kept as well: their
    name attributes them to no foreign observation, and dropping them would
    change what a single-obs field downloads today.

    A proposal is not a field.  The obs table is queried per PROPOSAL, and
    several proposals span two fields -- 2221 covers brick (nircam 001) and
    cloudc (nircam 002), 3958 brick and sickle, 2045 arches and quintuplet,
    1979 m4 and ngc6397 -- so this mask narrows those runs too: a brick reduce
    stops pulling cloudc's association products into brick's ``output_dir``.
    The consumer glob downstream is already ``-o{field}``-scoped, so what the
    narrowing removes is download volume rather than input.

    The prefix match is ANCHORED on the character after the observation
    number.  ``startswith('jw10678-o1')`` also accepts ``jw10678-o100`` and
    ``jw10678-o139`` while REJECTING tile 1 -- a non-empty mask over the wrong
    tiles, which the "kept 0" report below cannot catch.  ``observation_number``
    pads the field and the ``(?![0-9])`` lookahead closes the rest.

    A JOINT ``field`` names a SET of observations, each with its own MAST
    spelling: sgrb2's MIRI is registered ``002-998`` and sickle's ``001-002``,
    and no MAST row is ever named ``jw05365-o002-998``.  Tested as one prefix
    it matches nothing, so every attributed row would be dropped and the
    reduction would download no association at all.  That is how the m2
    checkpoint's per-frame filter emptied sgrb2's and sickle's F770W input
    (60 -> 0 catalogs) before it decomposed the joint token, so decompose and
    test MEMBERSHIP here as well, keeping the joint spelling in the set in case
    a product ever IS named that way.

    Keeping NOTHING is reported: an empty obs table reaches
    ``get_product_list`` looking like "no products released yet", and a wrong
    ``--field`` looks identical.
    """
    obs_ids = np.asarray(obs_ids, dtype=str)
    prefixes = list(dict.fromkeys(
        [obs_id_prefix(proposal_id, field)]
        + [obs_id_prefix(proposal_id, p) for p in str(field).split('-') if p]))
    anchored = re.compile('(?:' + '|'.join(re.escape(p) for p in prefixes)
                          + ')(?![0-9])')
    own = np.array([bool(anchored.match(str(o))) for o in obs_ids], dtype=bool)
    own = own.reshape(obs_ids.shape)
    unattributed = np.char.find(obs_ids, '-o') < 0
    mask = own | unattributed
    if len(obs_ids) and not mask.any():
        print(f"MAST obs scoping: kept 0 of {len(obs_ids)} obs-table row(s) -- "
              f"none of them spells {sorted(prefixes)}.  Either this "
              f"observation has no released products yet, or --field names an "
              f"observation this proposal does not have.", flush=True)
    return mask
