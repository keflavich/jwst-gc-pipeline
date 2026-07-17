"""Resolve a stage's UPSTREAM output facets from parent-product sidecars.

The rerun engine (``rerun.decide_stage``) keys the seeding cascade off each
record's ``inputs.upstream`` map -- ``{parent_stage: {data,wcs,meta}}``.  This
module builds that map at stamp time by reading the parent products' already-
written ``.prov.json`` sidecars, so a stage's record captures exactly which
upstream outputs it consumed.

A parent may resolve to MANY products (e.g. the ``imaging`` parent of ``m12`` is
every per-exposure ``_crf``; the ``m6`` parent of the cross-band ``m7`` is every
filter's ``m6`` catalog).  Those are POOLED into a single facet triple via
:func:`pool_facets` (an order-independent hash of the members' per-facet hashes),
so "did any upstream product change" is a single comparison.
"""
import hashlib

from . import prov_sidecar

# Immediate seeding parent(s) per stage.  'imaging' is the frame-data parent of
# every cataloging stage; the previous m-phase is the seed parent.  (m12's only
# parent is imaging; it has no previous cataloging phase.)
STAGE_PARENTS = {
    'imaging': [],
    'm12': ['imaging'],
    'm3': ['imaging', 'm12'],
    'm4': ['imaging', 'm3'],
    'm5': ['imaging', 'm4'],
    'm6': ['imaging', 'm5'],
    'm7': ['imaging', 'm6'],
    'm8': ['imaging', 'm7'],
}


def _pool_hash(values):
    """Order-independent sha256 over a set of hash strings (None-tolerant)."""
    present = sorted(v for v in values if v is not None)
    if not present:
        return None
    h = hashlib.sha256()
    for v in present:
        h.update(v.encode('utf-8'))
        h.update(b'\x00')
    return h.hexdigest()


def pool_facets(facet_dicts):
    """Pool a list of ``{data,wcs,meta}`` dicts into one triple.

    Each output facet is an order-independent hash of that facet across all
    members, so N per-exposure frames (or N per-filter catalogs) collapse to a
    single ``{data,wcs,meta}`` that changes iff ANY member's facet changed.
    """
    facet_dicts = [f for f in facet_dicts if f]
    return {
        'data': _pool_hash([f.get('data') for f in facet_dicts]),
        'wcs': _pool_hash([f.get('wcs') for f in facet_dicts]),
        'meta': _pool_hash([f.get('meta') for f in facet_dicts]),
    }


def _facets_from_sidecar(product_path):
    """Return the ``outputs`` facet dict recorded for ``product_path``, or None."""
    rec = prov_sidecar.read_sidecar(product_path)
    if rec is None:
        return None
    return rec.get('outputs')


def upstream_from_sidecars(parent_paths_by_stage):
    """Build the ``inputs.upstream`` map from parent product sidecars.

    Parameters
    ----------
    parent_paths_by_stage : dict
        ``{parent_stage: path | [paths]}``.  A single path or a list; a list is
        pooled via :func:`pool_facets`.  Parents with no readable sidecar (or an
        empty list) are omitted -- a missing upstream simply cannot contribute to
        the decision (the engine treats an absent facet as "unknown", never as
        "unchanged").

    Returns
    -------
    dict
        ``{parent_stage: {data,wcs,meta}}`` for every parent that yielded at
        least one readable sidecar.
    """
    out = {}
    for stage, paths in parent_paths_by_stage.items():
        if paths is None:
            continue
        if isinstance(paths, (str, bytes)):
            paths = [paths]
        facets = [_facets_from_sidecar(p) for p in paths]
        facets = [f for f in facets if f]
        if not facets:
            continue
        out[stage] = facets[0] if len(facets) == 1 else pool_facets(facets)
    return out
