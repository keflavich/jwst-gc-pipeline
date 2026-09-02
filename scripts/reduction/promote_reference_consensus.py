#!/usr/bin/env python
"""Promote one filter's consensus catalog to the field's JWST reference.

Every filter's m2 checkpoint writes ``catalogs/<filter>[_token]_consensus.fits``.
Exactly one of them is the field's ANCHOR — the filter closest to VIRAC2 in
wavelength and in which stars it leaves unsaturated
(``consensus_catalog.reference_filter``) — and that one is copied to
``catalogs/jwst_reference[_token]_consensus.fits``, which every other filter
ties to.

This is a separate step rather than part of the checkpoint because the
checkpoint runs per filter and cannot know which of the field's filters ranks
best.  Run it once all the m2 checkpoints for a field are done.

    python scripts/reduction/promote_reference_consensus.py \
        --basepath /orange/adamginsburg/jwst/brick --proposal-id 2221

With no ``--filter`` the field's filters are read from ``fields.yaml``; pass
``--from-disk`` to rank only the filters that actually have a consensus
catalog written (useful mid-campaign, but note it can promote a lesser filter
if the best one's m2 has not run yet — it says so when that happens).
"""
import argparse
import glob
import os
import sys

from jwst_gc_pipeline.photometry.consensus_catalog import (
    NoReferenceFilterError, consensus_obs_token, consensus_path,
    promote_reference_filter, reference_filter, reference_filter_rank)


def _filters_on_disk(basepath, token):
    """Filters whose consensus catalog exists, from the filenames."""
    pattern = consensus_path(basepath, '*', obs_token=token)
    out = []
    for path in sorted(glob.glob(pattern)):
        name = os.path.basename(path)
        stem = name[:-len('_consensus.fits')]
        if token and stem.endswith(token):
            stem = stem[:-len(token)]
        if stem and not stem.startswith('jwst_reference'):
            out.append(stem.upper())
    return out


def _filters_from_registry(field_name, proposal_id=None, obs=None):
    """The field's filters from ``fields.yaml``.

    Scoped to one proposal when given: a target observed by two proposals
    (ngc6334 6778/7213) has two independent reference catalogs, not one.

    Scoped further to ``obs``'s instrument when that is unambiguous.  The
    registry holds ONE flat filter list per proposal, so without this the
    ranking can pick a band the observation never took -- and the consensus it
    then looks for is written under the observation's own token, so the file
    cannot exist.  sickle is the case: obs 007 is NIRCam (F210M), obs 001-002 is
    MIRI (F770W), and neither token carries the other's bands.
    """
    from jwst_gc_pipeline import fields as fields_mod
    if obs and proposal_id:
        scoped = fields_mod.filters_for_observation(field_name, proposal_id, obs)
        if scoped:
            return scoped
    out = []
    for instrument in ('nircam', 'miri', 'niriss'):
        by_proposal = fields_mod.obs_filters(instrument).get(field_name, {})
        for proposal, filters in by_proposal.items():
            if proposal_id and str(proposal) != str(proposal_id):
                continue
            out.extend(str(f).upper() for f in filters)
    return sorted(set(out))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--basepath', required=True,
                   help='the field directory holding catalogs/')
    p.add_argument('--field-name', default=None,
                   help='fields.yaml key, to read the filter list from the registry')
    p.add_argument('--filter', dest='filters', action='append', default=[],
                   help='explicit filter (repeatable); overrides the registry')
    p.add_argument('--from-disk', action='store_true',
                   help='rank only the filters that already have a consensus catalog')
    p.add_argument('--proposal-id', default=None)
    p.add_argument('--obsid', default=None)
    p.add_argument('--dry-run', action='store_true',
                   help='report the pick without writing')
    args = p.parse_args(argv)

    token = consensus_obs_token(args.proposal_id, args.obsid)

    if args.filters:
        filters, source = [f.upper() for f in args.filters], 'command line'
    elif args.from_disk:
        filters, source = _filters_on_disk(args.basepath, token), 'consensus catalogs on disk'
    elif args.field_name:
        # The obs token is what `promote_reference_filter` resolves the
        # chosen band's consensus under, so the ranking must see the same
        # observation.
        filters = _filters_from_registry(args.field_name, args.proposal_id,
                                         obs=args.obsid)
        source = 'fields.yaml'
    else:
        p.error('need one of --filter, --from-disk or --field-name')

    if not filters:
        print(f'no filters found from {source}', file=sys.stderr)
        return 1

    try:
        chosen = reference_filter(filters)
    except NoReferenceFilterError as ex:
        print(f'{ex}', file=sys.stderr)
        return 1

    print(f'filters ({source}): ' + ', '.join(
        f'{f} {reference_filter_rank(f):.3f}' if _rankable(f) else f'{f} (unrankable)'
        for f in sorted(filters, key=_rank_or_inf)))
    print(f'reference filter: {chosen}')

    if args.from_disk:
        print('NOTE: --from-disk ranks only what is already written.  If a '
              'better-ranked filter\'s m2 has not run yet, this promotes the '
              'best filter AVAILABLE, not the best filter of the field.')

    if args.dry_run:
        print(f'--dry-run: would write '
              f'{os.path.join(args.basepath, "catalogs")}/jwst_reference{token}_consensus.fits')
        return 0

    chosen, path = promote_reference_filter(args.basepath, filters,
                                            obs_token=token)
    print(f'wrote {path} (REFFILT={chosen})')
    return 0


def _rankable(name):
    try:
        reference_filter_rank(name)
    except ValueError:
        return False
    return True


def _rank_or_inf(name):
    try:
        return reference_filter_rank(name)
    except ValueError:
        return float('inf')


if __name__ == '__main__':
    raise SystemExit(main())
