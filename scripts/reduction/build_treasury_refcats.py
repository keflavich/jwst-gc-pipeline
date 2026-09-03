#!/usr/bin/env python
"""Build ONE reference catalog PER TILE for the GC Treasury (program 10678).

Why per tile
------------
10678 is 139 pointings tiling the inner Galactic Centre.  Measured from the
planned MAST footprints (2026-09-03): the tile centres span l = -0.573..+0.705,
b = -0.138..+0.576 -- 1.28 x 0.71 degrees -- and neighbouring centres sit
2.17-3.78' apart.  One catalog cannot cover that, and handing a tile a catalog
built for a different pointing is not a degraded reference, it is the wrong sky:
that is how gc2211 o023 took a -9.28" per-exposure "correction" from o028's
catalog (``astrometry_utils.pick_refcat``).

So every observation gets its own file, stamped with its observation number:

    catalogs/gaia_virac2_refcat_epoch<tag>_o<NNN>.fits

which is the token ``pick_refcat`` matches on, and which both paths that
actually MEASURE a tie already read by globbing ``catalogs/`` --
``photometry.cataloging._astrom_checkpoint_refcat`` (the m2 checkpoint) and
``reduction.bulk_offset_step0.refcat_for_frame``.  Neither consults
fields.yaml, so a tile whose file is on disk is picked up with no registry
edit; the registry entry (``--emit-registry``) is what the REDUCE reads, for
provenance and for its own missing-file refusal.

The cone covers BOTH apertures
------------------------------
Every 10678 visit is NIRCam F212N+F480M prime with MIRI F770W in parallel, and
the MIRI aperture is far off the prime.  From the planned ``s_region``
polygons, identical for all 139 tiles:

    NIRCam prime footprint radius, about its own centre      3.21'
    NIRCam prime centre -> MIRI parallel centre              7.79'
    farthest MIRI vertex from the NIRCam prime centre        9.00'
    radius about the JOINT centre that covers both           7.09'

The per-tile builder's default ``--radius 0.1`` deg = 6.0' about the prime
centre therefore contains NO MIRI sky at all -- the nearest MIRI vertex is
6.6' out.  This driver instead centres each cone on the JOINT (NIRCam+MIRI)
footprint centre and sets the radius from the footprints themselves plus
``--margin-arcmin``, so one catalog serves both instruments of a tile.  A tile
whose ``s_region`` is missing falls back to the MAST target position with
``--fallback-radius``.

Usage
-----
    python scripts/reduction/build_treasury_refcats.py --observations 088-139 \
        --epoch 2026.69 --dry-run

``--dry-run`` prints the exact per-tile command and builds nothing.  Without it
the tiles are built one subprocess at a time; this is a long query against
VizieR (VIRAC2 II/387 + Gaia DR3), so run it as a batch job, not on a login
node.  A tile whose catalog is already on disk is skipped unless ``--force``.
"""
import argparse
import glob
import os
import subprocess
import sys

import numpy as np

from jwst_gc_pipeline import fields as field_registry
from jwst_gc_pipeline.reduction.build_gaia_virac2_refcat_byquery import (
    refcat_filename)

#: The GC Treasury.  Named here rather than defaulted in-line so the one place
#: the program number appears is greppable.
TREASURY_PROPOSAL = '10678'
TREASURY_FIELD = 'gc-treasury'

#: Extra radius beyond the planned footprints, in arcmin.  The polygons this
#: driver reads are the PLAN; the executed pointing moves by the dither pattern
#: and the achieved roll, and a reference catalog that stops at the footprint
#: edge loses the tie for the outermost exposures.
DEFAULT_MARGIN_ARCMIN = 1.5

#: Used only for a tile MAST gives no ``s_region`` for.  0.16 deg = 9.6',
#: which covers the 9.00' worst-case MIRI vertex about the NIRCam prime centre
#: (the target position MAST reports) with the same margin.
DEFAULT_FALLBACK_RADIUS_DEG = 0.16


class TileGeometryError(RuntimeError):
    """A tile's pointing could not be determined from the MAST rows."""


def parse_observations(spec):
    """``'088-139'``/``'001,005'``/``'001,088-090'`` -> zero-padded obsids.

    Returns them sorted and de-duplicated.  ``None`` or ``'all'`` returns
    ``None``, meaning "every observation the query found".  A range is
    INCLUSIVE at both ends: ``088-139`` is the 52 scheduled visits, 52 tiles.
    """
    if spec is None or str(spec).strip().lower() == 'all':
        return None
    out = set()
    for piece in str(spec).split(','):
        piece = piece.strip()
        if not piece:
            continue
        if '-' in piece:
            lo, hi = piece.split('-', 1)
            lo, hi = lo.strip(), hi.strip()
            if not (lo.isdigit() and hi.isdigit()):
                raise ValueError(
                    f'--observations range {piece!r} is not two numbers')
            if int(hi) < int(lo):
                raise ValueError(
                    f'--observations range {piece!r} runs backwards')
            out.update(f'{n:03d}' for n in range(int(lo), int(hi) + 1))
        else:
            if not piece.isdigit():
                raise ValueError(
                    f'--observations entry {piece!r} is not a number')
            out.add(f'{int(piece):03d}')
    if not out:
        raise ValueError(f'--observations {spec!r} selects nothing')
    return sorted(out)


def _polygon_vertices(s_region):
    """``(ra, dec)`` arrays from a MAST ``POLYGON`` string, or ``None``."""
    if s_region is None:
        return None
    text = str(s_region)
    if 'POLYGON' not in text.upper():
        return None
    numbers = []
    for token in text.split():
        # skip the shape word and an optional frame word ('ICRS')
        try:
            numbers.append(float(token))
        except ValueError:
            continue
    if len(numbers) < 6 or len(numbers) % 2:
        return None
    values = np.asarray(numbers, dtype=float).reshape(-1, 2)
    return values[:, 0], values[:, 1]


def _unit_vectors(ra_deg, dec_deg):
    ra = np.radians(np.asarray(ra_deg, dtype=float))
    dec = np.radians(np.asarray(dec_deg, dtype=float))
    return np.stack([np.cos(dec) * np.cos(ra),
                     np.cos(dec) * np.sin(ra),
                     np.sin(dec)], axis=-1)


def enclosing_cone(ra_deg, dec_deg):
    """``(ra, dec, radius_deg)`` of a cone about the mean direction.

    The mean of the unit vectors, renormalised, then the largest angle from it
    to any input point.  Pure geometry, so it is exercised by the tests without
    a network call.
    """
    vectors = _unit_vectors(ra_deg, dec_deg)
    if len(vectors) == 0:
        raise TileGeometryError('no positions to enclose')
    centre = vectors.mean(axis=0)
    norm = np.linalg.norm(centre)
    if not np.isfinite(norm) or norm <= 0:
        raise TileGeometryError('positions average to no direction')
    centre = centre / norm
    cosines = np.clip(vectors @ centre, -1.0, 1.0)
    radius = float(np.degrees(np.arccos(cosines).max()))
    ra = float(np.degrees(np.arctan2(centre[1], centre[0])) % 360.0)
    dec = float(np.degrees(np.arcsin(np.clip(centre[2], -1.0, 1.0))))
    return ra, dec, radius


class Tile:
    """One 10678 observation: where its cone goes and what it is called."""

    def __init__(self, obsid, target, ra, dec, radius_deg, epoch=None,
                 instruments=()):
        self.obsid = obsid
        self.target = target
        self.ra = ra
        self.dec = dec
        self.radius_deg = radius_deg
        self.epoch = epoch
        self.instruments = tuple(sorted(instruments))

    def __repr__(self):
        return (f'Tile({self.obsid}, {self.target}, ra={self.ra:.5f}, '
                f'dec={self.dec:.5f}, radius={self.radius_deg:.4f})')


def _epoch_from_tmin(t_min_values):
    """Observation epoch in jyear from the earliest finite ``t_min`` (MJD)."""
    finite = [float(v) for v in t_min_values
              if v is not None and np.isfinite(float(v))]
    if not finite:
        return None
    from astropy.time import Time
    return float(Time(min(finite), format='mjd').jyear)


def tiles_from_table(table, observations=None, proposal=TREASURY_PROPOSAL,
                     margin_arcmin=DEFAULT_MARGIN_ARCMIN,
                     fallback_radius_deg=DEFAULT_FALLBACK_RADIUS_DEG):
    """Group MAST exposure-level rows into per-observation tiles.

    Every instrument's rows for an observation go into ONE cone -- that is the
    point: the MIRI parallel is 7.79' off the NIRCam prime, so a cone fitted to
    the prime alone leaves the parallel with no reference.  The observation
    number is read off ``obs_id`` (``jw10678088001_...`` -> ``088``), which is
    where it lives for planned rows; ``t_min`` is NaN until a visit executes.
    """
    rows = {}
    for row in table:
        obs_id = str(row['obs_id'])
        prefix = f'jw{int(proposal):05d}'
        if not obs_id.startswith(prefix):
            continue
        obsid = obs_id[len(prefix):len(prefix) + 3]
        if not obsid.isdigit():
            continue
        if observations is not None and obsid not in observations:
            continue
        rows.setdefault(obsid, []).append(row)

    tiles = []
    for obsid in sorted(rows):
        group = rows[obsid]
        ras, decs = [], []
        for row in group:
            vertices = _polygon_vertices(row['s_region'])
            if vertices is None:
                continue
            ras.extend(vertices[0])
            decs.extend(vertices[1])
        if ras:
            ra, dec, radius = enclosing_cone(ras, decs)
            radius += margin_arcmin / 60.0
        else:
            # No footprint published: fall back to the target position with a
            # radius wide enough for the parallel aperture.
            centres = [(float(r['s_ra']), float(r['s_dec'])) for r in group
                       if np.isfinite(float(r['s_ra']))
                       and np.isfinite(float(r['s_dec']))]
            if not centres:
                raise TileGeometryError(
                    f'observation {obsid} has neither an s_region footprint '
                    f'nor a finite s_ra/s_dec; nothing says where to query')
            ra, dec, _ = enclosing_cone([c[0] for c in centres],
                                        [c[1] for c in centres])
            radius = float(fallback_radius_deg)
        targets = sorted({str(r['target_name']) for r in group})
        instruments = {str(r['instrument_name']).split('/')[0].lower()
                       for r in group}
        tiles.append(Tile(obsid=obsid, target=targets[0], ra=ra, dec=dec,
                          radius_deg=radius,
                          epoch=_epoch_from_tmin([r['t_min'] for r in group]),
                          instruments=instruments))
    return tiles


def query_tiles(proposal=TREASURY_PROPOSAL, observations=None, **kwargs):
    """Tile geometry straight from MAST.  Network call; not unit-tested."""
    from astroquery.mast import Observations
    table = Observations.query_criteria(proposal_id=str(proposal))
    if len(table) == 0:
        raise RuntimeError(f'MAST returned no rows for proposal {proposal}')
    return tiles_from_table(table, observations=observations,
                            proposal=proposal, **kwargs)


def existing_refcat(base, obsid):
    """Any epoch's refcat already built for this observation, or ``None``.

    Matched by TOKEN and not by epoch tag: the tag records when the tile was
    observed, and a rerun that recomputes the epoch must not build a second
    catalog for the same sky under a new name.
    """
    token = f'{int(str(obsid).lstrip("oO")):03d}'
    found = sorted(glob.glob(os.path.join(
        str(base), 'catalogs', f'gaia_virac2_refcat_epoch*_o{token}.fits')))
    return found[-1] if found else None


def build_command(tile, base, epoch, python=sys.executable,
                  min_ref_density=None):
    """The per-tile ``build_gaia_virac2_refcat_byquery`` command line."""
    if epoch is None:
        raise ValueError(
            f'observation {tile.obsid} has no epoch: MAST reports t_min NaN '
            f'(the visit has not executed), so pass --epoch.  The 52 visits '
            f'scheduled 2026-09-10..13 are epoch ~2026.69.')
    cmd = [python, '-m',
           'jwst_gc_pipeline.reduction.build_gaia_virac2_refcat_byquery',
           '--base', str(base),
           '--epoch', f'{float(epoch):.4f}',
           '--ra', f'{tile.ra:.6f}',
           '--dec', f'{tile.dec:.6f}',
           '--radius', f'{tile.radius_deg:.4f}',
           '--obs-token', tile.obsid]
    if min_ref_density is not None:
        cmd += ['--min-ref-density', repr(float(min_ref_density))]
    return cmd


def registry_block(tiles, epochs):
    """The fields.yaml ``reference_catalog:`` block for the tiles built.

    fields.yaml registers a catalog per observation; it has no templating, so
    the file a tile ends up with is written back here rather than guessed
    ahead of the build (the epoch tag is the tile's own observation date).
    """
    lines = ['        reference_catalog:']
    for tile in tiles:
        epoch = epochs[tile.obsid]
        tag = f'{float(epoch):.2f}'
        name = refcat_filename(tag, tile.obsid)
        lines.append(f"          '{tile.obsid}': catalogs/{name}")
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--proposal', default=TREASURY_PROPOSAL)
    ap.add_argument('--field', default=TREASURY_FIELD,
                    help='registry field name supplying the base path')
    ap.add_argument('--base', default=None,
                    help='target basepath; default: the field registry\'s')
    ap.add_argument('--observations', default=None, metavar='SPEC',
                    help="'088-139', '001,005', or 'all' (default)")
    ap.add_argument('--epoch', type=float, default=None,
                    help='observation epoch (jyear) for tiles MAST gives no '
                         't_min for.  A tile whose t_min IS published uses '
                         'its own value and ignores this.')
    ap.add_argument('--margin-arcmin', type=float,
                    default=DEFAULT_MARGIN_ARCMIN,
                    help='extra radius beyond the planned footprints')
    ap.add_argument('--fallback-radius', type=float,
                    default=DEFAULT_FALLBACK_RADIUS_DEG,
                    help='cone radius (deg) for a tile with no s_region')
    ap.add_argument('--min-ref-density', type=float, default=None,
                    help='passed through to the per-tile builder')
    ap.add_argument('--force', action='store_true',
                    help='rebuild a tile whose refcat is already on disk')
    ap.add_argument('--dry-run', action='store_true',
                    help='print what would be built and build nothing')
    ap.add_argument('--emit-registry', action='store_true',
                    help='print the fields.yaml reference_catalog block for '
                         'the selected tiles')
    ap.add_argument('--stop-on-error', action='store_true',
                    help='abort the batch on the first tile that fails '
                         '(default: report it and carry on)')
    args = ap.parse_args(argv)

    observations = parse_observations(args.observations)
    base = args.base or field_registry.basepath(args.field)
    tiles = query_tiles(proposal=args.proposal, observations=observations,
                        margin_arcmin=args.margin_arcmin,
                        fallback_radius_deg=args.fallback_radius)
    if observations is not None:
        missing = sorted(set(observations) - {t.obsid for t in tiles})
        if missing:
            print(f'WARNING: proposal {args.proposal} has no MAST rows for '
                  f'observation(s) {missing}', flush=True)

    print(f'{len(tiles)} tile(s) of proposal {args.proposal}; base {base}',
          flush=True)

    epochs, todo, skipped = {}, [], []
    for tile in tiles:
        epochs[tile.obsid] = tile.epoch if tile.epoch is not None else args.epoch
        already = existing_refcat(base, tile.obsid)
        if already and not args.force:
            skipped.append((tile, already))
        else:
            todo.append(tile)

    for tile, path in skipped:
        print(f'  o{tile.obsid} {tile.target}: already built '
              f'({os.path.basename(path)}); --force to rebuild', flush=True)

    failures = []
    for tile in todo:
        cmd = build_command(tile, base, epochs[tile.obsid],
                            min_ref_density=args.min_ref_density)
        span = tile.radius_deg * 60.0
        print(f'  o{tile.obsid} {tile.target}: cone '
              f'({tile.ra:.5f}, {tile.dec:.5f}) r={span:.2f}\' covering '
              f'{"+".join(tile.instruments)}\n    ' + ' '.join(cmd),
              flush=True)
        if args.dry_run:
            continue
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures.append((tile.obsid, result.returncode))
            print(f'  o{tile.obsid} FAILED (rc={result.returncode})',
                  flush=True)
            if args.stop_on_error:
                break

    if args.emit_registry:
        buildable = [t for t in tiles if epochs[t.obsid] is not None]
        print('\n# fields.yaml: fields.gc-treasury.observations.'
              f"'{args.proposal}'")
        print(registry_block(buildable, epochs))

    if args.dry_run:
        print(f'\n--dry-run: {len(todo)} tile(s) would be built, '
              f'{len(skipped)} already on disk.  Nothing was queried.',
              flush=True)
        return 0
    if failures:
        print(f'\n{len(failures)} tile(s) failed: '
              f'{[o for o, _ in failures]}', flush=True)
        return 1
    print(f'\nbuilt {len(todo)} tile(s), skipped {len(skipped)}', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
