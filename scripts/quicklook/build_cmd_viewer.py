#!/usr/bin/env python
"""Build the colour-magnitude explorer: pointings on sky, F212N vs F212N-F480M.

    python scripts/quicklook/build_cmd_viewer.py \
        --out /orange/adamginsburg/jwst/releases/site

Writes ``cmd_explorer.html`` and ``cmd_explorer_data.json`` into ``--out``.
``make_webpage.py`` links the page from the release index when it finds the
HTML there, so the two can be rebuilt independently and in either order.

WHAT IT COSTS, AND WHY THERE IS A CACHE

The expensive part is reading catalogs and cross-matching the two filters:
about 9 s and ~200 MB for a pointing with 2x10^5 rows per band.  Ten pointings
is a minute and a half; the programme is 139, so a full rebuild at completion
would be ~20 minutes and would repeat every previous tile's work.  Catalogs are
immutable once written -- a re-reduction writes a NEW file, at a new merge
stage -- so the matched (colour, magnitude) pairs are cached per pointing,
keyed on the input paths and their mtimes.  A rebuild after four new tiles land
reads four catalogs, not forty.  ``--no-cache`` forces the work.

Everything after the cross-match is cheap: binning 10^7 points onto a hex grid
is a couple of seconds of numpy, and the JSON that comes out is a few hundred
kilobytes.  The browser does no binning at all.

THE AXES ARE PINNED ON PURPOSE

``--extent`` fixes the plot limits across rebuilds.  Without it the limits come
from percentiles of whatever is loaded, so the same tile's diagram would shift
between builds as new tiles change the pooled distribution, and two versions of
the page could not be compared.  The default derives them and PRINTS them, so
the first build tells you what to pin.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from jwst_gc_pipeline.quicklook import catalogs as C          # noqa: E402
from jwst_gc_pipeline.quicklook import cmdview, hexbin        # noqa: E402

DEFAULT_CATALOGS = '/orange/adamginsburg/jwst/gc-treasury/catalogs'
DEFAULT_MAST = '/orange/adamginsburg/jwst/gc-treasury/mastDownload'
DEFAULT_FOOTPRINTS = '/orange/adamginsburg/jwst/monitor/footprints.json'
DEFAULT_CACHE = '/orange/adamginsburg/jwst/quicklook/cmd_cache'
DEFAULT_OUT = '/orange/adamginsburg/jwst/releases/site'


def cache_key(paths, match_arcsec):
    """Identity of one pointing's cached pairs: the inputs AND how they were paired.

    Content-addressing the catalogs themselves would mean hashing 50 MB per
    pointing, which is most of the cost we are avoiding.  A re-reduction writes
    a new filename anyway, so path+mtime turns over whenever the data does.

    ``match_arcsec`` is in the key because the cache holds MATCHED PAIRS, not
    catalogs: keyed on the inputs alone, a rebuild at a different radius reused
    pairs made at the old one and the page then printed the new radius in its
    own provenance block over data matched at the old one.  The band names are
    in the paths, so they need no separate entry.
    """
    parts = [f'match={float(match_arcsec):.4f}']
    for band in sorted(paths):
        path = Path(paths[band])
        parts.append(f'{band}:{path}:{os.path.getmtime(path):.0f}')
    return hashlib.sha256('|'.join(parts).encode()).hexdigest()[:16]


def load_pointing(obsid, entry, cache_dir, match_arcsec, allow_network=True,
                  use_cache=True):
    """``(colour, magnitude)`` for one pointing, from cache when it is current."""
    key = cache_key(entry['paths'], match_arcsec)
    cached = Path(cache_dir) / f'{obsid}_{key}.npz'
    if use_cache and cached.exists():
        with np.load(cached) as npz:
            return npz['colour'], npz['mag'], True
    t0 = time.time()
    colour, mag = C.pointing_cmd(entry['paths'], max_sep_arcsec=match_arcsec,
                                 allow_network=allow_network)
    print(f'  {obsid}: {len(colour):>7,} pairs from {entry["source"]} catalogs '
          f'in {time.time() - t0:.1f} s')
    if use_cache:
        cached.parent.mkdir(parents=True, exist_ok=True)
        # Written to a temp name and moved, so a build killed mid-write does not
        # leave a truncated .npz that the next build would read as good.
        # The temp name has to END in .npz: savez_compressed appends the
        # extension to anything that does not, so a `.npz.tmp` target writes
        # `.npz.tmp.npz` and the rename then fails on a missing file.
        tmp = cached.with_name(cached.stem + '.tmp.npz')
        np.savez_compressed(tmp, colour=colour.astype(np.float32),
                            mag=mag.astype(np.float32))
        tmp.replace(cached)
    return colour, mag, False


def _module_of(path):
    """``'merged'`` / ``'nrca'`` / ``'nrcb'`` for a pipeline catalog, else None."""
    m = C._JICAMA.match(Path(path).name)
    return m.group('module') if m else None


def load_footprints(path):
    """``{obsid: {'polys': [...], 'label': ..., 'centre': [ra, dec]}}``.

    Read from the monitor's own footprints file, so the two pages draw the same
    tiles in the same places.  The observation id is the tile number: APT
    numbers 10678's observations to match the ``GC_<n>`` target names, which is
    what ``o127`` and ``GC_127`` both are.
    """
    if not Path(path).exists():
        print(f'note: no footprints at {path}; pointings will be listed but '
              f'not outlined on the sky')
        return {}
    data = json.loads(Path(path).read_text())
    out = {}
    for entry in data.get('observed', []) + data.get('planned', []):
        number = str(entry.get('number', '')).strip()
        if not number.isdigit():
            continue
        obsid = f'o{int(number):03d}'
        polys = list(entry.get('nircam') or [])
        if not polys or obsid in out:
            continue
        out[obsid] = {'polys': polys, 'label': entry.get('target') or obsid,
                      'centre': [entry.get('ra'), entry.get('dec')]}
    return out


def build(args):
    jicama = C.find_jicama(args.catalogs)
    mast = C.find_mast(args.mast)
    chosen = C.choose_sources(jicama, mast)
    waiting = C.incomplete_pointings(args.catalogs, args.mast)
    if not chosen:
        raise SystemExit(
            f'no pointing has catalogs in both {C.BLUE_BAND.upper()} and '
            f'{C.RED_BAND.upper()}\n'
            f'  pipeline catalogs: {args.catalogs}\n'
            f'  MAST catalogs:     {args.mast}\n'
            f'  short a band:      {waiting or "none"}')
    if args.pointings:
        wanted = set(args.pointings)
        chosen = {o: e for o, e in chosen.items() if o in wanted}

    print(f'{len(chosen)} pointing(s): '
          f'{sum(1 for e in chosen.values() if e["source"] == "jicama")} from the '
          f'pipeline, {sum(1 for e in chosen.values() if e["source"] == "mast")} '
          f'from MAST')

    footprints = load_footprints(args.footprints)
    loaded, reused, empty = {}, 0, []
    t0 = time.time()
    for obsid, entry in chosen.items():
        colour, mag, from_cache = load_pointing(
            obsid, entry, args.cache, args.match_arcsec,
            allow_network=not args.offline, use_cache=not args.no_cache)
        reused += bool(from_cache)
        if len(colour):
            loaded[obsid] = (colour, mag)
        else:
            # Both bands present and nothing paired: a real condition (a failed
            # cross-match, an astrometric offset between the two filters) and
            # not the same thing as a missing band, so it is named rather than
            # dropped into the same silence.
            empty.append(obsid)
    print(f'loaded {len(loaded)} pointing(s) in {time.time() - t0:.1f} s '
          f'({reused} from cache)')
    if not loaded:
        raise SystemExit('every pointing cross-matched to zero pairs')

    all_colour = np.concatenate([c for c, _ in loaded.values()])
    all_mag = np.concatenate([m for _, m in loaded.values()])
    if args.extent:
        extent = tuple(args.extent)
    else:
        extent = hexbin.percentile_extent(all_colour, all_mag)
        print(f'derived extent (pin it with --extent to keep successive builds '
              f'comparable):\n  --extent {extent[0]:.3f} {extent[1]:.3f} '
              f'{extent[2]:.3f} {extent[3]:.3f}')
    grid = hexbin.Grid(*extent, nx=args.nx)

    t0 = time.time()
    all_cells, all_max, all_n = grid.bin(all_colour, all_mag)
    print(f'binned {len(all_colour):,} stars into {len(all_cells):,} cells '
          f'in {time.time() - t0:.1f} s (peak {all_max:,})')

    fields = []
    for obsid in sorted(loaded):
        colour, mag = loaded[obsid]
        cells, peak, n_in = grid.bin(colour, mag)
        foot = footprints.get(obsid)
        if foot is None:
            # Listed in the table and hoverable there, with no outline on the
            # sky.  Drawing a guessed rectangle would put a wrong footprint on a
            # page whose whole point is which tile you are looking at.
            print(f'note: {obsid} is not in the footprints file; it is listed '
                  f'but not outlined')
            foot = {'polys': None, 'label': f'GC_{int(obsid[1:])}',
                    'centre': [None, None]}
        polys = foot['polys']
        centre = foot['centre']
        # Which NIRCam module(s) the photometry came from.  Four of the ten
        # live pointings resolve to a single-module catalog because no
        # `merged` file exists for them yet, and one module is ~60% of a tile
        # (o127: nrca 133k rows vs merged 217k).  Ranking cannot fix that --
        # the wider file is absent, not out-ranked -- so the page has to SAY
        # it rather than label 60% of the sky `GC_128` and stop there.
        modules = sorted({_module_of(pth) for pth in
                          chosen[obsid]['paths'].values()} - {None})
        partial = bool(modules) and 'merged' not in modules
        fields.append({
            'id': obsid,
            'label': foot['label'],
            'source': chosen[obsid]['source'],
            'modules': modules,
            'partial': partial,
            'n': int(n_in),
            'max': int(peak),
            'cells': cells,
            'polys': polys or [],
            'centre': centre if centre and centre[0] is not None else None,
            'files': {b: Path(p).name for b, p in chosen[obsid]['paths'].items()},
        })

    centres = [f['centre'] for f in fields if f['centre']]
    data = {
        'built': time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()),
        'programme': args.programme,
        'bands': {'blue': C.BLUE_BAND, 'red': C.RED_BAND},
        'mag_system': 'vega',
        'match_arcsec': args.match_arcsec,
        'labels': {
            'x': f'{C.BLUE_BAND.upper()} − {C.RED_BAND.upper()} (Vega)',
            'y': f'{C.BLUE_BAND.upper()} (Vega)',
        },
        'grid': grid.to_json(),
        'all': {'cells': all_cells, 'max': all_max, 'n': int(all_n)},
        'fields': fields,
        'incomplete': waiting,
        'no_pairs': empty,
        'centre': ([float(np.average([c[0] for c in centres])),
                    float(np.average([c[1] for c in centres]))]
                   if centres else None),
        'fov': args.fov,
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_path = out_dir / 'cmd_explorer_data.json'
    html_path = out_dir / 'cmd_explorer.html'
    data_path.write_text(json.dumps(data, separators=(',', ':')))
    html_path.write_text(cmdview.render(data, data_path.name))
    print(f'wrote {html_path} ({html_path.stat().st_size / 1024:.0f} kB) and '
          f'{data_path} ({data_path.stat().st_size / 1024:.0f} kB)')
    partial = [f['id'] for f in fields if f['partial']]
    if partial:
        print('single-module (part of the tile, and the page says so): '
              + ', '.join(f"{o} [{'/'.join(f['modules'])}]"
                          for o in partial
                          for f in fields if f['id'] == o))
    if empty:
        print('both bands present but zero pairs matched (not plotted): '
              + ', '.join(empty))
    if waiting:
        print('short a band, so not plotted: ' +
              ', '.join(f'{o} needs {"/".join(b).upper()}'
                        for o, b in sorted(waiting.items())))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--catalogs', default=DEFAULT_CATALOGS,
                    help='pipeline ("jicama") catalog directory')
    ap.add_argument('--mast', default=DEFAULT_MAST,
                    help='MAST download tree, used per pointing where the '
                         'pipeline has no catalogs')
    ap.add_argument('--footprints', default=DEFAULT_FOOTPRINTS,
                    help="the monitor's footprints.json")
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--cache', default=DEFAULT_CACHE)
    ap.add_argument('--no-cache', action='store_true',
                    help='re-read and re-match every pointing')
    ap.add_argument('--offline', action='store_true',
                    help='use the cached SVO zeropoints without asking SVO')
    ap.add_argument('--nx', type=int, default=hexbin.DEFAULT_NX,
                    help='hexagons across the colour axis')
    ap.add_argument('--extent', type=float, nargs=4,
                    metavar=('CMIN', 'CMAX', 'MMIN', 'MMAX'),
                    help='pin the plot limits (colour then magnitude)')
    ap.add_argument('--match-arcsec', type=float, default=C.DEFAULT_MATCH_ARCSEC)
    ap.add_argument('--fov', type=float, default=1.4,
                    help='opening field of view, degrees')
    ap.add_argument('--programme', default='10678')
    ap.add_argument('--pointings', nargs='*',
                    help='restrict to these obsids (e.g. o127 o132)')
    return build(ap.parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
