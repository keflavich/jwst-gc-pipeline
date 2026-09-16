#!/usr/bin/env python
"""Build the slow-panner page and its tour.

The tour visits only pointings the coadd CONTAINS, read from the per-field
layers it was built from. Taking the list from the observing schedule instead
would pan across tiles that are observed but not yet rendered, and blank sky in
a viewer that moves on its own reads as a broken page rather than as a tile
that has not been built.

Ordering is nearest-neighbour from the westernmost tile, then closed into a
loop. A survey tiled along the Galactic plane is close to a line, so the greedy
path is near-optimal there; what matters more is that consecutive stops are
adjacent, since the pan crosses the gap between them in real time.
"""
import argparse
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '..', '..'))

from jwst_gc_pipeline.quicklook import panner        # noqa: E402

DEFAULT_HIPS_DIR = '/orange/adamginsburg/jwst/gc-treasury/pngs'
DEFAULT_FOOTPRINTS = '/orange/adamginsburg/jwst/monitor/footprints.json'
DEFAULT_OUT = '/orange/adamginsburg/jwst/releases/site'

#: Per-field layers of the flavour the coadd is made of.
_LAYER = re.compile(r'^GCTreasury_o(?P<obs>\d{3})_RGB_480-mean-212_vminmax_hips$')


def layers_in_the_coadd(hips_dir):
    """``{'o127', ...}`` -- the pointings with a rendered vminmax layer."""
    out = set()
    if not os.path.isdir(hips_dir):
        return out
    for name in os.listdir(hips_dir):
        match = _LAYER.match(name)
        if match and os.path.isfile(os.path.join(hips_dir, name, 'properties')):
            out.add(f"o{match.group('obs')}")
    return out


def centres(footprints_path, wanted):
    """``[{'id', 'label', 'ra', 'dec'}]`` for the wanted pointings."""
    with open(footprints_path) as fh:
        data = json.load(fh)
    found = {}
    for entry in data.get('observed', []) + data.get('planned', []):
        number = str(entry.get('number', '')).strip()
        if not number.isdigit():
            continue
        obsid = f'o{int(number):03d}'
        if obsid not in wanted or obsid in found:
            continue
        ra, dec = entry.get('ra'), entry.get('dec')
        if ra is None or dec is None:
            continue
        found[obsid] = {'id': obsid, 'label': entry.get('target') or obsid,
                        'ra': float(ra), 'dec': float(dec)}
    return found


def separation(a, b):
    """Angular separation in arcsec."""
    ra1, dec1 = math.radians(a['ra']), math.radians(a['dec'])
    ra2, dec2 = math.radians(b['ra']), math.radians(b['dec'])
    dot = (math.sin(dec1) * math.sin(dec2)
           + math.cos(dec1) * math.cos(dec2) * math.cos(ra1 - ra2))
    return math.degrees(math.acos(max(-1.0, min(1.0, dot)))) * 3600


def order_tour(stops):
    """Greedy nearest-neighbour from the westernmost stop."""
    remaining = list(stops)
    path = [min(remaining, key=lambda s: s['ra'])]
    remaining.remove(path[0])
    while remaining:
        here = path[-1]
        nxt = min(remaining, key=lambda s: separation(here, s))
        remaining.remove(nxt)
        path.append(nxt)
    return path


def build(args):
    wanted = layers_in_the_coadd(args.hips_dir)
    if not wanted:
        raise SystemExit(f'no vminmax per-field layers under {args.hips_dir}')
    found = centres(args.footprints, wanted)
    missing = sorted(wanted - set(found))
    if missing:
        # Named rather than dropped silently: a layer with no footprint entry
        # is imagery the tour cannot reach, which is worth knowing.
        print(f'note: {len(missing)} layer(s) have no footprint centre and are '
              f'not on the tour: {", ".join(missing)}')
    if not found:
        raise SystemExit('no pointing has both a rendered layer and a centre')

    path = order_tour(list(found.values()))
    legs = [separation(path[i], path[(i + 1) % len(path)])
            for i in range(len(path))]
    # A leg longer than the threshold is a CUT, not a pan. 10678 is not one
    # contiguous block -- the o040-o042 group sits about 55' from the rest --
    # and panning that gap at 2"/s is 27 minutes of empty sky, which is most
    # of a loop spent showing nothing. The page cuts across it instead, the
    # way any tour of separated fields would.
    for i, span in enumerate(legs):
        path[i]['jump'] = span > args.max_leg * 60
    panned = [span for span, stop in zip(legs, path) if not stop['jump']]
    if not panned:
        # Every leg a cut is a tour with nothing to watch, and the page has no
        # defence against it: its skip loop advances until it finds a leg to
        # pan and would spin forever inside requestAnimationFrame, freezing the
        # tab with no error. Refuse BEFORE writing -- the summary line used to
        # raise on max() of an empty list, after both files were on disk, so a
        # tour nobody could watch was published and the crash looked like the
        # build had failed to produce one.
        raise SystemExit(
            f'every leg is longer than --max-leg={args.max_leg}\' , so the '
            f'tour would cut between all {len(path)} stops and pan across '
            f'none of them. Raise --max-leg.')
    total = sum(panned)
    tour = {'survey': args.survey, 'fov': args.fov, 'rate': args.rate,
            'stops': path}
    os.makedirs(args.out, exist_ok=True)
    data_path = os.path.join(args.out, panner.DATA_FILE)
    with open(data_path, 'w') as fh:
        json.dump(tour, fh)
    page_path = os.path.join(args.out, panner.PAGE_FILE)
    with open(page_path, 'w') as fh:
        fh.write(panner.render_page(panner.DATA_FILE))

    minutes = total / args.rate / 60
    cuts = sum(1 for stop in path if stop['jump'])
    print(f'{len(path)} stops, {total / 60:.1f}\' panned, '
          f'{minutes:.0f} min per loop at {args.rate}"/s')
    print(f'{cuts} cut(s) over gaps longer than {args.max_leg:.0f}\'; '
          f'longest panned leg {max(panned) / 60:.1f}\' '
          f'({max(panned) / args.rate / 60:.1f} min)')
    print(f'wrote {page_path} and {data_path}')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hips-dir', default=DEFAULT_HIPS_DIR)
    ap.add_argument('--footprints', default=DEFAULT_FOOTPRINTS)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--survey', default=panner.SURVEY_URL)
    ap.add_argument('--fov', type=float, default=panner.DEFAULT_FOV,
                    help='degrees across the viewport')
    ap.add_argument('--rate', type=float, default=panner.DEFAULT_RATE,
                    help='arcsec of sky per second')
    ap.add_argument('--max-leg', type=float, default=5.0,
                    help='arcmin; a longer gap is cut across rather than panned')
    return build(ap.parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
