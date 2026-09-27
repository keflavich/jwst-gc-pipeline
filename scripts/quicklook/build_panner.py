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

Every product in ``panner.PRODUCTS`` gets its own tour, since they cover
different sky (MIRI parallels sit several arcminutes from their NIRCam
pointings; the stars-subtracted layers exist only for finished catalogs). A
product with nothing to pan is left off the page's menu and named; the default
product failing stops the build before anything is written.
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
#: The local copy of the served HiPS directories, for the one product with no
#: per-field layers of its own (the MIRI + NIRCam composite): its tour keeps a
#: pointing only if this copy has a tile there.
DEFAULT_HIPS_ROOT = '/orange/adamginsburg/web/public/avm_images'

#: Pointings the tour never visits, even with a rendered layer.  o138 and
#: o139 are the same pair the survey mosaics leave out (EXCLUDE in the
#: gc-treasury mosaics recipe); the maintainer asked for the panner to skip
#: them too.
DEFAULT_EXCLUDE = ('o138', 'o139')

#: The per-field layer family the default product's coadd is made of.
DEFAULT_FAMILY = 'RGB_480-mean-212_vminmax'


class NoTour(Exception):
    """A product has nothing the page could pan across."""


def _layer_pattern(family):
    # A tile with only one module finished is rendered per module
    # (GCTreasury_o063_nrca_...) and goes into the coadd all the same.
    return re.compile(r'^GCTreasury_o(?P<obs>\d{3})_(?:nrc[ab]_)?'
                      + re.escape(family) + r'_hips$')


def _finished_layers(hips_dir, family):
    """``[(obsid, path)]`` for the family's layers that have ``properties``."""
    pattern = _layer_pattern(family)
    out = []
    if not os.path.isdir(hips_dir):
        return out
    for name in sorted(os.listdir(hips_dir)):
        match = pattern.match(name)
        path = os.path.join(hips_dir, name)
        if match and os.path.isfile(os.path.join(path, 'properties')):
            out.append((f"o{match.group('obs')}", path))
    return out


def layers_in_the_coadd(hips_dir, family=DEFAULT_FAMILY):
    """``{'o127', ...}`` -- the pointings with a finished layer of the family.

    ``reproject_to_hips`` writes ``properties`` last, so a layer without it is
    mid-build and not yet imagery."""
    return {obsid for obsid, _ in _finished_layers(hips_dir, family)}


def read_properties(path):
    """A HiPS ``properties`` file as a dict of strings."""
    props = {}
    with open(path) as fh:
        for line in fh:
            key, sep, value = line.partition('=')
            if sep and not key.lstrip().startswith('#'):
                props[key.strip()] = value.strip()
    return props


def hips_order(hips_dir, family, default=panner.DEFAULT_ORDER):
    """The HiPS order of the family's layers, read from the first one."""
    for _, path in _finished_layers(hips_dir, family):
        order = read_properties(os.path.join(path, 'properties')).get('hips_order')
        if order is not None:
            return int(order)
    return default


def _polygon_centre(value):
    """Mean direction of a footprint's vertices, or None.

    footprints.json stores each instrument's footprint as a JSON string of a
    list of polygons, each a list of ``[ra, dec]``."""
    if isinstance(value, str):
        value = json.loads(value)
    vertices = [v for polygon in (value or []) for v in polygon]
    if not vertices:
        return None
    x = y = z = 0.0
    for ra, dec in vertices:
        r, d = math.radians(ra), math.radians(dec)
        x += math.cos(d) * math.cos(r)
        y += math.cos(d) * math.sin(r)
        z += math.sin(d)
    ra = math.degrees(math.atan2(y, x)) % 360
    dec = math.degrees(math.atan2(z, math.hypot(x, y)))
    return ra, dec


def centres(footprints_path, wanted, centre='nircam'):
    """``{obsid: {'id', 'label', 'ra', 'dec'}}`` for the wanted pointings.

    ``centre='nircam'`` is the pointing itself; ``'miri'`` is the middle of
    the MIRI parallel's footprint, several arcminutes away from it."""
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
        if centre == 'nircam':
            ra, dec = entry.get('ra'), entry.get('dec')
        else:
            at = _polygon_centre(entry.get(centre))
            ra, dec = at if at else (None, None)
        if ra is None or dec is None:
            continue
        found[obsid] = {'id': obsid, 'label': entry.get('target') or obsid,
                        'ra': float(ra), 'dec': float(dec)}
    return found


def tile_checker(hips_root):
    """A function ``(ra, dec) -> bool``: does the HiPS at ``hips_root`` have a
    tile at its deepest order there?  Raises NoTour if it is not a HiPS."""
    props_path = os.path.join(hips_root, 'properties')
    if not os.path.isfile(props_path):
        raise NoTour(f'{hips_root} is not a HiPS, so there is no telling '
                     f'where it has imagery')
    # Only this product needs HEALPix, so the import waits until it is asked.
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy_healpix import lonlat_to_healpix

    props = read_properties(props_path)
    order = int(props['hips_order'])
    galactic = props.get('hips_frame', 'equatorial') == 'galactic'
    fmt = props.get('hips_tile_format', 'png').split()[0]

    def has_tile(ra, dec):
        coord = SkyCoord(ra * u.deg, dec * u.deg, frame='icrs')
        lon, lat = ((coord.galactic.l, coord.galactic.b) if galactic
                    else (coord.ra, coord.dec))
        ipix = int(lonlat_to_healpix(lon, lat, 2 ** order, order='nested'))
        tile = os.path.join(hips_root, f'Norder{order}',
                            f'Dir{ipix // 10000 * 10000}', f'Npix{ipix}.{fmt}')
        return os.path.isfile(tile)

    has_tile.order = order
    return has_tile


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


def product_tour(product, args, survey):
    """``(tour, panned_arcsec)`` for one product; raises NoTour."""
    key, family = product['key'], product['layers']
    wanted = layers_in_the_coadd(args.hips_dir, family)
    if not wanted:
        raise NoTour(f'no {family} per-field layers under {args.hips_dir}')
    excluded = sorted(wanted & set(args.exclude))
    if excluded:
        print(f'{key}: excluded from the tour: {", ".join(excluded)}')
    wanted -= set(args.exclude)
    found = centres(args.footprints, wanted, product.get('centre', 'nircam'))
    missing = sorted(wanted - set(found))
    if missing:
        # Named rather than dropped silently: a layer with no footprint entry
        # is imagery the tour cannot reach, which is worth knowing.
        print(f'{key}: note: {len(missing)} layer(s) have no footprint centre '
              f'and are not on the tour: {", ".join(missing)}')
    order = hips_order(args.hips_dir, family)
    if product.get('coverage'):
        has_tile = tile_checker(os.path.join(args.hips_root, product['hips']))
        order = has_tile.order
        bare = sorted(k for k, stop in found.items()
                      if not has_tile(stop['ra'], stop['dec']))
        if bare:
            print(f'{key}: no imagery at {len(bare)} pointing(s), left off '
                  f'its tour: {", ".join(bare)}')
        for k in bare:
            del found[k]
    if not found:
        raise NoTour('no pointing has both a rendered layer and a centre')

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
        raise NoTour(
            f'every leg is longer than --max-leg={args.max_leg}\' , so the '
            f'tour would cut between all {len(path)} stops and pan across '
            f'none of them. Raise --max-leg.')
    # A coarser HiPS is shown wider, and panned proportionally faster so the
    # sky crosses the screen at the same speed whichever product is up.
    fov = panner.fov_for_order(order, args.fov)
    tour = {'key': key, 'label': product['label'], 'note': product['note'],
            'survey': survey, 'fov': fov, 'rate': args.rate * fov / args.fov,
            'stops': path}
    return tour, panned


def build(args):
    chosen = [p for p in panner.PRODUCTS
              if args.products is None or p['key'] in args.products]
    if not chosen:
        raise SystemExit(f'none of --products {args.products} is a product; '
                         f'choose from {[p["key"] for p in panner.PRODUCTS]}')
    tours, stats = [], []
    for i, product in enumerate(chosen):
        survey = args.hips_base + product['hips'] + '/'
        if i == 0 and args.survey:
            survey = args.survey
        try:
            tour, panned = product_tour(product, args, survey)
        except NoTour as why:
            if i == 0:
                # The page opens on the first product; without it there is
                # no page, and nothing has been written yet.
                raise SystemExit(str(why)) from why
            print(f'{product["key"]}: left off the menu: {why}')
            continue
        tours.append(tour)
        stats.append(panned)

    default = tours[0]
    # The top level is the default product in the shape the page read before
    # the menu existed, so a page and a tour file deployed out of step still
    # pan.
    data = {'survey': default['survey'], 'fov': default['fov'],
            'rate': default['rate'], 'stops': default['stops'],
            'products': tours}
    os.makedirs(args.out, exist_ok=True)
    data_path = os.path.join(args.out, panner.DATA_FILE)
    with open(data_path, 'w') as fh:
        json.dump(data, fh)
    page_path = os.path.join(args.out, panner.PAGE_FILE)
    with open(page_path, 'w') as fh:
        fh.write(panner.render_page(panner.DATA_FILE))

    for tour, panned in zip(tours, stats):
        total = sum(panned)
        cuts = sum(1 for stop in tour['stops'] if stop['jump'])
        print(f'{tour["key"]}: {len(tour["stops"])} stops, '
              f'{total / 60:.1f}\' panned, '
              f'{total / tour["rate"] / 60:.0f} min per loop at '
              f'{tour["rate"]:g}"/s (fov {tour["fov"]:g} deg); '
              f'{cuts} cut(s) over gaps longer than {args.max_leg:.0f}\'; '
              f'longest panned leg {max(panned) / 60:.1f}\'')
    print(f'wrote {page_path} and {data_path}')
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--hips-dir', default=DEFAULT_HIPS_DIR)
    ap.add_argument('--footprints', default=DEFAULT_FOOTPRINTS)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--survey', default=None,
                    help='URL of the default product\'s HiPS (default: '
                         '--hips-base plus its directory)')
    ap.add_argument('--hips-base', default=panner.HIPS_BASE,
                    help='URL the products\' HiPS directories are served under')
    ap.add_argument('--hips-root', default=DEFAULT_HIPS_ROOT,
                    help='local copy of the served HiPS directories, read for '
                         'products with no per-field layers')
    ap.add_argument('--products', nargs='*', default=None,
                    help='product keys to offer, in panner.PRODUCTS order '
                         '(default: all of them)')
    ap.add_argument('--fov', type=float, default=panner.DEFAULT_FOV,
                    help='degrees across the viewport')
    ap.add_argument('--rate', type=float, default=panner.DEFAULT_RATE,
                    help='arcsec of sky per second')
    ap.add_argument('--exclude', nargs='*', default=list(DEFAULT_EXCLUDE),
                    help='pointings (e.g. o138) left off the tour; pass '
                         'with no values to visit every rendered layer')
    ap.add_argument('--max-leg', type=float, default=5.0,
                    help='arcmin; a longer gap is cut across rather than panned')
    return build(ap.parse_args(argv))


if __name__ == '__main__':
    sys.exit(main())
