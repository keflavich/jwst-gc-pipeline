#!/usr/bin/env python
"""Full-field static overview images of the Treasury survey, from its HiPS.

The HiPS layers are the only place the survey exists as one consistently
stretched picture, and they are the right thing to ZOOM into.  They are the
wrong thing to put in a talk, a paper draft or a message: they need a client,
a network round trip per tile, and a viewer that knows what a HiPS is.  These
are the flat version -- one PNG per layer, the whole survey, screen sized.

Deliberate choices
------------------
**Galactic, and rectangular in it.**  The survey is a strip along the plane, so
in Galactic coordinates it lies across the frame instead of running diagonally
through it, and the framing wastes far less empty sky.  The projection is CAR
with ``CRVAL2 = 0``, which is the case where CAR is exactly linear in l and b:
the image IS an l/b rectangle, and a reader can measure off it.

**Screen resolution, not native.**  ~1.7"/px at the default width, against a
HiPS pixel of ~0.4" at order 14.  Anyone who wants to zoom should open the
HiPS; an overview that is secretly a 40000-pixel mosaic serves nobody and
takes a minute to load.  The sampling order is chosen to match the output
scale rather than to be as deep as possible, and `reproject_adaptive`
anti-aliases what downsampling remains -- `reproject_interp` point-samples,
which on a star field turns into a shimmer of dropped and doubled stars.

**Empty sky is transparent, not black.**  The survey is a ragged tiling with
real gaps in it (the skipped visits, the MIRI parallel's offset, the edges),
and a black background asserts "observed, and empty" over sky that was never
looked at.  Alpha is 0 wherever no tile contributed, so the gaps read as gaps
on any page background.

**The control field is excluded.**  o138/o139 sit at b = +0.58, half a degree
north of everything else, with nothing between.  Framing to include them would
spend ~60% of the image on blank sky to show two tiles.  They are dropped by
NUMBER rather than by a latitude cut, so this does not silently start dropping
main-survey tiles if the survey grows north.
"""
import argparse
import json
import os
import sys

import numpy as np
import astropy.units as u
from astropy.coordinates import Galactic, ICRS, SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy_healpix import HEALPix
from PIL import Image
from reproject import reproject_adaptive, reproject_interp
from reproject.mosaicking import reproject_and_coadd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hips_preview import choose_header, read_tile, tile_path   # noqa: E402

from reproject.hips._utils import tile_header_2d              # noqa: E402

BUILD_ROOT = '/orange/adamginsburg/jwst/gc-treasury/pngs'
MOSAIC_BUILD = '/orange/adamginsburg/jwst/gc-treasury/mosaics'
FOOTPRINTS = '/orange/adamginsburg/web/public/jwst-gc/footprints.json'

#: The observations that are the control field, not the survey.  By number,
#: because a latitude cut would quietly take main-survey tiles with it if the
#: footprint ever extends north.
CONTROL_FIELD = ('138', '139')

#: layer directory name -> (build path, output basename).  The same names the
#: HiPS publisher uses, so an overview is traceable to the layer it came from.
LAYERS = {
    'jwst_gc_treasury_vminmax_hips': (
        f'{BUILD_ROOT}/jwst_gc_treasury_vminmax_hips',
        'gc_treasury_overview_vminmax'),
    'jwst_gc_treasury_log_hips': (
        f'{BUILD_ROOT}/jwst_gc_treasury_log_hips',
        'gc_treasury_overview_log'),
    'jwst_gc_treasury_hips': (
        f'{BUILD_ROOT}/jwst_gc_treasury_hips',
        'gc_treasury_overview_pct'),
    'jwst_gc_treasury_miri_bgmatch_hips': (
        f'{BUILD_ROOT}/jwst_gc_treasury_miri_bgmatch_hips',
        'gc_treasury_overview_miri'),
    'gctreasury_mosaic_RGB_770-480-212_hips': (
        f'{MOSAIC_BUILD}/gctreasury_mosaic_RGB_770-480-212_hips',
        'gc_treasury_overview_rgb_770-480-212'),
}

#: Arcminutes of blank sky kept outside the footprint on every side, so the
#: outermost tiles are not flush against the frame edge.
DEFAULT_MARGIN_ARCMIN = 1.0


def survey_box(footprints=FOOTPRINTS, exclude=CONTROL_FIELD):
    """``(l_min, l_max, b_min, b_max)`` in degrees over the observed tiles.

    Measured from the footprint POLYGONS rather than the tile centres: the
    centres understate the box by half a tile on each side, and the MIRI
    parallel sits several arcminutes off its NIRCam prime, so a box drawn
    around the centres clips the MIRI strip off its own edge.
    """
    with open(footprints) as fh:
        data = json.load(fh)
    ls, bs, used = [], [], 0
    for entry in data.get('observed') or []:
        if str(entry.get('number')) in set(exclude):
            continue
        used += 1
        for key in ('nircam', 'miri'):
            for poly in (entry.get(key) or []):
                coord = SkyCoord([p[0] for p in poly], [p[1] for p in poly],
                                 unit='deg').galactic
                ls += list(coord.l.wrap_at(180 * u.deg).deg)
                bs += list(coord.b.deg)
    if not ls:
        raise SystemExit(f'no observed footprints in {footprints}')
    return (min(ls), max(ls), min(bs), max(bs)), used


def target_wcs(box, width, margin_deg):
    """A Galactic CAR grid covering ``box``, and its output shape.

    ``CRVAL2 = 0`` is not cosmetic: CAR is linear in l and b only on the
    equator, so putting the reference latitude anywhere else would make the
    pixel grid a slightly curved function of b and the image would no longer
    be a rectangle anyone can measure.  The field centre is placed with CRPIX
    instead.
    """
    l_min, l_max, b_min, b_max = box
    l_min, l_max = l_min - margin_deg, l_max + margin_deg
    b_min, b_max = b_min - margin_deg, b_max + margin_deg
    scale = (l_max - l_min) / width
    height = int(round((b_max - b_min) / scale))

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ['GLON-CAR', 'GLAT-CAR']
    wcs.wcs.crval = [0.5 * (l_min + l_max), 0.0]
    wcs.wcs.cdelt = [-scale, scale]
    # CRPIX2 puts b=0 where it belongs relative to the requested band, which is
    # what lets CRVAL2 stay on the equator.
    wcs.wcs.crpix = [width / 2 + 0.5, 0.5 - b_min / scale]
    return wcs, (height, width), scale


def sampling_level(scale_deg, tile_size=512):
    """The HiPS order whose pixels are nearest the output scale, from above.

    Sampling far below the output scale is wasted reads and, with a
    point-sampling reprojection, aliasing; sampling above it throws away
    resolution the overview is meant to show.  A HiPS pixel at order k is
    ``(58.63 deg / 2**k) / tile_size``, so this picks the smallest k whose
    pixel is finer than the output pixel, then keeps one more order of headroom
    for the adaptive kernel to average over.
    """
    base = 58.6323 / tile_size            # degrees per pixel at order 0
    level = 0
    while base / 2 ** level > scale_deg and level < 14:
        level += 1
    return min(level + 1, 14)


def render(hips, wcs, shape, level, frame_name='galactic', adaptive=True):
    """``(rgb, covered)`` -- the coadded tiles on the target grid."""
    frame = Galactic() if frame_name == 'galactic' else ICRS()
    hp = HEALPix(nside=2 ** level, order='nested', frame=frame)
    centre = wcs.pixel_to_world(shape[1] / 2, shape[0] / 2)
    corner = wcs.pixel_to_world(0, 0)
    radius = centre.separation(corner) + (58.6323 / 2 ** level) * u.deg

    inputs, missing = [], 0
    for index in hp.cone_search_skycoord(centre, radius=radius):
        path = tile_path(hips, level, int(index))
        if not os.path.exists(path):
            missing += 1
            continue
        header = choose_header(
            tile_header_2d(level=level, index=int(index), frame=frame,
                           tile_size=512), hp, int(index))
        inputs.append((read_tile(path), WCS(header)))
    if not inputs:
        raise SystemExit(f'no tiles of {hips} cover the field of view')

    function = reproject_adaptive if adaptive else reproject_interp
    kwargs = dict(kernel='gaussian', boundary_mode='ignore') if adaptive else {}
    # Coverage travels as a FOURTH PLANE through the same reprojection as the
    # colour, rather than being inferred afterwards from the result.
    #
    # Inferring it does not work, and fails in the direction that hides the
    # problem: `reproject_adaptive` returns 0.0, not NaN, for a pixel no input
    # covered, so `isfinite` was true over the whole frame and the first
    # rendering came out "100.0% covered" with alpha 255 everywhere -- on a
    # ragged tiling that is two thirds empty sky.  The coadd's own footprint is
    # no better: it counts a tile as covering a pixel wherever the TILE lies,
    # including the large blank margins of a HiPS border tile, so it reports
    # coverage the data does not have.
    #
    # The tile's own alpha is the only thing that knows, so it is carried
    # explicitly: 1 where that tile has pixels, 0 where it does not, averaged
    # by the same kernel as the colour so the edges land in the same place.
    planes = []
    for channel in range(4):
        if channel < 3:
            arrays = [(np.nan_to_num(data[..., channel], nan=0.0), tile_wcs)
                      for data, tile_wcs in inputs]
        else:
            arrays = [(np.isfinite(data).any(axis=-1).astype(float), tile_wcs)
                      for data, tile_wcs in inputs]
        array, _ = reproject_and_coadd(
            arrays, wcs, shape_out=shape, reproject_function=function,
            match_background=False, combine_function='mean', **kwargs)
        planes.append(np.nan_to_num(array, nan=0.0))
    rgb = np.stack(planes[:3], axis=-1)
    # Half a pixel's worth of a tile is enough to call it observed; below that
    # the pixel is mostly the blank side of an edge.
    covered = planes[3] >= 0.5
    return rgb, covered, len(inputs), missing


def save_rgba(path, rgb, covered):
    """Write RGBA, transparent where nothing was observed."""
    out = np.empty(rgb.shape[:2] + (4,), dtype=np.uint8)
    out[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    out[..., 3] = np.where(covered, 255, 0)
    # The array is in FITS orientation (row 0 at the bottom); a PNG's row 0 is
    # the top.  Same flip `hips_preview` applies on the way in.
    Image.fromarray(np.flipud(out)).save(path, optimize=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--layer', action='append',
                    help='render only this layer (repeatable); default all')
    ap.add_argument('--out-dir', default='/orange/adamginsburg/jwst/'
                                         'gc-treasury/pngs/overviews')
    ap.add_argument('--width', type=int, default=2400,
                    help='output width in pixels (default 2400, ~1.7 arcsec/px)')
    ap.add_argument('--margin-arcmin', type=float,
                    default=DEFAULT_MARGIN_ARCMIN)
    ap.add_argument('--footprints', default=FOOTPRINTS)
    ap.add_argument('--include-control-field', action='store_true',
                    help='frame to include o138/o139 as well (mostly blank sky)')
    ap.add_argument('--level', type=int,
                    help='HiPS order to sample (default: matched to the output)')
    ap.add_argument('--interp', action='store_true',
                    help='point-sample instead of the anti-aliased kernel')
    ap.add_argument('--publish', action='store_true',
                    help='copy the results to the docroot and starformation, '
                         'beside the HiPS layers they came from')
    ap.add_argument('--dry-run', action='store_true',
                    help='with --publish: print the rsyncs without running them')
    args = ap.parse_args(argv)

    exclude = () if args.include_control_field else CONTROL_FIELD
    box, used = survey_box(args.footprints, exclude)
    wcs, shape, scale = target_wcs(box, args.width, args.margin_arcmin / 60.0)
    level = args.level or sampling_level(scale)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'{used} observed tiles, {"with" if args.include_control_field else "without"} '
          f'the control field')
    print(f'  l {box[0]:+.4f}..{box[1]:+.4f}  b {box[2]:+.4f}..{box[3]:+.4f}')
    print(f'  {shape[1]}x{shape[0]} px at {scale * 3600:.2f}"/px, '
          f'sampling HiPS order {level}')

    names = args.layer or list(LAYERS)
    written = []
    for name in names:
        if name not in LAYERS:
            raise SystemExit(f'unknown layer {name}; known: {", ".join(LAYERS)}')
        hips, base = LAYERS[name]
        if not os.path.isdir(os.path.join(hips, f'Norder{level}')):
            print(f'  {name}: no Norder{level} in {hips}; skipping')
            continue
        rgb, covered, n_tiles, missing = render(hips, wcs, shape, level,
                                                adaptive=not args.interp)
        path = os.path.join(args.out_dir, base + '.png')
        save_rgba(path, rgb, covered)
        frac = float(covered.mean())
        print(f'  {name}: {n_tiles} tiles ({missing} absent) -> {path} '
              f'({frac:.1%} of the frame covered)')
        written.append(path)

    header = wcs.to_header()
    header['NAXIS1'], header['NAXIS2'] = shape[1], shape[0]
    wcs_path = os.path.join(args.out_dir, 'gc_treasury_overview.wcs')
    header.totextfile(wcs_path, overwrite=True)
    print(f'  wrote {wcs_path} (the grid every layer shares)')
    if args.publish and written:
        publish(args.out_dir, dry=args.dry_run)
    return 0 if written else 1


#: Where the overviews are served, beside the HiPS layers they came from.
DOCROOT = '/orange/adamginsburg/web/public/avm_images'
REMOTE = ('starformation:/h/cnswww-starformation.astro/'
          'starformation.astro.ufl.edu/htdocs/avm_images')


def publish(out_dir, dry=False):
    """Copy the overviews next to the HiPS layers on both hosts.

    A plain per-file rsync rather than the staged swap `publish_hips_layers`
    does for a HiPS: a HiPS is a tree of thousands of tiles that is unusable
    half-copied, so it is staged and swapped; each of these is ONE file, and
    rsync replaces a single file atomically enough that a reader gets the old
    one or the new one.  Restricted to the overview names, never a directory
    sync -- the docroot holds other people's files and there is no --delete
    anywhere near it.
    """
    import subprocess
    names = sorted(n for n in os.listdir(out_dir)
                   if n.startswith('gc_treasury_overview'))
    if not names:
        print('  nothing to publish')
        return
    paths = [os.path.join(out_dir, n) for n in names]
    for dest in (DOCROOT, REMOTE):
        cmd = ['rsync', '-a'] + paths + [dest + '/']
        print('  ' + ' '.join(cmd))
        if not dry:
            subprocess.run(cmd, check=True)
        print(f'  published {len(names)} file(s) to {dest}')


if __name__ == '__main__':
    sys.exit(main())
