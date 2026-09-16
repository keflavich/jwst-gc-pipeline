"""Render one standalone image of a HiPS over a given field of view.

Used for the release page's preview of a field whose mosaics do not exist yet
(programme 10678): the HiPS is the only place that survey exists as a single
consistently-stretched picture, so the preview is rendered FROM it.

The HiPS is the only place the whole 10678 mosaic exists as a single
consistently-stretched picture, so the release preview is rendered FROM it
rather than re-derived from the per-field PNGs: re-deriving would reintroduce
exactly the per-field stretch differences the vminmax build exists to remove.
"""
import argparse
import os

import numpy as np
import astropy.units as u
from astropy.coordinates import Galactic, ICRS, SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from astropy_healpix import HEALPix
from PIL import Image
from reproject import reproject_interp
from reproject.mosaicking import reproject_and_coadd
from reproject.hips._utils import tile_header_2d


def tile_path(root, level, index):
    return os.path.join(root, f'Norder{level}', f'Dir{(index // 10000) * 10000}',
                        f'Npix{index}.png')


def read_tile(path):
    """RGB planes of a HiPS PNG tile in FITS orientation.

    PNG row 0 is the TOP row; a FITS array's row 0 is the BOTTOM, and the tile
    WCS is a FITS WCS, so the tile is flipped on read. Skipping this puts every
    tile on the sky mirrored about its own centre, which at this scale looks
    like a plausible picture and is wrong everywhere.
    """
    img = Image.open(path).convert('RGBA')
    arr = np.asarray(img).astype(float)
    alpha = arr[..., 3]
    rgb = arr[..., :3]
    rgb[alpha == 0] = np.nan
    return np.flipud(rgb)


def choose_header(headers, hp, index):
    """The one of a border tile's two candidate headers that lands on the tile.

    `tile_header_2d` returns two for a tile the HPX projection splits; the
    right one is whichever puts the tile's own HEALPix centre inside it.
    """
    if isinstance(headers, fits.Header):
        return headers
    centre = hp.healpix_to_skycoord(index)
    for header in headers:
        wcs = WCS(header)
        x, y = wcs.world_to_pixel(centre)
        if -0.5 <= x <= header['NAXIS1'] - 0.5 and -0.5 <= y <= header['NAXIS2'] - 0.5:
            return header
    return headers[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('hips')
    ap.add_argument('--out', required=True)
    ap.add_argument('--ra', type=float, required=True)
    ap.add_argument('--dec', type=float, required=True)
    ap.add_argument('--fov', type=float, required=True, help='width, degrees')
    ap.add_argument('--width', type=int, default=1800)
    ap.add_argument('--height', type=int, default=1200)
    ap.add_argument('--level', type=int, default=7)
    ap.add_argument('--frame', default='galactic',
                    help='the HiPS frame, from its properties file')
    ap.add_argument('--out-frame', default='galactic',
                    choices=('galactic', 'icrs'))
    args = ap.parse_args()

    centre = SkyCoord(args.ra, args.dec, unit='deg', frame='icrs')
    scale = args.fov / args.width

    target = WCS(naxis=2)
    if args.out_frame == 'galactic':
        # A Galactic-plane survey tiles along l, so a Galactic frame packs the
        # footprint into far less empty sky than an equatorial one: the same
        # pointings run across the frame instead of diagonally through it.
        gal = centre.galactic
        target.wcs.ctype = ['GLON-TAN', 'GLAT-TAN']
        target.wcs.crval = [gal.l.deg, gal.b.deg]
    else:
        target.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        target.wcs.crval = [centre.ra.deg, centre.dec.deg]
    target.wcs.crpix = [args.width / 2 + 0.5, args.height / 2 + 0.5]
    target.wcs.cdelt = [-scale, scale]
    shape = (args.height, args.width)

    frame = Galactic() if args.frame == 'galactic' else ICRS()
    hp = HEALPix(nside=2 ** args.level, order='nested', frame=frame)
    # Half the frame diagonal, plus a tile's worth of margin so an edge tile
    # that only clips the corner is still read.
    radius = (np.hypot(args.fov, args.height * scale) / 2
              + 45 / 2 ** args.level) * u.deg
    indices = hp.cone_search_skycoord(centre, radius=radius)

    inputs = []
    missing = 0
    for index in indices:
        path = tile_path(args.hips, args.level, int(index))
        if not os.path.exists(path):
            missing += 1
            continue
        header = choose_header(tile_header_2d(level=args.level, index=int(index),
                                              frame=frame, tile_size=512),
                               hp, int(index))
        inputs.append((read_tile(path), WCS(header)))
    print(f'{len(inputs)} tiles read, {missing} of {len(indices)} absent')
    if not inputs:
        raise SystemExit('no tiles cover that field of view')

    planes = []
    for channel in range(3):
        array, _ = reproject_and_coadd(
            [(data[..., channel], wcs) for data, wcs in inputs],
            target, shape_out=shape, reproject_function=reproject_interp,
            match_background=False, combine_function='mean')
        planes.append(array)
    rgb = np.stack(planes, axis=-1)
    rgb = np.nan_to_num(rgb, nan=0.0)
    out = np.clip(rgb, 0, 255).astype(np.uint8)
    Image.fromarray(np.flipud(out)).save(args.out)
    filled = float(np.mean(np.any(out > 0, axis=-1)))
    print(f'wrote {args.out} ({args.width}x{args.height}, {filled:.1%} of the '
          f'frame has data)')


if __name__ == '__main__':
    main()
