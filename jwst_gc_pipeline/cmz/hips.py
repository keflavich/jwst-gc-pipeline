"""Mono-per-filter HiPS + derived two-color HiPS for the growing CMZ mosaic.

Design (see ``scripts/release/CMZ_HIPS_AND_CATALOG_SHARING_PLAN.md``): the
incremental substrate is a MONO (grayscale, real-flux) HiPS per filter; the
COLOR HiPS is a cheap DERIVED layer over two mono HiPS.

* :func:`build_mono_hips` — reproject a field's ``_i2d`` into a mono HiPS tile
  tree (astropy ``reproject.hips``), galactic frame, FITS tiles (numeric, so the
  color step and any future pyramid merge stay lossless).
* :class:`MemberRegistry` — tracks which ``_i2d`` files feed a filter's mono
  HiPS, so a new field appends to the registry.  (Correct incremental *pyramid*
  merging is what CDS Hipsgen.jar does natively; ``cmz.hipsgen`` wraps it.  This
  pure-python path rebuilds from the member registry -- honest about that rather
  than hand-rolling HEALPix nested-child pyramid math, the exact thing the repo's
  HiPS-orientation QA already has to police.)
* :func:`derive_two_color_hips` — per-tile ``R = long``, ``B = F212N``,
  ``G = 0.5*(R+B)`` with a GLOBAL asinh stretch (per-tile stretch would seam),
  emitting PNG tiles Aladin renders directly.
"""
import json
import os

import numpy as np

TILE_WIDTH = 512


# --------------------------------------------------------------------------
# tile-tree addressing (HiPS standard: Norder{k}/Dir{d}/Npix{p}.{ext})
# --------------------------------------------------------------------------
def tile_path(root, order, npix, ext):
    d = (npix // 10000) * 10000
    return os.path.join(root, f'Norder{order}', f'Dir{d}', f'Npix{npix}.{ext}')


def iter_tiles(root, order, ext):
    """Yield ``(npix, path)`` for every tile at ``order`` under ``root``."""
    import glob
    base = os.path.join(root, f'Norder{order}')
    for p in sorted(glob.glob(os.path.join(base, 'Dir*', f'Npix*.{ext}'))):
        npix = int(os.path.basename(p).split('Npix')[1].split('.')[0])
        yield npix, p


def read_properties(root):
    props = {}
    path = os.path.join(root, 'properties')
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                if '=' in line and not line.strip().startswith('#'):
                    k, v = line.split('=', 1)
                    props[k.strip()] = v.strip()
    return props


def _max_order(root, ext):
    import glob
    orders = [int(os.path.basename(d)[6:])
              for d in glob.glob(os.path.join(root, 'Norder*'))
              if os.path.basename(d)[6:].isdigit()]
    return max(orders) if orders else None


# --------------------------------------------------------------------------
# mono HiPS build (reproject.hips)
# --------------------------------------------------------------------------
def build_mono_hips(i2d_paths, out_dir, order=None, coord_system_out='galactic',
                    tile_format='fits', threads=8, sci_ext='SCI'):
    """Build/refresh a mono HiPS from one or more ``_i2d`` mosaics.

    Thin wrapper over ``reproject.hips.reproject_to_hips``.  Returns ``out_dir``.
    NOTE: reproject.hips writes a fresh tree; for a GROWING survey pass the full
    member list (see :class:`MemberRegistry`) or use ``cmz.hipsgen`` for native
    incremental tiling.
    """
    from reproject.hips import reproject_to_hips
    from reproject import reproject_interp
    from astropy.io import fits
    from astropy.wcs import WCS

    inputs = []
    for p in ([i2d_paths] if isinstance(i2d_paths, str) else list(i2d_paths)):
        with fits.open(p) as hdul:
            hdu = hdul[sci_ext] if sci_ext in [h.name for h in hdul] else hdul[0]
            inputs.append((np.asarray(hdu.data, float), WCS(hdu.header)))
    os.makedirs(out_dir, exist_ok=True)
    kwargs = dict(reproject_function=reproject_interp,
                  output_directory=out_dir, coord_system_out=coord_system_out,
                  threads=threads, tile_format=tile_format)
    if order is not None:
        kwargs['level'] = order
    # reproject_to_hips accepts a single (array, wcs) or a list to coadd.
    reproject_to_hips(inputs if len(inputs) > 1 else inputs[0], **kwargs)
    return out_dir


class MemberRegistry:
    """Track the ``_i2d`` members of a filter's mono HiPS (a JSON sidecar)."""

    def __init__(self, path):
        self.path = path
        self.members = []
        if os.path.exists(path):
            with open(path) as fh:
                self.members = json.load(fh).get('members', [])

    def add(self, i2d_path, field, tag=None):
        entry = {'i2d': os.path.abspath(i2d_path), 'field': field, 'tag': tag}
        if entry['i2d'] not in {m['i2d'] for m in self.members}:
            self.members.append(entry)
        return self

    def save(self):
        with open(self.path, 'w') as fh:
            json.dump({'members': self.members}, fh, indent=2)
        return self.path

    def i2d_paths(self):
        return [m['i2d'] for m in self.members]


# --------------------------------------------------------------------------
# global stretch + two-color derivation
# --------------------------------------------------------------------------
def _load_tile_array(path):
    from astropy.io import fits
    with fits.open(path) as hdul:
        for hdu in hdul:
            if getattr(hdu, 'data', None) is not None:
                return np.asarray(hdu.data, float)
    return None


def global_limits(root, sample_order=3, percentiles=(1.0, 99.5)):
    """Global (vmin, vmax) for a mono HiPS from its coarse ``sample_order`` tiles.

    Using ONE global stretch (not per-tile) is what keeps the color mosaic
    seamless.  Falls back to the deepest available order if ``sample_order`` is
    absent.
    """
    order = sample_order
    if _max_order(root, 'fits') is not None and order > _max_order(root, 'fits'):
        order = _max_order(root, 'fits')
    vals = []
    for _, p in iter_tiles(root, order, 'fits'):
        arr = _load_tile_array(p)
        if arr is not None:
            v = arr[np.isfinite(arr)]
            if v.size:
                vals.append(v)
    if not vals:
        raise ValueError(f'no finite pixels at order {order} under {root}')
    allv = np.concatenate(vals)
    lo, hi = np.percentile(allv, percentiles)
    return float(lo), float(hi)


def _asinh_norm(arr, vmin, vmax):
    """Asinh stretch to [0,1] (matches the jwst_rgb house stretch)."""
    a = 0.1
    x = (np.asarray(arr, float) - vmin) / max(vmax - vmin, 1e-30)
    x = np.clip(x, 0, 1)
    out = np.arcsinh(x / a) / np.arcsinh(1.0 / a)
    return np.clip(out, 0, 1)


def two_color_tile(blue_arr, red_arr, blue_lims, red_lims):
    """One RGBA tile: R=long, B=short (F212N), G=0.5*(R+B); NaN -> transparent."""
    b = _asinh_norm(blue_arr, *blue_lims)
    r = _asinh_norm(red_arr, *red_lims)
    g = 0.5 * (r + b)
    finite = np.isfinite(blue_arr) | np.isfinite(red_arr)
    # NaN channels -> 0 before casting (alpha carries the transparency), so the
    # uint8 cast never sees NaN.
    r = np.nan_to_num(r, nan=0.0)
    g = np.nan_to_num(g, nan=0.0)
    b = np.nan_to_num(b, nan=0.0)
    rgba = np.zeros(r.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = (r * 255).astype(np.uint8)
    rgba[..., 1] = (g * 255).astype(np.uint8)
    rgba[..., 2] = (b * 255).astype(np.uint8)
    rgba[..., 3] = np.where(finite, 255, 0).astype(np.uint8)
    return rgba


def derive_two_color_hips(blue_hips, red_hips, out_dir, blue_lims=None,
                          red_lims=None, hips_frame='galactic'):
    """Derive a two-color (+interpolated green) PNG HiPS from two mono HiPS.

    ``blue_hips`` = F212N mono HiPS, ``red_hips`` = long-band mono HiPS.  For
    every tile present in BOTH (at every order), writes an RGBA PNG tile with a
    GLOBAL stretch.  Returns the number of tiles written.
    """
    from PIL import Image
    blue_lims = blue_lims or global_limits(blue_hips)
    red_lims = red_lims or global_limits(red_hips)
    max_order = min(o for o in (_max_order(blue_hips, 'fits'),
                                _max_order(red_hips, 'fits')) if o is not None)
    os.makedirs(out_dir, exist_ok=True)
    n_written = 0
    for order in range(max_order + 1):
        blue_tiles = dict(iter_tiles(blue_hips, order, 'fits'))
        red_tiles = dict(iter_tiles(red_hips, order, 'fits'))
        for npix in sorted(set(blue_tiles) & set(red_tiles)):
            barr = _load_tile_array(blue_tiles[npix])
            rarr = _load_tile_array(red_tiles[npix])
            if barr is None or rarr is None or barr.shape != rarr.shape:
                continue
            rgba = two_color_tile(barr, rarr, blue_lims, red_lims)
            outp = tile_path(out_dir, order, npix, 'png')
            os.makedirs(os.path.dirname(outp), exist_ok=True)
            Image.fromarray(rgba, mode='RGBA').save(outp)
            n_written += 1
    _write_color_properties(out_dir, max_order, hips_frame,
                            blue_lims, red_lims, blue_hips, red_hips)
    return n_written


def _write_color_properties(out_dir, max_order, frame, blue_lims, red_lims,
                            blue_hips, red_hips, pipeline_tag=None):
    if pipeline_tag is None:
        try:
            from jwst_gc_pipeline.versioning.tags import get_pipeline_tag
            pipeline_tag = get_pipeline_tag()
        except (ImportError, OSError, ValueError):
            pipeline_tag = 'unknown'
    with open(os.path.join(out_dir, 'properties'), 'w') as fh:
        fh.write(
            f'creator_did          = ivo://jwst-gc/cmz/two-color\n'
            f'obs_title            = CMZ two-color (F212N + long, G=0.5(R+B))\n'
            f'hips_builder         = jwst_gc_pipeline.cmz.hips\n'
            f'hips_release_date    = \n'
            f'hips_frame           = {frame}\n'
            f'hips_order           = {max_order}\n'
            f'hips_tile_width      = {TILE_WIDTH}\n'
            f'hips_tile_format     = png\n'
            f'dataproduct_type     = image\n'
            f'gc_pipeline_tag      = {pipeline_tag}\n'
            f'gc_blue_hips         = {os.path.abspath(blue_hips)}\n'
            f'gc_red_hips          = {os.path.abspath(red_hips)}\n'
            f'gc_blue_limits       = {blue_lims[0]:.6g},{blue_lims[1]:.6g}\n'
            f'gc_red_limits        = {red_lims[0]:.6g},{red_lims[1]:.6g}\n')
