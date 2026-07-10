print("Starting crowdsource_catalogs_long", flush=True)
import sys
import tracemalloc
import resource
import glob
import time
import json
import re
import inspect
import numpy
import regions
import numpy as np
from pathlib import Path
from functools import cache
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel, interpolate_replace_nans
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
from astropy.visualization import simple_norm
from astropy.modeling.fitting import LevMarLSQFitter
from astropy import wcs
from astropy import table
from astropy import stats
from astropy import units as u
from astropy.nddata import NDData
from astropy.io import fits
from scipy import ndimage
from scipy.spatial import cKDTree
import requests
import requests.exceptions
import urllib3
import urllib3.exceptions
from jwst.datamodels import dqflags
from jwst.datamodels import ImageModel
from jwst.associations import asn_from_list
from jwst.associations.lib.rules_level3_base import DMS_Level3_Base
from jwst.resample import ResampleStep
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf import extract_stars, EPSFStars, EPSFBuilder
# EPSFModel was deprecated in photutils 2.0 in favour of ImagePSF
try:
    from photutils.psf import ImagePSF as EPSFModel
except ImportError:
    from photutils.psf import EPSFModel
# PSFPhotometry, IterativePSFPhotometry, SourceGrouper present since photutils 1.9
from photutils.psf import PSFPhotometry, IterativePSFPhotometry, SourceGrouper, GriddedPSFModel
# LocalBackground present since photutils 1.9
from photutils.background import MMMBackground, MADStdBackgroundRMS, MedianBackground, Background2D, LocalBackground

import warnings
from astropy.utils.exceptions import AstropyWarning, AstropyDeprecationWarning
warnings.simplefilter('ignore', category=AstropyWarning)


# ---------------------------------------------------------------------------
# Monkey-patch around astropy.nddata.utils.overlap_slices bug (hit via
# photutils.make_model_image when photutils passes small_array_shape as an
# ndarray and an out-of-frame source yields e_max == 0).
#
# At line ~138 of astropy/nddata/utils.py:
#   if e_max < 0 or (e_max == 0 and small_array_shape != (0, 0)):
# When small_array_shape is an ndarray, `ndarray != (0, 0)` returns an array
# and the `or` branch raises:
#   ValueError: The truth value of an array with more than one element
#   is ambiguous. Use a.any() or a.all()
#
# This is triggered by IterativePSFPhotometry when a fit converges to a
# source centered at e.g. y = -7.63 with sub_shape=(15, 15): then
# e_max = int(-15.13) + 15 = 0 and the comparison explodes.
#
# We wrap the function so that small_array_shape is coerced to a tuple of
# ints before the comparison, matching the function's own semantics.
# ---------------------------------------------------------------------------
import astropy.nddata.utils as _astropy_nddata_utils
_original_overlap_slices = _astropy_nddata_utils.overlap_slices


def _overlap_slices_tuple_shape(large_array_shape, small_array_shape, position,
                                mode='partial', **kwargs):
    small_array_shape = tuple(int(x) for x in small_array_shape)
    return _original_overlap_slices(large_array_shape, small_array_shape,
                                    position, mode=mode, **kwargs)


_astropy_nddata_utils.overlap_slices = _overlap_slices_tuple_shape
# photutils imports overlap_slices at module import time in several places;
# update those module-level bindings too so the patched version is used.
import photutils.utils.cutouts as _photutils_cutouts
_photutils_cutouts.overlap_slices = _overlap_slices_tuple_shape
import photutils.datasets.images as _photutils_datasets_images
_photutils_datasets_images.overlap_slices = _overlap_slices_tuple_shape
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)


# ---------------------------------------------------------------------------
# Photutils 2.x <-> 3.x compatibility shims.
#
# In photutils 3.0 several keyword arguments were renamed:
#   * PSFPhotometry / IterativePSFPhotometry: ``localbkg_estimator`` ->
#     ``local_bkg_estimator``
#   * make_model_image / make_residual_image: ``include_localbkg`` ->
#     ``include_local_bkg``
# Both old names are retained as deprecation-warning aliases until 4.0,
# but the deprecation warnings flood the per-frame logs.  These small
# wrappers detect the installed version once and dispatch to the right
# kwarg, so the same source works on 2.3.0 and 3.0+.
# ---------------------------------------------------------------------------
# photutils-compat shims + PSF-fit helpers factored into photometry/psf_fitting.py
# (2026-06-09 restructure).  Imported here so existing references keep working.
from jwst_gc_pipeline.photometry.psf_fitting import (
    _PHOTUTILS_GE_3, _LOCAL_BKG_KW, _INCLUDE_LOCAL_BKG_KW,
    _make_psfphotometry, _make_iterative_psfphotometry,
    CachingGriddedPSFModel, forced_psf_photometry,
    _make_model_image, _dedup_close_sources,
)


# ---------------------------------------------------------------------------
# Forced photometry & caching PSF model (experimental low-level fits)
#
# Two related accelerators for the per-source fit step:
#
# 1.  ``CachingGriddedPSFModel`` -- subclass of ``GriddedPSFModel`` that
#     memoizes the rendered PSF stamp on (x_0, y_0, stamp pixel grid).
#     Drop-in replacement for ``GriddedPSFModel`` in any photutils call;
#     accelerates LM fits where the inner finite-difference Jacobian
#     perturbs ONLY the flux parameter (a frequent late-iteration case)
#     or where (x_0, y_0) are pinned via ``xy_bounds``.
#
# 2.  ``forced_psf_photometry()`` -- standalone closed-form linear flux
#     solve at fixed (x_0, y_0).  Bypasses photutils' LM entirely:
#         f = sum(d*p*w) / sum(p^2*w),   sigma_f = 1 / sqrt(sum(p^2*w))
#     ~80x faster than the full LM path on iter2/iter3 chains where
#     positions come from a union seed catalog.
# ---------------------------------------------------------------------------
# CachingGriddedPSFModel and forced_psf_photometry are imported from
# photometry/psf_fitting.py (see the import near the top of this module).


# ---------------------------------------------------------------------------
# Parallel-chunked PSFPhotometry (experimental; off unless --parallel-workers>1)
#
# The serial PSFPhotometry call ends up spending most wall-time in
# per-source LevMar + LocalBackground (annulus sigma-clip).  Sources whose
# PSFs do not overlap can be fit independently, so we can split the
# init_params table into chunks of mutually-non-touching groups and run
# one PSFPhotometry per chunk in a forked worker.  Forked workers inherit
# the SCI image, error array, mask, and PSF model read-only via
# copy-on-write (Linux); only the small init_params chunk is pickled.
#
# Bit-exact agreement with the serial path was verified in
# brick2221/analysis/bench_psf_parallel.py for the non-LocalBackground
# case; LocalBackground is per-source and deterministic, so agreement
# should carry over.  We still keep the serial path as the default and
# only activate parallel when the caller passes --parallel-workers > 1.
# ---------------------------------------------------------------------------
import multiprocessing as _mp_par  # noqa: E402 -- kept near user

# Module-level state populated by _par_worker_init; only read in workers.
def resolve_max_group_size(raw):
    """Resolve the --max-group-size option to either None (unlimited, no cap) or
    a positive int (the cap).  The value 0 is REJECTED as ambiguous: it used to
    mean "no cap" but reads like "no grouping", so callers must now be explicit
    ('unlimited' or a positive integer).  Raises SystemExit on an invalid value.
    """
    if raw is None:
        raise SystemExit(
            "--max-group-size must be set explicitly to 'unlimited' or a positive "
            "integer (it has no implicit default; 0 is not allowed).")
    s = str(raw).strip().lower()
    if s in ('unlimited', 'inf', 'infinite', 'nocap', 'none'):
        return None
    try:
        n = int(s)
    except ValueError:
        raise SystemExit(
            f"--max-group-size={raw!r} is invalid; use 'unlimited' or a positive "
            f"integer.")
    if n == 0:
        raise SystemExit(
            "--max-group-size=0 is ambiguous and no longer allowed: it meant "
            "'unlimited group size' but reads like 'no grouping'.  Pass "
            "--max-group-size=unlimited for no cap, or a positive integer "
            "(e.g. 10-15) for a cap.")
    if n < 0:
        raise SystemExit(
            f"--max-group-size={n} is invalid; use 'unlimited' or a positive integer.")
    return n


class CappedSourceGrouper:
    """SourceGrouper wrapper that caps the maximum group size.

    photutils SourceGrouper has only a min_separation knob: any cluster of
    sources mutually within that distance becomes one group, however large.
    In dense fields with cross-band union seeds, those clusters routinely
    reach 50-100+ sources, and the joint LevMar fit cost grows roughly
    cubically with group size; brick LW iter3 saw the per-source rate
    collapse from ~5 it/s to <1 it/s in dense regions, hitting the 96 h
    walltime at ~14% completion.

    This wrapper post-processes the inner grouper's output: any group
    larger than ``max_size`` is split into ``ceil(N/max_size)`` spatially
    coherent sub-groups by sorting along the group's principal axis.
    Sparse-region groups (already <= max_size) are returned untouched.
    """

    def __init__(self, min_separation, max_size=0):
        self._inner = SourceGrouper(min_separation)
        self.min_separation = min_separation
        self.max_size = int(max_size) if max_size else 0

    def __call__(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        group_ids = np.asarray(self._inner(x, y)).astype(np.int64, copy=True)
        if self.max_size <= 0 or group_ids.size == 0:
            return group_ids
        next_id = int(group_ids.max()) + 1
        unique_ids, counts = np.unique(group_ids, return_counts=True)
        oversized = unique_ids[counts > self.max_size]
        if oversized.size == 0:
            return group_ids
        n_split = 0
        for gid in oversized:
            mask = group_ids == gid
            n = int(mask.sum())
            n_sub = int(np.ceil(n / self.max_size))
            xy = np.column_stack([x[mask], y[mask]])
            centered = xy - xy.mean(axis=0)
            cov = centered.T @ centered
            try:
                eigvals, eigvecs = np.linalg.eigh(cov)
                principal = eigvecs[:, int(np.argmax(eigvals))]
                if not np.all(np.isfinite(principal)) or np.linalg.norm(principal) == 0:
                    raise ValueError('degenerate principal axis')
                proj = centered @ principal
            except (ValueError, np.linalg.LinAlgError):
                proj = xy[:, 0]
            order = np.argsort(proj, kind='stable')
            sub_size = int(np.ceil(n / n_sub))
            sub_label = np.empty(n, dtype=np.int64)
            for k in range(n_sub):
                sub_label[order[k * sub_size:(k + 1) * sub_size]] = k
            new_ids = np.empty(n, dtype=group_ids.dtype)
            new_ids[sub_label == 0] = gid
            for k in range(1, n_sub):
                new_ids[sub_label == k] = next_id
                next_id += 1
            group_ids[mask] = new_ids
            n_split += 1
        print(f"CappedSourceGrouper: split {n_split} oversized "
              f"group{'s' if n_split != 1 else ''} (cap={self.max_size}); "
              f"total groups: {len(unique_ids)} -> {next_id - int(unique_ids[0])}",
              flush=True)
        return group_ids


# Filename-token helpers factored into photometry/naming.py (2026-06-09
# restructure).  Imported here so existing references keep working unchanged.
from jwst_gc_pipeline.photometry.naming import (
    _CHUNK_TOKEN_RE, _chunk_token, _strip_chunk, _iteration_token, _bgsub_token,
    MIRI_FILTERS, _instrument_from_filter, _inst_token,
    residual_to_smoothed_bg_i2d, residual_to_model_i2d, residual_to_infilled_i2d,
)
from jwst_gc_pipeline.photometry.psf_paths import (
    resolve_merged_psf_grid_path, central_psf_dir,
)


def _seed_table_chunk_subset(seed_table, ww, image_shape,
                             chunk_index, n_seed_chunks):
    """Subset a seed table to one tile of an N-way image-pixel chunking.

    The image is split into a ``Gx x Gy`` grid where
    ``Gx = ceil(sqrt(N))`` and ``Gy = ceil(N / Gx)`` (so Gx * Gy >= N;
    chunks beyond N never receive sources).  Each chunk owns a half-open
    pixel rectangle; a seed is assigned to the chunk that contains its
    pixel position.  Seeds with non-finite pixel coordinates (e.g. far
    outside the FOV) are dropped.

    Returns a NEW Table; the input is not modified.
    """
    n = int(n_seed_chunks)
    if n <= 1:
        return seed_table
    ny, nx = int(image_shape[0]), int(image_shape[1])
    gx = int(np.ceil(np.sqrt(n)))
    gy = int(np.ceil(n / gx))
    cx = chunk_index % gx
    cy = chunk_index // gx
    x_lo = (nx * cx) / gx
    x_hi = (nx * (cx + 1)) / gx
    y_lo = (ny * cy) / gy
    y_hi = (ny * (cy + 1)) / gy

    seed_table = _as_table(seed_table)
    if 'skycoord' in seed_table.colnames:
        sc = seed_table['skycoord']
    else:
        sc = SkyCoord(ra=seed_table['ra'], dec=seed_table['dec'], unit='deg')
    xpix, ypix = ww.world_to_pixel(sc)
    xpix = np.asarray(xpix, dtype=float)
    ypix = np.asarray(ypix, dtype=float)
    in_tile = (
        np.isfinite(xpix) & np.isfinite(ypix)
        & (xpix >= x_lo) & (xpix < x_hi)
        & (ypix >= y_lo) & (ypix < y_hi)
    )
    n_kept = int(in_tile.sum())
    print(
        f"Seed chunking: chunk {chunk_index + 1}/{n} (grid {gx}x{gy}, "
        f"x=[{x_lo:.1f},{x_hi:.1f}) y=[{y_lo:.1f},{y_hi:.1f})): "
        f"{len(seed_table)} -> {n_kept} seeds",
        flush=True,
    )
    return seed_table[in_tile]


def _sanitize_cutout_label(s):
    """Make a cutout label safe to embed in a directory name."""
    return re.sub(r'[^A-Za-z0-9._+-]', '', str(s)) or 'cutout'


def _cutout_label_for(options):
    """Final cutout output label (filesystem-safe).  ``--cutout-label``
    overrides; else derived from the .reg basename or the ra,dec of the spec.
    Single source of truth shared by _prepare_cutout_input and the driver's
    cutout residual-mosaic so both resolve the same <basepath>/cutouts/<label>.
    """
    if getattr(options, 'cutout_label', ''):
        return _sanitize_cutout_label(options.cutout_label)
    spec = str(getattr(options, 'cutout_region', '')).strip()
    if spec.lower().endswith('.reg') or (os.path.exists(spec) and ',' not in spec):
        return _sanitize_cutout_label(os.path.splitext(os.path.basename(spec))[0])
    parts = spec.split(',')
    return _sanitize_cutout_label(f"{float(parts[0]):.5f}{float(parts[1]):+.5f}")


def _cutout_out_basepath(basepath, options):
    """``<basepath>/cutouts/<label>`` for a --cutout-region run; ``basepath``
    itself for a full-frame run (no ``--cutout-region``).  The manual pipeline
    runs full-frame in place under ``basepath`` and only namespaces into the
    ``cutouts/`` tree when a cutout region is requested."""
    if not getattr(options, 'cutout_region', ''):
        return basepath
    return os.path.join(basepath, 'cutouts', _cutout_label_for(options))


def _parse_cutout_region(spec, ww, default_size_arcsec=5.0):
    """Parse a ``--cutout-region`` spec into ``(center, size, label)``.

    ``spec`` is either a DS9 ``.reg`` file (first region; circle / box /
    point) or a comma string ``'ra,dec,size'`` or ``'ra,dec,w,h'`` (deg,
    deg, arcsec).  ``size`` is returned as an astropy Quantity suitable for
    ``Cutout2D`` (scalar -> square; ``(h, w)`` tuple otherwise).  ``label``
    is a filesystem-safe string used to namespace the cutout outputs.
    """
    spec = str(spec).strip()
    if spec.lower().endswith('.reg') or (os.path.exists(spec) and ',' not in spec):
        reg = regions.Regions.read(spec)[0]
        center = reg.center
        if not isinstance(center, SkyCoord):  # pixel region
            center = ww.pixel_to_world(center.x, center.y)
        if hasattr(reg, 'radius'):
            size = 2.0 * u.Quantity(reg.radius)
        elif hasattr(reg, 'width') and hasattr(reg, 'height'):
            size = (u.Quantity(reg.height), u.Quantity(reg.width))  # (ny, nx)
        else:  # point region: no extent -> use the default square size
            size = default_size_arcsec * u.arcsec
        label = os.path.splitext(os.path.basename(spec))[0]
    else:
        parts = [float(x) for x in spec.split(',')]
        center = SkyCoord(parts[0] * u.deg, parts[1] * u.deg)
        if len(parts) == 3:
            size = parts[2] * u.arcsec
        elif len(parts) >= 4:
            size = (parts[3] * u.arcsec, parts[2] * u.arcsec)  # (ny, nx)
        else:
            raise ValueError("--cutout-region string must be 'ra,dec,size' "
                             f"or 'ra,dec,w,h' (deg,deg,arcsec); got {spec!r}")
        label = f"{parts[0]:.5f}{parts[1]:+.5f}"
    return center, size, _sanitize_cutout_label(label)


class CutoutNoOverlap(ValueError):
    """Raised when a ``--cutout-region`` does not overlap a given exposure.

    The per-exposure driver catches this to skip the frame; it errors only if
    NO frame in the run overlaps the region.
    """


def _shift_gwcs(gwcs_obj, x0, y0):
    """GWCS for a cutout whose origin is full-frame pixel ``(x0, y0)``:
    cutout ``(x, y)`` -> full ``(x + x0, y + y0)`` -> world.

    Prepends a pixel ``Shift`` to the forward transform so the cutout keeps
    the exact (rectified) astrometry of the parent i2d.
    """
    import gwcs as _gwcs
    from astropy.modeling.models import Shift
    shifted = (Shift(float(x0)) & Shift(float(y0))) | gwcs_obj.forward_transform
    return _gwcs.WCS(forward_transform=shifted,
                     input_frame=gwcs_obj.input_frame,
                     output_frame=gwcs_obj.output_frame)


def _crop_to_slices(a, ny, nx, yslc, xslc):
    """Crop a datamodel array to ``[yslc, xslc]``: a 2-D ``(ny, nx)`` plane or a
    3-D ``(..., ny, nx)`` cube; pass None / empty / other-shaped arrays through
    unchanged.  Shared by the cutout-input and i2d-finite-crop datamodel
    croppers (both crop the same attribute set against a fixed slice pair)."""
    if a is None or np.size(a) == 0:
        return a
    if a.ndim == 2 and a.shape == (ny, nx):
        return a[yslc, xslc]
    if a.ndim == 3 and a.shape[-2:] == (ny, nx):
        return a[:, yslc, xslc]
    return a


def _prepare_cutout_input(filename, basepath, filtername, options):
    """Write a cropped *datamodel* copy of ``filename`` for a
    ``--cutout-region`` run.  Returns ``(label, cutout_filename, out_basepath)``.

    The cutout keeps a VALID GWCS -- the parent i2d's GWCS shifted by the
    cutout origin -- in the ASDF extension, plus a matching FITS WCS in the
    SCI header.  So (a) catalog RA/Dec are exact, (b) the per-frame residual
    / model datamodels stay resample-able, and (c) the residual-mosaic
    ResampleStep spans only the cutout region (the shifted GWCS carries a
    cutout-sized bounding box).  The cutout file lives under
    ``<basepath>/cutouts/<label>/<filtername>/pipeline/`` so every
    filename-derived output (satstar models, bg dumps, residuals) is
    namespaced and never overwrites full-frame products.
    """
    from astropy.nddata import Cutout2D, NoOverlapError
    from astropy.wcs import NoConvergence
    with fits.open(filename) as hdul:
        sci_ww = wcs.WCS(hdul['SCI'].header)
        center, size, _ = _parse_cutout_region(
            options.cutout_region, sci_ww,
            default_size_arcsec=float(getattr(options, 'cutout_size_arcsec', 5.0)))
        label = _cutout_label_for(options)
        try:
            cut = Cutout2D(np.asarray(hdul['SCI'].data), position=center, size=size,
                           wcs=sci_ww, mode='trim', copy=True)
        except (NoOverlapError, NoConvergence) as ex:
            # NoOverlapError: region projects just off the array.
            # NoConvergence: region is far off-field, so the SIP world->pixel
            # inverse diverges.  Either way the frame doesn't cover the region.
            raise CutoutNoOverlap(
                f"cutout region does not overlap {os.path.basename(filename)}") from ex
    yslc, xslc = cut.slices_original
    x0, y0 = int(xslc.start), int(yslc.start)
    ny_c, nx_c = cut.data.shape

    # A region that only grazes the detector edge trims (mode='trim') to a
    # degenerate (zero- or near-zero-size) crop without raising NoOverlapError.
    # That is not a usable frame -- treat it as a non-overlap so the per-exposure
    # driver SKIPS it (legitimate for a cutout) instead of feeding an empty image
    # downstream (empty-array reductions then crash, e.g. get_saturated_stars).
    if ny_c < 2 or nx_c < 2:
        raise CutoutNoOverlap(
            f"cutout region grazes {os.path.basename(filename)} -> degenerate "
            f"{ny_c}x{nx_c} crop")

    out_basepath = os.path.join(basepath, 'cutouts', label)
    out_dir = os.path.join(out_basepath, filtername, 'pipeline')
    os.makedirs(out_dir, exist_ok=True)
    cutout_filename = os.path.join(
        out_dir, os.path.basename(filename).replace('.fits', f'_cutout_{label}.fits'))

    with ImageModel(filename) as m:
        ny, nx = m.data.shape

        new = m.copy()
        for attr in ('data', 'err', 'dq', 'wht', 'con', 'var_poisson',
                     'var_rnoise', 'var_flat', 'area'):
            if hasattr(m, attr):
                setattr(new, attr, _crop_to_slices(getattr(m, attr), ny, nx, yslc, xslc))
        shifted = _shift_gwcs(m.meta.wcs, x0, y0)
        shifted.bounding_box = ((-0.5, nx_c - 0.5), (-0.5, ny_c - 0.5))
        new.meta.wcs = shifted
        new.save(cutout_filename)

    # Write the matching cutout FITS WCS into the SCI header so the pipeline's
    # ``wcs.WCS(im1['SCI'].header)`` reads the correct (cutout) WCS; the
    # shifted GWCS stays in the ASDF extension for datamodels / resample.
    #
    # relax=True is REQUIRED: the detector-frame parent WCS is RA---TAN-SIP, and
    # plain to_header() drops the '-SIP' CTYPE suffix while leaving the A_*/B_*
    # SIP coefficients in the header.  astropy still applies SIP (with a warning),
    # so catalog RA/Dec stay correct, but viewers that honor CTYPE strictly (CARTA)
    # then IGNORE the distortion -> every per-frame cutout image (residual, model,
    # _resbg_reproj, _srcfind_input) displays ~0.3-0.5 px offset from the catalog
    # and from the rectified i2d.  relax=True writes CTYPE='RA---TAN-SIP' so the
    # SIP is declared and applied consistently everywhere.
    with fits.open(cutout_filename, mode='update') as h:
        h['SCI'].header.update(cut.wcs.to_header(relax=True))
        h.flush()

    print(f"CUTOUT '{label}': wrote {cut.data.shape} cutout centered "
          f"{center.to_string('hmsdms')} (origin x0={x0},y0={y0}) -> "
          f"{cutout_filename}", flush=True)
    # x0,y0 = cutout origin in the PARENT frame's pixel coords; the caller
    # uses it to re-origin the spatially-varying PSF grid so the cutout's PSF
    # matches what the full-frame fit would use at the same source positions.
    return label, cutout_filename, out_basepath, x0, y0


def _crop_datamodel_to_finite(filename, pad=4):
    """Crop an i2d datamodel in place to the bounding box of its finite,
    nonzero SCI data (plus ``pad`` px), shifting the GWCS by the crop origin.

    ResampleStep auto-allocates a near-full-frame output canvas even for a
    small cutout (the cutout data ends up filling <1% of it).  This trims the
    mosaic back to the cutout region.  i2d products are plain RA--TAN (no
    SIP), so the FITS WCS crop is an exact ``CRPIX -= origin``.
    """
    with ImageModel(filename) as m:
        d = np.asarray(m.data)
        finite = np.isfinite(d) & (d != 0)
        if not finite.any():
            return
        ys, xs = np.where(finite)
        ny, nx = d.shape
        y0 = max(0, int(ys.min()) - pad); y1 = min(ny, int(ys.max()) + 1 + pad)
        x0 = max(0, int(xs.min()) - pad); x1 = min(nx, int(xs.max()) + 1 + pad)
        if (y0, x0, y1, x1) == (0, 0, ny, nx):
            return  # already tight
        yslc, xslc = slice(y0, y1), slice(x0, x1)

        for attr in ('data', 'err', 'dq', 'wht', 'con', 'var_poisson',
                     'var_rnoise', 'var_flat', 'area'):
            if hasattr(m, attr):
                setattr(m, attr, _crop_to_slices(getattr(m, attr), ny, nx, yslc, xslc))
        ny_c, nx_c = (y1 - y0), (x1 - x0)
        shifted = _shift_gwcs(m.meta.wcs, x0, y0)
        shifted.bounding_box = ((-0.5, nx_c - 0.5), (-0.5, ny_c - 0.5))
        m.meta.wcs = shifted
        m.save(filename)

    # Keep the SCI FITS WCS consistent for astropy consumers (i2d is TAN).
    with fits.open(filename, mode='update') as h:
        for ext in ('SCI', 'ERR', 'CON', 'WHT', 'VAR_POISSON', 'VAR_RNOISE',
                    'VAR_FLAT', 'AREA'):
            if ext in h and 'CRPIX1' in h[ext].header:
                h[ext].header['CRPIX1'] -= x0
                h[ext].header['CRPIX2'] -= y0
        h.flush()
    print(f"cutout: cropped {os.path.basename(filename)} to finite region "
          f"[{x0}:{x1},{y0}:{y1}] ({nx_c}x{ny_c})", flush=True)


# _make_model_image is imported from photometry/psf_fitting.py.

import crowdsource
from crowdsource import crowdsource_base
from crowdsource.crowdsource_base import fit_im, psfmod

from jwst_gc_pipeline.reduction.saturated_star_finding import (
    remove_saturated_stars, correct_dq_first_group_saturation)

from astroquery.svo_fps import SvoFps

import pylab as pl
pl.rcParams['figure.facecolor'] = 'w'
pl.rcParams['image.origin'] = 'lower'

import os
print("Importing webbpsf", flush=True)
import stpsf as webbpsf
import stpsf
print(f"Webbpsf version: {webbpsf.__version__}")
from stpsf.utils import to_griddedpsfmodel
import datetime
print("Done with imports", flush=True)

FWHM_TABLE = Path(__file__).resolve().parents[1] / 'reduction' / 'fwhm_table.ecsv'
REGIONS_DIR = Path(__file__).resolve().parents[2] / 'regions_'

# MIRI_FILTERS / _instrument_from_filter / _inst_token are imported from
# photometry/naming.py (see top-of-file import) so there is one source of truth.

# DQ flags to drop from photometry.  DO_NOT_USE is the umbrella set
# the JWST pipeline already curates; SATURATED is added explicitly so
# the saturated-star branch can split it out for special handling.
# MIRI gets NON_SCIENCE (imager region masks) and PERSISTENCE (latent
# images from prior bright sources, common in MIRI long-wavelength).
_BAD_DQ_FLAGS_NIRCAM = ('DO_NOT_USE', 'SATURATED')
_BAD_DQ_FLAGS_MIRI = ('DO_NOT_USE', 'SATURATED', 'NON_SCIENCE', 'PERSISTENCE')

def _bad_dq_bitmask(instrument):
    from jwst.datamodels import dqflags as _dq
    flags = _BAD_DQ_FLAGS_MIRI if str(instrument).upper() == 'MIRI' else _BAD_DQ_FLAGS_NIRCAM
    bm = 0
    for f in flags:
        bm |= int(_dq.pixel[f])
    return bm


def normalize_vgroup_id(vgroup_id):
    if vgroup_id is None or vgroup_id == '':
        return '', None

    vgroup_token = str(vgroup_id)
    if vgroup_token.startswith('_vgroup'):
        vgroup_token = vgroup_token.removeprefix('_vgroup')

    if vgroup_token.isdigit():
        return f'_vgroup{vgroup_token}', int(vgroup_token)

    digit_match = re.search(r'\d+', vgroup_token)
    if digit_match is not None:
        return f'_vgroup{vgroup_token}', int(digit_match.group(0))

    return f'_vgroup{vgroup_token}', None


def print(*args, **kwargs):
    now = datetime.datetime.now().isoformat()
    from builtins import print as printfunc
    return printfunc(f"{now}:", *args, **kwargs)


class WrappedPSFModel(crowdsource.psf.SimplePSF):
    """
    wrapper for photutils GriddedPSFModel
    """
    def __init__(self, psfgridmodel, stampsz=19):
        self.psfgridmodel = psfgridmodel
        self.default_stampsz = stampsz

    def __call__(self, col, row, stampsz=None, deriv=False):

        if stampsz is None:
            stampsz = self.default_stampsz

        parshape = numpy.broadcast(col, row).shape
        tparshape = parshape if len(parshape) > 0 else (1,)

        # numpy uses row, column notation
        rows, cols = np.indices((stampsz, stampsz)) - (np.array([stampsz, stampsz])-1)[:, None, None] / 2.

        # explicitly broadcast
        col = np.atleast_1d(col)
        row = np.atleast_1d(row)
        #rows = rows[:, :, None] + row[None, None, :]
        #cols = cols[:, :, None] + col[None, None, :]

        # photutils seems to use column, row notation
        # only works with photutils <= 1.6.0 - but is wrong there
        #stamps = self.psfgridmodel.evaluate(cols, rows, 1, col, row)
        # it returns something in (nstamps, row, col) shape
        # pretty sure that ought to be (col, row, nstamps) for crowdsource

        # andrew saydjari's version here:
        # it returns something in (nstamps, row, col) shape
        #
        # NOTE: this loop CANNOT be batched into a single
        # ``self.psfgridmodel.evaluate(...)`` call with vector
        # ``x_0``/``y_0``.  ``photutils.psf.GriddedPSFModel.evaluate``
        # explicitly forces those args to scalars (``if not
        # np.isscalar(x_0): x_0 = x_0[0]``) because the bilinear
        # interpolation between adjacent grid ePSFs is computed for a
        # single subpixel offset per call.  Vectorising would require a
        # custom interpolator that reproduces ``_calc_model_values`` for
        # an array of (x_0, y_0) at once and bypasses photutils.  Not
        # worth the maintenance cost in this code path -- WrappedPSFModel
        # is only used by the crowdsource fit_im branch, which the main
        # iter3 pipeline no longer runs (daophot path is preferred).
        stamps = []
        for i in range(len(col)):
            # the +0.5 is required to actually center the PSF (empirically)
            #stamps.append(self.psfgridmodel.evaluate(cols+col[i]+0.5, rows+row[i]+0.5, 1, col[i], row[i]))
            # the above may have been true when we were using (incorrectly) offset PSFs
            stamps.append(self.psfgridmodel.evaluate(cols+col[i], rows+row[i], 1, col[i], row[i]))

        stamps = np.array(stamps)

        # for oversampled stamps, they may not be normalized
        stamps /= stamps.sum(axis=(1,2))[:,None,None]
        # this is evidently an incorrect transpose
        #stamps = np.transpose(stamps, axes=(0,2,1))

        if deriv:
            dpsfdrow, dpsfdcol = np.gradient(stamps, axis=(1, 2))

        ret = stamps
        if parshape != tparshape:
            ret = ret.reshape(stampsz, stampsz)
            if deriv:
                dpsfdrow = dpsfdrow.reshape(stampsz, stampsz)
                dpsfdcol = dpsfdcol.reshape(stampsz, stampsz)
        if deriv:
            ret = (ret, dpsfdcol, dpsfdrow)

        return ret

    def render_model(self, col, row, stampsz=None):
        """
        this function likely does nothing?
        """
        if stampsz is not None:
            self.stampsz = stampsz

        rows, cols = np.indices(self.stampsz, dtype=float) - (np.array(self.stampsz)-1)[:, None, None] / 2.

        return self.psfgridmodel.evaluate(cols, rows, 1, col, row).T.squeeze()


def write_via_local_scratch(final_path, write_fn):
    """Build a per-frame product on node-local ``$SLURM_TMPDIR``, then copy it to
    ``final_path`` on shared storage.

    The many-small-write FITS/table construction (header padding, table
    serialisation, checksums) lands on fast local disk, and only a single
    sequential streaming copy touches the shared filesystem.  This cuts the
    random-I/O + metadata burst that ``N`` array-tasks x ``M`` workers create
    during the per-frame m12 write storm -- the failure mode behind the
    multi-hour shared-FS stall documented in ``PERFORMANCE_BRICK.md``.

    ``write_fn`` is a callable taking a path and writing the product there, e.g.
    ``lambda p: tbl.write(p, overwrite=True)``.  Falls back to writing
    ``final_path`` directly when no local scratch is configured, or if the
    local write/copy fails (e.g. scratch full), so behaviour is unchanged off
    SLURM and never loses the output.
    """
    scratch = os.environ.get('SLURM_TMPDIR')
    if not scratch or not os.path.isdir(scratch):
        write_fn(final_path)
        return
    import tempfile
    import shutil
    fd, tmp = tempfile.mkstemp(prefix='wlc_',
                               suffix='_' + os.path.basename(final_path),
                               dir=scratch)
    os.close(fd)
    try:
        write_fn(tmp)
        shutil.copyfile(tmp, final_path)
    except OSError as ex:
        print(f"[io] local-scratch write for {final_path} failed ({ex}); "
              f"writing directly to shared storage", flush=True)
        write_fn(final_path)
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def save_epsf(epsf, filename, overwrite=True):
    header = {}
    header['OVERSAMP'] = list(epsf.oversampling)
    hdu = fits.PrimaryHDU(data=epsf.data, header=header)
    hdu.writeto(filename, overwrite=overwrite)


def read_epsf(filename):
    fh = fits.open(filename)
    hdu = fh[0]
    return EPSFModel(data=hdu.data, oversampling=hdu.header['OVERSAMP'])


# Set True for --cutout-region runs: diagnostic PNGs are disabled (the fixed
# zoom regions rarely intersect a small cutout and the plots aren't useful at
# that scale).  Only affects cutout runs; full-frame diagnostics are unchanged.
_SUPPRESS_DIAGNOSTICS = False


def _noop_savefig(*args, **kwargs):
    return None


def catalog_zoom_diagnostic(data, modsky, zoomcut, stars):

    # Disabled entirely for cutout runs (see _SUPPRESS_DIAGNOSTICS).
    if _SUPPRESS_DIAGNOSTICS:
        return

    # A fixed zoom region (e.g. slice(128,256)) may not intersect the image
    # -- notably for a small --cutout-region run -- leaving an empty array
    # that crashes simple_norm's percentile.  These diagnostics are
    # non-essential, so skip cleanly when the zoom is empty.
    if np.asarray(data[zoomcut]).size == 0:
        print(f"catalog_zoom_diagnostic: zoom region {zoomcut} is empty for "
              f"data shape {np.asarray(data).shape}; skipping plot", flush=True)
        return

    # make sure stars is a table
    try:
        'qf' in stars.colnames
    except AttributeError:
        stars = Table(stars)

    pl.figure(figsize=(12,12))
    im = pl.subplot(2,2,1).imshow(data[zoomcut],
                                  norm=simple_norm(data[zoomcut],
                                                   stretch='log',
                                                   max_percent=99.95,
                                                   vmin=0), cmap='gray')
    pl.xticks([]); pl.yticks([]); pl.title("Data")
    pl.colorbar(mappable=im)
    im = pl.subplot(2,2,2).imshow(modsky[zoomcut],
                                  norm=simple_norm(modsky[zoomcut],
                                                   stretch='log',
                                                   max_percent=99.95,
                                                   vmin=0), cmap='gray')
    pl.xticks([]); pl.yticks([]); pl.title("fit_im model+sky")
    pl.colorbar(mappable=im)

    resid = (data[zoomcut] - modsky[zoomcut])
    rms = stats.mad_std(resid, ignore_nan=True)
    if np.isnan(rms):
        raise ValueError("RMS is nan, this shouldn't happen")

    norm = (simple_norm(resid, stretch='asinh', max_percent=99.95, min_percent=0.5)
            if np.nanmin(resid) > 0 else
            simple_norm(resid, stretch='log', vmax=np.nanpercentile(resid, 99.95), vmin=-2*rms))

    im = pl.subplot(2,2,3).imshow(resid,
                                  norm=norm,
                                  cmap='gray')
    pl.xticks([]); pl.yticks([]); pl.title(f"data-modsky (rms={rms:10.3g})")
    pl.colorbar(mappable=im)
    im = pl.subplot(2,2,4).imshow(data[zoomcut],
                                  norm=simple_norm(data[zoomcut],
                                                   stretch='log',
                                                   max_percent=99.95,
                                                   vmin=0), cmap='gray')

    if 'qf' in stars.colnames:
        # used in analysis
        qgood = ((stars['qf'] > 0.9) &
                 (stars['spread_model'] < 0.25) &
                 (stars['fracflux'] > 0.75)
                )
        neg = stars['flux'] < 0
    elif 'qfit' in stars.colnames:
        # guesses, no tests don
        qgood = ((stars['qfit'] < 0.4) &
                 (stars['cfit'] < 0.1) &
                 (stars['flags'] == 0))
        neg = stars['flux_fit'] < 0
    else:
        qgood = np.ones(len(stars), dtype='bool')
        neg = np.zeros(len(stars), dtype='bool')

    axlims = pl.axis()
    if zoomcut[0].start:
        # pl.axis([0,zoomcut[0].stop-zoomcut[0].start, 0, zoomcut[1].stop-zoomcut[1].start])
        ok = ((stars['x'] >= zoomcut[1].start) &
              (stars['x'] <= zoomcut[1].stop) &
              (stars['y'] >= zoomcut[0].start) &
              (stars['y'] <= zoomcut[0].stop))
        pl.subplot(2,2,4).scatter(stars['x'][ok & ~qgood]-zoomcut[1].start,
                                  stars['y'][ok & ~qgood]-zoomcut[0].start,
                                  marker='+', color='y', s=8, linewidth=0.5)
        pl.subplot(2,2,4).scatter(stars['x'][ok & qgood]-zoomcut[1].start,
                                  stars['y'][ok & qgood]-zoomcut[0].start,
                                  marker='x', color='r', s=8, linewidth=0.5)
        pl.subplot(2,2,4).scatter(stars['x'][neg]-zoomcut[1].start,
                                  stars['y'][neg]-zoomcut[0].start,
                                  marker='1', color='b', s=8, linewidth=0.5)
    else:
        pl.subplot(2,2,4).scatter(stars['x'][~qgood],
                                  stars['y'][~qgood], marker='+', color='lime', s=5, linewidth=0.5)
        pl.subplot(2,2,4).scatter(stars['x'][qgood],
                                  stars['y'][qgood], marker='x', color='r', s=5, linewidth=0.5)
        pl.subplot(2,2,4).scatter(stars['x'][neg],
                                  stars['y'][neg], marker='1', color='b', s=5, linewidth=0.5)
    pl.axis(axlims)
    pl.xticks([]); pl.yticks([]); pl.title("Data with stars");
    pl.colorbar(mappable=im)
    pl.tight_layout()


# Table column-convention resolvers factored into photometry/column_utils.py
# (bloat refactor).  Imported here so existing references (_get_source_xy etc.,
# and _L.<name> access from cataloging.py) keep working unchanged.
from jwst_gc_pipeline.photometry.column_utils import (
    _XY_COLUMN_CANDIDATES, _get_source_xy, _column_to_float_array,
    _best_available_xy, _has_any_xy_columns, _skycoord_radec_arrays,
)


def _resolve_seed_skycoords(seed_table, ww=None, preferred_skycoord_col=None):
    """Ensure ``seed_table`` has a vectorised ``SkyCoord``-mixin
    ``skycoord`` column suitable for direct ``ww.world_to_pixel`` and
    bulk ``.ra.deg`` / ``.dec.deg`` access.

    Resolution order (each is one vector SkyCoord construction):
      1. Already a SkyCoord-mixin ``skycoord`` column -> verify no masked
         rows; if any are masked, fall through to step 3 to backfill.
      2. Plain ``ra``/``dec`` columns -> build mixin from those.
         (Common for union seed catalogues built by build_union_seed_catalog.py.)
      3. Merge across SkyCoord-mixin candidate columns, taking the first
         UNMASKED value per row (preferred col first).  Critical when
         vstacking heterogeneous tables (e.g. iter1 daophot rows carry
         ``skycoord_centroid``, satstar rows carry ``skycoord_fit``;
         each column is masked on the rows belonging to the other table).
         The previous implementation picked the first existing column by
         name and called ``.unmasked``, which exposed the (0,0) fill
         sentinel for masked rows.  Star B regression 2026-06-02.
      4. Only ``(x, y)`` available + a WCS -> bulk ``ww.pixel_to_world``
         backfill for any rows still missing sky.

    Object-dtype columns of SkyCoord scalars are NOT supported: producing
    one is a bug at the call site (it forces a per-row Python loop in
    every consumer).  Raise loudly if no resolution path applies.
    """
    seed_table = _as_table(seed_table)
    nsrc = len(seed_table)
    if nsrc == 0:
        return seed_table

    def _column_mask(col):
        """Return per-row mask (True=masked) for a SkyCoord mixin column."""
        m = np.zeros(len(col), dtype=bool)
        for axis in ('ra', 'dec'):
            sub = getattr(col, axis, None)
            sub_mask = getattr(sub, 'mask', None)
            if sub_mask is None:
                continue
            sub_mask = np.asarray(sub_mask)
            if sub_mask.shape == m.shape:
                m |= sub_mask
        return m

    # 1. Already a SkyCoord column.  Accept ONLY if no rows are masked
    # (otherwise (0,0) sentinel rows would silently pass through to
    # world_to_pixel).
    if ('skycoord' in seed_table.colnames
            and isinstance(seed_table['skycoord'], SkyCoord)):
        if not np.any(_column_mask(seed_table['skycoord'])):
            return seed_table

    # 2. Plain ra/dec columns -> single vector SkyCoord construction.
    # IMPORTANT: only short-circuit on ra/dec when no preferred SkyCoord-mixin
    # column is requested.  Otherwise the iter3 per-filter seed snap (which
    # adds ``skycoord_{filter}`` to override the union's SW-only ra/dec
    # astrometry) would be ignored.  Detected 2026-06-03 on sickle F480M
    # iter3 source 55 init at unsnapped pix (310.26,126.62) despite union
    # row 13911 having been correctly snapped to (311.80,127.07): consumer's
    # _resolve_seed_skycoords hit ra/dec fallback before checking
    # skycoord_f480m.
    if ('skycoord' not in seed_table.colnames
            and 'ra' in seed_table.colnames and 'dec' in seed_table.colnames
            and preferred_skycoord_col is None):
        ra_arr = np.asarray(seed_table['ra'], dtype=float)
        dec_arr = np.asarray(seed_table['dec'], dtype=float)
        seed_table['skycoord'] = SkyCoord(ra=ra_arr * u.deg,
                                          dec=dec_arr * u.deg,
                                          frame='icrs')
        return seed_table

    # 3. Merge across candidate SkyCoord-mixin columns.  For each row,
    # take the first UNMASKED ra/dec across the candidate list.  Rows
    # with no valid candidate are left as NaN so downstream callers
    # filter them explicitly (not silently via a (0,0) fill).
    sky_columns = []
    if preferred_skycoord_col is not None:
        sky_columns.append(preferred_skycoord_col)
    sky_columns.extend(['skycoord', 'skycoord_fit',
                        'skycoord_centroid', 'skycoord_ref'])
    ra_master = np.full(nsrc, np.nan, dtype=float)
    dec_master = np.full(nsrc, np.nan, dtype=float)
    for colname in sky_columns:
        if colname not in seed_table.colnames:
            continue
        col = seed_table[colname]
        if not isinstance(col, SkyCoord):
            continue
        mask = _column_mask(col)
        unmasked_col = col.unmasked if hasattr(col, 'unmasked') else col
        col_ra = np.asarray(unmasked_col.ra.deg, dtype=float)
        col_dec = np.asarray(unmasked_col.dec.deg, dtype=float)
        need = np.isnan(ra_master) & ~mask
        if np.any(need):
            ra_master[need] = col_ra[need]
            dec_master[need] = col_dec[need]
        if not np.any(np.isnan(ra_master)):
            break

    # 4. Backfill any still-NaN rows from (x, y) + ww when available.
    if ww is not None and np.any(np.isnan(ra_master)) and _has_any_xy_columns(seed_table):
        xvals, yvals = _best_available_xy(seed_table)
        need = (np.isnan(ra_master)
                & np.isfinite(np.asarray(xvals, dtype=float))
                & np.isfinite(np.asarray(yvals, dtype=float)))
        if np.any(need):
            derived = ww.pixel_to_world(np.asarray(xvals)[need],
                                        np.asarray(yvals)[need])
            ra_master[need] = np.asarray(derived.ra.deg, dtype=float)
            dec_master[need] = np.asarray(derived.dec.deg, dtype=float)

    if (not np.any(np.isfinite(ra_master))
            and not any(c in seed_table.colnames
                        for c in (['ra', 'dec', 'skycoord']
                                  + sky_columns
                                  + (['x', 'y'] if ww is not None else [])))):
        raise ValueError(
            'Could not determine sky coordinates: seed table has no '
            f'skycoord/ra/dec/(x,y)+ww input. Columns: {seed_table.colnames}'
        )

    seed_table['skycoord'] = SkyCoord(ra=ra_master * u.deg,
                                      dec=dec_master * u.deg,
                                      frame='icrs')
    return seed_table


def _sample_background_map(background_map, xvals, yvals):
    """Sample a 2D background image at source coordinates using nearest-neighbor lookup."""
    sampled = np.full(len(xvals), np.nan, dtype='float32')
    if background_map is None:
        return sampled

    xi = np.rint(np.asarray(xvals)).astype(int)
    yi = np.rint(np.asarray(yvals)).astype(int)
    inbounds = ((xi >= 0) & (yi >= 0) &
                (yi < background_map.shape[0]) &
                (xi < background_map.shape[1]))
    sampled[inbounds] = background_map[yi[inbounds], xi[inbounds]]
    return sampled


# _iteration_token / _bgsub_token are imported from photometry/naming.py (see
# the import near the top of this module, beside _strip_chunk).


def obs_token(proposal_id, field):
    """Per-observation filename disambiguator for multi-obs targets.

    Proposal 2211 (gc2211) comprises 5 GC pointings (obs 023/028/046/049/050)
    that REUSE the same ``(visit, vgroup, exp)`` tuples, so the obs-less per-frame
    catalog-table name ``{filter}_{module}_visit001_vgroup02201_exp00001_...`` is
    identical across obs that share a filter and silently overwrites (= data loss;
    F200W: o023/o046/o049/o050; F277W: all 5).  Insert ``_o{field}`` for prop 2211
    so each obs writes a distinct catalog table.  The per-frame residual/model
    products under ``{filter}/pipeline/`` already carry ``-o{field}`` and are
    unaffected.  Other proposals are single-obs-per-basepath and get the empty
    token, so their filenames and existing products are unchanged.
    """
    if str(proposal_id) == '2211' and field not in (None, ''):
        return f'_o{field}'
    # ngc6334: TWO proposals (7213 + 6778) share the same target dir and share
    # the filters F200W + F470N, cataloged with the SAME obs number (001) and the
    # SAME (visit, vgroup, exp) tuples -- so the per-frame catalog-table name
    # ``f200w_{module}_visit001_vgroup02101_exp00001_...`` is identical across the
    # two proposals and the second run silently overwrites the first (= data loss:
    # 6778 clobbered 7213's F200W/F470N catalogs, 2026-07-09).  These two proposal
    # ids appear ONLY in ngc6334, so disambiguate by proposal id (no field/target
    # threading needed).  Non-shared filters get the token too (harmless -- one
    # proposal per filter -> nothing to collide with), keeping the scheme uniform.
    if str(proposal_id) in ('7213', '6778'):
        return f'_j{proposal_id}'
    return ''


def _obs_token_from_options(options):
    return obs_token(getattr(options, 'proposal_id', None),
                     getattr(options, 'field', None))


def _predict_output_tokens(options, visit_id=None, vgroup_id=None,
                           exposure_id=None, iteration_label=None):
    """Reproduce the per-exposure tokens used when writing catalog outputs.

    Kept in sync with save_photutils_results / save_crowdsource_results /
    do_photometry_step so --skip-if-done and --list-missing-tasks can check
    whether the expected output already exists without running photometry.
    """
    visitid_ = f'_visit{int(visit_id):03d}' if visit_id not in (None, '') else ''
    vgroupid_, _ = normalize_vgroup_id(vgroup_id)
    if exposure_id in (None, ''):
        exposure_ = ''
    else:
        exposure_ = f'_exp{int(exposure_id):05d}'
    desat = '_unsatstar' if options.desaturated else ''
    bgsub = _bgsub_token(options)
    epsf_ = '_epsf' if options.epsf else ''
    blur_ = '_blur' if options.blur else ''
    group_ = '_group' if options.group else ''
    if iteration_label is None:
        iter_label = options.iteration_label or None
    else:
        iter_label = iteration_label if iteration_label != '' else None
    iter_ = _iteration_token(iter_label)
    return visitid_, vgroupid_, exposure_, desat, bgsub, epsf_, blur_, group_, iter_


def _predict_tblfilename(basepath, filtername, module, options,
                         visit_id, vgroup_id, exposure_id,
                         iteration_label=None, method='daophot',
                         basic_or_iterative='iterative'):
    (visitid_, vgroupid_, exposure_, desat, bgsub,
     epsf_, blur_, group_, iter_) = _predict_output_tokens(
        options, visit_id, vgroup_id, exposure_id, iteration_label)
    obs_ = _obs_token_from_options(options)
    if method == 'daophot':
        return (f'{basepath}/{filtername}/'
                f'{filtername.lower()}_{module}{obs_}{visitid_}{vgroupid_}{exposure_}'
                f'{desat}{bgsub}{epsf_}{blur_}{group_}{iter_}'
                f'_daophot_{basic_or_iterative}.fits')
    return (f'{basepath}/{filtername}/'
            f'{filtername.lower()}_{module}{obs_}{visitid_}{vgroupid_}{exposure_}'
            f'{desat}{bgsub}{blur_}{iter_}'
            f'_crowdsource_unweighted.fits')


def _expected_output_exists(basepath, filtername, module, options,
                            visit_id, vgroup_id, exposure_id,
                            iteration_label=None):
    """Main output sentinel for --skip-if-done / --list-missing-tasks.

    daophot-iterative is the final step when --daophot is set (or basic when
    --basic-only); crowdsource_unweighted is the final step otherwise.
    """
    if options.daophot:
        method = 'daophot'
        basic_or_iterative = 'basic' if options.basic_only else 'iterative'
    else:
        method = 'crowdsource'
        basic_or_iterative = 'unweighted'
    path = _predict_tblfilename(basepath, filtername, module, options,
                                visit_id, vgroup_id, exposure_id,
                                iteration_label=iteration_label,
                                method=method,
                                basic_or_iterative=basic_or_iterative)
    return os.path.exists(path)


def _as_table(data):
    if isinstance(data, Table):
        return Table(data, copy=True)
    if isinstance(data, str):
        return Table.read(data)
    return Table(data)


def _combine_seed_and_satstars(seed_catalog, satstar_table):
    seed_table = _as_table(seed_catalog)
    if 'is_saturated' not in seed_table.colnames:
        seed_table['is_saturated'] = np.zeros(len(seed_table), dtype=bool)

    if satstar_table is None:
        return seed_table

    satstar_table = _as_table(satstar_table)
    if len(satstar_table) == 0:
        return seed_table

    if 'is_saturated' not in satstar_table.colnames:
        satstar_table['is_saturated'] = np.ones(len(satstar_table), dtype=bool)

    return vstack([seed_table, satstar_table], metadata_conflicts='silent')


def _augment_seed_catalog_with_detections(seed_catalog, detection_catalog, match_radius_pix=1.0):
    raise RuntimeError('Use _augment_seed_catalog_with_detections_sky for seeded augmentation')


def _augment_seed_catalog_with_detections_sky(seed_catalog, detection_catalog, ww,
                                              match_radius_pix=1.0,
                                              preferred_seed_skycoord_col=None,
                                              return_stats=False):
    seed_table = _resolve_seed_skycoords(_as_table(seed_catalog), ww=ww,
                                         preferred_skycoord_col=preferred_seed_skycoord_col)
    detection_table = _as_table(detection_catalog)
    stats = {
        'seed_input': len(seed_table),
        'detection_input': len(detection_table),
        'detection_finite_xy': 0,
        'detection_added': 0,
        'detection_rejected_match': 0,
    }

    if len(seed_table) == 0:
        stats['detection_added'] = len(detection_table)
        if return_stats:
            return detection_table, stats
        return detection_table
    if len(detection_table) == 0:
        if return_stats:
            return seed_table, stats
        return seed_table

    det_x, det_y = _best_available_xy(detection_table)
    det_finite = np.isfinite(det_x) & np.isfinite(det_y)
    if not np.any(det_finite):
        if return_stats:
            return seed_table, stats
        return seed_table

    stats['detection_finite_xy'] = int(np.sum(det_finite))

    det_sky = ww.pixel_to_world(det_x[det_finite], det_y[det_finite])
    det_ra = np.asarray(det_sky.ra.deg, dtype=float)
    det_dec = np.asarray(det_sky.dec.deg, dtype=float)
    detection_table = detection_table[det_finite]
    # Assign the SkyCoord array directly as a mixin column.  The previous
    # per-cell loop produced an object-dtype list of scalar SkyCoords which
    # forced every downstream consumer into a Python-level per-row scan.
    detection_table['skycoord'] = det_sky
    if 'is_saturated' not in detection_table.colnames:
        detection_table['is_saturated'] = np.zeros(len(detection_table), dtype=bool)

    seed_ra, seed_dec = _skycoord_radec_arrays(seed_table, 'skycoord')
    valid_seed_idx = np.isfinite(seed_ra) & np.isfinite(seed_dec)
    if not np.any(valid_seed_idx):
        combined = vstack([seed_table, detection_table], metadata_conflicts='silent')
        if 'is_saturated' not in combined.colnames:
            combined['is_saturated'] = np.zeros(len(combined), dtype=bool)
        stats['detection_added'] = len(detection_table)
        stats['detection_rejected_match'] = 0
        if return_stats:
            return combined, stats
        return combined

    seed_sky = SkyCoord(ra=seed_ra[valid_seed_idx] * u.deg,
                        dec=seed_dec[valid_seed_idx] * u.deg,
                        frame='icrs')
    det_sky_all = SkyCoord(ra=det_ra * u.deg,
                           dec=det_dec * u.deg,
                           frame='icrs')
    _, sep2d, _ = det_sky_all.match_to_catalog_sky(seed_sky)
    pixscale = ww.proj_plane_pixel_area()**0.5
    match_radius = (match_radius_pix * pixscale).to(u.arcsec)
    keep = sep2d > match_radius

    stats['detection_added'] = int(np.sum(keep))
    stats['detection_rejected_match'] = int(len(keep) - np.sum(keep))

    combined = vstack([seed_table, detection_table[keep]], metadata_conflicts='silent')
    if 'is_saturated' not in combined.colnames:
        combined['is_saturated'] = np.zeros(len(combined), dtype=bool)
    if return_stats:
        return combined, stats
    return combined


def _protect_mask(x, y, protect_xy, protect_radius_pix):
    """Boolean mask over (x, y) fits within ``protect_radius_pix`` of any coadd-
    confirmed seed position in ``protect_xy`` (Nx2 pixel array).  Used to exempt
    real, independently-confirmed stars from the satstar-wing / near-saturation
    drops in dense fields (W51 IRS2): a real star on a bright neighbour's wing
    would otherwise be culled as a wing artifact."""
    if protect_xy is None or len(protect_xy) == 0 or protect_radius_pix <= 0:
        return np.zeros(len(x), dtype=bool)
    from scipy.spatial import cKDTree
    pts = np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])
    finite = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    prot = np.zeros(len(x), dtype=bool)
    if not np.any(finite):
        return prot
    tree = cKDTree(np.asarray(protect_xy, dtype=float))
    dd, _ = tree.query(pts[finite], k=1)
    prot[finite] = dd <= float(protect_radius_pix)
    return prot


def _filter_near_saturation(phot_obj, dq, *, max_sat_dist_pix,
                            label, max_log_rows=50,
                            protect_xy=None, protect_radius_pix=0.0):
    """Drop fits from ``phot_obj.results`` whose center is within
    ``max_sat_dist_pix`` pixels of any SATURATED-DQ pixel and keep the
    PSFPhotometry object's state consistent.

    Rationale: regular ``phot_basic``/``phot_iter`` fits placed on a
    saturated pixel are unreliable -- the central data value is "stuck"
    while the unsaturated wings drive the fit toward enormous fluxes,
    producing model-image holes of order -10 000 counts.  The dedicated
    ``satstar`` catalog handles those stars separately and lives in a
    different table, so this filter does not touch it.

    A no-op when ``dq`` is None or has no SATURATED pixels.
    """
    if dq is None:
        return 0
    sat_mask = (dq & dqflags.pixel['SATURATED']).astype(bool)
    n_sat = int(sat_mask.sum())
    if n_sat == 0:
        return 0
    # distance_transform_edt: distance to nearest True in the input mask.
    # We want distance to nearest saturated pixel, so feed ~sat_mask.
    sat_dist_map = ndimage.distance_transform_edt(~sat_mask)

    res = phot_obj.results
    if res is None or len(res) == 0:
        return 0

    x = np.asarray(res['x_fit'], dtype=float)
    y = np.asarray(res['y_fit'], dtype=float)
    flux_arr = np.asarray(res['flux_fit'], dtype=float)
    ny, nx = sat_mask.shape
    ix = np.rint(x).astype(int)
    iy = np.rint(y).astype(int)
    in_frame = (np.isfinite(x) & np.isfinite(y)
                & (ix >= 0) & (ix < nx)
                & (iy >= 0) & (iy < ny))

    sat_dist = np.full(len(res), np.inf, dtype=float)
    if np.any(in_frame):
        sat_dist[in_frame] = sat_dist_map[iy[in_frame], ix[in_frame]]

    drop = sat_dist <= float(max_sat_dist_pix)
    if drop.any() and protect_xy is not None and protect_radius_pix > 0:
        prot = _protect_mask(x, y, protect_xy, protect_radius_pix)
        n_prot = int(np.sum(drop & prot))
        if n_prot:
            drop = drop & ~prot
            print(f"Saturation-proximity filter ({label}): protected {n_prot} "
                  f"coadd-confirmed seeds from drop", flush=True)
    n_drop = int(np.sum(drop))
    if n_drop == 0:
        return 0

    print(f"Saturation-proximity filter ({label}): dropping {n_drop} fits "
          f"within {max_sat_dist_pix:.1f} pix of a SATURATED-DQ pixel "
          f"({len(res)} -> {len(res) - n_drop}); "
          f"sat_pixels_in_frame={n_sat}", flush=True)

    # Log a sample of dropped rows for forensics.
    drop_idx = np.where(drop)[0]
    log_idx = drop_idx[np.argsort(sat_dist[drop_idx])][:max_log_rows]
    if len(log_idx) > 0:
        id_col = res['id'] if 'id' in res.colnames else np.arange(len(res))
        print(f"  dropped fits ({label}, up to {max_log_rows} closest to sat):", flush=True)
        print(f"    {'id':>6} {'x_fit':>9} {'y_fit':>9} {'flux_fit':>12} {'sat_dist':>9}",
              flush=True)
        for i in log_idx:
            sid = id_col[i]
            print(f"    {int(sid):>6d} {x[i]:>9.2f} {y[i]:>9.2f} "
                  f"{flux_arr[i]:>12.2f} {sat_dist[i]:>9.2f}", flush=True)
        if len(drop_idx) > len(log_idx):
            print(f"    ... ({len(drop_idx) - len(log_idx)} more not shown)",
                  flush=True)

    keep = ~drop
    phot_obj.results = phot_obj.results[keep]
    inner_phot = getattr(phot_obj, '_psfphot', None)
    if (inner_phot is not None
            and inner_phot.init_params is not None
            and len(inner_phot.init_params) == len(keep)):
        inner_phot.init_params = inner_phot.init_params[keep]
    if (hasattr(phot_obj, 'init_params')
            and phot_obj.init_params is not None
            and len(phot_obj.init_params) == len(keep)):
        phot_obj.init_params = phot_obj.init_params[keep]

    # IterativePSFPhotometry rebuilds its _model_image_params from the
    # per-iteration deepcopied snapshots in ``fit_results``, not from
    # ``self.results`` -- so updating ``self.results`` alone leaves the
    # rendered model image unchanged.  Filter each per-iteration snapshot
    # by the same sat-distance rule so make_model_image() agrees with
    # the saved catalog.
    fit_results = getattr(phot_obj, 'fit_results', None)
    if fit_results:
        for fr in fit_results:
            sub = getattr(fr, 'results', None)
            if sub is None or len(sub) == 0:
                continue
            sx = np.asarray(sub['x_fit'], dtype=float)
            sy = np.asarray(sub['y_fit'], dtype=float)
            s_ix = np.rint(sx).astype(int)
            s_iy = np.rint(sy).astype(int)
            s_in = (np.isfinite(sx) & np.isfinite(sy)
                    & (s_ix >= 0) & (s_ix < nx)
                    & (s_iy >= 0) & (s_iy < ny))
            s_dist = np.full(len(sub), np.inf, dtype=float)
            if np.any(s_in):
                s_dist[s_in] = sat_dist_map[s_iy[s_in], s_ix[s_in]]
            sub_keep = s_dist > float(max_sat_dist_pix)
            if sub_keep.sum() < len(sub):
                fr.results = sub[sub_keep]
                if (fr.init_params is not None
                        and len(fr.init_params) == len(sub_keep)):
                    fr.init_params = fr.init_params[sub_keep]
                fr.__dict__.pop('_model_image_params', None)

    phot_obj.__dict__.pop('_model_image_params', None)
    return n_drop


def _filter_satstar_artifacts(phot_obj, satstar_model, err, *,
                              sig_K, ratio_cut, label, max_log_rows=50,
                              protect_xy=None, protect_radius_pix=0.0):
    """Drop fits sitting inside significant satstar PSF wings whose own
    model contribution is < ``ratio_cut`` x ``satstar_model`` at that pixel.

    Gate: only fits where ``satstar_model[y,x] > sig_K * median(err)`` are
    candidates.  Within the gate, fits with ``dao_model[y,x] / satstar_model[y,x]
    < ratio_cut`` are dropped as PSF-wing artifacts.  The dao_model is
    rendered from ``phot_obj`` itself; we then re-render after filtering
    via the standard ``_model_image_params`` cache invalidation.

    Rationale: bright-star PSF wings carry enough flux that DAOStarFinder
    triggers along them and PSFPhotometry then fits "stars" whose total
    contribution at the detection pixel is smaller than the satstar wing
    already modeled there.  Those fits double-count satstar flux.
    """
    if satstar_model is None or ratio_cut <= 0:
        return 0
    res = phot_obj.results
    if res is None or len(res) == 0:
        return 0
    err_finite = err[np.isfinite(err) & (err > 0)]
    if err_finite.size == 0:
        return 0
    err_med = float(np.median(err_finite))
    thresh = sig_K * err_med

    modsky = _make_model_image(phot_obj, satstar_model.shape,
                               psf_shape=(21, 21), include_local_bkg=False)

    x = np.asarray(res['x_fit'], dtype=float)
    y = np.asarray(res['y_fit'], dtype=float)
    flux_arr = np.asarray(res['flux_fit'], dtype=float)
    ny, nx = satstar_model.shape
    ix = np.rint(x).astype(int)
    iy = np.rint(y).astype(int)
    in_frame = (np.isfinite(x) & np.isfinite(y)
                & (ix >= 0) & (ix < nx)
                & (iy >= 0) & (iy < ny))
    sat_val = np.zeros(len(res), dtype=float)
    dao_val = np.zeros(len(res), dtype=float)
    if np.any(in_frame):
        sat_val[in_frame] = satstar_model[iy[in_frame], ix[in_frame]]
        dao_val[in_frame] = modsky[iy[in_frame], ix[in_frame]]
    in_gate = np.isfinite(sat_val) & (sat_val > thresh)
    safe_sat = np.where(sat_val > 0, sat_val, np.nan)
    ratio = np.where(in_gate, dao_val / safe_sat, np.inf)
    drop = in_gate & np.isfinite(ratio) & (ratio < ratio_cut)
    if drop.any() and protect_xy is not None and protect_radius_pix > 0:
        prot = _protect_mask(x, y, protect_xy, protect_radius_pix)
        n_prot = int(np.sum(drop & prot))
        if n_prot:
            drop = drop & ~prot
            print(f"Satstar-artifact filter ({label}): protected {n_prot} "
                  f"coadd-confirmed seeds from drop", flush=True)
    n_drop = int(np.sum(drop))
    if n_drop == 0:
        print(f"Satstar-artifact filter ({label}): no drops "
              f"(sig_K={sig_K}, ratio_cut={ratio_cut}, "
              f"in_gate={int(in_gate.sum())}, err_med={err_med:.3g})",
              flush=True)
        return 0

    print(f"Satstar-artifact filter ({label}): dropping {n_drop} fits "
          f"with dao_model < {ratio_cut:.2f}*sat_model on sat>{sig_K:.1f}*err_med "
          f"(err_med={err_med:.3g}); {len(res)} -> {len(res) - n_drop}",
          flush=True)

    drop_idx = np.where(drop)[0]
    log_idx = drop_idx[np.argsort(ratio[drop_idx])][:max_log_rows]
    if len(log_idx) > 0:
        id_col = res['id'] if 'id' in res.colnames else np.arange(len(res))
        print(f"  dropped fits ({label}, up to {max_log_rows} smallest ratio):",
              flush=True)
        print(f"    {'id':>6} {'x_fit':>9} {'y_fit':>9} {'flux_fit':>12} "
              f"{'sat_val':>9} {'dao_val':>9} {'ratio':>7}", flush=True)
        for i in log_idx:
            sid = id_col[i]
            print(f"    {int(sid):>6d} {x[i]:>9.2f} {y[i]:>9.2f} "
                  f"{flux_arr[i]:>12.2f} {sat_val[i]:>9.2f} {dao_val[i]:>9.2f} "
                  f"{ratio[i]:>7.3f}", flush=True)
        if len(drop_idx) > len(log_idx):
            print(f"    ... ({len(drop_idx) - len(log_idx)} more not shown)",
                  flush=True)

    keep = ~drop
    phot_obj.results = phot_obj.results[keep]
    inner_phot = getattr(phot_obj, '_psfphot', None)
    if (inner_phot is not None
            and inner_phot.init_params is not None
            and len(inner_phot.init_params) == len(keep)):
        inner_phot.init_params = inner_phot.init_params[keep]
    if (hasattr(phot_obj, 'init_params')
            and phot_obj.init_params is not None
            and len(phot_obj.init_params) == len(keep)):
        phot_obj.init_params = phot_obj.init_params[keep]

    fit_results = getattr(phot_obj, 'fit_results', None)
    if fit_results:
        for fr in fit_results:
            sub = getattr(fr, 'results', None)
            if sub is None or len(sub) == 0:
                continue
            sx = np.asarray(sub['x_fit'], dtype=float)
            sy = np.asarray(sub['y_fit'], dtype=float)
            s_ix = np.rint(sx).astype(int)
            s_iy = np.rint(sy).astype(int)
            s_in = (np.isfinite(sx) & np.isfinite(sy)
                    & (s_ix >= 0) & (s_ix < nx)
                    & (s_iy >= 0) & (s_iy < ny))
            s_sat = np.zeros(len(sub), dtype=float)
            s_dao = np.zeros(len(sub), dtype=float)
            if np.any(s_in):
                s_sat[s_in] = satstar_model[s_iy[s_in], s_ix[s_in]]
                s_dao[s_in] = modsky[s_iy[s_in], s_ix[s_in]]
            s_in_gate = s_sat > thresh
            s_safe = np.where(s_sat > 0, s_sat, np.nan)
            s_ratio = np.where(s_in_gate, s_dao / s_safe, np.inf)
            sub_drop = s_in_gate & np.isfinite(s_ratio) & (s_ratio < ratio_cut)
            if sub_drop.any():
                sub_keep = ~sub_drop
                fr.results = sub[sub_keep]
                if (fr.init_params is not None
                        and len(fr.init_params) == len(sub_keep)):
                    fr.init_params = fr.init_params[sub_keep]
                fr.__dict__.pop('_model_image_params', None)

    phot_obj.__dict__.pop('_model_image_params', None)
    return n_drop


# _dedup_close_sources is imported from photometry/psf_fitting.py.


class SeededFinder:
    def __init__(self, seed_table, ww=None, preferred_skycoord_col=None):
        self.seed_table = _as_table(seed_table)
        self.ww = ww
        self.preferred_skycoord_col = preferred_skycoord_col

    def __call__(self, data, mask=None):
        # Empty seed (0 rows): a genuinely source-poor frame -- common at long
        # MIRI wavelengths (e.g. w51 F2100W at 21um is almost all extended
        # emission, so a single dither can detect zero point sources).
        # _resolve_seed_skycoords early-returns an empty table WITHOUT a
        # 'skycoord' column for nsrc==0, so indexing seeds['skycoord'] below
        # would KeyError and abort the whole filter on one empty frame.  Return
        # an empty result carrying the columns downstream expects instead.
        if len(self.seed_table) == 0:
            empty = Table(self.seed_table, copy=True)
            for _c in ('flux', 'xcentroid', 'ycentroid', 'x_init', 'y_init',
                       'flux_init'):
                if _c not in empty.colnames:
                    empty[_c] = np.zeros(0, dtype=float)
            return empty
        seeds = _resolve_seed_skycoords(
            Table(self.seed_table, copy=True),
            ww=self.ww,
            preferred_skycoord_col=self.preferred_skycoord_col,
        )
        if self.ww is None:
            xvals, yvals = _best_available_xy(seeds)
        else:
            # ``_resolve_seed_skycoords`` now guarantees ``seeds['skycoord']``
            # is a vectorised SkyCoord-mixin column, so we can hand it
            # directly to ``ww.world_to_pixel`` without rebuilding a fresh
            # SkyCoord from per-row floats.  This removes the previous
            # ra/dec round-trip which dominated SeededFinder runtime
            # (~487 s on 2.5M rows; new direct path is sub-second).  NaN
            # ra/dec values pass straight through gwcs as NaN x/y and are
            # caught by the finite filter below.
            sc_col = seeds['skycoord']
            xx, yy = self.ww.world_to_pixel(sc_col)
            xvals = np.asarray(xx, dtype=float)
            yvals = np.asarray(yy, dtype=float)

        # Drop seeds whose sky->pixel projection produced NaN (typically
        # rows whose sky column was masked and got resolved to the
        # (0,0) fill before _resolve_seed_skycoords was fixed
        # 2026-06-02).  Log explicitly so silent losses surface in the
        # task log instead of being invisible.
        finite = np.isfinite(xvals) & np.isfinite(yvals)
        n_finite_drop = int(np.sum(~finite))
        if n_finite_drop > 0:
            print(f"SeededFinder dropping {n_finite_drop} sources with "
                  f"non-finite sky->pixel position (input={len(seeds)}); "
                  f"check seed catalog for masked / (0,0) sky entries",
                  flush=True)
        seeds = seeds[finite]
        xvals = xvals[finite]
        yvals = yvals[finite]

        ny, nx = data.shape
        in_field = (xvals >= 0) & (yvals >= 0) & (xvals < nx) & (yvals < ny)
        if np.any(~in_field):
            print(f"SeededFinder dropping {np.sum(~in_field)} out-of-field sources (nx={nx}, ny={ny})", flush=True)
        seeds = seeds[in_field]
        xvals = xvals[in_field]
        yvals = yvals[in_field]

        if 'flux' not in seeds.colnames:
            if 'flux_fit' in seeds.colnames:
                seeds['flux'] = np.asarray(seeds['flux_fit'], dtype=float)
            else:
                seeds['flux'] = np.ones(len(seeds), dtype=float)
        seeds['xcentroid'] = np.asarray(xvals, dtype=float)
        seeds['ycentroid'] = np.asarray(yvals, dtype=float)
        seeds['x_init'] = np.asarray(xvals, dtype=float)
        seeds['y_init'] = np.asarray(yvals, dtype=float)
        seeds['flux_init'] = np.asarray(seeds['flux'], dtype=float)
        return seeds


def build_hybrid_saturated_artifact_mask(shape, satstar_table, core_radius_pix=12, halo_radius_pix=28,
                                         flux_scale_pix=1.0):
    mask = np.zeros(shape, dtype=bool)
    if satstar_table is None:
        return mask

    satstar_table = _as_table(satstar_table)
    if len(satstar_table) == 0:
        return mask

    xvals, yvals = _get_source_xy(satstar_table)
    if 'flux_fit' in satstar_table.colnames:
        fluxvals = np.asarray(satstar_table['flux_fit'], dtype=float)
    else:
        fluxvals = np.ones(len(satstar_table), dtype=float)

    yy, xx = np.indices(shape)
    for xval, yval, fluxval in zip(xvals, yvals, fluxvals):
        if not (np.isfinite(xval) and np.isfinite(yval)):
            continue
        flux_term = flux_scale_pix * np.log10(max(float(fluxval), 1.0))
        core_radius = max(float(core_radius_pix), core_radius_pix + flux_term)
        halo_radius = max(core_radius + 2.0, halo_radius_pix + flux_term)
        distance2 = (xx - xval) ** 2 + (yy - yval) ** 2
        mask |= distance2 <= halo_radius ** 2
        mask |= distance2 <= core_radius ** 2

    return mask


def postprocess_residual_image(data, fwhm_pix, negative_threshold=0.0, satstar_table=None,
                               core_radius_pix=12, halo_radius_pix=28, flux_scale_pix=1.0):
    processed = np.array(data, dtype=float, copy=True)
    kernel = Gaussian2DKernel(x_stddev=fwhm_pix / 2.355)

    if negative_threshold is not None:
        negative_mask = processed < negative_threshold
        if np.any(negative_mask):
            processed[negative_mask] = np.nan

    if satstar_table is not None:
        saturated_mask = build_hybrid_saturated_artifact_mask(
            processed.shape,
            satstar_table,
            core_radius_pix=core_radius_pix,
            halo_radius_pix=halo_radius_pix,
            flux_scale_pix=flux_scale_pix,
        )
        if np.any(saturated_mask):
            processed[saturated_mask] = np.nan

    if np.any(np.isnan(processed)):
        processed = interpolate_replace_nans(processed, kernel, convolve=convolve_fft,
                                             allow_huge=True)

    return processed


def compute_local_noise_map(data, smooth_sigma_pix=3.0):
    """
    Build a local noise map from an image using the sequence:
    1) Gaussian smooth
    2) high-pass residual = original - smooth
    3) local variance from smoothed residual**2
    4) local noise = sqrt(local variance)
    """
    image = np.asarray(np.nan_to_num(data), dtype=float)
    smoothed = ndimage.gaussian_filter(image, sigma=float(smooth_sigma_pix))
    residual = image - smoothed
    local_var = ndimage.gaussian_filter(residual ** 2, sigma=float(smooth_sigma_pix))
    local_var = np.where(local_var < 0, 0, local_var)
    noise_map = np.sqrt(local_var)
    return noise_map


def _sample_map_at_positions(image_map, xvals, yvals):
    xpix = np.rint(np.asarray(xvals, dtype=float)).astype(int)
    ypix = np.rint(np.asarray(yvals, dtype=float)).astype(int)

    sampled = np.full(len(xpix), np.nan, dtype=float)
    valid = ((xpix >= 0) & (ypix >= 0) &
             (ypix < image_map.shape[0]) & (xpix < image_map.shape[1]))
    sampled[valid] = image_map[ypix[valid], xpix[valid]]
    return sampled


def annotate_and_filter_by_local_snr(detection_table, noise_map, snr_threshold=5.0):
    tbl = _as_table(detection_table)
    if len(tbl) == 0:
        if 'local_noise' not in tbl.colnames:
            tbl['local_noise'] = np.array([], dtype=float)
        if 'local_snr' not in tbl.colnames:
            tbl['local_snr'] = np.array([], dtype=float)
        return tbl, {'input_count': 0, 'kept_count': 0, 'dropped_count': 0}

    xvals, yvals = _best_available_xy(tbl)
    local_noise = _sample_map_at_positions(noise_map, xvals, yvals)

    if 'peak' in tbl.colnames:
        signal = np.asarray(tbl['peak'], dtype=float)
    elif 'flux' in tbl.colnames:
        signal = np.asarray(tbl['flux'], dtype=float)
    elif 'flux_fit' in tbl.colnames:
        signal = np.asarray(tbl['flux_fit'], dtype=float)
    elif 'flux_init' in tbl.colnames:
        signal = np.asarray(tbl['flux_init'], dtype=float)
    else:
        signal = np.full(len(tbl), np.nan, dtype=float)

    with np.errstate(divide='ignore', invalid='ignore'):
        local_snr = np.abs(signal) / local_noise

    tbl['local_noise'] = np.asarray(local_noise, dtype=float)
    tbl['local_snr'] = np.asarray(local_snr, dtype=float)

    keep = (np.isfinite(local_snr) & np.isfinite(local_noise) &
            (local_noise > 0) & (local_snr >= float(snr_threshold)))
    filtered = tbl[keep]
    stats = {
        'input_count': int(len(tbl)),
        'kept_count': int(np.sum(keep)),
        'dropped_count': int(len(tbl) - np.sum(keep)),
    }
    return filtered, stats


def load_or_make_satstar_catalog(filename, path_prefix, use_merged_psf_for_merged=False, overwrite=False,
                                 outside_star_pixels=None, outside_star_fit_box=512,
                                 forced_grid_search_radius=5,
                                 flux_overrides=None,
                                 flux_drops=None,
                                 oversub_clamp_percentile=10.0,
                                 file_suffix='',
                                 seed_gate_image=None, seed_gate_wcs=None,
                                 deblend_with_zeroframe=False):
    """
    ``file_suffix`` is inserted into the satstar output filenames before
    the ``_satstar_catalog`` / ``_satstar_model`` / ``_satstar_residual``
    tag, so that concurrent runs which differ by post-processing options
    (e.g. ``--bgsub`` and ``--iteration-label=iter2`` vs their non-bgsub
    counterparts) write to distinct files and do not race on the shared
    name when astropy's ``writeto(overwrite=True)`` tries to remove an
    existing file.
    """
    # Prefer the *_extended_satstar_catalog.fits produced by
    # ``force_union_satstar.py`` when present: it contains the original
    # per-frame DQ-based satstar fits PLUS forced fits at positions that
    # are saturated in OTHER frames of the same filter (so the same set
    # of saturated stars is consistently fit across the whole filter).
    # See project_force_union_satstar.md for the design rationale.
    # When flux_overrides are supplied (cross-frame out-of-field reconciliation),
    # force a refit: the cached catalog predates the override and must be redone.
    if flux_overrides or flux_drops:
        overwrite = True
    extended_filename = filename.replace(
        '.fits', f'{file_suffix}_extended_satstar_catalog.fits')
    if os.path.exists(extended_filename) and not overwrite:
        return Table.read(extended_filename)
    satstar_filename = filename.replace('.fits', f'{file_suffix}_satstar_catalog.fits')
    if os.path.exists(satstar_filename) and not overwrite:
        return Table.read(satstar_filename)

    remove_saturated_stars(filename, overwrite=overwrite, path_prefix=path_prefix,
                           use_merged_psf_for_merged=use_merged_psf_for_merged,
                           outside_star_pixels=outside_star_pixels,
                           outside_star_fit_box=outside_star_fit_box,
                           forced_grid_search_radius=forced_grid_search_radius,
                           flux_overrides=flux_overrides,
                           flux_drops=flux_drops,
                           oversub_clamp_percentile=oversub_clamp_percentile,
                           file_suffix=file_suffix,
                           seed_gate_image=seed_gate_image,
                           seed_gate_wcs=seed_gate_wcs,
                           deblend_with_zeroframe=deblend_with_zeroframe)
    if os.path.exists(satstar_filename):
        return Table.read(satstar_filename)
    return None


def load_outside_fov_satstar_pixels(basepath, ww, data_shape=None,
                                    max_offset_arcsec=40.0):
    """Return outside-FOV satstar seed pixel positions, filtered by proximity.

    Diffraction spikes extend ~40" along the linear image axes and ~30"
    diagonally.  Stars whose nearest image pixel is more than
    ``max_offset_arcsec`` away contribute no measurable signal — drop them
    so the satstar fitter doesn't waste time on unfittable forced sources
    (which return NaN flux and stall the pipeline).

    A second region file ``saturated_stars_outside_fov_locked.reg`` (same
    point-region format) takes precedence when present — it contains
    REFINED celestial positions verified in one or two reference filters
    and is used as-is, with no position grid search.  Returned tuple's
    ``locked`` flag is True when the locked file was used; callers should
    set ``forced_grid_search_radius=0`` in that case.

    Returns
    -------
    pixels : list[tuple[float,float]]
    locked : bool
        True when positions come from ``_locked.reg``; False when from
        the original ``_outside_fov.reg``.

    Parameters
    ----------
    data_shape : tuple (ny, nx), optional
        Image shape used to compute the proximity cut.  If omitted we try
        ``ww.array_shape``; if that's also None, no proximity filter is
        applied.
    """
    # Precedence: a Spitzer/external-catalog PRIOR (refined further on the
    # diffraction spikes by the forced-fit grid search) > a fully LOCKED
    # hand-verified file (no position search) > the original coarse seeds.
    spitzer_fn = f'{basepath}/regions_/saturated_stars_outside_fov_spitzer.reg'
    locked_fn = f'{basepath}/regions_/saturated_stars_outside_fov_locked.reg'
    if os.path.exists(spitzer_fn):
        regfn = spitzer_fn
        # NOT locked: the Spitzer position is only ~50 mas; the forced-fit grid
        # search refines it onto the in-frame diffraction spikes (whose high PSF
        # value constrains the amplitude that the faint outer wings cannot).
        locked = False
        print(f"Using SPITZER-refined outside-FOV seed PRIOR: {regfn} "
              f"(grid-search refines on diffraction spikes)", flush=True)
    elif os.path.exists(locked_fn):
        regfn = locked_fn
        locked = True
        print(f"Using LOCKED outside-FOV seeds: {regfn}", flush=True)
    else:
        regfn = f'{basepath}/regions_/saturated_stars_outside_fov.reg'
        locked = False
    if not os.path.exists(regfn):
        return [], False

    reglist = regions.Regions.read(regfn)
    outside_pixels = []
    raw_pixels = []
    for reg in reglist:
        preg = reg
        if hasattr(reg, 'to_pixel'):
            preg = reg.to_pixel(ww)

        center = getattr(preg, 'center', None)
        if center is None:
            continue

        xval = float(center.x)
        yval = float(center.y)
        if np.isfinite(xval) and np.isfinite(yval):
            raw_pixels.append((xval, yval))

    # Determine image shape for the proximity cut
    if data_shape is None:
        data_shape = getattr(ww, 'array_shape', None)
    if data_shape is None or max_offset_arcsec is None:
        outside_pixels = raw_pixels
    else:
        ny, nx = int(data_shape[0]), int(data_shape[1])
        # Pixel scale in arcsec/pixel.  Use ww.proj_plane_pixel_scales as the
        # one-true source; if it isn't available, fail loudly — silently
        # picking a hard-coded number would produce wrong proximity cuts.
        scales = ww.proj_plane_pixel_scales()
        pix_arcsec = float(scales[0].to('arcsec').value)
        max_offset_pix = max_offset_arcsec / pix_arcsec
        for xv, yv in raw_pixels:
            dx = max(0.0, xv - (nx - 1)) if xv > nx - 1 else (0.0 if xv >= 0 else -xv)
            dy = max(0.0, yv - (ny - 1)) if yv > ny - 1 else (0.0 if yv >= 0 else -yv)
            dist_pix = (dx * dx + dy * dy) ** 0.5
            dist_arcsec = dist_pix * pix_arcsec
            if dist_arcsec <= max_offset_arcsec:
                outside_pixels.append((xv, yv))
                print(f"  outside-FOV seed at ({xv:.0f},{yv:.0f}) kept "
                      f"(dist={dist_arcsec:.1f}\" <= {max_offset_arcsec}\")",
                      flush=True)
            else:
                print(f"  outside-FOV seed at ({xv:.0f},{yv:.0f}) DROPPED "
                      f"(dist={dist_arcsec:.1f}\" > {max_offset_arcsec}\")",
                      flush=True)

    print(f"Loaded {len(outside_pixels)} outside-FOV saturated-star seeds "
          f"from {regfn} (of {len(raw_pixels)} in file)", flush=True)
    return outside_pixels, locked


def save_photutils_results(result, ww, filename,
                           im1, detector,
                           basepath, filtername, module, desat, bgsub, exposure_, visitid_, vgroupid_,
                           psf=None,
                           blur=False,
                           basic_or_iterative='basic',
                           options=None,
                           epsf_="",
                           group="",
                           fpsf="",
                           background_map=None,
                           iteration_label=None):
    print("Saving photutils results.")
    blur_ = "_blur" if blur else ""

    pixscale = (ww.proj_plane_pixel_area()**0.5).to(u.arcsec)
    if 'x_fit' in result.colnames:
        if hasattr(result['x_fit'], 'mask'):
            bad = result['x_fit'].mask
        else:
            bad = ~np.isfinite(result['x_fit'])
        print(f'Found and removed {np.sum(bad)} bad fits out of {len(result)} total [fit resulted in masked x_fit, y_fit]', flush=True)
        result = result[~bad]
        coords = ww.pixel_to_world(result['x_fit'], result['y_fit'])
        result['skycoord_centroid'] = coords
    elif 'xcentroid' in result.colnames:
        coords = ww.pixel_to_world(result['xcentroid'], result['ycentroid'])
    elif 'x_centroid' in result.colnames:
        # photutils >=3.0 emits x_centroid / y_centroid (with underscore)
        coords = ww.pixel_to_world(result['x_centroid'], result['y_centroid'])
        result['skycoord_centroid'] = coords
    elif 'x_init' in result.colnames:
        coords = ww.pixel_to_world(result['x_init'], result['y_init'])
        result['skycoord_init'] = coords
    else:
        raise KeyError(f"No x value found in {result.colnames}")
    print(f'len(result) = {len(result)}, len(coords) = {len(coords)}, type(result)={type(result)}', flush=True)
    if options.each_exposure:
        result.meta['exposure'] = exposure_
    if visitid_ is not None and visitid_ != '':
        result.meta['visit'] = int(visitid_[-3:])
    if vgroupid_ is not None and vgroupid_ != '':
        result.meta['vgroup'] = vgroupid_.removeprefix('_vgroup')

    result.meta['filename'] = filename
    result.meta['filter'] = filtername
    result.meta['module'] = module
    result.meta['detector'] = detector
    result.meta['pixscale'] = pixscale.to(u.deg).value
    result.meta['pixscale_as'] = pixscale.to(u.arcsec).value
    result.meta['proposal_id'] = options.proposal_id

    if 'RAOFFSET' in im1[0].header:
        result.meta['RAOFFSET'] = im1[0].header['RAOFFSET']
        result.meta['DEOFFSET'] = im1[0].header['DEOFFSET']
    elif 'RAOFFSET' in im1[1].header:
        result.meta['RAOFFSET'] = im1[1].header['RAOFFSET']
        result.meta['DEOFFSET'] = im1[1].header['DEOFFSET']

    if 'x_err' in result.colnames:
        result['dra'] = result['x_err'] * pixscale
        result['ddec'] = result['y_err'] * pixscale

    if iteration_label not in (None, ''):
        result.meta['iteration'] = str(iteration_label)

    if 'local_bkg' in result.colnames:
        result.meta['BKGCOL'] = 'local_bkg'
        result.meta['BKGMETH'] = 'photutils_local'
    else:
        xpos, ypos = _get_source_xy(result)
        result['local_bkg'] = _sample_background_map(background_map, xpos, ypos)
        result.meta['BKGCOL'] = 'local_bkg'
        result.meta['BKGMETH'] = 'bkg2d_sampled' if background_map is not None else 'none'

    iter_ = _iteration_token(iteration_label)
    # Per-observation disambiguator (prop 2211/gc2211 only; empty elsewhere).
    # gc2211's 5 obs reuse the same visit/vgroup/exp tuples, so without _o{field}
    # the catalog tables collide across obs and silently overwrite.  MUST match
    # _predict_tblfilename and the merge_catalogs.py glob.  See obs_token().
    obs_ = _obs_token_from_options(options)
    # Historical bug: this used to be `{module}{detector}` with no
    # separator, which produced doubled tokens like ``nrcbnrcb`` /
    # ``nrcanrca`` whenever ``module == detector`` (which is always
    # the case for the eachexp call paths) and broke the
    # ``merge_catalogs.py`` glob that expects just ``{module}``.
    # The original iter1 convention used only ``{module}`` and that's
    # what every other filename slot in this file (and the seed-catalog
    # inference at line ~1931) still uses.  Restored.
    tblfilename = f"{basepath}/{filtername}/{filtername.lower()}_{module}{obs_}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}{iter_}_daophot_{basic_or_iterative}.fits"

    long_keys = [k for k in result.meta if len(k) > 8]
    for k in long_keys:
        result.meta[k[:8]] = result.meta[k]
        del result.meta[k]

    print(f"tblfilename={tblfilename}, filename={filename}, filtername={filtername}, module={module}, desat={desat}, bgsub={bgsub}, fpsf={fpsf} blur={blur}")

    # Per-frame catalog write -- the highest-frequency per-frame output (once per
    # frame per pass).  Build on node-local scratch then copy, to spare the
    # shared FS the metadata/random-I/O burst during the m12 write storm.
    write_via_local_scratch(tblfilename, lambda p: result.write(p, overwrite=True))
    print(f"Completed {basic_or_iterative} photometry, and wrote out file {tblfilename}")

    return result


def load_data(filename):
    fh = fits.open(filename)
    im1 = fh
    data = im1['SCI'].data
    try:
        wht = im1['WHT'].data
    except KeyError:
        wht = None
    err = im1['ERR'].data
    instrument = im1[0].header['INSTRUME']
    telescope = im1[0].header['TELESCOP']
    obsdate = im1[0].header['DATE-OBS']
    return fh, im1, data, wht, err, instrument, telescope, obsdate


def get_psf_model(filtername, proposal_id, field,
                  module,
                  use_webbpsf=False,
                  obsdate=None,
                  use_grid=False,
                  blur=False,
                  target='brick',
                  stampsz=19,
                  oversample=1,
                  basepath='/blue/adamginsburg/adamginsburg/jwst/',
                  psf_cache_dir=None,
                  instrument=None):
    """
    Return two types of PSF model, the first for DAOPhot and the second for Crowdsource

    instrument: 'NIRCam' or 'MIRI'.  If None, derived from filtername.
    """

    # jwst_root is the shared root above the per-target trees; keep it so the
    # centralized PSF store (psfs_shared/) can live as a sibling of {target}/.
    jwst_root = basepath
    basepath = f'{basepath}/{target}'

    blur_ = "_blur" if blur else ""

    if instrument is None:
        instrument = _instrument_from_filter(filtername)
    inst_token = instrument.lower()

    if use_webbpsf:
        # PSF cache check: if a pre-built fovp101 samp2 file exists, load it directly
        # and skip the expensive MAST download + Poppy PSF generation (~17-20 min, ~300 GB peak).
        # Naming convention mirrors WebbPSF: {instrument}_{detector}_{filter}_fovp101_samp2_npsf16.fits
        _psf_oversample = 2
        # Default the webbpsf cache to the centralized shared store so freshly
        # downloaded grids are reused across targets/fields, not re-downloaded.
        _psf_outdir = psf_cache_dir or central_psf_dir(jwst_root)
        if instrument == 'MIRI':
            # MIRI imaging: single detector (MIRIM); no module split.
            _cache_detector = 'MIRIM'
        elif module in ('nrca', 'nrcb'):
            if 'F4' in filtername.upper() or 'F3' in filtername.upper():
                _cache_detector = f'{module.upper()}5'
            else:
                _cache_detector = f'{module.upper()}1'
        elif module.lower() in ('nrcalong', 'nrcblong'):
            # Per-frame path passes the physical DETECTOR as ``module``
            # (cataloging sets file_module=file_detector), so LW frames arrive
            # here as 'nrcblong'.  WebbPSF names its LW grid by detector
            # 'NRCB5', so map long->5; otherwise the lookup builds
            # nircam_nrcblong_* and misses the cached nircam_nrcb5_* grid,
            # forcing a full MAST/Poppy rebuild on every frame.
            _cache_detector = f'{module.lower()[:4].upper()}5'
        elif 'nrc' in module:
            _cache_detector = module.upper()
        else:
            _cache_detector = None  # all_detectors path — handled below

        grid = None
        if _cache_detector is not None:
            # Try the requested oversample first, then accept whatever is on disk
            # (samp4 has higher fidelity than samp2; either is fine for fitting).
            _samp_candidates = [_psf_oversample, 4, 2, 1]
            seen = set()
            for _samp in _samp_candidates:
                if _samp in seen:
                    continue
                seen.add(_samp)
                _psf_fn = os.path.join(_psf_outdir,
                    f'{inst_token}_{_cache_detector.lower()}_{filtername.lower()}'
                    f'_fovp101_samp{_samp}_npsf16.fits')
                if os.path.exists(_psf_fn):
                    print(f"Loading cached PSF grid (skipping MAST/Poppy): {_psf_fn}", flush=True)
                    grid = to_griddedpsfmodel(_psf_fn)
                    if isinstance(grid, list):
                        grid = grid[0]
                    break

        if grid is None:
            with open(os.path.expanduser('~/.mast_api_token'), 'r') as fh:
                api_token = fh.read().strip()
            from astroquery.mast import Mast

            for ii in range(10):
                try:
                    Mast.login(api_token.strip())
                    break
                except (requests.exceptions.ReadTimeout,
                        requests.exceptions.ConnectionError,
                        urllib3.exceptions.ReadTimeoutError,
                        urllib3.exceptions.ProtocolError,
                        ConnectionError,
                        TimeoutError) as ex:
                    # Transient MAST hiccup (incl. RemoteDisconnected wrapped
                    # in ConnectionError; this killed brick 34252892 + cloudc
                    # 34252893 mid-run on 2026-06-10 after 3-7h of work).
                    backoff = min(30, 2 ** ii)
                    print(f"Attempt {ii} to log in to MAST: {type(ex).__name__}: {ex}; sleeping {backoff}s",
                          flush=True)
                    time.sleep(backoff)
            os.environ['MAST_API_TOKEN'] = api_token.strip()

            has_downloaded = False
            ntries = 0
            while not has_downloaded:
                ntries += 1
                try:
                    print(f"Attempting to download WebbPSF data ({instrument})", flush=True)
                    if instrument == 'MIRI':
                        nrc = webbpsf.MIRI()
                    else:
                        nrc = webbpsf.NIRCam()
                    nrc.load_wss_opd_by_date(f'{obsdate}T00:00:00')
                    nrc.filter = filtername
                    if instrument == 'MIRI':
                        # MIRI imaging only has the MIRIM detector for imaging filters.
                        nrc.detector = 'MIRIM'
                        grid = nrc.psf_grid(num_psfs=16, all_detectors=False, verbose=True, save=True,
                                           fov_pixels=101, oversample=_psf_oversample, outdir=_psf_outdir)
                    elif module in ('nrca', 'nrcb'):
                        if 'F4' in filtername.upper() or 'F3' in filtername.upper():
                            nrc.detector = f'{module.upper()}5'
                        else:
                            nrc.detector = f'{module.upper()}1'
                        grid = nrc.psf_grid(num_psfs=16, all_detectors=False, verbose=True, save=True,
                                           fov_pixels=101, oversample=_psf_oversample, outdir=_psf_outdir)
                    elif 'nrc' in module:
                        nrc.detector = module.upper()
                        grid = nrc.psf_grid(num_psfs=16, all_detectors=False, verbose=True, save=True,
                                           fov_pixels=101, oversample=_psf_oversample, outdir=_psf_outdir)
                    else:
                        grid = nrc.psf_grid(num_psfs=16, all_detectors=True, verbose=True, save=True,
                                           fov_pixels=101, oversample=_psf_oversample, outdir=_psf_outdir)
                    has_downloaded = True
                except (urllib3.exceptions.ReadTimeoutError, requests.exceptions.ReadTimeout, requests.HTTPError) as ex:
                    print(f"Failed to build PSF: {ex}", flush=True)
                except Exception as ex:
                    print(ex, flush=True)
                    if ntries > 10:
                        # avoid infinite loops
                        raise ValueError("Failed to download PSF, probably because of an error listed above")
                    else:
                        continue

        if use_grid:
            # 2026-04-24: stpsf's to_griddedpsfmodel sometimes returns
            # a list of grids when module='merged' is used for the LW
            # detectors (one grid per detector), not a single
            # GriddedPSFModel.  The downstream code (e.g.
            # ``dao_psf_model.flux.min = 0``) treats the return as a
            # single grid and crashes with
            # ``AttributeError: 'list' object has no attribute 'flux'``.
            # The use_grid=False branch already handles this; mirror it
            # here so both LW (cloudc + brick + sgrb2 etc) iter3 runs
            # don't fail on PSF setup.
            if isinstance(grid, list):
                grid = grid[0]
            return grid, WrappedPSFModel(grid, stampsz=stampsz)
        else:
            # there's no way to use a grid across all detectors.
            # the right way would be to use this as a grid of grids, but that apparently isn't supported.
            if isinstance(grid, list):
                grid = grid[0]

            #yy, xx = np.indices([31,31], dtype=float)
            #grid.x_0 = grid.y_0 = 15.5
            #psf_model = crowdsource.psf.SimplePSF(stamp=grid(xx,yy))

            # bigger PSF probably needed
            yy, xx = np.indices([61, 61], dtype=float)
            grid.x_0 = grid.y_0 = 30
            psf_model = crowdsource.psf.SimplePSF(stamp=grid(xx, yy))

            return grid, psf_model
    else:

        # Centralized PSF store: prefer the shared psfs_shared/ grid, fall back
        # to the legacy per-(proposal, field) path (resolve_merged_psf_grid_path).
        _psf_grid_path = resolve_merged_psf_grid_path(
            jwst_root, target, instrument, module, filtername,
            proposal_id, field, oversample=oversample, blur=blur)
        grid = psfgrid = to_griddedpsfmodel(_psf_grid_path)

        # if isinstance(grid, list):
        #     print(f"Grid is a list: {grid}")
        #     psf_model = WrappedPSFModel(grid[0])
        #     dao_psf_model = grid[0]
        # else:

        psf_model = WrappedPSFModel(grid, stampsz=stampsz)
        dao_psf_model = grid

        return grid, psf_model


def get_uncertainty(err, data, dq=None, wht=None):

    if dq is None:
        dq = np.zeros(data.shape, dtype='int')

    # crowdsource uses inverse-sigma, not inverse-variance
    weight = err**-1
    #maxweight = np.percentile(weight[np.isfinite(weight)], 95)
    #minweight = np.percentile(weight[np.isfinite(weight)], 5)
    #badweight =  np.percentile(weight[np.isfinite(weight)], 1)
    #weight[err < 1e-5] = 0
    #weight[(err == 0) | (wht == 0)] = np.nanmedian(weight)
    #weight[np.isnan(weight)] = 0
    bad = np.isnan(weight) | (data == 0) | np.isnan(data) | (weight == 0) | (err == 0)
    #if dq is not None:
    #    # only 0 is OK
    #    bad |= (dq != 0)
    if wht is not None:
        bad |= (wht == 0)

    #weight[weight > maxweight] = maxweight
    #weight[weight < minweight] = minweight
    # it seems that crowdsource doesn't like zero weights
    # may have caused broked f466n? weight[bad] = badweight
    #weight[bad] = minweight
    # crowdsource explicitly handles weight=0, so this _should_ work.
    weight[bad] = 0

    # Expand bad pixel zones for dq
    #bad_for_dq = ndimage.binary_dilation(bad, iterations=2)
    #dq[bad_for_dq] = 2 | 2**30 | 2**31
    #print(f"Total bad pixels = {bad.sum()}, total bad for dq={bad_for_dq.sum()}")

    return dq, weight, bad


def mosaic_each_exposure_residuals(basepath, filtername, proposal_id, field, module,
                                   residual_kind='iterative', desat=False, bgsub=False,
                                   epsf=False, blur=False, group=False, pupil='clear',
                                   iteration_label=None, resbgsub=False,
                                   make_starless=True, crop_to_data=False):
    """
    Resample per-exposure residual images into one JWST-style *_residual_i2d.fits product.
    """
    if residual_kind not in ('basic', 'iterative'):
        raise ValueError(f"residual_kind must be one of ('basic', 'iterative'), got {residual_kind}")

    pipeline_dir = f'{basepath}/{filtername}/pipeline'
    desat_ = '_unsatstar' if desat else ''
    # Mirror _bgsub_token: the iter3-residual-bg run appends _resbgsub after
    # _bgsub so this glob finds the residuals do_photometry_step wrote.
    bgsub_ = ('_bgsub' if bgsub else '') + ('_resbgsub' if resbgsub else '')
    epsf_ = '_epsf' if epsf else ''
    blur_ = '_blur' if blur else ''
    group_ = '_group' if group else ''
    iter_ = _iteration_token(iteration_label)
    inst_token = _inst_token(filtername)

    if proposal_id == '3958' and field == '007' and filtername in ('F187N', 'F210M') and module == 'nrcb':
        module_patterns = [f'nrcb{number}' for number in range(1, 5)]
    elif proposal_id == '5365' and field == '001' and module in ('nrca', 'nrcb'):
        module_patterns = [f'{module}{number}' for number in range(1, 5)]
    elif module == 'merged':
        # Combined nrca+nrcb mosaic.  Some pipelines save residuals with the
        # literal 'merged' token (brick/cloudc/sgrc — chunked iter3 path);
        # others save with per-detector tokens.  Include both so glob.glob
        # finds whichever variant exists.  LW per-exposures use
        # nrcalong/nrcblong tokens; SW per-exposures use nrca1-4 + nrcb1-4.
        if _instrument_from_filter(filtername) == 'MIRI':
            module_patterns = ['merged', 'mirimage']
        else:
            module_patterns = ['merged',
                               'nrcalong', 'nrcblong',
                               'nrca1', 'nrca2', 'nrca3', 'nrca4',
                               'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
    else:
        module_patterns = [module]

    residual_files = []
    iter_regex = re.compile(r'_iter[^_]*_daophot_')
    iter_marker = f'{iter_}_daophot_' if iter_ else None
    flag_tokens = {
        '_unsatstar': desat,
        '_bgsub': bgsub,
        '_resbgsub': resbgsub,
        '_epsf': epsf,
        '_blur': blur,
        '_group': group,
    }

    def _matches_expected_tokens(residual_path):
        name = os.path.basename(residual_path)

        # Enforce exact flag matching: no accidental mixing of bgsub/non-bgsub,
        # desaturated/non-desaturated, etc.
        for token, enabled in flag_tokens.items():
            has_token = token in name
            if enabled and not has_token:
                return False
            if (not enabled) and has_token:
                return False

        # Enforce exact iteration matching: unlabeled mosaics must exclude iter* files.
        if iter_marker is None:
            if iter_regex.search(name):
                return False
        else:
            if iter_marker not in name:
                return False

        return True

    # Chunked iter3 LW runs (brick 2221 / cloudc) split each frame's sources
    # across N seed chunks; each chunk writes its own
    # ``..._iter3_chunkXXofYY_daophot_{kind}_residual.fits`` where
    # ``residual_K = data_for_residual - model_K`` (data_for_residual is shared
    # across chunks).  Mosaicing chunk residuals directly would double-count
    # source flux not fit in that chunk.  Glob both un-chunked and chunked
    # variants; chunked variants are combined into a single frame-level
    # residual below before mosaicing.
    # Chunked variants embed ``_chunkXXofYY_`` between the iter token and
    # ``_daophot_``: e.g. ``..._iter3_chunk00of08_daophot_basic_residual.fits``.
    chunked_iter_regex = (re.compile(re.escape(iter_) + r'_chunk\d+of\d+_daophot_')
                          if iter_ else re.compile(r'_chunk\d+of\d+_daophot_'))
    for module_pattern in module_patterns:
        for chunk_pat in ('', '_chunk*of*'):
            residual_glob = (
                f'{pipeline_dir}/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-'
                f'{module_pattern}_visit*_vgroup*_exp*{desat_}{bgsub_}{epsf_}{blur_}{group_}'
                f'{iter_}{chunk_pat}_daophot_{residual_kind}_residual.fits'
            )
            residual_files.extend(glob.glob(residual_glob))

    def _accept(fn):
        if not _matches_expected_tokens(fn):
            # Chunked variants have ``_iter3_chunkXXofYY_daophot_`` rather than
            # ``_iter3_daophot_``; accept those explicitly when iter_marker is set.
            if iter_marker and chunked_iter_regex and chunked_iter_regex.search(os.path.basename(fn)):
                # still enforce flag tokens
                name = os.path.basename(fn)
                for token, enabled in flag_tokens.items():
                    has_token = token in name
                    if enabled and not has_token:
                        return False
                    if (not enabled) and has_token:
                        return False
                return True
            return False
        return True

    residual_files = sorted(set(fn for fn in residual_files if _accept(fn)))
    if len(residual_files) == 0:
        raise ValueError(
            f'No per-exposure residuals found for module={module} '
            f'patterns={module_patterns} filter={filtername} residual_kind={residual_kind}'
        )

    # Combine chunked per-frame residuals into a single frame-level residual.
    # For chunks of the same frame:
    #   chunk_K_residual + chunk_K_model == data_for_residual  (shared)
    # Therefore:
    #   combined_residual = data_for_residual - sum_K(model_K)
    #                     = chunk_0_residual - sum_{K>0}(model_K)
    # The combined file is written to the chunk-stripped path so the existing
    # mosaicing logic (which expects one residual per frame) works unchanged.
    _chunk_strip_re = re.compile(r'_chunk\d+of\d+')
    frame_groups = {}
    for fn in residual_files:
        key = _chunk_strip_re.sub('', fn)
        frame_groups.setdefault(key, []).append(fn)

    combined_residuals = []
    for frame_key, chunks in frame_groups.items():
        if len(chunks) == 1 and '_chunk' not in os.path.basename(chunks[0]):
            combined_residuals.append(chunks[0])
            continue
        chunks_sorted = sorted(chunks)
        base_path = chunks_sorted[0]
        with fits.open(base_path) as hdul:
            sci_ext = None
            for i, hdu in enumerate(hdul):
                if hdu.name == 'SCI' or (i == 1 and hdu.data is not None):
                    sci_ext = i
                    break
            if sci_ext is None:
                sci_ext = 1
            combined_data = hdul[sci_ext].data.astype(np.float64, copy=True)
        for ch in chunks_sorted[1:]:
            model_path = ch.replace('_residual.fits', '_model.fits')
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f'Chunk combination needs model file {model_path} '
                    f'(missing companion to {ch})'
                )
            with fits.open(model_path) as mhdul:
                msci = None
                for i, hdu in enumerate(mhdul):
                    if hdu.name == 'SCI' or (i == 1 and hdu.data is not None):
                        msci = i
                        break
                if msci is None:
                    msci = 1
                combined_data -= mhdul[msci].data.astype(np.float64)
        print(f'  combining {len(chunks_sorted)} chunks -> {os.path.basename(frame_key)}')
        save_residual_datamodel(base_path, frame_key, combined_data.astype(np.float32))
        combined_residuals.append(frame_key)

    residual_files = sorted(set(combined_residuals))

    product_name = (
        f'jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}'
        f'{desat_}{bgsub_}{epsf_}{blur_}{group_}{iter_}_daophot_{residual_kind}_residual'
    )
    asn = asn_from_list.asn_from_list(
        residual_files,
        rule=DMS_Level3_Base,
        product_name=product_name,
    )

    asn_filename = f'{pipeline_dir}/{product_name}_asn.json'
    with open(asn_filename, 'w') as asn_fh:
        _, serialized = asn.dump()
        asn_fh.write(serialized)

    output_filename = f'{pipeline_dir}/{product_name}_i2d.fits'
    print(f'Resampling {len(residual_files)} residual exposures into {product_name}_i2d.fits')
    resampled = ResampleStep.call(asn_filename, output_dir=pipeline_dir, save_results=False)
    resampled.save(output_filename, overwrite=True)
    if hasattr(resampled, 'close'):
        resampled.close()

    if not os.path.exists(output_filename):
        raise FileNotFoundError(f'Expected output was not created: {output_filename}')
    print(f'Wrote residual mosaic {output_filename}')

    if crop_to_data:
        # ResampleStep allocates a ~full-frame canvas even for a small cutout;
        # trim the mosaic back to the cutout region.  Done before the infilled
        # step so the infilled mosaic is cropped too.
        _crop_datamodel_to_finite(output_filename)

    fwhm_tbl = Table.read(FWHM_TABLE)
    row = fwhm_tbl[fwhm_tbl['Filter'] == filtername]
    fwhm_pix = float(row['PSF FWHM (pixel)'][0])
    with ImageModel(output_filename) as model:
        # Only infill genuine NaNs (sat-pixel holes / DQ gaps).  Previously
        # ``negative_threshold=0.0`` NaN'd every slightly-negative-noise pixel
        # and then interpolated them away, which flattened the background
        # and reduced contrast without improving real features.  User policy
        # (2026-05-15): infill only NaNs or stars, not background noise.
        infilled_data = postprocess_residual_image(
            model.data,
            fwhm_pix,
            negative_threshold=None,
            satstar_table=None,
        )
        model.data = infilled_data
        infilled_filename = residual_to_infilled_i2d(output_filename)
        model.save(infilled_filename, overwrite=True)
    print(f'Wrote residual infilled mosaic {infilled_filename}')

    # Always run make_starless after producing the infilled mosaic -- unless
    # disabled (cutout runs: the target-catalog config keyed on basepath does
    # not exist for the cutout tree, and a starless map isn't needed there).
    if not make_starless:
        print("Skipping make_starless (make_starless=False).", flush=True)
        return infilled_filename

    from jwst_gc_pipeline.photometry.make_starless_image import TARGETS, make_starless_filter

    # Reverse-lookup the target config by basepath.
    cfg = None
    for tgt_cfg in TARGETS.values():
        if os.path.abspath(tgt_cfg['basepath']) == os.path.abspath(basepath):
            cfg = tgt_cfg
            break

    if cfg is None:
        raise ValueError(f'no TARGETS entry in make_starless_image for basepath={basepath!r}')

    # Some targets have per-obs seed catalogs (e.g. gc2211 has one per
    # pointing) and provide a ``catalog_rel_template`` that we format
    # with the current ``field`` (obs id).  Fall back to the static
    # ``catalog_rel`` for the common one-catalog-per-target case.
    if 'catalog_rel_template' in cfg:
        cat_rel = cfg['catalog_rel_template'].format(field=field)
    else:
        cat_rel = cfg['catalog_rel']
    cat_path = os.path.join(basepath, cat_rel)
    out_dir  = os.path.join(basepath, 'catalogs', 'starless')
    os.makedirs(out_dir, exist_ok=True)
    cfg_regs = [os.path.join(basepath, p) for p in cfg.get('force_mask_regs', [])]
    make_starless_filter(
        filtername, basepath, cat_path, out_dir,
        method=residual_kind,
        bgsub=bgsub,
        force_mask_regs=cfg_regs or None,
    )

    return infilled_filename


def save_residual_datamodel(input_filename, output_filename, data, clear_dq=False):
    """
    TODO: profile this code, it seems to take a minute or more even for cutouts

    ``clear_dq`` (2026-06-20): for the MODEL image, reset DQ to GOOD and make
    ERR/variance finite.  The model is a SYNTHETIC rendered image -- every pixel
    is a valid model value, even where the DATA was DQ-SATURATED (the bright
    saturated-star CORES).  If we inherit the data's DQ, ResampleStep zero-
    weights the DO_NOT_USE/SATURATED pixels and the satstar model CORE is dropped
    from the model i2d mosaic (sickle F770W stars A/B: per-frame model has the
    core at 3378/10547, but the mosaic shows ~0 -- the saturated-core pixels were
    weighted out).  Unsaturated stars (C/D) keep good DQ so they survive; only the
    saturated cores were being lost.  Clearing DQ lets the full model resample.
    Used ONLY for the model; the RESIDUAL keeps the data DQ (NaN cores there are
    honest -- the data IS missing).
    """
    import astropy.io.fits as fits_io

    # Read S_REGION before opening with ImageModel; ImageModel.save() drops CONTINUE cards,
    # truncating polygons longer than one FITS card (~68 chars).
    s_region = None
    s_region_ext = None
    with fits_io.open(input_filename) as hdul:
        for i, hdu in enumerate(hdul):
            sr = hdu.header.get('S_REGION', None)
            if sr:
                s_region = sr
                s_region_ext = i
                break

    with ImageModel(input_filename) as model:
        wcs = model.meta.wcs
        model.data = data
        model.meta.wcs = wcs  # explicit re-assignment ensures GWCS is serialized to ASDF extension
        if clear_dq:
            # synthetic model -> every pixel valid; drop the data's DQ/SATURATED
            # flags so ResampleStep keeps the satstar model cores (see docstring).
            try:
                model.dq = np.zeros_like(model.dq)
            except Exception:
                pass
            # ResampleStep ivm-weights by variance; make it finite & uniform so no
            # model pixel is dropped for a NaN/inf inherited variance at the cores.
            for _vn in ('err', 'var_rnoise', 'var_poisson', 'var_flat'):
                try:
                    _arr = getattr(model, _vn)
                    if _arr is not None and np.size(_arr):
                        _bad = ~np.isfinite(_arr)
                        if _bad.any():
                            _arr[_bad] = 1.0
                            setattr(model, _vn, _arr)
                except Exception:
                    pass
        # The residual/model image is sky-pedestal-free BY CONSTRUCTION: the
        # model is rendered point sources (>= 0) and the residual already has the
        # source+satstar model subtracted.  But this datamodel inherits the input
        # frame's meta.background.level (the sky level measured during reduction)
        # with subtracted=False, so when ResampleStep coadds these per-frame
        # images into the i2d mosaic it RE-SUBTRACTS that sky level -> a spurious
        # uniform NEGATIVE pedestal (verified: F480M model i2d median -11.25 with
        # the inherited level vs +1.07 with it zeroed).  Mark the background as
        # already removed so resample adds nothing.
        try:
            model.meta.background.level = 0.0
            model.meta.background.subtracted = True
        except Exception as _ex:
            print(f"save_residual_datamodel: could not zero meta.background "
                  f"({_ex}); i2d mosaic may show a negative pedestal", flush=True)
        model.save(output_filename, overwrite=True)

    # Restore full S_REGION (with CONTINUE support) that ImageModel.save() may have truncated.
    if s_region is not None:
        with fits_io.open(output_filename, mode='update') as hdul:
            for hdu in hdul:
                if 'S_REGION' in hdu.header:
                    hdu.header['S_REGION'] = s_region
                    break
            hdul.flush()


def _cutout_smooth_residual_bg(residual_i2d_path, median_size=3, overwrite=False):
    """Median-smooth a cutout residual mosaic into its ``_smoothed_bg`` sibling.

    Mirrors make_iter3_residual_bgmaps.smooth_one (kept in-package so the
    in-process cutout pipeline has no dependency on the brick analysis dir).
    Returns the smoothed-bg path.  ``..._residual_i2d.fits`` ->
    ``..._residual_smoothed_bg_i2d.fits``.
    """
    out_path = residual_to_smoothed_bg_i2d(residual_i2d_path)
    if os.path.exists(out_path) and not overwrite:
        return out_path
    with fits.open(residual_i2d_path) as hdul:
        names = [h.name for h in hdul]
        if 'SCI' in names:
            sci_idx = names.index('SCI')
            data = hdul[sci_idx].data.astype(np.float32)
            header = hdul[sci_idx].header
            primary_header = hdul[0].header
        else:
            data = hdul[0].data.astype(np.float32)
            header = hdul[0].header
            primary_header = None
    finite = np.isfinite(data)
    work = np.where(finite, data, 0.0)
    smoothed = ndimage.median_filter(work, size=median_size, mode='nearest')
    smoothed = np.where(finite, smoothed, np.nan).astype(np.float32)
    out = fits.HDUList()
    if primary_header is not None:
        out.append(fits.PrimaryHDU(header=primary_header))
        out.append(fits.ImageHDU(data=smoothed, header=header, name='SCI'))
    else:
        out.append(fits.PrimaryHDU(data=smoothed, header=header))
    out[0].header['HISTORY'] = (
        f'cutout pipeline: {median_size}x{median_size} median filter of '
        f'{os.path.basename(residual_i2d_path)}')
    out.writeto(out_path, overwrite=True)
    return out_path


_REDUCTION_WCS_ASDF_CACHE = {}


def _reduction_mosaic_output_wcs(pipeline_dir, proposal_id, field, inst_token,
                                 filtername):
    """Path to an ASDF holding the reduction-mosaic (image3 i2d) GWCS, or None.

    When a reduction mosaic ``jw0{prop}-o{field}_t001_{inst}_{filt}_i2d.fits``
    exists, the cataloging ``_data_i2d`` should be resampled onto its EXACT grid
    so the 'data' image matches the canonical reduction mosaic pixel-for-pixel
    (verified byte-identical: same crf + same output WCS -> max|diff|=0).
    Returns None for fields with no reduction mosaic (NIRCam fields built only by
    cataloging) or cutout runs, leaving the legacy tight-bbox behaviour intact.
    """
    mosaic = os.path.join(
        pipeline_dir,
        f'jw0{proposal_id}-o{field}_t001_{inst_token}_{filtername.lower()}_i2d.fits')
    if not os.path.exists(mosaic):
        # This pipeline names its reduction mosaic with the custom
        # 'clear-{filt}-merged' token (PipelineRerunNIRCAM-*), not the STScI
        # default '{inst}_{filt}'. Fall back to it so data_i2d lands on the exact
        # reduction grid instead of a tight crop-to-data bbox.
        alt = os.path.join(
            pipeline_dir,
            f'jw0{proposal_id}-o{field}_t001_{inst_token}_clear-{filtername.lower()}-merged_i2d.fits')
        if os.path.exists(alt):
            mosaic = alt
        else:
            return None
    out = os.path.join(pipeline_dir,
                       f'_reduction_grid_o{field}_{filtername.lower()}.asdf')
    key = os.path.getmtime(mosaic)
    if _REDUCTION_WCS_ASDF_CACHE.get(out) == key and os.path.exists(out):
        return out
    import asdf as _asdf
    with ImageModel(mosaic) as _m:
        _asdf.AsdfFile({'wcs': _m.meta.wcs}).write_to(out)
    _REDUCTION_WCS_ASDF_CACHE[out] = key
    return out


def _resample_to_i2d(files, pipeline_dir, product_name, crop_to_data=True,
                     output_wcs=None):
    """Resample an explicit list of per-frame datamodels into one i2d mosaic.

    Generic ResampleStep coadd shared by the cutout data-i2d and merged-catalog
    residual mosaics.  ``crop_to_data`` trims the (over-allocated) canvas back
    to the finite-data bbox.  ``output_wcs`` (path to an ASDF custom WCS) forces
    the output onto an explicit grid -- used to land the data_i2d on the exact
    reduction-mosaic grid; it implies no cropping (the mosaic grid is already the
    full footprint).
    """
    files = sorted(set(files))
    if not files:
        return None
    asn = asn_from_list.asn_from_list(files, rule=DMS_Level3_Base,
                                      product_name=product_name)
    asn_filename = f'{pipeline_dir}/{product_name}_asn.json'
    with open(asn_filename, 'w') as fh:
        _, serialized = asn.dump()
        fh.write(serialized)
    output_filename = f'{pipeline_dir}/{product_name}_i2d.fits'
    print(f'Resampling {len(files)} frames into {product_name}_i2d.fits', flush=True)
    _call_kw = {}
    if output_wcs:
        _call_kw['output_wcs'] = output_wcs
        crop_to_data = False
    resampled = ResampleStep.call(asn_filename, output_dir=pipeline_dir,
                                  save_results=False, **_call_kw)
    resampled.save(output_filename, overwrite=True)
    if hasattr(resampled, 'close'):
        resampled.close()
    if crop_to_data:
        _crop_datamodel_to_finite(output_filename)
    return output_filename


def mosaic_cutout_input_data(cut_bp, filtername, proposal_id, field, module,
                             label, pupil='clear', input_files=None):
    """Resample the per-frame INPUT data into a ``_data_i2d`` mosaic.

    This is the ORIGINAL (non-residual) image on the same i2d grid as the
    residual mosaics, so catalog sky positions can be overplotted on real data
    (the residuals subtract the sources, leaving nothing to register against).

    Cutout runs resample the per-frame ``*_cutout_<label>.fits`` crops; full-frame
    runs pass the original frames explicitly via ``input_files`` (no cutout crop
    exists), reprojecting them onto a common grid the same way.
    """
    pipeline_dir = f'{cut_bp}/{filtername}/pipeline'
    inst_token = _inst_token(filtername)
    if input_files is not None:
        files = sorted(input_files)
    else:
        files = sorted(glob.glob(f'{pipeline_dir}/*_cutout_{label}.fits'))
    if not files:
        print(f"mosaic_cutout_input_data: no inputs "
              f"(label={label!r}, input_files={input_files is not None}) "
              f"in {pipeline_dir}", flush=True)
        return None
    product_name = (f'jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-'
                    f'{filtername.lower()}-{module}_data')
    # Full-frame runs (input_files passed): if a reduction mosaic exists, land the
    # data_i2d on its EXACT grid so the cataloging 'data' image == the canonical
    # reduction i2d.  Cutout runs (input_files None) keep the tight-bbox crop.
    output_wcs = None
    if input_files is not None:
        output_wcs = _reduction_mosaic_output_wcs(pipeline_dir, proposal_id,
                                                  field, inst_token, filtername)
    out = _resample_to_i2d(files, pipeline_dir, product_name,
                           crop_to_data=(output_wcs is None),
                           output_wcs=output_wcs)
    print(f'Wrote cutout data i2d {out}', flush=True)
    return out


def mosaic_cutout_satstar_flags(cut_bp, filtername, proposal_id, field, module,
                                label, pupil='clear'):
    """OR-combine the per-frame saturated-star flag images onto the data i2d grid.

    Flags are a uint8 bitmask (1=partly saturated/nonlinear, 2=totally
    saturated/unrecoverable NaN, 4=included in a satstar fit).  Uses
    nearest-neighbour reprojection + bitwise-OR so the integer bits are
    preserved (ResampleStep's interpolation would corrupt them).  Cutout path.
    """
    from reproject import reproject_interp
    pipeline_dir = f'{cut_bp}/{filtername}/pipeline'
    inst_token = _inst_token(filtername)
    ref = (f'{pipeline_dir}/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-'
           f'{filtername.lower()}-{module}_data_i2d.fits')
    flagfiles = sorted(glob.glob(f'{pipeline_dir}/*_cutout_{label}_satstar_flags.fits'))
    if not (os.path.exists(ref) and flagfiles):
        return None
    with fits.open(ref) as h:
        names = [x.name for x in h]
        di = names.index('SCI') if 'SCI' in names else 0
        refhdr = h[di].header
        refsh = h[di].data.shape
        refw = wcs.WCS(refhdr)
    acc = np.zeros(refsh, dtype=np.uint8)
    for fn in flagfiles:
        with fits.open(fn) as h:
            rep, _ = reproject_interp((h[0].data.astype(float), wcs.WCS(h[0].header)),
                                      refw, shape_out=refsh, order='nearest-neighbor')
        acc |= np.nan_to_num(rep).astype(np.uint8)
    out = fits.PrimaryHDU(data=acc, header=refhdr)
    out.header['FLAGBIT1'] = (1, 'partly saturated (nonlinear)')
    out.header['FLAGBIT2'] = (2, 'totally saturated (unrecoverable NaN)')
    out.header['FLAGBIT4'] = (4, 'included in saturated-star fit')
    out_fn = (f'{pipeline_dir}/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-'
              f'{filtername.lower()}-{module}_satstar_flags_i2d.fits')
    out.writeto(out_fn, overwrite=True)
    print(f"Wrote satstar flags i2d {out_fn}", flush=True)
    return out_fn


def _build_cutout_model_i2d(cut_bp, filtername, proposal_id, field, module,
                            iteration_label, resbgsub, options, pupil='clear'):
    """Write the final model i2d = data_i2d - residual_i2d (same grid).

    Prefers the merged-catalog residual (the vetted-catalog model) when present,
    else the raw residual.  ``model = data - residual`` includes everything that
    was subtracted (satstar + point-source models), so it overplots/compares
    directly against the data in CARTA.
    """
    inst_token = _inst_token(filtername)
    desat = '_unsatstar' if options.desaturated else ''
    bgsub = ('_bgsub' if options.bgsub else '') + ('_resbgsub' if resbgsub else '')
    epsf = '_epsf' if options.epsf else ''
    blur = '_blur' if options.blur else ''
    group = '_group' if options.group else ''
    iter_ = _iteration_token(iteration_label)
    pdir = f'{cut_bp}/{filtername}/pipeline'
    stem = (f'{pdir}/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-'
            f'{filtername.lower()}-{module}{desat}{bgsub}{epsf}{blur}{group}{iter_}'
            f'_daophot_iterative')
    data_i2d = (f'{pdir}/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-'
                f'{filtername.lower()}-{module}_data_i2d.fits')
    for resid_i2d in (f'{stem}_mergedcat_residual_i2d.fits',
                      f'{stem}_residual_i2d.fits'):
        if os.path.exists(resid_i2d) and os.path.exists(data_i2d):
            with fits.open(data_i2d) as hd, fits.open(resid_i2d) as hr:
                dn = [h.name for h in hd]; di = dn.index('SCI') if 'SCI' in dn else 1
                rn = [h.name for h in hr]; ri = rn.index('SCI') if 'SCI' in rn else 1
                model = hd[di].data.astype('float32') - hr[ri].data.astype('float32')
                out = fits.HDUList([fits.PrimaryHDU(header=hd[0].header),
                                    fits.ImageHDU(data=model, header=hd[di].header,
                                                  name='SCI')])
                out_fn = residual_to_model_i2d(resid_i2d)
                out.writeto(out_fn, overwrite=True)
            print(f"Wrote final model i2d {out_fn}", flush=True)
            return out_fn
    print(f"model i2d: data/residual i2d not found under {pdir}; skipped", flush=True)
    return None


def _cutout_origin(orig_filename, options):
    """Return the (x0, y0) parent-frame origin of the cutout in ``orig_filename``
    (same math as _prepare_cutout_input) WITHOUT writing anything.  Returns None
    if the region doesn't overlap the frame.  Used to re-origin the PSF grid in
    the merged-catalog residual render pass.  Full-frame runs (no
    ``--cutout-region``) have no crop, so the origin is ``(0, 0)``."""
    from astropy.nddata import Cutout2D, NoOverlapError
    from astropy.wcs import NoConvergence
    if not getattr(options, 'cutout_region', ''):
        return 0, 0
    with fits.open(orig_filename) as hdul:
        sci_ww = wcs.WCS(hdul['SCI'].header)
        center, size, _ = _parse_cutout_region(
            options.cutout_region, sci_ww,
            default_size_arcsec=float(getattr(options, 'cutout_size_arcsec', 5.0)))
        try:
            cut = Cutout2D(np.asarray(hdul['SCI'].data), position=center, size=size,
                           wcs=sci_ww, mode='trim', copy=False)
        except (NoOverlapError, NoConvergence):
            return None
    yslc, xslc = cut.slices_original
    return int(xslc.start), int(yslc.start)


def _resolve_render_psf_shape(fwhm_pix, grid, default=(21, 21)):
    """Render-stamp size scaled to the PSF FWHM (detector pixels).

    The default (21,21) = +-10px is NIRCam-calibrated.  For the broad MIRI
    long-wavelength PSFs (F2550W FWHM 7.3px) it clips the diffraction wings at
    ~1.4 FWHM, so a bright star's real flux out to r~16px is never subtracted --
    it lingers as a square halo in the residual and is missing from the model
    (cloudc F2550W: "bright stars use too small a PSF footprint").  Grow the
    stamp to +-MULT*FWHM (default 3 -> 6*FWHM across), forced odd, never smaller
    than ``default``, and capped by the PSF grid's own pixel stamp (can't render
    wings the grid does not contain).  NIRCam / short-MIRI (FWHM<~3.3px) stay at
    the 21px default.  Env overrides: MERGE_RENDER_PSF_SHAPE (absolute odd int)
    and MERGE_RENDER_FWHM_MULT (float, default 3.0).
    """
    _env = os.environ.get('MERGE_RENDER_PSF_SHAPE')
    if _env is not None:
        n = int(_env)
        n = n if n % 2 else n + 1
        return (n, n)
    if not (fwhm_pix and np.isfinite(fwhm_pix) and fwhm_pix > 0):
        return default
    mult = float(os.environ.get('MERGE_RENDER_FWHM_MULT', 3.0))
    n = 2 * int(np.ceil(mult * float(fwhm_pix))) + 1   # odd, centred
    n = max(n, int(default[0]))
    # cap by the grid's own pixel stamp (over-sampled data / oversampling)
    try:
        over = int(np.atleast_1d(getattr(grid, 'oversampling', [1]))[0])
        gstamp = int(min(np.asarray(grid.data).shape[-2:]) / max(1, over))
        if gstamp >= int(default[0]):
            n = min(n, gstamp if gstamp % 2 else gstamp - 1)
    except (AttributeError, TypeError, ValueError):
        pass
    return (int(n), int(n))


def _cap_render_to_pedestal(model, base, bgbox):
    """Cap a rendered satstar model to the ABOVE-PEDESTAL data.

    An uncovered saturated star re-rendered at its merged (saturation-clipped)
    flux as a peaked point source over-predicts the star in the frames where it
    is UNsaturated, and the excess gouges the (large MIRI thermal) background
    pedestal into a negative hole (cloudc F2550W bright star A: model 2879 vs
    data-bg 2185 -> residual -711 below the ~1000 pedestal).  Capping the model
    to ``clip(base - bg_coarse, 0)`` makes the core model = data-bg -> residual
    = bg (flat, no hole) while leaving the fainter wings (model < data-bg)
    subtracted.  ``bg_coarse`` is a coarse (bgbox) median filter of ``base``;
    NaN pixels in ``base`` are filled with its global median for the filter and
    left uncapped in the result.

    Returns the capped model (same shape).
    """
    from scipy.ndimage import median_filter as _medfilt
    base = np.asarray(base, dtype=float)
    finite = np.isfinite(base)
    fill = float(np.nanmedian(base[finite])) if finite.any() else 0.0
    bg_coarse = _medfilt(np.where(finite, base, fill), size=int(bgbox))
    cap = np.clip(base - bg_coarse, 0, None)          # above-pedestal signal
    return np.where(np.isfinite(cap), np.minimum(model, cap), model)


def _render_model_from_table(table, psf_model, shape, psf_shape):
    """Render a model image by evaluating ``psf_model`` at each
    (x_fit, y_fit, flux_fit) row of ``table``.  Used by the active mergedcat
    residual builder (build_mergedcat_residuals) AND by the legacy parallel
    photometry path (which sees it via namespace injection), so it is shared
    and lives here in the host module.

    Stamp size is ``psf_shape`` (psf_h, psf_w); the stamp is placed
    centered on the source rounded to integer pixel coords, but the
    PSF is evaluated at exact (x_fit, y_fit) so sub-pixel registration
    is preserved.
    """
    img = np.zeros(shape, dtype=np.float32)
    if len(table) == 0:
        return img
    psf_h, psf_w = int(psf_shape[0]), int(psf_shape[1])
    half_h, half_w = psf_h // 2, psf_w // 2
    ny, nx = shape

    xfit = np.asarray(table['x_fit'], dtype=float)
    yfit = np.asarray(table['y_fit'], dtype=float)
    flux = np.asarray(table['flux_fit'], dtype=float)

    for i in range(len(table)):
        x0, y0, f0 = xfit[i], yfit[i], flux[i]
        if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(f0)):
            continue
        ix = int(round(x0))
        iy = int(round(y0))
        y_lo = max(0, iy - half_h)
        y_hi = min(ny, iy - half_h + psf_h)
        x_lo = max(0, ix - half_w)
        x_hi = min(nx, ix - half_w + psf_w)
        if y_hi <= y_lo or x_hi <= x_lo:
            continue
        yy, xx = np.mgrid[y_lo:y_hi, x_lo:x_hi]
        stamp = psf_model.evaluate(xx.astype(float), yy.astype(float),
                                   f0, x0, y0)
        img[y_lo:y_hi, x_lo:x_hi] += np.asarray(stamp, dtype=np.float32)
    return img


def build_mergedcat_residuals(cut_bp, basepath, merged_cat_path, filtername,
                              proposal_id, field, module, options,
                              overlapping_frames, iteration_label, kinds,
                              pupil='clear', psf_shape=(21, 21),
                              satstar_label=None, satstar_cover_thresh=10.0):
    """Build residual i2d mosaics from the VETTED MERGED catalog (cutout path).

    The per-frame RAW residuals subtract every fitted source, including spurious
    detections that eat into extended emission.  The merged catalog drops those
    (quality cuts + cross-frame vetting), so a residual rendered from it shows
    the extended emission the raw residual over-subtracts.

    Construction is exact and cheap: the saved per-frame raw residual + raw model
    recover ``data_for_residual`` (= original data minus satstar, exactly as the
    fitter saw it); we re-render the model from the merged catalog (projected onto
    each frame, with the same cutout-re-origined PSF) and subtract.  No data /
    satstar reload, no re-fit.

    ``satstar_label`` (manual-path phase token, e.g. ``'m12'``/``'m3'``): when
    given, the per-frame saturated-star MODEL for that phase is added back into
    the merged-catalog MODEL mosaic *for display only* -- so the model image
    shows the saturated stars alongside the fitted point sources.  The RESIDUAL
    is unaffected (it already has the satstar model subtracted via ``base``).
    """
    from astropy.nddata import NDData as _NDData
    merged = Table.read(merged_cat_path)
    # Drop saturated-star rows: ``base`` (= data_for_residual) already has the
    # per-frame satstar MODEL subtracted, and merge_individual_frames'
    # replace_saturated() re-inserts those stars (replaced_saturated=True) with
    # the satstar flux.  Rendering them here would subtract them a SECOND time
    # (and with the wrong, non-saturated PSF).  Exclude them so the merged-cat
    # residual treats satstars exactly like the raw residual does (once, via the
    # satstar model in base).
    fcol = 'flux_fit' if 'flux_fit' in merged.colnames else 'flux'
    if 'skycoord' in merged.colnames:
        _all_sc = SkyCoord(merged['skycoord'])
    else:
        _all_sc = SkyCoord(merged['ra'], merged['dec'], unit='deg')
    _all_flux = np.asarray(merged[fcol], dtype=float)
    # A ``replaced_saturated`` star is removed per-frame via the satstar MODEL
    # (subtracted into ``base``), so rendering it here with the small point-source
    # PSF would DOUBLE-subtract it.  BUT a star can be saturated in only SOME
    # frames (satstar-fit there) and merely bright in OTHERS (no satstar fit in
    # those frames) -- in the latter ``base`` never removed it, so it MUST be
    # rendered or it lingers in the residual AND is missing from the model
    # (observed: sickle F480M bright stars left at ~88% of data peak).  So we keep
    # the saturated rows separate and decide PER FRAME (in the loop) whether THIS
    # frame's satstar model actually covers each one; uncovered ones are rendered
    # as ordinary point sources.  (See the QA POLICY block below.)
    if 'replaced_saturated' in merged.colnames:
        _satrow = np.asarray(merged['replaced_saturated'], dtype=bool)
    else:
        _satrow = np.zeros(len(merged), dtype=bool)
    msc = _all_sc[~_satrow]
    mflux = _all_flux[~_satrow]
    sat_sc = _all_sc[_satrow]
    sat_flux = _all_flux[_satrow]
    print(f"mergedcat: {int((~_satrow).sum())} unsaturated + {int(_satrow.sum())} "
          f"saturated rows; saturated rendered per-frame ONLY where this frame's "
          f"satstar model does not cover them (thresh={satstar_cover_thresh})",
          flush=True)

    pipeline_dir = f'{cut_bp}/{filtername}/pipeline'
    inst_token = _inst_token(filtername)

    # Load the PSF grid once (obsdate/instrument from the first frame).  Cutout
    # runs read the cropped copy; full-frame runs read the original frame.
    if getattr(options, 'cutout_region', ''):
        first_cut = os.path.join(
            pipeline_dir,
            os.path.basename(overlapping_frames[0]).replace(
                '.fits', f"_cutout_{_cutout_label_for(options)}.fits"))
    else:
        first_cut = overlapping_frames[0]
    fh, im1, _d, _w, _e, instrument, telescope, obsdate = load_data(first_cut)
    fh.close()
    grid, _psf = get_psf_model(filtername, proposal_id, field, module=module,
                               use_webbpsf=True, use_grid=options.each_exposure,
                               blur=options.blur, target=options.target,
                               obsdate=obsdate,
                               basepath='/blue/adamginsburg/adamginsburg/jwst/',
                               psf_cache_dir=os.path.join(basepath, 'psfs'),
                               instrument=instrument)
    # FWHM-scale the render stamp so broad MIRI long-wavelength PSFs are not
    # clipped (see _resolve_render_psf_shape).  Falls back to the (21,21) default
    # for NIRCam / short-MIRI and if the FWHM lookup fails.
    from jwst_gc_pipeline.reduction.filtering import get_fwhm as _get_fwhm
    _fwhm_pix = None
    try:
        _hdr = fits.getheader(first_cut)
        _, _fwhm_pix = _get_fwhm(_hdr)
    except (KeyError, IndexError, FileNotFoundError, OSError):
        _fwhm_pix = None
    _new_shape = _resolve_render_psf_shape(_fwhm_pix, grid, default=psf_shape)
    if tuple(_new_shape) != tuple(psf_shape):
        print(f"mergedcat render psf_shape {tuple(psf_shape)} -> {tuple(_new_shape)} "
              f"(filter {filtername}, FWHM {_fwhm_pix:.1f}px detector)", flush=True)
    psf_shape = _new_shape
    half_h, half_w = int(psf_shape[0]) // 2, int(psf_shape[1]) // 2

    # satstar-model suffix matching _prepare_frame_for_photometry's
    # ``satstar_file_suffix`` (manual path); used to add the saturated-star model
    # back into the MODEL mosaic for display (never the residual).
    sat_suffix = (_bgsub_token(options) + _iteration_token(satstar_label)
                  if satstar_label is not None else None)

    written = {k: [] for k in kinds}
    written_model = {k: [] for k in kinds}
    for orig in overlapping_frames:
        origin = _cutout_origin(orig, options)
        if origin is None:
            continue
        x0, y0 = origin
        # the per-frame satstar model sits next to the FITTER INPUT (the cutout
        # crop for cutout runs, the original frame full-frame), same pixel grid
        # as the per-frame residual/model
        satstar_sm = None
        if sat_suffix is not None:
            if getattr(options, 'cutout_region', ''):
                _fitter_in = os.path.join(
                    pipeline_dir, os.path.basename(orig).replace(
                        '.fits', f"_cutout_{_cutout_label_for(options)}.fits"))
            else:
                _fitter_in = orig
            for _smp in (_fitter_in.replace('.fits', f'{sat_suffix}_extended_satstar_model.fits'),
                         _fitter_in.replace('.fits', f'{sat_suffix}_satstar_model.fits')):
                if os.path.exists(_smp):
                    try:
                        _sm = fits.getdata(_smp).astype('float32')
                        satstar_sm = np.where(np.isfinite(_sm), _sm, 0.0)
                    except (OSError, ValueError) as _ex:
                        print(f"mergedcat: could not read satstar model {_smp}: {_ex}",
                              flush=True)
                    break
        # re-origin the spatially-varying PSF grid to cutout pixel coords
        shifted_xy = [(gx - x0, gy - y0) for (gx, gy) in grid.grid_xypos]
        rg = type(grid)(_NDData(np.asarray(grid.data),
                                meta={'grid_xypos': shifted_xy,
                                      'oversampling': grid.oversampling}))
        bn = os.path.basename(orig)
        visit_id = bn.split('_')[0][-3:]
        vgroup_id = bn.split('_')[1]
        exposure_id = bn.split('_')[2]
        # JOINT multi-obs runs (field like '002-998'): the per-frame products fold
        # the observation number into vgroup_id (cataloging.py do_photometry frame
        # loop) so two obs that share a visit+vgroup+exposure (e.g. sgrb2 obs998's
        # redo reused obs002's tile 02101) don't collide.  Mirror that fold here so
        # the raw per-frame residual/model are FOUND (else this build aborts with
        # "missing raw basic products").  Single-obs runs (no '-') unchanged.
        if '-' in str(field):
            vgroup_id = f'{bn.split("_")[0][-6:-3]}{vgroup_id}'
        # Per-frame products are named by the actual DETECTOR (the manual path
        # writes them per-detector to avoid SW filename collisions); use it in the
        # stem so the raw residual/model are found and the mergedcat per-frame
        # products are written per-detector too.  The FINAL coadded mosaic below
        # keeps the requested ``module`` (merged-module) name.
        frame_detector = bn.split('_')[3]
        (visitid_, vgroupid_, exposure_, desat, bgsub,
         epsf_, blur_, group_, iter_) = _predict_output_tokens(
            options, visit_id, vgroup_id, exposure_id, iteration_label)
        for kind in kinds:
            stem = (f'{pipeline_dir}/jw0{proposal_id}-o{field}_t001_{inst_token}_'
                    f'{pupil}-{filtername.lower()}-{frame_detector}{visitid_}{vgroupid_}'
                    f'{exposure_}{desat}{bgsub}{epsf_}{blur_}{group_}{iter_}'
                    f'_daophot_{kind}')
            raw_resid = f'{stem}_residual.fits'
            raw_model = f'{stem}_model.fits'
            if not (os.path.exists(raw_resid) and os.path.exists(raw_model)):
                # This frame is one of the successfully-fit overlapping frames, so
                # its per-frame residual+model MUST exist.  Missing products would
                # silently drop the frame from the residual/model mosaic, punching
                # a hole in the "final" image -- never acceptable.  Hard-crash.
                raise FileNotFoundError(
                    f"mergedcat: missing raw {kind} products for {bn} "
                    f"(expected {os.path.basename(raw_resid)} + "
                    f"{os.path.basename(raw_model)} in {pipeline_dir}).  This frame "
                    f"was fit successfully, so its per-frame products must exist; a "
                    f"missing one would punch a hole in the {kind} mosaic.  Aborting.")
            with fits.open(raw_resid) as h:
                ww = wcs.WCS(h['SCI'].header)
                base = h['SCI'].data.astype(float)
            with fits.open(raw_model) as h:
                base = base + h['SCI'].data.astype(float)  # = data_for_residual
            xx, yy = ww.world_to_pixel(msc)
            ny, nx = base.shape
            keep = (np.isfinite(xx) & np.isfinite(yy) & np.isfinite(mflux)
                    & (xx > -half_w) & (xx < nx + half_w)
                    & (yy > -half_h) & (yy < ny + half_h))
            rx = list(np.asarray(xx)[keep])
            ry = list(np.asarray(yy)[keep])
            rf = list(np.asarray(mflux)[keep])
            # Saturated stars: render only the ones THIS frame's satstar model does
            # NOT cover (else they linger in the residual -- see split above).  A
            # star is "covered" when the per-frame satstar model has a real peak at
            # its position; where it was not satstar-fit the model is exactly 0.
            # These UNCOVERED saturated stars are rendered SEPARATELY (srx/sry/srf)
            # so their model can be capped to the data (below) -- rendering them at
            # the merged (saturation-clipped) flux as a peaked point source
            # over-predicts the star in the frames where it is UNsaturated (cloudc
            # F2550W bright star A: model 2797 vs true star 2296) and gouges the
            # thermal-background pedestal (a -501 hole).
            srx, sry, srf = [], [], []
            if len(sat_sc):
                sxx, syy = ww.world_to_pixel(sat_sc)
                for k in range(len(sat_sc)):
                    sx, sy, sf = float(sxx[k]), float(syy[k]), float(sat_flux[k])
                    if not (np.isfinite(sx) and np.isfinite(sy) and np.isfinite(sf)):
                        continue
                    if not (-half_w < sx < nx + half_w and -half_h < sy < ny + half_h):
                        continue
                    covered = False
                    if satstar_sm is not None:
                        xi, yi = int(round(sx)), int(round(sy))
                        if 0 <= xi < nx and 0 <= yi < ny:
                            sub = satstar_sm[max(0, yi - 3):yi + 4,
                                             max(0, xi - 3):xi + 4]
                            covered = (np.isfinite(sub).any()
                                       and np.nanmax(sub) > satstar_cover_thresh)
                    if not covered:
                        srx.append(sx); sry.append(sy); srf.append(sf)
                if srx:
                    print(f"mergedcat {kind} {os.path.basename(orig)}: rendering "
                          f"{len(srx)} saturated stars NOT covered by this "
                          f"frame's satstar model (would otherwise stay in the "
                          f"residual)", flush=True)
            tbl = Table({'x_fit': np.asarray(rx, dtype=float),
                         'y_fit': np.asarray(ry, dtype=float),
                         'flux_fit': np.asarray(rf, dtype=float)})
            mc_model = _render_model_from_table(tbl, rg, base.shape, psf_shape)
            # Uncovered saturated stars: render separately and CAP to the above-
            # pedestal data so a peaked/over-flux model cannot gouge the (large
            # MIRI thermal) background pedestal.  In the star CORE the cap makes
            # model = data-bg -> residual = bg (flat, no hole); in the wings the
            # model is < data-bg and is kept (the star wing is still subtracted).
            # MIRI-only; env-gated MERGE_SATSTAR_RENDER_CAP (default off).
            if (srx and _instrument_from_filter(filtername) == 'MIRI'
                    and int(os.environ.get('MERGE_SATSTAR_RENDER_CAP', 0))):
                stbl = Table({'x_fit': np.asarray(srx, dtype=float),
                              'y_fit': np.asarray(sry, dtype=float),
                              'flux_fit': np.asarray(srf, dtype=float)})
                mc_sat = _render_model_from_table(stbl, rg, base.shape, psf_shape)
                _bgbox = int(max(15, round(6 * (_fwhm_pix or 3.0))))
                mc_sat_capped = _cap_render_to_pedestal(mc_sat, base, _bgbox)
                mc_model = mc_model + mc_sat_capped
                print(f"  [miri satstar render cap] capped {len(srx)} uncovered "
                      f"satstar renders to above-pedestal data (bgbox={_bgbox})",
                      flush=True)
            elif srx:
                # cap disabled (or non-MIRI): render at merged flux as before
                stbl = Table({'x_fit': np.asarray(srx, dtype=float),
                              'y_fit': np.asarray(sry, dtype=float),
                              'flux_fit': np.asarray(srf, dtype=float)})
                mc_model = mc_model + _render_model_from_table(
                    stbl, rg, base.shape, psf_shape)
            # ============================ QA POLICY ============================
            # The residual and model QA images obey a STRICT content policy that
            # downstream evaluation (and the residual-bg feedback loop) depend on:
            #
            #   RESIDUAL  = background ONLY.  NO stars -- neither saturated nor
            #               unsaturated.  Built as base - mc_model, where
            #                 base     = data - satstar_model  (data_for_residual;
            #                            saturated stars ALREADY removed per-frame)
            #                 mc_model = rendered UNSATURATED point sources only
            #                            (replaced_saturated rows were dropped from
            #                            `merged` above; rendering them here would
            #                            double-subtract the satstars).
            #               So the INTERMEDIATE model subtracted to make the
            #               residual MUST EXCLUDE saturated stars (they are gone
            #               via `base`).  A clean residual = pure extended bg.
            #               (If saturated stars LINGER in the residual, the
            #               per-frame satstar model under-fit them -- a satstar
            #               problem, not a policy problem here.)
            #
            #   MODEL i2d = stars ONLY (saturated AND unsaturated), NO background.
            #               mc_model (unsat point sources) + satstar_sm (the
            #               saturated-star model added back FOR DISPLAY).  The
            #               written datamodel also zeroes meta.background so the
            #               resampled mosaic carries no sky pedestal.
            #
            # CRITICAL ASYMMETRY: the model SUBTRACTED to form the residual must
            # NOT include satstars; the MODEL written to disk MUST include them.
            # Regression: tests/test_residual_model_policy.py.
            # ==================================================================
            mc_resid = (base - mc_model).astype('float32')   # residual: bg only (no stars)
            # NaN-mask deep over-subtraction PITS (MIRI).  A satstar model with
            # over-estimated amplitude gouges a deep NEGATIVE pit; at the masked
            # saturated CORE (sickle pillar -755k) and -- crucially -- out in the
            # UNSATURATED WINGS, where the DATA is valid (~700) but the broad model
            # subtracts ~1e6 (w51 F770W 11.2e6-flux satstar: resid -1.02e6 at
            # data=667, NOT a DQ-SATURATED pixel).  The legacy mask required
            # DQ-SATURATED and so MISSED every wing pit, leaving -1e6 holes in the
            # residual the iterative background must not see.  Broaden: NaN ANY
            # deeply-negative pit (over-subtraction signature) regardless of DQ,
            # plus the legacy saturated-core pits, then dilate to fill the basin.
            # Positive residual (unsubtracted bright flux the user wants) is KEPT --
            # a pit must be < 0.  Far fewer pixels than the broad SATURATED flag,
            # so no resample balloon.  Thresholds env-tunable.
            if _instrument_from_filter(filtername) == 'MIRI':
                try:
                    _fin = np.isfinite(mc_resid)
                    if _fin.any():
                        _med = np.nanmedian(mc_resid[_fin])
                        _nmad = 1.4826 * np.nanmedian(
                            np.abs(mc_resid[_fin] - _med)) + 1e-6
                        _k = float(os.environ.get('MIRI_RESID_PIT_NMAD', 15.0))
                        # over-subtraction pit: deeply negative below the bg
                        _pit = _fin & (mc_resid < _med - _k * _nmad) & (mc_resid < 0)
                        # legacy: saturated-core pits at a gentler threshold
                        with fits.open(orig) as _oh:
                            _onames = [h.name for h in _oh]
                            _dq = _oh['DQ'].data if 'DQ' in _onames else None
                        if _dq is not None and _dq.shape == mc_resid.shape:
                            _satmask = (_dq.astype(np.int64) & 2) > 0
                            _pit = _pit | (_satmask
                                           & (mc_resid < _med - 10.0 * _nmad)
                                           & (mc_resid < 0))
                        from scipy.ndimage import binary_dilation as _bd
                        _pit = _bd(_pit, iterations=int(
                            os.environ.get('MIRI_RESID_PIT_DILATE', 2)))
                        mc_resid[_pit] = np.nan
                        print(f"  [miri resid] NaN'd {int(_pit.sum())} over-sub pit "
                              f"px (deep-negative + sat-core; med={_med:.0f} "
                              f"nmad={_nmad:.0f})", flush=True)
                except (OSError, ValueError) as _dqex:
                    print(f"mergedcat: pit NaN-mask skipped for "
                          f"{os.path.basename(orig)}: {_dqex}", flush=True)
            mc_model_display = mc_model.astype('float32')     # i2d model: stars, no bg
            if satstar_sm is not None and satstar_sm.shape == mc_model.shape:
                mc_model_display = mc_model_display + satstar_sm
            out_resid = f'{stem}_mergedcat_residual.fits'
            out_model = f'{stem}_mergedcat_model.fits'
            save_residual_datamodel(raw_resid, out_resid, mc_resid)
            save_residual_datamodel(raw_model, out_model, mc_model_display,
                                    clear_dq=True)
            written[kind].append(out_resid)
            written_model[kind].append(out_model)
    # mosaic each kind's merged-catalog residuals
    outpaths = {}
    bgsub_tok = _bgsub_token(options)
    iter_tok = _iteration_token(iteration_label)
    desat_tok = '_unsatstar' if options.desaturated else ''
    epsf_tok = '_epsf' if options.epsf else ''
    blur_tok = '_blur' if options.blur else ''
    group_tok = '_group' if options.group else ''
    for kind in kinds:
        if not written[kind]:
            continue
        product_name = (f'jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-'
                        f'{filtername.lower()}-{module}{desat_tok}{bgsub_tok}'
                        f'{epsf_tok}{blur_tok}{group_tok}{iter_tok}'
                        f'_daophot_{kind}_mergedcat_residual')
        outpaths[kind] = _resample_to_i2d(written[kind], pipeline_dir,
                                          product_name, crop_to_data=True)
        print(f"mergedcat: wrote {kind} residual i2d {outpaths[kind]}", flush=True)
        # FINAL coadd-level over-subtraction pit cleanup (MIRI).  The per-frame
        # pit NaN-mask + resample averaging is LEAKY: a pixel that is a deep pit
        # in only SOME overlapping frames (shallow/positive in others) averages
        # to a moderate-but-still-negative pit that survives on the mosaic
        # (w51 F770W: per-frame fix took -1.02e6 -> coadd still had -5.9e3 pits).
        # NaN any deep-negative pit that REMAINS on the assembled residual_i2d so
        # the displayed/iterated residual has NO negative holes.  Positive flux
        # KEPT (pit must be < 0).
        if (_instrument_from_filter(filtername) == 'MIRI'
                and outpaths[kind] and os.path.exists(outpaths[kind])):
            try:
                with fits.open(outpaths[kind], mode='update') as _rh:
                    _rd = _rh['SCI'].data
                    _fin = np.isfinite(_rd)
                    if _fin.any():
                        _med = np.nanmedian(_rd[_fin])
                        _nmad = 1.4826 * np.nanmedian(
                            np.abs(_rd[_fin] - _med)) + 1e-6
                        _k = float(os.environ.get('MIRI_RESID_PIT_NMAD', 15.0))
                        _pit = _fin & (_rd < _med - _k * _nmad) & (_rd < 0)
                        from scipy.ndimage import binary_dilation as _bd
                        _pit = _bd(_pit, iterations=int(
                            os.environ.get('MIRI_RESID_PIT_DILATE', 2)))
                        _rd[_pit] = np.nan
                        _rh['SCI'].data = _rd
                        _rh.flush()
                        print(f"  [miri resid i2d] NaN'd {int(_pit.sum())} coadd "
                              f"pit px (med={_med:.0f} nmad={_nmad:.0f})", flush=True)
            except (OSError, ValueError) as _e:
                print(f"mergedcat: resid i2d pit cleanup skipped: {_e}", flush=True)
        # also mosaic the merged-catalog MODEL so the CARTA loaders can show
        # data / model / residual / bg side by side
        model_product = product_name.replace('_mergedcat_residual',
                                             '_mergedcat_model')
        try:
            mpath = _resample_to_i2d(written_model[kind], pipeline_dir,
                                     model_product, crop_to_data=True)
            print(f"mergedcat: wrote {kind} model i2d {mpath}", flush=True)
        except Exception as ex:
            print(f"mergedcat: model i2d build failed ({ex})", flush=True)
    return outpaths


def _flag_likely_extended_iter4(merged4_path, merged2_path, pixscale_arcsec):
    """Add a ``likely_extended`` column to the iter4 merged catalog.

    Heuristic (pillar_head F480M, 2026-06-08): a source with ``qfit > 0.1`` whose
    centroid moved > 0.5 px between iter2 and iter4 is probably a bump in extended
    emission, not a confident star (real point sources have tight centroids and
    good fits; emission bumps fit poorly and wander as the background model
    changes).  Also records ``centroid_move_iter2to4_pix``.  See
    NOTES_star_vs_extended_emission.md.
    """
    if not (os.path.exists(merged4_path) and os.path.exists(merged2_path)):
        return
    t4 = Table.read(merged4_path)
    t2 = Table.read(merged2_path)
    if len(t4) == 0:
        return
    def _sc(t):
        return (t['skycoord'] if 'skycoord' in t.colnames
                else SkyCoord(t['ra'], t['dec'], unit='deg'))
    sc4, sc2 = _sc(t4), _sc(t2)
    qcol = 'qfit' if 'qfit' in t4.colnames else ('qfit_avg' if 'qfit_avg' in t4.colnames else None)
    move_px = np.full(len(t4), np.nan)
    flag = np.zeros(len(t4), dtype=bool)
    if len(t2) > 0:
        idx, sep, _ = sc4.match_to_catalog_sky(sc2)
        mp = (sep.arcsec / pixscale_arcsec)
        # only treat as the SAME source (so the move is meaningful) within 5 px
        same = mp < 5.0
        move_px[same] = mp[same]
        if qcol is not None:
            qf = np.asarray(t4[qcol], dtype=float)
            flag = same & (qf > 0.1) & (mp > 0.5)
    t4['centroid_move_iter2to4_pix'] = move_px
    t4['likely_extended'] = flag
    t4.write(merged4_path, overwrite=True)
    print(f"iter4: flagged {int(flag.sum())}/{len(t4)} sources likely_extended "
          f"(qfit>0.1 & iter2->iter4 move>0.5px) in {os.path.basename(merged4_path)}",
          flush=True)


def build_filtered_iter2_residual_bg(cut_bp, basepath, filtername, proposal_id,
                                     field, module, options, overlapping_frames,
                                     pupil='clear', qfit_max=0.2,
                                     peak_over_bkg=20.0, psf_shape=(21, 21)):
    """EXPERIMENTAL iter4 background: smoothed iter2 residual where only confident
    STARS are subtracted, so extended-emission false detections stay in the
    background (and iter4 no longer inflates fluxes to absorb them).

    A source is subtracted (kept in the model) iff it is a confident star:
      qfit <= qfit_max  OR  flags == 1 (central-saturation real star)  OR
      peak surface brightness > peak_over_bkg * local_bkg  (bright real star;
      peak SB = the peak PIXEL value at the source, NOT the integrated flux).
    Everything else is left in the residual (treated as extended emission).
    Returns the smoothed-bg i2d path.  See NOTES_star_vs_extended_emission.md.
    """
    from astropy.nddata import NDData as _NDData
    pipeline_dir = f'{cut_bp}/{filtername}/pipeline'
    inst_token = _inst_token(filtername)
    first_cut = os.path.join(
        pipeline_dir,
        os.path.basename(overlapping_frames[0]).replace(
            '.fits', f"_cutout_{_cutout_label_for(options)}.fits"))
    fh, im1, _d, _w, _e, instrument, telescope, obsdate = load_data(first_cut)
    fh.close()
    grid, _psf = get_psf_model(filtername, proposal_id, field, module=module,
                               use_webbpsf=True, use_grid=options.each_exposure,
                               blur=options.blur, target=options.target,
                               obsdate=obsdate,
                               basepath='/blue/adamginsburg/adamginsburg/jwst/',
                               psf_cache_dir=os.path.join(basepath, 'psfs'),
                               instrument=instrument)
    written = []
    n_kept = n_drop = 0
    for orig in overlapping_frames:
        origin = _cutout_origin(orig, options)
        if origin is None:
            continue
        x0, y0 = origin
        shifted_xy = [(gx - x0, gy - y0) for (gx, gy) in grid.grid_xypos]
        rg = type(grid)(_NDData(np.asarray(grid.data),
                                meta={'grid_xypos': shifted_xy,
                                      'oversampling': grid.oversampling}))
        bn = os.path.basename(orig)
        visit_id, vgroup_id, exposure_id = bn.split('_')[0][-3:], bn.split('_')[1], bn.split('_')[2]
        catfn = _predict_tblfilename(cut_bp, filtername, module, options,
                                     visit_id, vgroup_id, exposure_id,
                                     iteration_label='iter2', method='daophot',
                                     basic_or_iterative='iterative')
        (visitid_, vgroupid_, exposure_, desat, bgsub,
         epsf_, blur_, group_, iter_) = _predict_output_tokens(
            options, visit_id, vgroup_id, exposure_id, 'iter2')
        stem = (f'{pipeline_dir}/jw0{proposal_id}-o{field}_t001_{inst_token}_'
                f'{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}'
                f'{exposure_}{desat}{bgsub}{epsf_}{blur_}{group_}{iter_}'
                f'_daophot_iterative')
        raw_resid, raw_model = f'{stem}_residual.fits', f'{stem}_model.fits'
        if not (os.path.exists(catfn) and os.path.exists(raw_resid)
                and os.path.exists(raw_model)):
            print(f"qfilt-bg: missing iter2 products for {bn}; skip", flush=True)
            continue
        cat = Table.read(catfn)
        with fits.open(raw_resid) as h:
            base = h['SCI'].data.astype(float)
        with fits.open(raw_model) as h:
            base = base + h['SCI'].data.astype(float)  # data_for_residual
        ny, nx = base.shape
        xf = np.asarray(cat['x_fit'], float); yf = np.asarray(cat['y_fit'], float)
        qf = np.asarray(cat['qfit'], float)
        flg = np.asarray(cat['flags'], float) if 'flags' in cat.colnames else np.zeros(len(cat))
        lbk = np.asarray(cat['local_bkg'], float) if 'local_bkg' in cat.colnames else np.zeros(len(cat))
        # peak surface brightness = peak PIXEL value in a 3x3 box at the source
        peaksb = np.full(len(cat), np.nan)
        for i in range(len(cat)):
            ix, iy = int(round(xf[i])), int(round(yf[i]))
            if 0 <= iy < ny and 0 <= ix < nx:
                box = base[max(0, iy-1):iy+2, max(0, ix-1):ix+2]
                box = box[np.isfinite(box)]
                if box.size:
                    peaksb[i] = float(box.max())
        keep = ((qf <= qfit_max) | (flg == 1)
                | (np.isfinite(peaksb) & (lbk > 0) & (peaksb > peak_over_bkg * lbk)))
        n_kept += int(keep.sum()); n_drop += int((~keep).sum())
        tbl = Table({'x_fit': xf[keep], 'y_fit': yf[keep],
                     'flux_fit': np.asarray(cat['flux_fit'], float)[keep]})
        model = _render_model_from_table(tbl, rg, base.shape, psf_shape)
        resid = (base - model).astype('float32')
        out_resid = f'{stem}_qfilt_residual.fits'
        save_residual_datamodel(raw_resid, out_resid, resid)
        written.append(out_resid)
    if not written:
        return None
    desat_tok = '_unsatstar' if options.desaturated else ''
    bgsub_tok = _bgsub_token(options)
    blur_tok = '_blur' if options.blur else ''
    epsf_tok = '_epsf' if options.epsf else ''
    group_tok = '_group' if options.group else ''
    product_name = (f'jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-'
                    f'{filtername.lower()}-{module}{desat_tok}{bgsub_tok}'
                    f'{epsf_tok}{blur_tok}{group_tok}_iter2_daophot_iterative_qfilt_residual')
    i2d = _resample_to_i2d(written, pipeline_dir, product_name, crop_to_data=True)
    bg = _cutout_smooth_residual_bg(i2d, overwrite=True)
    print(f"qfilt-bg: kept {n_kept} subtracted / left {n_drop} in residual; "
          f"smoothed bg {bg}", flush=True)
    return bg


def main(smoothing_scales={'f182m': 0.25, 'f187n':0.25, 'f212n':0.55,
                           'f410m': 0.55, 'f405n':0.55, 'f466n':0.55,
                           'f335m': 0.55, 'f470n': 0.55, 'f480m': 0.55,
                           # MIRI starting values; revisit once tuned
                           'f560w': 0.55, 'f770w': 0.55, 'f1000w': 0.55,
                           'f1130w': 0.55, 'f1280w': 0.55, 'f1500w': 0.55,
                           'f1800w': 0.55, 'f2100w': 0.55, 'f2550w': 0.55},
        bg_boxsizes={'f182m': 19, 'f187n':11, 'f212n':11,
                     'f210m': 11, 'f150w': 19,
                     'f410m': 11, 'f405n':11, 'f466n':11,
                     'f444w': 11, 'f356w':11, 'f335m': 11, 'f470n': 11, 'f480m': 11,
                     'f300m': 11, 'f360m': 11,
                     'f200w':19, 'f115w':19,
                     # MIRI: ~5-6x FWHM (pix), odd; Sgr B2 backgrounds may want larger
                     'f560w': 11, 'f770w': 15, 'f1000w': 19,
                     'f1130w': 21, 'f1280w': 23, 'f1500w': 27,
                     'f1800w': 33, 'f2100w': 37, 'f2550w': 45,
                    },
        crowdsource_default_kwargs={'maxstars': 500000, },
        ):
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-f", "--filternames", dest="filternames",
                    default='F466N,F405N,F410M',
                    help="filter name list", metavar="filternames")
    parser.add_option("-m", "--modules", dest="modules",
                    default='nrca,nrcb,merged',
                    help="module list", metavar="modules")
    parser.add_option("-d", "--desaturated", dest="desaturated",
                    default=False,
                    action='store_true',
                    help="use image with saturated stars removed?", metavar="desaturated")
    parser.add_option("--daophot", dest="daophot",
                    default=False,
                    action='store_true',
                    help="run daophot?", metavar="daophot")
    parser.add_option("--skip-crowdsource", dest="nocrowdsource",
                    default=False,
                    action='store_true',
                    help="skip crowdsource?", metavar="nocrowdsource")
    parser.add_option("--bgsub", dest="bgsub",
                    default=False,
                    action='store_true',
                    help="perform global background-subtraction first?", metavar="bgsub")
    parser.add_option("--use-iter3-residual-bg", dest="use_iter3_residual_bg",
                    default=False,
                    action='store_true',
                    help=("Subtract the MERGED iter3 residual smoothed-background "
                          "mosaic (..-merged_iter3_daophot_iterative_residual_"
                          "smoothed_bg_i2d.fits), reprojected onto each exposure's "
                          "grid, before fitting.  Uses the whole-field merged "
                          "residual (max background S/N).  Output catalogs/residuals "
                          "get a '_resbgsub' filename token.  Built by "
                          "make_iter3_residual_bgmaps.py."),
                    metavar="use_iter3_residual_bg")
    parser.add_option("--residual-source", dest="residual_source",
                    type="choice", choices=["mergedcat", "rawcat", "both"],
                    default="mergedcat",
                    help=("Which residual i2d(s) to build (cutout pipeline): "
                          "'mergedcat' (default) renders the model from the VETTED "
                          "MERGED catalog (spurious sources removed -- recovers "
                          "extended emission the raw residual over-subtracts); "
                          "'rawcat' uses each frame's own fit; 'both' builds both. "
                          "Rendering is extra work, so default to mergedcat only."),
                    metavar="residual_source")
    parser.add_option("--iter4-bg-exclude-badfit", dest="iter4_bg_exclude_badfit",
                    default=False, action='store_true',
                    help=("EXPERIMENTAL (cutout iter4): build the iter4 background "
                          "from an iter2 residual that subtracts only confident "
                          "stars (qfit<=0.2, OR flags==1, OR peak-SB>20x local_bkg), "
                          "leaving extended-emission false detections in the "
                          "background so iter4 stops inflating fluxes on extended "
                          "emission.  See NOTES_star_vs_extended_emission.md."),
                    metavar="iter4_bg_exclude_badfit")
    parser.add_option("--profile-memory", dest="profile_memory",
                    default=False,
                    action='store_true',
                    help=("DEBUG ONLY: enable per-frame tracemalloc memory "
                          "profiling in do_photometry_step.  OFF by default -- it "
                          "instruments every process-wide allocation and costs "
                          "tens of seconds per snapshot, dominating run time.  "
                          "Use only when chasing a memory leak."),
                    metavar="profile_memory")
    # --- manual-iteration path (cataloging.py; replaces IterativePSFPhotometry) ---
    parser.add_option("--manual-iterations", dest="manual_iterations",
                    default=True, action='store_true',
                    help=("Use the default PSF photometry pipeline "
                          "(jwst_gc_pipeline.photometry.cataloging).  This is THE "
                          "default; pass --legacy-iterations to use the old "
                          "IterativePSFPhotometry pipeline instead.  BASIC "
                          "single-pass fits with explicit iter1..iter7 reseeding, "
                          "model/data-peak overshoot QC, and a strict ban on "
                          "negative-peak sources.  (End-to-end coverage is cutout "
                          "in-process today; full-frame routing is pending.)  See "
                          "PHOTOMETRY_PIPELINE.md."),
                    metavar="manual_iterations")
    parser.add_option("--legacy-iterations", dest="manual_iterations",
                    action='store_false',
                    help=("Opt out of the default manual-iteration path and use "
                          "the legacy IterativePSFPhotometry cutout pipeline "
                          "(_run_cutout_pipeline).  Also re-enables the per-filter "
                          "single-module restriction policy."))
    parser.add_option("--manual-overshoot-ratio", dest="manual_overshoot_ratio",
                    type='float', default=1.2,
                    help="Flag a fit when its model peak > this x the local data peak (default 1.2).")
    parser.add_option("--manual-overshoot-action", dest="manual_overshoot_action",
                    type='choice', choices=['flag', 'drop', 'refit'], default='refit',
                    help="What to do with overshooting fits: flag|drop|refit (default refit at seed position).")
    parser.add_option("--manual-iter2-local-snr", dest="manual_iter2_local_snr",
                    type='float', default=3.0,
                    help="Local-S/N threshold for daofind on residual/bg-subtracted images (default 3.0).")
    parser.add_option("--manual-ext-qfit-max", dest="manual_ext_qfit_max",
                    type='float', default=0.2,
                    help="Extended-emission vetting: keep sources with qfit <= this (default 0.2).")
    parser.add_option("--manual-ext-prom-min", dest="manual_ext_prom_min",
                    type='float', default=-1.0,
                    help="Extended-emission NIRCam vetting: HARD prominence floor on "
                         "the deep data_i2d (rise above local emission) that the "
                         "qfit/peakSB/flags/bright-isolated star tests CANNOT bypass "
                         "-- the single robust real-star-vs-emission discriminator on "
                         "bright extended fields (W51 darkfil: real median ~5, false "
                         "~0.9).  -1 (default) = AUTO: 3.0 for extended-emission NIRCam "
                         "fields, off otherwise.  0 = force off.  Satstar force-keep "
                         "(model==catalog) still overrides it.")
    parser.add_option("--manual-ext-peak-over-bkg", dest="manual_ext_peak_over_bkg",
                    type='float', default=20.0,
                    help="Extended-emission vetting: keep if peak-SB > this x local bkg (default 20).")
    parser.add_option("--manual-ext-local-snr-min", dest="manual_ext_local_snr_min",
                    type='float', default=5.0,
                    help="Extended-emission vetting: require local S/N >= this (default 5).")
    parser.add_option("--manual-ext-snr-high-keep", dest="manual_ext_snr_high_keep",
                    type='float', default=20.0,
                    help="Extended-emission vetting BRIGHT-ISOLATED keep: a "
                         "group_size==1 source (trustworthy S/N) with S/N >= this "
                         "AND qfit < --manual-ext-qfit-high-keep-max is kept even "
                         "if its qfit is above the star_like qfit_max (real bright/"
                         "faint stars on bright emission whose qfit/peakSB are "
                         "degraded by the background; default 20).")
    parser.add_option("--manual-ext-qfit-high-keep-max", dest="manual_ext_qfit_high_keep_max",
                    type='float', default=0.4,
                    help="Upper qfit cap for the bright-isolated keep (default 0.4); "
                         "extended-emission knots have worse qfit so stay rejected.")
    parser.add_option("--manual-ext-qfit-recover-max", dest="manual_ext_qfit_recover_max",
                    type='float', default=0.2,
                    help="RECOVER-tier qfit ceiling for the extended-emission vetting "
                         "(NIRCam).  Keep a source whose qfit is in "
                         "(--manual-ext-qfit-max, this] AND S/N >= "
                         "--manual-ext-local-snr-min AND NOT within "
                         "--manual-ext-recover-satstar-guard-arcsec of a saturated "
                         "star.  Recovers real neighbour-blended stars (median qfit "
                         "~0.30) that the strict qfit<=0.2 cut deletes; spikes/emission "
                         "(qfit>~0.5) stay rejected.  Cataloged => subtracted "
                         "(invariant preserved).  Default 0.2 == qfit_max => NO-OP "
                         "(byte-identical to prior behaviour); set 0.5 to enable.")
    parser.add_option("--manual-ext-recover-satstar-guard-arcsec",
                    dest="manual_ext_recover_satstar_guard_arcsec",
                    type='float', default=2.0,
                    help="Merged-level satstar-proximity backstop for the recover "
                         "tier: a recovered source within this many arcsec of a "
                         "catalog saturated star is rejected (diffraction-spike "
                         "guard, on top of the per-frame _filter_satstar_artifacts). "
                         "Default 2.0\".")
    parser.add_option("--manual-ext-recover-prom-log-intercept",
                    dest="manual_ext_recover_prom_log_intercept",
                    type='float', default=-0.77,
                    help="Recover-tier prominence gate, SLOPED in (qfit, log10 prom): "
                         "a recovered source must satisfy log10(prominence) >= "
                         "intercept + slope*qfit on the data_i2d (rise above the "
                         "local emission, scaled by fit quality).  Intercept "
                         "(default -0.77).  Fit on sickle + W51 cutouts (69 real / "
                         "142 emission, F480M+F187N): keeps 63/69 real at 5/142 "
                         "emission vs 52/69 for a flat prom>=5.  NaN -> NOT recovered.")
    parser.add_option("--manual-ext-recover-prom-log-slope",
                    dest="manual_ext_recover_prom_log_slope",
                    type='float', default=5.6,
                    help="Slope of the recover-tier prominence gate in log10(prom) "
                         "per unit qfit (default 5.6): the prominence floor RISES "
                         "with qfit so a well-fit (low-qfit) source is trusted at "
                         "lower prominence and a poorly-fit one needs much more "
                         "(high-qfit recovery is unsafe on emission fields).")
    parser.add_option("--manual-ext-recover-no-prom-gate",
                    dest="manual_ext_recover_prom_gate",
                    action='store_false', default=True,
                    help="Disable the recover-tier prominence gate (UNSAFE on "
                         "extended emission; for diagnostics only).")
    parser.add_option("--manual-ext-nmatch-confirm", dest="manual_ext_nmatch_confirm",
                    type='int', default=0,
                    help="MULTI-FRAME CONFIRMATION keep (Hosek ndet-style): keep any "
                         "source detected in >= N exposures (nmatch>=N) with qfit <= "
                         "--manual-ext-nmatch-confirm-qfit-max, regardless of the "
                         "single-catalog qfit/snr cuts.  Recovers faint stars we "
                         "DETECT but the vetting drops (Arches F212N vs Hosek: 0.43 "
                         "-> 0.58).  Default 0 = OFF; set 3 to match Hosek's ndet>=3. "
                         "STAR-FIELD TOOL: extended emission is fixed on-sky so it "
                         "ALSO repeats across dithers with a stable centroid -- the "
                         "position guard does NOT reject it (W51 dark filament: +110 "
                         "emission knots admitted).  Leave OFF on emission-dominated "
                         "fields.")
    parser.add_option("--manual-ext-nmatch-confirm-qfit-max",
                    dest="manual_ext_nmatch_confirm_qfit_max",
                    type='float', default=0.6,
                    help="qfit ceiling for the multi-frame confirmation keep "
                         "(default 0.6).")
    parser.add_option("--manual-seed-round-max", dest="manual_seed_round_max",
                    type='float', default=0.5,
                    help="DAOStarFinder roundness bound for the i2d-augmented "
                         "RESIDUAL seed detection (roundlo=-x, roundhi=+x).  Default "
                         "0.5 (tight, rejects extended emission).  On STAR-dominated "
                         "fields loosen to ~1.0: faint companions sitting on a bright "
                         "neighbour's residual gradient are distorted and fail the "
                         "tight cut (Arches: ~3x more recovered).  Do NOT loosen on "
                         "emission fields (shape is what rejects emission knots).")
    parser.add_option("--manual-seed-sharp-lo", dest="manual_seed_sharp_lo",
                    type='float', default=0.4,
                    help="DAOStarFinder sharpness lower bound for the residual seed "
                         "detection (default 0.4; star-field loosen ~0.2).")
    parser.add_option("--manual-seed-sharp-hi", dest="manual_seed_sharp_hi",
                    type='float', default=1.2,
                    help="DAOStarFinder sharpness upper bound for the residual seed "
                         "detection (default 1.2; star-field loosen ~1.5).")
    parser.add_option("--manual-ext-nmatch-confirm-maxpos-mas",
                    dest="manual_ext_nmatch_confirm_maxpos_mas",
                    type='float', default=0.0,
                    help="Position-stability guard for the multi-frame keep: only "
                         "keep if the across-exposure centroid scatter "
                         "hypot(std_ra,std_dec) <= this many mas.  Rejects "
                         "position-UNSTABLE spurious (cosmic-ray / noise "
                         "coincidences).  Does NOT reject extended emission (fixed "
                         "on-sky -> stable centroid).  Default 0 = off.")
    # Structure-noise prune + coarse-bg detection.  These shape/physics-based
    # discriminators reject broad extended-emission (PAH/nebulosity) bumps while
    # keeping faint point sources (which stay sharp and outpeak the local
    # structure regardless of brightness) -- unlike raising an S/N threshold,
    # which also kills faint real stars.  Previously only the miri_tuning path
    # set these; expose them so NIRCam LW (W51/Sickle/WD2) extended-emission
    # fields can enable them.  Default 0 = disabled (prior NIRCam behaviour).
    # The miri_tuning per-phase schedule still overrides these for MIRI.
    parser.add_option("--manual-struct-noise-x", dest="struct_x",
                    type='float', default=0.0,
                    help=("Structure-noise prune: keep a detection only if its peak "
                          "exceeds smoothed_bg + struct_x*real_noise + "
                          "struct_y*structure_noise.  struct_x scales the per-pixel "
                          "photon-noise term.  0 disables (default 0.0); NIRCam LW "
                          "PAH fields try ~2-3."))
    parser.add_option("--manual-struct-noise-y", dest="struct_y",
                    type='float', default=0.0,
                    help=("Structure-noise prune: struct_y scales the local STRUCTURE "
                          "noise (RMS of data-minus-smoothed-bg) -- the term that "
                          "rejects PAH/filament bumps while sparing point sources.  "
                          "0 disables (default 0.0); NIRCam LW try ~3-4."))
    parser.add_option("--extended-emission", dest="extended_emission",
                    action="store_true", default=None,
                    help=("Force extended-emission handling ON regardless of "
                          "--target: structure-noise prune (struct_x=1/struct_y=2), "
                          "emission noise-floor / prominence detection gates, and "
                          "coadd-seed protection.  Use for bright PAH/dust "
                          "nebulosity fields not in the built-in list "
                          "(w51/sickle/wd2/...)."))
    parser.add_option("--no-extended-emission", dest="extended_emission",
                    action="store_false",
                    help=("Force extended-emission handling OFF even for a target "
                          "that is in the built-in extended-emission list."))
    parser.add_option("--nircam-prom-m1", dest="nircam_prom_m1",
                    type='float', default=0.0,
                    help=("Extended-emission NIRCam (w51/sickle/wd2) per-pass "
                          "prominence-reject threshold for the m1 (iter1) pass: drop "
                          "fits whose data core does not rise >= this * annulus-MAD "
                          "above the local emission.  0 disables (default).  Use a "
                          "CONSERVATIVE (low) value here -- iter1 has no background "
                          "model yet."))
    parser.add_option("--nircam-prom-m2", dest="nircam_prom_m2",
                    type='float', default=0.0,
                    help=("Same as --nircam-prom-m1 but for the m2 (iter2) pass, which "
                          "detects on the iter1 source-subtracted residual.  Use a "
                          "more AGGRESSIVE (higher) value -- real stars stand out more "
                          "once iter1 sources are removed.  0 disables (default)."))
    parser.add_option("--nircam-prom-m3plus", dest="nircam_prom_m3plus",
                    type='float', default=0.0,
                    help=("Same prominence gate for m3..m6 (the background-subtracted "
                          "passes).  0 disables (default)."))
    parser.add_option("--manual-detect-noise-floor-box", dest="detect_noise_floor_box",
                    type='int', default=0,
                    help=("Extended-emission NIRCam (w51/sickle/wd2) detection cost cut: "
                          "when >0, daofind detects on the S/N image data/floor at a "
                          "fixed --manual-detect-noise-floor-k, where floor = the local "
                          "noise map median-filtered over this many px.  Raises the "
                          "detection bar on bright clumpy emission (fewer fake stars to "
                          "FIT) without a global threshold hike.  0 = off (default; the "
                          "historical min-noise threshold).  Try 61."))
    parser.add_option("--manual-detect-noise-floor-k", dest="detect_noise_floor_k",
                    type='float', default=5.0,
                    help=("S/N threshold for --manual-detect-noise-floor-box (peak must "
                          "exceed k * local emission-noise floor).  Default 5."))
    parser.add_option("--manual-detect-noise-floor-i2dseed", dest="detect_noise_floor_i2dseed",
                    type='int', default=0,
                    help=("Also apply the emission-noise-floor detection to the i2d "
                          "coadd-augmented seed (default 0 = per-frame passes only, so "
                          "the deep coadd still recovers faint-on-emission stars the "
                          "per-frame cut drops).  Set 1 for the most aggressive cost cut "
                          "at the expense of faint completeness."))
    parser.add_option("--manual-coarse-bg-box", dest="coarse_bg_box",
                    type='int', default=0,
                    help=("Detect on a coarse-background-subtracted image: subtract a "
                          "<box>px median before daofind so faint stars on a bright "
                          "but smooth extended pedestal are not lost.  0 disables "
                          "(default 0); MIRI raw rounds use 51."))
    parser.add_option("--saturation-data-floor", dest="saturation_data_floor",
                    type='float', default=-1.0,
                    help=("Only treat a SATURATED-DQ pixel as un-fittable when its "
                          "data exceeds this floor.  Guards against JUMP/persistence "
                          "artifacts mis-tagged SATURATED on unsaturated sources, "
                          "which otherwise drop a seeded real star from every frame "
                          "(e.g. W51 F480M, peak ~355 vs true saturation >1e4).  "
                          "-1 (default) = per-filter auto (NIRCam table; MIRI/unlisted "
                          "-> mask all); 0 = mask all SATURATED; >0 = explicit floor."))
    parser.add_option("--manual-group-min-sep-fwhm", dest="manual_group_min_sep_fwhm",
                    type='float', default=2.0,
                    help=("SourceGrouper grouping radius in FWHM (manual path; "
                          "requires --group).  Sources closer than this are fit "
                          "jointly.  Raise above 2.0 to jointly fit wider close "
                          "pairs that otherwise over-subtract in the valley "
                          "between them (default 2.0)."))
    parser.add_option("--resbg-mosaic-module", dest="resbg_mosaic_module",
                    default='',
                    help=("Module token of the iter3 residual mosaic to use as "
                          "the --use-iter3-residual-bg background (default: "
                          "'merged').  Targets whose whole-field co-add is a "
                          "single detector pass that detector (e.g. sickle LW "
                          "= 'nrcb')."),
                    metavar="resbg_mosaic_module")
    parser.add_option("--cutout-region", dest="cutout_region",
                    default='',
                    help=("Run the full per-exposure pipeline on only a small "
                          "cutout.  Either a DS9 .reg file (first region) or "
                          "'ra,dec,size' / 'ra,dec,w,h' (deg,deg,arcsec).  All "
                          "outputs are written under "
                          "<basepath>/cutouts/<label>/ so they never overwrite "
                          "full-frame photometry."),
                    metavar="cutout_region")
    parser.add_option("--cutout-label", dest="cutout_label",
                    default='',
                    help="Override the auto-derived cutout output-dir label.",
                    metavar="cutout_label")
    parser.add_option("--cutout-size-arcsec", dest="cutout_size_arcsec",
                    default=5.0, type='float',
                    help="Square cutout size (arcsec) for DS9 point regions / "
                         "fallback.  Default 5.0.",
                    metavar="cutout_size_arcsec")
    # Fit saturated stars whose centres lie OUTSIDE this frame's FOV (from
    # regions_/saturated_stars_outside_fov[_locked].reg) so their wings are
    # subtracted.  Tri-state: default (None) -> ON for normal runs, OFF for
    # --cutout-region runs.  The two flags force it on/off regardless.
    parser.add_option("--fit-satstar-outside-fov", dest="fit_satstar_outside_fov",
                    default=None, action='store_true',
                    help=("Force-enable fitting of saturated stars outside the "
                          "frame FOV (default: on for full-frame runs, off for "
                          "--cutout-region runs)."),
                    metavar="fit_satstar_outside_fov")
    parser.add_option("--no-fit-satstar-outside-fov", dest="fit_satstar_outside_fov",
                    action='store_false',
                    help="Force-disable fitting of saturated stars outside the FOV.")
    parser.add_option("--deblend-satstars", dest="deblend_satstars",
                    default=False, action='store_true',
                    help=("ZEROFRAME-deblend merged saturated cores: in crowded GC "
                          "fields (gc2211) bright stars' saturated cores touch and "
                          "label as one component, so the single seed lands between "
                          "stars.  Load the matching _ramp.fits ZEROFRAME and split "
                          "each merged component into one seed per star.  Auto-degrades "
                          "to legacy where a frame has no sibling _ramp.fits."),
                    metavar="deblend_satstars")
    parser.add_option("--epsf", dest="epsf",
                    default=False,
                    action='store_true',
                    help="try to make & use an ePSF?", metavar="epsf")
    parser.add_option("--blur", dest="blur",
                    default=False,
                    action='store_true',
                    help="blur the PSF?", metavar="blur")
    parser.add_option("--proposal_id", dest="proposal_id",
                    default='2221',
                    help="proposal_id", metavar="proposal_id")
    parser.add_option("--target", dest="target",
                    default='brick',
                    help="target", metavar="target")
    parser.add_option("--field", dest="field",
                    default=None,
                    help="Explicit field (e.g. '023' for proposal 2211 obs 023). "
                    "Required when a target maps to multiple fields under one "
                    "proposal (e.g. gc2211 has fields 023/028/046/049/050); "
                    "otherwise the field is derived from --target via "
                    "reg_to_field_mapping.", metavar="field")
    parser.add_option("--group", dest="group",
                      default=False,
                      action='store_true')
    parser.add_option('--each-exposure', dest='each_exposure',
                      default=False, action='store_true',
                      help='Photometer each exposure (REQUIRED for new runs; '
                           'mosaic-mode photometry is deprecated as of '
                           '2026-05-25 -- it skips satstar fitting and '
                           'bypasses iter3 plumbing).',
                      metavar='each_exposure')
    parser.add_option('--each-suffix', dest='each_suffix',
                      default='destreak_o001_crf',
                      help='Suffix for the level-2 products', metavar='each_suffix')
    parser.add_option('--each-suffix-overrides', dest='each_suffix_overrides',
                      default=None,
                      help=('Per-filter override of --each-suffix as a CSV of '
                            'FILTER:suffix pairs, e.g. '
                            '"F187N:destreak_o007_crf,F210M:destreak_o007_crf". '
                            'Filters not listed use --each-suffix.  Manual path '
                            'only; lets one run mix input crfs per filter (sickle '
                            'SW=destreak, LW=align) while keeping the m7 '
                            'cross-band seed.'),
                      metavar='each_suffix_overrides')
    parser.add_option('--manual-crossband-ref-filter', dest='manual_crossband_ref_filter',
                      default='',
                      help=('Astrometric reference filter for the manual-path '
                            'cross-band merge (final multifilter step).  Must be '
                            'one of --filternames.  Default: auto-select the '
                            'reddest broad/medium band (e.g. sickle -> F480M).'),
                      metavar='manual_crossband_ref_filter')
    parser.add_option('--manual-crossband-seed-dedup-mas', type=float,
                      dest='manual_crossband_seed_dedup_mas', default=30.0,
                      help=('Dedup radius (mas) for the m7 cross-band union seed: '
                            'a star seen in N co-aligned filters appears N times '
                            'within ~25 mas, so the union is deduped to one seed '
                            'per star before fitting (avoids degenerate co-located '
                            'groups that split flux).  Stay below the SW blend '
                            'limit (~64 mas) so resolvable pairs survive.  0 '
                            'disables.  Default 30.'),
                      metavar='manual_crossband_seed_dedup_mas')
    parser.add_option('--manual-crossband-seed-min-filters', type=int,
                      dest='manual_crossband_seed_min_filters', default=2,
                      help=('STRINGENT m7 cross-band seed: only seed positions '
                            'independently confirmed (SNR>min, qfit<max) in >= this '
                            'many filters.  Prevents a single-band (or i2d-structure) '
                            'detection from being force-fit into all bands as a fake '
                            'multi-band source (2026-06-30 fix).  Default 2.  Set 1 '
                            'to restore the legacy union seed (NOT recommended).'),
                      metavar='manual_crossband_seed_min_filters')
    parser.add_option('--manual-crossband-seed-snr-min', type=float,
                      dest='manual_crossband_seed_snr_min', default=5.0,
                      help='Per-filter SNR threshold for m7 cross-band seed confirmation. Default 5.',
                      metavar='manual_crossband_seed_snr_min')
    parser.add_option('--manual-crossband-seed-qfit-max', type=float,
                      dest='manual_crossband_seed_qfit_max', default=0.2,
                      help='Per-filter qfit ceiling for m7 cross-band seed confirmation. Default 0.2.',
                      metavar='manual_crossband_seed_qfit_max')
    parser.add_option('--manual-crossband-seed-max-sep-mas', type=float,
                      dest='manual_crossband_seed_max_sep_mas', default=30.0,
                      help='Cross-filter match radius (mas) for m7 cross-band seed confirmation clustering. Default 30.',
                      metavar='manual_crossband_seed_max_sep_mas')
    parser.add_option('--manual-start-phase', dest='manual_start_phase',
                      default='',
                      help=('Start the manual pipeline partway through (e.g. '
                            '"m7") reusing on-disk products from earlier phases, '
                            'so the big multifilter run can be split into small '
                            'per-filter jobs (m12..m6) + one finalize job (m7 + '
                            'cross-band merge).  Requires the previous phase '
                            'complete on disk.'),
                      metavar='manual_start_phase')
    parser.add_option('--manual-stop-after-phase', dest='manual_stop_after_phase',
                      default='',
                      help=('Run the manual pipeline only through this phase '
                            '(inclusive), then stop.  Combined with '
                            '--manual-start-phase this runs exactly one phase per '
                            'job -- the unit the per-frame SLURM fan-out '
                            '(submit_cataloging_perframe.sh) schedules.'),
                      metavar='manual_stop_after_phase')
    parser.add_option('--manual-frame-shard', dest='manual_frame_shard',
                      default='',
                      help=('"I/N": in the per-frame fit, process only frames '
                            'whose index %% N == I.  Lets one phase be fanned out '
                            'across N independent small SLURM array tasks.  Default '
                            'empty = fit all frames (monolithic).'),
                      metavar='manual_frame_shard')
    parser.add_option('--manual-skip-finalize', dest='manual_skip_finalize',
                      default=False, action='store_true',
                      help=('Fan-out worker mode: fit the (sharded) frames and '
                            'write per-frame completion markers, then STOP before '
                            'the per-phase barrier (reconcile/merge/vet/residual/'
                            'bg).  Pairs with --manual-finalize-only.'))
    parser.add_option('--manual-finalize-only', dest='manual_finalize_only',
                      default=False, action='store_true',
                      help=('Barrier job: do NOT fit frames; verify every '
                            'candidate frame has its phase completion marker '
                            '(hard-crash on any miss -> no silent exposure drop), '
                            'then run the per-phase barrier over the per-frame '
                            'products already on disk.'))
    parser.add_option('--no-forced-fill-m8', dest='forced_fill_m8',
                      default=True, action='store_false',
                      help=('Skip the inline m8 forced cross-band fill at the end of '
                            'the multifilter finalize (m7 + cross-band merge).  Use '
                            'when the m8 fill is fanned out into per-filter jobs '
                            '(perfilter_m8.sbatch + m8_merge_partials.py) instead, '
                            'since the monolithic fill over all 264 frames overruns '
                            'the 18h wall.'))
    parser.add_option('--manual-m8-partial', dest='manual_m8_partial',
                      default=False, action='store_true',
                      help=('With --manual-start-phase=m8 and a single --filternames '
                            'entry: run the forced cross-band fill for that one band '
                            'only and write a partial catalog '
                            '(..._resbgsub_m8_partial_<FILT>.fits) instead of the '
                            'combined m8.  Lets the m8 fill (which otherwise sweeps '
                            'all 264 frames serially and overruns the 18h wall) fan '
                            'out into 5 independent per-filter jobs; '
                            'scripts/reduction/m8_merge_partials.py column-merges '
                            'them into the final ..._resbgsub_m8.fits.'))
    parser.add_option('--no-m8-dedup', dest='m8_dedup',
                      default=True, action='store_false',
                      help=('Skip the post-m8 split-source de-duplication '
                            '(dedup_catalog.dedup_merged_catalog).  By default the '
                            'combined m8 is de-duplicated into a sibling '
                            '..._resbgsub_m8_dedup.fits: crowded-field cross-band '
                            'merges split a single star into two reference rows; the '
                            'dedup collapses complementary-coverage pairs while '
                            'preserving resolved binaries.'))
    parser.add_option('--seed-catalog', dest='seed_catalog',
                      default='',
                      help='Optional seed catalog for a seeded photometry rerun', metavar='seed_catalog')
    parser.add_option('--iteration-label', dest='iteration_label',
                      default='',
                      help='Optional iteration label to embed in output filenames', metavar='iteration_label')
    parser.add_option('--postprocess-residuals', dest='postprocess_residuals',
                      default=False,
                      action='store_true',
                      help='Apply negative-pixel masking and saturated-star infill before detection')
    parser.add_option('--basic-only', dest='basic_only',
                      default=False,
                      action='store_true',
                      help='Run only BASIC daophot photometry and residual generation')
    parser.add_option('--residual-negative-threshold', dest='residual_negative_threshold',
                      default=0.0,
                      type='float',
                      help='Pixels below this threshold are replaced with Gaussian infill before detection')
    parser.add_option('--local-snr-threshold', dest='local_snr_threshold',
                      default=5.0,
                      type='float',
                      help='Per-source local S/N threshold for retaining DAO detections')
    parser.add_option('--daofind-roundlo', dest='daofind_roundlo',
                      default=-1.0,
                      type='float',
                      help='DAOStarFinder roundness lower bound')
    parser.add_option('--daofind-roundhi', dest='daofind_roundhi',
                      default=1.0,
                      type='float',
                      help='DAOStarFinder roundness upper bound')
    parser.add_option('--satstar-artifact-ratio', dest='satstar_artifact_ratio',
                      default=1.0, type='float',
                      help='Reject post-fit sources where dao_model[y,x] '
                           '< RATIO * satstar_model[y,x] inside the gate. '
                           'Default 1.0 rejects fits dimmer than the artifact. '
                           'Set to 0 to disable.')
    parser.add_option('--satstar-artifact-sigK', dest='satstar_artifact_sigK',
                      default=3.0, type='float',
                      help='Sigma multiplier on median ERR for the gate. '
                           'Filter only applies where '
                           'satstar_model[y,x] > SIGK * median(err). '
                           'Default 3.')
    parser.add_option('--satstar-oversub-clamp-percentile',
                      dest='satstar_oversub_clamp_percentile',
                      default=10.0, type='float',
                      help='Percentile of data/model used as the over-subtraction '
                           'clamp scale for OUT-OF-FIELD (forced) saturated stars: '
                           'a smaller value enforces model<=data on a larger fraction '
                           'of the >5sigma footprint (10 -> 90%% of pixels, 1 -> 99%%, '
                           '0 -> every pixel), under-subtracting slightly instead of '
                           'leaving deep negative spike residuals. Default 10. Lower '
                           '(e.g. 1-2) for single-detector LW filters over bright '
                           'background (F335M).')
    parser.add_option("--satstar-zeroframe-recover", dest="satstar_zeroframe_recover",
                      default=False, action='store_true',
                      help=('Recover the charge-migration-INFLATED rim of the most-'
                            'saturated stars from the ramp first read (sibling '
                            '_ramp.fits group-0): replace the saturated rim with '
                            'R*group0 (R=median cal/group0 over bright unsat px) so '
                            'the PSF-subtracted residual no longer leaves a positive '
                            'ring/dot.  Deep core (group-0 also saturated) falls back '
                            'to model-replacement.  No-op where no _ramp.fits exists.'))
    parser.add_option("--satstar-zeroframe-dilate", type='int',
                      dest="satstar_zeroframe_dilate", default=3,
                      help=('Dilation (px) of the DQ-SATURATED mask for the ZEROFRAME '
                            'rim recovery, to catch charge-migration pixels spreading '
                            'beyond the hard DQ flag.  Default 3.'),
                      metavar="satstar_zeroframe_dilate")
    parser.add_option('--skip-mosaic-each-exposure-residuals',
                      dest='skip_mosaic_each_exposure_residuals',
                      default=False,
                      action='store_true',
                      help='After --each-exposure, resample all per-exposure residuals into a residual_i2d product by default; this parameter skips that step. Residual kinds are auto-determined based on enabled photometry types.')
    parser.add_option('--bundle-size', dest='bundle_size',
                      default=1, type='int',
                      help='Number of consecutive per-exposure iterations each SLURM array task handles (default 1 = one exposure per task).')
    parser.add_option('--skip-if-done', dest='skip_if_done',
                      default=False, action='store_true',
                      help='In --each-exposure mode, skip any exposure whose main output catalog already exists.')
    parser.add_option('--finalize-only', dest='finalize_only',
                      default=False, action='store_true',
                      help='Skip photometry and only run mosaic_each_exposure_residuals for the filter/module/iteration labels requested.')
    parser.add_option('--iteration-labels', dest='iteration_labels',
                      default='',
                      help='Comma-separated iteration labels for --finalize-only (empty entry = None). Example: ",iter2" runs both None and iter2.')
    parser.add_option('--list-missing-tasks', dest='list_missing_tasks',
                      default=False, action='store_true',
                      help='Enumerate --each-exposure work and print a comma-separated SLURM --array spec of only the bundled task indices that still need to run. Writes only the spec to stdout; logs go to stderr.')
    parser.add_option('--max-group-size', dest='max_group_size',
                      default='unlimited', type='string',
                      help=("Cap on photutils SourceGrouper group size.  Must be "
                            "EXPLICIT: 'unlimited' (no cap) or a POSITIVE integer. "
                            "The value 0 is REJECTED -- it was ambiguous (read as "
                            "'no grouping' but actually meant 'unlimited group "
                            "size').  Groups larger than the cap are split into "
                            "spatially coherent sub-groups via principal-axis "
                            "sorting before the joint fit.  Use 10-15 for dense "
                            "fields to keep joint fits tractable; 'unlimited' only "
                            "where blends are rare."))
    parser.add_option('--n-seed-chunks', dest='n_seed_chunks',
                      default=1, type='int',
                      help=('Split the seed catalog into N image-pixel tiles '
                            'and fit only the seeds in this chunk\'s tile.  '
                            'Combine with --seed-chunk-index.  Output filenames '
                            'gain a _chunkXXofYY token between the iter token '
                            'and _daophot_*; merge_catalogs.py vstacks chunks '
                            'back into one per-frame catalog.'))
    parser.add_option('--seed-chunk-index', dest='seed_chunk_index',
                      default=0, type='int',
                      help='Zero-based chunk index in [0, --n-seed-chunks).')
    parser.add_option('--parallel-workers', dest='parallel_workers',
                      default=1, type='int',
                      help=('EXPERIMENTAL: parallelize per-source PSF fitting '
                            'across N forked worker processes (default 1 = '
                            'serial, original behavior).  Sources are chunked '
                            'by SourceGrouper output so spatially overlapping '
                            'sources stay in the same chunk and never get '
                            'double-fit.  Off by default; only takes effect '
                            'when >1.  Validate against the serial path before '
                            'switching production jobs.'))
    parser.add_option('--parallel-chunk-size', dest='parallel_chunk_size',
                      default=100, type='int',
                      help=('Target sources per chunk when --parallel-workers '
                            '> 1.  Larger = fewer fork events but coarser '
                            'load-balance; ~100 is reasonable for nrca1-class '
                            'frames.'))
    (options, args) = parser.parse_args()

    # Deprecate mosaic-mode photometry (2026-05-25).  Reasons:
    # * mosaic-mode skips satstar fitting (the per-frame DQ_SATURATED gate is
    #   only meaningful in the original cal files, not in the drizzled
    #   mosaic), so saturated stars stay un-subtracted -- contradicts the
    #   "satstar first, before any daofind/daophot" requirement.
    # * Iter3 chains, --postprocess-residuals, force-union satstar, the
    #   satstar-artifact filter, seed_union, and all the recent fixes assume
    #   per-frame inputs.  Mosaic mode misses those benefits silently.
    # * No live launcher in /orange/.../shellscripts uses non-each-exposure
    #   mode for new runs; the remaining call sites are legacy.  Refuse
    #   here to force callers to use --each-exposure.
    # --finalize-only is allowed without --each-exposure because it only runs
    # mosaic_each_exposure_residuals on already-produced per-frame residuals.
    if (not options.each_exposure
            and not options.finalize_only
            and not options.list_missing_tasks):
        raise SystemExit(
            'mosaic-mode photometry (no --each-exposure) is deprecated. '
            'It cannot run satstar fitting (no per-frame DQ) and bypasses '
            'iter3/postprocess/force-union plumbing.  Re-invoke with '
            '--each-exposure.  See project_satstar_artifact_filter and '
            'project_seed_union_stale_data memories for rationale.'
        )

    # Validate chunking args and fold the chunk token into iteration_label so
    # every downstream filename composition (catalog, residual, satstar,
    # diagnostic plots) automatically picks it up without further plumbing.
    if options.n_seed_chunks and int(options.n_seed_chunks) > 1:
        n_chunks = int(options.n_seed_chunks)
        chunk_index = int(options.seed_chunk_index)
        if chunk_index < 0 or chunk_index >= n_chunks:
            raise SystemExit(
                f'--seed-chunk-index={chunk_index} out of range '
                f'for --n-seed-chunks={n_chunks}')
        chunk_tok = _chunk_token(chunk_index, n_chunks)
        if options.iteration_label:
            options.iteration_label = f'{options.iteration_label}{chunk_tok}'
        else:
            # Bare-chunk (no base iter label) is meaningful for ad-hoc runs
            # but unusual; tag the label so output files don't collide.
            options.iteration_label = chunk_tok.lstrip('_')
        # Also fold into the comma-separated --iteration-labels (used by
        # --finalize-only) when set, so chunked finalize calls find the
        # correct per-chunk filenames.
        if options.iteration_labels:
            options.iteration_labels = ','.join(
                (f'{tok}{chunk_tok}' if tok else chunk_tok.lstrip('_'))
                for tok in options.iteration_labels.split(','))

    filternames = options.filternames.split(",")
    modules = options.modules.split(",")
    proposal_id = options.proposal_id
    target = options.target

    nvisits = {'2221': {'brick': 2, 'cloudc': 2},
               # 2526 obs 021 = "G0" CMZ cloud-c filament F770W (1 visit),
               # routed into the cloudc tree.
               '2526': {'cloudc': 1},
               '1182': {'brick': 2},
               '3958': {'sickle': 1},
               # cloudef = Cloud E (obs 002) + Cloud F (obs 005), two
               # NIRCam pointings reduced together as one target.  Each
               # obs has only 1 visit; the catalog script's visit loop
               # is scoped to a single --field=<obs>, so nvisits is
               # per-obs (=1), not per-target (=2).  The per-obs cat
               # arrays in the wrapper iterate fields externally.
               '2092': {'cloudef': 1},
               '4147': {'sgrc': 1},
               '5365': {'sgrb2': 1},
               '2045': {'arches': 1, 'quintuplet': 1},
               '1939': {'sgra': 1},
               '2211': {'gc2211': 1},
               # Westerlund 1 (1905) + Westerlund 2 (3523): each proposal has one
               # main pointing per target.
               # Wd1 (1905) o001 is a 3-VISIT NIRCam mosaic (visits 001/002/003 are
               # offset tiles stepping ~2.7' E; together they span the full reduction
               # footprint). nvisits=1 previously cataloged only visit 001 (the west
               # tile) -> catalog + data_i2d covered ~1/3. Wd2 (3523) o005 is a single
               # visit (32 frames), so 1 is correct there.
               '1905': {'wd1': 3},
               '3523': {'wd2': 1},
               # w51 already exists in this codebase under proposals 1182 (obs 004)
               # and 6151 (obs 001).  Re-assert Gaia as ref via PipelineRerunNIRCAM-LONG.
               '6151': {'w51': 2},
               # Globular clusters (Jay Anderson co-I; added 2026-07-01). 1 visit
               # per field; 1979 'm4' spans two fields (o002, o003) run separately.
               '1334': {'m92': 1},
               '1979': {'ngc6397': 1, 'm4': 1},
               # NGC 6334 (Cat's Paw): 7213 o001 = 2 visits, 6778 o001 = 3 visits.
               '7213': {'ngc6334': 2},
               '6778': {'ngc6334': 3},
               }
    # 2211 is an asteroid-survey program with 5 separate GC pointings; all
    # map to the same 'gc2211' target/basepath, distinguished only by field.
    field_to_reg_mapping = {'2221': {'001': 'brick', '002': 'cloudc'},
                            '1182': {'004': 'brick'},
                            # 3958: 007 = NIRCam (sickle); MIRI pointings
                            # 001/002 = sickle, but 003 = the BRICK MIRI field
                            # (shares the 3958 program id, routed to brick/ so
                            # its catalogs do not land in / clash with sickle/).
                            # '001-002' = JOINT sickle run (both MIRI obs
                            # cataloged together; see get_filenames joint glob).
                            '3958': {'007': 'sickle', '001': 'sickle',
                                     '002': 'sickle', '003': 'brick',
                                     '001-002': 'sickle'},
                            # 2092: 002/005 = NIRCam; 004/006/008 = MIRI
                            '2092': {'002': 'cloudef', '005': 'cloudef',
                                     '004': 'cloudef', '006': 'cloudef',
                                     '008': 'cloudef'},
                            '4147': {'012': 'sgrc'},
                            # 5365 sgrb2 MIRI: obs 002 + obs 998 (skipped_redo);
                            # '002-998' = JOINT run combining both obs's 4 tiles.
                            '5365': {'001': 'sgrb2', '002': 'sgrb2',
                                     '998': 'sgrb2', '002-998': 'sgrb2'},
                            '2045': {'001': 'arches', '003': 'quintuplet'},
                            '1939': {'001': 'sgra'},
                            '2211': {'023': 'gc2211', '028': 'gc2211',
                                     '046': 'gc2211', '049': 'gc2211',
                                     '050': 'gc2211'},
                            '1905': {'001': 'wd1', '003': 'wd1'},
                            '3523': {'003': 'wd2', '005': 'wd2'},
                            '6151': {'001': 'w51'},
                            # 2526 obs 021 = "G0" CMZ cloud-c filament F770W
                            '2526': {'021': 'cloudc'},
                            # Globular clusters (Jay Anderson co-I; added 2026-07-01)
                            '1334': {'001': 'm92'},
                            '1979': {'001': 'ngc6397', '002': 'm4', '003': 'm4'},
                            # NGC 6334 (Cat's Paw SFR; extended emission)
                            '7213': {'001': 'ngc6334'},
                            '6778': {'001': 'ngc6334'},
                            }[proposal_id]
    # Instrument-dependent field numbering for MIRI (mirimage).  The map above is
    # NIRCam-era; proposal 2221 numbers the brick/cloudc MIRI pointings OPPOSITE
    # to its NIRCam pointings (NIRCam brick=001/cloudc=002; MIRI brick=002/
    # cloudc=001), and the w51 (6151) / sgrb2 (5365) MIRI pointings are obs 002,
    # which the NIRCam-era map omits.  Override only for mirimage so NIRCam runs
    # are untouched.
    if 'mirimage' in [str(m).lower() for m in modules]:
        if proposal_id == '2221':
            field_to_reg_mapping = {'002': 'brick', '001': 'cloudc'}
        elif proposal_id == '6151':
            field_to_reg_mapping = {'001': 'w51', '002': 'w51'}
        elif proposal_id == '5365':
            field_to_reg_mapping = {'001': 'sgrb2', '002': 'sgrb2',
                                    '998': 'sgrb2', '002-998': 'sgrb2'}
    reg_to_field_mapping = {v:k for k,v in field_to_reg_mapping.items()}
    # When multiple fields share a target (e.g. proposal 2211 / gc2211 has
    # 5 GC pointings 023/028/046/049/050), the inverted mapping collapses to
    # one entry, so prefer the explicit --field value when it's available.
    if getattr(options, 'field', None):
        field = str(options.field)
    else:
        field = reg_to_field_mapping[target]

    # Module restrictions per proposal/field/filter for single-module datasets
    # Sickle is NRCB-only (SUB640 subarray) but detectors differ by wavelength:
    # - Short-wavelength (F187N, F210M): nrcb1, nrcb2, nrcb3, nrcb4
    # - Long-wavelength (F335M, F470N, F480M): nrcb only
    modules_by_proposal_field_filter = {
        '3958': {
            '007': {
                'F187N': ('nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'),
                'F210M': ('nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'),
                'F335M': ('nrcb',),
                'F470N': ('nrcb',),
                'F480M': ('nrcb',),
            }
        }
    }
    # Check if there's a filter-specific policy
    allowed_modules = None
    if proposal_id in modules_by_proposal_field_filter:
        if field in modules_by_proposal_field_filter[proposal_id]:
            field_policy = modules_by_proposal_field_filter[proposal_id][field]
            # Check if any of the requested filters have a policy
            for filt in filternames:
                if filt in field_policy:
                    allowed_modules = field_policy[filt]
                    break
    
    # The manual-iteration path abstracts detectors behind the module TOKEN
    # (per-frame catalogs are saved as ``<filt>_<module>_...`` and get_filenames
    # / merge_individual_frames resolve SW nrcb->nrcb1-4 and LW nrcb->nrcblong
    # per filter internally), so a single token (e.g. 'nrcb') serves both a SW
    # and a LW filter in one multifilter (crossband) run.  The legacy
    # per-detector restriction below picks ONE filter's policy for all filters
    # and pre-expands SW to nrcb1-4, which breaks a mixed SW+LW manual run; skip
    # it for --manual-iterations (validated on arches F212N+F323N).
    if allowed_modules is not None and not getattr(options, 'manual_iterations', False):
        expanded_modules = []
        for module in modules:
            if proposal_id == '3958' and field == '007' and module in ('nrca', 'nrcb'):
                if any(filt in ('F187N', 'F210M') for filt in filternames):
                    expanded_modules.extend([f'{module}{number}' for number in range(1, 5)])
                    continue
            expanded_modules.append(module)

        filtered_modules = [module for module in expanded_modules if module in allowed_modules]
        if len(filtered_modules) == 0:
            raise ValueError(
                f"No requested modules are allowed for proposal_id={proposal_id} field={field} "
                f"Requested modules={modules}, expanded_modules={expanded_modules}, allowed modules={allowed_modules}"
            )
        if tuple(filtered_modules) != tuple(modules):
            print(
                f"Restricting modules for proposal_id={proposal_id} field={field} filters={filternames} "
                f"to {filtered_modules} because this dataset is explicitly single-module."
            )
        modules = filtered_modules

    if field_to_reg_mapping[field] in ('sickle', 'cloudef', 'sgrc', 'sgrb2', 'arches', 'quintuplet', 'sgra', 'gc2211', 'wd1', 'wd2', 'w51',
                                       # globular clusters (Anderson co-I) live on /orange
                                       'm92', 'ngc6397', 'm4',
                                       'ngc6334'):  # NGC 6334 (Cat's Paw SFR)
        basepath = f'/orange/adamginsburg/jwst/{field_to_reg_mapping[field]}/'
    else:
        basepath = f'/blue/adamginsburg/adamginsburg/jwst/{field_to_reg_mapping[field]}/'

    pl.close('all')

    if options.finalize_only:
        if options.iteration_labels != '':
            iteration_labels = [tok if tok != '' else None
                                for tok in options.iteration_labels.split(',')]
        else:
            iteration_labels = [options.iteration_label or None]
        mosaic_residual_kinds = []
        if options.daophot:
            mosaic_residual_kinds = ['basic'] if options.basic_only else ['basic', 'iterative']
        print(f"--finalize-only: running mosaics for labels={iteration_labels} kinds={mosaic_residual_kinds}",
              file=sys.stderr)
        for module in modules:
            for filtername in filternames:
                for lbl in iteration_labels:
                    for kind in mosaic_residual_kinds:
                        mosaic_each_exposure_residuals(basepath=basepath,
                                                      filtername=filtername,
                                                      proposal_id=proposal_id,
                                                      field=field,
                                                      module=module,
                                                      residual_kind=kind,
                                                      desat=options.desaturated,
                                                      bgsub=options.bgsub,
                                                      epsf=options.epsf,
                                                      blur=options.blur,
                                                      group=options.group,
                                                      pupil='clear',
                                                      iteration_label=lbl,
                                                      resbgsub=getattr(options, 'use_iter3_residual_bg', False))
        return

    if options.list_missing_tasks:
        bundle = max(1, options.bundle_size)
        # Silence log chatter so only the final array spec reaches stdout.
        _real_stdout = sys.stdout
        sys.stdout = sys.stderr
        try:
            index = -1
            missing_tasks = set()
            max_task = -1
            for module in modules:
                for filtername in filternames:
                    if not options.each_exposure:
                        continue
                    for visitid in range(1, nvisits[proposal_id][target] + 1):
                        visitid = f'{visitid:03d}'
                        try:
                            filenames = get_filenames(basepath, filtername, proposal_id,
                                                      field, visitid=visitid,
                                                      each_suffix=options.each_suffix,
                                                      module=module, pupil='clear')
                        except Exception as ex:
                            print(f"list-missing-tasks: no files for {filtername} {module} visit {visitid}: {ex}")
                            continue
                        for filename in sorted(filenames):
                            index += 1
                            task_idx = index // bundle
                            if task_idx > max_task:
                                max_task = task_idx
                            exposure_id = filename.split("_")[2]
                            visit_id = filename.split("_")[0][-3:]
                            vgroup_id = filename.split("_")[1]
                            # Match the per-file detector token convention
                            # used by the main each-exposure loop above.
                            file_detector = filename.split("_")[3]
                            file_module = file_detector if module == 'merged' else module
                            if not _expected_output_exists(
                                    basepath, filtername, file_module, options,
                                    visit_id, vgroup_id, exposure_id,
                                    iteration_label=options.iteration_label or None):
                                missing_tasks.add(task_idx)
        finally:
            sys.stdout = _real_stdout
        # Emit a sentinel-prefixed line so callers can grep it out of stdout
        # regardless of any module-import chatter that landed in stdout before
        # main() rerouted it.
        spec = ','.join(str(i) for i in sorted(missing_tasks))
        sys.stdout.write(f'__MISSING_TASKS__:{spec}\n')
        sys.stdout.flush()
        return

    print(f"options: {options}")

    bundle_size = max(1, options.bundle_size)
    # need to have incrementing _before_ test
    index = -1

    # --cutout-region bookkeeping: frames that don't overlap the region are
    # skipped; if NO frame in the run overlaps, that's an error (below).
    _cutout_run = bool(getattr(options, 'cutout_region', ''))
    _cutout_overlap_count = 0

    # The manual-iteration pipeline (cataloging.py) is the default path.  Its
    # phases are sequential (each detects on the previous phase's merged residual
    # mosaic), so the whole multi-phase pipeline runs IN-PROCESS as a single job
    # -- full-frame OR cutout -- never split across a SLURM array.  It is NOT
    # cutout-specific: with no --cutout-region it processes the full frames in
    # place under ``basepath``.  Run it as a single (non-array) job.
    #
    # The legacy --cutout-region in-process path (_run_cutout_pipeline) is kept
    # only for --legacy-iterations cutout runs.  Full-frame legacy runs still
    # fall through to the SLURM-array each-exposure loop below (byte-identical).
    if options.each_exposure and os.getenv('SLURM_ARRAY_TASK_ID') is None:
        if getattr(options, 'manual_iterations', False):
            # Imported lazily so the legacy path never depends on it and there is
            # no import cycle (cataloging imports mosaicking/IO from this module).
            from jwst_gc_pipeline.photometry import cataloging as _cataloging
            _cataloging.run_manual_pipeline(
                options, modules, filternames, nvisits, proposal_id, target,
                field, basepath, crowdsource_default_kwargs, bg_boxsizes)
            return
        if _cutout_run:
            # Legacy crowdsource cutout path, sequestered to photometry/legacy/.
            from jwst_gc_pipeline.photometry.legacy.crowdsource_step import _run_cutout_pipeline
            _run_cutout_pipeline(options, modules, filternames, nvisits, proposal_id,
                                 target, field, basepath, crowdsource_default_kwargs,
                                 bg_boxsizes)
            return
    if getattr(options, 'manual_iterations', False) and options.each_exposure \
            and os.getenv('SLURM_ARRAY_TASK_ID') is not None:
        raise SystemExit(
            "manual-iteration pipeline runs as a single in-process job; its phases "
            "are sequential and cannot be split across a SLURM array.  Re-submit "
            "without --array (unset SLURM_ARRAY_TASK_ID), or pass --legacy-iterations "
            "for the array-parallel per-exposure path.")

    for module in modules:
        detector = module # no sub-detectors for long-NIRCAM or for MIRI
        for filtername in filternames:
            if options.each_exposure:
                for visitid in range(1, nvisits[proposal_id][target] + 1):
                    visitid = f'{visitid:03d}'
                    filenames = get_filenames(basepath, filtername, proposal_id,
                                              field, visitid=visitid,
                                              each_suffix=options.each_suffix,
                                              module=module, pupil='clear')
                    if len(filenames) > 0:
                        print(f"Looping over filenames {filenames} for filter={filtername} proposal={proposal_id} field={field} visitid={visitid}")
                        # jw02221001001_07101_00024_nrcblong_destreak_o001_crf.fits
                        for filename in sorted(filenames):

                            index += 1
                            # enable array jobs with bundle-size K: task j
                            # handles indices [j*K, j*K + K)
                            task_env = os.getenv('SLURM_ARRAY_TASK_ID')
                            if task_env is not None:
                                task_idx = int(task_env)
                                lo = task_idx * bundle_size
                                hi = lo + bundle_size
                                if index < lo or index >= hi:
                                    print(f'Task={task_env} (bundle {bundle_size}, range [{lo},{hi})) skipping index {index}')
                                    continue

                            exposure_id = filename.split("_")[2]
                            visit_id = filename.split("_")[0][-3:]
                            vgroup_id = filename.split("_")[1]
                            # Extract the actual detector token from the
                            # filename (e.g. nrca1, nrcb3, nrcalong).  When
                            # ``module='merged'`` is passed, get_filenames()
                            # returned files across all detectors; saving
                            # all of them under the literal 'merged' token
                            # caused the 8 detector outputs per exposure to
                            # overwrite each other, dropping nmatch_good
                            # from the expected 6 (dithers) to 1.  Pass the
                            # per-file detector through so save_photutils_results
                            # writes a unique filename per (exposure, detector).
                            file_detector = filename.split("_")[3]
                            if module == 'merged':
                                file_module = file_detector
                            else:
                                file_module = module
                            if options.skip_if_done and _expected_output_exists(
                                    basepath, filtername, file_module, options,
                                    visit_id, vgroup_id, exposure_id,
                                    iteration_label=options.iteration_label or None):
                                print(f'skip-if-done: expected output exists for '
                                      f'{filtername} {file_module} visit={visit_id} '
                                      f'vgroup={vgroup_id} exp={exposure_id}; skipping.')
                                continue
                            try:
                                # Legacy crowdsource path, sequestered to photometry/legacy/.
                                from jwst_gc_pipeline.photometry.legacy.crowdsource_step import do_photometry_step
                                do_photometry_step(options, filtername, file_module, file_detector,
                                                   field, basepath, filename, proposal_id,
                                                   crowdsource_default_kwargs, exposurenumber=int(exposure_id),
                                                   visit_id=visit_id, vgroup_id=vgroup_id,
                                                   use_webbpsf=True,
                                                   bg_boxsizes=bg_boxsizes,
                                                   seed_catalog=options.seed_catalog or None,
                                                   iteration_label=options.iteration_label or None,
                                                   postprocess_residuals=options.postprocess_residuals or bool(options.seed_catalog),
                                                   residual_negative_threshold=options.residual_negative_threshold,
                                                   local_snr_threshold=options.local_snr_threshold,
                                                   daofind_roundlo=options.daofind_roundlo,
                                                   daofind_roundhi=options.daofind_roundhi)
                            except CutoutNoOverlap as ex:
                                # Frame doesn't cover the cutout region -- skip it.
                                print(f"cutout: skipping non-overlapping frame {filename} ({ex})",
                                      flush=True)
                                continue
                            if _cutout_run:
                                _cutout_overlap_count += 1

                if not options.skip_mosaic_each_exposure_residuals:
                    if os.getenv('SLURM_ARRAY_TASK_ID') is None:
                        # For a cutout run, mosaic the cutout residuals (under
                        # <basepath>/cutouts/<label>/) and skip make_starless
                        # (its target config is keyed on the full-frame
                        # basepath).  ResampleStep rectifies just the cutout
                        # region via the shifted GWCS on each cutout residual.
                        _mosaic_basepath = (_cutout_out_basepath(basepath, options)
                                            if _cutout_run else basepath)
                        _mosaic_starless = not _cutout_run
                        # Determine which residual kinds to mosaic based on enabled photometry types
                        mosaic_residual_kinds = []
                        if options.daophot:
                            mosaic_residual_kinds = ['basic'] if options.basic_only else ['basic', 'iterative']

                        for residual_kind in mosaic_residual_kinds:
                            mosaic_each_exposure_residuals(basepath=_mosaic_basepath,
                                                          filtername=filtername,
                                                          proposal_id=proposal_id,
                                                          field=field,
                                                          module=module,
                                                          residual_kind=residual_kind,
                                                          desat=options.desaturated,
                                                          bgsub=options.bgsub,
                                                          epsf=options.epsf,
                                                          blur=options.blur,
                                                          group=options.group,
                                                          pupil='clear',
                                                          iteration_label=options.iteration_label or None,
                                                          resbgsub=getattr(options, 'use_iter3_residual_bg', False),
                                                          make_starless=_mosaic_starless,
                                                          crop_to_data=_cutout_run)
                    else:
                        print('Skipping residual mosaicking in SLURM array-task mode.')

                # For a cutout run, also merge the per-exposure catalogs into a
                # single across-exposure catalog under the cutout tree
                # (combine_singleframe).  Saturated-star replacement runs too,
                # using the cutout's own satstar catalogs (basepath=_cut_bp);
                # only the per-filter reduction/fwhm_table.ecsv is read from
                # the real target basepath (fwhm_basepath) since the cutout
                # tree doesn't contain it.
                if (_cutout_run and os.getenv('SLURM_ARRAY_TASK_ID') is None
                        and options.daophot):
                    from jwst_gc_pipeline.photometry import merge_catalogs as _merge_catalogs
                    _cut_bp = _cutout_out_basepath(basepath, options)
                    os.makedirs(os.path.join(_cut_bp, 'catalogs'), exist_ok=True)
                    _merge_methods = [('dao', '_basic')]
                    if not options.basic_only:
                        _merge_methods.append(('daoiterative', '_iterative'))
                    for _mname, _msuffix in _merge_methods:
                        try:
                            _merge_catalogs.merge_individual_frames(
                                module=module, filtername=filtername.lower(),
                                progid=proposal_id, method=_mname, suffix=_msuffix,
                                target=target, basepath=_cut_bp,
                                iteration_label=options.iteration_label or None,
                                bgsub=options.bgsub, desat=options.desaturated,
                                epsf=options.epsf, blur=options.blur,
                                resbgsub=getattr(options, 'use_iter3_residual_bg', False),
                                fwhm_basepath=basepath)
                            print(f"cutout: wrote merged {_mname} catalog under "
                                  f"{_cut_bp}/catalogs/", flush=True)
                        except Exception as ex:
                            print(f"cutout: merge_individual_frames({_mname}) "
                                  f"failed: {ex}", flush=True)
            else:
                # Mosaic-mode photometry deprecated 2026-05-25 (see main()
                # deprecation guard).  Unreachable in normal CLI use, but
                # raise here too as a defensive backstop for any caller that
                # imports the module and bypasses the guard.
                raise RuntimeError(
                    'mosaic-mode photometry is deprecated; pass '
                    '--each-exposure')

    # If a cutout region was requested but overlapped NO frame at all, that's
    # almost certainly a wrong region / target -- fail loudly rather than
    # exit having silently done nothing.  (In SLURM array mode a given task
    # legitimately handles only a subset of frames, so don't raise there.)
    if (_cutout_run and os.getenv('SLURM_ARRAY_TASK_ID') is None
            and _cutout_overlap_count == 0):
        raise ValueError(
            f"--cutout-region={options.cutout_region!r} overlapped none of the "
            f"processed frames (filters={filternames}, modules={modules}).  "
            f"Check the region coordinates/target.")


def get_filenames(basepath, filtername, proposal_id, field, each_suffix, module, pupil='clear', visitid='001', allow_empty=False):

    # jw01182004002_02101_00012_nrcalong_destreak_o004_crf.fits
    # jw02221001001_07101_00012_nrcalong_destreak_o001_crf.fits
    # jw02221001001_05101_00022_nrcb3_destreak_o001_crf.fits
    # 2026-04-24: when module='merged' (used by the LW per-frame
    # photometry runs to indicate "both NIRCam long-wavelength
    # detectors"), the per-frame data files actually carry the
    # detector-specific tokens 'nrcalong' / 'nrcblong' in their names.
    # The earlier ``*{module}*`` glob with module='merged' returned 0
    # files for any LW filter because those files don't have the
    # literal substring 'merged'.  Expand the search so module='merged'
    # matches both detector tokens.
    # MIRI imaging has a single detector ('mirimage'); 'merged' there
    # collapses to that single token.
    if module == 'merged':
        if _instrument_from_filter(filtername) == 'MIRI':
            glob_modules = ['mirimage']
        else:
            # Include both LW (nrcalong/nrcblong) and SW (nrca1-4/nrcb1-4)
            # detector tokens.  glob.glob only matches tokens that actually
            # appear in filenames for this filter, so LW filters pick up only
            # nrcalong/nrcblong and SW filters pick up only nrca1..nrcb4.
            glob_modules = ['nrcalong', 'nrcblong',
                            'nrca1', 'nrca2', 'nrca3', 'nrca4',
                            'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']
    else:
        glob_modules = [module]
    # JOINT MULTI-OBS (2026-06-19): a hyphen-joined field token (e.g.
    # ``001-002``) means "catalog both observations together" -- glob each real
    # obs's frames so candidate_frames spans both pointings.  Everything
    # downstream (merge globs by vgroup* not obs; data_i2d / residual i2d
    # ResampleStep auto-unions the frame WCSs) is already obs-agnostic, so the
    # only obs-locked step is this glob.  The leading filename token encodes the
    # real obs number (jw0{proposal}{obs}{visit}); the ``o{obs}_crf`` suffix
    # token does too, so derive a per-obs suffix by substituting the obs digits.
    subfields = field.split('-') if '-' in str(field) else [field]
    fglob = []
    glstr_list = []
    for sf in subfields:
        if len(subfields) > 1:
            sf_suffix = re.sub(r'o\d{3}_crf', f'o{sf}_crf', each_suffix)
        else:
            sf_suffix = each_suffix
        for gm in glob_modules:
            glstr = f'{basepath}/{filtername}/pipeline/jw0{proposal_id}{sf}{visitid}*{gm}*{sf_suffix}.fits'
            glstr_list.append(glstr)
            fglob.extend(glob.glob(glstr))
    if len(fglob) == 0:
        if allow_empty:
            # Tolerated by callers sweeping a visit range (run_manual_pipeline):
            # a target's configured nvisits can exceed the visits actually
            # present for a given filter/obs (e.g. cloudc NIRCam has 2 visits but
            # the MIRI F2550W obs 001 has only visit 001), so an absent visit
            # must contribute zero frames, not crash the whole run.
            return []
        raise ValueError(f"No matches found to any of {glstr_list}")
    else:
        return sorted(set(fglob))


def get_filename(basepath, filtername, proposal_id, field, module, options, pupil='clear'):
    desat = '_unsatstar' if options.desaturated else ''
    bgsub = '_bgsub' if options.bgsub else ''
    #epsf_ = "_epsf" if options.epsf else ""
    #blur_ = "_blur" if options.blur else ""
    inst_token = _inst_token(filtername)

    filename = f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}_i2d.fits'
    if os.path.exists(filename):
        return filename

    candidate_patterns = [
        f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}_realigned-to-refcat.fits',
        f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}_i2d{desat}.fits',
        f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_F444W-{filtername.lower()}-{module}_nodestreak_realigned-to-refcat.fits',
        f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t*_{inst_token}_*{filtername.lower()}*{module}*i2d*.fits',
        f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t*_{inst_token}_*{filtername.lower()}*i2d*.fits',
        f'{basepath}/mastDownload/JWST/**/jw0{proposal_id}-o{field}_t*_{inst_token}_*{filtername.lower()}*{module}*i2d*.fits',
        f'{basepath}/mastDownload/JWST/**/jw0{proposal_id}-o{field}_t*_{inst_token}_*{filtername.lower()}*i2d*.fits',
    ]

    for glstr in candidate_patterns:
        fglob = glob.glob(glstr, recursive=True)
        if len(fglob) == 1:
            return fglob[0]
        if len(fglob) > 1:
            return sorted(fglob)[-1]

    raise ValueError(f"No input file found for filter={filtername} proposal={proposal_id} field={field} module={module} in {basepath}")



if __name__ == "__main__":
    main()
