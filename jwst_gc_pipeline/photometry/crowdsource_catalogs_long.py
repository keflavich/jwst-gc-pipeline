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
_PAR_IMAGE = None
_PAR_ERR = None
_PAR_MASK = None
_PAR_PHOT_KWARGS = None
_PAR_NEED_MODEL = False
_PAR_MODEL_PSF_SHAPE = None


def _par_worker_init(image, err, mask, phot_kwargs, need_model, model_psf_shape):
    global _PAR_IMAGE, _PAR_ERR, _PAR_MASK, _PAR_PHOT_KWARGS
    global _PAR_NEED_MODEL, _PAR_MODEL_PSF_SHAPE
    _PAR_IMAGE = image
    _PAR_ERR = err
    _PAR_MASK = mask
    _PAR_PHOT_KWARGS = phot_kwargs
    _PAR_NEED_MODEL = bool(need_model)
    _PAR_MODEL_PSF_SHAPE = model_psf_shape


def _par_worker_fit(args):
    """Worker entrypoint: fit one chunk; optionally also render its
    contribution to the model image and return it cropped to a tight
    bounding box so we don't pickle 16 MB per chunk."""
    chunk_idx, chunk_init = args
    photom = _make_psfphotometry(**_PAR_PHOT_KWARGS)
    tbl = photom(_PAR_IMAGE, error=_PAR_ERR, mask=_PAR_MASK,
                 init_params=chunk_init)
    if not _PAR_NEED_MODEL:
        return chunk_idx, tbl, None

    # Render this chunk's model contribution and crop to bbox.
    full_model = photom.make_model_image(
        _PAR_IMAGE.shape,
        psf_shape=_PAR_MODEL_PSF_SHAPE,
        include_local_bkg=False,
    )
    # Compute bbox from fit positions; pad by psf_shape//2 + 1.
    psf_h, psf_w = _PAR_MODEL_PSF_SHAPE
    pad_y, pad_x = psf_h // 2 + 1, psf_w // 2 + 1
    xfit = np.asarray(tbl['x_fit'], dtype=float)
    yfit = np.asarray(tbl['y_fit'], dtype=float)
    if len(xfit) == 0:
        return chunk_idx, tbl, None
    # Non-converged fits return NaN x_fit/y_fit; exclude from bbox.
    finite = np.isfinite(xfit) & np.isfinite(yfit)
    if not finite.any():
        return chunk_idx, tbl, None
    xfit = xfit[finite]
    yfit = yfit[finite]
    ymin = max(0, int(np.floor(yfit.min())) - pad_y)
    ymax = min(_PAR_IMAGE.shape[0], int(np.ceil(yfit.max())) + pad_y + 1)
    xmin = max(0, int(np.floor(xfit.min())) - pad_x)
    xmax = min(_PAR_IMAGE.shape[1], int(np.ceil(xfit.max())) + pad_x + 1)
    sub = full_model[ymin:ymax, xmin:xmax].copy()
    return chunk_idx, tbl, (ymin, ymax, xmin, xmax, sub)


def _chunk_init_by_group(init_params, group_id, target_size):
    """Partition init_params row-indices into chunks, never splitting a
    group across chunks.  Returns list[np.ndarray[int]]."""
    by_group = {}
    for i, gid in enumerate(group_id):
        by_group.setdefault(int(gid), []).append(int(i))
    # Order from largest group to smallest, then pack greedily.
    groups = sorted(by_group.values(), key=lambda g: -len(g))
    chunks = []
    current = []
    for grp in groups:
        if current and len(current) + len(grp) > target_size:
            chunks.append(current)
            current = []
        current.extend(grp)
    if current:
        chunks.append(current)
    return [np.asarray(c, dtype=np.int64) for c in chunks]


def _kdtree_group_ids(x, y, min_separation):
    """O(N log N) replacement for ``SourceGrouper`` used only for the
    chunk-partition step.  ``photutils.psf.SourceGrouper`` builds the
    full pairwise distance matrix via ``scipy.cluster.hierarchy.fclusterdata``
    -- O(N**2) memory, which OOMs (>100 GB) at N ~ 1.5e5 dense-field
    iter3 seed counts.  Here we only need a connectivity grouping
    (any two sources within ``min_separation`` are in the same chunk),
    so KDTree.query_pairs + a union-find over those edges suffices.

    Returns an integer group label array of shape (N,), labels >= 1.
    """
    from scipy.spatial import cKDTree
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    parent = np.arange(n, dtype=np.int64)

    def find(i):
        # iterative path-compression
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    if n > 1:
        tree = cKDTree(np.column_stack([x, y]))
        for i, j in tree.query_pairs(r=float(min_separation), output_type='ndarray'):
            union(int(i), int(j))

    # Compact root labels to 1..n_groups
    roots = np.array([find(i) for i in range(n)], dtype=np.int64)
    _, inverse = np.unique(roots, return_inverse=True)
    return (inverse + 1).astype(np.int64)


def _parallel_psfphotometry(image, *, photometry_kwargs, init_params,
                             error, mask, n_workers, chunk_size,
                             group_min_separation,
                             return_model=False, model_psf_shape=(15, 15)):
    """Run PSFPhotometry on init_params in parallel, returning the
    vstacked result table (and optionally a model image)."""
    group_id = _kdtree_group_ids(init_params['x_init'],
                                 init_params['y_init'],
                                 group_min_separation)
    chunk_idx_lists = _chunk_init_by_group(init_params, group_id, chunk_size)
    print(f"_parallel_psfphotometry: {len(init_params)} sources, "
          f"{len(np.unique(group_id))} groups, {len(chunk_idx_lists)} chunks, "
          f"{n_workers} workers", flush=True)
    if len(chunk_idx_lists) == 0:
        return init_params[:0], (np.zeros_like(image) if return_model else None)

    payload = [(i, init_params[idx]) for i, idx in enumerate(chunk_idx_lists)]
    ctx = _mp_par.get_context("fork")
    with ctx.Pool(processes=n_workers,
                  initializer=_par_worker_init,
                  initargs=(image, error, mask, photometry_kwargs,
                            return_model, model_psf_shape)) as pool:
        results = pool.map(_par_worker_fit, payload)
    results.sort(key=lambda r: r[0])
    tables = [r[1] for r in results]
    result_tbl = vstack(tables)

    model_image = None
    if return_model:
        model_image = np.zeros(image.shape, dtype=np.float32)
        for _, _, payload_model in results:
            if payload_model is None:
                continue
            ymin, ymax, xmin, xmax, sub = payload_model
            model_image[ymin:ymax, xmin:xmax] += sub
    return result_tbl, model_image


def _render_model_from_table(table, psf_model, shape, psf_shape):
    """Render a model image by evaluating ``psf_model`` at each
    (x_fit, y_fit, flux_fit) row of ``table``.  Used by the parallel
    path's stand-in photometry object so downstream make_model_image
    calls re-render from the (possibly filtered) results table without
    needing the underlying photutils _fit_models state.

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


class _FakePhot:
    """Stand-in for an IterativePSFPhotometry/PSFPhotometry instance,
    exposing only the attributes downstream code reads/writes after the
    fit:

      - ``.results``                         : Table (mutable)
      - ``.init_params``                     : Table or None
      - ``._psfphot.init_params``            : same Table (compat with
                                               dedup that updates inner)
      - ``.fit_results``                     : empty list (no per-iter
                                               snapshots; make_model_image
                                               re-renders from results)
      - ``.make_model_image(shape, psf_shape=, include_local_bkg=)``

    Used only when ``--parallel-workers > 1``; the serial path keeps
    real photutils objects so make_model_image's optimized _fit_models
    path is unaffected.
    """

    def __init__(self, results, psf_model, init_params=None):
        self.results = results
        self.init_params = init_params
        self.fit_results = []  # _filter_near_saturation tolerates empty
        self._psf_model = psf_model
        class _Inner:
            pass
        self._psfphot = _Inner()
        self._psfphot.init_params = init_params

    def make_model_image(self, shape, *, psf_shape=None, include_local_bkg=False):
        # include_local_bkg ignored: serial path also passes False for
        # all known production call sites.
        if psf_shape is None:
            psf_shape = (15, 15)
        return _render_model_from_table(self.results, self._psf_model,
                                        shape, psf_shape)


def _parallel_iterative_psfphotometry(image, *, photometry_kwargs, finder,
                                       init_params, error, mask,
                                       maxiters, sub_shape, psf_model,
                                       n_workers, chunk_size,
                                       group_min_separation):
    """Reimplement IterativePSFPhotometry mode='new' with chunked fits.

    On each iteration, run the (serial, cheap) finder on the current
    residual to discover new sources, parallel-fit them, subtract their
    rendered model, and continue.  When init_params is provided, the
    first iteration uses those instead of running the finder, mirroring
    IterativePSFPhotometry(init_params=...).
    """
    residual = image.copy()
    accumulated_tables = []
    for it in range(maxiters):
        if it == 0 and init_params is not None and len(init_params) > 0:
            iter_init = init_params
        else:
            sources = finder(residual, mask=mask)
            if sources is None or len(sources) == 0:
                print(f"  iter {it}: no new sources from finder; stopping",
                      flush=True)
                break
            iter_init = Table()
            # photutils finders return x_centroid/y_centroid in 3.x and
            # xcentroid/ycentroid in 2.x; handle both.
            xcol = 'x_centroid' if 'x_centroid' in sources.colnames else 'xcentroid'
            ycol = 'y_centroid' if 'y_centroid' in sources.colnames else 'ycentroid'
            iter_init['x_init'] = sources[xcol]
            iter_init['y_init'] = sources[ycol]
            iter_init['flux_init'] = sources['flux']

        print(f"  parallel iter {it}: fitting {len(iter_init)} sources",
              flush=True)
        tbl, model_img = _parallel_psfphotometry(
            residual,
            photometry_kwargs=photometry_kwargs,
            init_params=iter_init,
            error=error, mask=mask,
            n_workers=n_workers, chunk_size=chunk_size,
            group_min_separation=group_min_separation,
            return_model=True, model_psf_shape=sub_shape,
        )
        tbl['iter_detected'] = np.full(len(tbl), it + 1, dtype=np.int32)
        accumulated_tables.append(tbl)
        residual = residual - model_img

    if not accumulated_tables:
        return image[:0]  # bogus placeholder, but callers should handle len()==0
    return vstack(accumulated_tables)


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

    out_basepath = os.path.join(basepath, 'cutouts', label)
    out_dir = os.path.join(out_basepath, filtername, 'pipeline')
    os.makedirs(out_dir, exist_ok=True)
    cutout_filename = os.path.join(
        out_dir, os.path.basename(filename).replace('.fits', f'_cutout_{label}.fits'))

    with ImageModel(filename) as m:
        ny, nx = m.data.shape

        def _crop(a):
            if a is None or np.size(a) == 0:
                return a
            if a.ndim == 2 and a.shape == (ny, nx):
                return a[yslc, xslc]
            if a.ndim == 3 and a.shape[-2:] == (ny, nx):
                return a[:, yslc, xslc]
            return a

        new = m.copy()
        for attr in ('data', 'err', 'dq', 'wht', 'con', 'var_poisson',
                     'var_rnoise', 'var_flat', 'area'):
            if hasattr(m, attr):
                setattr(new, attr, _crop(getattr(m, attr)))
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

        def _crop(a):
            if a is None or np.size(a) == 0:
                return a
            if a.ndim == 2 and a.shape == (ny, nx):
                return a[yslc, xslc]
            if a.ndim == 3 and a.shape[-2:] == (ny, nx):
                return a[:, yslc, xslc]
            return a

        for attr in ('data', 'err', 'dq', 'wht', 'con', 'var_poisson',
                     'var_rnoise', 'var_flat', 'area'):
            if hasattr(m, attr):
                setattr(m, attr, _crop(getattr(m, attr)))
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

from jwst_gc_pipeline.reduction.saturated_star_finding import remove_saturated_stars

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

MIRI_FILTERS = frozenset(['f560w', 'f770w', 'f1000w', 'f1130w', 'f1280w',
                          'f1500w', 'f1800w', 'f2100w', 'f2550w'])

def _instrument_from_filter(filtername):
    """Return 'MIRI' or 'NIRCam' based on filter name (no header read needed)."""
    return 'MIRI' if str(filtername).lower() in MIRI_FILTERS else 'NIRCam'

def _inst_token(filtername):
    """Lowercased instrument token used in JWST i2d filename conventions."""
    return _instrument_from_filter(filtername).lower()

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


def _get_source_xy(tbl):
    """Return source x/y columns using the first available coordinate convention."""
    if 'x_fit' in tbl.colnames and 'y_fit' in tbl.colnames:
        return np.asarray(tbl['x_fit']), np.asarray(tbl['y_fit'])
    if 'xcentroid' in tbl.colnames and 'ycentroid' in tbl.colnames:
        return np.asarray(tbl['xcentroid']), np.asarray(tbl['ycentroid'])
    # photutils >=3.0 emits ``x_centroid``/``y_centroid``
    if 'x_centroid' in tbl.colnames and 'y_centroid' in tbl.colnames:
        return np.asarray(tbl['x_centroid']), np.asarray(tbl['y_centroid'])
    if 'x_init' in tbl.colnames and 'y_init' in tbl.colnames:
        return np.asarray(tbl['x_init']), np.asarray(tbl['y_init'])
    if 'x' in tbl.colnames and 'y' in tbl.colnames:
        return np.asarray(tbl['x']), np.asarray(tbl['y'])
    raise KeyError(f"No recognized x/y coordinate columns in {tbl.colnames}")


def _column_to_float_array(tbl, colname):
    col = tbl[colname]
    if hasattr(col, 'filled'):
        return np.asarray(col.filled(np.nan), dtype=float)
    return np.asarray(col, dtype=float)


def _best_available_xy(tbl):
    # photutils >=3.0 emits ``x_centroid``/``y_centroid`` from DAOStarFinder;
    # 2.x emits ``xcentroid``/``ycentroid``.  Accept both.
    candidates = [
        ('xcentroid', 'ycentroid'),
        ('x_centroid', 'y_centroid'),
        ('x_fit', 'y_fit'),
        ('x_init', 'y_init'),
        ('x', 'y'),
    ]
    best_pair = None
    best_score = -1
    best_x = None
    best_y = None
    for xname, yname in candidates:
        if xname in tbl.colnames and yname in tbl.colnames:
            xvals = _column_to_float_array(tbl, xname)
            yvals = _column_to_float_array(tbl, yname)
            score = np.isfinite(xvals).sum() + np.isfinite(yvals).sum()
            if score > best_score:
                best_score = score
                best_pair = (xname, yname)
                best_x = xvals
                best_y = yvals
    if best_pair is None:
        raise KeyError(f"No recognized x/y coordinate columns in {tbl.colnames}")
    return best_x, best_y


def _has_any_xy_columns(tbl):
    return any(
        xname in tbl.colnames and yname in tbl.colnames
        for xname, yname in (('xcentroid', 'ycentroid'),
                             ('x_centroid', 'y_centroid'),
                             ('x_fit', 'y_fit'),
                             ('x_init', 'y_init'), ('x', 'y'))
    )


def _skycoord_radec_arrays(tbl, colname):
    """Return ``(ra_deg, dec_deg)`` numpy arrays for every row of
    ``tbl[colname]``.

    ``tbl[colname]`` MUST be a vectorised ``SkyCoord``-mixin column.
    All producers in this module (``_resolve_seed_skycoords`` and
    ``_augment_seed_catalog_with_detections_sky``) now build mixin
    columns; an object-dtype column of SkyCoord scalars is treated as a
    bug at the producer site, not something to silently work around.
    """
    n = len(tbl)
    ra = np.full(n, np.nan, dtype=float)
    dec = np.full(n, np.nan, dtype=float)
    if n == 0 or colname not in tbl.colnames:
        return ra, dec

    col = tbl[colname]
    if not isinstance(col, SkyCoord):
        raise TypeError(
            f"_skycoord_radec_arrays expected tbl['{colname}'] to be a "
            f"SkyCoord-mixin column, got {type(col).__name__}.  Fix the "
            f"producer to assign a SkyCoord array, not an object-dtype "
            f"list of SkyCoord scalars."
        )
    ra_v = np.asarray(col.ra.deg, dtype=float)
    dec_v = np.asarray(col.dec.deg, dtype=float)
    if hasattr(col, 'mask') and col.mask is not None:
        valid = ~np.asarray(col.mask, dtype=bool)
        ra[valid] = ra_v[valid]
        dec[valid] = dec_v[valid]
    else:
        ra[:] = ra_v
        dec[:] = dec_v
    return ra, dec


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
    if method == 'daophot':
        return (f'{basepath}/{filtername}/'
                f'{filtername.lower()}_{module}{visitid_}{vgroupid_}{exposure_}'
                f'{desat}{bgsub}{epsf_}{blur_}{group_}{iter_}'
                f'_daophot_{basic_or_iterative}.fits')
    return (f'{basepath}/{filtername}/'
            f'{filtername.lower()}_{module}{visitid_}{vgroupid_}{exposure_}'
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


def _filter_near_saturation(phot_obj, dq, *, max_sat_dist_pix,
                            label, max_log_rows=50):
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
                              sig_K, ratio_cut, label, max_log_rows=50):
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
                                 file_suffix='',
                                 seed_gate_image=None, seed_gate_wcs=None):
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
                           file_suffix=file_suffix,
                           seed_gate_image=seed_gate_image,
                           seed_gate_wcs=seed_gate_wcs)
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
    locked_fn = f'{basepath}/regions_/saturated_stars_outside_fov_locked.reg'
    if os.path.exists(locked_fn):
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
    # Historical bug: this used to be `{module}{detector}` with no
    # separator, which produced doubled tokens like ``nrcbnrcb`` /
    # ``nrcanrca`` whenever ``module == detector`` (which is always
    # the case for the eachexp call paths) and broke the
    # ``merge_catalogs.py`` glob that expects just ``{module}``.
    # The original iter1 convention used only ``{module}`` and that's
    # what every other filename slot in this file (and the seed-catalog
    # inference at line ~1931) still uses.  Restored.
    tblfilename = f"{basepath}/{filtername}/{filtername.lower()}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}{iter_}_daophot_{basic_or_iterative}.fits"

    long_keys = [k for k in result.meta if len(k) > 8]
    for k in long_keys:
        result.meta[k[:8]] = result.meta[k]
        del result.meta[k]

    print(f"tblfilename={tblfilename}, filename={filename}, filtername={filtername}, module={module}, desat={desat}, bgsub={bgsub}, fpsf={fpsf} blur={blur}")

    result.write(tblfilename, overwrite=True)
    print(f"Completed {basic_or_iterative} photometry, and wrote out file {tblfilename}")

    return result


def save_crowdsource_results(results, ww, filename, suffix,
                             im1, detector,
                             basepath, filtername, module, desat, bgsub, exposure_, visitid_, vgroupid_,
                             psf=None,
                             blur=False,
                             options=None,
                             fpsf="",
                             iteration_label=None):
    print("Saving crowdsource results.")
    blur_ = "_blur" if blur else ""

    stars, modsky, skymsky, psf_ = results
    stars = Table(stars)
    coords = ww.pixel_to_world(stars['y'], stars['x'])
    stars['skycoord'] = coords
    stars['x'], stars['y'] = stars['y'], stars['x']
    stars['dx'], stars['dy'] = stars['dy'], stars['dx']

    pixscale = (ww.proj_plane_pixel_area()**0.5).to(u.arcsec)
    stars['dra'] = stars['dx'] * pixscale
    stars['ddec'] = stars['dy'] * pixscale
    if visitid_ is not None and visitid_ != '':
        stars.meta['visit'] = int(visitid_[-3:])
    if vgroupid_ is not None and vgroupid_ != '':
        stars.meta['vgroup'] = vgroupid_.removeprefix('_vgroup')
    stars.meta['filename'] = filename
    stars.meta['filter'] = filtername
    stars.meta['module'] = module
    stars.meta['detector'] = detector
    stars.meta['pixscale'] = pixscale.to(u.deg).value
    stars.meta['pixscale_as'] = pixscale.to(u.arcsec).value
    stars.meta['proposal_id'] = options.proposal_id
    if exposure_:
        stars.meta['exposure'] = exposure_
    if iteration_label not in (None, ''):
        stars.meta['iteration'] = str(iteration_label)
    if visitid_:
        stars.meta['visit'] = int(visitid_[-3:])
    if vgroupid_:
        stars.meta['vgroup'] = vgroupid_.removeprefix('_vgroup')

    if 'RAOFFSET' in im1[0].header:
        stars.meta['RAOFFSET'] = im1[0].header['RAOFFSET']
        stars.meta['DEOFFSET'] = im1[0].header['DEOFFSET']
    elif 'RAOFFSET' in im1[1].header:
        stars.meta['RAOFFSET'] = im1[1].header['RAOFFSET']
        stars.meta['DEOFFSET'] = im1[1].header['DEOFFSET']

    iter_ = _iteration_token(iteration_label)
    tblfilename = (f"{basepath}/{filtername}/"
                   f"{filtername.lower()}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}{iter_}"
                   f"_crowdsource_{suffix}.fits")

    print(f"tblfilename={tblfilename}, filename={filename}, suffix={suffix}, filtername={filtername}, module={module}, desat={desat}, bgsub={bgsub}, fpsf={fpsf} blur={blur}")

    stars.write(tblfilename, overwrite=True)
    with fits.open(tblfilename, mode='update', output_verify='fix') as fh:
        fh[0].header.update(im1[1].header)
    skymskyhdu = fits.PrimaryHDU(data=skymsky, header=im1[1].header)
    modskyhdu = fits.ImageHDU(data=modsky, header=im1[1].header)
    hdul = fits.HDUList([skymskyhdu, modskyhdu])
    hdul.writeto(f"{basepath}/{filtername}/{filtername.lower()}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}{iter_}_crowdsource_skymodel_{suffix}.fits", overwrite=True)

    if psf is not None:
        if hasattr(psf, 'stamp'):
            psfhdu = fits.PrimaryHDU(data=psf.stamp)
            psf_fn = (f"{basepath}/{filtername}/"
                      f"{filtername.lower()}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}{iter_}"
                      f"_crowdsource_{suffix}_psf.fits")
            psfhdu.writeto(psf_fn, overwrite=True)
        else:
            raise ValueError(f"PSF did not have a stamp attribute.  It was: {psf}, type={type(psf)}")

    return stars


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
        _psf_outdir = psf_cache_dir or '.'
        if instrument == 'MIRI':
            # MIRI imaging: single detector (MIRIM); no module split.
            _cache_detector = 'MIRIM'
        elif module in ('nrca', 'nrcb'):
            if 'F4' in filtername.upper() or 'F3' in filtername.upper():
                _cache_detector = f'{module.upper()}5'
            else:
                _cache_detector = f'{module.upper()}1'
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

        grid = psfgrid = to_griddedpsfmodel(f'{basepath}/psfs/{filtername.upper()}_{proposal_id}_{field}_merged_PSFgrid_oversample{oversample}{blur_}.fits')

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
        infilled_filename = output_filename.replace('_residual_i2d.fits', '_residual_infilled_i2d.fits')
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
    out_path = residual_i2d_path.replace('_residual_i2d.fits',
                                         '_residual_smoothed_bg_i2d.fits')
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


def _resample_to_i2d(files, pipeline_dir, product_name, crop_to_data=True):
    """Resample an explicit list of per-frame datamodels into one i2d mosaic.

    Generic ResampleStep coadd shared by the cutout data-i2d and merged-catalog
    residual mosaics.  ``crop_to_data`` trims the (over-allocated) canvas back
    to the finite-data bbox.
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
    resampled = ResampleStep.call(asn_filename, output_dir=pipeline_dir,
                                  save_results=False)
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
    out = _resample_to_i2d(files, pipeline_dir, product_name, crop_to_data=True)
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
                out_fn = resid_i2d.replace('_residual_i2d.fits', '_model_i2d.fits')
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
            if len(sat_sc):
                sxx, syy = ww.world_to_pixel(sat_sc)
                n_render_sat = 0
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
                        rx.append(sx); ry.append(sy); rf.append(sf)
                        n_render_sat += 1
                if n_render_sat:
                    print(f"mergedcat {kind} {os.path.basename(orig)}: rendering "
                          f"{n_render_sat} saturated stars NOT covered by this "
                          f"frame's satstar model (would otherwise stay in the "
                          f"residual)", flush=True)
            tbl = Table({'x_fit': np.asarray(rx, dtype=float),
                         'y_fit': np.asarray(ry, dtype=float),
                         'flux_fit': np.asarray(rf, dtype=float)})
            mc_model = _render_model_from_table(tbl, rg, base.shape, psf_shape)
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
            # NaN-mask ONLY deep over-subtraction PITS at saturated cores (MIRI).
            # The clipped saturated core, minus a TRUE-amplitude satstar model, can
            # gouge a deep negative pit (sickle bright pillar star: -755k) -- THOSE
            # pixels are undefined and get NaN'd.  But the MIRI SATURATED DQ flag is
            # broad (~16% of the detector: it tags the bright stars' spikes/wings,
            # whose DATA is perfectly valid), and NaN-ing ALL of them (a) erased the
            # bright flux the user needs in the residual and (b) BALLOONED through
            # resample (16% detector -> 12.5% i2d).  So restrict the NaN to
            # saturated pixels that are also a DEEP NEGATIVE pit; positive/normal
            # residual at saturated pixels (the unsubtracted bright flux) is KEPT.
            # Full-frame only (cutout shapes won't match the frame DQ -> guard skips).
            if _instrument_from_filter(filtername) == 'MIRI':
                try:
                    with fits.open(orig) as _oh:
                        _onames = [h.name for h in _oh]
                        _dq = _oh['DQ'].data if 'DQ' in _onames else None
                    if _dq is not None and _dq.shape == mc_resid.shape:
                        _satmask = (_dq.astype(np.int64) & 2) > 0
                        _fin = np.isfinite(mc_resid)
                        if _fin.any():
                            _med = np.nanmedian(mc_resid[_fin])
                            _nmad = 1.4826 * np.nanmedian(
                                np.abs(mc_resid[_fin] - _med)) + 1e-6
                            # pit = saturated AND >10 robust-sigma below the median
                            _pit = _satmask & (mc_resid < _med - 10.0 * _nmad)
                            mc_resid[_pit] = np.nan
                            print(f"  [miri resid] NaN'd {int(_pit.sum())} deep-pit "
                                  f"sat px (of {int(_satmask.sum())} sat); kept "
                                  f"bright flux", flush=True)
                except Exception as _dqex:
                    print(f"mergedcat: DQ-sat NaN-mask skipped for "
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


def _run_cutout_pipeline(options, modules, filternames, nvisits, proposal_id,
                         target, field, basepath, crowdsource_default_kwargs,
                         bg_boxsizes):
    """In-process multi-phase pipeline for a ``--cutout-region`` run.

    A cutout is small enough to run every phase sequentially in one process
    (no SLURM array), so the subsequent-step orchestration that the full-frame
    pipeline does via dependent SLURM jobs is done here as plain calls.

    Phases (single filter):  iter1 -> iter2 -> iter4
      * iter1  : unseeded per-frame photometry; merge per-frame catalogs;
                 build the iter1 residual i2d mosaic.
      * iter2  : re-fit each frame seeded by the MERGED iter1 catalog (the
                 merged catalog fed back in); merge; build iter2 residual i2d.
      * iter4  : median-smooth the iter2 residual mosaic, subtract it from the
                 input image, and re-fit seeded by the MERGED iter2 catalog
                 (residual built against the ORIGINAL data); merge.

    Multi-filter adds an ``iter3`` phase between iter2 and iter4 that seeds
    every filter from the cross-filter union of the iter2 merged catalogs.

    All outputs land under ``<basepath>/cutouts/<label>/`` (disjoint from
    full-frame products).  Frames not overlapping the region are skipped; if
    NO frame overlaps, raises (wrong region/target).
    """
    import copy
    from jwst_gc_pipeline.photometry import merge_catalogs as _merge_catalogs

    cut_bp = _cutout_out_basepath(basepath, options)
    os.makedirs(os.path.join(cut_bp, 'catalogs'), exist_ok=True)
    pupil = 'clear'
    multifilter = len(filternames) > 1

    phases = ['iter1', 'iter2']
    if multifilter:
        phases.append('iter3')
    phases.append('iter4')
    print(f"CUTOUT PIPELINE: label={_cutout_label_for(options)} "
          f"phases={phases} filters={filternames} modules={modules}", flush=True)

    def _merged_iter_path(phase, module, filt, kind='iterative'):
        """Reconstruct the merged minimal catalog path that
        merge_individual_frames writes for ``phase`` (matches its token logic).
        ``kind`` selects the iterative (daoiterative) or basic (dao) catalog."""
        desat = '_unsatstar' if options.desaturated else ''
        bgsub = ('_bgsub' if options.bgsub else '') + ('_resbgsub' if phase == 'iter4' else '')
        blur_ = '_blur' if options.blur else ''
        iter_token = '' if phase == 'iter1' else f'_{phase}'
        method_suffix = 'daoiterative_iterative' if kind == 'iterative' else 'dao_basic'
        return (f'{cut_bp}/catalogs/{filt.lower()}_{module}_indivexp_merged'
                f'{desat}{bgsub}{blur_}{iter_token}_{method_suffix}.fits')

    overlap_total = 0
    # mosaic infilled-paths recorded per (phase, module, filt) for iter4 bg build
    mosaic_paths = {}
    # overlapping-frame list recorded in iter1, reused by later phases
    frame_cache = {}

    for phase in phases:
        is_iter1 = (phase == 'iter1')
        iteration_label = None if is_iter1 else phase
        resbgsub = (phase == 'iter4')

        opts_phase = copy.copy(options)
        opts_phase.iteration_label = iteration_label or ''
        opts_phase.seed_catalog = ''
        # iter4 carries the _resbgsub filename token (it subtracts a residual
        # bg); drives _bgsub_token so per-frame/mosaic/merge names agree.
        opts_phase.use_iter3_residual_bg = resbgsub

        for module in modules:
            for filt in filternames:
                # --- determine seed + resbg for this (phase, module, filt) ---
                seed_catalog = None
                resbg_path = None
                if phase == 'iter2':
                    seed_catalog = _merged_iter_path('iter1', module, filt)
                elif phase == 'iter3':
                    seed_catalog = _build_cutout_union_seed(
                        cut_bp, modules, filternames, options)
                elif phase == 'iter4':
                    seed_src = 'iter3' if multifilter else 'iter2'
                    seed_catalog = _merged_iter_path(seed_src, module, filt)
                    if getattr(options, 'iter4_bg_exclude_badfit', False):
                        # EXPERIMENTAL: smoothed iter2 residual that subtracts
                        # only confident stars, leaving extended-emission false
                        # detections in the background so iter4 doesn't inflate
                        # fluxes to absorb them.  See NOTES_star_vs_extended_emission.md
                        resbg_path = build_filtered_iter2_residual_bg(
                            cut_bp, basepath, filt, proposal_id, field, module,
                            options, frame_cache.get((module, filt), []),
                            pupil=pupil)
                        print(f"iter4: built qfit-filtered smoothed bg "
                              f"{resbg_path}", flush=True)
                    if not resbg_path:
                        # standard smoothed-bg from the seed-source residual mosaic
                        src_infilled = mosaic_paths.get((seed_src, module, filt))
                        if src_infilled is None:
                            raise ValueError(
                                f"iter4 needs the {seed_src} residual mosaic for "
                                f"module={module} filt={filt}; none was produced.")
                        src_residual = src_infilled.replace(
                            '_residual_infilled_i2d.fits', '_residual_i2d.fits')
                        resbg_path = _cutout_smooth_residual_bg(src_residual)
                        print(f"iter4: built smoothed bg {resbg_path}", flush=True)

                if seed_catalog is not None and not os.path.exists(seed_catalog):
                    raise ValueError(
                        f"{phase}: seed catalog {seed_catalog} missing "
                        f"(prior phase merge did not produce it).")

                postprocess = options.postprocess_residuals or (seed_catalog is not None)

                # --- candidate frames ---
                # iter1 (first phase) scans every exposure of every visit and
                # records which ones overlap the cutout region (the overlap test
                # inside do_photometry_step costs ~10 s/frame); later phases reuse
                # that cached overlapping-frame list instead of re-scanning the
                # non-overlapping frames every phase.
                if phase == phases[0]:
                    candidate_frames = []
                    for visitid in range(1, nvisits[proposal_id][target] + 1):
                        candidate_frames.extend(sorted(get_filenames(
                            basepath, filt, proposal_id, field,
                            visitid=f'{visitid:03d}', each_suffix=options.each_suffix,
                            module=module, pupil='clear')))
                else:
                    candidate_frames = frame_cache.get((module, filt), [])

                n_overlap_phase = 0
                overlapping_now = []
                for filename in candidate_frames:
                    exposure_id = filename.split("_")[2]
                    visit_id = filename.split("_")[0][-3:]
                    vgroup_id = filename.split("_")[1]
                    file_detector = filename.split("_")[3]
                    file_module = file_detector if module == 'merged' else module
                    if options.skip_if_done and _expected_output_exists(
                            cut_bp, filt, file_module, opts_phase,
                            visit_id, vgroup_id, exposure_id,
                            iteration_label=iteration_label):
                        print(f'skip-if-done [{phase}]: {filt} {file_module} '
                              f'visit={visit_id} exp={exposure_id}', flush=True)
                        overlapping_now.append(filename)
                        n_overlap_phase += 1
                        continue
                    try:
                        do_photometry_step(
                            opts_phase, filt, file_module, file_detector,
                            field, basepath, filename, proposal_id,
                            crowdsource_default_kwargs,
                            exposurenumber=int(exposure_id),
                            visit_id=visit_id, vgroup_id=vgroup_id,
                            use_webbpsf=True, bg_boxsizes=bg_boxsizes,
                            seed_catalog=seed_catalog,
                            iteration_label=iteration_label,
                            postprocess_residuals=postprocess,
                            residual_negative_threshold=options.residual_negative_threshold,
                            local_snr_threshold=options.local_snr_threshold,
                            daofind_roundlo=options.daofind_roundlo,
                            daofind_roundhi=options.daofind_roundhi,
                            resbg_path=resbg_path)
                    except CutoutNoOverlap as ex:
                        print(f"cutout [{phase}]: skipping non-overlapping "
                              f"frame {filename} ({ex})", flush=True)
                        continue
                    overlapping_now.append(filename)
                    n_overlap_phase += 1

                if phase == phases[0]:
                    frame_cache[(module, filt)] = overlapping_now

                if n_overlap_phase == 0:
                    raise ValueError(
                        f"--cutout-region={options.cutout_region!r} overlapped "
                        f"none of the {filt}/{module} frames in phase {phase}.")
                if phase == phases[0]:
                    overlap_total += n_overlap_phase

                # --- merge per-frame catalogs FIRST (the merged-catalog
                # residual needs the vetted merged catalog) ---
                if options.daophot:
                    _merge_methods = [('dao', '_basic')]
                    if not options.basic_only:
                        _merge_methods.append(('daoiterative', '_iterative'))
                    for _mname, _msuffix in _merge_methods:
                        _merge_catalogs.merge_individual_frames(
                            module=module, filtername=filt.lower(),
                            progid=proposal_id, method=_mname, suffix=_msuffix,
                            target=target, basepath=cut_bp,
                            iteration_label=iteration_label,
                            bgsub=options.bgsub, desat=options.desaturated,
                            epsf=options.epsf, blur=options.blur,
                            resbgsub=resbgsub, fwhm_basepath=basepath)
                        print(f"cutout [{phase}]: merged {_mname} catalog under "
                              f"{cut_bp}/catalogs/", flush=True)
                    # iter4: flag likely-extended (non-star) detections by
                    # comparing iter2->iter4 centroid motion (diagnostic column).
                    if phase == 'iter4' and not options.basic_only:
                        try:
                            _pixscale = float(np.sqrt(np.abs(np.linalg.det(
                                wcs.WCS(fits.getheader(
                                    f'{cut_bp}/{filt}/pipeline/jw0{proposal_id}-o'
                                    f'{field}_t001_{_inst_token(filt)}_clear-'
                                    f'{filt.lower()}-{module}_data_i2d.fits',
                                    extname='SCI')).pixel_scale_matrix))) * 3600.0)
                            _flag_likely_extended_iter4(
                                _merged_iter_path('iter4', module, filt, 'iterative'),
                                _merged_iter_path('iter2', module, filt, 'iterative'),
                                _pixscale)
                        except Exception as ex:
                            print(f"cutout [iter4]: likely_extended flag failed: "
                                  f"{ex}", flush=True)

                # --- residual i2d(s) for this phase ---
                # --residual-source: 'mergedcat' (default), 'rawcat', or 'both'.
                # The merged-catalog residual drops spurious sources (which eat
                # extended emission in the raw residual).  The raw iterative
                # mosaic is also built when it is the iter4 background source
                # (seed_src), regardless of --residual-source.
                if not options.skip_mosaic_each_exposure_residuals and options.daophot:
                    residual_source = getattr(options, 'residual_source', 'mergedcat')
                    kinds = ['basic'] if options.basic_only else ['basic', 'iterative']
                    seed_src = 'iter3' if multifilter else 'iter2'
                    need_raw_iter_for_bg = (phase == seed_src)
                    build_raw = residual_source in ('rawcat', 'both')
                    for residual_kind in kinds:
                        do_raw = build_raw or (residual_kind == 'iterative'
                                               and need_raw_iter_for_bg)
                        if not do_raw:
                            continue
                        infilled = mosaic_each_exposure_residuals(
                            basepath=cut_bp, filtername=filt,
                            proposal_id=proposal_id, field=field, module=module,
                            residual_kind=residual_kind,
                            desat=options.desaturated, bgsub=options.bgsub,
                            epsf=options.epsf, blur=options.blur,
                            group=options.group, pupil=pupil,
                            iteration_label=iteration_label, resbgsub=resbgsub,
                            make_starless=False, crop_to_data=True)
                        if residual_kind == 'iterative':
                            mosaic_paths[(phase, module, filt)] = infilled

                    # merged-catalog residual i2d (default deliverable).  Built
                    # for the science kind (iterative, or basic if --basic-only),
                    # from the matching vetted merged catalog.
                    if residual_source in ('mergedcat', 'both'):
                        mc_kind = 'basic' if options.basic_only else 'iterative'
                        try:
                            # opts_phase (not options) carries the phase's bgsub
                            # token (iter4 -> _resbgsub) so the per-frame raw
                            # product names are reconstructed correctly.
                            build_mergedcat_residuals(
                                cut_bp, basepath,
                                _merged_iter_path(phase, module, filt, mc_kind),
                                filt, proposal_id, field, module, opts_phase,
                                frame_cache.get((module, filt), []),
                                iteration_label, [mc_kind], pupil=pupil)
                        except Exception as ex:
                            print(f"cutout [{phase}]: mergedcat residual failed: "
                                  f"{ex}", flush=True)

                    # original-data i2d (once, during iter1) so catalog sky
                    # positions can be overplotted on real data
                    if phase == phases[0]:
                        try:
                            mosaic_cutout_input_data(
                                cut_bp, filt, proposal_id, field, module,
                                _cutout_label_for(options), pupil=pupil)
                        except Exception as ex:
                            print(f"cutout: data i2d build failed: {ex}", flush=True)
                        try:
                            mosaic_cutout_satstar_flags(
                                cut_bp, filt, proposal_id, field, module,
                                _cutout_label_for(options), pupil=pupil)
                        except Exception as ex:
                            print(f"cutout: satstar flags i2d failed: {ex}", flush=True)

                    # final model image = data_i2d - residual_i2d (same grid),
                    # built for the final phase so the model can be overplotted /
                    # compared in CARTA.  Prefers the mergedcat residual when
                    # present (the vetted-catalog model), else the raw residual.
                    if phase == phases[-1]:
                        try:
                            _build_cutout_model_i2d(
                                cut_bp, filt, proposal_id, field, module,
                                iteration_label, resbgsub, options, pupil=pupil)
                        except Exception as ex:
                            print(f"cutout: model i2d build failed: {ex}", flush=True)

    print(f"CUTOUT PIPELINE DONE: {overlap_total} overlapping frames, "
          f"phases={phases}", flush=True)


def _build_cutout_union_seed(cut_bp, modules, filternames, options):
    """Build the cross-filter union seed for a multi-filter cutout iter3.

    Stacks the iter2 merged catalogs across all filters (and modules) into one
    skycoord seed table, writes it under the cutout catalogs/ dir, and returns
    its path.  Single-filter cutouts skip iter3 and never call this.
    """
    from astropy.table import vstack as _vstack
    desat = '_unsatstar' if options.desaturated else ''
    bgsub = '_bgsub' if options.bgsub else ''
    blur_ = '_blur' if options.blur else ''
    tbls = []
    for module in modules:
        for filt in filternames:
            p = (f'{cut_bp}/catalogs/{filt.lower()}_{module}_indivexp_merged'
                 f'{desat}{bgsub}{blur_}_iter2_daoiterative_iterative.fits')
            if os.path.exists(p):
                t = Table.read(p)
                if 'skycoord' in t.colnames:
                    tbls.append(Table({'skycoord': t['skycoord']}))
    if not tbls:
        raise ValueError("iter3 union seed: no iter2 merged catalogs found "
                         f"under {cut_bp}/catalogs/")
    union = _vstack(tbls, metadata_conflicts='silent')
    out = f'{cut_bp}/catalogs/union_seed_iter2_cutout.fits'
    union.write(out, overwrite=True)
    print(f"iter3: wrote cross-filter union seed {out} (n={len(union)})", flush=True)
    return out


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
    parser.add_option("--manual-ext-peak-over-bkg", dest="manual_ext_peak_over_bkg",
                    type='float', default=20.0,
                    help="Extended-emission vetting: keep if peak-SB > this x local bkg (default 20).")
    parser.add_option("--manual-ext-local-snr-min", dest="manual_ext_local_snr_min",
                    type='float', default=5.0,
                    help="Extended-emission vetting: require local S/N >= this (default 5).")
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

    nvisits = {'2221': {'brick': 1, 'cloudc': 2},
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
               '1905': {'wd1': 1},
               '3523': {'wd2': 1},
               # w51 already exists in this codebase under proposals 1182 (obs 004)
               # and 6151 (obs 001).  Re-assert Gaia as ref via PipelineRerunNIRCAM-LONG.
               '6151': {'w51': 1},
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
                            '5365': {'001': 'sgrb2'},
                            '2045': {'001': 'arches', '003': 'quintuplet'},
                            '1939': {'001': 'sgra'},
                            '2211': {'023': 'gc2211', '028': 'gc2211',
                                     '046': 'gc2211', '049': 'gc2211',
                                     '050': 'gc2211'},
                            '1905': {'001': 'wd1', '003': 'wd1'},
                            '3523': {'003': 'wd2', '005': 'wd2'},
                            '6151': {'001': 'w51'},
                            }[proposal_id]
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

    if field_to_reg_mapping[field] in ('sickle', 'cloudef', 'sgrc', 'sgrb2', 'arches', 'quintuplet', 'sgra', 'gc2211', 'wd1', 'wd2', 'w51'):
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


def get_filenames(basepath, filtername, proposal_id, field, each_suffix, module, pupil='clear', visitid='001'):

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


def do_photometry_step(options, filtername, module, detector, field, basepath,
                       filename, proposal_id, crowdsource_default_kwargs, exposurenumber=None,
                       visit_id=None, vgroup_id=None,
                       bg_boxsizes=None,
                       use_webbpsf=False,
                       nsigma=5,
                       local_snr_threshold=5.0,
                       daofind_roundlo=-1.0,
                       daofind_roundhi=1.0,
                       pupil='clear',
                       seed_catalog=None,
                       iteration_label=None,
                       postprocess_residuals=False,
                       residual_negative_threshold=0.0,
                       resbg_path=None):
    """
    nsigma is the threshold to multiply the error estimate by to get the detection threshold
    """
    print(f"Starting {field} filter {filtername} module {module} detector {detector} {exposurenumber}", flush=True)

    # Memory profiling is OFF by default everywhere -- it is debugging-only.
    # tracemalloc.start(25) instruments EVERY process-wide allocation and the
    # ~16 _mem_report() snapshots per frame (take_snapshot + statistics, some
    # deep=True) cost tens of seconds each on a multi-GB process, dwarfing the
    # actual photometry (it made each frame ~9 min).  Enable only with the
    # explicit --profile-memory flag when chasing a leak.
    _profile_mem = bool(getattr(options, 'profile_memory', False))
    if _profile_mem and not tracemalloc.is_tracing():
        tracemalloc.start(25)

    def _mem_report(label, deep=False):
        if not _profile_mem:
            return
        snap = tracemalloc.take_snapshot()
        top = snap.statistics('lineno')
        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        try:
            curr_kb = int(open('/proc/self/status').read().split('VmRSS:')[1].split()[0])
        except (OSError, IndexError, ValueError):
            # /proc/self/status may not exist (non-Linux), or VmRSS may be
            # absent / unparseable; mem report is diagnostic so 0 is fine.
            curr_kb = 0
        print(f"=MEM= {label}: curr={curr_kb/1e6:.2f}GB peak={peak_kb/1e6:.2f}GB", flush=True)
        for s in top[:12]:
            print(f"  {s.size/1e9:.3f}GB {s.traceback[0]}", flush=True)
        if deep and top:
            print(f"  --- traceback for #1 allocator ({top[0].size/1e9:.3f}GB) ---", flush=True)
            for frame in top[0].traceback.format():
                print(f"  {frame}", flush=True)
    fwhm_tbl = Table.read(FWHM_TABLE)
    row = fwhm_tbl[fwhm_tbl['Filter'] == filtername]
    fwhm = fwhm_arcsec = float(row['PSF FWHM (arcsec)'][0])
    fwhm_pix = float(row['PSF FWHM (pixel)'][0])

    # redundant, saves me renaming variables....
    filt = filtername

    # LocalBackground annulus, filter-scaled.  Inner must exceed the
    # photometric aperture (2*fwhm_pix) and sit beyond the first PSF
    # sidelobe (~1.6 FWHM peak; ~2.2 FWHM outer edge).  Floor at the
    # historical NIRCam-LW value (6,10) so existing NIRCam configs
    # are unchanged.
    aperture_radius_pix = 2.0 * fwhm_pix
    localbkg_inner = max(6, int(round(aperture_radius_pix + 0.5 * fwhm_pix)))
    localbkg_outer = localbkg_inner + max(4, int(round(fwhm_pix)))

    # file naming suffixes
    desat = '_unsatstar' if options.desaturated else ''
    bgsub = _bgsub_token(options)
    epsf_ = "_epsf" if options.epsf else ""
    exposure_ = f'_exp{exposurenumber:05d}' if exposurenumber is not None else ''
    visitid_ = f'_visit{int(visit_id):03d}' if visit_id is not None else ''
    vgroupid_, vgroup_numeric = normalize_vgroup_id(vgroup_id)
    blur_ = "_blur" if options.blur else ""
    group = "_group" if options.group else ""
    iter_ = _iteration_token(iteration_label)

    print(f"Starting cataloging on {filename}", flush=True)
    # ---- Optional small-region cutout ----------------------------------
    # Run the whole per-exposure pipeline on just a hand-specified region.
    # We write a cropped copy of the input into <basepath>/cutouts/<label>/
    # and run on THAT, so every output -- both basepath-derived (catalog,
    # residual, diagnostics) and filename-derived (satstar models, background
    # dumps) -- lands in the cutout tree and never overwrites full-frame
    # products.  basepath itself is redirected to out_basepath further below
    # (after all basepath INPUT reads are done).
    cutout_label = ''
    out_basepath = basepath
    _cutout_x0, _cutout_y0 = 0, 0
    if getattr(options, 'cutout_region', ''):
        cutout_label, filename, out_basepath, _cutout_x0, _cutout_y0 = _prepare_cutout_input(
            filename, basepath, filtername, options)
    _cutout_active = bool(cutout_label)
    if _cutout_active:
        # Disable diagnostic PNGs for cutout runs (whole invocation is a
        # cutout run, so this never affects full-frame output): make the
        # zoom-diagnostic a no-op and suppress all savefig writes.
        global _SUPPRESS_DIAGNOSTICS
        _SUPPRESS_DIAGNOSTICS = True
        pl.savefig = _noop_savefig

    fh, im1, data, wht, err, instrument, telescope, obsdate = load_data(filename)
    background_map = None
    inst_token = instrument.lower()

    # set up coordinate system
    ww = wcs.WCS(im1[1].header)
    pixscale = ww.proj_plane_pixel_area()**0.5
    cen = ww.pixel_to_world(im1[1].shape[1]/2, im1[1].shape[0]/2)

    # iter4resbgrefit builds its residual against the pristine image, so keep a
    # copy of the data *before* any background subtraction.  (The authoritative
    # is_resbg_refit flag is recomputed in the iteration block below.)
    _is_resbg_refit_early = (
        (_strip_chunk(iteration_label) or '').lower() in ('iter4resbgrefit', 'iter4'))
    original_data = data.copy() if _is_resbg_refit_early else None

    if options.bgsub:
        # background subtraction
        # see BackgroundEstimationExperiments.ipynb
        bkg = Background2D(data, box_size=bg_boxsizes[filt.lower()], bkg_estimator=MedianBackground())
        background_map = bkg.background
        fits.PrimaryHDU(data=bkg.background,
                        header=im1['SCI'].header).writeto(filename.replace(".fits",
                                                                           "_background.fits"),
                                                          overwrite=True)

        # subtract background, but then re-zero the edges
        zeros = data == 0
        data = data - bkg.background
        data[zeros] = 0

        fits.PrimaryHDU(data=data, header=im1['SCI'].header).writeto(filename.replace(".fits", "_bgsub.fits"), overwrite=True)

    if resbg_path or getattr(options, 'use_iter3_residual_bg', False):
        # 2026-04-25: alternative background subtraction that uses the
        # iter3 photometry residual (3x3-median-smoothed) as the
        # background estimate.  Built by make_iter3_residual_bgmaps.py
        # and consumed by the iter2-residbg / iter3-residbg cascade.
        #
        # 2026-06-06: use the whole-field iter3 residual *mosaic*, smoothed,
        # instead of the per-exposure residual.  The mosaic co-adds every
        # exposure so its background has much higher S/N.  It lives on the
        # mosaic pixel grid, so reproject it onto this exposure's WCS before
        # subtracting.  Built by make_iter3_residual_bgmaps.py.
        #
        # The mosaic's module token is configurable via
        # ``--resbg-mosaic-module`` (default 'merged').  Targets whose
        # whole-field co-add is a single detector (e.g. sickle LW = 'nrcb')
        # pass that token; SW four-detector co-adds use 'merged'.
        #
        # 2026-06-07: ``resbg_path`` (in-process cutout pipeline) overrides
        # the iter3-filename construction -- the cutout wrapper passes the
        # smoothed iter2 (or iter3) residual mosaic it just built.
        from reproject import reproject_interp
        if resbg_path:
            residbg_path = resbg_path
        else:
            _inst = _inst_token(filtername)
            _bg_module = getattr(options, 'resbg_mosaic_module', '') or 'merged'
            residbg_path = (
                f'{basepath}/{filtername}/pipeline/'
                f'jw0{proposal_id}-o{field}_t001_{_inst}_{pupil}-{filtername.lower()}-'
                f'{_bg_module}_iter3_daophot_iterative_residual_smoothed_bg_i2d.fits'
            )
        if not os.path.exists(residbg_path):
            raise ValueError(
                f"residual-bg subtraction requires the smoothed-bg mosaic "
                f"{residbg_path} to exist; run "
                f"`python make_iter3_residual_bgmaps.py --target=<target>` "
                f"after the iter3 residual mosaic is complete (or, for cutout "
                f"runs, the wrapper builds it from the iter2 residual mosaic)."
            )
        with fits.open(residbg_path) as bgh:
            if 'SCI' in [h.name for h in bgh]:
                bg_hdu = bgh['SCI']
            else:
                bg_hdu = bgh[0]
            bg_wcs = wcs.WCS(bg_hdu.header)
            bg_data = bg_hdu.data.astype(float)
        # Reproject the merged-grid background onto this exposure's grid.
        # Surface-brightness units (MJy/sr) are resolution-independent, so
        # interpolation across the grid change is valid without rescaling.
        bg_reproj, _ = reproject_interp((bg_data, bg_wcs), ww,
                                        shape_out=data.shape)
        n_nan = int(np.sum(~np.isfinite(bg_reproj)))
        bg_finite = np.where(np.isfinite(bg_reproj), bg_reproj, 0.0)
        zeros = data == 0
        data = data - bg_finite
        data[zeros] = 0
        background_map = bg_finite
        print(f"Subtracted merged iter3-residual-smoothed bg ({residbg_path}) "
              f"reprojected onto exposure grid: sum={float(np.nansum(bg_finite)):.3e} "
              f"MJy/sr-equiv, {n_nan} pix outside merged FOV (set to 0)", flush=True)
        # Diagnostics (both cutout and full-frame paths): save the reprojected
        # smoothed residual that was subtracted and the resulting source-finding
        # input (data - smoothed_residual), with the frame's SCI WCS so they
        # overplot against the catalog.  Tokens distinguish iter/bgsub variants.
        _diag_suffix = f"{_bgsub_token(options)}{_iteration_token(iteration_label)}"
        _sci_hdr = im1['SCI'].header
        try:
            fits.PrimaryHDU(data=bg_finite.astype('float32'), header=_sci_hdr).writeto(
                filename.replace('.fits', f'{_diag_suffix}_resbg_reproj.fits'),
                overwrite=True)
            fits.PrimaryHDU(data=data.astype('float32'), header=_sci_hdr).writeto(
                filename.replace('.fits', f'{_diag_suffix}_srcfind_input.fits'),
                overwrite=True)
            print(f"  wrote resbg diagnostics: *{_diag_suffix}_resbg_reproj.fits "
                  f"and *_srcfind_input.fits", flush=True)
        except (OSError, ValueError) as _ex:
            print(f"  WARNING: failed to write resbg diagnostics: {_ex}", flush=True)

    # try to limit memory use before we start photometry
    data = data.astype('float32')

    # Load PSF model
    _mem_report("before PSF load")
    grid, psf_model = get_psf_model(filtername, proposal_id, field,
                                    module=module,
                                    use_webbpsf=use_webbpsf,
                                    # if we're doing each exposure, we want the full grid
                                    use_grid=options.each_exposure,
                                    blur=options.blur,
                                    target=options.target,
                                    obsdate=obsdate,
                                    basepath='/blue/adamginsburg/adamginsburg/jwst/',
                                    psf_cache_dir=os.path.join(basepath, 'psfs'),
                                    instrument=instrument)
    dao_psf_model = grid
    _mem_report("after PSF load")

    if _cutout_active and (_cutout_x0 or _cutout_y0):
        # The spatially-varying PSF grid is indexed in the PARENT frame's
        # pixel coords, but the cutout data is 0-origin.  Re-origin the grid
        # by the cutout offset so a source at cutout pixel (cx, cy) is fit
        # with the SAME PSF the full-frame run would use at parent pixel
        # (cx + x0, cy + y0).  Without this the cutout silently uses a
        # mis-positioned (wrong) PSF.  Exact (verified maxdiff 0).
        from astropy.nddata import NDData as _NDData
        _shifted_xy = [(gx - _cutout_x0, gy - _cutout_y0)
                       for (gx, gy) in dao_psf_model.grid_xypos]
        dao_psf_model = type(dao_psf_model)(_NDData(
            np.asarray(dao_psf_model.data),
            meta={'grid_xypos': _shifted_xy,
                  'oversampling': dao_psf_model.oversampling}))
        grid = dao_psf_model
        print(f"CUTOUT: re-origined PSF grid by (-{_cutout_x0}, -{_cutout_y0}) "
              f"so the spatially-varying PSF matches the parent-frame fit",
              flush=True)

    # bound the flux to be >= 0 (no negative peak fitting)
    dao_psf_model.flux.min = 0

    dq, weight, bad = get_uncertainty(err, data, wht=wht, dq=im1['DQ'].data if 'DQ' in im1 else None)

    # SVO FPS uses mixed-case instrument names (e.g. NIRCam) while FITS headers
    # use all-caps (NIRCAM). Map to SVO conventions before lookup.
    _svo_inst_map = {'NIRCAM': 'NIRCam', 'NIRISS': 'NIRISS', 'NIRSPEC': 'NIRSpec', 'MIRI': 'MIRI'}
    _svo_instrument = _svo_inst_map.get(instrument.upper(), instrument)
    filter_table = SvoFps.get_filter_list(facility=telescope, instrument=_svo_instrument)
    filter_table.add_index('filterID')
    eff_wavelength = filter_table.loc[f'{telescope}/{_svo_instrument}.{filt}']['WavelengthEff'] * u.AA

    # DAO Photometry setup.  Use a CappedSourceGrouper when the caller
    # asks for a group-size cap (--max-group-size); the inner SourceGrouper
    # uses the same 2*FWHM linking distance as before.  ``resolve_max_group_size``
    # rejects the ambiguous 0 and returns None for 'unlimited'.
    _max_group_size = resolve_max_group_size(getattr(options, 'max_group_size', 'unlimited'))
    if _max_group_size is not None:
        grouper = CappedSourceGrouper(2 * fwhm_pix, max_size=_max_group_size)
    else:
        print("max_group_size=unlimited: SourceGrouper has no group-size cap.", flush=True)
        grouper = SourceGrouper(2 * fwhm_pix)
    mmm_bkg = MMMBackground()

    # empirically determined in debugging session with Taehwa on 2025-12-09:
    # with just nan_to_num, setting pixels to zero, some stars got "erased"
    kernel = Gaussian2DKernel(x_stddev=fwhm_pix/2.355)
    mask = np.isnan(data) | bad
    if 'DQ' in im1:
        dqarr = im1['DQ'].data
        is_saturated = (dqarr & dqflags.pixel['SATURATED']) != 0
        # we want original data_ to be untouched for imshowing diagnostics etc.
        data_ = data.copy()
        data_[is_saturated] = np.nan
        mask |= is_saturated
        # Honor the broader bad-DQ bitmask for the instrument.  For MIRI
        # this also drops NON_SCIENCE imager regions and PERSISTENCE
        # latents; for NIRCam it adds DO_NOT_USE coverage that wasn't
        # previously enforced here.
        bad_bitmask = _bad_dq_bitmask(instrument)
        is_baddq = (dqarr & bad_bitmask) != 0
        mask |= is_baddq
    else:
        data_ = data

    nan_replaced_data = interpolate_replace_nans(data_, kernel, convolve=convolve_fft,
                                                  allow_huge=True)

    # Infer a per-exposure ``_daophot_basic.fits`` seed for iter2, but *not*
    # for iter3 -- iter3 must use the cross-band union seed catalog that
    # the caller passes in explicitly.  Silently falling back to the basic
    # per-frame catalog would defeat the purpose of iter3 entirely.
    # Use the *base* iteration label so a chunk-suffixed compound label
    # (e.g. 'iter3_chunk03of08' from --n-seed-chunks > 1) still triggers
    # the iter3-specific code path (explicit-seed requirement, xy_bounds,
    # tighter local-SNR threshold).
    _base_label = _strip_chunk(iteration_label)
    is_iter3 = (_base_label is not None
                and str(_base_label).lower() == 'iter3')
    # iter4resbgrefit: final residual-bg refit step appended after iter3.
    # Re-fits the EXACT per-frame iter3 catalog as seeds with the iter3 tight
    # xy_bounds, on the residual-bg-subtracted data, and writes its residual
    # against the ORIGINAL (non-bg-subtracted) data.  Purely additive -- the
    # iter1/iter2/iter3 code paths are unchanged.
    # 'iter4' is the cutout-pipeline final refit (in-process wrapper): same
    # residual-from-original + tight-bounds behavior as iter4resbgrefit, but it
    # is seeded by an EXPLICIT merged catalog (passed by the wrapper), so the
    # per-frame iter3-seed inference below is gated on 'iter4resbgrefit' only.
    is_resbg_refit = (_base_label is not None
                      and str(_base_label).lower() in ('iter4resbgrefit', 'iter4'))
    if (seed_catalog is None and iteration_label not in (None, '')
            and not is_iter3 and not is_resbg_refit):
        inferred_seed_catalog = (
            f'{basepath}/{filtername}/'
            f'{filtername.lower()}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_daophot_basic.fits'
        )
        if os.path.exists(inferred_seed_catalog):
            seed_catalog = inferred_seed_catalog
    if (is_resbg_refit and seed_catalog is None
            and str(_base_label).lower() == 'iter4resbgrefit'):
        # Seed from this frame's own iter3 iterative catalog (the "exact
        # iter3 catalog").  That file carries NO bgsub token and the
        # ``_iter3`` iter token -- even though this run's bgsub token is
        # ``_resbgsub`` and its iter token is ``_iter4resbgrefit``.
        inferred_iter3_catalog = (
            f'{basepath}/{filtername}/'
            f'{filtername.lower()}_{module}{visitid_}{vgroupid_}{exposure_}{desat}{epsf_}{blur_}{group}_iter3_daophot_iterative.fits'
        )
        if not os.path.exists(inferred_iter3_catalog):
            raise ValueError(
                f"iteration_label='iter4resbgrefit' requires the per-frame "
                f"iter3 catalog {inferred_iter3_catalog} to exist; run iter3 "
                f"photometry first."
            )
        seed_catalog = inferred_iter3_catalog
    if is_iter3 and seed_catalog is None:
        raise ValueError(
            "iteration_label='iter3' requires an explicit seed_catalog "
            "pointing at the cross-band union seed file "
            "(build_union_seed_catalog.py output); no fallback is allowed."
        )

    is_second_iteration = seed_catalog is not None
    # iter3 position bound: ±1 SW NIRCam pixel (0.031"), expressed in the
    # current frame's pixel units.  On LW this is ~0.5 pix, on SW it is
    # 1 pix.  Kept None for iter1/iter2 so their behavior is unchanged.
    iter3_xy_bounds_pix = None
    if is_iter3 or is_resbg_refit:
        pixscale_arcsec = float(pixscale.to(u.arcsec).value)
        sw_pix_arcsec = 0.031
        iter3_xy_bounds_pix = float(sw_pix_arcsec / pixscale_arcsec)
        _xyb_label = 'iter4resbgrefit' if is_resbg_refit else 'iter3'
        print(f"{_xyb_label}: pixscale={pixscale_arcsec:.4f}\"/pix -> "
              f"xy_bounds=±{iter3_xy_bounds_pix:.3f} pix per source",
              flush=True)
    if is_second_iteration:
        # iter2 / iter3 tuned finder settings: lower local-SNR cut plus tighter
        # roundness/sharpness to suppress diffuse-background false detections.
        # iter3 is seed-dominated (the union catalog already knows where the
        # sources are), so its post-seed DAOStarFinder augmentation threshold
        # is raised to reduce the flood of low-SNR "discoveries" that add
        # little beyond what the union catalog provides.
        iter2_local_snr_threshold = 6.0 if is_iter3 else 3.0
        iter2_roundlo = -0.3
        iter2_roundhi = 0.3
        iter2_sharplo = 0.50
        iter2_sharphi = 1.00

        # Local-noise-map DAO thresholding for second-iteration residual search.
        local_noise_map = compute_local_noise_map(nan_replaced_data, smooth_sigma_pix=3.0)
        finite_noise = np.isfinite(local_noise_map) & (local_noise_map > 0)
        if not np.any(finite_noise):
            raise ValueError('Local noise map has no positive finite values')
        daofind_threshold = float(np.nanmin(local_noise_map[finite_noise]))
        daofind_tuned = DAOStarFinder(threshold=daofind_threshold,
                                      fwhm=fwhm_pix, roundhi=iter2_roundhi, roundlo=iter2_roundlo,
                                      sharplo=iter2_sharplo, sharphi=iter2_sharphi)
        print(
            f'DAO iter2 local-noise threshold={daofind_threshold}; '
            f'local_snr_threshold={iter2_local_snr_threshold}; '
            f'roundlo={iter2_roundlo}; roundhi={iter2_roundhi}; '
            f'sharplo={iter2_sharplo}; sharphi={iter2_sharphi}',
            flush=True,
        )
    else:
        # Keep original first-pass starfinding behavior unchanged.
        filtered_errest = np.nanmedian(err)
        print(f'Error estimate for DAO from median(err): {filtered_errest}', flush=True)
        # sigma_clipped stats get _much_ lower uncertainty for frames dominated by extended emission (maybe?).  At least, Sickle F470N had 3x too high error
        mean, med, std = stats.sigma_clipped_stats(data, stdfunc='mad_std')
        print(f'Error estimate for DAO from stats.: std={std}', flush=True)
        filtered_errest = min([filtered_errest, std])

        daofind_threshold = nsigma * filtered_errest
        daofind_tuned = DAOStarFinder(threshold=daofind_threshold,
                                      fwhm=fwhm_pix, roundhi=daofind_roundhi, roundlo=daofind_roundlo,
                                      sharplo=0.30, sharphi=1.40)
        print(
            f'DAO first-pass threshold={daofind_threshold}; '
            f'roundlo={daofind_roundlo}; roundhi={daofind_roundhi}',
            flush=True,
        )

    print("Finding stars with daofind_tuned", flush=True)

    satstar_table = None
    # Holds the (NaN-replaced) satstar model image after it has been
    # subtracted from ``nan_replaced_data``.  Re-applied to the residual
    # written to disk so the saved per-frame residual matches what the
    # fitter actually saw (i.e. data minus satstar wings minus phot model).
    satstar_model_subtracted = None
    # Satstar fitting + subtraction runs for EVERY photometry pass --
    # iter1, iter2, iter3 -- before any daofind/daophot.  The previous
    # gate ``and seed_catalog is not None`` left iter1 with bright
    # un-subtracted saturated stars.  Mosaic-mode photometry was
    # deprecated 2026-05-25 (see main() guard), so reaching this point
    # implies each-exposure mode and the per-frame DQ_SATURATED gate the
    # satstar fitter relies on is available.
    if True:
        # Optionally fit saturated stars whose centres lie OUTSIDE this frame's
        # FOV (their wings still bleed into the field).  Default: ON for
        # full-frame runs, OFF for cutout runs (a small cutout rarely benefits
        # and the off-FOV forced fits are wasteful there).  --fit-satstar-
        # outside-fov / --no-fit-satstar-outside-fov force it.  In-FOV satstar
        # fitting below is unaffected either way.
        _fit_outside = getattr(options, 'fit_satstar_outside_fov', None)
        if _fit_outside is None:
            _fit_outside = not _cutout_active
        if _fit_outside:
            # Cut at 32" — matches the radius of the large PSF grid used for forced
            # fits (fovp2048 SW × 0.031"/pix = 31.7"; fovp1024 LW × 0.063"/pix
            # = 32.3").  Anything farther falls outside PSF support so the
            # cutout would contain zero usable pixels and the fit would raise.
            outside_star_pixels, outside_locked = load_outside_fov_satstar_pixels(
                basepath, ww, data_shape=nan_replaced_data.shape, max_offset_arcsec=32.0)
        else:
            outside_star_pixels, outside_locked = [], False
        # When seeds came from the verified ``_locked.reg`` file, skip the
        # ±5 px grid search (radius=0 → single-point flux-only fit at the
        # locked position).  Default radius=5 otherwise.
        forced_grid_search_radius = 0 if outside_locked else 5
        # Namespace the satstar outputs by bgsub/iteration_label so that
        # the non-bgsub and bgsub iter2 array jobs (which can run concurrently
        # on the same frame) don't race each other on a shared filename.
        # The prior shared name (`<frame>_satstar_residual.fits`) caused
        # FileNotFoundError from astropy's writeto(overwrite=True) when a
        # sibling job deleted the file between the existence check and
        # the os.remove call.
        iter_tag = _iteration_token(iteration_label)
        satstar_file_suffix = f'{bgsub}{iter_tag}'
        # MIRI: feed the DEEP coadded data_i2d to the satstar seed gate so the
        # extended-emission phantom rejection (prominence + faint-core) is
        # measured on the noise-averaged coadd, not this single frame.  A
        # per-frame measurement lets a phantom escape via one frame's noise
        # spike (the cross-frame satstar merge then keeps it).  Frame-invariant
        # + matches what the user sees in the final i2d.  NIRCam: not loaded
        # (the gate is MIRI-only inside get_saturated_stars anyway).
        _seed_gate_image = _seed_gate_wcs = None
        if module == 'mirimage':
            _di2d_path = (f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-'
                          f'o{field}_t001_{_inst_token(filtername)}_{pupil}-'
                          f'{filtername.lower()}-{module}_data_i2d.fits')
            if os.path.exists(_di2d_path):
                try:
                    with fits.open(_di2d_path) as _dih:
                        _ext = 'SCI' if 'SCI' in [h.name for h in _dih] else 0
                        _seed_gate_image = _dih[_ext].data.astype(float)
                        _seed_gate_wcs = wcs.WCS(_dih[_ext].header)
                except Exception as _gex:
                    print(f"satstar seed gate: could not load coadd "
                          f"{_di2d_path}: {_gex}", flush=True)
            else:
                print(f"satstar seed gate: coadd data_i2d not found "
                      f"({_di2d_path}); gate falls back to per-frame data",
                      flush=True)
        satstar_table = load_or_make_satstar_catalog(
            filename,
            path_prefix=f'{basepath}/psfs',
            use_merged_psf_for_merged=(module == 'merged'),
            overwrite=bool(outside_star_pixels),
            outside_star_pixels=outside_star_pixels,
            outside_star_fit_box=512,
            forced_grid_search_radius=forced_grid_search_radius,
            file_suffix=satstar_file_suffix,
            seed_gate_image=_seed_gate_image,
            seed_gate_wcs=_seed_gate_wcs,
        )

        # Pipeline-plumbing fix (2026-04-21):
        # The satstar finder fits the bright/saturated stars and writes a
        # satstar_model.fits, but historically `phot_basic`/`phot_iter` ran on
        # ``nan_replaced_data`` (i.e. bgsub-only -- the satstar model was NOT
        # subtracted).  That left the wings of saturated stars fully visible
        # to the regular fitter, which then placed inflated fits at the
        # "stuck-low" central pixel and produced ~-15000-count holes in the
        # final residual image.  Subtract the satstar model here so the
        # downstream photometry sees the satstar-cleaned data.
        #
        # Filenames mirror those produced by remove_saturated_stars()
        # (saturated_star_finding.py) and load_or_make_satstar_catalog().
        # Prefer the extended model image (force-fit union of per-frame
        # satstar positions across the filter) when present.
        extended_model_path = filename.replace(
            '.fits', f'{satstar_file_suffix}_extended_satstar_model.fits')
        satstar_model_path = filename.replace(
            '.fits', f'{satstar_file_suffix}_satstar_model.fits')
        if os.path.exists(extended_model_path):
            satstar_model_path = extended_model_path
        if os.path.exists(satstar_model_path):
            try:
                satstar_model_image = fits.getdata(satstar_model_path).astype(float)
            except (OSError, ValueError) as exc:
                print(f"Could not read satstar_model {satstar_model_path}: {exc}; "
                      f"skipping satstar subtraction", flush=True)
            else:
                if satstar_model_image.shape != nan_replaced_data.shape:
                    print(f"satstar_model shape {satstar_model_image.shape} does not "
                          f"match data shape {nan_replaced_data.shape}; skipping "
                          f"satstar subtraction", flush=True)
                else:
                    finite_model = np.where(np.isfinite(satstar_model_image),
                                            satstar_model_image, 0.0)
                    n_pos = int(np.sum(finite_model > 0))
                    total = float(np.nansum(finite_model))
                    # At SATURATED-DQ pixels we replace the data with the
                    # satstar model BEFORE subtracting, forcing residual=0
                    # at those pixels.  Rationale: JWST ramp-fitter retains
                    # numeric values at saturated pixels (from group-1
                    # fits), often >> 1e3 MJy/sr.  interpolate_replace_nans
                    # only touches actual NaN, so retained values pass
                    # through.  Earlier "skip subtraction" attempt left
                    # those large positive data values in the residual
                    # (sickle F480M: 7078 MJy/sr at a worst-subtracted
                    # star, baseline 682; see 2026-06-01).  Substituting
                    # model-for-data at sat pixels is the user-directed
                    # behaviour: "saturated pixels should be replaced with
                    # model values from the fitted saturated stars".
                    if 'DQ' in im1:
                        was_saturated = (im1['DQ'].data
                                         & dqflags.pixel['SATURATED']) != 0
                        n_sat = int(was_saturated.sum())
                        nan_replaced_data = np.where(was_saturated,
                                                     finite_model,
                                                     nan_replaced_data)
                    else:
                        n_sat = 0
                    nan_replaced_data = nan_replaced_data - finite_model
                    satstar_model_subtracted = finite_model
                    print(f"Subtracted satstar_model ({satstar_model_path}) "
                          f"from nan_replaced_data: {n_pos} positive pixels, "
                          f"sum={total:.3e} counts; "
                          f"replaced {n_sat} SATURATED-DQ pixels with model "
                          f"before subtract (residual=0 there)",
                          flush=True)
        else:
            print(f"No satstar_model file at {satstar_model_path}; "
                  f"phot_basic/phot_iter will see the satstar wings unmodified",
                  flush=True)

    seeded_init_params = None
    if seed_catalog is not None:
        preferred_seed_skycoord_col = f'skycoord_{filtername.lower()}'
        merged_seed_table = _as_table(seed_catalog)

        # Snap seed positions to this filter's own iter2 astrometry where
        # available.  build_union_seed_catalog.py records only a single
        # ``skycoord_ref`` column taken from the SHORTEST-WAVELENGTH
        # filter that detected each cluster.  When fitting a long-
        # wavelength filter like F480M, the SW position can be offset
        # several pixels from the LW position (filter-dependent
        # astrometry + saturated-core centroid bias on SW, or simply there is no short-wavelength detection of this exact star).  With
        # iter3's tight xy_bounds (=1 SW pix, ~0.5 LW pix), fits cannot
        # move far enough to reach the true LW star: they end up at the
        # boundary (flags=48 = near_bound + no_covariance) with low
        # flux, and the unmodelled star reappears as a large positive
        # residual.  Diagnosed 2026-06-03 on sickle F480M union seed
        # source_id_union=7288 (flux_f480m=52333, skycoord_ref taken
        # from F187N detection ~5 LW pix away from the true F480M
        # position).
        #
        # Solution: load THIS filter's cross-exposure-merged iter2
        # daoiterative catalog (produced by merge_catalogs.py at
        # {basepath}/catalogs/{filt}_merged_indivexp_merged_iter2_
        # daoiterative_iterative.fits) and add a
        # ``skycoord_<filter>`` mixin column on the seed table whose
        # value is the iter2 skycoord of the nearest in-filter match
        # within ``match_radius`` of each seed.  ``SeededFinder``'s
        # ``preferred_skycoord_col`` already prefers
        # ``skycoord_{filter}`` (line above), so the snapped position
        # gets used automatically when it exists.  Seeds with no
        # nearby per-filter detection fall back to ``skycoord_ref``
        # (existing behaviour).
        try:
            _iter2_cat_path = os.path.join(
                basepath, 'catalogs',
                f'{filtername.lower()}_merged_indivexp_merged_iter2_'
                f'daoiterative_iterative.fits',
            )
            if os.path.exists(_iter2_cat_path):
                _it2 = Table.read(_iter2_cat_path)
                if 'skycoord' in _it2.colnames and len(_it2) > 0:
                    _it2_sk = _it2['skycoord']
                    if not isinstance(_it2_sk, SkyCoord):
                        _it2_sk = SkyCoord(_it2_sk)
                    _seed_table_resolved = _resolve_seed_skycoords(
                        Table(merged_seed_table, copy=True), ww=ww,
                        preferred_skycoord_col=preferred_seed_skycoord_col,
                    )
                    _seed_sk = _seed_table_resolved['skycoord']
                    if not isinstance(_seed_sk, SkyCoord):
                        _seed_sk = SkyCoord(_seed_sk)
                    _seed_sk = _seed_sk.unmasked if hasattr(_seed_sk, 'unmasked') else _seed_sk
                    _it2_sk = _it2_sk.unmasked if hasattr(_it2_sk, 'unmasked') else _it2_sk
                    # Bulk nearest-neighbor match.  Use 3 LW pix ~ 0.2"
                    # for LW filters; SW filters get 1.5 SW pix ~ 0.05".
                    _pixscale_arcsec = (ww.proj_plane_pixel_area()**0.5).to(u.arcsec).value if hasattr(ww, 'proj_plane_pixel_area') else 0.063
                    _match_radius_arcsec = max(0.15, 3.0 * _pixscale_arcsec)
                    _idx, _sep2d, _ = _seed_sk.match_to_catalog_sky(_it2_sk)
                    _good = (_sep2d.arcsec < _match_radius_arcsec)
                    if np.any(_good):
                        _new_ra = np.asarray(_seed_sk.ra.deg, dtype=float).copy()
                        _new_dec = np.asarray(_seed_sk.dec.deg, dtype=float).copy()
                        _new_ra[_good] = np.asarray(_it2_sk.ra.deg)[_idx[_good]]
                        _new_dec[_good] = np.asarray(_it2_sk.dec.deg)[_idx[_good]]
                        merged_seed_table[preferred_seed_skycoord_col] = SkyCoord(
                            ra=_new_ra * u.deg, dec=_new_dec * u.deg, frame='icrs')
                        seed_catalog = merged_seed_table
                        print(f"Snapped {int(np.sum(_good))} of {len(merged_seed_table)} "
                              f"seed positions to per-filter iter2 catalog "
                              f"({_iter2_cat_path.split('/')[-1]}) within "
                              f"{_match_radius_arcsec:.2f}\"; "
                              f"populated {preferred_seed_skycoord_col} column.",
                              flush=True)
                    else:
                        print(f"No seeds within {_match_radius_arcsec:.2f}\" of any "
                              f"per-filter iter2 source; leaving seed positions as-is",
                              flush=True)
                else:
                    print(f"Per-filter iter2 catalog {_iter2_cat_path} has no "
                          f"'skycoord' column or is empty; skipping snap",
                          flush=True)
            else:
                print(f"No per-filter iter2 cat at {_iter2_cat_path}; "
                      f"using cross-band union positions unchanged",
                      flush=True)
        except Exception as _snap_exc:
            print(f"Per-filter iter2 snap failed ({_snap_exc!r}); "
                  f"continuing with cross-band union positions",
                  flush=True)

        # V12: inject per-filter iter2 sources as NEW seed rows.
        #
        # V11 snaps existing union rows to nearby iter2 positions but does
        # not fix two failure modes diagnosed 2026-06-04 on the
        # f480_toinvestgiate_june4 reg list:
        #
        # Mode A (stars 1, 2, 12, 13): iter2 detects a faint target ~0.21"
        # from a bright neighbor.  Union has 2+ SW fragments in the area;
        # V11 snaps each to its closest iter2 source, so faint AND bright
        # iter2 positions each get a snapped union row.  But the snapped
        # union rows inherit flux_init=1 (det_f480m=False, flux_f480m
        # masked).  The pre-fit deduplication (line ~4129, 1.0 * FWHM ~
        # 2.5 LW pix for iter3) keeps the BRIGHTER seed in each cluster;
        # in this case the bright iter2 source has nearby union rows with
        # detected_f480m=True / flux_f480m=5787 carried in -- they win,
        # the faint target's seed (flux_init=1) is dropped along with all
        # other "flux=1" near-coincident union fragments.  Empirically
        # verified: for star 1, V11 snapped u[14153] to iter2[5304] at
        # pix (344.21, 121.83) but basic iter3 has zero inits within 2
        # px of that position.
        #
        # Mode B (stars 7, 11, 15): the nearest union seed is 0.197-
        # 0.253" from its iter2 source - just outside V11's 0.19" snap
        # radius (3.0 * 0.063"/pix).  No snap fires, faint iter2 target
        # has no representative in iter3 seeds.
        #
        # Solution: append every per-filter iter2 source as a fresh seed
        # row, carrying its iter2 flux as flux_init.  Then dedup picks
        # the iter2 row (high flux) over nearby low-flux_init union
        # fragments.  Union rows with detected_f480m=True still have
        # their own flux populated and survive on their own merit.
        try:
            _iter2_cat_path = os.path.join(
                basepath, 'catalogs',
                f'{filtername.lower()}_merged_indivexp_merged_iter2_'
                f'daoiterative_iterative.fits',
            )
            if os.path.exists(_iter2_cat_path):
                _it2 = Table.read(_iter2_cat_path)
                if 'skycoord' in _it2.colnames and len(_it2) > 0:
                    _it2_sk = _it2['skycoord']
                    if not isinstance(_it2_sk, SkyCoord):
                        _it2_sk = SkyCoord(_it2_sk)
                    _it2_sk = _it2_sk.unmasked if hasattr(_it2_sk, 'unmasked') else _it2_sk

                    _n_iter2 = len(_it2)
                    _flux_col_lower = f'flux_{filtername.lower()}'
                    _it2_flux = (np.asarray(_it2['flux'], dtype=float)
                                 if 'flux' in _it2.colnames
                                 else np.ones(_n_iter2, dtype=float))

                    _injected = Table()
                    # Ensure injected rows carry iter2 flux as 'flux'
                    # (SeededFinder reads this as flux_init).  This is
                    # the load-bearing column for dedup brightest-wins.
                    _injected['flux'] = _it2_flux
                    _injected['flux_fit'] = _it2_flux
                    for _col in merged_seed_table.colnames:
                        if _col in ('flux', 'flux_fit'):
                            continue
                        _src = merged_seed_table[_col]
                        if isinstance(_src, SkyCoord):
                            _injected[_col] = SkyCoord(
                                ra=_it2_sk.ra, dec=_it2_sk.dec, frame='icrs')
                        elif _col == 'ra':
                            _injected[_col] = np.asarray(_it2_sk.ra.deg, dtype=float)
                        elif _col == 'dec':
                            _injected[_col] = np.asarray(_it2_sk.dec.deg, dtype=float)
                        elif _col == 'source_id_union':
                            _injected[_col] = np.ma.masked_array(
                                np.full(_n_iter2, -1, dtype=np.int64),
                                mask=np.ones(_n_iter2, dtype=bool))
                        elif _col == 'seed_filter_origin':
                            _injected[_col] = np.array(
                                [f'{filtername.upper()}_ITER2'] * _n_iter2)
                        elif _col == 'is_saturated':
                            _injected[_col] = np.zeros(_n_iter2, dtype=bool)
                        elif _col == 'n_filters':
                            _injected[_col] = np.ones(_n_iter2, dtype=np.int32)
                        elif _col == _flux_col_lower:
                            _injected[_col] = _it2_flux
                        elif _col == f'detected_{filtername.lower()}':
                            _injected[_col] = np.ones(_n_iter2, dtype=bool)
                        elif _col == 'flux_fit':
                            _injected[_col] = _it2_flux
                        elif _col.startswith('detected_'):
                            _injected[_col] = np.zeros(_n_iter2, dtype=bool)
                        else:
                            # Fill numeric columns with NaN, others with
                            # the column's default; use a masked array to
                            # preserve dtype across vstack.
                            _dt = _src.dtype if hasattr(_src, 'dtype') else None
                            if _dt is not None and np.issubdtype(_dt, np.floating):
                                _injected[_col] = np.full(_n_iter2, np.nan, dtype=_dt)
                            elif _dt is not None and np.issubdtype(_dt, np.integer):
                                _injected[_col] = np.ma.masked_array(
                                    np.zeros(_n_iter2, dtype=_dt),
                                    mask=np.ones(_n_iter2, dtype=bool))
                            elif _dt is not None and np.issubdtype(_dt, np.bool_):
                                _injected[_col] = np.zeros(_n_iter2, dtype=bool)
                            else:
                                _injected[_col] = np.ma.masked_array(
                                    np.zeros(_n_iter2, dtype=object),
                                    mask=np.ones(_n_iter2, dtype=bool))

                    from astropy.table import vstack as _vstack
                    merged_seed_table = _vstack(
                        [merged_seed_table, _injected],
                        join_type='outer', metadata_conflicts='silent')
                    seed_catalog = merged_seed_table
                    print(f"Injected {_n_iter2} per-filter iter2 sources as "
                          f"new seed rows (flux_init carried from iter2 "
                          f"'flux' column); dedup will collapse against "
                          f"nearby union fragments. Seed table now "
                          f"{len(merged_seed_table)} rows.", flush=True)
        except Exception as _inject_exc:
            print(f"Per-filter iter2 seed injection failed ({_inject_exc!r}); "
                  f"continuing with snap-only union catalog",
                  flush=True)

        # Also snap satstar_table positions to per-filter iter2 where matched.
        # The force-union satstar table (project_force_union_satstar 2026-05-18)
        # adds entries at cross-frame union (skycoord_ref) positions for stars
        # saturated in OTHER filters' frames -- these land on this frame at
        # the SW astrometric position, which is several LW pixels off from
        # the true F480M position.  Pre-fit dedup then merges the snapped
        # union seed with the unsnapped satstar entry; the satstar entry
        # (which carries is_saturated=True and a valid flux) wins and the
        # snapped position is lost.  Detected 2026-06-03 on sickle F480M
        # region (266.57045,-28.80021): union row 13911 was correctly snapped
        # to pix (311.80,127.07), but iter3 source 55 init landed at
        # (310.26,126.62) -- the unsnapped union/satstar position.  Snapping
        # satstar_table's (x_fit,y_fit) to per-filter iter2 positions
        # closes this gap.
        try:
            if (satstar_table is not None and len(satstar_table) > 0
                    and 'x_fit' in satstar_table.colnames
                    and 'y_fit' in satstar_table.colnames):
                _iter2_cat_path = os.path.join(
                    basepath, 'catalogs',
                    f'{filtername.lower()}_merged_indivexp_merged_iter2_'
                    f'daoiterative_iterative.fits',
                )
                if os.path.exists(_iter2_cat_path):
                    _it2 = Table.read(_iter2_cat_path)
                    if 'skycoord' in _it2.colnames and len(_it2) > 0:
                        _it2_sk = _it2['skycoord']
                        if not isinstance(_it2_sk, SkyCoord):
                            _it2_sk = SkyCoord(_it2_sk)
                        _it2_sk = _it2_sk.unmasked if hasattr(_it2_sk, 'unmasked') else _it2_sk
                        _sat_x = np.asarray(satstar_table['x_fit'], dtype=float)
                        _sat_y = np.asarray(satstar_table['y_fit'], dtype=float)
                        _sat_finite = np.isfinite(_sat_x) & np.isfinite(_sat_y)
                        if np.any(_sat_finite):
                            _sat_sk = ww.pixel_to_world(_sat_x[_sat_finite],
                                                        _sat_y[_sat_finite])
                            _pixscale_arcsec = (ww.proj_plane_pixel_area()**0.5).to(u.arcsec).value if hasattr(ww, 'proj_plane_pixel_area') else 0.063
                            _match_radius_arcsec = max(0.15, 3.0 * _pixscale_arcsec)
                            _idx, _sep2d, _ = _sat_sk.match_to_catalog_sky(_it2_sk)
                            _good = _sep2d.arcsec < _match_radius_arcsec
                            if np.any(_good):
                                _it2_xy = ww.world_to_pixel(_it2_sk[_idx[_good]])
                                _new_x = _sat_x.copy()
                                _new_y = _sat_y.copy()
                                _finite_idx = np.where(_sat_finite)[0]
                                _move_idx = _finite_idx[_good]
                                _new_x[_move_idx] = np.asarray(_it2_xy[0], dtype=float)
                                _new_y[_move_idx] = np.asarray(_it2_xy[1], dtype=float)
                                satstar_table['x_fit'] = _new_x
                                satstar_table['y_fit'] = _new_y
                                if 'x_init' in satstar_table.colnames:
                                    _init_x = np.asarray(satstar_table['x_init'], dtype=float).copy()
                                    _init_y = np.asarray(satstar_table['y_init'], dtype=float).copy()
                                    _init_x[_move_idx] = np.asarray(_it2_xy[0], dtype=float)
                                    _init_y[_move_idx] = np.asarray(_it2_xy[1], dtype=float)
                                    satstar_table['x_init'] = _init_x
                                    satstar_table['y_init'] = _init_y
                                print(f"Snapped {int(np.sum(_good))} of "
                                      f"{len(satstar_table)} satstar_table "
                                      f"entries to per-filter iter2 positions "
                                      f"within {_match_radius_arcsec:.2f}\".",
                                      flush=True)
        except Exception as _snap_sat_exc:
            print(f"Per-filter iter2 satstar snap failed ({_snap_sat_exc!r}); "
                  f"continuing with original satstar positions",
                  flush=True)

        # Optional spatial chunking: split the seed catalog into N image-pixel
        # tiles and fit only the regular seeds whose pixel position lies in
        # this chunk's tile.  Each chunk runs as its own SLURM array job and
        # writes a _chunkXXofYY-tagged per-frame catalog; merge_catalogs.py
        # vstacks them back into one per-frame table.  Used to stay under
        # the 96 h walltime for brick LW iter3 (433 k seeds per frame).
        # satstar_table is NOT subset here -- it stays full-frame so the
        # postprocess-residual masking continues to see every satstar; the
        # merge-time dedup catches the resulting cross-chunk satstar
        # duplicates.
        _n_seed_chunks = int(getattr(options, 'n_seed_chunks', 1) or 1)
        _seed_chunk_index = int(getattr(options, 'seed_chunk_index', 0) or 0)
        if _n_seed_chunks > 1:
            merged_seed_table = _seed_table_chunk_subset(
                merged_seed_table, ww=ww, image_shape=data.shape,
                chunk_index=_seed_chunk_index,
                n_seed_chunks=_n_seed_chunks,
            )
            seed_catalog = merged_seed_table

        # Populate flux_fit on is_saturated union-catalog rows from the
        # per-filter flux column so dedup can compare them against current-
        # frame satstar entries.  Previously these rows were stripped en
        # bloc, which removed bright stars that are saturated in some
        # filters but not in THIS one (e.g. seed flagged saturated from
        # F187N but well within linear range in F480M).  Such stars then
        # had no satstar fit (no DQ-saturated pixels in this filter) AND
        # no daophot fit (stripped) — they remained fully un-subtracted
        # in the residual mosaic.  Diagnosed 2026-05-16 on Sickle F480M:
        # user reg position (266.56408,-28.80118) corresponded to
        # seed[11992] is_saturated=True flux_f480m=4.03e5; nearby NaN-
        # flux duplicate seeds (sep~0.3") were used instead, fits hit
        # xy_bounds=±0.5px and gave flag=48 with flux~1500.  Filling
        # flux_fit from per-filter flux lets _dedup_close_sources keep
        # the correct (brightest, on-target) seed.
        if satstar_table is not None:
            st = _as_table(seed_catalog)
            if 'is_saturated' in st.colnames:
                is_sat_mask = np.asarray(st['is_saturated'], dtype=bool)
                n_sat_in_union = int(np.sum(is_sat_mask))
                if n_sat_in_union > 0:
                    # Find the per-filter flux column for the current filter.
                    _flux_col = f'flux_{filtername.lower()}'
                    if _flux_col in st.colnames:
                        if 'flux_fit' not in st.colnames:
                            st['flux_fit'] = np.full(len(st), np.nan, dtype=float)
                        _f = np.asarray(st[_flux_col], dtype=float)
                        # Only fill where flux_fit is currently NaN AND
                        # per-filter flux is finite.
                        _need = is_sat_mask & np.isnan(np.asarray(st['flux_fit'], dtype=float)) & np.isfinite(_f)
                        if np.any(_need):
                            st['flux_fit'] = np.where(_need, _f, st['flux_fit'])
                            print(f"Filled flux_fit from {_flux_col} on "
                                  f"{int(np.sum(_need))} is_saturated union-catalog "
                                  f"seeds (so dedup can compare them against "
                                  f"current-frame satstar entries)",
                                  flush=True)
                    seed_catalog = st
                    merged_seed_table = _as_table(seed_catalog)
                    print(f"Kept {n_sat_in_union} is_saturated=True rows in union seed catalog "
                          f"(flux_fit populated; dedup handles overlap with current-frame satstar)",
                          flush=True)

        seed_catalog = _combine_seed_and_satstars(seed_catalog, satstar_table)
        seed_after_sat_table = _as_table(seed_catalog)
        sat_seed_count = int(np.sum(np.asarray(seed_after_sat_table['is_saturated'], dtype=bool)))
        nonsat_seed_count = int(len(seed_after_sat_table) - sat_seed_count)
        detection_image = nan_replaced_data
        assert not np.any(np.isnan(nan_replaced_data))
        if postprocess_residuals:
            detection_image = postprocess_residual_image(
                nan_replaced_data,
                fwhm_pix,
                negative_threshold=residual_negative_threshold,
                satstar_table=satstar_table,
            )
        if postprocess_residuals:
            extra_noise_map = compute_local_noise_map(detection_image, smooth_sigma_pix=3.0)
            finite_extra_noise = np.isfinite(extra_noise_map) & (extra_noise_map > 0)
            if not np.any(finite_extra_noise):
                raise ValueError('Postprocessed local noise map has no positive finite values')
            extra_noise_floor = float(np.nanmin(extra_noise_map[finite_extra_noise]))
            extra_finder = DAOStarFinder(threshold=extra_noise_floor,
                                         fwhm=fwhm_pix, roundhi=iter2_roundhi, roundlo=iter2_roundlo,
                                         sharplo=iter2_sharplo, sharphi=iter2_sharphi)
            extra_detections = extra_finder(detection_image, mask=mask)
            extra_noise_for_snr = extra_noise_map
            print(f'Postprocessed DAO local-noise threshold: {extra_noise_floor}', flush=True)
        else:
            extra_detections = daofind_tuned(detection_image, mask=mask)
            extra_noise_for_snr = local_noise_map

        if extra_detections is None:
            extra_detections = Table()
        extra_detections, extra_snr_stats = annotate_and_filter_by_local_snr(
            extra_detections,
            extra_noise_for_snr,
            snr_threshold=iter2_local_snr_threshold,
        )
        print(
            'Extra DAO detections local-SNR filter: '
            f'in={extra_snr_stats["input_count"]} '
            f'kept={extra_snr_stats["kept_count"]} '
            f'dropped={extra_snr_stats["dropped_count"]}',
            flush=True,
        )
        seed_catalog, seed_aug_stats = _augment_seed_catalog_with_detections_sky(
            seed_catalog,
            extra_detections,
            ww=ww,
            match_radius_pix=max(1.0, 0.5 * fwhm_pix),
            preferred_seed_skycoord_col=preferred_seed_skycoord_col,
            return_stats=True,
        )
        print(
            'Seed composition: '
            f'merged_seed_rows={len(merged_seed_table)} '
            f'sat_seed_rows={sat_seed_count} '
            f'nonsat_seed_rows={nonsat_seed_count} '
            f'dao_detect_total={seed_aug_stats["detection_input"]} '
            f'dao_detect_finite_xy={seed_aug_stats["detection_finite_xy"]} '
            f'dao_added={seed_aug_stats["detection_added"]} '
            f'dao_rejected_duplicates={seed_aug_stats["detection_rejected_match"]} '
            f'seed_rows_final={len(_as_table(seed_catalog))}'
        )
        _mem_report("before SeededFinder")
        finstars = SeededFinder(seed_catalog, ww=ww,
                                preferred_skycoord_col=preferred_seed_skycoord_col)(nan_replaced_data, mask=mask)
        _mem_report("after SeededFinder call", deep=True)
        seeded_init_params = Table()
        seeded_init_params['x_init'] = np.asarray(finstars['x_init'], dtype=float)
        seeded_init_params['y_init'] = np.asarray(finstars['y_init'], dtype=float)
        seeded_init_params['flux_init'] = np.asarray(finstars['flux_init'], dtype=float)
        # Carry is_saturated through so dedup can preferentially keep
        # known bright/saturated seeds over nearby NaN-flux duplicates.
        if 'is_saturated' in finstars.colnames:
            seeded_init_params['is_saturated'] = np.asarray(
                finstars['is_saturated'], dtype=bool)

        # Deduplicate seeds: remove entries within 0.5 FWHM of a brighter seed.
        # Merged catalogs can contain sub-pixel duplicate entries from multiple
        # per-exposure fits landing at slightly different positions for the same
        # star.  Two seeds at the same position each receive the full star flux,
        # doubling the model and producing large negative residuals.  No
        # quality metric exists at the seed stage, so the brightest init flux
        # wins any flux-disagreement tie.
        #
        # For iter3 the union seed catalog contains sources from all filters,
        # including SW-only detections that are unresolved at LW wavelengths.
        # Fitting many seeds within one PSF FWHM of each other produces an
        # ill-conditioned LSQ system with wildly oscillating positive/negative
        # fluxes that contaminate the residual image.  Tighten the dedup to
        # 1.0×FWHM for iter3 to collapse seeds within one resolution element
        # to a single fit position before entering the solver.
        min_sep_pix = 1.0 * fwhm_pix if is_iter3 else 0.5 * fwhm_pix
        n_before = len(seeded_init_params)
        if n_before > 1:
            keep, n_disagree = _dedup_close_sources(
                xy=np.column_stack([
                    np.asarray(seeded_init_params['x_init'], dtype=float),
                    np.asarray(seeded_init_params['y_init'], dtype=float),
                ]),
                flux=np.asarray(seeded_init_params['flux_init'], dtype=float),
                min_sep_pix=min_sep_pix,
                quality=None,
                is_saturated=(np.asarray(seeded_init_params['is_saturated'], dtype=bool)
                              if 'is_saturated' in seeded_init_params.colnames
                              else None),
            )
            n_removed = int(n_before - np.sum(keep))
            if n_removed > 0:
                seeded_init_params = seeded_init_params[keep]
                print(f"Pre-fit deduplication removed {n_removed} seeds within "
                      f"{min_sep_pix:.2f} pix ({n_before} -> {len(seeded_init_params)}); "
                      f"{n_disagree} clusters had disagreeing init fluxes",
                      flush=True)
        _mem_report("after seed dedup")

        finding_label = 'seeded'
    else:
        finstars = daofind_tuned(nan_replaced_data,
                                 mask=mask)
        if finstars is None:
            finstars = Table()
        finding_label = 'daofind'

    print(f"Found {len(finstars)} with daofind_tuned", flush=True)
    # for diagnostic plotting convenience
    # photutils >=3.0 emits x_centroid/y_centroid; 2.x emits xcentroid/ycentroid.
    if 'xcentroid' in finstars.colnames:
        finstars['x'] = finstars['xcentroid']
        finstars['y'] = finstars['ycentroid']
    elif 'x_centroid' in finstars.colnames:
        finstars['x'] = finstars['x_centroid']
        finstars['y'] = finstars['y_centroid']
    finstars['skycoord'] = ww.pixel_to_world(finstars['x'], finstars['y'])

    # All basepath-based INPUT reads (resbg, seed, satstar, union catalogs)
    # are done by this point; redirect every subsequent OUTPUT to the cutout
    # directory so cutout products never overwrite full-frame photometry.
    if _cutout_active:
        os.makedirs(f'{out_basepath}/{filtername}/pipeline', exist_ok=True)
        basepath = out_basepath

    result = save_photutils_results(finstars, ww, filename,
                                    im1=im1, detector=detector,
                                    basepath=basepath,
                                    filtername=filtername, module=module,
                                    desat=desat, bgsub=bgsub,
                                    blur="",
                                    exposure_=exposure_,
                                    visitid_=visitid_, vgroupid_=vgroupid_,
                                    basic_or_iterative=finding_label,
                                    options=options,
                                    epsf_="",
                                    fpsf="",
                                    group=group,
                                    psf=None,
                                    background_map=background_map,
                                    iteration_label=iteration_label)

    stars = finstars # because I'm copy-pasting code...

    # Set up visualization
    reg = regions.RectangleSkyRegion(center=cen, width=1.5*u.arcmin, height=1.5*u.arcmin)
    preg = reg.to_pixel(ww)
    #mask = preg.to_mask()
    #cutout = mask.cutout(im1[1].data)
    #err = mask.cutout(im1[2].data)
    # Zoom regions live in a shared regions_/ directory across fields (brick,
    # sgrb2, cloudc, sickle).  Two guards keep cross-field regions out:
    # (1) sky-separation gate — if the region center sits more than
    #     max_field_sep from this detector's pointing, it belongs to a
    #     different field and projects to extreme pixel coords.
    # (2) bbox sanity check — `to_mask()` allocates a (ny,nx) float64 array
    #     sized to the bounding box (regions/shapes/rectangle.py:165).  A
    #     half-degree-misplaced rectangle yields a TB-scale allocation that
    #     drove sgrb2 nrca1 photometry to 564 GB peak.  Refuse anything
    #     larger than 4× the detector.
    region_list = [y for x in glob.glob(str(REGIONS_DIR / '*zoom*.reg')) for y in
                   regions.Regions.read(x)]
    ny, nx = data.shape
    data_center_sky = ww.pixel_to_world(nx / 2.0, ny / 2.0)
    max_field_sep = 6 * u.arcmin
    zoomcut_list = {}
    for _reg in region_list:
        reg_center = getattr(_reg, 'center', None)
        if isinstance(reg_center, SkyCoord):
            if reg_center.separation(data_center_sky) > max_field_sep:
                continue
        _pix = _reg.to_pixel(ww)
        bb = _pix.bounding_box
        if bb.ixmax < 0 or bb.iymax < 0 or bb.ixmin >= nx or bb.iymin >= ny:
            continue
        bb_w = bb.ixmax - bb.ixmin
        bb_h = bb.iymax - bb.iymin
        if bb_w * bb_h > 4 * nx * ny:
            print(f"Skipping oversized zoom region '{_reg.meta.get('text', '?')}': "
                  f"bbox {bb_w}x{bb_h} exceeds 4x detector ({nx}x{ny})",
                  flush=True)
            continue
        _slc_tuple = _pix.to_mask().get_overlap_slices(data.shape)
        if _slc_tuple is None or _slc_tuple[0] is None:
            continue
        _slc = _slc_tuple[0]
        if (_slc[0].start > 0 and _slc[1].start > 0
                and _slc[0].stop < data.shape[0] and _slc[1].stop < data.shape[1]):
            zoomcut_list[_reg.meta['text']] = _slc

    _mem_report("after region zoomcut loop", deep=True)
    zoomcut = slice(128, 256), slice(128, 256)
    modsky = data*0 # no model for daofind
    nullslice = (slice(None), slice(None))

    _mem_report("before daofind catalog_zoom_diagnostic block")
    try:
        catalog_zoom_diagnostic(data, modsky, nullslice, stars)
        pl.suptitle(f"daofind Catalog Diagnostics zoomed {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}")
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}_catalog_diagnostics_daofind.png',
                bbox_inches='tight')

        catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
        pl.suptitle(f"daofind Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}")
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}_catalog_diagnostics_zoom_daofind.png',
                bbox_inches='tight')

        for name, zoomcut in zoomcut_list.items():
            catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
            pl.suptitle(f"daofind Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub} zoom {name}")
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}_catalog_diagnostics_zoom{name.replace(" ","_")}_daofind.png',
                    bbox_inches='tight')
    except Exception as ex:
        print(f'FAILURE to produce catalog zoom diagnostics for module {module} and filter {filtername} for basic daofinder: {ex}')
    _mem_report("after daofind catalog_zoom_diagnostic block", deep=True)

    if not options.nocrowdsource:

        t0 = time.time()

        if False: # why do the unweighted version?
            print()
            print("starting crowdsource unweighted", flush=True)
            results_unweighted = fit_im(nan_replaced_data, psf_model,
                                        weight=np.ones_like(data)*np.nanmedian(weight)*(~mask),
                                        # psfderiv=np.gradient(-psf_initial[0].data),
                                        dq=dq,
                                        nskyx=0, nskyy=0, refit_psf=False, verbose=True,
                                        **crowdsource_default_kwargs,
                                        )
            print(f"Done with unweighted crowdsource. dt={time.time() - t0}")
            stars, modsky, skymsky, psf = results_unweighted
            stars = save_crowdsource_results(results_unweighted, ww, filename,
                                             im1=im1, detector=detector,
                                             basepath=basepath,
                                             filtername=filtername, module=module,
                                             desat=desat, bgsub=bgsub,
                                             blur=options.blur,
                                             exposure_=exposure_,
                                             visitid_=visitid_,
                                             vgroupid_=vgroupid_,
                                             options=options,
                                             suffix="unweighted", psf=None,
                                             iteration_label=iteration_label)

            zoomcut = slice(128, 256), slice(128, 256)

            try:
                catalog_zoom_diagnostic(data, modsky, nullslice, stars)
                pl.suptitle(f"Crowdsource nsky=0 unweighted Catalog Diagnostics zoomed {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_}")
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_}_catalog_diagnostics_unweighted.png',
                        bbox_inches='tight')

                catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                pl.suptitle(f"Crowdsource nsky=0 unweighted Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_}")
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_}_catalog_diagnostics_zoom_unweighted.png',
                        bbox_inches='tight')
                for name, zoomcut in zoomcut_list.items():
                    catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                    pl.suptitle(f"Crowdsource nsky=0 Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_} zoom {name}")
                    pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{blur_}_catalog_diagnostics_zoom{name.replace(" ","_")}_unweighted.png',
                            bbox_inches='tight')
            except Exception as ex:
                print(f'FAILURE to produce catalog zoom diagnostics for module {module} and filter {filtername} for unweighted crowdsource: {ex}')
                exc_tb = sys.exc_info()[2]
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                print(f"Exception {ex} was in {fname} line {exc_tb.tb_lineno}")

            fig = pl.figure(0, figsize=(10,10))
            fig.clf()
            ax = fig.gca()
            im = ax.imshow(weight, norm=simple_norm(weight, stretch='log'))
            pl.colorbar(mappable=im)
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}_weights.png',
                    bbox_inches='tight')

        #for refit_psf, fpsf in zip((False, True), ('', '_fitpsf',)):
        for refit_psf, fpsf in zip((False, ), ('', )):
            for nsky in (0, ): #1, ):
                t0 = time.time()
                print()
                print(f"Running crowdsource fit_im with weights & nskyx=nskyy={nsky} & fpsf={fpsf} & blur={blur_}")
                print(f"data.shape={data.shape} weight_shape={weight.shape}", flush=True)
                _mem_report("before crowdsource fit_im")
                results = fit_im(nan_replaced_data, psf_model, weight=weight * (~mask),
                                 nskyx=nsky, nskyy=nsky, refit_psf=refit_psf, verbose=True,
                                 dq=dq,
                                 **crowdsource_default_kwargs
                                 )
                _mem_report("after crowdsource fit_im", deep=True)
                print(f"Done with weighted, refit={fpsf}, nsky={nsky} crowdsource. dt={time.time() - t0}")
                stars, modsky, skymsky, psf = results
                stars = save_crowdsource_results(results, ww, filename,
                                                 im1=im1, detector=detector,
                                                 basepath=basepath,
                                                 filtername=filtername,
                                                 module=module, desat=desat,
                                                 bgsub=bgsub, fpsf=fpsf,
                                                 blur=options.blur,
                                                 exposure_=exposure_,
                                                 visitid_=visitid_,
                                                 vgroupid_=vgroupid_,
                                                 psf=psf if refit_psf else None,
                                                 options=options,
                                                 suffix=f"nsky{nsky}",
                                                 iteration_label=iteration_label)
                _mem_report("after save_crowdsource_results")

                zoomcut = slice(128, 256), slice(128, 256)

                try:
                    catalog_zoom_diagnostic(data, modsky, nullslice, stars)
                    pl.suptitle(f"Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_} nsky={nsky} weighted")
                    pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}_nsky{nsky}_weighted_catalog_diagnostics.png',
                            bbox_inches='tight')

                    catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                    pl.suptitle(f"Catalog Diagnostics zoomed {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_} nsky={nsky} weighted")
                    pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}_nsky{nsky}_weighted_catalog_diagnostics_zoom.png',
                            bbox_inches='tight')

                    for name, zoomcut in zoomcut_list.items():
                        catalog_zoom_diagnostic(data, modsky, zoomcut, stars)
                        pl.suptitle(f"Crowdsource nsky={nsky} weighted Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_} zoom {name}")
                        pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{fpsf}{blur_}_nsky{nsky}_weighted_catalog_diagnostics_zoom{name.replace(" ","_")}.png',
                                bbox_inches='tight')
                except Exception as ex:
                    print(f'FAILURE to produce catalog zoom diagnostics for module {module} and filter {filtername} for crowdsource nsky={nsky} refitpsf={refit_psf} blur={options.blur}: {ex}')
                    exc_tb = sys.exc_info()[2]
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    print(f"Exception {ex} was in {fname} line {exc_tb.tb_lineno}")
                _mem_report("after crowdsource diag plotting", deep=True)

    if options.daophot:
        t0 = time.time()
        print("Starting basic PSF photometry", flush=True)
        _mem_report("before phot_basic setup")

        basic_finder = None if seeded_init_params is not None else daofind_tuned
        _phot_basic_extra = {}
        if iter3_xy_bounds_pix is not None:
            _phot_basic_extra['xy_bounds'] = (iter3_xy_bounds_pix,
                                              iter3_xy_bounds_pix)

        _parallel_workers = int(getattr(options, 'parallel_workers', 1) or 1)
        if _parallel_workers > 1:
            # EXPERIMENTAL parallel path.  Run the finder serially (cheap
            # relative to fitting), then chunk-parallel the fit.  See
            # _parallel_psfphotometry / _FakePhot for details.  Off
            # unless --parallel-workers > 1.
            if seeded_init_params is not None:
                _basic_init = seeded_init_params
            else:
                print("Running basic finder (serial, pre-chunking)", flush=True)
                _basic_sources = daofind_tuned(nan_replaced_data, mask=mask)
                _basic_init = Table()
                _xcol = ('x_centroid' if _basic_sources is not None
                         and 'x_centroid' in _basic_sources.colnames
                         else 'xcentroid')
                _ycol = ('y_centroid' if _basic_sources is not None
                         and 'y_centroid' in _basic_sources.colnames
                         else 'ycentroid')
                if _basic_sources is None or len(_basic_sources) == 0:
                    _basic_init['x_init'] = np.zeros(0)
                    _basic_init['y_init'] = np.zeros(0)
                    _basic_init['flux_init'] = np.zeros(0)
                else:
                    _basic_init['x_init'] = _basic_sources[_xcol]
                    _basic_init['y_init'] = _basic_sources[_ycol]
                    _basic_init['flux_init'] = _basic_sources['flux']

            _phot_basic_kwargs = dict(
                psf_model=dao_psf_model,
                fitter=LevMarLSQFitter(),
                fit_shape=(5, 5),
                aperture_radius=aperture_radius_pix,
                progress_bar=False,
                grouper=grouper if options.group else None,
                finder=None,
            )
            _phot_basic_kwargs['localbkg_estimator'] = LocalBackground(
                localbkg_inner, localbkg_outer)
            _phot_basic_kwargs.update(_phot_basic_extra)
            _chunk_size = int(getattr(options, 'parallel_chunk_size', 100))
            print(f"About to do BASIC photometry (PARALLEL, "
                  f"n_workers={_parallel_workers}, chunk={_chunk_size}, "
                  f"n_sources={len(_basic_init)})....", flush=True)
            _mem_report("before phot_basic call")
            result, _ = _parallel_psfphotometry(
                nan_replaced_data,
                photometry_kwargs=_phot_basic_kwargs,
                init_params=_basic_init,
                error=np.where(bad, 1e10, err),
                mask=mask,
                n_workers=_parallel_workers,
                chunk_size=_chunk_size,
                group_min_separation=2 * fwhm_pix,
                return_model=False,
            )
            phot_basic = _FakePhot(results=result,
                                   psf_model=dao_psf_model,
                                   init_params=_basic_init)
        else:
            phot_basic = _make_psfphotometry(
                                       finder=basic_finder,
                                       # filter-scaled: inner clears aperture
                                       # (2*FWHM) plus first Airy sidelobe;
                                       # see header comment near aperture_radius_pix.
                                       localbkg_estimator=LocalBackground(localbkg_inner, localbkg_outer),
                                       grouper=grouper if options.group else None,
                                       psf_model=dao_psf_model,
                                       fitter=LevMarLSQFitter(),
                                       fit_shape=(5, 5),
                                       aperture_radius=aperture_radius_pix,
                                       progress_bar=True,
                                       **_phot_basic_extra,
                                      )

            print("About to do BASIC photometry....")
            _mem_report("before phot_basic call")
            if seeded_init_params is not None:
                result = phot_basic(nan_replaced_data, mask=mask, init_params=seeded_init_params, error=np.where(bad, 1e10, err))
            else:
                result = phot_basic(nan_replaced_data, mask=mask, error=np.where(bad, 1e10, err))
        print(f"Done with BASIC photometry. len(result)={len(result)}  dt={time.time() - t0}")
        _mem_report("after phot_basic")

        # Post-fit deduplication: the unseeded DAO finder can detect multiple
        # local maxima near a single bright star; each is fit independently
        # without a grouper, and they can converge to the same (x_fit, y_fit).
        # Summing those PSFs in make_model_image() produces 2x-4x overfits.
        # When a cluster of fits converges to the same position, the same
        # physical source should yield matching fluxes; if fluxes disagree
        # the fits reached different minima and we keep the one with the
        # best qfit (smallest chi-squared/pixel).  Filter the phot_basic
        # object's own state so the saved catalog and the rendered model
        # image are both built from the deduplicated set.
        # 2026-04-24: threshold loosened from 0.5 * fwhm_pix to
        # 1.5 * fwhm_pix.  The tighter threshold left seeded-duplicate
        # fits at 2.5-3.2 LW pix apart uncaught, which drove
        # progressively deeper negative residuals in iter2 (p0.1=-96)
        # and iter3 (p0.1=-172).
        # 2026-06-04 (V13): tighten to 1.0 px to preserve real
        # adjacent stars from V12 iter2 inject (e.g. sickle F480M
        # star 1 faint target 3.32 px from bright neighbor was being
        # qfit-merged into bright at 1.5*FWHM=3.86 px).  Risk: may
        # re-introduce 2026-04-24 deep negative residuals if drift-
        # together duplicates beyond 1.0 px exist; will be measured
        # on V13 mosaic.
        min_sep_pix = 1.0
        xfit_arr = np.asarray(result['x_fit'], dtype=float)
        yfit_arr = np.asarray(result['y_fit'], dtype=float)
        flux_arr = np.asarray(result['flux_fit'], dtype=float)
        qfit_arr = (np.asarray(result['qfit'], dtype=float)
                    if 'qfit' in result.colnames else None)
        keep_full, n_disagree = _dedup_close_sources(
            xy=np.column_stack([xfit_arr, yfit_arr]),
            flux=flux_arr,
            min_sep_pix=min_sep_pix,
            quality=qfit_arr,
        )
        n_removed = int(len(keep_full) - np.sum(keep_full))
        if n_removed > 0:
            tiebreak = "qfit" if qfit_arr is not None else "brightest"
            print(f"Post-fit deduplication: dropping {n_removed} drift-together "
                  f"fits within {min_sep_pix:.2f} pix "
                  f"({len(result)} -> {int(np.sum(keep_full))}); "
                  f"{n_disagree} clusters had disagreeing fitted fluxes "
                  f"(resolved by {tiebreak})", flush=True)
            # Filter the PSFPhotometry object's own state so the saved catalog
            # AND the model image (via make_model_image) are both built from
            # the deduplicated set.
            phot_basic.results = phot_basic.results[keep_full]
            if (phot_basic.init_params is not None
                    and len(phot_basic.init_params) == len(keep_full)):
                phot_basic.init_params = phot_basic.init_params[keep_full]
            # Invalidate the @lazyproperty cache so _model_image_params
            # regenerates from the filtered results on next access.
            phot_basic.__dict__.pop('_model_image_params', None)
            # Keep the local `result` name in sync with the filtered table.
            result = phot_basic.results

        # Saturation-proximity filter: regular phot_basic fits placed on
        # or right next to a saturated DQ pixel are unreliable (the central
        # data value is "stuck" while the wings drive the flux up), so drop
        # them before the catalog and the model image are written.  The
        # dedicated satstar catalog lives in a separate file and is not
        # touched.
        _dqarr_for_satfilter = im1['DQ'].data if 'DQ' in im1 else None
        _filter_near_saturation(phot_basic, _dqarr_for_satfilter,
                                max_sat_dist_pix=1.0,  # changed from 5.0 -> 1.0 (2026-05-28): the
                                # original intent is to drop fits whose CENTER
                                # is on a saturated pixel ("stuck-low"
                                # central data drives wing-only fit to bogus
                                # flux).  5.0 also killed legitimate stars
                                # within 5 px of a saturated neighbour --
                                # detected on Star B (17:46:14.175, 3500
                                # MJy/sr) sitting 2.24 px from the donut
                                # neighbour's nearest saturated edge pixel.
                                # 1.0 keeps the strict "center on or
                                # immediately adjacent to a sat pixel" guard
                                # while letting nearby real stars survive.
                                label='basic')
        _filter_satstar_artifacts(phot_basic, satstar_model_subtracted, err,
                                  sig_K=float(options.satstar_artifact_sigK),
                                  ratio_cut=float(options.satstar_artifact_ratio),
                                  label='basic')
        result = phot_basic.results

        result = save_photutils_results(result, ww, filename,
                                        im1=im1, detector=detector,
                                        basepath=basepath,
                                        filtername=filtername, module=module,
                                        desat=desat, bgsub=bgsub,
                                        blur=options.blur,
                                        exposure_=exposure_,
                                        visitid_=visitid_,
                                        vgroupid_=vgroupid_,
                                        basic_or_iterative='basic',
                                        options=options,
                                        epsf_=epsf_,
                                        group=group,
                                        psf=None,
                                        background_map=background_map,
                                        iteration_label=iteration_label)

        stars = result
        stars['x'] = stars['x_fit']
        stars['y'] = stars['y_fit']
        print("Creating BASIC residual image, using 21x21 patches")
        _mem_report("before basic model image")
        modsky = _make_model_image(phot_basic, data.shape, psf_shape=(21, 21), include_local_bkg=False)
        _mem_report("after basic model image")
        # The fitter saw ``data - satstar_model`` (when a satstar model
        # exists), so the saved residual must subtract the satstar model
        # too -- otherwise the bright-star wings reappear and dominate
        # the residual mosaic.  See pipeline-plumbing block above.
        # iter4resbgrefit: build the residual against the ORIGINAL (pre-bg-
        # subtraction) data so the saved residual = original - star models
        # (background retained).  All other iterations use ``data`` as before.
        _resid_base = (original_data if (is_resbg_refit and original_data is not None)
                       else data)
        data_for_residual = (_resid_base if satstar_model_subtracted is None
                             else _resid_base - satstar_model_subtracted)
        residual = data_for_residual - modsky
        print("Done creating BASIC residual image, using 21x21 patches")
        save_residual_datamodel(
            filename,
            f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}{iter_}_daophot_basic_residual.fits',
            residual,
        )
        save_residual_datamodel(
            filename,
            f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}{iter_}_daophot_basic_model.fits',
            modsky,
        )
        print("Saved BASIC residual image, now making diagnostics.")
        catalog_zoom_diagnostic(data_for_residual, modsky, nullslice, stars)
        pl.suptitle(f"daophot basic Catalog Diagnostics zoomed {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}")
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_catalog_diagnostics_daophot_basic.png',
                bbox_inches='tight')

        catalog_zoom_diagnostic(data_for_residual, modsky, zoomcut, stars)
        pl.suptitle(f"daophot basic Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}")
        pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_catalog_diagnostics_zoom_daophot_basic.png',
                bbox_inches='tight')

        for name, zoomcut in zoomcut_list.items():
            catalog_zoom_diagnostic(data_for_residual, modsky, zoomcut, stars)
            pl.suptitle(f"daophot basic Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group} zoom {name}")
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}__catalog_diagnostics_zoom_daophot_basic{name.replace(" ","_")}.png',
                    bbox_inches='tight')

        print(f"Done with diagnostics for BASIC photometry.  dt={time.time() - t0}")
        pl.close('all')

        if not options.basic_only:
            t0 = time.time()
            print("Iterative PSF photometry")
            if options.epsf:
                print("Building EPSF")
                epsf_builder = EPSFBuilder(oversampling=3, maxiters=10,
                                           smoothing_kernel='quadratic',
                                           progress_bar=True)

                epsfsel = ((finstars['peak'] > 200) &
                           (finstars['roundness1'] > -0.25) &
                           (finstars['roundness1'] < 0.25) &
                           (finstars['roundness2'] > -0.25) &
                           (finstars['roundness2'] < 0.25) &
                           (finstars['sharpness'] > 0.4) &
                           (finstars['sharpness'] < 0.8))

                print(f"Extracting {epsfsel.sum()} stars")
                stars = extract_stars(NDData(data=nan_replaced_data), finstars[epsfsel], size=35)

                for star in stars:
                    background = np.nanpercentile(star.data, 5)
                    star.data[:] -= background

                epsf, fitted_stars = epsf_builder(stars)
                epsf._data = epsf.data[2:-2, 2:-2]

                norm = simple_norm(epsf.data, 'log', percent=99.0)
                pl.figure(1).clf()
                pl.imshow(epsf.data, norm=norm, origin='lower', cmap='viridis')
                pl.colorbar()
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_daophot_epsf.png',
                           bbox_inches='tight')
                dao_psf_model = epsf

            _phot_iter_extra = {}
            if iter3_xy_bounds_pix is not None:
                _phot_iter_extra['xy_bounds'] = (iter3_xy_bounds_pix,
                                                 iter3_xy_bounds_pix)

            _parallel_workers = int(getattr(options, 'parallel_workers', 1) or 1)
            if _parallel_workers > 1:
                # EXPERIMENTAL parallel path.  Reimplements
                # IterativePSFPhotometry mode='new' with chunked
                # PSFPhotometry calls.  Returns a `_FakePhot` stand-in
                # whose `.results`, `.make_model_image()` etc. satisfy
                # the downstream dedup / sat-filter / model-rendering
                # code paths without depending on photutils internal
                # _fit_models state (which is per-worker and not
                # reconstructable).  Off unless --parallel-workers > 1.
                _phot_iter_kwargs = dict(
                    psf_model=dao_psf_model,
                    fitter=LevMarLSQFitter(),
                    fit_shape=(5, 5),
                    aperture_radius=aperture_radius_pix,
                    progress_bar=False,
                    grouper=grouper if options.group else None,
                )
                _phot_iter_kwargs['localbkg_estimator'] = LocalBackground(
                    localbkg_inner, localbkg_outer)
                _phot_iter_kwargs.update(_phot_iter_extra)
                _chunk_size = int(getattr(options, 'parallel_chunk_size', 100))
                print(f"About to do ITERATIVE photometry (PARALLEL, "
                      f"n_workers={_parallel_workers}, chunk={_chunk_size})....",
                      flush=True)
                _mem_report("before phot_iter call")
                result2 = _parallel_iterative_psfphotometry(
                    nan_replaced_data,
                    photometry_kwargs=_phot_iter_kwargs,
                    finder=daofind_tuned,
                    init_params=seeded_init_params,
                    error=np.where(bad, 1e10, err),
                    mask=mask,
                    maxiters=5,
                    sub_shape=(15, 15),
                    psf_model=dao_psf_model,
                    n_workers=_parallel_workers,
                    chunk_size=_chunk_size,
                    group_min_separation=2 * fwhm_pix,
                )
                phot_iter = _FakePhot(results=result2,
                                      psf_model=dao_psf_model,
                                      init_params=seeded_init_params)
            else:
                phot_iter = _make_iterative_psfphotometry(
                                                   finder=daofind_tuned,
                                                   localbkg_estimator=LocalBackground(localbkg_inner, localbkg_outer),
                                                   grouper=grouper if options.group else None,
                                                   psf_model=dao_psf_model,
                                                   fitter=LevMarLSQFitter(),
                                                   maxiters=5,
                                                   fit_shape=(5, 5),
                                                   sub_shape=(15, 15),
                                                   aperture_radius=aperture_radius_pix,
                                                   progress_bar=True,
                                                   **_phot_iter_extra,
                                                  )

                print("About to do ITERATIVE photometry....")
                _mem_report("before phot_iter call")
                if seeded_init_params is not None:
                    result2 = phot_iter(nan_replaced_data, mask=mask, init_params=seeded_init_params, error=np.where(bad, 1e10, err))
                else:
                    result2 = phot_iter(nan_replaced_data, mask=mask, error=np.where(bad, 1e10, err))
            print(f"Done with ITERATIVE photometry. len(result2)={len(result2)}  dt={time.time() - t0}")
            _mem_report("after phot_iter")

            # Apply the same post-fit deduplication to the iterative results.
            # IterativePSFPhotometry can produce drift-together duplicates across
            # iterations (a source detected in the residual of iter N matches the
            # same star fit in iter N-1).  Left in place, these duplicates
            # (i) double the flux in the rendered model image, and
            # (ii) trigger a photutils bug in make_model_image where the
            #      composite model parameters become array-valued and overlap_slices
            #      raises ValueError("The truth value of an array ... is ambiguous").
            # 2026-04-24: threshold loosened to 1.5 * fwhm_pix, same
            # rationale as in phot_basic above.
            # 2026-06-04 (V13): tighten to 1.0 px to preserve real
            # adjacent stars (see phot_basic note above).
            min_sep_pix = 1.0
            xfit_arr = np.asarray(result2['x_fit'], dtype=float)
            yfit_arr = np.asarray(result2['y_fit'], dtype=float)
            flux_arr = np.asarray(result2['flux_fit'], dtype=float)
            qfit_arr = (np.asarray(result2['qfit'], dtype=float)
                        if 'qfit' in result2.colnames else None)
            iter_keep, iter_n_disagree = _dedup_close_sources(
                xy=np.column_stack([xfit_arr, yfit_arr]),
                flux=flux_arr,
                min_sep_pix=min_sep_pix,
                quality=qfit_arr,
            )
            iter_n_removed = int(len(iter_keep) - np.sum(iter_keep))
            if iter_n_removed > 0:
                iter_tiebreak = "qfit" if qfit_arr is not None else "brightest"
                print(f"Post-fit deduplication (iterative): dropping {iter_n_removed} "
                      f"drift-together fits within {min_sep_pix:.2f} pix "
                      f"({len(result2)} -> {int(np.sum(iter_keep))}); "
                      f"{iter_n_disagree} clusters had disagreeing fitted fluxes "
                      f"(resolved by {iter_tiebreak})", flush=True)
                phot_iter.results = phot_iter.results[iter_keep]
                # IterativePSFPhotometry has no init_params attribute of its
                # own, but its internal PSFPhotometry (self._psfphot) does.
                inner_phot = getattr(phot_iter, '_psfphot', None)
                if (inner_phot is not None
                        and inner_phot.init_params is not None
                        and len(inner_phot.init_params) == len(iter_keep)):
                    inner_phot.init_params = inner_phot.init_params[iter_keep]
                phot_iter.__dict__.pop('_model_image_params', None)
                result2 = phot_iter.results

            # photutils.datasets.images.make_model_image uses a per-row
            # 'model_shape' column in the params table when present; if that
            # column has been populated with array-valued entries during the
            # iterative fit it triggers the ndarray-vs-tuple comparison inside
            # astropy.nddata.utils.overlap_slices.  Drop that column so the
            # caller's psf_shape=(21, 21) argument governs stamp size for all
            # sources uniformly.
            if 'model_shape' in phot_iter.results.colnames:
                phot_iter.results.remove_column('model_shape')
                phot_iter.__dict__.pop('_model_image_params', None)

            # Saturation-proximity filter for the iterative path -- same
            # rationale as the basic path: drop fits placed within
            # max_sat_dist_pix of any saturated DQ pixel.  Iterative
            # photometry is more aggressive and produces more of these
            # spurious near-saturation fits than basic, so the filter has
            # bigger impact here.
            _dqarr_for_satfilter_iter = im1['DQ'].data if 'DQ' in im1 else None
            _filter_near_saturation(phot_iter, _dqarr_for_satfilter_iter,
                                    max_sat_dist_pix=1.0,  # changed from 5.0 -> 1.0 (2026-05-28): the
                                # original intent is to drop fits whose CENTER
                                # is on a saturated pixel ("stuck-low"
                                # central data drives wing-only fit to bogus
                                # flux).  5.0 also killed legitimate stars
                                # within 5 px of a saturated neighbour --
                                # detected on Star B (17:46:14.175, 3500
                                # MJy/sr) sitting 2.24 px from the donut
                                # neighbour's nearest saturated edge pixel.
                                # 1.0 keeps the strict "center on or
                                # immediately adjacent to a sat pixel" guard
                                # while letting nearby real stars survive.
                                    label='iterative')
            _filter_satstar_artifacts(phot_iter, satstar_model_subtracted, err,
                                      sig_K=float(options.satstar_artifact_sigK),
                                      ratio_cut=float(options.satstar_artifact_ratio),
                                      label='iterative')
            result2 = phot_iter.results

            result2 = save_photutils_results(result2, ww, filename,
                                             im1=im1, detector=detector,
                                             basepath=basepath,
                                             filtername=filtername, module=module,
                                             desat=desat, bgsub=bgsub,
                                             blur=options.blur,
                                             exposure_=exposure_,
                                             visitid_=visitid_,
                                             vgroupid_=vgroupid_,
                                             basic_or_iterative='iterative',
                                             options=options,
                                             epsf_=epsf_,
                                             group=group,
                                             psf=None,
                                             background_map=background_map,
                                             iteration_label=iteration_label)

            stars = result2
            stars['x'] = stars['x_fit']
            stars['y'] = stars['y_fit']

            print("Creating iterative residual")
            _mem_report("before iter model image")
            modsky = _make_model_image(phot_iter, data.shape, psf_shape=(21, 21), include_local_bkg=False)
            _mem_report("after iter model image")
            # iter4resbgrefit: residual against the ORIGINAL (pre-bg-subtraction)
            # data; all other iterations use ``data`` (see basic block above).
            _resid_base = (original_data if (is_resbg_refit and original_data is not None)
                           else data)
            data_for_residual = (_resid_base if satstar_model_subtracted is None
                                 else _resid_base - satstar_model_subtracted)
            residual = data_for_residual - modsky
            print("finished iterative residual")
            save_residual_datamodel(
                filename,
                f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}{iter_}_daophot_iterative_residual.fits',
                residual,
            )
            save_residual_datamodel(
                filename,
                f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}{iter_}_daophot_iterative_model.fits',
                modsky,
            )
            print("Saved iterative residual")
            catalog_zoom_diagnostic(data_for_residual, modsky, nullslice, stars)
            pl.suptitle(f"daophot iterative Catalog Diagnostics zoomed {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}")
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_catalog_diagnostics_daophot_iterative.png',
                    bbox_inches='tight')

            catalog_zoom_diagnostic(data_for_residual, modsky, zoomcut, stars)
            pl.suptitle(f"daophot iterative Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}")
            pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}_catalog_diagnostics_zoom_daophot_iterative.png',
                    bbox_inches='tight')

            for name, zoomcut in zoomcut_list.items():
                catalog_zoom_diagnostic(data_for_residual, modsky, zoomcut, stars)
                pl.suptitle(f"daophot iterative Catalog Diagnostics {filtername} {module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group} zoom {name}")
                pl.savefig(f'{basepath}/{filtername}/pipeline/jw0{proposal_id}-o{field}_t001_{inst_token}_{pupil}-{filtername.lower()}-{module}{visitid_}{vgroupid_}{exposure_}{desat}{bgsub}{epsf_}{blur_}{group}__catalog_diagnostics_zoom_daophot_iterative{name.replace(" ","_")}.png',
                        bbox_inches='tight')

            print(f"Done with diagnostics for ITERATIVE photometry.  dt={time.time() - t0}")
            pl.close('all')
        else:
            print("Skipping ITERATIVE photometry because --basic-only was requested")


if __name__ == "__main__":
    main()
