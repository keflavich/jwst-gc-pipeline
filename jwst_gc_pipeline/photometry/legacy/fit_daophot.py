"""LEGACY daophot / photutils per-exposure fitting machinery -- BENCHMARKS ONLY.

Split out of ``legacy/photometry_step.py`` (which now orchestrates the legacy
per-exposure step but no longer DEFINES the fitting backends).  This module
holds the daophot machinery: the chunk-parallel PSFPhotometry workers, the
O(N log N) KDTree grouping, the ``_FakePhot`` stand-in, the chunked
reimplementation of IterativePSFPhotometry(mode='new'), the source grouper,
the saturated-star model subtraction, and the first-pass DAOStarFinder.

FROZEN: superseded by the manual-iteration pipeline (``cataloging``); kept for
benchmark reproduction.  Reuses the shared helpers/constants/third-party
imports from ``catalog_long`` via the namespace copy below, exactly as
``photometry_step`` does.
"""
import jwst_gc_pipeline.photometry.catalog_long as _host

# Reproduce the exact module namespace the relocated code had while it lived
# in catalog_long (shared helpers + that module's imports), so every
# bare-name reference below resolves unchanged.
globals().update({_k: _v for _k, _v in vars(_host).items()
                  if not _k.startswith('__')})


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


def _make_grouper(options, fwhm_pix):
    """Build the source grouper (2*FWHM linking) honoring --max-group-size.

    Factored out of ``do_photometry_step``.  ``resolve_max_group_size`` rejects
    the ambiguous 0 and maps 'unlimited' -> None (no cap -> plain SourceGrouper);
    a positive cap -> CappedSourceGrouper.
    """
    max_group_size = resolve_max_group_size(
        getattr(options, 'max_group_size', 'unlimited'))
    if max_group_size is not None:
        return CappedSourceGrouper(2 * fwhm_pix, max_size=max_group_size)
    print("max_group_size=unlimited: SourceGrouper has no group-size cap.",
          flush=True)
    return SourceGrouper(2 * fwhm_pix)


def _subtract_satstar_model(nan_replaced_data, satstar_model_image, saturated_mask):
    """Replace saturated-DQ pixels with the satstar model, then subtract it.

    Factored out of ``do_photometry_step`` (block K).  At SATURATED-DQ pixels
    the data is REPLACED by the model before subtracting, forcing residual=0
    there: the JWST ramp-fitter retains large numeric values at saturated pixels
    (group-1 fits, often >> 1e3 MJy/sr) that interpolate_replace_nans does not
    touch, and leaving them produced huge positive residuals (sickle F480M:
    7078 MJy/sr at a worst star, baseline 682; 2026-06-01).  Non-finite model
    pixels are treated as 0.  ``saturated_mask`` may be None (no DQ).

    Returns ``(new_data, finite_model)``.
    """
    finite_model = np.where(np.isfinite(satstar_model_image),
                            satstar_model_image, 0.0)
    if saturated_mask is not None:
        nan_replaced_data = np.where(saturated_mask, finite_model,
                                     nan_replaced_data)
    return nan_replaced_data - finite_model, finite_model


def _first_pass_daofinder(data, err, nsigma, fwhm_pix, roundlo, roundhi):
    """First-pass (iter1) DAOStarFinder + its threshold.

    Factored out of ``do_photometry_step`` (block J else-branch).  Threshold is
    ``nsigma * min(median(err), mad_std(data))`` -- the sigma-clipped mad_std of
    the data guards against err being over-estimated on extended-emission frames
    (Sickle F470N had ~3x too high err).  Returns ``(finder, threshold)``.
    """
    filtered_errest = np.nanmedian(err)
    print(f'Error estimate for DAO from median(err): {filtered_errest}', flush=True)
    # sigma_clipped stats get _much_ lower uncertainty for frames dominated by extended emission (maybe?).  At least, Sickle F470N had 3x too high error
    mean, med, std = stats.sigma_clipped_stats(data, stdfunc='mad_std')
    print(f'Error estimate for DAO from stats.: std={std}', flush=True)
    filtered_errest = min([filtered_errest, std])
    threshold = nsigma * filtered_errest
    finder = DAOStarFinder(threshold=threshold, fwhm=fwhm_pix,
                           roundhi=roundhi, roundlo=roundlo,
                           sharplo=0.30, sharphi=1.40)
    return finder, threshold
