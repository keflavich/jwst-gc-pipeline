r"""Astrometric diagnostic figures.

Two questions, one figure each.

**Internal** (:func:`internal_astrometry`) -- how repeatable is a position?
Each per-filter catalogue carries ``std_ra``/``std_dec``, the scatter of the
per-exposure centroids that were averaged into the merged position, and
``nmatch``, how many exposures contributed.  Their bright-end floor is the
single-epoch centroiding precision of the field, and the way that floor rises
towards faint magnitudes is the photon-noise term.  The cross-band table adds
the *inter-filter* consistency: the same star measured in two filters is an
independent realisation of the whole chain (different detector pixels,
different distortion solution, different PSF), so the median inter-filter
separation bounds the systematic floor from below in a way a single filter
cannot.

**Absolute** (:func:`absolute_astrometry`) -- where is the field on the sky?
Measured against the registered reference catalogue with the offset-histogram
estimator and mapped per tile, never as a nearest-neighbour median against a
dense catalogue.  ``CLAUDE.md`` states the rule and
``measure_offsets.assert_sparse_reference_for_nn_median`` enforces it; the
short version is that when the true shift exceeds the reference's
nearest-neighbour spacing, nearest-neighbour pairing matches the *wrong* star
and the median collapses towards zero, fabricating agreement.  A single
whole-field number is also insufficient on its own -- a field can read ~0 in
bulk while half of it is displaced -- so the bulk tie is always accompanied
by the per-tile map.
"""

import warnings

import numpy as np
import astropy.units as u

from jwst_gc_pipeline.diagnostics import loaders, style
from jwst_gc_pipeline.diagnostics.figures import FigureResult, save
from jwst_gc_pipeline.photometry.astrometry_offsets import (
    KDTreeReference, measure_offset, measure_offset_grid, WINDOW_EDGE_FRACTION)

# Tiles across the field for the per-tile offset map.  12x12 is the floor the
# release checklist asks for: the brick-1182 F200W seam was a ~90 mas residual
# confined to one strip that a 4x4 grid diluted away.
GRID_N = 12

# Sources fed to the per-tile tie.  This is a runtime ceiling, not a
# statistical one, and it is set high on purpose: the histogram peak's
# contrast grows with the number of pairs, and a tile with too few pairs does
# not fail loudly -- it widens its search window and returns an offset that is
# a property of the window.  Thinning arches from 241k to 60k sources took the
# per-tile measurement from 125 usable tiles to 43, so the cheap-looking
# saving buys a mostly-empty map.
MAX_TIE_SOURCES = 250000

# Margin (arcsec) kept around the field when cropping the reference catalogue.
# Wide enough to contain any offset the sweep could legitimately find, narrow
# enough that a sparse tile cannot latch onto a peak from far outside the field.
REFERENCE_MARGIN_ARCSEC = 120.0

MAS = 3.6e6  # degrees -> milliarcsec

# A per-exposure centroid scatter below this is not a measurement.  JWST
# centroiding does not repeat to ten microarcseconds; a value this small means
# the position was copied from a fixed seed (a forced or seeded refit) and
# never re-measured per exposure, so the "scatter" is the absence of one.
# Counting these as a precision floor would report a fabricated 0.00 mas.
MIN_MEASURABLE_SCATTER_MAS = 0.01


def internal_astrometry(inv, outdir, max_sources=400000):
    """Per-exposure repeatability and inter-filter positional agreement."""
    plt = style.use_style()
    filters = [f for f in inv.measured_filters]
    if not filters:
        return None

    per_filter = {}
    zps = loaders.photometric_zeropoints(inv.crossband_catalog, filters)
    for filt in filters:
        tbl = loaders.read_columns(
            inv.per_filter_catalogs[filt],
            ['flux', 'std_ra', 'std_dec', 'nmatch', 'nmatch_good'],
            label=f'{inv.name} {filt}')
        flux = loaders.column(tbl, 'flux')
        mag, maglabel = loaders.magnitudes(flux, zps.get(filt))
        # std_ra/std_dec are stored in degrees; a per-source value of exactly
        # zero is the forced-position artefact (a copied seed whose position
        # was never measured per frame), not a 0 mas measurement.
        sra = loaders.column(tbl, 'std_ra') * MAS
        sdec = loaders.column(tbl, 'std_dec') * MAS
        nmatch = loaders.column(tbl, 'nmatch')
        scatter = np.hypot(sra, sdec) / np.sqrt(2.0)
        unmeasured = scatter < MIN_MEASURABLE_SCATTER_MAS
        scatter[unmeasured] = np.nan
        n_unmeasured = int(unmeasured.sum())
        if scatter.size > max_sources:
            rng = np.random.default_rng(0)
            pick = rng.choice(scatter.size, max_sources, replace=False)
            mag, scatter, nmatch = mag[pick], scatter[pick], nmatch[pick]
        per_filter[filt] = dict(mag=mag, scatter=scatter, nmatch=nmatch,
                                maglabel=maglabel,
                                n_zero=n_unmeasured, n=len(tbl))

    has_cross = inv.has_crossband and len(filters) > 1
    npanels = len(filters) + (1 if has_cross else 0)
    fig, axes = style.panel_grid(npanels, panel=(2.7, 2.3))
    colors = style.filter_colors(filters)

    floors = {}
    for ax, filt in zip(axes, filters):
        d = per_filter[filt]
        good = np.isfinite(d['mag']) & np.isfinite(d['scatter']) & (d['scatter'] > 0)
        if good.sum() < 50:
            ax.text(0.5, 0.5, 'too few measured positions', ha='center',
                    va='center', transform=ax.transAxes, fontsize=7)
            ax.set_title(filt.upper())
            continue
        ax.hexbin(d['mag'][good], d['scatter'][good], bins='log', gridsize=45,
                  mincnt=1, cmap='Greys', yscale='log',
                  extent=(*style.robust_range(d['mag'][good], 0.5, 99.5),
                          np.log10(max(np.nanpercentile(d['scatter'][good], 0.5), 0.1)),
                          np.log10(np.nanpercentile(d['scatter'][good], 99.9))))
        centres, pct = style.running_percentiles(d['mag'][good], d['scatter'][good])
        if centres.size:
            ax.plot(centres, pct[50], color=colors[filt], lw=1.4)
            ax.fill_between(centres, pct[16], pct[84], color=colors[filt], alpha=0.25,
                            lw=0)
        # The floor: median scatter over the brightest well-measured decile.
        bright = good & (d['mag'] <= np.nanpercentile(d['mag'][good], 10))
        floor = float(np.nanmedian(d['scatter'][bright])) if bright.sum() > 20 else np.nan
        floors[filt] = floor
        if np.isfinite(floor):
            ax.axhline(floor, color=colors[filt], ls=':', lw=1)
            style.annotate(ax, f'floor {floor:.1f} mas\n'
                               f'N={d["n"]:,}', loc='upper left')
        ax.set_yscale('log')
        ax.set_title(filt.upper())
        ax.set_xlabel(d['maglabel'])
        ax.set_ylabel('per-exposure scatter (mas)')

    cross = {}
    if has_cross:
        cross = _crossband_separations(inv, filters)
        ax = axes[-1]
        if cross:
            names = cross['filters']
            img = cross['median_sep_mas']
            im = ax.imshow(img, cmap='viridis', vmin=0,
                           vmax=np.nanpercentile(img, 95) or 1)
            ax.set_xticks(range(len(names)))
            ax.set_yticks(range(len(names)))
            ax.set_xticklabels([n.upper() for n in names], rotation=90, fontsize=5.5)
            ax.set_yticklabels([n.upper() for n in names], fontsize=5.5)
            ax.grid(False)
            cb = ax.figure.colorbar(im, ax=ax, fraction=0.046)
            cb.set_label('median inter-filter sep. (mas)', fontsize=6.5)
            cb.ax.tick_params(labelsize=6)
            ax.set_title('cross-band agreement')
        else:
            ax.text(0.5, 0.5, 'no cross-band positions', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7)
            ax.set_title('cross-band agreement')

    fig.suptitle(f'{inv.name}: internal astrometry', fontsize=10, y=1.005)
    fig.tight_layout()
    path = save(fig, outdir, 'D2_astrometry_internal')

    measurements = dict(floors_mas=floors,
                        n_zero_scatter={f: per_filter[f]['n_zero'] for f in filters},
                        n_sources={f: per_filter[f]['n'] for f in filters},
                        crossband=cross)
    caption = (
        f'Internal astrometry of {inv.name}. '
        'Per filter: the scatter of the individual-exposure centroids about '
        'the merged position, against brightness; greyscale is source density, '
        'the coloured line and band are the running median and 16--84th '
        'percentile, and the dotted line marks the bright-decile floor quoted '
        'in the panel. '
        + ('The final panel is the median separation between the same stars '
           'measured independently in each pair of filters; only sources '
           'independently detected in both bands are counted, since a forced '
           'or seeded position imported from another band would measure the '
           'merge radius rather than the astrometry.' if has_cross else ''))
    return FigureResult('D2_astrometry_internal', path, caption, 'astrometry',
                        measurements)


def _crossband_separations(inv, filters, max_pairs=200000):
    """Median same-source separation between every pair of filters (mas).

    Only sources INDEPENDENTLY DETECTED in both filters count.  A row of the
    merged table can carry a position in a band where nothing was detected --
    a forced or seeded fit, placed at a position imported from another band --
    and comparing that against its own source of truth measures the merge
    radius, not the astrometry.  On W51 the difference is a median of 3.7 mas
    and an 84th percentile of 623 mas over all rows, against 1.6 and 5.4 mas
    over independently detected ones.
    """
    from astropy.coordinates import SkyCoord
    wanted = []
    for filt in filters:
        wanted += [f'skycoord_{filt}.ra', f'skycoord_{filt}.dec',
                   f'independently_detected_{filt}']
    tbl = loaders.read_columns(inv.crossband_catalog, wanted,
                               label=f'{inv.name} cross-band')
    # On disk these are ``skycoord_<filt>.ra``/``.dec``; Table.read reassembles
    # each pair into a single mixin column ``skycoord_<filt>``, so the presence
    # test has to be against the mixin name, not the on-disk one.
    present = [f for f in filters if f'skycoord_{f}' in tbl.colnames]
    if len(present) < 2:
        return {}
    coords, detected = {}, {}
    any_flag = False
    for filt in present:
        coords[filt] = (loaders.column(tbl, f'skycoord_{filt}.ra'),
                        loaders.column(tbl, f'skycoord_{filt}.dec'))
        col = f'independently_detected_{filt}'
        if col in tbl.colnames:
            detected[filt] = loaders.column(tbl, col, fill=0) > 0
            any_flag = True
        else:
            detected[filt] = np.ones(len(tbl), dtype=bool)
    n = len(present)
    img = np.full((n, n), np.nan)
    p84 = np.full((n, n), np.nan)
    counts = np.zeros((n, n), dtype=int)
    for i, fi in enumerate(present):
        for j, fj in enumerate(present):
            if j <= i:
                continue
            ra_i, dec_i = coords[fi]
            ra_j, dec_j = coords[fj]
            good = (np.isfinite(ra_i) & np.isfinite(ra_j) &
                    np.isfinite(dec_i) & np.isfinite(dec_j) &
                    detected[fi] & detected[fj])
            if good.sum() < 20:
                continue
            idx = np.flatnonzero(good)
            if idx.size > max_pairs:
                idx = np.random.default_rng(0).choice(idx, max_pairs, replace=False)
            # Same ROW of the merged table: these are the same source by
            # construction, so this is a same-star residual and not a
            # nearest-neighbour match at all.
            c_i = SkyCoord(ra_i[idx] * u.deg, dec_i[idx] * u.deg)
            c_j = SkyCoord(ra_j[idx] * u.deg, dec_j[idx] * u.deg)
            sep = c_i.separation(c_j).to(u.mas).value
            img[i, j] = img[j, i] = float(np.median(sep))
            p84[i, j] = p84[j, i] = float(np.percentile(sep, 84))
            counts[i, j] = counts[j, i] = idx.size
    return dict(filters=present, median_sep_mas=img, p84_sep_mas=p84,
                n_pairs=counts, independent_only=bool(any_flag))


def absolute_astrometry(inv, outdir, anchor=None):
    """Bulk tie to the reference catalogue plus the per-tile residual map."""
    plt = style.use_style()
    if not inv.reference_catalogs or not inv.measured_filters:
        return None
    from astropy.coordinates import SkyCoord
    # The reference is picked by registry order, not string order, so a field
    # with several registered references does not silently prefer one by
    # alphabet; the choice and the discards are recorded.
    ref_items = list(inv.reference_catalogs.items())
    ref_key, refpath = ref_items[0]
    reftbl = loaders.read_columns(refpath, ['skycoord.ra', 'skycoord.dec',
                                            'ra', 'dec', 'source'],
                                  label=f'{inv.name} reference')
    ref = _refcoords(reftbl)
    if ref is None or len(ref) < 100:
        return None

    filters = list(inv.measured_filters)
    fig, axes = style.panel_grid(len(filters) + 1, panel=(2.8, 2.5))
    colors = style.filter_colors(filters)
    bulk = {}

    # PASS 1: read every filter's usable coordinates and accumulate the field
    # footprint.  The KD-tree is cropped to the UNION of all filters' bounds,
    # not the first filter's -- otherwise any band whose footprint extends past
    # the first band's box (SW vs LW, or a second-proposal band) loses reference
    # coverage at its edges and its edge tiles fail as "too sparse", which the
    # prose would misread as sparsity rather than missing reference.
    per_filter = {}
    rmin = dmin = np.inf
    rmax = dmax = -np.inf
    for filt in filters:
        tbl = loaders.read_columns(inv.per_filter_catalogs[filt],
                                   ['skycoord.ra', 'skycoord.dec', 'flux',
                                    'qfit', 'is_saturated'],
                                   label=f'{inv.name} {filt}')
        ra = loaders.column(tbl, 'skycoord.ra')
        dec = loaders.column(tbl, 'skycoord.dec')
        usable = np.isfinite(ra) & np.isfinite(dec)
        # Saturated stars carry a centroid bias (the F187N Pa-alpha case), so
        # they are excluded from the tie rather than allowed to drag it.
        sat = loaders.column(tbl, 'is_saturated', fill=0) > 0
        if (usable & ~sat).sum() > 200:
            usable &= ~sat
        if usable.sum() < 200:
            per_filter[filt] = None
            continue
        idx = np.flatnonzero(usable)
        n_used = idx.size
        if idx.size > MAX_TIE_SOURCES:
            idx = np.sort(np.random.default_rng(1182).choice(
                idx, MAX_TIE_SOURCES, replace=False))
            n_used = MAX_TIE_SOURCES
        coords = SkyCoord(ra[idx] * u.deg, dec[idx] * u.deg)
        per_filter[filt] = (coords, int(n_used))
        rmin = min(rmin, float(coords.ra.deg.min()))
        rmax = max(rmax, float(coords.ra.deg.max()))
        dmin = min(dmin, float(coords.dec.deg.min()))
        dmax = max(dmax, float(coords.dec.deg.max()))

    if not np.isfinite(rmin):
        return None
    # A four-corner SkyCoord of the union box; _crop_reference reads its
    # min/max, so this crops the reference to the union footprint.
    union_coords = SkyCoord([rmin, rmax, rmin, rmax] * u.deg,
                            [dmin, dmin, dmax, dmax] * u.deg)
    ref_tree = KDTreeReference(_crop_reference(ref, union_coords))

    # PASS 2: measure and draw per filter against the shared, union-cropped tree.
    for ax, filt in zip(axes, filters):
        got = per_filter.get(filt)
        if got is None:
            ax.text(0.5, 0.5, 'too few sources', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7)
            ax.set_title(filt.upper())
            continue
        coords, n_used = got
        # confirm_windows: an unconfirmed bulk peak that is large relative to
        # its own search window can be footprint geometry or a sparse-histogram
        # chance bin rather than a real tie (issues #158, #600).  The write-up
        # escalates such a bulk to a "grossly shifted" verdict, so the peak has
        # to be confirmed before it earns that; the cost is one or two extra
        # measurements, and only on a swept or edge-riding result.
        result = measure_offset(coords, ref_tree, context=f'{inv.name}/{filt}',
                                confirm_windows=True)
        bounds = (float(coords.ra.deg.min()), float(coords.ra.deg.max()),
                  float(coords.dec.deg.min()), float(coords.dec.deg.max()))
        grid = measure_offset_grid(coords, ref_tree, nx=GRID_N, ny=GRID_N,
                                   ra_bounds=bounds[:2], dec_bounds=bounds[2:],
                                   context=f'{inv.name}/{filt}')
        bulk[filt] = _tie_record(result, grid)
        bulk[filt]['n_sources_used'] = int(n_used)
        bulk[filt]['reference'] = ref_key
        _draw_grid(ax, grid, bounds)
        ax.set_title(filt.upper())
        if result is not None:
            style.annotate(
                ax,
                f"bulk {result['dra']:+.0f}, {result['ddec']:+.0f} mas\n"
                f"contrast {result['contrast']:.1f}"
                + ('  SWEPT' if result.get('swept') else ''),
                loc='upper left')

    ax = axes[-1]
    _draw_bulk_summary(ax, bulk, colors)
    fig.suptitle(f'{inv.name}: absolute astrometry vs '
                 f'{refpath.rsplit("/", 1)[-1]}', fontsize=10, y=1.005)
    fig.tight_layout()
    path = save(fig, outdir, 'D3_astrometry_absolute')

    caption = (
        f'Absolute astrometry of {inv.name} against its registered reference '
        f'catalogue. Each per-filter panel maps the offset-histogram tie on a '
        f'{GRID_N}$\\times${GRID_N} tile grid (arrows, exaggerated; colour is '
        'the per-tile peak contrast), with the whole-field bulk tie quoted in '
        'the corner. Grey crosses mark tiles whose tie was only found by '
        'widening the search window and whose offset is therefore not '
        'trustworthy. The final panel collects the bulk ties. Offsets are '
        'measured by offset-histogram stacking, which is immune to the '
        'nearest-neighbour pairing collapse that a median against a dense '
        'reference suffers; the per-tile map is shown because a bulk offset '
        'near zero does not by itself mean the field is registered.')
    return FigureResult('D3_astrometry_absolute', path, caption, 'astrometry',
                        dict(reference=refpath, reference_key=ref_key,
                             references_available=[k for k, _ in ref_items],
                             bulk=bulk, grid_n=GRID_N))


def _crop_reference(ref, coords, margin_arcsec=REFERENCE_MARGIN_ARCSEC):
    """Reference sources within *margin* of the field's bounding box.

    Two reasons.  Cost: the pair search is over the reference, so trimming a
    survey catalogue to the footprint is close to free accuracy.  Correctness:
    a tile with few sources sweeps its search window out to a minute of arc,
    and a reference extending far beyond the field gives that sweep somewhere
    spurious to land.
    """
    margin_deg = margin_arcsec / 3600.0
    dec0 = float(np.median(coords.dec.deg))
    cosdec = max(np.cos(np.radians(dec0)), 1e-6)
    r0, r1 = float(coords.ra.deg.min()), float(coords.ra.deg.max())
    d0, d1 = float(coords.dec.deg.min()), float(coords.dec.deg.max())
    keep = ((ref.ra.deg >= r0 - margin_deg / cosdec) &
            (ref.ra.deg <= r1 + margin_deg / cosdec) &
            (ref.dec.deg >= d0 - margin_deg) &
            (ref.dec.deg <= d1 + margin_deg))
    # Never crop down to something unusable; fall back to the full catalogue.
    return ref[keep] if keep.sum() >= 100 else ref


def _coords_from(tbl, candidates=(('skycoord.ra', 'skycoord.dec'), ('ra', 'dec'))):
    """First usable sky-coordinate representation in *tbl*, or None.

    Reference catalogues are written by several different producers, so both
    the mixin form and a bare ``ra``/``dec`` pair have to be accepted.
    """
    from astropy.coordinates import SkyCoord
    for racol, deccol in candidates:
        ra = loaders.column(tbl, racol)
        dec = loaders.column(tbl, deccol)
        good = np.isfinite(ra) & np.isfinite(dec)
        if good.sum() < 10:
            continue
        return SkyCoord(ra[good] * u.deg, dec[good] * u.deg)
    return None


_refcoords = _coords_from
_catcoords = _coords_from


def _tile_trustworthy(c):
    """A tile whose offset describes the DATA, not the search window.

    Excluded: tiles that only tied after the window was swept (``swept``), and
    tiles whose peak rides the edge of the window (``window_edge_fraction`` >=
    ``WINDOW_EDGE_FRACTION``) -- in both the offset is a property of the window,
    which is exactly the ``measure_offset`` blind spot (see its docstring).
    """
    return bool(c.get('ok') and not c.get('swept')
                and float(c.get('window_edge_fraction', 0.0)) < WINDOW_EDGE_FRACTION)


def _tie_record(result, grid):
    """Flatten a measure_offset / measure_offset_grid pair into plain numbers.

    The per-tile statistics are computed on the residual ABOUT THE BULK TIE,
    not on the raw tile offsets: the tile map exists to expose spatially varying
    registration errors that the whole-field number cannot, so it must be made
    independent of that whole-field shift.  A clean 60 mas bulk tie with a
    perfectly flat map otherwise reads as a 60 mas "locally displaced region".
    The raw (bulk-inclusive) offsets are kept alongside for reference.
    """
    rec = dict(bulk=None, tiles=None)
    bulk_dra = bulk_ddec = 0.0
    if result is not None:
        rec['bulk'] = {k: (float(v) if isinstance(v, (int, float, np.floating))
                           else v)
                       for k, v in result.items() if k != 'windows'}
        if result.get('dra') is not None:
            bulk_dra, bulk_ddec = float(result['dra']), float(result['ddec'])
    if grid:
        all_cells = grid.get('cells', [])
        n_ok = sum(1 for c in all_cells if c.get('ok'))
        cells = [c for c in all_cells if _tile_trustworthy(c)]
        swept = [c for c in all_cells if c.get('ok') and c.get('swept')]
        edge = [c for c in all_cells if c.get('ok') and not c.get('swept')
                and float(c.get('window_edge_fraction', 0.0)) >= WINDOW_EDGE_FRACTION]
        # Tiles that found NO coherent tie at all (contrast failure): counted so
        # the fractions in the write-up add up.
        n_no_tie = len(all_cells) - n_ok
        common = dict(n_measured=len(cells), n_total=len(all_cells),
                      n_swept=len(swept), n_window_edge=len(edge),
                      n_no_tie=int(n_no_tie))
        if cells:
            # residual about the bulk tie (the field-independent quantity)
            res_off = np.array([np.hypot(c['dra'] - bulk_dra, c['ddec'] - bulk_ddec)
                                for c in cells], dtype=float)
            raw_off = np.array([c['off'] for c in cells], dtype=float)
            cons = np.array([c['contrast'] for c in cells], dtype=float)
            rec['tiles'] = dict(
                **common,
                median_off_mas=float(np.median(res_off)),
                # p95 as well as the maximum: one edge tile with an odd peak
                # should not be able to speak for the field, and the gap
                # between the two says whether a large "worst" is structural
                # or a single outlier.
                p95_off_mas=float(np.percentile(res_off, 95)),
                worst_off_mas=float(np.max(res_off)),
                n_above_50mas=int(np.sum(res_off > 50.0)),
                median_contrast=float(np.median(cons)),
                bulk_off_mas=float(np.hypot(bulk_dra, bulk_ddec)),
                raw_median_off_mas=float(np.median(raw_off)),
                raw_worst_off_mas=float(np.max(raw_off)))
        else:
            rec['tiles'] = dict(**common)
    return rec


def _draw_grid(ax, grid, bounds):
    """Quiver of the per-tile offsets, coloured by peak contrast.

    ``measure_offset_grid`` reports tiles by index, so the tile centres are
    reconstructed here from the same bounds that were handed to it.  Axes are
    arcseconds from the field centre rather than absolute degrees: at these
    field sizes a degree axis is all shared leading digits.
    """
    all_cells = (grid or {}).get('cells', [])
    cells = [c for c in all_cells if _tile_trustworthy(c)]
    # marked, not drawn: swept OR window-edge-riding -- an arrow would be the
    # widened/edge window's artefact, not a tie.
    swept = [c for c in all_cells if c.get('ok') and not _tile_trustworthy(c)]
    r0, r1, d0, d1 = bounds
    rc, dc = 0.5 * (r0 + r1), 0.5 * (d0 + d1)
    cosdec = max(np.cos(np.radians(dc)), 1e-6)
    redges = np.linspace(r0, r1, GRID_N + 1)
    dedges = np.linspace(d0, d1, GRID_N + 1)

    def centres(cs):
        x = np.array([0.5 * (redges[c['ix']] + redges[c['ix'] + 1]) for c in cs])
        y = np.array([0.5 * (dedges[c['iy']] + dedges[c['iy'] + 1]) for c in cs])
        return (x - rc) * 3600.0 * cosdec, (y - dc) * 3600.0

    ax.set_xlabel(r'$\Delta$RA from centre (arcsec)')
    ax.set_ylabel(r'$\Delta$Dec from centre (arcsec)')
    if swept:
        # Mark, do not plot: a swept tile has no coherent tie at the nominal
        # window and its arrow would be the widened window's artefact.
        sx, sy = centres(swept)
        ax.plot(sx, sy, 'x', color='0.6', ms=3, mew=0.7, zorder=1)
    if not cells:
        ax.text(0.5, 0.5, 'no coherent per-tile tie', ha='center', va='center',
                transform=ax.transAxes, fontsize=7)
        ax.invert_xaxis()
        return
    x, y = centres(cells)
    # Offsets come back in milliarcseconds; the axes are in arcseconds.
    dra = np.array([c['dra'] for c in cells], dtype=float) / 1000.0
    ddec = np.array([c['ddec'] for c in cells], dtype=float) / 1000.0
    contrast = np.array([c['contrast'] for c in cells], dtype=float)

    # Scale so the median offset spans about half a tile, then clip the drawn
    # length at one tile.  Without the clip a single outlier tile -- and there
    # is usually one at the edge -- draws an arrow across the whole panel and
    # hides the structure the map exists to show.
    tile_arcsec = (r1 - r0) * 3600.0 * cosdec / GRID_N
    mag = np.hypot(dra, ddec)
    typical = float(np.median(mag[mag > 0])) if np.any(mag > 0) else 1.0
    gain = (tile_arcsec / 2.0) / max(typical, 1e-9)
    drawn = np.minimum(mag * gain, tile_arcsec)
    unit = np.where(mag > 0, drawn / np.maximum(mag, 1e-12), 0.0)
    q = ax.quiver(x, y, dra * unit, ddec * unit, contrast,
                  cmap='plasma', angles='xy', scale_units='xy', scale=1.0,
                  width=0.005, pivot='tail')
    cb = ax.figure.colorbar(q, ax=ax, fraction=0.046)
    cb.set_label('peak contrast', fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    style.annotate(ax, f'arrow $\\times${gain:.0f}\n'
                       f'median {1000 * typical:.0f} mas', loc='lower right')
    ax.invert_xaxis()


def _draw_bulk_summary(ax, bulk, colors):
    """Each filter's whole-field tie in the (dRA, dDec) plane.

    Points are labelled in place rather than through a legend: with ten
    filters a legend covers the very clustering the panel is meant to show.
    """
    ax.axhline(0, color='0.6', lw=0.8)
    ax.axvline(0, color='0.6', lw=0.8)
    drawn = 0
    # Filters in one field usually tie to nearly the same offset, so the
    # labels would sit on top of each other; fan them out around the point.
    fan = [(6, 4), (6, -8), (-6, 4), (-6, -8), (0, 8), (0, -12)]
    for filt, rec in bulk.items():
        b = rec.get('bulk')
        if not b or b.get('dra') is None:
            continue
        ax.errorbar(b['dra'], b['ddec'],
                    xerr=b.get('dra_err'), yerr=b.get('ddec_err'),
                    marker='o', ms=4, color=colors.get(filt, 'k'),
                    lw=0.8, capsize=1.5, zorder=3)
        dx, dy = fan[drawn % len(fan)]
        ax.annotate(filt.upper(), (b['dra'], b['ddec']),
                    textcoords='offset points', xytext=(dx, dy), fontsize=5.5,
                    ha='left' if dx >= 0 else 'right',
                    color=colors.get(filt, 'k'))
        drawn += 1
    ax.set_xlabel(r'bulk $\Delta$RA (mas)')
    ax.set_ylabel(r'bulk $\Delta$Dec (mas)')
    ax.set_title('bulk tie per filter')
    if drawn:
        ax.set_aspect('equal', adjustable='datalim')
        ax.margins(0.25)
    else:
        ax.text(0.5, 0.5, 'no bulk tie measured', ha='center', va='center',
                transform=ax.transAxes, fontsize=7)
