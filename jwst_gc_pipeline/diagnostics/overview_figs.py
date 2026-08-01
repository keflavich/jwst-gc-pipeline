"""Field overview: coverage, source density, and how deep each filter went.

This is the orientation figure.  Everything after it is a measurement of some
property of the catalogue; this one establishes what the catalogue *is* --
which filters cover which sky, how crowded the field is, and how the number
counts turn over.  The turnover magnitude is the practical completeness limit
and is worth having in front of the reader before any precision claim.
"""

import numpy as np

from jwst_gc_pipeline.diagnostics import loaders, style
from jwst_gc_pipeline.diagnostics.figures import FigureResult, save

DENSITY_BINS = 72


def overview(inv, outdir, max_sources=400000):
    """Source-density map of the deepest filter, coverage, and number counts."""
    style.use_style()
    filters = list(inv.measured_filters)
    if not filters:
        return None
    colors = style.filter_colors(filters)
    zps = loaders.photometric_zeropoints(inv.crossband_catalog, filters)

    per_filter = {}
    for filt in filters:
        tbl = loaders.read_columns(inv.per_filter_catalogs[filt],
                                   ['flux', 'skycoord.ra', 'skycoord.dec'],
                                   label=f'{inv.name} {filt}')
        flux = loaders.column(tbl, 'flux')
        mag, maglabel = loaders.magnitudes(flux, zps.get(filt))
        per_filter[filt] = dict(
            ra=loaders.column(tbl, 'skycoord.ra'),
            dec=loaders.column(tbl, 'skycoord.dec'),
            mag=mag, maglabel=maglabel, n=len(tbl))

    richest = max(filters, key=lambda f: per_filter[f]['n'])

    fig, axes = style.panel_grid(3, ncols=3, panel=(3.1, 2.7))
    ax_map, ax_cov, ax_counts = axes

    d = per_filter[richest]
    good = np.isfinite(d['ra']) & np.isfinite(d['dec'])
    density, extent = _density_map(d['ra'][good], d['dec'][good])
    im = ax_map.imshow(density, extent=extent, aspect='auto', cmap='inferno',
                       vmin=0, vmax=np.nanpercentile(density, 99))
    ax_map.invert_xaxis()
    ax_map.set_xlabel('RA (deg)')
    ax_map.set_ylabel('Dec (deg)')
    ax_map.set_title(f'source density, {richest.upper()}')
    cb = fig.colorbar(im, ax=ax_map, fraction=0.046)
    cb.set_label(r'sources arcsec$^{-2}$', fontsize=6.5)
    cb.ax.tick_params(labelsize=5.5)

    areas = {}
    for filt in filters:
        p = per_filter[filt]
        ok = np.isfinite(p['ra']) & np.isfinite(p['dec'])
        if ok.sum() < 10:
            continue
        # A convex outline would overstate a tiled footprint; the occupied
        # fraction of a coarse grid is a closer estimate of real coverage.
        area = _occupied_area_arcsec2(p['ra'][ok], p['dec'][ok])
        areas[filt] = area
        ax_cov.plot(np.median(p['ra'][ok]), np.median(p['dec'][ok]), 'x',
                    color=colors[filt], ms=4)
        ax_cov.scatter(p['ra'][ok][::max(ok.sum() // 4000, 1)],
                       p['dec'][ok][::max(ok.sum() // 4000, 1)],
                       s=0.4, color=colors[filt], alpha=0.35, lw=0,
                       label=filt.upper())
    ax_cov.invert_xaxis()
    ax_cov.set_xlabel('RA (deg)')
    ax_cov.set_ylabel('Dec (deg)')
    ax_cov.set_title('per-filter coverage')
    ax_cov.legend(ncol=2, fontsize=5.0, markerscale=8)

    turnovers = {}
    for filt in filters:
        p = per_filter[filt]
        mag = p['mag'][np.isfinite(p['mag'])]
        if mag.size < 100:
            continue
        lo, hi = np.percentile(mag, [0.2, 99.8])
        bins = np.linspace(lo, hi, 60)
        counts, edges = np.histogram(mag, bins=bins)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax_counts.plot(centres, np.maximum(counts, 0.5), color=colors[filt],
                       lw=1.1, label=filt.upper())
        if counts.max() > 0:
            turnovers[filt] = float(centres[int(np.argmax(counts))])
    ax_counts.set_yscale('log')
    ax_counts.set_xlabel(per_filter[richest]['maglabel'])
    ax_counts.set_ylabel('sources per bin')
    ax_counts.set_title('number counts')
    ax_counts.legend(ncol=2, fontsize=5.0)

    fig.suptitle(f'{inv.name}: field overview', fontsize=10, y=1.01)
    fig.tight_layout()
    path = save(fig, outdir, 'D1_overview')

    measurements = dict(
        n_sources={f: per_filter[f]['n'] for f in filters},
        area_arcsec2=areas,
        peak_density=float(np.nanmax(density)) if np.isfinite(density).any() else np.nan,
        median_density=float(np.nanmedian(density[density > 0]))
        if np.isfinite(density).any() else np.nan,
        turnover=turnovers,
        richest_filter=richest)
    caption = (
        f'Overview of {inv.name}. Left: source surface density in '
        f'{DENSITY_BINS}$\\times${DENSITY_BINS} sky cells for {richest.upper()}, '
        'the filter with the most detections. Centre: the sky coverage of '
        'each filter, sub-sampled. Right: number counts per filter; the peak '
        'of each curve is the turnover magnitude, the practical completeness '
        'limit beyond which the counts are set by detectability rather than '
        'by the stellar population.')
    return FigureResult('D1_overview', path, caption, 'overview', measurements)


def _density_map(ra, dec, nbins=DENSITY_BINS):
    """Source counts per square arcsecond on a regular sky grid."""
    if ra.size == 0:
        return np.zeros((nbins, nbins)), (0, 1, 0, 1)
    r0, r1 = np.percentile(ra, [0.05, 99.95])
    d0, d1 = np.percentile(dec, [0.05, 99.95])
    counts, xe, ye = np.histogram2d(ra, dec, bins=nbins,
                                    range=[[r0, r1], [d0, d1]])
    cosdec = np.cos(np.radians(0.5 * (d0 + d1)))
    cell_arcsec2 = ((xe[1] - xe[0]) * 3600.0 * cosdec) * \
                   ((ye[1] - ye[0]) * 3600.0)
    return counts.T / max(cell_arcsec2, 1e-12), (r0, r1, d0, d1)


def _occupied_area_arcsec2(ra, dec, nbins=64):
    """Sky area covered, as the occupied fraction of the bounding grid."""
    r0, r1 = np.percentile(ra, [0.02, 99.98])
    d0, d1 = np.percentile(dec, [0.02, 99.98])
    counts, xe, ye = np.histogram2d(ra, dec, bins=nbins,
                                    range=[[r0, r1], [d0, d1]])
    cosdec = np.cos(np.radians(0.5 * (d0 + d1)))
    cell = ((xe[1] - xe[0]) * 3600.0 * cosdec) * ((ye[1] - ye[0]) * 3600.0)
    return float((counts > 0).sum() * cell)
