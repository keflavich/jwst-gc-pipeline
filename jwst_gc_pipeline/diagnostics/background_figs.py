r"""Background (diffuse flux) diagnostic figures.

The pipeline measures the light that is *not* in the stars in two distinct
ways, and the difference between them is the whole point of these figures.

``local_bkg``
    What ``photutils``' ``LocalBackground`` handed the PSF fitter: a
    sigma-clipped median in an annulus around each source, evaluated on
    **whatever image the fitter was given**.  In the early stages that image
    is the calibrated frame, so ``local_bkg`` measures the astrophysical
    diffuse emission.  In the later stages the frame has already had a
    smoothed residual background removed, so the same column measures only
    what that subtraction left behind, and collapses towards zero.  The
    column is therefore *stage-dependent by construction*: comparing it
    between two stages compares two different questions.

``modelsub_bkg``
    The residual-footprint background added in 2026-07: the mean over a
    :math:`3\times3` pixel box centred on the star, measured on the
    per-exposure residual after the star-only model is subtracted, and then
    combined across exposures as a sigma-clipped weighted mean.  Writing
    :math:`D_i` for the exposure data, :math:`S_i` for the smoothed background
    that was subtracted before fitting and :math:`P_i` for the fitted PSF
    model,

    .. math::

       A_i = D_i - S_i - B_i, \qquad R_i = D_i - S_i - P_i,

    where :math:`A_i` is what the fitter saw and :math:`R_i` is what this
    column measures.  Because it starts from the raw data and removes only the
    star, it is invariant across pipeline stages, which is exactly what
    ``local_bkg`` is not.

Two figures follow: the distributions and their brightness dependence
(:func:`background_distributions`), and the spatial structure set against the
mosaic that produced it (:func:`background_spatial`).  The second is the test
of the physical claim -- that this quantity tracks the extended emission --
and it is made quantitative with a rank correlation rather than left to the
eye.
"""

import numpy as np

from jwst_gc_pipeline.diagnostics import loaders, style
from jwst_gc_pipeline.diagnostics.figures import FigureResult, save

# Sources brighter than this percentile are the regime where both background
# estimators are contaminated by the star's own wings.
BRIGHT_PERCENTILE = 99.0

MAP_BINS = 56


def _read_background(inv, filt):
    """``(local_bkg, modelsub_bkg, mag, maglabel, coords, n)`` for one filter."""
    zps = loaders.photometric_zeropoints(inv.crossband_catalog,
                                         inv.measured_filters)
    tbl = loaders.read_columns(
        inv.per_filter_catalogs[filt],
        ['flux', 'local_bkg', 'mean_modelsub_bkg', 'modelsub_bkg',
         'mean_modelsub_bkg_std', 'modelsub_bkg_rms_avg',
         'skycoord.ra', 'skycoord.dec'],
        label=f'{inv.name} {filt}')
    flux = loaders.column(tbl, 'flux')
    mag, maglabel = loaders.magnitudes(flux, zps.get(filt))
    local = loaders.column(tbl, 'local_bkg')
    # The merged column is the one to prefer; the per-frame column only
    # appears in a single-frame product.
    modelsub = loaders.column(tbl, 'mean_modelsub_bkg')
    if not np.isfinite(modelsub).any():
        modelsub = loaders.column(tbl, 'modelsub_bkg')
    ra = loaders.column(tbl, 'skycoord.ra')
    dec = loaders.column(tbl, 'skycoord.dec')
    return dict(local=local, modelsub=modelsub, mag=mag, maglabel=maglabel,
                ra=ra, dec=dec, n=len(tbl))


def background_distributions(inv, outdir, max_sources=400000):
    """Distribution of each background estimator and its brightness dependence."""
    style.use_style()
    filters = list(inv.measured_filters)
    if not filters:
        return None
    colors = style.filter_colors(filters)

    data = {f: _read_background(inv, f) for f in filters}
    has_modelsub = {f: np.isfinite(data[f]['modelsub']).any() for f in filters}

    # One distribution panel for all filters, then one brightness panel each.
    fig, axes = style.panel_grid(len(filters) + 1, panel=(2.7, 2.3))
    stats = {}

    ax = axes[0]
    for filt in filters:
        d = data[filt]
        vals = d['local'][np.isfinite(d['local'])]
        if vals.size < 50:
            continue
        lo, hi = np.percentile(vals, [0.5, 99.5])
        if hi <= lo:
            continue
        ax.hist(vals, bins=np.linspace(lo, hi, 80), histtype='step',
                density=True, color=colors[filt], lw=1.1, label=filt.upper())
        if has_modelsub[filt]:
            mvals = d['modelsub'][np.isfinite(d['modelsub'])]
            ax.hist(mvals, bins=np.linspace(lo, hi, 80), histtype='step',
                    density=True, color=colors[filt], lw=1.0, ls='--')
    ax.set_xlabel('background (image units)')
    ax.set_ylabel('normalised density')
    ax.set_title('distributions (solid: local_bkg)')
    ax.legend(ncol=2, fontsize=5.5)

    for ax, filt in zip(axes[1:], filters):
        d = data[filt]
        mag, local, modelsub = d['mag'], d['local'], d['modelsub']
        if mag.size > max_sources:
            pick = np.random.default_rng(0).choice(mag.size, max_sources,
                                                   replace=False)
            mag, local, modelsub = mag[pick], local[pick], modelsub[pick]
        good = np.isfinite(mag) & np.isfinite(local)
        if good.sum() < 50:
            ax.text(0.5, 0.5, 'no background column', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7)
            ax.set_title(filt.upper())
            continue
        centres, pct = style.running_percentiles(mag[good], local[good])
        ax.plot(centres, pct[50], color=colors[filt], lw=1.4, label='local_bkg')
        ax.fill_between(centres, pct[16], pct[84], color=colors[filt],
                        alpha=0.2, lw=0)
        entry = dict(median_local=float(np.nanmedian(local[good])),
                     n=int(d['n']))
        if has_modelsub[filt]:
            gm = np.isfinite(mag) & np.isfinite(modelsub)
            cm, pm = style.running_percentiles(mag[gm], modelsub[gm])
            ax.plot(cm, pm[50], color='k', lw=1.1, ls='--',
                    label='mean_modelsub_bkg')
            entry['median_modelsub'] = float(np.nanmedian(modelsub[gm]))
        # The bright end: where the star's own wings leak into the estimator.
        if good.sum() > 200:
            bright = good & (mag <= np.nanpercentile(mag[good],
                                                     100 - BRIGHT_PERCENTILE))
            faint = good & (mag > np.nanpercentile(mag[good], 50))
            if bright.sum() > 20 and faint.sum() > 20:
                entry['bright_excess_local'] = float(
                    np.nanmedian(local[bright]) - np.nanmedian(local[faint]))
                ax.axvline(np.nanpercentile(mag[good], 100 - BRIGHT_PERCENTILE),
                           color='0.5', ls=':', lw=0.8)
        stats[filt] = entry
        ax.set_title(filt.upper())
        ax.set_xlabel(d['maglabel'])
        ax.set_ylabel('background (image units)')
        ax.legend(fontsize=5.5, loc='upper right')

    fig.suptitle(f'{inv.name}: background estimators', fontsize=10, y=1.005)
    fig.tight_layout()
    path = save(fig, outdir, 'D6_background_distributions')
    n_with = sum(has_modelsub.values())
    caption = (
        f'Background estimators for {inv.name}. First panel: the distribution '
        'of the annulus background handed to the fitter (solid) and, where '
        'available, the residual-footprint background (dashed). Remaining '
        'panels: the running median of each against brightness, with the '
        '16--84th percentile band for the annulus estimator; the vertical '
        f'dotted line marks the brightest {100 - BRIGHT_PERCENTILE:.0f} per '
        'cent, where a star\'s own wings begin to enter its own background '
        'aperture. '
        + (f'The residual-footprint column is present for {n_with} of '
           f'{len(filters)} filters; it was added in 2026-07, so catalogues '
           'produced before then carry only the annulus estimator.'
           if n_with < len(filters) else ''))
    return FigureResult('D6_background_distributions', path, caption,
                        'background',
                        dict(per_filter=stats,
                             has_modelsub={f: bool(v) for f, v
                                           in has_modelsub.items()}))


def background_spatial(inv, outdir, max_filters=6, downsample=8):
    """Spatial background structure against the mosaic that produced it."""
    style.use_style()
    filters = [f for f in inv.measured_filters if f in inv.mosaics]
    if not filters:
        return None
    # Keep the figure legible: the widest-baseline subset, wavelength-ordered.
    if len(filters) > max_filters:
        idx = np.linspace(0, len(filters) - 1, max_filters).round().astype(int)
        filters = [filters[i] for i in sorted(set(idx.tolist()))]

    fig, axes = style.panel_grid(2 * len(filters), ncols=2, panel=(3.0, 2.6))
    correlations = {}

    for row, filt in enumerate(filters):
        ax_map, ax_corr = axes[2 * row], axes[2 * row + 1]
        d = _read_background(inv, filt)
        values = d['modelsub'] if np.isfinite(d['modelsub']).any() else d['local']
        which = ('mean_modelsub_bkg' if np.isfinite(d['modelsub']).any()
                 else 'local_bkg')
        # min_count=5: a cell median built from one or two stars is noise, and
        # a map of noise looks exactly like a map of fine structure.
        img, extent = style.binned_median_image(d['ra'], d['dec'], values,
                                                nbins=MAP_BINS, min_count=5)
        vmin, vmax = style.robust_range(img[np.isfinite(img)], 2, 98, pad=0.0)
        im = ax_map.imshow(img, extent=extent, aspect='auto', cmap='cividis',
                           vmin=vmin, vmax=vmax)
        ax_map.invert_xaxis()
        ax_map.set_xlabel('RA (deg)')
        ax_map.set_ylabel('Dec (deg)')
        ax_map.set_title(f'{filt.upper()}: median {which} per cell')
        cb = fig.colorbar(im, ax=ax_map, fraction=0.046)
        cb.set_label('image units', fontsize=6)
        cb.ax.tick_params(labelsize=5.5)

        sb = _mosaic_at_sources(inv.mosaics[filt], d, downsample=downsample)
        good = np.isfinite(sb) & np.isfinite(values)
        if good.sum() < 100:
            ax_corr.text(0.5, 0.5, 'no overlap with mosaic', ha='center',
                         va='center', transform=ax_corr.transAxes, fontsize=7)
            ax_corr.set_title(f'{filt.upper()}: vs mosaic')
            continue
        rho, _p = style.spearman(sb[good], values[good])
        correlations[filt] = dict(spearman=rho, n=int(good.sum()),
                                  estimator=which.replace('\\', ''))
        ax_corr.hexbin(sb[good], values[good], bins='log', gridsize=55, mincnt=1,
                       cmap='Greys',
                       extent=(*style.robust_range(sb[good], 1, 99),
                               *style.robust_range(values[good], 1, 99)))
        centres, pct = style.running_percentiles(sb[good], values[good])
        if centres.size:
            ax_corr.plot(centres, pct[50], color='crimson', lw=1.3)
        style.annotate(ax_corr, f'Spearman $\\rho$ = {rho:.3f}\nN={good.sum():,}',
                       loc='upper left')
        ax_corr.set_xlabel('mosaic surface brightness (MJy/sr)')
        ax_corr.set_ylabel(f'per-source {which}')
        ax_corr.set_title(f'{filt.upper()}: vs mosaic')

    fig.suptitle(f'{inv.name}: background structure vs diffuse emission',
                 fontsize=10, y=1.003)
    fig.tight_layout()
    path = save(fig, outdir, 'D7_background_spatial')
    caption = (
        f'Spatial background structure in {inv.name}. Left column: the median '
        f'per-source background in {MAP_BINS}$\\times${MAP_BINS} sky cells -- a '
        'map of the diffuse emission built entirely from the star catalogue. '
        'Right column: the same per-source value against the surface '
        'brightness of the drizzled mosaic sampled at that source\'s '
        'position, with the running median in red and the Spearman rank '
        'correlation quoted. A high rank correlation is the evidence that the '
        'per-source background is measuring the astrophysical extended '
        'emission rather than a fitting artefact.')
    return FigureResult('D7_background_spatial', path, caption, 'background',
                        dict(correlations=correlations,
                             filters=list(filters), map_bins=MAP_BINS))


def _mosaic_at_sources(mosaic_path, d, downsample=8):
    """Mosaic surface brightness at each source position in *d*."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    good = np.isfinite(d['ra']) & np.isfinite(d['dec'])
    out = np.full(d['ra'].size, np.nan)
    if good.sum() == 0:
        return out
    coords = SkyCoord(d['ra'][good] * u.deg, d['dec'][good] * u.deg)
    out[good] = loaders.sample_mosaic(mosaic_path, coords, downsample=downsample)
    return out
