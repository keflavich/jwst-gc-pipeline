r"""Photometric diagnostic figures.

Three questions.

**Precision and depth** (:func:`photometric_precision`) -- the fractional flux
uncertainty against brightness gives both the noise model and, where it
crosses :math:`1/5`, the :math:`5\sigma` depth.  Two uncertainties are
plotted because the pipeline reports two and they answer different questions:
``flux_err`` is the fitter's formal covariance from the single merged fit,
while ``flux_err_prop`` propagates the per-exposure scatter.  Where the
propagated error exceeds the formal one, something beyond photon noise --
crowding, an imperfect PSF, residual background structure -- is moving the
flux between exposures, and the formal error understates reality.

**Fit quality** (:func:`photometric_quality`) -- ``qfit`` is the normalised
residual of the PSF fit,

.. math::

   \mathrm{qfit} = \frac{\sum_p |D_p - M_p|}{\sum_p M_p},

summed over the fitting footprint for data :math:`D` and model :math:`M`.  A
well-fit isolated star sits near the noise floor; a blend, a saturated core or
a source sitting on structured emission does not.  Plotting it against
brightness separates the two regimes that the pipeline's vetting has to tell
apart, and the flag census alongside it says how much of the catalogue each
special-case path touched.

**Colour diagrams** (:func:`color_diagrams`) -- the closing sanity check.  A
colour--magnitude and a colour--colour diagram will show a photometric
systematic that no per-filter statistic reveals, because they are differential
between filters: a zero-point error, a saturation-correction discontinuity or
a filter-dependent background over-subtraction all appear as structure in the
colour that has no counterpart in any single band.
"""

import numpy as np

from jwst_gc_pipeline.diagnostics import loaders, style
from jwst_gc_pipeline.diagnostics.figures import FigureResult, save

# qfit above this is the pipeline's "poorly fit" regime (see the vetting code).
QFIT_WARN = 0.2


def photometric_precision(inv, outdir, max_sources=400000):
    """Fractional flux error against brightness, per filter, with 5-sigma depth."""
    style.use_style()
    filters = list(inv.measured_filters)
    if not filters:
        return None
    zps = loaders.photometric_zeropoints(inv.crossband_catalog, filters)
    fig, axes = style.panel_grid(len(filters), panel=(2.7, 2.3))
    colors = style.filter_colors(filters)
    depths = {}
    err_ratio = {}

    for ax, filt in zip(axes, filters):
        tbl = loaders.read_columns(
            inv.per_filter_catalogs[filt],
            ['flux', 'flux_err', 'flux_err_prop', 'nmatch'],
            label=f'{inv.name} {filt}')
        flux = loaders.column(tbl, 'flux')
        ferr = loaders.column(tbl, 'flux_err')
        fprop = loaders.column(tbl, 'flux_err_prop')
        mag, maglabel = loaders.magnitudes(flux, zps.get(filt))
        with np.errstate(divide='ignore', invalid='ignore'):
            frac = np.where(flux > 0, ferr / flux, np.nan)
            frac_prop = np.where(flux > 0, fprop / flux, np.nan)
        if mag.size > max_sources:
            pick = np.random.default_rng(0).choice(mag.size, max_sources,
                                                   replace=False)
            mag, frac, frac_prop = mag[pick], frac[pick], frac_prop[pick]

        good = np.isfinite(mag) & np.isfinite(frac) & (frac > 0)
        if good.sum() < 50:
            ax.text(0.5, 0.5, 'too few measured fluxes', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7)
            ax.set_title(filt.upper())
            continue
        ax.hexbin(mag[good], frac[good], bins='log', gridsize=45, mincnt=1,
                  cmap='Greys', yscale='log')
        centres, pct = style.running_percentiles(mag[good], frac[good])
        if centres.size:
            ax.plot(centres, pct[50], color=colors[filt], lw=1.4,
                    label='formal')
        good_p = np.isfinite(mag) & np.isfinite(frac_prop) & (frac_prop > 0)
        if good_p.sum() > 50:
            cp, pp = style.running_percentiles(mag[good_p], frac_prop[good_p])
            if cp.size:
                ax.plot(cp, pp[50], color=colors[filt], lw=1.2, ls='--',
                        label='propagated')
            ratio = np.nanmedian(frac_prop[good_p] / frac[good_p])
            err_ratio[filt] = float(ratio)
        ax.axhline(0.2, color='0.4', lw=0.8, ls=':')
        depth = _crossing(centres, pct[50], 0.2) if centres.size else np.nan
        depths[filt] = depth
        note = f'N={len(tbl):,}'
        if np.isfinite(depth):
            note += f'\n5$\\sigma$ at {depth:.2f}'
        if filt in err_ratio:
            note += f'\nprop/formal {err_ratio[filt]:.2f}'
        style.annotate(ax, note, loc='upper left')
        ax.set_yscale('log')
        ax.set_ylim(max(np.nanpercentile(frac[good], 0.2), 1e-4), 2)
        ax.set_title(filt.upper())
        ax.set_xlabel(maglabel)
        ax.set_ylabel(r'$\sigma_F/F$')
        ax.legend(loc='lower right', fontsize=5.5)

    fig.suptitle(f'{inv.name}: photometric precision and depth', fontsize=10,
                 y=1.005)
    fig.tight_layout()
    path = save(fig, outdir, 'D4_photometry_precision')
    caption = (
        f'Photometric precision of {inv.name}. Greyscale is source density; '
        'the solid line is the running median of the fitter\'s formal '
        r'uncertainty $\sigma_F/F$ and the dashed line the same quantity '
        'propagated from the per-exposure scatter. The dotted horizontal line '
        r'is $\sigma_F/F = 0.2$, and where the median crosses it is quoted as '
        r'the $5\sigma$ depth. A propagated-to-formal ratio above unity means '
        'the exposures disagree by more than the fit covariance predicts.')
    return FigureResult('D4_photometry_precision', path, caption, 'photometry',
                        dict(depth=depths, err_ratio=err_ratio))


def _crossing(x, y, level):
    """First *x* where *y* crosses *level* going up, by linear interpolation."""
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 3:
        return np.nan
    above = y >= level
    if not above.any() or above.all():
        return np.nan
    idx = int(np.argmax(above))
    if idx == 0:
        return np.nan
    x0, x1, y0, y1 = x[idx - 1], x[idx], y[idx - 1], y[idx]
    if y1 == y0:
        return float(x1)
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


def photometric_quality(inv, outdir, max_sources=400000):
    """PSF-fit residual against brightness, plus the special-case flag census."""
    style.use_style()
    filters = list(inv.measured_filters)
    if not filters:
        return None
    zps = loaders.photometric_zeropoints(inv.crossband_catalog, filters)
    fig, axes = style.panel_grid(len(filters) + 1, panel=(2.7, 2.3))
    colors = style.filter_colors(filters)

    flag_cols = ['is_saturated', 'replaced_saturated', 'satstar_gate_rejected',
                 'satclip_corrected']
    census = {}
    qfit_stats = {}

    for ax, filt in zip(axes, filters):
        tbl = loaders.read_columns(
            inv.per_filter_catalogs[filt],
            ['flux', 'qfit', 'cfit', 'group_size', 'flags'] + flag_cols,
            label=f'{inv.name} {filt}')
        flux = loaders.column(tbl, 'flux')
        qfit = loaders.column(tbl, 'qfit')
        mag, maglabel = loaders.magnitudes(flux, zps.get(filt))
        n = len(tbl)
        census[filt] = {c: int(np.nansum(loaders.column(tbl, c, fill=0) > 0))
                        for c in flag_cols}
        census[filt]['total'] = n

        if mag.size > max_sources:
            pick = np.random.default_rng(0).choice(mag.size, max_sources,
                                                   replace=False)
            mag, qfit = mag[pick], qfit[pick]
        good = np.isfinite(mag) & np.isfinite(qfit) & (qfit > 0)
        if good.sum() < 50:
            ax.text(0.5, 0.5, 'no qfit', ha='center', va='center',
                    transform=ax.transAxes, fontsize=7)
            ax.set_title(filt.upper())
            continue
        # qfit has a long tail in both directions -- a perfectly-fit isolated
        # star and a catastrophically blended one differ by many decades -- so
        # the axis is set from percentiles, not from the extremes.
        ylo = max(np.percentile(qfit[good], 0.5), 1e-4)
        yhi = max(np.percentile(qfit[good], 99.5), ylo * 10)
        ax.hexbin(mag[good], qfit[good], bins='log', gridsize=45, mincnt=1,
                  cmap='Greys', yscale='log',
                  extent=(*style.robust_range(mag[good], 0.5, 99.5),
                          np.log10(ylo), np.log10(yhi)))
        centres, pct = style.running_percentiles(mag[good], qfit[good])
        if centres.size:
            ax.plot(centres, pct[50], color=colors[filt], lw=1.4)
        ax.axhline(QFIT_WARN, color='0.4', ls=':', lw=0.8)
        frac_bad = float(np.mean(qfit[good] > QFIT_WARN))
        qfit_stats[filt] = dict(median=float(np.median(qfit[good])),
                                frac_above_warn=frac_bad)
        style.annotate(ax, f'median {np.median(qfit[good]):.3f}\n'
                            f'{100 * frac_bad:.1f}% > {QFIT_WARN}', loc='upper left')
        ax.set_yscale('log')
        # After set_yscale, which re-autoscales -- otherwise the decades-long
        # tail sets the axis and the body of the distribution is a thin band.
        ax.set_ylim(ylo, yhi)
        ax.set_title(filt.upper())
        ax.set_xlabel(maglabel)
        ax.set_ylabel('qfit')

    _draw_census(axes[-1], census, flag_cols)
    fig.suptitle(f'{inv.name}: PSF fit quality and special-case census',
                 fontsize=10, y=1.005)
    fig.tight_layout()
    path = save(fig, outdir, 'D5_photometry_quality')
    caption = (
        f'PSF-fit quality for {inv.name}. Per filter, the normalised fit '
        'residual qfit against brightness (greyscale: source density; line: '
        f'running median; dotted: the qfit\\,=\\,{QFIT_WARN} vetting '
        'threshold). The final panel is the fraction of each filter\'s '
        'catalogue touched by the saturation-handling paths: flagged '
        'saturated, core-replaced, rejected by the saturated-star gate, and '
        'clip-corrected.')
    return FigureResult('D5_photometry_quality', path, caption, 'photometry',
                        dict(qfit=qfit_stats, census=census))


def _draw_census(ax, census, flag_cols):
    filters = list(census)
    if not filters:
        ax.set_visible(False)
        return
    width = 0.8 / max(len(flag_cols), 1)
    xs = np.arange(len(filters))
    for k, col in enumerate(flag_cols):
        fracs = [100.0 * census[f].get(col, 0) / max(census[f]['total'], 1)
                 for f in filters]
        ax.bar(xs + k * width - 0.4 + width / 2, fracs, width=width,
               label=col.replace('_', ' '))
    ax.set_xticks(xs)
    ax.set_xticklabels([f.upper() for f in filters], rotation=90, fontsize=5.5)
    ax.set_ylabel('per cent of catalogue')
    ax.set_title('saturation-path census')
    # Below the axes: with a dozen filters the bars fill the panel and an
    # inset legend sits on top of the data it is labelling.
    ax.legend(fontsize=5.0, ncol=2, loc='upper center',
              bbox_to_anchor=(0.5, -0.28))
    ax.set_yscale('symlog', linthresh=0.01)


def color_diagrams(inv, outdir, max_sources=300000):
    """Colour--magnitude and colour--colour diagrams from the cross-band merge."""
    style.use_style()
    if not inv.has_crossband:
        return None
    filters = [f for f in inv.filters]
    wanted = []
    for filt in filters:
        wanted += [f'mag_vega_{filt}', f'emag_ab_{filt}', f'qfit_{filt}']
    tbl = loaders.read_columns(inv.crossband_catalog, wanted,
                               label=f'{inv.name} cross-band')
    have = [f for f in filters if f'mag_vega_{f}' in tbl.colnames]
    have.sort(key=style.filter_wavelength)
    if len(have) < 2:
        return None
    mags = {f: loaders.column(tbl, f'mag_vega_{f}') for f in have}
    if len(tbl) > max_sources:
        pick = np.random.default_rng(0).choice(len(tbl), max_sources, replace=False)
        mags = {f: v[pick] for f, v in mags.items()}

    # Bluest and reddest available bands make the longest colour baseline; a
    # third band in between gives the colour-colour plane.
    blue, red = have[0], have[-1]
    panels = [('cmd', blue, red)]
    if len(have) >= 4:
        mid1, mid2 = have[1], have[-2]
        panels.append(('ccd', (blue, mid1), (mid2, red)))
    elif len(have) == 3:
        panels.append(('ccd', (have[0], have[1]), (have[1], have[2])))

    fig, axes = style.panel_grid(len(panels), ncols=len(panels), panel=(3.2, 3.0))
    stats = {}
    for ax, spec in zip(axes, panels):
        if spec[0] == 'cmd':
            _b, _r = spec[1], spec[2]
            color = mags[_b] - mags[_r]
            ax.hexbin(color, mags[_r], bins='log', gridsize=70, mincnt=1,
                      cmap='magma_r',
                      extent=(*style.robust_range(color, 1, 99),
                              *style.robust_range(mags[_r], 0.2, 99.8)))
            ax.invert_yaxis()
            ax.set_xlabel(f'{_b.upper()} $-$ {_r.upper()} (Vega)')
            ax.set_ylabel(f'{_r.upper()} (Vega)')
            ax.set_title('colour--magnitude')
            good = np.isfinite(color)
            stats['cmd'] = dict(bands=[_b, _r], n=int(good.sum()),
                                median_color=float(np.nanmedian(color)))
        else:
            (b1, r1), (b2, r2) = spec[1], spec[2]
            c1 = mags[b1] - mags[r1]
            c2 = mags[b2] - mags[r2]
            ax.hexbin(c1, c2, bins='log', gridsize=70, mincnt=1, cmap='magma_r',
                      extent=(*style.robust_range(c1, 1, 99),
                              *style.robust_range(c2, 1, 99)))
            ax.set_xlabel(f'{b1.upper()} $-$ {r1.upper()}')
            ax.set_ylabel(f'{b2.upper()} $-$ {r2.upper()}')
            ax.set_title('colour--colour')
            stats['ccd'] = dict(bands=[b1, r1, b2, r2],
                                n=int(np.isfinite(c1 * c2).sum()))

    fig.suptitle(f'{inv.name}: colour diagrams '
                 f'(m{inv.crossband_stage} cross-band merge)', fontsize=10,
                 y=1.005)
    fig.tight_layout()
    path = save(fig, outdir, 'D8_color_diagrams')
    caption = (
        f'Colour diagrams for {inv.name}, from the stage-m{inv.crossband_stage} '
        'cross-band merge, in Vega magnitudes. These are differential between '
        'filters and so respond to systematics that per-filter statistics '
        'cannot see: a zero-point error shifts a sequence bodily, a '
        'saturation-correction discontinuity breaks it at a specific '
        'magnitude, and a filter-dependent background error tilts it.')
    return FigureResult('D8_color_diagrams', path, caption, 'photometry', stats)
