"""Shared plotting style and small statistics helpers for the write-ups.

Everything here exists so the seventeen field documents look like one series
rather than seventeen one-off scripts: one rcParams block, one panel-grid
helper, one running-percentile routine, one colour per filter.
"""

import numpy as np

# A vector format, because these go into a paper.
FIGURE_FORMAT = 'pdf'

RCPARAMS = {
    'figure.dpi': 120,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'legend.frameon': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'axes.axisbelow': True,
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
}


def use_style():
    """Apply :data:`RCPARAMS` to the current matplotlib session."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update(RCPARAMS)
    return plt


def filter_wavelength(filtername):
    """Pivot wavelength in microns, parsed from the filter name."""
    digits = ''.join(c for c in filtername if c.isdigit())
    return float(digits[:3]) / 100.0 * (10.0 if len(digits) > 3 else 1.0)


def filter_colors(filternames):
    """A wavelength-ordered colour per filter, blue -> red."""
    plt = use_style()
    order = sorted(filternames, key=filter_wavelength)
    cmap = plt.get_cmap('turbo')
    n = max(len(order) - 1, 1)
    return {f: cmap(0.08 + 0.84 * i / n) for i, f in enumerate(order)}


def panel_grid(n, ncols=None, panel=(2.5, 2.2), sharex=False, sharey=False):
    """A figure with *n* panels laid out in a near-square grid.

    Returns ``(fig, axes)`` with ``axes`` a flat list of exactly *n* live axes;
    unused cells are removed rather than left blank.
    """
    plt = use_style()
    if ncols is None:
        ncols = int(np.ceil(np.sqrt(n)))
        ncols = min(max(ncols, 1), 4)
    nrows = int(np.ceil(n / ncols))
    fig, axgrid = plt.subplots(nrows, ncols, sharex=sharex, sharey=sharey,
                               figsize=(panel[0] * ncols, panel[1] * nrows),
                               squeeze=False)
    flat = axgrid.ravel().tolist()
    for ax in flat[n:]:
        fig.delaxes(ax)
    return fig, flat[:n]


def running_percentiles(x, y, bins=20, percentiles=(16, 50, 84), min_count=10):
    """Percentiles of *y* in bins of *x*.

    Returns ``(centres, {percentile: values})``; bins holding fewer than
    *min_count* finite points are NaN so a sparse tail cannot masquerade as a
    measured trend.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < min_count:
        return np.array([]), {p: np.array([]) for p in percentiles}
    if np.isscalar(bins):
        edges = np.linspace(np.nanpercentile(x, 0.5),
                            np.nanpercentile(x, 99.5), int(bins) + 1)
    else:
        edges = np.asarray(bins, dtype=float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    idx = np.digitize(x, edges) - 1
    out = {p: np.full(centres.size, np.nan) for p in percentiles}
    for i in range(centres.size):
        sel = idx == i
        if sel.sum() < min_count:
            continue
        for p in percentiles:
            out[p][i] = np.percentile(y[sel], p)
    return centres, out


def robust_range(values, lo=1.0, hi=99.0, pad=0.05):
    """Percentile range of *values*, padded, immune to a few wild outliers."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)
    a, b = np.percentile(values, [lo, hi])
    if not np.isfinite(a) or not np.isfinite(b) or b <= a:
        return (float(np.min(values)), float(np.max(values)) or 1.0)
    span = b - a
    return (a - pad * span, b + pad * span)


def binned_median_image(x, y, values, nbins=64, min_count=3, bounds=None):
    """Median of *values* on a regular ``nbins x nbins`` grid in (x, y).

    Returns ``(image, extent)``; cells with fewer than *min_count* points are
    NaN.  This is the workhorse behind every spatial map in the write-ups: it
    is what turns a per-source column into something comparable with a mosaic.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(values)
    x, y, values = x[good], y[good], values[good]
    if x.size == 0:
        return np.full((nbins, nbins), np.nan), (0, 1, 0, 1)
    if bounds is None:
        x0, x1 = np.nanpercentile(x, [0.05, 99.95])
        y0, y1 = np.nanpercentile(y, [0.05, 99.95])
    else:
        x0, x1, y0, y1 = bounds
    xe = np.linspace(x0, x1, nbins + 1)
    ye = np.linspace(y0, y1, nbins + 1)
    ix = np.clip(np.digitize(x, xe) - 1, 0, nbins - 1)
    iy = np.clip(np.digitize(y, ye) - 1, 0, nbins - 1)
    flat = iy * nbins + ix
    order = np.argsort(flat, kind='stable')
    flat, values = flat[order], values[order]
    starts = np.searchsorted(flat, np.arange(nbins * nbins), side='left')
    ends = np.searchsorted(flat, np.arange(nbins * nbins), side='right')
    img = np.full(nbins * nbins, np.nan)
    for cell, (s, e) in enumerate(zip(starts, ends)):
        if e - s >= min_count:
            img[cell] = np.median(values[s:e])
    return img.reshape(nbins, nbins), (x0, x1, y0, y1)


def spearman(x, y, max_points=200000, seed=0):
    """Spearman rank correlation of the jointly finite entries of *x*, *y*.

    Sub-samples above *max_points* -- the rank transform is the expensive part
    and a 2e5 sample already pins rho to well under 0.01.
    """
    from scipy import stats
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 10:
        return np.nan, np.nan
    if x.size > max_points:
        rng = np.random.default_rng(seed)
        pick = rng.choice(x.size, max_points, replace=False)
        x, y = x[pick], y[pick]
    res = stats.spearmanr(x, y)
    return float(res.statistic), float(res.pvalue)


def annotate(ax, text, loc='upper left', **kwargs):
    """Small boxed annotation in axes coordinates."""
    xy = {'upper left': (0.03, 0.97), 'upper right': (0.97, 0.97),
          'lower left': (0.03, 0.03), 'lower right': (0.97, 0.03)}[loc]
    ha = 'left' if 'left' in loc else 'right'
    va = 'top' if 'upper' in loc else 'bottom'
    kwargs.setdefault('fontsize', 6.5)
    ax.text(xy[0], xy[1], text, transform=ax.transAxes, ha=ha, va=va,
            bbox=dict(boxstyle='round,pad=0.25', fc='w', ec='0.7', alpha=0.85),
            **kwargs)
