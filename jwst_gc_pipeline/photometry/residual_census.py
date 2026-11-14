"""Unsubtracted-star fraction of a residual image, binned by magnitude.

Issue #972 measured how many weakly saturated F480M stars (12.0-12.5 mag) were
left in the residual image after the satstar and daophot channels: 30% at o111
m6 and 27% at o132 m7, against 2.2-3.6% in the end-to-end test of the fixes.
The script that produced those figures was not kept, so the figure could not be
repeated on a later release.  This module holds the estimator so the same
number can be re-measured and compared.

The estimator, for each star-like peak of the pre-subtraction image:

    res_frac = residual extreme in a small box / pre-subtraction peak in that box

where the residual extreme is the value of largest magnitude, so that an
over-subtracted core reads negative.

``res_frac`` near 0 is a clean subtraction, near 1 an untouched star, and
negative an over-subtraction.  A star counts as unsubtracted when
``res_frac > UNSUB_THRESHOLD`` and as over-subtracted when
``res_frac < OVERSUB_THRESHOLD``.  The functions take plain arrays, so the
caller chooses the peak list (catalogue positions or a peak finder) and the
magnitude column.
"""
import numpy as np

UNSUB_THRESHOLD = 0.5
OVERSUB_THRESHOLD = -0.2
DEFAULT_BOX_HALF = 2


def box_peak(image, x, y, half=DEFAULT_BOX_HALF, signed_extreme=False):
    """Maximum of ``image`` in a (2*half+1)^2 box about each (x, y).

    ``x`` and ``y`` are 0-indexed pixel coordinates.  The box is clipped at the
    image edge.  A box with no finite pixel, or a centre off the image, gives
    NaN.  With ``signed_extreme`` the value of largest magnitude is returned
    with its sign, so an over-subtracted core reads negative.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    ny, nx = image.shape
    out = np.full(x.shape, np.nan)
    for i, (xi, yi) in enumerate(zip(x, y)):
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        xc, yc = int(round(xi)), int(round(yi))
        if xc < 0 or yc < 0 or xc >= nx or yc >= ny:
            continue
        cut = image[max(yc - half, 0):yc + half + 1,
                    max(xc - half, 0):xc + half + 1]
        if np.isfinite(cut).any():
            if signed_extreme:
                out[i] = cut.flat[np.nanargmax(np.abs(cut))]
            else:
                out[i] = np.nanmax(cut)
    return out


def residual_fraction(pre, residual, x, y, half=DEFAULT_BOX_HALF):
    """Fraction of the pre-subtraction peak left in the residual at (x, y).

    Peaks at or below zero in the pre-subtraction image give NaN, since the
    ratio carries no meaning there.
    """
    pre_peak = box_peak(pre, x, y, half)
    res_peak = box_peak(residual, x, y, half, signed_extreme=True)
    with np.errstate(invalid='ignore', divide='ignore'):
        frac = np.where(pre_peak > 0, res_peak / pre_peak, np.nan)
    return frac


def unsubtracted_fraction_by_bin(mag, res_frac, edges,
                                 unsub_threshold=UNSUB_THRESHOLD,
                                 oversub_threshold=OVERSUB_THRESHOLD):
    """Count unsubtracted and over-subtracted stars per magnitude bin.

    ``edges`` are the bin edges (for example ``np.arange(11.5, 14.76, 0.5)``);
    a star falls in ``[edges[i], edges[i+1])``.  Stars with a non-finite
    magnitude or ``res_frac`` are left out of every count.  Returns a list of
    dicts with keys ``lo``, ``hi``, ``n``, ``n_unsub``, ``frac_unsub``
    (NaN for an empty bin), ``n_over`` and ``frac_over``.
    """
    mag = np.asarray(mag, dtype=float)
    res_frac = np.asarray(res_frac, dtype=float)
    good = np.isfinite(mag) & np.isfinite(res_frac)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = good & (mag >= lo) & (mag < hi)
        n = int(sel.sum())
        n_unsub = int((res_frac[sel] > unsub_threshold).sum())
        n_over = int((res_frac[sel] < oversub_threshold).sum())
        rows.append(dict(lo=float(lo), hi=float(hi), n=n,
                         n_unsub=n_unsub,
                         frac_unsub=n_unsub / n if n else np.nan,
                         n_over=n_over,
                         frac_over=n_over / n if n else np.nan))
    return rows


def bin_edges(lo, hi, step):
    """Bin edges from ``lo`` to ``hi`` in steps of ``step``.

    The last edge is always ``hi``; when ``hi - lo`` is not a whole number of
    steps the final bin is narrower than ``step`` rather than dropped.
    """
    n_full = int(np.floor((hi - lo) / step + 1e-9))
    edges = lo + step * np.arange(n_full + 1)
    if hi - edges[-1] > 1e-9:
        edges = np.append(edges, hi)
    else:
        edges[-1] = hi
    return edges


def format_table(rows):
    """Plain-text table of the rows from ``unsubtracted_fraction_by_bin``."""
    lines = ['  mag bin        n  unsub  unsub%  over  over%']
    for r in rows:
        lines.append('{lo:5.1f}-{hi:5.1f} {n:8d} {n_unsub:6d} {pu:7.1f} '
                     '{n_over:5d} {po:6.1f}'.format(
                         pu=100 * r['frac_unsub'], po=100 * r['frac_over'], **r))
    return '\n'.join(lines)
