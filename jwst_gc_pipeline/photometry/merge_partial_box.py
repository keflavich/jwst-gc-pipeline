"""Opt-in: a per-frame fit that used few of its pixels does not vote on a star that a full-box fit measured.

``combine_singleframe`` averages the per-frame rows of a star with the weights
1 / ``flux_err``^2.  The per-frame fit (``fit_shape=(5, 5)``, 25 pixels) loses
the pixels that are masked, saturated or off the image, and a row fitted to a
part of its box comes back with a smaller ``flux_err`` than a full-box row
whenever its flux is smaller, because the error grows with the flux.  The
inverse-variance mean then follows the row with the smaller flux.  With two
rows the MAD clip of ``combine_singleframe`` cannot reject either of them, so
nothing stops the partial row.

gc-treasury 10678 o130 F212N, mosaic pixel (5065, 68): the full-box row has
``flux_fit`` 14433.84, ``flux_err`` 64.715 and ``n_pixels_fit`` 25.  The other
row has ``n_pixels_fit`` 10 of 25, ``flux_fit`` 1295.42, ``flux_err`` 31.101,
reduced chi^2 71.8, and lies 2.96 pixels (0.093") from the full-box row.  Its
weight is 4.3 times that of the full-box row, so the production merged flux is
3761 against 14335 in the vetted catalogue of the previous merge step (m6), and
the merged position is the mean of the two rows, 0.046" (1.5 pixels) from the
full-box row.  The mean position is not weighted (``pos_weights``), so a
partial row that is far from the star moves the position as much as a full-box
row does.

With ``MERGE_PARTIAL_BOX_FRAC_MIN`` above 0, a row whose ``n_pixels_fit`` is
below that fraction of the box is left out of the flux and position averages
of its star (and of the other averaged columns: they share the flux weights)
when the star has at least one row that is kept by the sigma clip, has a
finite positive flux error and is at or above the fraction.  The box is the
largest ``n_pixels_fit`` of the rows of the call.  A star whose kept rows are
all below the fraction is averaged as before, a row without a finite positive
flux error or without ``n_pixels_fit`` is never left out and never serves as
the full-box row, and ``nmatch_good`` still counts every kept row.  With ``MERGE_PARTIAL_BOX_ZMAX`` above 0 only the rows that
disagree with the rest of their star are left out: those whose flux differs from
the inverse-variance mean of the star's rows at or above the fraction by more
than that many standard deviations of the difference.

It is off unless ``MERGE_PARTIAL_BOX_FRAC_MIN`` is above 0.
"""
import os

import numpy as np

FRAC_MIN_ENV = 'MERGE_PARTIAL_BOX_FRAC_MIN'
ZMAX_ENV = 'MERGE_PARTIAL_BOX_ZMAX'

#: Stars per block of the reductions below; bounds the float64 temporaries.
_BLOCK = 65536


def _read_float(name, default):
    """``name`` as a finite float, ``default`` when unset or blank."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a number") from exc
    if not np.isfinite(value):
        raise ValueError(f"{name}={raw!r} is not finite")
    return value


def settings():
    """``(frac_min, zmax)``, or None when the switch is off.

    Off when ``MERGE_PARTIAL_BOX_FRAC_MIN`` is unset, blank or 0; the other
    variable is not read then.  ``frac_min`` must be at most 1 (a row cannot
    use more than the box), and ``zmax`` 0 or above; ``zmax`` is None when it
    is unset, blank or 0, which leaves out every row below ``frac_min``.  A
    value outside those ranges raises ``ValueError`` naming the variable.
    """
    frac_min = _read_float(FRAC_MIN_ENV, 0.0)
    if frac_min == 0.0:
        return None
    if not 0.0 < frac_min <= 1.0:
        raise ValueError(f"{FRAC_MIN_ENV}={frac_min:g} must be in (0, 1] (the fraction of the "
                         f"fit box a row used), or 0 for off")
    zmax = _read_float(ZMAX_ENV, 0.0)
    if zmax < 0.0:
        raise ValueError(f"{ZMAX_ENV}={zmax:g} must be 0 (every partial row) or above")
    return frac_min, (zmax if zmax > 0.0 else None)


def partial_rows(n_fit, usable, flux, flux_err, frac_min, zmax=None):
    """Rows to leave out of the averages of their star.

    Parameters
    ----------
    n_fit : `numpy.ndarray`, shape (n_src, n_frames)
        ``n_pixels_fit`` of each row, NaN where the star has no row in the
        frame or the frame has no such column.
    usable : `numpy.ndarray` of bool, same shape
        Rows kept by the sigma clip.
    flux, flux_err : `numpy.ndarray`, same shape
        Flux and flux error of each row.
    frac_min : float
        A row below this fraction of the box (the largest ``n_fit``) is partial.
    zmax : float, optional
        Leave out only the partial rows whose flux differs from the
        inverse-variance mean of the star's other usable rows (those at or above
        ``frac_min`` of the box) by more than ``zmax`` times the quadrature sum
        of the two errors.  Stars without such a row keep all their rows.

    Returns
    -------
    drop : `numpy.ndarray` of bool, same shape
        True for the rows to leave out; a star never loses all its usable rows.
    box : float
        The fit box, NaN when ``n_fit`` has no finite entry.
    """
    n_fit = np.asarray(n_fit)
    drop = np.zeros(n_fit.shape, dtype=bool)
    if not np.isfinite(n_fit).any():
        return drop, np.nan
    box = float(np.nanmax(n_fit))
    for lo in range(0, n_fit.shape[0], _BLOCK):
        sl = slice(lo, lo + _BLOCK)
        nf = n_fit[sl].astype('float64')
        fl = np.asarray(flux[sl], dtype='float64')
        er = np.asarray(flux_err[sl], dtype='float64')
        with np.errstate(invalid='ignore', divide='ignore'):
            ok = (np.asarray(usable[sl], dtype=bool) & np.isfinite(nf) & np.isfinite(fl)
                  & np.isfinite(er) & (er > 0))
            below = ok & (nf < frac_min * box)
            above = ok & (nf >= frac_min * box)
            if zmax is None:
                drop[sl] = below & above.any(axis=1)[:, None]
                continue
            w = np.where(above, 1.0 / er**2, 0.0)
            sw = w.sum(axis=1)
            mean = np.where(sw > 0, (np.where(above, fl, 0.0) * w).sum(axis=1) / sw, np.nan)
            sig = np.where(sw > 0, 1.0 / np.sqrt(sw), np.nan)
            z = (fl - mean[:, None]) / np.sqrt(er**2 + sig[:, None]**2)
            drop[sl] = below & (sw > 0)[:, None] & (np.abs(z) > zmax)
    return drop, box
