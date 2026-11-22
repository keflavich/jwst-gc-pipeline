"""Opt-in: keep a bright star that ``combine_singleframe`` would merge into a faint row.

``combine_singleframe`` builds its base catalog frame by frame (Loop 1): a
frame's row starts a new base row only when it lies more than ``min_offset``
(0.1") from every base row so far.  Phase 1 then hands each frame's rows to the
base rows by mutual nearest neighbour inside ``max_offset``, and each base row's
flux is the inverse-variance average of the rows it receives.  When the first
frame that covers a star holds only a faint row within ``min_offset`` of it
(the star itself is missing from that frame's catalog), the faint row takes the
base row, and the star's own rows go one of two ways:

* a frame that also holds a faint row of its own near the base row loses the
  mutual test, because that faint row is nearer to the base row.  Every such
  frame drops the star's row, and the star leaves the merged catalog
  (gc-treasury 10678 o045 F212N, mosaic pixel (8753, 3176): every one of the
  six frames holds a faint row of 108-244 within 0.012" of the base row, the
  first of them seeded it, and the five later frames also fit the star at
  ``flux_fit`` 1.50e4-1.57e4, 0.077-0.080" from the base row; merged flux 167);
* a frame without a faint row passes the test, and the base row averages the
  bright row with the faint ones, weighted by 1 / ``flux_err``^2, so the faint
  rows set the merged flux (mosaic pixel (1377, 4559): rows of 7748, 0.097" from
  the base row, and 357; merged flux 1775).

With ``MERGE_PAIR_RESCUE_RATIO`` above 1, a frame row also starts a base row
when it lies farther than the split-row radius (``split_row_radius``) from its
nearest base row, its flux is at least ``MERGE_PAIR_RESCUE_FLUX_MIN`` and at
least ``MERGE_PAIR_RESCUE_RATIO`` times the flux of the row that seeded that
base row.  Later rows of the star then find the new base row as their nearest
one, and the faint rows keep theirs.  A row inside the split-row radius never
starts one, because ``fold_split_rows`` would merge the two base rows again.
Every other row is placed as before.

It is off unless ``MERGE_PAIR_RESCUE_RATIO`` is above 0.
"""
import os

import numpy as np

RATIO_ENV = 'MERGE_PAIR_RESCUE_RATIO'
FLUX_MIN_ENV = 'MERGE_PAIR_RESCUE_FLUX_MIN'


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
    """``(ratio, flux_min)``, or None when the rescue is off.

    Off when ``MERGE_PAIR_RESCUE_RATIO`` is unset, blank or 0; the flux floor
    is not read then.  A ratio that is not above 1 would start base rows for
    rows no brighter than the row that seeded theirs, and a negative floor
    means nothing, so both raise ``ValueError`` naming the variable.
    """
    ratio = _read_float(RATIO_ENV, 0.0)
    if ratio == 0.0:
        return None
    if ratio <= 1.0:
        raise ValueError(f"{RATIO_ENV}={ratio:g} must be above 1 (the flux ratio "
                         f"to the row that seeded the nearest base row), or 0 for off")
    flux_min = _read_float(FLUX_MIN_ENV, 0.0)
    if flux_min < 0.0:
        raise ValueError(f"{FLUX_MIN_ENV}={flux_min:g} must be 0 or above")
    return ratio, flux_min


def rescue_rows(sep, flux, seed_flux, split_radius, ratio, flux_min):
    """Rows of one frame that start a base row although they lie within the dedup radius of one.

    Parameters
    ----------
    sep : `~astropy.coordinates.Angle`
        Distance of each row to its nearest base row.
    flux, seed_flux : `numpy.ndarray`
        Flux of each row, and flux of the row that seeded its nearest base row.
    split_radius : `~astropy.units.Quantity`
        A row this close to its nearest base row never qualifies.
    ratio, flux_min : float
        A row qualifies when its flux is positive, at least ``flux_min`` and at
        least ``ratio`` times ``seed_flux``.

    Returns
    -------
    `numpy.ndarray` of bool
        True for the qualifying rows.  A NaN flux or seed flux never qualifies.
    """
    with np.errstate(invalid='ignore'):
        return ((sep > split_radius) & np.isfinite(flux) & np.isfinite(seed_flux)
                & (flux > 0) & (flux >= flux_min) & (flux >= ratio * seed_flux))
