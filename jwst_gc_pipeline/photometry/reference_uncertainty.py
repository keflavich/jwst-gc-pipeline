"""Per-star predicted reference-position uncertainty for a Gaia+VIRAC2 refcat.

Context (issue #965 item 1, following #957): the m2 same-star region map flags
45" cells against the Gaia+VIRAC2 reference on 15 mas / 3-sigma residuals, and
five gc-treasury tiles (#965: o075/o077/o080/o084/o087) refuse on 1-2 cells
that sit only slightly above the tolerance the reference's own per-star scatter
already produces.  VIRAC2 (II/387) carries a per-star position error
(``e_RAJ2000``/``e_DEJ2000``, mas) and a per-star proper-motion error
(``e_pmRA``/``e_pmDE``, mas/yr) that the refcat previously discarded; propagated
to the observation epoch these predict which reference stars are too uncertain
to test a 15 mas cell against in the first place.  Gaia DR3 carries the same
pair (``e_RA_ICRS``/``e_DE_ICRS`` and ``e_pmRA``/``e_pmDE`` from the VizieR
I/355/gaiadr3 mirror, or ``ra_error``/``dec_error``/``pmra_error``/
``pmdec_error`` from the ESA TAP) and is propagated the same way from its own
2016.0 epoch.

This module holds only the quadrature-sum math and the column names, so it is
testable with plain arrays and importable from the refcat builder
(``reduction/build_gaia_virac2_refcat_byquery.py``), the loader
(``photometry/visit_consensus.load_reference_catalog``) and the region-map cut
(``photometry/astrometry_offsets.same_star_region_map`` /
``local_residual_map``) without pulling in Vizier, FITS I/O or the region-map
machinery.

This is NOT a NN-median astrometric measurement (CLAUDE.md ASTROMETRY RULE #1):
it never estimates an offset, only a per-star PREDICTED uncertainty used to cut
or weight pairs that a sanctioned estimator (``measure_offset`` /
``local_residual_map`` / ``same_star_region_map``) has already formed.
"""
import numpy as np

#: Column written to the refcat: the combined RA+Dec predicted 1-sigma
#: position uncertainty at the refcat's own observation epoch (mas).
SIGMA_PRED_COLUMN = 'sigma_pred_mas'
#: Per-axis raw components, kept alongside the combined column so a consumer
#: can recompute at a DIFFERENT epoch (e.g. a re-tie against an older catalog)
#: without re-querying VizieR.
SIGMA_POS_RA_COLUMN = 'sigma_pos_ra_mas'
SIGMA_POS_DEC_COLUMN = 'sigma_pos_dec_mas'
SIGMA_PM_RA_COLUMN = 'sigma_pm_ra_masyr'
SIGMA_PM_DEC_COLUMN = 'sigma_pm_dec_masyr'


def predicted_axis_sigma_mas(sigma_pos_mas, sigma_pm_mas_yr, dt_yr):
    """``sqrt(sigma_pos^2 + (dt * sigma_pm)^2)`` for one axis (RA or Dec).

    All three arguments broadcast together.  A non-finite entry in either
    input (VizieR leaves a blank field as NaN for e.g. a star fit with only
    ``Nep`` too small for a PM solution) propagates to a non-finite result --
    it is not silently treated as zero uncertainty, which would make a
    poorly-measured star look perfectly known.
    """
    sigma_pos_mas = np.asarray(sigma_pos_mas, dtype=float)
    sigma_pm_mas_yr = np.asarray(sigma_pm_mas_yr, dtype=float)
    return np.sqrt(sigma_pos_mas ** 2 + (float(dt_yr) * sigma_pm_mas_yr) ** 2)


def combined_sigma_pred_mas(sigma_ra_mas, sigma_dec_mas):
    """RA/Dec axis sigmas combined into one predicted-offset sigma (mas).

    Matched-pair residuals in this codebase are reported as a single
    ``hypot(dra, ddec)`` offset (``local_residual_map``, ``measure_offset``),
    so the predicted sigma that gates or weights them is combined the same
    way rather than kept as a 2-vector.
    """
    return np.hypot(np.asarray(sigma_ra_mas, dtype=float),
                    np.asarray(sigma_dec_mas, dtype=float))


def sigma_pred_mas(sigma_pos_ra_mas, sigma_pos_dec_mas,
                   sigma_pm_ra_masyr, sigma_pm_dec_masyr, dt_yr):
    """One call: per-axis position + PM errors, propagation baseline ``dt_yr``,
    combined RA/Dec -> the single ``sigma_pred_mas`` column value.
    """
    sra = predicted_axis_sigma_mas(sigma_pos_ra_mas, sigma_pm_ra_masyr, dt_yr)
    sdec = predicted_axis_sigma_mas(sigma_pos_dec_mas, sigma_pm_dec_masyr, dt_yr)
    return combined_sigma_pred_mas(sra, sdec)


def weighted_median(values, weights):
    """A robust weighted median: the value at which the cumulative weight
    first reaches half the total, with a plain-median TIE-BREAK when it lands
    EXACTLY on that boundary -- average the two straddling values, the way
    ``np.median`` averages the two middle elements of an even-length array.
    Uniform weights therefore reproduce ``np.median`` exactly for both odd and
    even counts: ``weighted_median([0, 10], [1, 1]) == 5.0``, not the lower
    value 0.0 an unbroken "first index reaching half" rule would return.

    Falls back to the plain median when every weight is non-finite or
    non-positive (so a caller can pass an all-NaN-sigma weight array -- the
    "no per-star sigma known" case -- and get today's unweighted behaviour
    rather than a crash or a silent NaN).

    Not a NN-median astrometric measurement: this operates on residuals a
    sanctioned matched-pair estimator already produced, not on raw
    nearest-neighbour separations.
    """
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    finite = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not finite.any():
        return float(np.median(v)) if v.size else float('nan')
    v, w = v[finite], w[finite]
    order = np.argsort(v)
    v, w = v[order], w[order]
    cum = np.cumsum(w)
    total = cum[-1]
    half = 0.5 * total
    idx = int(np.searchsorted(cum, half))
    idx = min(idx, len(v) - 1)
    if idx + 1 < len(v) and np.isclose(cum[idx], half, rtol=1e-9, atol=1e-9 * total):
        return float(0.5 * (v[idx] + v[idx + 1]))
    return float(v[idx])


def inverse_variance_weights(sigma_mas, floor_mas=1.0):
    """``1 / sigma^2`` weights for ``weighted_median``, floored at
    ``floor_mas`` so a suspiciously tiny reported sigma cannot dominate a cell
    outright; non-finite or non-positive sigma gets weight 0 (excluded, not
    infinitely trusted).
    """
    s = np.asarray(sigma_mas, dtype=float)
    s = np.where(np.isfinite(s) & (s > 0), np.maximum(s, float(floor_mas)),
                np.nan)
    w = np.zeros_like(s)
    ok = np.isfinite(s)
    w[ok] = 1.0 / (s[ok] ** 2)
    return w


def weights_with_unknown_fallback(sigma_mas, floor_mas=1.0):
    """Inverse-variance weights, with an UNKNOWN sigma (NaN -- a star with no
    error columns, e.g. a Gaia row queried before this refcat carried them)
    given the MEDIAN weight of the stars in the same call whose sigma IS
    known, rather than excluded outright (``inverse_variance_weights`` gives
    it 0).  A cell mixing a few sigma-less legacy rows with mostly-known ones
    should not have those rows silently vanish from the statistic; treating
    them as "typical" is the closest a weighted median can come to today's
    unweighted behaviour for exactly those rows.

    When NOTHING in the input has a known sigma, every weight is 1 (uniform),
    which makes ``weighted_median`` degenerate to the plain median -- the
    correct fallback for an old refcat with no sigma column at all.
    """
    w = inverse_variance_weights(sigma_mas, floor_mas=floor_mas)
    unknown = ~np.isfinite(np.asarray(sigma_mas, dtype=float))
    if not unknown.any():
        return w
    known_positive = w[~unknown & (w > 0)]
    fallback = float(np.median(known_positive)) if known_positive.size else 1.0
    w = w.copy()
    w[unknown] = fallback
    return w
