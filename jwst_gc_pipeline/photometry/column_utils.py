"""Table column-convention resolvers shared across the photometry pipeline.

Different producers (photutils DAOStarFinder 2.x vs >=3.0, PSFPhotometry,
seed/satstar tables) emit source positions under different column names
(``xcentroid``/``x_centroid``/``x_fit``/``x_init``/``x``).  These small,
dependency-light helpers pick the first available convention so callers don't
re-implement the candidate list.

Factored out of ``catalog_long.py`` (bloat refactor); that module
imports these names so there is a single source of truth and ``_L.<name>``
access keeps working unchanged.  Covered by test_seed_skycoord_resolution.py and
test_crowdsource_long_regressions.py.
"""
import numpy as np
from astropy.coordinates import SkyCoord

# Source x/y column conventions, in preference order for "first available".
_XY_COLUMN_CANDIDATES = (
    ('xcentroid', 'ycentroid'),
    ('x_centroid', 'y_centroid'),   # photutils >=3.0
    ('x_fit', 'y_fit'),
    ('x_init', 'y_init'),
    ('x', 'y'),
)


def _get_source_xy(tbl):
    """Return source x/y columns using the first available coordinate convention."""
    if 'x_fit' in tbl.colnames and 'y_fit' in tbl.colnames:
        return np.asarray(tbl['x_fit']), np.asarray(tbl['y_fit'])
    if 'xcentroid' in tbl.colnames and 'ycentroid' in tbl.colnames:
        return np.asarray(tbl['xcentroid']), np.asarray(tbl['ycentroid'])
    # photutils >=3.0 emits ``x_centroid``/``y_centroid``
    if 'x_centroid' in tbl.colnames and 'y_centroid' in tbl.colnames:
        return np.asarray(tbl['x_centroid']), np.asarray(tbl['y_centroid'])
    if 'x_init' in tbl.colnames and 'y_init' in tbl.colnames:
        return np.asarray(tbl['x_init']), np.asarray(tbl['y_init'])
    if 'x' in tbl.colnames and 'y' in tbl.colnames:
        return np.asarray(tbl['x']), np.asarray(tbl['y'])
    raise KeyError(f"No recognized x/y coordinate columns in {tbl.colnames}")


def _column_to_float_array(tbl, colname):
    col = tbl[colname]
    if hasattr(col, 'filled'):
        return np.asarray(col.filled(np.nan), dtype=float)
    return np.asarray(col, dtype=float)


def _best_available_xy(tbl):
    # photutils >=3.0 emits ``x_centroid``/``y_centroid`` from DAOStarFinder;
    # 2.x emits ``xcentroid``/``ycentroid``.  Accept both.
    candidates = [
        ('xcentroid', 'ycentroid'),
        ('x_centroid', 'y_centroid'),
        ('x_fit', 'y_fit'),
        ('x_init', 'y_init'),
        ('x', 'y'),
    ]
    best_pair = None
    best_score = -1
    best_x = None
    best_y = None
    for xname, yname in candidates:
        if xname in tbl.colnames and yname in tbl.colnames:
            xvals = _column_to_float_array(tbl, xname)
            yvals = _column_to_float_array(tbl, yname)
            score = np.isfinite(xvals).sum() + np.isfinite(yvals).sum()
            if score > best_score:
                best_score = score
                best_pair = (xname, yname)
                best_x = xvals
                best_y = yvals
    if best_pair is None:
        raise KeyError(f"No recognized x/y coordinate columns in {tbl.colnames}")
    return best_x, best_y


def _has_any_xy_columns(tbl):
    return any(
        xname in tbl.colnames and yname in tbl.colnames
        for xname, yname in (('xcentroid', 'ycentroid'),
                             ('x_centroid', 'y_centroid'),
                             ('x_fit', 'y_fit'),
                             ('x_init', 'y_init'), ('x', 'y'))
    )


def _skycoord_radec_arrays(tbl, colname):
    """Return ``(ra_deg, dec_deg)`` numpy arrays for every row of
    ``tbl[colname]``.

    ``tbl[colname]`` MUST be a vectorised ``SkyCoord``-mixin column.
    All producers in this module (``_resolve_seed_skycoords`` and
    ``_augment_seed_catalog_with_detections_sky``) now build mixin
    columns; an object-dtype column of SkyCoord scalars is treated as a
    bug at the producer site, not something to silently work around.
    """
    n = len(tbl)
    ra = np.full(n, np.nan, dtype=float)
    dec = np.full(n, np.nan, dtype=float)
    if n == 0 or colname not in tbl.colnames:
        return ra, dec

    col = tbl[colname]
    if not isinstance(col, SkyCoord):
        raise TypeError(
            f"_skycoord_radec_arrays expected tbl['{colname}'] to be a "
            f"SkyCoord-mixin column, got {type(col).__name__}.  Fix the "
            f"producer to assign a SkyCoord array, not an object-dtype "
            f"list of SkyCoord scalars."
        )
    ra_v = np.asarray(col.ra.deg, dtype=float)
    dec_v = np.asarray(col.dec.deg, dtype=float)
    if hasattr(col, 'mask') and col.mask is not None:
        valid = ~np.asarray(col.mask, dtype=bool)
        ra[valid] = ra_v[valid]
        dec[valid] = dec_v[valid]
    else:
        ra[:] = ra_v
        dec[:] = dec_v
    return ra, dec


def confident_star_mask(cat, *, qfit_max=0.2, sky_clean_prom_min=5.0,
                        sky_clean_snr_min=3.0, drop_overshoot=True):
    """Boolean mask of CONFIDENT STARS for plotting / science analysis.

    The vetted catalog deliberately retains some lower-confidence tiers (e.g.
    peakSB / bright-isolated keeps) so the model==catalog subtraction invariant
    holds; for CMDs and completeness plots you usually want only sources whose
    star-nature is positively established.  A source qualifies via ANY of:

    - qfit-confident: ``qfit <= qfit_max`` (well-fit point source), or
    - sky-clean tier: ``sky_clean`` (measured local emission consistent with
      dark sky -- emission contamination physically impossible there) AND
      ``prominence >= sky_clean_prom_min`` AND fit S/N >= sky_clean_snr_min,
      qfit ignored (blend-degraded), or
    - subtracted saturated star (``replaced_saturated``): real bright star,
      but note its flux comes from the wing/spike fit -- plot with care.

    Sources flagged ``model_overshoot`` are excluded (inflated fits) unless
    ``drop_overshoot=False``.  Columns absent from ``cat`` make their tier
    contribute False (e.g. a pre-sky-clean catalog just falls back to the
    qfit tier), so the mask is safe on any catalog generation.
    """
    n = len(cat)

    def _col(name, default=np.nan):
        if name not in cat.colnames:
            return np.full(n, default)
        c = cat[name]
        c = c.filled(default) if hasattr(c, 'filled') else c
        return np.asarray(c, dtype=float)

    qf = _col('qfit')
    tier_qfit = np.isfinite(qf) & (qf <= qfit_max)

    sk = _col('sky_clean', 0.0).astype(bool)
    prom = _col('prominence')
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = _col('flux') / _col('flux_err')
    tier_sky = (sk & np.isfinite(prom) & (prom >= sky_clean_prom_min)
                & np.isfinite(snr) & (snr >= sky_clean_snr_min))

    tier_sat = _col('replaced_saturated', 0.0).astype(bool)

    mask = tier_qfit | tier_sky | tier_sat
    if drop_overshoot:
        mask &= ~_col('model_overshoot', 0.0).astype(bool)
    return mask


def alignment_star_mask(cat, *, suffix='', qfit_max=0.4):
    """Boolean mask of rows safe to use as ASTROMETRIC anchors (frame
    alignment, epoch-to-epoch transformations, cross-band registration).

    Wing-constrained satstar centroids are good (3 mas frame-to-frame
    repeatability, 14 mas vs clean narrow-band centroids; Brick 2026-07-11)
    but their errors are not guaranteed benign-Gaussian: a coherent ~34 mas
    bulk offset relative to the daophot frame was measured on the demo
    rebuild, exactly the class of systematic that does not average down in a
    transformation fit.  Clipped/suppressed daophot centroids of saturated
    and near-saturated stars are worse (core-skewed).  So transformation
    star sets must EXCLUDE, per band:

    - ``replaced_saturated``  (position = wing fit; possible coherent offset)
    - ``is_saturated``        (clipped daophot centroid, unrepaired)
    - ``forced_filled``       (position copied from another band)
    - poor fits (``qfit > qfit_max``)

    ``suffix`` selects per-band columns in merged catalogs (e.g.
    ``suffix='_f182m'``).  Missing columns contribute no exclusion, so the
    mask is safe on any catalog generation.
    """
    n = len(cat)

    def _col(name, default=np.nan):
        name = name + suffix
        if name not in cat.colnames:
            return np.full(n, default)
        c = cat[name]
        c = c.filled(default) if hasattr(c, 'filled') else c
        return np.asarray(c, dtype=float)

    bad = (_col('replaced_saturated', 0.0).astype(bool)
           | _col('is_saturated', 0.0).astype(bool)
           | _col('forced_filled', 0.0).astype(bool))
    qf = _col('qfit')
    bad |= np.isfinite(qf) & (qf > qfit_max)
    return ~bad


def color_reliable_mask(cat, band1, band2):
    """Boolean mask: the ``band1 - band2`` color of each row is trustworthy.

    Brick CMD audit (2026-07-10): rows saturated in one band whose OTHER-band
    magnitude came from the m8 forced fill produce colors wrong by 5-9 mag --
    the cross-band merge missed the partner band's real detection (median
    0.214" away; saturated-centroid scatter), so the fill measured the
    star-subtracted residual.  92-100% of such colors are >2 mag off the
    locus; they form the diagonal plumes on the CMD bright end.

    A color is UNRELIABLE iff:

    - one band is ``forced_filled`` while the row is
      ``replaced_saturated``/``is_saturated`` in the other (either direction),
      or
    - ONE band was repaired by satstar substitution (``replaced_saturated``)
      while the other is saturation-affected but NOT repaired
      (``is_saturated`` without ``replaced_saturated``).  Brick strip audit
      (2026-07-11): such one-band-repaired rows carry colors wrong by up to
      several magnitudes (a substituted-medium/clipped-narrow pair, e.g.
      F410M substituted 1490 rows vs F405N only 513); rows repaired in BOTH
      bands are magnitude-flat through the suppression-strip windows.

    Ordinary fill-vs-detection and fill-vs-fill colors are kept: fills at
    unsaturated positions are genuine measurements/limits (5% bad, same as
    detections).  Bands are the lowercase merged-catalog suffixes ('f182m').
    Missing columns contribute False (a catalog with no fill or saturation
    columns is fully reliable).
    """
    n = len(cat)

    def _bool(name):
        if name not in cat.colnames:
            return np.zeros(n, dtype=bool)
        c = cat[name]
        c = c.filled(False) if hasattr(c, 'filled') else c
        return np.asarray(c, dtype=bool)

    def _sat(band):
        return _bool(f'replaced_saturated_{band}') | _bool(f'is_saturated_{band}')

    def _repaired(band):
        return _bool(f'replaced_saturated_{band}')

    def _clipped_unrepaired(band):
        # saturation-clipped daophot value that was never repaired: either the
        # DQ-saturated veto/miss case, or a gate-rejected strip candidate whose
        # gray-zone correction did NOT apply (satclip_corrected False; see
        # merge_catalogs gray-zone clip correction, PR #83).
        return ((_bool(f'is_saturated_{band}') | (_bool(f'satstar_gate_rejected_{band}')
                                                  & ~_bool(f'satclip_corrected_{band}')))
                & ~_bool(f'replaced_saturated_{band}'))

    bad = ((_bool(f'forced_filled_{band1}') & _sat(band2))
           | (_bool(f'forced_filled_{band2}') & _sat(band1))
           # one-band-repaired: substituted in one band, saturation-clipped
           # and unrepaired in the other -> spurious color (strip audit).
           | (_repaired(band1) & _clipped_unrepaired(band2))
           | (_repaired(band2) & _clipped_unrepaired(band1)))
    return ~bad
