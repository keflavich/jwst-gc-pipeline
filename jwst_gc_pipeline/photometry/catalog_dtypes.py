"""Downcast per-frame catalog columns at write time, without dropping any
column.

Per-frame ``*_daophot_basic.fits`` (``crowdsource_catalogs_long.save_photutils_results``)
and ``*_m<N>_satstar_catalog.fits`` (written in
``saturated_star_finding.remove_saturated_stars``) are the only catalogs carried in ORIGINAL DETECTOR PIXEL coordinates,
so they must be kept -- but on disk they carry ``float64``/``int64`` columns
for quantities that do not need that width: photometric metrics (flux, local
background, fit-quality statistics) and small integer counters/bitmasks.

This module holds one explicit allowlist per direction. A column not on either
list is left untouched -- a newly added column is never silently shrunk, and
a column that is renamed or dropped upstream is simply absent from the
allowlist intersection (``table.colnames``), not an error.

Hard rule (CLAUDE.md astrometry rules): pixel positions/errors and sky
coordinates are NEVER downcast here -- ``x_fit``/``y_fit``/``x_init``/
``y_init``/``x_err``/``y_err``/``x_0``/``y_0``/``xcentroid``/``ycentroid``,
every ``skycoord_*`` column, and ``sat_com_ra``/``sat_com_dec`` stay
``float64``. The per-row ``dra``/``ddec`` arcsec-error columns
(``crowdsource_catalogs_long.save_photutils_results``, derived from
``x_err``/``y_err``) are likewise left alone -- they are position errors in a
different unit, not a photometric metric.
"""
import numpy as np

__all__ = ['FLOAT32_COLUMNS', 'INT32_COLUMNS', 'downcast_catalog_dtypes']

#: float64 -> float32: photometric / fit-quality metric columns only.
#: Verified against a real per-frame catalog
#: (cloudef F210M nrca2 ..._m7_daophot_basic.fits, 21442 rows) and a real
#: satstar catalog (cloudef F770W MIRI ..._m12_satstar_catalog.fits, 40 rows):
#: values run from ~1e-8 to ~2.6e7, comfortably inside float32's +-3.4e38
#: range, and float32's ~1e-7 relative precision is far below these columns'
#: measurement uncertainty.
FLOAT32_COLUMNS = frozenset({
    'local_bkg', 'local_bkg_resbgsub',
    'flux_init', 'flux_fit', 'flux_err', 'flux_fit_raw',
    'qfit', 'cfit', 'reduced_chi2', 'model_data_peak_ratio',
    'modelsub_bkg', 'modelsub_bkg_rms',
    'sidelobe_resid_sigma', 'ssr_ratio',
    # wingcal_ratio stays float64: merge_catalogs tests `ratio == 1.0` as
    # the 'no per-star C(r) applied' sentinel, and a float32 rounding of a
    # real ratio near 1 onto exactly 1.0 would apply the pooled C(r) twice.
    'wingcal_rmask',
    'sat_severity_floor', 'satstar_implied_peak', 'satstar_observed_peak',
})

#: int64 -> int32: small counters / bitmasks. Checked against production
#: catalogs: ``flags`` tops out at bit 5 there (nowhere near the int32 sign
#: bit, 31), and ``id``/``group_id``/``group_size``/``n_pixels_fit``/
#: ``sat_area``/``modelsub_bkg_npix`` are all per-frame small counts (tens of
#: thousands at most, int32 max is ~2.1e9). ``downcast_catalog_dtypes`` also
#: checks the actual min/max of each column before casting, so a column that
#: somehow exceeds int32 range is left at int64 rather than silently
#: overflowed.
INT32_COLUMNS = frozenset({
    'id', 'group_id', 'group_size', 'flags',
    'n_pixels_fit', 'sat_area', 'modelsub_bkg_npix',
})

_INT32_MIN = np.iinfo(np.int32).min
_INT32_MAX = np.iinfo(np.int32).max


def downcast_catalog_dtypes(table, *, float_columns=FLOAT32_COLUMNS,
                             int_columns=INT32_COLUMNS):
    """Downcast ``table``'s photometric-metric / counter columns in place.

    Mutates and returns ``table``. Only columns present AND on one of the two
    allowlists are touched; everything else (including every pixel-position
    and sky-coordinate column) is left exactly as it was. Masked columns keep
    their mask -- only the underlying dtype changes.

    A column on ``int_columns`` is downcast only if its current values fit in
    ``int32`` (checked via the column's own min/max); a column that doesn't
    fit is left at its original dtype rather than overflowed.
    """
    for name in table.colnames:
        col = table[name]
        dtype = getattr(col, 'dtype', None)
        if dtype is None:
            # Non-array columns (e.g. a SkyCoord mixin column) have no numpy
            # dtype of their own and are never on either allowlist anyway.
            continue
        # Compare by kind + itemsize, NOT ``dtype == np.float64`` / ``np.int64``
        # directly: a table just read from FITS carries big-endian dtypes
        # (``>f8``, ``>i8``), and ``np.dtype('>f8') == np.float64`` is False,
        # which would silently make this whole function a no-op on exactly
        # the tables it exists to shrink.
        is_f8 = dtype.kind == 'f' and dtype.itemsize == 8
        is_i8 = dtype.kind == 'i' and dtype.itemsize == 8
        if name in float_columns:
            if is_f8:
                table[name] = col.astype(np.float32)
        elif name in int_columns:
            if is_i8 and len(col):
                # Range-check the UNMASKED values only: np.min on a masked
                # column whose entries are all masked returns ``masked``,
                # and int(masked) raises MaskError (seeded fits leave
                # group_id fully masked).  A column with no unmasked value
                # has nothing that can overflow and is cast as-is.
                vals = np.ma.compressed(np.ma.asarray(col))
                if (vals.size == 0
                        or (_INT32_MIN <= int(vals.min())
                            and int(vals.max()) <= _INT32_MAX)):
                    table[name] = col.astype(np.int32)
    return table
