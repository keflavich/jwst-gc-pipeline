"""Readers for the ZEROFRAME fit-treatment columns of the satstar catalogs (#681).

A saturated star is fitted either with its recovered rim (the mask is the
group-0-saturated deep core only) or with the whole any-group SATURATED blob
masked, depending on whether a sibling ``_ramp.fits`` gave a usable group-0
scale.  The two treatments differ by 0.5-3.6 mag on marginally saturated
stars.  ``get_saturated_stars`` records the outcome per row:

* ``zf_fit`` (int32): 1 when the group-0 deep core was the mask, 0 when the
  blob was;
* ``zf_rim_in_core`` (int32): 1 when the source's own saturated component held
  at least one rewritten rim pixel;
* ``zf_nrim_src`` (int32): that pixel count, -1 for a forced or outside-FOV
  seed with no component.

Catalogs written before #681 have none of these columns.  They read as
UNKNOWN, never as 0: :func:`satstar_zf_status` returns a masked array whose
mask marks those rows.  The columns are integers because an integer mask
survives the FITS write of the consolidated cache, and a boolean one does not
(masked rows read back as True).
"""
import numpy as np

__all__ = ['ZF_FIT_COLUMN', 'satstar_zf_status', 'satstar_zf_summary',
           'zf_fit_fraction']

ZF_FIT_COLUMN = 'zf_fit'


def satstar_zf_status(tbl):
    """``zf_fit`` of ``tbl`` as a masked int array: 1, 0, or masked (unknown).

    A table with no ``zf_fit`` column returns an array that is masked
    throughout.  A masked row stays masked.
    """
    n = len(tbl)
    if ZF_FIT_COLUMN not in tbl.colnames:
        return np.ma.masked_all(n, dtype=np.int32)
    col = tbl[ZF_FIT_COLUMN]
    mask = np.ma.getmaskarray(col).copy()
    vals = np.asarray(np.ma.getdata(col))
    # A column of an unexpected dtype (float NaN, say) is unknown where it is
    # not exactly 0 or 1.
    if vals.dtype.kind == 'f':
        mask |= ~np.isfinite(vals)
        vals = np.where(mask, 0, vals)
    mask |= ~np.isin(vals, (0, 1))
    return np.ma.array(vals.astype(np.int32), mask=mask)


def satstar_zf_summary(tbl):
    """One-line count of rows fitted on the deep core, the blob, or unknown."""
    st = satstar_zf_status(tbl)
    known = ~np.ma.getmaskarray(st)
    vals = np.ma.getdata(st)
    n_unknown = int((~known).sum())
    n_one = int((known & (vals == 1)).sum())
    n_zero = int((known & (vals == 0)).sum())
    return (f"satstar ZEROFRAME fit mode: {n_one} deep-core mask (rim "
            f"recovered), {n_zero} blob mask, {n_unknown} unknown (catalog "
            f"predates #681) of {len(tbl)} rows")


def zf_fit_fraction(status, group, n_groups, usable=None):
    """Fraction of each group's members with ``zf_fit == 1``, over known rows.

    ``status`` is the masked array of :func:`satstar_zf_status`, ``group[i]``
    the group of member ``i``.  ``usable`` (optional boolean array) drops
    members from the count.  Groups with no known member get NaN.
    """
    group = np.asarray(group, dtype=np.int64)
    known = ~np.ma.getmaskarray(status)
    if usable is not None:
        known &= np.asarray(usable, dtype=bool)
    ones = known & (np.ma.getdata(status) == 1)
    n_known = np.bincount(group[known], minlength=n_groups).astype(float)
    n_ones = np.bincount(group[ones], minlength=n_groups).astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(n_known > 0, n_ones / np.where(n_known > 0, n_known, 1.0),
                        np.nan).astype(np.float32)
