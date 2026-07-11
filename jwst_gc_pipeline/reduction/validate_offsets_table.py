"""Sanity checks for the per-visit offsets table that ``fix_alignment`` consumes.

The operative ``Offsets_JWST_Brick<pid>_VIRAC2locked.csv`` is ad-hoc/hand-curated
(no clean single builder; the .bak trail crosstie/2stage/PREF410MSPLIT/pervisit/
v001fix). That curation once **collapsed brick-1182 visit-001 onto visit-002's value**
(both ~+1.9" for a visit truly ~20" off) -- every independent MEASUREMENT table had
v001 = -17.5" correct, but the collapsed locked table is the one that got applied, so
half the mosaic stayed ~20" off.

The tell of that failure: two DISTINCT visits of the same filter carrying (near-)
identical offsets. Independent per-visit pointing errors do not agree to a few mas by
chance -- especially not across every filter at once. This module flags that pattern
so a collapsed table cannot be silently applied again.

This is a cheap STATIC check (no re-measurement). The strong dynamic check -- remeasure
each row vs VIRAC2 with the window-swept helper -- lives in the standalone validator
(brick/analysis/verify_v001_fix_independent.py) and in
``photometry.astrometry_offsets.measure_offset``.
"""
import numpy as np


def flag_collapsed_visits(offsets_tbl, tol_arcsec=0.02):
    """Flag filters whose distinct visits carry (near-)identical offsets.

    Parameters
    ----------
    offsets_tbl : astropy Table
        Must have ``Visit``, ``Filter``, and ``dra (arcsec)`` / ``ddec (arcsec)``
        (or ``dra`` / ``ddec``) columns.
    tol_arcsec : float
        Two visits whose offsets agree within this are treated as suspiciously
        identical (default 0.02" = 20 mas; real per-visit pointing errors differ by
        much more).

    Returns
    -------
    list of dict
        One entry per suspicious (filter, visit-pair): ``dict(filter, visit_a,
        visit_b, sep_arcsec, dra, ddec)``.  Empty list = clean.
    """
    cols = offsets_tbl.colnames
    if 'Visit' not in cols or 'Filter' not in cols:
        return []
    dc = 'dra (arcsec)' if 'dra (arcsec)' in cols else ('dra' if 'dra' in cols else None)
    ec = 'ddec (arcsec)' if 'ddec (arcsec)' in cols else ('ddec' if 'ddec' in cols else None)
    if dc is None or ec is None:
        return []

    vis = np.asarray(offsets_tbl['Visit'])
    filt = np.asarray(offsets_tbl['Filter'])
    dra = np.asarray(offsets_tbl[dc], dtype=float)
    ddec = np.asarray(offsets_tbl[ec], dtype=float)

    issues = []
    for f in np.unique(filt):
        fm = filt == f
        visits = np.unique(vis[fm])
        # per-visit mean offset (tables may be per-exposure)
        vals = {}
        for v in visits:
            vm = fm & (vis == v)
            vals[v] = (float(np.nanmedian(dra[vm])), float(np.nanmedian(ddec[vm])))
        vlist = list(vals)
        for i in range(len(vlist)):
            for j in range(i + 1, len(vlist)):
                a, b = vlist[i], vlist[j]
                sep = float(np.hypot(vals[a][0] - vals[b][0], vals[a][1] - vals[b][1]))
                if sep <= tol_arcsec:
                    issues.append(dict(filter=str(f), visit_a=str(a), visit_b=str(b),
                                       sep_arcsec=sep, dra=vals[a][0], ddec=vals[a][1]))
    return issues


class CollapsedOffsetsTableError(RuntimeError):
    """Raised when an offsets table has visits collapsed onto one value."""


def assert_offsets_table_sane(offsets_tbl, tol_arcsec=0.02, context="", raise_on_issue=False):
    """Warn (or raise) if ``offsets_tbl`` shows the visit-collapse signature.

    Returns the issue list (empty = clean).  Set ``raise_on_issue`` (or env
    ``OFFSETS_TABLE_COLLAPSE_RAISE=1``) to raise instead of warn.
    """
    import os
    import warnings
    issues = flag_collapsed_visits(offsets_tbl, tol_arcsec=tol_arcsec)
    if issues:
        lines = [f"  {i['filter']}: visits {i['visit_a']} & {i['visit_b']} both "
                 f"({i['dra']:+.4f},{i['ddec']:+.4f})\" (agree to {i['sep_arcsec']*1000:.1f} mas)"
                 for i in issues]
        msg = (f"COLLAPSED OFFSETS TABLE{(' ' + context) if context else ''}: distinct "
               f"visits share (near-)identical offsets -- the brick-1182 v001 failure "
               f"signature (a visit's real offset was overwritten by another's). Do NOT "
               f"trust this table; re-measure the flagged visits with a window-swept "
               f"histogram (photometry.astrometry_offsets.measure_offset).\n" + "\n".join(lines))
        if raise_on_issue or os.environ.get('OFFSETS_TABLE_COLLAPSE_RAISE') == '1':
            raise CollapsedOffsetsTableError(msg)
        warnings.warn(msg)
    return issues
