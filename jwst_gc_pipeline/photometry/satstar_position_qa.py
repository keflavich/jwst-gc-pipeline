"""Per-catalog QA of satstar positions against their saturated-component anchor.

Every row of a satstar catalog records ``sat_com_ra`` / ``sat_com_dec``, the sky
position of the bounding-box centre of the saturated component the row was
seeded from (``reduction/saturated_star_finding.py``; the box centre, not a
flux-weighted centre of mass).  :func:`satstar_position_qa` compares that anchor
with the fitted ``skycoord_fit`` and summarises the separation, so a catalog
whose fits wandered off their seeds, or whose positions were locked onto the
seed, shows up in the log without a bespoke script (issue #951).

The comparison is row against its own anchor.  It involves no catalogue
matching, so ASTROMETRY RULE #1 does not apply.

The module is kept free of heavy imports so that it can be unit-tested without
the satstar fitter.  It reads a table and returns numbers; it writes no column
and takes no part in any cache signature.

Interpretation: peak and sub-floor seeds (most rows) take their anchor from a
component that may hold little DQ-SATURATED area, so the statistic measures
agreement with the seed component, not with a true saturated core.
"""
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

ANCHOR_RA_COL = 'sat_com_ra'
ANCHOR_DEC_COL = 'sat_com_dec'
FIT_SKY_COL = 'skycoord_fit'

#: Separation below which a row counts as locked onto its anchor.  The anchor
#: transport in ``satstar_wcs_refresh`` leaves ~0.001 mas, well inside this.
LOCK_TOL_MAS = 0.05
#: ``lock_suspect`` is raised when more than this fraction of rows are locked.
LOCK_SUSPECT_FRACTION = 0.25
LOCK_SUSPECT_MIN_ROWS = 10
#: Separation beyond which a row is reported as a gross outlier.
GROSS_MAS = 200.0


def _as_float_array(col):
    """Column values as a float array with masked entries set to NaN."""
    return np.ma.filled(np.ma.asarray(col, dtype=float), np.nan)


def frame_pixel_scale_mas(wcs, x, y):
    """Mean pixel scale in mas at pixel ``(x, y)``, from a GWCS-backed WCS.

    ``wcs`` is the object returned by ``frame_wcs``; the scale comes from
    finite differences of ``pixel_to_world`` (the GWCS), not from the SIP
    header.  Returns ``None`` when the WCS yields no finite value there.
    """
    x = float(x)
    y = float(y)
    c0 = wcs.pixel_to_world(x, y)
    cx = wcs.pixel_to_world(x + 1.0, y)
    cy = wcs.pixel_to_world(x, y + 1.0)
    if not all(isinstance(c, SkyCoord) for c in (c0, cx, cy)):
        return None
    sx = c0.separation(cx).to(u.mas).value
    sy = c0.separation(cy).to(u.mas).value
    scale = float(np.sqrt(sx * sy))
    return scale if np.isfinite(scale) and scale > 0 else None


def satstar_position_qa(tbl, pixscale_mas=None):
    """Summarise the ``skycoord_fit`` to ``sat_com`` separation of a satstar table.

    Parameters
    ----------
    tbl : astropy.table.Table
        Satstar catalog with a ``skycoord_fit`` SkyCoord column and, for a
        current-vintage catalog, ``sat_com_ra`` / ``sat_com_dec`` in degrees.
    pixscale_mas : float, optional
        Detector pixel scale in mas.  When given, ``frac_gt_halfpix`` is
        reported.

    Returns
    -------
    dict
        ``{'available': False}`` when the table has no anchor columns (a
        catalog from before ``sat_com_*`` existed) or no ``skycoord_fit``.
        Otherwise ``available`` (True), ``n`` (rows), ``n_finite_anchor`` (rows
        compared; rows with a NaN anchor or fit, such as forced outside-FOV
        seeds, are excluded), ``median_mas``, ``p16_mas``, ``p84_mas``,
        ``frac_gt_200mas``, ``frac_locked`` (separation < ``LOCK_TOL_MAS``),
        ``lock_suspect`` (more than 25% locked among at least 10 rows), and
        ``frac_gt_halfpix`` when ``pixscale_mas`` is given.  ``median_by_seed_kind``
        maps each ``seed_kind`` to its median when that column exists.  When no
        row has a finite anchor the statistics are NaN.
    """
    if (ANCHOR_RA_COL not in tbl.colnames or ANCHOR_DEC_COL not in tbl.colnames
            or FIT_SKY_COL not in tbl.colnames):
        return {'available': False}

    n = len(tbl)
    fit = tbl[FIT_SKY_COL]
    if isinstance(fit, SkyCoord):
        fit_ra = np.asarray(fit.ra.deg, dtype=float)
        fit_dec = np.asarray(fit.dec.deg, dtype=float)
    else:
        fit = SkyCoord(fit)
        fit_ra = np.asarray(fit.ra.deg, dtype=float)
        fit_dec = np.asarray(fit.dec.deg, dtype=float)
    anc_ra = _as_float_array(tbl[ANCHOR_RA_COL])
    anc_dec = _as_float_array(tbl[ANCHOR_DEC_COL])

    ok = (np.isfinite(fit_ra) & np.isfinite(fit_dec)
          & np.isfinite(anc_ra) & np.isfinite(anc_dec))
    out = {'available': True, 'n': int(n), 'n_finite_anchor': int(ok.sum())}

    # Tangent-plane separation about the anchor: RA difference scaled by cos(dec)
    dra = ((fit_ra[ok] - anc_ra[ok] + 180.0) % 360.0 - 180.0) \
        * np.cos(np.deg2rad(anc_dec[ok]))
    ddec = fit_dec[ok] - anc_dec[ok]
    sep = np.hypot(dra, ddec) * 3.6e6  # deg -> mas

    if sep.size == 0:
        for key in ('median_mas', 'p16_mas', 'p84_mas', 'frac_gt_200mas',
                    'frac_locked'):
            out[key] = float('nan')
        out['lock_suspect'] = False
        if pixscale_mas is not None:
            out['frac_gt_halfpix'] = float('nan')
        return out

    p16, med, p84 = np.percentile(sep, [16, 50, 84])
    out['median_mas'] = float(med)
    out['p16_mas'] = float(p16)
    out['p84_mas'] = float(p84)
    out['frac_gt_200mas'] = float(np.mean(sep > GROSS_MAS))
    out['frac_locked'] = float(np.mean(sep < LOCK_TOL_MAS))
    out['lock_suspect'] = bool(out['frac_locked'] > LOCK_SUSPECT_FRACTION
                               and sep.size >= LOCK_SUSPECT_MIN_ROWS)
    if pixscale_mas is not None:
        out['frac_gt_halfpix'] = float(np.mean(sep > 0.5 * float(pixscale_mas)))

    if 'seed_kind' in tbl.colnames:
        kinds = np.array([k.decode() if isinstance(k, bytes) else str(k)
                          for k in np.asarray(tbl['seed_kind'])])[ok]
        out['median_by_seed_kind'] = {
            str(k): float(np.median(sep[kinds == k])) for k in np.unique(kinds)}
    return out


def format_satstar_position_qa(qa, label=''):
    """One log line for a :func:`satstar_position_qa` result (informational).

    The wording avoids every token in ``monitoring.jobs.LOG_SIGNATURES``, so the
    line is scanned as ``info``.
    """
    head = f"Satstar position QA{' ' + label if label else ''}:"
    if not qa.get('available'):
        return (f"{head} pre-anchor catalog vintage (no sat_com_ra/dec); "
                f"fit-vs-component separation not computed")
    if qa['n_finite_anchor'] == 0:
        return (f"{head} 0/{qa['n']} rows carry a finite component anchor")
    parts = [f"fit-to-component separation over {qa['n_finite_anchor']}/{qa['n']} rows"
             f": median {qa['median_mas']:.1f} mas"
             f" (p16 {qa['p16_mas']:.1f}, p84 {qa['p84_mas']:.1f})"]
    if 'frac_gt_halfpix' in qa:
        parts.append(f"{100 * qa['frac_gt_halfpix']:.1f}% beyond half a pixel")
    parts.append(f"{100 * qa['frac_gt_200mas']:.2f}% beyond {GROSS_MAS:.0f} mas")
    parts.append(f"{100 * qa['frac_locked']:.0f}% within {LOCK_TOL_MAS} mas of the anchor")
    if qa.get('lock_suspect'):
        parts.append("positions locked to the anchor in a large share of rows")
    by_kind = qa.get('median_by_seed_kind')
    if by_kind:
        parts.append("median by seed kind " + ", ".join(
            f"{k} {v:.1f}" for k, v in sorted(by_kind.items())))
    parts.append("reference is the seed component's box centre")
    return f"{head} " + "; ".join(parts)


def _catalog_pixel_scale_mas(tbl, frame_path):
    """Pixel scale in mas at the catalog's median centroid, or ``None``.

    The catalog meta carries no ``PIXAR_A2``, so the scale comes from the
    frame's GWCS.  Any failure to read the frame or evaluate the WCS leaves the
    half-pixel fraction out of the QA line.
    """
    if frame_path is None or 'xcentroid' not in tbl.colnames \
            or 'ycentroid' not in tbl.colnames:
        return None
    x = _as_float_array(tbl['xcentroid'])
    y = _as_float_array(tbl['ycentroid'])
    ok = np.isfinite(x) & np.isfinite(y)
    if not ok.any():
        return None
    from jwst_gc_pipeline.frame_wcs import frame_wcs
    try:
        wcs = frame_wcs(frame_path, warn_missing=False)
        return frame_pixel_scale_mas(wcs, np.median(x[ok]), np.median(y[ok]))
    except (OSError, ValueError, KeyError, RuntimeError):
        return None


def log_satstar_position_qa(tbl, frame_path=None, label=''):
    """Print the ``Satstar position QA:`` line for ``tbl`` and return the dict.

    Read-only: the table is not modified and no file is written.
    """
    qa = satstar_position_qa(
        tbl, pixscale_mas=(_catalog_pixel_scale_mas(tbl, frame_path)
                           if 'sat_com_ra' in tbl.colnames else None))
    print(format_satstar_position_qa(qa, label=label), flush=True)
    return qa
