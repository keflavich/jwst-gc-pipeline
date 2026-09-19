"""Luminosity-function continuity across the satstar/daophot seam (#925 item 5).

Why this exists
---------------
A merged catalog measures bright stars with the saturated-star channel
(``replaced_saturated_{band}``) and fainter stars with daophot.  Where the
two channels hand over, the star COUNTS must be continuous.  In GC Treasury
(program 10678) m8 tile o132 they are not: inside the tile's own footprint the
F480M 0.1-mag bins hold 156, 187, 112 stars at 11.7-11.9 and then 1, 2, 5 at
12.0-12.2 (#925).  Stars in that range were deleted from both channels (the
satstar severity gate rejected them and the near-saturation veto dropped their
daophot fit).  A colour metric cannot see a hole, because the missing stars
have no colour; this module measures the hole directly.

It complements :mod:`jwst_gc_pipeline.photometry.saturation_continuity`,
which tests whether the two channels share a FLUX SCALE (colour jump across
the boundary).  This module tests whether they share a POPULATION (counts).

Statistic
---------
For one band of one observation's catalog:

1. Rows: finite ``mag_vega_{band}``, not ``forced_filled_{band}`` (a forced
   fill is a measurement at another band's position, not a detection), and
   inside the observation footprint (by default :func:`occupancy_footprint`:
   10" sky cells holding >= 20 rows that are not ``replaced_saturated`` in
   any band, i.e. cells the observation's own daophot pass covers).  The
   footprint cut makes the check blind to satstar rows pooled in from other
   observations (#925 defect 1), which otherwise dominate the counts.
2. Seam ``b``: the faint edge of the satstar channel, the upper edge of the
   faintest 0.1-mag bin (with >= 5 rows) in which >= 50% of rows are
   ``replaced_saturated``.
3. Reference LF: a Poisson maximum-likelihood fit of
   ``log N(m) = poly(m - b)`` (degree 1 by default) to 0.05-mag bins in the
   two FLANKS ``[b - flank_outer, b - flank_inner)`` and
   ``[b + flank_inner, b + flank_outer)``.  The flanks exclude the seam so a
   defect there cannot pull the fit toward itself.
4. Windows ``B = [b - window, b)`` (bright side, satstar channel) and
   ``F = [b, b + window)`` (faint side, daophot channel).  With observed
   counts ``nB, nF`` and fitted expectations ``eB, eF``:

   - ``ratio_bright = nB / eB``, ``ratio_faint = nF / eF``
   - ``step = ratio_bright / ratio_faint``

   A hole on the faint side drives ``ratio_faint`` toward 0 and ``step`` up;
   a pile-up of mis-measured stars on the bright side (#925 defect 3) drives
   ``ratio_bright`` and ``step`` up.

Decision
--------
The check FAILS when any of ``ratio_bright``, ``ratio_faint``, ``step``
lies outside ``[1/max_ratio, max_ratio]`` AND that departure is significant:
the Poisson tail probability of the window count given its expectation (for
``step``, the binomial split of ``nB + nF``) is below ``p_fail``.  Both
conditions are required: the ratio bound sets the smallest defect worth
failing on, and the significance bound stops a 3-star bin from failing.  When
either window's expectation is below ``min_expected`` the verdict is
``insufficient``, which is not a failure.

The defaults (``max_ratio=2``, ``p_fail=1e-6``, ``min_expected=20``) were set
from measured m8 catalogs; see the table in the Notes of
:func:`lf_continuity`.
"""
import argparse
import json
import sys

import numpy as np
from scipy import stats as _sps

__all__ = ['occupancy_footprint', 'find_seam', 'lf_continuity',
           'assert_lf_continuity', 'main']

# Verdicts
PASS, FAIL, INSUFFICIENT, NO_SAT = 'pass', 'fail', 'insufficient', 'no-sat-population'


def _colnames(cat):
    try:
        return list(cat.colnames)
    except AttributeError:
        return list(cat.keys())


def _col(cat, name, fill, n):
    """Column ``name`` as a plain array (masked values filled), or a constant
    array of ``fill`` when the column is absent."""
    if name not in _colnames(cat):
        return np.full(n, fill)
    col = cat[name]
    try:
        return np.asarray(col.filled(fill))
    except AttributeError:
        return np.asarray(col)


def _len(cat):
    return len(cat[_colnames(cat)[0]])


def _radec(cat):
    """Reference-position RA/Dec in degrees, from either a ``skycoord_ref``
    SkyCoord column or the flattened ``skycoord_ref.ra``/``.dec`` columns.
    Returns ``None`` when neither is present."""
    names = _colnames(cat)
    if 'skycoord_ref.ra' in names and 'skycoord_ref.dec' in names:
        return (np.asarray(cat['skycoord_ref.ra'], float),
                np.asarray(cat['skycoord_ref.dec'], float))
    if 'skycoord_ref' in names:
        sc = cat['skycoord_ref']
        return np.asarray(sc.ra.deg, float), np.asarray(sc.dec.deg, float)
    return None


def _bands(cat):
    pre = 'mag_vega_'
    return [c[len(pre):] for c in _colnames(cat) if c.startswith(pre)]


def occupancy_footprint(cat, cell_arcsec=10.0, min_count=20, bands=None):
    """Boolean mask of rows inside the observation's own daophot footprint.

    The sky is cut into ``cell_arcsec`` square cells (RA scaled by
    cos(Dec)).  A cell is IN the footprint when it holds at least
    ``min_count`` rows that are not ``replaced_saturated`` in any of
    ``bands`` (default: every band with a ``mag_vega_`` column).  Satstar
    rows pooled from other observations sit in cells with no daophot rows of
    this observation and fall outside.

    At GC-field densities (>> 20 rows per 10" cell) this loses only a thin
    rim.  For a sparse field lower ``min_count`` or pass an explicit mask to
    :func:`lf_continuity`.  Returns all-True when the catalog has no
    reference coordinates.
    """
    n = _len(cat)
    rd = _radec(cat)
    if rd is None:
        return np.ones(n, bool)
    ra, dec = rd
    anyrep = np.zeros(n, bool)
    for b in (bands if bands is not None else _bands(cat)):
        anyrep |= _col(cat, f'replaced_saturated_{b}', False, n).astype(bool)
    good = np.isfinite(ra) & np.isfinite(dec)
    cosd = np.cos(np.deg2rad(np.nanmedian(dec[good]))) if good.any() else 1.0
    scale = 3600.0 / cell_arcsec
    ix = np.floor(np.where(good, ra, 0) * cosd * scale).astype(np.int64)
    iy = np.floor(np.where(good, dec, 0) * scale).astype(np.int64)
    key = ix * 10_000_019 + iy
    u, cnt = np.unique(key[good & ~anyrep], return_counts=True)
    return good & np.isin(key, u[cnt >= min_count])


def find_seam(mag, rep, binwidth=0.1, min_n=5, min_frac=0.5):
    """Faint edge of the satstar channel: upper edge of the faintest
    ``binwidth`` bin holding >= ``min_n`` rows of which >= ``min_frac`` are
    ``rep`` (replaced_saturated).  NaN when no bin qualifies."""
    ok = np.isfinite(mag)
    if not np.any(ok & rep):
        return np.nan
    lo = np.floor(np.min(mag[ok]) / binwidth) * binwidth
    hi = np.max(mag[ok & rep]) + 2 * binwidth
    edges = np.arange(lo, hi + binwidth, binwidth)
    n_all, _ = np.histogram(mag[ok], edges)
    n_rep, _ = np.histogram(mag[ok & rep], edges)
    qual = (n_all >= min_n) & (n_rep >= min_frac * n_all)
    if not qual.any():
        return np.nan
    return float(edges[np.nonzero(qual)[0].max() + 1])


def _poisson_logpoly_fit(x, y, deg, n_iter=50):
    """Maximum-likelihood fit of counts ``y ~ Poisson(exp(poly(x)))`` by
    iteratively reweighted least squares.  Returns ``np.polyval``-ordered
    coefficients."""
    X = np.vander(x, deg + 1)
    beta = np.linalg.lstsq(X, np.log(np.maximum(y, 0.5)), rcond=None)[0]
    for _ in range(n_iter):
        mu = np.exp(X @ beta)
        z = X @ beta + (y - mu) / mu
        w = np.sqrt(mu)
        new = np.linalg.lstsq(X * w[:, None], z * w, rcond=None)[0]
        if np.allclose(new, beta, rtol=0, atol=1e-10):
            beta = new
            break
        beta = new
    return beta


def _two_sided_poisson_p(n, mu):
    """Two-sided tail probability of observing ``n`` given Poisson mean
    ``mu`` (twice the smaller tail, capped at 1)."""
    lo = _sps.poisson.cdf(n, mu)
    hi = _sps.poisson.sf(n - 1, mu)
    return float(min(1.0, 2 * min(lo, hi)))


def _two_sided_binom_p(k, n, p):
    if n == 0:
        return 1.0
    return float(_sps.binomtest(int(k), int(n), p).pvalue)


def lf_continuity(cat, band, footprint='auto', seam=None, window=0.25,
                  flank_inner=0.5, flank_outer=1.5, binwidth=0.05, deg=1,
                  max_ratio=2.0, p_fail=1e-6, min_expected=20.0,
                  exclude_forced=True):
    """Star-count continuity across the satstar/daophot seam for one band.

    Parameters
    ----------
    cat : `~astropy.table.Table` or mapping of column name -> array
        One observation's merged catalog (needs ``mag_vega_{band}`` and
        ``replaced_saturated_{band}``; ``forced_filled_{band}`` and
        ``skycoord_ref`` are used when present).
    band : str
        Lower-case filter name, e.g. ``'f480m'``.
    footprint : 'auto', None, or bool array
        ``'auto'`` uses :func:`occupancy_footprint`; ``None`` keeps every
        row; an array is used as the in-footprint mask.
    seam : float, optional
        Seam magnitude; default from :func:`find_seam`.
    window, flank_inner, flank_outer, binwidth : float
        Window half-width and flank limits (mag, relative to the seam) and
        the fit bin width.  See the module docstring.
    deg : int
        Degree of the log-count polynomial.
    max_ratio : float
        Smallest departure (factor) that can fail.
    p_fail : float
        Significance a departure must reach to fail.
    min_expected : float
        Minimum fitted count in each window for a verdict.
    exclude_forced : bool
        Drop ``forced_filled_{band}`` rows.

    Returns
    -------
    dict
        ``verdict`` ('pass', 'fail', 'insufficient', 'no-sat-population'),
        ``band``, ``seam``, ``n_bright``, ``n_faint``, ``exp_bright``,
        ``exp_faint``, ``ratio_bright``, ``ratio_faint``, ``step``,
        ``p_bright``, ``p_faint``, ``p_step``, ``failed`` (names of the
        failing quantities), ``n_rows`` (rows used), ``n_sat`` (satstar rows
        used), ``slope`` (fitted d log10 N / dm at the seam).

    Notes
    -----
    Threshold calibration, measured 2026-09-19 with the defaults on shipped
    m8 catalogs (read-only; occupancy footprint):

    ==========================  =====  ======  ======  =======  =======
    catalog / band              seam   r_B     r_F     step     verdict
    ==========================  =====  ======  ======  =======  =======
    gc-treasury o132 F480M      12.00  1.11    0.000   inf      fail
    gc-treasury o132 F212N      15.60  0.87    0.825   1.05     pass
    brick o001 F212N            15.60  1.11    1.018   1.09     pass
    cloudc F212N                15.50  0.87    0.631   1.37     pass
    sgrc F480M                  13.80  0.66    0.798   0.83     pass
    cloudef F480M               11.90  2.28    0.066   34.8     fail
    ==========================  =====  ======  ======  =======  =======

    GC Treasury o127, o129, o130 and o135 F480M also fail (seam 12.00,
    r_B 0.99-1.57, r_F 0.000-0.027, step >= 36).  Seams known clean from
    the #925 comparisons (the F212N rows and sgrc F480M) lie within
    0.63-1.37 on every ratio; the 10678 F480M hole sits at r_F <= 0.027.
    ``max_ratio=2`` lies between the two, with a factor >= 1.46 margin to
    the clean side and >= 18 to the defect.  The clean seams reach
    p ~ 1e-55 (cloudc F212N r_F = 0.63), so significance alone cannot be
    the criterion at these sample sizes; the ratio bound carries the
    decision and ``p_fail`` only protects thin windows.
    """
    n = _len(cat)
    mag = _col(cat, f'mag_vega_{band}', np.nan, n).astype(float)
    rep = _col(cat, f'replaced_saturated_{band}', False, n).astype(bool)
    use = np.isfinite(mag)
    if exclude_forced:
        use &= ~_col(cat, f'forced_filled_{band}', False, n).astype(bool)
    if isinstance(footprint, str):
        if footprint != 'auto':
            raise ValueError(f"footprint must be 'auto', None or a mask, got {footprint!r}")
        use &= occupancy_footprint(cat)
    elif footprint is not None:
        use &= np.asarray(footprint, bool)
    mag, rep = mag[use], rep[use]

    out = dict(band=band, verdict=NO_SAT, seam=np.nan, n_rows=int(use.sum()),
               n_sat=int(rep.sum()), n_bright=0, n_faint=0,
               exp_bright=np.nan, exp_faint=np.nan, ratio_bright=np.nan,
               ratio_faint=np.nan, step=np.nan, p_bright=np.nan,
               p_faint=np.nan, p_step=np.nan, slope=np.nan, failed=[])
    b = find_seam(mag, rep) if seam is None else float(seam)
    out['seam'] = b
    if not np.isfinite(b):
        # satstar rows exist but too few per bin to locate the seam
        if out['n_sat'] > 0:
            out['verdict'] = INSUFFICIENT
        return out

    edges = np.arange(-flank_outer, flank_outer + binwidth / 2, binwidth)
    cnt, _ = np.histogram(mag - b, edges)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    flank = np.abs(ctr) > flank_inner
    if cnt[flank].sum() == 0:
        out['verdict'] = INSUFFICIENT
        return out
    beta = _poisson_logpoly_fit(ctr[flank], cnt[flank].astype(float), deg)
    out['slope'] = float(np.polyval(np.polyder(beta), 0.0) / np.log(10))

    def expect(lo, hi):
        # integrate the fitted density (per binwidth) over [lo, hi)
        xs = np.linspace(lo, hi, 201)
        return float(np.trapezoid(np.exp(np.polyval(beta, xs)), xs) / binwidth)

    nB = int(np.sum((mag >= b - window) & (mag < b)))
    nF = int(np.sum((mag >= b) & (mag < b + window)))
    eB, eF = expect(-window, 0.0), expect(0.0, window)
    out.update(n_bright=nB, n_faint=nF, exp_bright=eB, exp_faint=eF)
    if min(eB, eF) < min_expected:
        out['verdict'] = INSUFFICIENT
        return out
    rB, rF = nB / eB, nF / eF
    out.update(ratio_bright=rB, ratio_faint=rF,
               step=(rB / rF) if rF > 0 else np.inf,
               p_bright=_two_sided_poisson_p(nB, eB),
               p_faint=_two_sided_poisson_p(nF, eF),
               p_step=_two_sided_binom_p(nB, nB + nF, eB / (eB + eF)))
    lo_r, hi_r = 1.0 / max_ratio, max_ratio
    for name, pname in (('ratio_bright', 'p_bright'),
                        ('ratio_faint', 'p_faint'), ('step', 'p_step')):
        r = out[name]
        if (r < lo_r or r > hi_r) and out[pname] < p_fail:
            out['failed'].append(name)
    out['verdict'] = FAIL if out['failed'] else PASS
    return out


def format_result(r):
    """One-line human-readable summary of a :func:`lf_continuity` result."""
    if r['verdict'] == NO_SAT:
        return f"{r['band']}: no satstar population ({r['n_sat']} rows)"
    s = (f"{r['band']}: {r['verdict'].upper()} seam={r['seam']:.2f} "
         f"bright {r['n_bright']}/{r['exp_bright']:.1f} "
         f"faint {r['n_faint']}/{r['exp_faint']:.1f}")
    if np.isfinite(r['ratio_faint']):
        s += (f" ratio_bright={r['ratio_bright']:.2f} "
              f"ratio_faint={r['ratio_faint']:.3f} step={r['step']:.2f} "
              f"(p_faint={r['p_faint']:.1e}, p_step={r['p_step']:.1e})")
    if r['failed']:
        s += f" failed={','.join(r['failed'])}"
    return s


def assert_lf_continuity(cat, bands=None, **kwargs):
    """Raise ``AssertionError`` naming every band whose LF breaks at the
    satstar/daophot seam.  ``bands`` defaults to every band with a
    ``replaced_saturated_`` column; other keywords go to
    :func:`lf_continuity`.  Returns the list of results when all pass.
    The footprint is computed once and shared across bands."""
    if bands is None:
        pre = 'replaced_saturated_'
        bands = [c[len(pre):] for c in _colnames(cat) if c.startswith(pre)]
    fp = kwargs.get('footprint', 'auto')
    if isinstance(fp, str) and fp == 'auto':
        kwargs['footprint'] = occupancy_footprint(cat)
    results = [lf_continuity(cat, b, **kwargs) for b in bands]
    fails = [format_result(r) for r in results if r['verdict'] == FAIL]
    assert not fails, ('luminosity-function discontinuity at the '
                       'satstar/daophot seam:\n  ' + '\n  '.join(fails))
    return results


def _read_columns(path, bands):
    """Read only the columns this check needs from a (large) FITS catalog."""
    from astropy.io import fits
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        names = data.columns.names
        allb = [c[len('mag_vega_'):] for c in names if c.startswith('mag_vega_')]
        want = {'skycoord_ref.ra', 'skycoord_ref.dec'}
        for b in allb:
            want |= {f'mag_vega_{b}', f'replaced_saturated_{b}',
                     f'forced_filled_{b}'}
        cols = {c: np.array(data[c]) for c in names if c in want}
    if bands is None:
        bands = [b for b in allb if f'replaced_saturated_{b}' in cols]
    return cols, bands


def main(argv=None):
    """CLI: ``python -m jwst_gc_pipeline.photometry.lf_continuity CAT.fits
    [--band f480m ...] [--no-footprint] [--json]``.  Exit status 1 when any
    band fails."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('catalogs', nargs='+')
    ap.add_argument('--band', action='append', dest='bands',
                    help='band to check (repeatable; default: all with satstar rows)')
    ap.add_argument('--no-footprint', action='store_true',
                    help='use every row instead of the occupancy footprint')
    ap.add_argument('--window', type=float, default=0.25)
    ap.add_argument('--max-ratio', type=float, default=2.0)
    ap.add_argument('--p-fail', type=float, default=1e-6)
    ap.add_argument('--min-expected', type=float, default=20.0)
    ap.add_argument('--json', action='store_true', help='print JSON lines')
    a = ap.parse_args(argv)
    bad = False
    for path in a.catalogs:
        cols, bands = _read_columns(path, [b.lower() for b in a.bands] if a.bands else None)
        fp = None if a.no_footprint else occupancy_footprint(cols)
        for b in bands:
            r = lf_continuity(cols, b, footprint=fp, window=a.window,
                              max_ratio=a.max_ratio, p_fail=a.p_fail,
                              min_expected=a.min_expected)
            bad |= r['verdict'] == FAIL
            if a.json:
                print(json.dumps(dict(r, catalog=path), default=float))
            else:
                print(f'{path}  {format_result(r)}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
