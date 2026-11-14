"""Two-band pair coverage of a merged catalog (#1269).

Why this exists
---------------
In GC Treasury (program 10678) the pre-#1178 m7 catalogues lost the F480M
measurement of weakly saturated stars.  A star detected in F212N then had no
usable independent F480M partner, and 24.8% of the F212N 13-14.5 detections
(40% at l = -0.20...-0.13) fell into channel B2 of the gc-3dmap tiles.  The
2026-10-28 rerun reduced the share to 3.9%.  Nothing in the pipeline reported
the stale input, so the defect surfaced two months later as a notch in a
depth map.  This module measures the share directly from one catalogue.

Channels
--------
Channel logic follows ``gc-3dmap-2026/scripts/02_extract_tiles.py``, for the
band pair ``(band_a, band_b)`` (default F212N, F480M).  A row is *clean* when

- it is independently detected in both bands,
- neither band is forced-filled,
- ``qfit < qfit_max`` or ``replaced_saturated`` holds in both bands, and
- both ``mag_vega`` values are finite.

Rows detected independently in ``band_a`` and inside ``mag_range`` (a range
in ``mag_vega_{band_a}``) are then split into

- ``BB``: clean, ``band_a`` not replaced-saturated;
- ``BSAT``: clean, ``band_a`` replaced-saturated;
- ``B2``: the remainder, including forced ``band_b`` rows.

``R4`` counts rows in the same magnitude range that are detected
independently in ``band_b`` only.  The statistic is

    b2_share = B2 / (B2 + BB + BSAT)

Decision
--------
The verdict is ``warn`` above ``warn_share`` (default 0.10), ``fail`` above
``fail_share`` (default 0.20) and ``insufficient`` when fewer than
``min_count`` rows are detected in ``band_a``.  The defaults come from the
GC Treasury m8 catalogues at F212N 13-14.5: the pre-#1178 set had a 10th
percentile of 12% with 36 of 68 observations above 20%, and the 2026-10-28
set a maximum of 8.6%.  Other fields and filter pairs need their own
``mag_range`` and thresholds.  The command line reports the verdicts and
exits with status 1 on ``fail`` only; no pipeline stage gates on it.
"""
import argparse
import json
import sys

import numpy as np

from .lf_continuity import _colnames

__all__ = ['pair_coverage', 'format_result', 'main']

PASS, WARN, FAIL, INSUFFICIENT = 'pass', 'warn', 'fail', 'insufficient'

_NEEDED = ('independently_detected', 'forced_filled', 'qfit',
           'replaced_saturated', 'mag_vega')


def _columns(cat, bands):
    """The columns this check needs, as plain arrays.  A missing column
    raises ``KeyError`` naming it."""
    have = set(_colnames(cat))
    out = {}
    for b in bands:
        for stem in _NEEDED:
            name = f'{stem}_{b}'
            if name not in have:
                raise KeyError(f'pair_coverage: catalog has no column {name!r}')
            col = cat[name]
            if hasattr(col, 'filled'):
                col = col.filled(False if stem not in ('qfit', 'mag_vega') else np.nan)
            out[name] = np.asarray(col)
    return out


def pair_coverage(cat, band_a='f212n', band_b='f480m', mag_range=(13.0, 14.5),
                  qfit_max=0.3, warn_share=0.10, fail_share=0.20, min_count=50):
    """B2 share of the ``band_a`` detections of ``cat`` (see the module
    docstring).  ``cat`` is an astropy Table, a FITS record array's column
    dict, or any mapping of column name to array.  Returns a dict with
    ``n_BB``, ``n_BSAT``, ``n_B2``, ``n_R4``, ``b2_share`` and ``verdict``."""
    band_a, band_b = band_a.lower(), band_b.lower()
    c = _columns(cat, (band_a, band_b))

    def flag(stem, b):
        return c[f'{stem}_{b}'].astype(bool)

    ia, ib = flag('independently_detected', band_a), flag('independently_detected', band_b)
    fa, fb = flag('forced_filled', band_a), flag('forced_filled', band_b)
    rsa, rsb = flag('replaced_saturated', band_a), flag('replaced_saturated', band_b)
    ma = np.asarray(c[f'mag_vega_{band_a}'], float)
    mb = np.asarray(c[f'mag_vega_{band_b}'], float)
    with np.errstate(invalid='ignore'):
        qa = (np.asarray(c[f'qfit_{band_a}'], float) < qfit_max) | rsa
        qb = (np.asarray(c[f'qfit_{band_b}'], float) < qfit_max) | rsb
        in_range = (ma >= mag_range[0]) & (ma < mag_range[1])
    clean = ia & ib & ~fa & ~fb & qa & qb & np.isfinite(ma) & np.isfinite(mb)
    det = ia & in_range
    n_bb = int((det & clean & ~rsa).sum())
    n_bsat = int((det & clean & rsa).sum())
    n_b2 = int((det & ~clean).sum())
    n_r4 = int((~ia & ib & in_range).sum())
    n_det = n_bb + n_bsat + n_b2
    share = n_b2 / n_det if n_det else float('nan')
    if n_det < min_count:
        verdict = INSUFFICIENT
    elif share > fail_share:
        verdict = FAIL
    elif share > warn_share:
        verdict = WARN
    else:
        verdict = PASS
    return dict(band_a=band_a, band_b=band_b, mag_range=tuple(mag_range),
                qfit_max=qfit_max, n_BB=n_bb, n_BSAT=n_bsat, n_B2=n_b2,
                n_R4=n_r4, b2_share=share, verdict=verdict,
                warn_share=warn_share, fail_share=fail_share)


def format_result(r):
    """One-line human-readable summary of a :func:`pair_coverage` result."""
    lo, hi = r['mag_range']
    return (f"{r['band_a']}/{r['band_b']} mag_vega_{r['band_a']} {lo:g}-{hi:g}: "
            f"{r['verdict'].upper()} B2 share {100 * r['b2_share']:.1f}% "
            f"(BB {r['n_BB']} BSAT {r['n_BSAT']} B2 {r['n_B2']} R4 {r['n_R4']})")


def _read_columns(path, bands):
    """Read only the columns this check needs from a (large) FITS catalog."""
    from astropy.io import fits
    want = {f'{stem}_{b}' for b in bands for stem in _NEEDED}
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        return {c: np.array(data[c]) for c in data.columns.names if c in want}


def main(argv=None):
    """CLI: ``python -m jwst_gc_pipeline.photometry.pair_coverage CAT.fits
    [CAT.fits ...] [--band-a f212n --band-b f480m] [--json]``.  Exit status 1
    when any catalog fails."""
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('catalogs', nargs='+')
    ap.add_argument('--band-a', default='f212n')
    ap.add_argument('--band-b', default='f480m')
    ap.add_argument('--mag-min', type=float, default=13.0)
    ap.add_argument('--mag-max', type=float, default=14.5)
    ap.add_argument('--qfit-max', type=float, default=0.3)
    ap.add_argument('--warn-share', type=float, default=0.10)
    ap.add_argument('--fail-share', type=float, default=0.20)
    ap.add_argument('--min-count', type=int, default=50)
    ap.add_argument('--json', action='store_true', help='print JSON lines')
    a = ap.parse_args(argv)
    bad = False
    for path in a.catalogs:
        cols = _read_columns(path, (a.band_a.lower(), a.band_b.lower()))
        r = pair_coverage(cols, a.band_a, a.band_b,
                          mag_range=(a.mag_min, a.mag_max), qfit_max=a.qfit_max,
                          warn_share=a.warn_share, fail_share=a.fail_share,
                          min_count=a.min_count)
        bad |= r['verdict'] == FAIL
        if a.json:
            print(json.dumps(dict(r, catalog=path), default=float))
        else:
            print(f'{path}  {format_result(r)}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
