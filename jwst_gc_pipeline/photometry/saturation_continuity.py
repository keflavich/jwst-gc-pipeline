"""Saturation-continuity metric — certification test for merged catalogs.

Definition
----------
For a band pair (A = band that saturates, B = reference band), using rows
CLEAN in B (independently_detected, not replaced_saturated, not
forced_filled):

  color = mag_vega_A - mag_vega_B
  In 0.5-mag bins of mag_B, split rows into
     SAT   = replaced_saturated_A
     UNFLG = no saturation flags in A
  TRANSITION bins: both classes have n >= 10 AND each is >= 20% of the bin
  (these straddle the saturation boundary; brighter bins have broken UNFLG
  rows, fainter SAT rows are phantoms, so neither is a fair comparison).

  jump(bin)  = median(color | SAT) - median(color | UNFLG)
  C1 = max |jump| over transition bins.

  Fallback when no transition bin exists (weakly-saturating bands): C2 =
  max |median(color|SAT) - locus(mag_B)| over SAT bins with n >= 10
  brightward of boundary+1, where locus = robust linear fit of median UNFLG
  color over mag_B in [boundary+0.5, boundary+3.5].

PASS: metric < 0.05 mag (goal) / < 0.10 mag (certification floor).
A discontinuity means saturation-handled photometry is on a different flux
scale than normal photometry — the CMD breaks at the saturation boundary.

Current status (Brick, 2026-07-09 catalogs): only f200w-f212n passes the
0.10 floor (C1 = 0.094).
"""
import numpy as np


def _get(cat, name, fill, n):
    """Column with masked-fill, or a constant array when absent (catalogs
    from other pipeline generations may lack some flag columns)."""
    if name not in cat.colnames:
        return np.full(n, fill)
    return _filled(cat[name], fill)


def _filled(col, fill):
    try:
        return np.asarray(col.filled(fill))
    except AttributeError:
        return np.asarray(col)


def saturation_continuity(cat, bandA, bandB, binwidth=0.5, min_n=10,
                          min_frac=0.20, frac_boundary=0.25):
    """Return dict(metric, kind, worst_bin, bins) for pair (A, B)."""
    n = len(cat)
    magA = _get(cat, f'mag_vega_{bandA}', np.nan, n).astype(float)
    magB = _get(cat, f'mag_vega_{bandB}', np.nan, n).astype(float)
    repA = _get(cat, f'replaced_saturated_{bandA}', False, n).astype(bool)
    ffA = _get(cat, f'forced_filled_{bandA}', False, n).astype(bool)
    satA = _get(cat, f'is_saturated_{bandA}', False, n).astype(bool)
    cleanB = (np.isfinite(magB)
              & _get(cat, f'independently_detected_{bandB}', True, n).astype(bool)
              & ~_get(cat, f'replaced_saturated_{bandB}', False, n).astype(bool)
              & ~_get(cat, f'forced_filled_{bandB}', False, n).astype(bool)
              & ~_get(cat, f'is_saturated_{bandB}', False, n).astype(bool))
    ok = np.isfinite(magA) & cleanB
    color = magA - magB
    SAT = ok & repA
    UNFLG = ok & ~repA & ~ffA & ~satA
    if SAT.sum() < min_n:
        return dict(metric=np.nan, kind='no-sat-population', worst_bin=None, bins=[])

    lo = np.floor(np.nanmin(magB[SAT | UNFLG]) * 2) / 2
    hi = np.ceil(np.nanmax(magB[SAT | UNFLG]) * 2) / 2
    edges = np.arange(lo, hi, binwidth)

    bins, boundary = [], None
    for e in edges:
        inb = (magB >= e) & (magB < e + binwidth)
        ns, nu = int((SAT & inb).sum()), int((UNFLG & inb).sum())
        tot = ns + nu
        if tot == 0:
            continue
        frac = ns / tot
        if frac >= frac_boundary and ns >= 3:
            boundary = e + binwidth
        rec = dict(magB_lo=e, n_sat=ns, n_unflg=nu, frac=frac, jump=np.nan)
        if ns >= min_n and nu >= min_n and min(frac, 1 - frac) >= min_frac:
            rec['jump'] = float(np.median(color[SAT & inb])
                                - np.median(color[UNFLG & inb]))
        bins.append(rec)

    trans = [b for b in bins if np.isfinite(b['jump'])]
    if trans:
        worst = max(trans, key=lambda b: abs(b['jump']))
        return dict(metric=abs(worst['jump']), kind='C1-boundary-jump',
                    worst_bin=worst, bins=bins)

    # C2 fallback: locus-referenced satstar offset
    if boundary is None:
        boundary = np.percentile(magB[SAT], 75)
    pts = [(b['magB_lo'] + binwidth / 2,
            np.median(color[UNFLG & (magB >= b['magB_lo'])
                            & (magB < b['magB_lo'] + binwidth)]), b['n_unflg'])
           for b in bins
           if boundary + 0.5 <= b['magB_lo'] <= boundary + 3.0
           and b['n_unflg'] >= 50]
    if len(pts) < 2:
        return dict(metric=np.nan, kind='no-locus', worst_bin=None, bins=bins)
    x, y, n = map(np.array, zip(*pts))
    locus = np.poly1d(np.polyfit(x, y, 1, w=np.sqrt(n)))
    offs = []
    for b in bins:
        if b['n_sat'] >= min_n and b['magB_lo'] + binwidth <= boundary + 1.0:
            inb = (magB >= b['magB_lo']) & (magB < b['magB_lo'] + binwidth)
            off = float(np.median(color[SAT & inb])
                        - locus(b['magB_lo'] + binwidth / 2))
            offs.append((abs(off), dict(b, jump=off)))
    if not offs:
        return dict(metric=np.nan, kind='no-sat-bins', worst_bin=None, bins=bins)
    m, worst = max(offs, key=lambda t: t[0])
    return dict(metric=m, kind='C2-locus-offset', worst_bin=worst, bins=bins)


def assert_saturation_continuity(cat, pairs, threshold=0.10):
    """Regression-test entry point: raises AssertionError listing failures."""
    fails = []
    for a, b in pairs:
        r = saturation_continuity(cat, a, b)
        if np.isfinite(r['metric']) and r['metric'] >= threshold:
            w = r['worst_bin']
            fails.append(f"{a} vs {b}: {r['metric']:.3f} mag ({r['kind']}) "
                         f"at mag_{b}={w['magB_lo']:.1f} "
                         f"(n_sat={w['n_sat']}, n_unflg={w['n_unflg']})")
    assert not fails, ('saturation-boundary photometric discontinuity '
                       f'>= {threshold} mag:\n  ' + '\n  '.join(fails))
