#!/usr/bin/env python
"""Validator for astrometric per-visit/per-exposure offsets tables
(``offsets/Offsets_JWST_Brick<prop>_VIRAC2locked.csv`` and friends) consumed by
``PipelineRerunNIRCAM-LONG.fix_alignment``.

It catches the class of bug that corrupted brick-1182 in 2026-07: a builder
COLLAPSED distinct visits onto a single visit's offset -- visit-001's true
-17.5"/+13.5" was overwritten with visit-002's +1.9"/+1.0" for every filter, so
fix_alignment mis-corrected every visit-001 exposure by ~20" and the top half of
every mosaic + catalog was untied to VIRAC2.

Key physical fact: guide-star pointing errors are PER-VISIT and independent; two
DISTINCT visits of the same filter agreeing to a few mas does not happen by
chance (real per-visit offsets differ by tens of mas to arcseconds).  So
identical per-visit offsets are the collapse signature.

Usage (CLI):  python -m jwst_gc_pipeline.reduction.validate_offsets_table TABLE.csv [--strict]
Programmatic: validate_offsets_table(table_or_path) -> list[str] of issues.
"""
from __future__ import annotations

import sys
import numpy as np
from astropy.table import Table


# Two DISTINCT visits agreeing to within this on-sky separation are treated as a
# builder collapse (independent guide-star errors never match this closely).
COLLAPSE_TOL_MAS = 5.0
# A rigid per-visit pointing offset larger than this is implausible -> likely a
# unit error (mas written as arcsec) or a runaway solve.
MAX_PLAUSIBLE_OFFSET_ARCSEC = 60.0


def _load(table_or_path) -> Table:
    if isinstance(table_or_path, Table):
        return table_or_path
    return Table.read(table_or_path, format='csv')


def _per_visit_offsets(t: Table):
    """{filter: {visit: (dra_arcsec, ddec_arcsec)}} using the median over the
    visit's exposures/detectors (rows sharing Visit+Filter)."""
    need = {'Visit', 'Filter', 'dra', 'ddec'}
    missing = need - set(t.colnames)
    if missing:
        raise KeyError(f"offsets table missing columns {sorted(missing)}; "
                       f"has {t.colnames}")
    vis = np.asarray([str(v) for v in t['Visit']])
    filt = np.asarray([str(f) for f in t['Filter']])
    dra = np.asarray(t['dra'], dtype=float)
    ddec = np.asarray(t['ddec'], dtype=float)
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for f in sorted(set(filt)):
        out[f] = {}
        fm = filt == f
        for v in sorted(set(vis[fm])):
            m = fm & (vis == v)
            out[f][v] = (float(np.nanmedian(dra[m])), float(np.nanmedian(ddec[m])))
    return out


def flag_collapsed_visits(table_or_path, tol_mas: float = COLLAPSE_TOL_MAS) -> list[str]:
    """Flag filters whose DISTINCT visits carry (near-)identical offsets.

    Returns a list of human-readable issue strings (empty = clean).
    """
    t = _load(table_or_path)
    per = _per_visit_offsets(t)
    cosd = np.cos(np.radians(-28.7))  # GC; converts coordinate dRA to on-sky
    issues: list[str] = []
    for f, byv in per.items():
        if len(byv) < 2:
            continue
        visits = list(byv)
        # max pairwise on-sky separation between distinct visits
        maxsep = 0.0
        pair = None
        for i in range(len(visits)):
            for j in range(i + 1, len(visits)):
                (ra1, de1), (ra2, de2) = byv[visits[i]], byv[visits[j]]
                sep = np.hypot((ra1 - ra2) * cosd, de1 - de2) * 1000.0  # mas
                if sep > maxsep:
                    maxsep = sep; pair = (visits[i], visits[j])
        if maxsep < tol_mas:
            (ra, de) = byv[visits[0]]
            issues.append(
                f"COLLAPSE: filter {f}: {len(visits)} distinct visits all share "
                f"offset ~({ra:+.3f},{de:+.3f})\" (max pairwise sep {maxsep:.1f} mas "
                f"< {tol_mas} mas). Independent per-visit guide-star errors do not "
                f"agree this closely -> the builder likely collapsed visits "
                f"(e.g. gave every visit the dominant visit's shift). Visits: {visits}.")
    return issues


def flag_insane_magnitudes(table_or_path, max_arcsec: float = MAX_PLAUSIBLE_OFFSET_ARCSEC) -> list[str]:
    """Flag offsets larger than a plausible pointing error (unit-error / runaway)."""
    t = _load(table_or_path)
    dra = np.abs(np.asarray(t['dra'], dtype=float))
    ddec = np.abs(np.asarray(t['ddec'], dtype=float))
    mag = np.hypot(dra, ddec)
    bad = np.isfinite(mag) & (mag > max_arcsec)
    if not bad.any():
        return []
    worst = float(np.nanmax(mag[bad]))
    return [f"INSANE MAGNITUDE: {int(bad.sum())} rows have |offset| > {max_arcsec}\" "
            f"(worst {worst:.1f}\"). A pointing offset this large usually means a "
            f"unit error (mas stored as arcsec) or a runaway solve."]


def validate_offsets_table(table_or_path, tol_mas: float = COLLAPSE_TOL_MAS,
                           max_arcsec: float = MAX_PLAUSIBLE_OFFSET_ARCSEC) -> list[str]:
    """Run all offsets-table sanity checks; return the combined issue list."""
    return (flag_collapsed_visits(table_or_path, tol_mas=tol_mas)
            + flag_insane_magnitudes(table_or_path, max_arcsec=max_arcsec))


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('table', help='offsets CSV to validate')
    ap.add_argument('--tol-mas', type=float, default=COLLAPSE_TOL_MAS)
    ap.add_argument('--max-arcsec', type=float, default=MAX_PLAUSIBLE_OFFSET_ARCSEC)
    ap.add_argument('--strict', action='store_true',
                    help='exit non-zero if any issue is found (default also exits '
                         'non-zero on issues; --strict is kept for scripting clarity)')
    args = ap.parse_args(argv)
    issues = validate_offsets_table(args.table, tol_mas=args.tol_mas, max_arcsec=args.max_arcsec)
    if not issues:
        print(f"OK: {args.table} passed offsets-table validation.")
        return 0
    print(f"FAIL: {args.table} has {len(issues)} issue(s):")
    for i in issues:
        print(f"  - {i}")
    return 1


if __name__ == '__main__':
    sys.exit(main())
