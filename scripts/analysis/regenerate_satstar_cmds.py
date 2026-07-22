#!/usr/bin/env python
r"""Regenerate the saturated-star CMD figures for the astrometry paper appendix
(``satstar_photometry.tex``, ``app:satstar``).

The original generator scripts were lost; this rebuilds them from the recovered
figure specs in ``docs/reports/SATURATED_STAR_PHOTOMETRY_ARTICLE.md`` plus the
surviving certification code in ``jwst_gc_pipeline.photometry.saturation_continuity``.
The annotated per-panel numbers are exactly that module's ``metric`` output, so
the figures and the CI gate cannot drift apart.

Input
-----
A COMBINED, all-band merged catalog carrying, per band, the columns
``mag_vega_<band>`` and (where available) ``replaced_saturated_<band>``,
``is_saturated_<band>``, ``forced_filled_<band>``, ``independently_detected_<band>``.
For the Brick this is the ``ok2221or1182`` union table under ``<basepath>/catalogs/``.
Missing flag columns fall back the same way the metric does
(is_saturated->False, forced_filled->False, independently_detected->True), so the
script runs against older tables too (SAT vs UNFLG is then keyed on
``replaced_saturated`` alone).

Dependency chain to feed this with FRESH satstar-recovered data
---------------------------------------------------------------
  running per-exposure cataloging (job 37553031, phases m12..m7, 2221 bands)
    -> per-filter merged tables            (merge_catalogs.py)
    -> combined all-band union table       (the ok2221or1182 combine step)
    -> THIS script                         (--after <union.fits>)
So this must be re-run only after the union table is rebuilt from the new m7
per-band catalogs; it does not itself trigger the reduction.

Usage
-----
    python regenerate_satstar_cmds.py --after <combined_after.fits> \
        [--before <combined_production.fits>] [--outdir DIR] [--tag DATE]

Without ``--before`` the multipair/stripfix panels are single-row (after only).
``--tag`` overrides the date token in the output filenames; the default keeps the
exact filenames the LaTeX ``\includegraphics`` already references (overwrite in
place), so the paper needs no edit.
"""
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table

from jwst_gc_pipeline.photometry.saturation_continuity import (
    saturation_continuity, degenerate_pair_flatness)

# ---------------------------------------------------------------------------
# Pair conventions (A = earlier-saturating band, B = reference; color = A - B,
# magnitude axis = mag_vega_B), from the article spec.
# ---------------------------------------------------------------------------
MULTIPAIR = [('f182m', 'f212n'), ('f405n', 'f410m'),
             ('f405n', 'f466n'), ('f212n', 'f405n')]
CALIBRATED_PAIR = ('f182m', 'f187n')          # headline: 0.74 -> 0.11
CLASS_PAIRS = [('f182m', 'f187n'), ('f182m', 'f212n'),
               ('f405n', 'f410m'), ('f405n', 'f466n')]
DEGENERATE_PAIRS = [('f405n', 'f410m'), ('f182m', 'f187n')]

# Default output filenames = the exact names the LaTeX already references, so a
# re-run overwrites in place. --tag replaces the date token in every value.
FIGNAMES = {
    'calibrated':  'brick_cmd_calibrated_substituted_2026-07-10.png',
    'multipair':   'brick_multipair_cmds_beforeafter_2026-07-11.png',
    'class':       'brick_multiband_class_cmds_2026-07-10.png',
    'degen_trend': 'brick_degenerate_pair_trends_2026-07-11.png',
    'degen_hist':  'brick_degen_pair_hists.png',
}
DEFAULT_OUTDIR = '/orange/adamginsburg/jwst/brick/astrometry_paper/figures/evidence'


def _col(cat, name, fill):
    """Column with masked-fill, or a constant array when absent -- mirrors
    ``saturation_continuity._get`` so this script and the metric agree on which
    rows are SAT/UNFLG/clean."""
    n = len(cat)
    if name not in cat.colnames:
        return np.full(n, fill)
    c = cat[name]
    try:
        return np.asarray(c.filled(fill))
    except AttributeError:
        return np.asarray(c)


def _sat_unflg_clean(cat, bandA, bandB):
    """Return (color, magB, SAT, UNFLG) using the metric's exact definitions."""
    n = len(cat)
    magA = _col(cat, f'mag_vega_{bandA}', np.nan).astype(float)
    magB = _col(cat, f'mag_vega_{bandB}', np.nan).astype(float)
    repA = _col(cat, f'replaced_saturated_{bandA}', False).astype(bool)
    ffA = _col(cat, f'forced_filled_{bandA}', False).astype(bool)
    satA = _col(cat, f'is_saturated_{bandA}', False).astype(bool)
    cleanB = (np.isfinite(magB)
              & _col(cat, f'independently_detected_{bandB}', True).astype(bool)
              & ~_col(cat, f'replaced_saturated_{bandB}', False).astype(bool)
              & ~_col(cat, f'forced_filled_{bandB}', False).astype(bool)
              & ~_col(cat, f'is_saturated_{bandB}', False).astype(bool))
    ok = np.isfinite(magA) & cleanB
    color = magA - magB
    SAT = ok & repA
    UNFLG = ok & ~repA & ~ffA & ~satA
    return color, magB, SAT, UNFLG


def _metric_str(cat, bandA, bandB):
    r = saturation_continuity(cat, bandA, bandB)
    if not np.isfinite(r['metric']):
        return f"C={r['kind']}", r
    return f"C={r['metric']:.2f} ({r['kind'].split('-')[0]})", r


def _cmd_panel(ax, cat, bandA, bandB, title=None, legend=False):
    """One CMD panel: UNFLG locus (gray) + SAT/substituted (orange), Y=mag_B
    inverted, annotated with the continuity metric."""
    color, magB, SAT, UNFLG = _sat_unflg_clean(cat, bandA, bandB)
    ax.scatter(color[UNFLG], magB[UNFLG], s=1, c='0.6', rasterized=True,
               label=f'unsaturated ({int(UNFLG.sum())})')
    ax.scatter(color[SAT], magB[SAT], s=2, c='tab:orange', rasterized=True,
               label=f'substituted ({int(SAT.sum())})')
    mstr, _ = _metric_str(cat, bandA, bandB)
    ax.set_xlabel(f'{bandA.upper()} - {bandB.upper()}')
    ax.set_ylabel(f'{bandB.upper()} (Vega)')
    if not ax.yaxis_inverted():
        ax.invert_yaxis()
    ax.set_title((title + '\n' if title else '') + mstr, fontsize=9)
    if legend:
        ax.legend(loc='upper right', fontsize=6, markerscale=4, framealpha=0.9)


# ---------------------------------------------------------------------------
# Figure builders
# ---------------------------------------------------------------------------
def fig_calibrated(after, outpath):
    """fig:calibrated_cmd -- single F182M-F187N CMD, calibrated + substituted."""
    a, b = CALIBRATED_PAIR
    fig, ax = plt.subplots(figsize=(5, 6))
    _cmd_panel(ax, after, a, b, title='F182M-F187N calibrated + substituted',
               legend=True)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def fig_multipair(after, before, outpath, tag_title='wing-cal + substitution'):
    """fig:multipair (and fig:stripfix) -- 4 pairs x (before/after) rows."""
    rows = [('production', before), (tag_title, after)] if before is not None \
        else [(tag_title, after)]
    fig, axes = plt.subplots(len(rows), len(MULTIPAIR),
                             figsize=(4 * len(MULTIPAIR), 5 * len(rows)),
                             squeeze=False)
    for ri, (rowlabel, cat) in enumerate(rows):
        for ci, (a, b) in enumerate(MULTIPAIR):
            _cmd_panel(axes[ri, ci], cat, a, b,
                       title=f'{rowlabel}: {a.upper()}-{b.upper()}',
                       legend=(ci == 0))
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def fig_class(after, outpath):
    """fig:class_cmds -- class-decomposed CMDs, one panel per pair.

    Four classes from the flag columns:
      unsaturated          = no sat flags
      substituted          = replaced_saturated
      dq-saturated (unrep) = is_saturated & ~replaced_saturated
      forced-filled        = forced_filled
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 11), squeeze=False)
    for ax, (a, b) in zip(axes.ravel(), CLASS_PAIRS):
        n = len(after)
        magA = _col(after, f'mag_vega_{a}', np.nan).astype(float)
        magB = _col(after, f'mag_vega_{b}', np.nan).astype(float)
        color = magA - magB
        rep = _col(after, f'replaced_saturated_{a}', False).astype(bool)
        sat = _col(after, f'is_saturated_{a}', False).astype(bool)
        ff = _col(after, f'forced_filled_{a}', False).astype(bool)
        base = np.isfinite(magA) & np.isfinite(magB)
        classes = [
            ('unsaturated', base & ~rep & ~sat & ~ff, '0.6', 1),
            ('dq-sat (unrepaired)', base & sat & ~rep, 'tab:blue', 3),
            ('forced-filled', base & ff & ~rep, 'tab:green', 3),
            ('substituted', base & rep, 'tab:orange', 3),
        ]
        for label, m, c, s in classes:
            if m.any():
                ax.scatter(color[m], magB[m], s=s, c=c, rasterized=True,
                           label=f'{label} ({int(m.sum())})')
        mstr, _ = _metric_str(after, a, b)
        ax.set_xlabel(f'{a.upper()} - {b.upper()}')
        ax.set_ylabel(f'{b.upper()} (Vega)')
        if not ax.yaxis_inverted():
            ax.invert_yaxis()
        ax.set_title(f'{a.upper()}-{b.upper()}   {mstr}', fontsize=9)
        ax.legend(loc='upper right', fontsize=6, markerscale=4, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _binned_median_color(cat, a, b, binwidth=0.25, min_n=20):
    """Binned-median color vs mag_B, following degenerate_pair_flatness bins
    (all rows incl. replaced_saturated; forced_filled excluded)."""
    magA = _col(cat, f'mag_vega_{a}', np.nan).astype(float)
    magB = _col(cat, f'mag_vega_{b}', np.nan).astype(float)
    ok = (np.isfinite(magA) & np.isfinite(magB)
          & ~_col(cat, f'forced_filled_{a}', False).astype(bool)
          & ~_col(cat, f'forced_filled_{b}', False).astype(bool))
    color = magA - magB
    lo = np.floor(np.nanmin(magB[ok]) / binwidth) * binwidth
    hi = np.nanmax(magB[ok])
    edges = np.arange(lo, hi, binwidth)
    xs, ys = [], []
    for e in edges:
        inb = ok & (magB >= e) & (magB < e + binwidth)
        if int(inb.sum()) >= min_n:
            xs.append(e + binwidth / 2)
            ys.append(np.median(color[inb]))
    return np.array(xs), np.array(ys)


def fig_degen_trends(after, before, outpath):
    """fig:degenerate_pairs -- binned-median color vs magnitude for the two
    near-degenerate pairs, with the flatness metric annotated."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    for ax, (a, b) in zip(axes.ravel(), DEGENERATE_PAIRS):
        if before is not None:
            xb, yb = _binned_median_color(before, a, b)
            ax.plot(xb, yb, '-', color='0.6', label='production')
        xa, ya = _binned_median_color(after, a, b)
        ax.plot(xa, ya, '-', color='k', label='after')
        r = degenerate_pair_flatness(after, a, b)
        if np.isfinite(r['plateau']):
            ax.axhline(r['plateau'], ls=':', color='tab:red',
                       label=f"plateau {r['plateau']:+.2f}")
        drift = ('n/a' if not np.isfinite(r['metric'])
                 else f"drift {r['metric']:.3f}")
        ax.set_xlabel(f'{b.upper()} (Vega)')
        ax.set_ylabel(f'{a.upper()} - {b.upper()}')
        ax.set_title(f'{a.upper()}-{b.upper()}   {drift}', fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def fig_degen_hists(after, before, outpath):
    """fig:degenhist -- 1-D color histograms per pair, strip-window vs plateau."""
    fig, axes = plt.subplots(len(DEGENERATE_PAIRS), 1,
                             figsize=(7, 4 * len(DEGENERATE_PAIRS)),
                             squeeze=False)
    for ax, (a, b) in zip(axes[:, 0], DEGENERATE_PAIRS):
        magA = _col(after, f'mag_vega_{a}', np.nan).astype(float)
        magB = _col(after, f'mag_vega_{b}', np.nan).astype(float)
        ff = (_col(after, f'forced_filled_{a}', False).astype(bool)
              | _col(after, f'forced_filled_{b}', False).astype(bool))
        color = (magA - magB)[np.isfinite(magA) & np.isfinite(magB) & ~ff]
        color = color[np.isfinite(color)]
        if before is not None:
            mAb = _col(before, f'mag_vega_{a}', np.nan).astype(float)
            mBb = _col(before, f'mag_vega_{b}', np.nan).astype(float)
            cb = (mAb - mBb)
            cb = cb[np.isfinite(cb)]
            ax.hist(cb, bins=120, range=(-1, 1), histtype='step',
                    color='0.6', label='production', density=True)
        ax.hist(color, bins=120, range=(-1, 1), histtype='step',
                color='k', label='after', density=True)
        ax.set_xlabel(f'{a.upper()} - {b.upper()}')
        ax.set_ylabel('density')
        ax.set_title(f'{a.upper()}-{b.upper()}  median {np.median(color):+.3f}',
                     fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def _retag(name, tag):
    """Replace the date token in a default filename with ``tag`` (or append)."""
    if tag is None:
        return name
    import re
    stem, ext = os.path.splitext(name)
    stem = re.sub(r'_\d{4}-\d{2}-\d{2}$', '', stem)
    return f'{stem}_{tag}{ext}'


def print_metric_summary(after):
    print('\n=== saturation-continuity metric (after catalog) ===')
    for a, b in MULTIPAIR + [CALIBRATED_PAIR]:
        s, r = _metric_str(after, a, b)
        w = r['worst_bin']
        extra = '' if w is None else f" @mag_{b}={w['magB_lo']:.1f}"
        print(f'  {a}-{b:6s}: {s}{extra}')
    print('=== degenerate-pair flatness ===')
    for a, b in DEGENERATE_PAIRS:
        r = degenerate_pair_flatness(after, a, b)
        m = 'n/a' if not np.isfinite(r['metric']) else f"{r['metric']:.3f}"
        print(f'  {a}-{b}: drift {m}  plateau {r["plateau"]:+.3f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--after', required=True,
                    help='combined all-band catalog with the rebuilt satstar channel')
    ap.add_argument('--before', default=None,
                    help='optional production catalog for before/after rows')
    ap.add_argument('--outdir', default=DEFAULT_OUTDIR)
    ap.add_argument('--tag', default=None,
                    help='date token for output filenames; default keeps the '
                         'exact names the LaTeX references')
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f'reading after: {args.after}')
    after = Table.read(args.after)
    before = None
    if args.before:
        print(f'reading before: {args.before}')
        before = Table.read(args.before)

    print_metric_summary(after)

    def out(key):
        return os.path.join(args.outdir, _retag(FIGNAMES[key], args.tag))

    print('\nwriting figures:')
    fig_calibrated(after, out('calibrated'));      print(' ', out('calibrated'))
    fig_multipair(after, before, out('multipair')); print(' ', out('multipair'))
    fig_class(after, out('class'));                 print(' ', out('class'))
    fig_degen_trends(after, before, out('degen_trend')); print(' ', out('degen_trend'))
    fig_degen_hists(after, before, out('degen_hist'));   print(' ', out('degen_hist'))
    print('done.')


if __name__ == '__main__':
    main()
