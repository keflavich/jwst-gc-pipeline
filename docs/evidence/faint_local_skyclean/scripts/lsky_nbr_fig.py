"""Realness of the #1019 additions and of base-kept stars against distance to
the nearest brighter base-kept star, per S/N range, from lsky_snr_diag.py
outputs (top row: reference-catalog match); also the distance-matched
expectation for the additions.  When img_realness_<field>.json exists, a
second row shows the image-peak realness from img_realness.py (a local
maximum of the reference image within 1.5 px, chance-corrected).

usage: python lsky_nbr_fig.py <out.png> [field ...]
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {'brick': 'Brick F182M (ref: F200W, independent visit)',
         'sgrb2': 'Sgr B2 F187N (ref: F182M, same visit)',
         'w51': 'W51 F187N (ref: F182M, same visit)'}
MIN_N = 20     # plot a bin only with at least this many sources


def matched_expectation(rows):
    """Added-weighted mean of the base-kept realness over distance bins
    (bins with a base-kept value), and the added realness over the same bins."""
    num_e = num_a = den = 0.0
    for r in rows:
        if r['n_added'] == 0 or r['rel_basekept'] is None or r['rel_added'] is None:
            continue
        num_e += r['n_added'] * r['rel_basekept']
        num_a += r['n_added'] * r['rel_added']
        den += r['n_added']
    return (num_a / den, num_e / den, int(den)) if den else (np.nan, np.nan, 0)


def matched_image(rows):
    """img_realness.py rows: added-weighted image realness of the additions
    and of base-kept stars over the same distance bins."""
    num_e = num_a = den = 0.0
    for r in rows:
        a, b = r['added'], r['basekept']
        if a.get('img_rel') is None or b.get('img_rel') is None:
            continue
        num_a += a['n'] * a['img_rel']
        num_e += a['n'] * b['img_rel']
        den += a['n']
    return (num_a / den, num_e / den, int(den)) if den else (np.nan, np.nan, 0)


def main(out, *fields):
    fields = fields or ('brick', 'sgrb2', 'w51')
    have_img = all(os.path.exists(f'{HERE}/img_realness_{f}.json') for f in fields)
    nrow = 2 if have_img else 1
    fig, axes = plt.subplots(nrow, len(fields), figsize=(5.2 * len(fields), 4.2 * nrow), sharey='row',
                             layout='constrained', squeeze=False)
    for ax, f in zip(axes[0], fields):
        with open(f'{HERE}/lsky_snr_diag_{f}.json') as fh:
            d = json.load(fh)['by_neighbour_distance']
        for c, (key, rows) in zip(('C0', 'C1', 'C3'), d.items()):
            lab = [f'{r["d_lo"]:g}-{r["d_hi"]:g}' if np.isfinite(r['d_hi']) else f'>{r["d_lo"]:g}' for r in rows]
            x = np.arange(len(rows))
            ya = [r['rel_added'] if r['n_added'] >= MIN_N and r['rel_added'] is not None else np.nan for r in rows]
            yb = [r['rel_basekept'] if r['n_basekept'] >= MIN_N and r['rel_basekept'] is not None else np.nan for r in rows]
            ra, re, n = matched_expectation(rows)
            ax.plot(x, yb, '--', color=c, lw=1, marker='s', ms=3,
                    label=f'base-kept, S/N {key}')
            ax.plot(x, ya, '-', color=c, lw=2, marker='o', ms=5,
                    label=f'added, S/N {key}: {ra:.2f} vs {re:.2f} expected (n {n})')
            ax.set_xticks(x, lab)
        ax.axhline(0, color='k', lw=0.5)
        ax.axhline(1, color='0.6', lw=0.5)
        ax.set_xlabel('distance to nearest brighter base-kept star (px)')
        ax.set_title(NAMES.get(f, f), fontsize=9)
        ax.legend(fontsize=6.5, loc='lower right')
    axes[0, 0].set_ylabel('catalog realness (chance-corrected match rate\nrelative to base-kept of the same flux)')
    if have_img:
        for ax, f in zip(axes[1], fields):
            with open(f'{HERE}/img_realness_{f}.json') as fh:
                d = json.load(fh)
            for c, (key, rows) in zip(('C0', 'C1', 'C3'), d.items()):
                lab = [f'{r["d_lo"]:g}-{r["d_hi"]:g}' if np.isfinite(r['d_hi']) else f'>{r["d_lo"]:g}' for r in rows]
                x = np.arange(len(rows))
                ya = [r['added']['img_rel'] if r['added']['n'] >= MIN_N and 'img_rel' in r['added'] else np.nan
                      for r in rows]
                yb = [r['basekept']['img_rel'] if r['basekept']['n'] >= MIN_N and 'img_rel' in r['basekept']
                      else np.nan for r in rows]
                ax.plot(x, yb, '--', color=c, lw=1, marker='s', ms=3, label=f'base-kept, S/N {key}')
                ra, re, n = matched_image(rows)
                ax.plot(x, ya, '-', color=c, lw=2, marker='o', ms=5,
                        label=f'added, S/N {key}: {ra:.2f} vs {re:.2f} expected (n {n})')
                ax.set_xticks(x, lab)
            ax.axhline(0, color='k', lw=0.5)
            ax.axhline(1, color='0.6', lw=0.5)
            ax.set_ylim(-0.1, 1.1)
            ax.set_xlabel('distance to nearest brighter base-kept star (px)')
            ax.set_title(NAMES.get(f, f).split(' (')[0] + ': reference IMAGE has a peak within 1.5 px', fontsize=9)
            ax.legend(fontsize=6.5, loc='lower right')
        axes[1, 0].set_ylabel('image realness (chance-corrected fraction\nwith a reference-image local maximum)')
    fig.suptitle('#1019 local sky-clean additions vs base-kept stars, by distance to a brighter star '
                 f'(bins with >= {MIN_N} sources; "expected" = base-kept realness weighted by the '
                 'additions\' distance distribution)', fontsize=8.5)
    fig.savefig(out, dpi=110)
    print(out)
    for f in fields:
        with open(f'{HERE}/lsky_snr_diag_{f}.json') as fh:
            d = json.load(fh)['by_neighbour_distance']
        for key, rows in d.items():
            ra, re, n = matched_expectation(rows)
            print(f'{f} S/N {key}: added {ra:.2f}, distance-matched expectation {re:.2f}, ratio {ra / re:.2f} (n {n})')
        if have_img:
            with open(f'{HERE}/img_realness_{f}.json') as fh:
                d = json.load(fh)
            for key, rows in d.items():
                ra, re, n = matched_image(rows)
                print(f'{f} S/N {key} image: added {ra:.2f}, distance-matched expectation {re:.2f}, '
                      f'ratio {ra / re:.2f} (n {n})')


if __name__ == '__main__':
    main(*sys.argv[1:])
