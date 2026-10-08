"""Aggregate injection-recovery truth tables -> bias tables + fig_ast_bias.png"""
import glob, os, sys
import numpy as np
from astropy.table import Table, vstack
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
FWHM = {'F410M': 2.179, 'F405N': 2.165, 'F277W': 1.444, 'F200W': 2.141}
RANGE = {'F410M': (16, 23), 'F405N': (15, 22), 'F277W': (16, 23), 'F200W': (16, 23)}
BANDS = ['F200W', 'F277W', 'F410M', 'F405N']
VARS = ['raw', 'resbg']
MATCHED = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/matched_Q_mainfcbg.fits'


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


def stats(mag_in, dm, matched, rec, edges, minn=10):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (mag_in >= lo) & (mag_in < hi)
        m = s & matched
        n = s.sum()
        rows.append((0.5 * (lo + hi), n, m.sum() / max(n, 1), rec[s].sum() / max(n, 1),
                     np.median(dm[m]) if m.sum() >= minn else np.nan,
                     mad(dm[m]) if m.sum() >= minn else np.nan))
    return np.array(rows)


def load(band, var):
    fns = sorted(glob.glob(f'{HERE}/truth/{band}_*_{var}_truth.fits'))
    ts = [Table.read(f) for f in fns]
    return (vstack(ts) if ts else None), fns


def ours_minus_dolphot(band, edges):
    t = Table.read(MATCHED)
    b = band[1:]
    r = np.asarray(t[f'ref_{b}'], float); o = np.asarray(t[f'our_{b}'], float)
    sat = np.asarray(t[f'our_is_saturated_{b}']).astype(bool) | np.asarray(t[f'our_replaced_saturated_{b}']).astype(bool)
    ok = np.isfinite(r) & np.isfinite(o) & ~sat & (r < 90) & (o < 90)
    x = 0.5 * (r + o)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = ok & (x >= lo) & (x < hi)
        out.append((0.5 * (lo + hi), np.median(o[s] - r[s]) if s.sum() >= 20 else np.nan, s.sum()))
    return np.array(out)


def main():
    lines = []
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6), sharey=True)
    colors = {'raw': 'C0', 'resbg': 'C3'}
    labels = {'raw': 'injection: LocalBackground on crf (a)', 'resbg': 'injection: after smoothed-resid bg (b)'}
    allsplit = []
    for ax, band in zip(axes, BANDS):
        lo, hi = RANGE[band]
        edges = np.arange(lo, hi + 0.01, 0.5)
        od = ours_minus_dolphot(band, np.arange(14, 24.01, 0.5))
        ax.plot(od[:, 0], od[:, 1], 'k-s', ms=4, label='ours - dolphot (matched catalog, mean mag)')
        for var in VARS:
            t, fns = load(band, var)
            if t is None:
                continue
            mag_in = np.asarray(t['mag_in']); mo = np.asarray(t['mag_out'])
            dm = mo - mag_in
            matched = np.asarray(t['matched']).astype(bool); rec = np.asarray(t['recovered']).astype(bool)
            st = stats(mag_in, dm, matched, rec, edges)
            ok = np.isfinite(st[:, 4])
            ax.plot(st[ok, 0], st[ok, 4], '-o', color=colors[var], ms=4, label=labels[var])
            ax.fill_between(st[ok, 0], st[ok, 4] - st[ok, 5], st[ok, 4] + st[ok, 5], color=colors[var], alpha=0.15)
            lines.append(f'\n#### {band} variant={var} (frames: {", ".join(os.path.basename(f).split("_"+var)[0] for f in fns)}; N_inj={len(t)})\n')
            lines.append('| mag_in | N_inj | matched frac | recovered frac (0.75 tol) | median dm (all matches) | 1.4826 MAD |')
            lines.append('|---|---|---|---|---|---|')
            for r in st:
                lines.append(f'| {r[0]:.2f} | {int(r[1])} | {r[2]:.2f} | {r[3]:.2f} | ' +
                             (f'{r[4]:+.3f} | {r[5]:.3f} |' if np.isfinite(r[4]) else 'n/a | n/a |'))
            # per-frame 19-21
            for f in fns:
                tt = Table.read(f)
                s = (tt['mag_in'] >= 19) & (tt['mag_in'] < 21) & tt['matched']
                lines.append(f'- per-frame {os.path.basename(f)}: matched {int(s.sum())}, median dm(19-21) = '
                             f'{np.median((tt["mag_out"]-tt["mag_in"])[s]):+.3f}')
            # splits, coarse 1-mag bins
            e1 = np.arange(lo, hi + 0.01, 1.0)
            sb = np.asarray(t['local_sb'], float)
            qs = np.nanpercentile(sb, [25, 50, 75])
            qi = np.digitize(sb, qs)
            near = np.asarray(t['d_nearest_base_pix'], float) < 3 * FWHM[band]
            splits = [(f'SB quartile {k+1}', qi == k) for k in range(4)] + \
                     [('nearest source < 3 FWHM', near), ('nearest source >= 3 FWHM', ~near)]
            lines.append(f'\nSplits for {band} {var} (median dm of all matches / matched / N_inj per 1-mag bin; SB quartile edges {qs[0]:.2f},{qs[1]:.2f},{qs[2]:.2f} MJy/sr):\n')
            lines.append('| subset | ' + ' | '.join(f'{a:.0f}-{b:.0f}' for a, b in zip(e1[:-1], e1[1:])) + ' |')
            lines.append('|---|' + '---|' * (len(e1) - 1))
            for name, sel in splits:
                st1 = stats(mag_in[sel], dm[sel], matched[sel], rec[sel], e1)
                cells = [(f'{r[4]:+.3f} ({r[2]*100:.0f}%, n={int(r[1])})' if np.isfinite(r[4]) else f'n/a ({r[2]*100:.0f}%, n={int(r[1])})') for r in st1]
                lines.append(f'| {name} | ' + ' | '.join(cells) + ' |')
        ax.axhline(0, color='gray', lw=0.8)
        ax.set_title(band); ax.set_xlabel('injected mag (Vega); ours-dolphot curve: mean mag')
        ax.set_xlim(lo - 0.5, hi + 0.5); ax.set_ylim(-0.4, 0.6)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel('median (mag_out - mag_in)  or  median (ours - dolphot) [mag]')
    axes[0].legend(fontsize=7, loc='upper left')
    fig.suptitle('wd2 injection-recovery (band shading = 1.4826 MAD of injection bias)')
    fig.tight_layout()
    fig.savefig(f'{HERE}/fig_ast_bias.png', dpi=140)
    open(f'{HERE}/bias_tables.md', 'w').write('\n'.join(lines))
    print('\n'.join(lines))

if __name__ == '__main__':
    main()
