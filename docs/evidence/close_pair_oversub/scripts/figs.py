"""Figures for the close-pair over-subtraction issue, from nn_oversub.py outputs.

curves.png : fraction of m7 vetted stars with residual S/N < -5 against the
             nearest-neighbour separation (in FWHM), every field (after) plus
             brick before (dashed); cumulative fraction of stars with a
             neighbour closer than x; neighbour flux ratio per separation bin.
gallery_<field>_<filt>.png : data | m7 residual stamps (catalogue positions
             marked) at random over-subtracted stars in three separation bins.
usage: python figs.py <gallery-field> <FILT> <before|after>
"""
import glob
import json
import os
import sys

import numpy as np
from astropy.table import Table

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(S, '..', '..', 'phaseloss'))
from phase_loss import Mosaic  # noqa: E402

OUT = os.path.join(S, 'out')
HALF = 8
SEPBINS = [(0, 0.75), (0.75, 1.25), (1.25, 2.0)]


def curves():
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for f in sorted(f for f in glob.glob(os.path.join(OUT, '*.json')) if f.endswith(('_after.json', '_before.json'))):
        d = json.load(open(f))
        name = os.path.basename(f)[:-5]
        ls = '--' if d['which'] == 'before' else '-'
        b = d['bins']
        c = np.array([0.5 * (r['lo'] + min(r['hi'], 6.0)) for r in b])
        y = np.array([np.nan if r['frac_lt_m5'] is None or r['n'] < 20 else r['frac_lt_m5'] for r in b])
        ln, = ax[0].plot(c, y, ls, marker='o', ms=3, label=name)
        s = Table.read(f[:-5] + '_sample.fits')
        xs = np.sort(np.asarray(s['nn_fwhm']))
        ax[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs), ls, color=ln.get_color())
    ax[0].set_xlabel('nearest catalogued neighbour / FWHM')
    ax[0].set_ylabel('fraction with m7 residual S/N < -5')
    ax[0].axvline(0.5, color='0.5', lw=0.8, ls=':')
    ax[0].text(0.52, 0.02, 'seed dedup 0.5 FWHM', fontsize=7, color='0.4', rotation=90)
    ax[0].legend(fontsize=7)
    ax[1].set_xlim(0, 2.5)
    ax[1].set_ylim(0, 0.35)
    ax[1].set_xlabel('nearest catalogued neighbour / FWHM')
    ax[1].set_ylabel('cumulative fraction of m7 vetted stars')
    ax[1].set_title('stars with a neighbour closer than x (dashed = brick before)', fontsize=9)
    for k, (name, col) in enumerate((('brick_f182m_after', 'C1'), ('sgrc_f212n_after', 'C6'),
                                     ('brick_f405n_after', 'C5'))):
        s = Table.read(os.path.join(OUT, name + '_sample.fits'))
        r = np.log10(np.asarray(s['nbr_flux_ratio']))
        for j, (lo, hi) in enumerate(SEPBINS + [(3, 5)]):
            m = (s['nn_fwhm'] >= lo) & (s['nn_fwhm'] < hi) & np.isfinite(r)
            if m.sum() < 10:
                continue
            q = np.percentile(r[m], [16, 50, 84])
            xx = j + 0.25 * (k - 1)
            ax[2].errorbar(xx, q[1], yerr=[[q[1] - q[0]], [q[2] - q[1]]], fmt='o', color=col,
                           label=name if j == 0 else None)
    ax[2].set_xticks(range(4))
    ax[2].set_xticklabels([f'{lo}-{hi}' for lo, hi in SEPBINS + [(3, 5)]])
    ax[2].set_xlabel('separation bin / FWHM')
    ax[2].set_ylabel('log10 neighbour flux / own flux (16-50-84%)')
    ax[2].axhline(0, color='0.5', lw=0.8)
    ax[2].legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'curves.png'), dpi=100)


def stamps(mos, x, y):
    out = np.full((len(x), 2 * HALF + 1, 2 * HALF + 1), np.nan, np.float32)
    ny, nx = mos.data.shape
    for k, (xi, yi) in enumerate(zip(np.round(x).astype(int), np.round(y).astype(int))):
        y0, y1 = max(yi - HALF, 0), min(yi + HALF + 1, ny)
        x0, x1 = max(xi - HALF, 0), min(xi + HALF + 1, nx)
        out[k, y0 - (yi - HALF):y1 - (yi - HALF), x0 - (xi - HALF):x1 - (xi - HALF)] = mos.data[y0:y1, x0:x1]
    return out


def gallery(field, filt, which, ncol=6):
    name = f'{field}_{filt.lower()}_{which}'
    d = json.load(open(os.path.join(OUT, name + '.json')))
    s = Table.read(os.path.join(OUT, name + '_sample.fits'))
    snap = json.load(open(os.path.join(S, '..', '..', 'prod_before', field, filt.lower(), 'snapshot.json')))
    datap = snap['files']['data']['path']
    if d['resid'].startswith('/blue/') and datap.startswith('/orange/'):
        datap = datap.replace('/orange/adamginsburg/jwst/brick', '/blue/adamginsburg/adamginsburg/jwst/brick', 1)
    rng = np.random.default_rng(2)
    rows = []
    for lo, hi in SEPBINS:
        m = np.flatnonzero((s['nn_fwhm'] >= lo) & (s['nn_fwhm'] < hi) & (s['snr'] < -5))
        rows.append(rng.choice(m, size=min(ncol, len(m)), replace=False))
    allx = np.asarray(s['x'])
    ally = np.asarray(s['y'])
    t = Table.read(d['vetted'])
    res = Mosaic(d['resid'])
    from astropy.coordinates import SkyCoord
    cx, cy = res.wcs.world_to_pixel(SkyCoord(t['skycoord']))
    dat = Mosaic(datap)
    fig, axs = plt.subplots(2 * len(SEPBINS), ncol, figsize=(1.9 * ncol, 3.9 * len(SEPBINS)))
    for r, (idx, (lo, hi)) in enumerate(zip(rows, SEPBINS)):
        xs, ys = allx[idx], ally[idx]
        sd, sr = stamps(dat, xs, ys), stamps(res, xs, ys)
        for c in range(ncol):
            for k, (st, lab) in enumerate(((sd, 'data'), (sr, 'm7 resid'))):
                ax = axs[2 * r + k, c]
                ax.set_xticks([])
                ax.set_yticks([])
                if c >= len(idx):
                    ax.axis('off')
                    continue
                med = np.nanmedian(sd[c])
                w = max(np.nanmax(sd[c][HALF - 2:HALF + 3, HALF - 2:HALF + 3]) - med, 1e-6)
                ax.imshow(st[c], origin='lower', cmap='gray', vmin=med - 0.5 * w, vmax=med + w,
                          interpolation='nearest')
                near = (np.abs(cx - xs[c]) <= HALF) & (np.abs(cy - ys[c]) <= HALF)
                ax.plot(cx[near] - np.round(xs[c]) + HALF, cy[near] - np.round(ys[c]) + HALF,
                        'x', color='orange', ms=5, mew=1.2)
                if k == 1:
                    ax.text(0.03, 0.03, f'S/N {s["snr"][idx[c]]:.0f}\nsep {s["nn_fwhm"][idx[c]]:.2f}',
                            transform=ax.transAxes, color='yellow', fontsize=7)
                if c == 0:
                    ax.set_ylabel(f'{lo}-{hi} FWHM\n{lab}', fontsize=8)
    fig.suptitle(f'{field} {filt} m7 ({which}): random over-subtracted (resid S/N < -5) vetted stars\n'
                 f'by nearest-neighbour separation; orange x = m7 vetted sources;\n'
                 f'data and residual share one stretch per column', fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(os.path.join(OUT, f'gallery_{name}.png'), dpi=100)
    res.close()
    dat.close()


if __name__ == '__main__':
    curves()
    if len(sys.argv) > 3:
        gallery(*sys.argv[1:4])
