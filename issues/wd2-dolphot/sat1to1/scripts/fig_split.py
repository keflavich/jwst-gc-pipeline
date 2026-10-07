"""Figure: complementary split rows of one star in an arm's m8 catalog.  Per example: mosaic cutouts of the
creating band and F410M with the dolphot star and both rows marked, and a band strip showing which row holds each
band and its dm against dolphot.  usage: python fig_split.py ARM [N]"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import search_around_sky, SkyCoord
from astropy.visualization import simple_norm
import astropy.units as u
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

arm = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 6
an.ZPWIN.update(an.zp_windows())
A = an.Arm(arm)
cat = A.cat
bands = [b for b in an.BANDS if f'mag_vega_f{b.lower()}' in cat.colnames]
M = {b: np.asarray(an.fl(cat[f'mag_vega_f{b.lower()}']), float) for b in bands}
F = np.array([np.isfinite(M[b]) for b in bands]).T
REP = np.array([np.ma.filled(cat[f'replaced_saturated_f{b.lower()}'], False).astype(bool)
                if f'replaced_saturated_f{b.lower()}' in cat.colnames else np.zeros(len(cat), bool) for b in bands]).T
mk = np.where(A.matched)[0]
mrow = A.idx[mk]
i1, i2, sep, _ = search_around_sky(A.sky[mrow], A.sky, 0.08 * u.arcsec)
keep = mrow[i1] != i2
i1, i2, sep = i1[keep], i2[keep], sep[keep]
disj = ~(F[mrow[i1]] & F[i2]).any(axis=1)
i1, i2, sep = i1[disj], i2[disj], sep[disj].to_value(u.mas)
best = {}
for a, b2, s in zip(i1, i2, sep):
    if a not in best or s < best[a][1]:
        best[a] = (b2, s)
reff = np.asarray(cat['skycoord_ref_filtername']).astype(str)
# examples: per separation bin, the pairs whose partner carries the most band values
ex = []
for lo, hi, n in ((0, 3, 2), (3, 20, 2), (20, 50, 2)):
    sel = [kv for kv in best.items() if lo <= kv[1][1] < hi]
    ex += sorted(sel, key=lambda kv: -F[kv[1][0]].sum())[:n]
ex = ex[:N]
D = '/orange/adamginsburg/jwst/wd2'
mos = {}


def cut(band, sc, size=1.2):
    if band not in mos:
        fn = f'{D}/{band.upper()}/pipeline/jw03523-o005_t001_nircam_clear-{band.lower()}-merged_i2d.fits'
        h = fits.open(fn, memmap=True)
        mos[band] = (h['SCI'].data, WCS(h['SCI'].header))
    data, w = mos[band]
    return Cutout2D(data, sc, size * u.arcsec, wcs=w, mode='partial', fill_value=np.nan)


fig = plt.figure(figsize=(17, 2.7 * len(ex)))
gs = fig.add_gridspec(len(ex), 3, width_ratios=[1, 1, 4.2], wspace=0.12, hspace=0.55, right=0.9)
norm_dm = TwoSlopeNorm(vcenter=0, vmin=-0.5, vmax=0.5)
for e, (a, (p, s)) in enumerate(ex):
    k = mk[a]
    r = mrow[a]
    dsc = SkyCoord(an.fl(A.m['RA'])[k], an.fl(A.m['DEC'])[k], unit='deg')
    for j, band in enumerate([reff[r].lower(), reff[p].lower()]):
        ax = fig.add_subplot(gs[e, j])
        c = cut(band, dsc)
        d = c.data
        fin = np.isfinite(d)
        ax.imshow(d, origin='lower', cmap='gray_r',
                  norm=simple_norm(d[fin], 'asinh', percent=99.5) if fin.any() else None)
        for sc, mkr, col, lab in ((dsc, 'o', 'red', 'dolphot'), (A.sky[r], '+', 'lime', 'row 1'),
                                  (A.sky[p], 'x', 'cyan', 'row 2')):
            x, y = c.wcs.world_to_pixel(sc)
            ax.plot(x, y, mkr, mfc='none', mec=col, color=col, ms=14 if mkr == 'o' else 10, mew=1.2, label=lab)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f'{band.upper()} (row {j + 1} band), 1.2"', fontsize=8)
        if e == 0 and j == 0:
            ax.legend(fontsize=7, loc='lower left', framealpha=0.6)
    ax = fig.add_subplot(gs[e, 2])
    img = np.full((2, len(bands)), np.nan)
    has = np.zeros((2, len(bands)), bool)
    for jj, b in enumerate(bands):
        for ii, row in enumerate((r, p)):
            if F[row, jj]:
                has[ii, jj] = True
                img[ii, jj] = M[b][row] - A.ref[b][k] - A.zp[b] if np.isfinite(A.ref[b][k]) else np.nan
    ax.imshow(np.where(has, 1, 0), cmap='Greys', vmin=0, vmax=4, aspect='auto')
    im = ax.imshow(img, cmap='RdBu_r', norm=norm_dm, aspect='auto')
    for ii in range(2):
        for jj in range(len(bands)):
            if has[ii, jj]:
                t = f'{img[ii, jj]:+.2f}' if np.isfinite(img[ii, jj]) else 'no ref'
                t += '*' if REP[(r, p)[ii], jj] else ''
                ax.text(jj, ii, t, ha='center', va='center', fontsize=7)
    ax.set_xticks(range(len(bands)))
    ax.set_xticklabels([f'F{b}' for b in bands], rotation=60, fontsize=7)
    ax.set_yticks([0, 1])
    ax.set_yticklabels([f'row 1 ({reff[r]})', f'row 2 ({reff[p]})'], fontsize=8)
    ax.set_title(f'rows {s:.1f} mas apart; dolphot F200W {A.ref["200W"][k]:.2f}; '
                 f'cell = ours - dolphot - ZP; * = satstar-replaced; blank = no value', fontsize=8)
cax = fig.add_axes([0.92, 0.35, 0.012, 0.3])
fig.colorbar(im, cax=cax, label='dm (mag)')
out = f'/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/sat1to1/fig/split_rows_{arm}.png'
fig.savefig(out, dpi=110, bbox_inches='tight')
print('wrote', out)
