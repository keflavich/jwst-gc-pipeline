"""Figure for the #1150 review item: within-detector satstar dm against 2.5 log10(flat), F150W/F200W x nrcb1/nrcb3.
Rows: (top) transposed-grid refit (rstar/out7f), (bottom) fixed-grid refit (flatarea/out7g, #1154 loader).
Each panel: per-star dm after per-bin median removal, without the flat (H, grey) and with it (Hf, blue); the bottom row
also subtracts pred (the dolphot pixel-area double count, #1153).  Lines: Theil-Sen fits.
usage: python fig_flatarea.py out.png"""
import re
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import theilslopes
from astropy.coordinates import SkyCoord
import astropy.units as u
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sys.path.insert(0, f'{Q}/satrefit/capbind')
sys.path.insert(0, f'{Q}/satrefit/rstar')
sys.path.insert(0, Q)
from cb_lib import Band
from an4 import REFV
import score7f
import analyze as an

DIRS = {'transposed grid (out7f)': f'{Q}/satrefit/rstar/out7f', 'fixed grid (out7g), minus pred': f'{Q}/flatarea/out7g'}
P = np.load(f'{Q}/photomver/areapred_main2.npz')
PT = np.load(f'{Q}/psfsum/predT_main2.npz')
M = an.Table.read(an.PATH['main2'][1])
dsky = SkyCoord(np.asarray(M['RA'], float) * u.deg, np.asarray(M['DEC'], float) * u.deg)


def subset_arrays(band, outd, sub_pred):
    score7f.D = outd
    B = Band(band)
    col, G = score7f.load(B)
    det_r = np.array([re.search(r'_(nrc\w+?)_align', B.map['files'][k]).group(1) for k in B.frame_of])
    fl = col('flat_rimw')
    D = {}
    for nm in ('H', 'Hf'):
        c = score7f.caps(B, G, nm) * B.rcor
        c = np.where(np.isfinite(c), c, np.inf)
        D[nm] = B.dm_of(np.minimum(col('a_' + nm), c)) - REFV[band]
    isb3 = B.med_per_star((det_r == 'nrcb3').astype(float))
    sfl = B.med_per_star(fl)
    ra, dec = B.med_per_star(B.col('ra')), B.med_per_star(B.col('dec'))
    okp = np.isfinite(ra) & np.isfinite(dec)
    jj = np.full(B.n, -1)
    idx, sep, _ = SkyCoord(ra[okp] * u.deg, dec[okp] * u.deg).match_to_catalog_sky(dsky)
    jj[np.flatnonzero(okp)[sep.arcsec < 0.06]] = idx[sep.arcsec < 0.06]
    p = np.where(jj >= 0, P[band][np.maximum(jj, 0)], np.nan)
    pt = np.where(jj >= 0, PT[band][np.maximum(jj, 0)], np.nan)
    bins = score7f.BINS[band]
    out = {}
    for det, v in (('nrcb1', 0), ('nrcb3', 1)):
        s = (B.have0 & (isb3 == v) & np.isfinite(sfl) & np.isfinite(D['H']) & np.isfinite(D['Hf'])
             & (B.ref >= bins[0][0]) & (B.ref < 17) & np.isfinite(2 * p - pt))
        x = 2.5 * np.log10(sfl[s])
        Y = {}
        for nm in ('H', 'Hf'):
            y = D[nm][s].copy()
            for lo, hi in bins:
                b = (B.ref[s] >= lo) & (B.ref[s] < hi)
                if b.any():
                    y[b] -= np.median(y[b])
            # same order as satflat_area.py: bin medians first, then pred
            Y[nm] = y - (p[s] - np.median(p[s])) if sub_pred else y
        out[det] = (x, Y)
    return out


fig, axes = plt.subplots(2, 4, figsize=(15, 7.5), sharex=True, sharey=True, constrained_layout=True)
for r, (lab, outd) in enumerate(DIRS.items()):
    sub_pred = 'minus pred' in lab
    for c0, band in enumerate(('150W', '200W')):
        arr = subset_arrays(band, outd, sub_pred)
        for c1, det in enumerate(('nrcb1', 'nrcb3')):
            ax = axes[r, 2 * c0 + c1]
            x, Y = arr[det]
            xx = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 2)
            txt = []
            for nm, colr in (('H', '0.55'), ('Hf', 'tab:blue')):
                ax.scatter(x, Y[nm], s=6, color=colr, alpha=0.45, lw=0)
                s_, i_, l1, l2 = theilslopes(Y[nm], x)
                ax.plot(xx, i_ + s_ * xx, color=colr, lw=2)
                txt.append(f'{nm}: {s_:+.2f} [{l1:+.2f}, {l2:+.2f}]')
            ax.plot(xx, -xx + np.median(Y['H']), color='k', ls=':', lw=1)
            ax.axhline(0, color='0.8', lw=0.8, zorder=0)
            ax.text(0.03, 0.97, '\n'.join(txt), transform=ax.transAxes, va='top', fontsize=9)
            ax.set_title(f'F{band} {det} ({len(x)} stars)', fontsize=10)
            if c0 == 0 and c1 == 0:
                ax.set_ylabel(f'{lab}\nsatstar dm, bin median removed')
            if r == 1:
                ax.set_xlabel('2.5 log10(flat), PSF$^2$-weighted rim')
axes[0, 0].set_ylim(-0.15, 0.15)
axes[0, 0].set_xlim(-0.08, 0.06)
for ax in axes.flat:
    ax.set_xticks([-0.06, -0.03, 0, 0.03])
fig.suptitle('Satstar dm against the pixel flat: H = rim without 1/flat, Hf = with 1/flat (#1150); dotted: slope -1', fontsize=11)
fig.savefig(sys.argv[1], dpi=110)
print('wrote', sys.argv[1])
