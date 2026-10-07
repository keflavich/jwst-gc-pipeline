"""Figure for the wd2 F150W registration offset.
(a) per-band median position offset of free-fit rows to dolphot;
(b) per-frame median (x_fit - x_init, y_fit - y_init) of bright free m7 fits;
(c) dm vs residual offset for F150W/F115W/F200W rows with forced_refit_frac >= 0.5.
usage: python fig_f150w.py"""
import glob
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

T = f'{an.Q}/tree_mainfcbg'
an.ZPWIN.update(an.zp_windows())
A = an.Arm('mainfcbg')
dsk = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)
B16 = ['115W', '150W', '162M', '164N', '182M', '187N', '200W', '212N',
       '250M', '277W', '300M', '323N', '335M', '405N', '410M', '466N']
off = {}
forced = {}
for b in B16:
    ref = A.ref[b]
    t = Table.read(f'{T}/catalogs/f{b.lower()}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
    sk = t['skycoord']
    fl = np.asarray(t['flux'], float)
    ok = np.isfinite(sk.ra.deg) & (fl > 0)
    t, sk, fl = t[ok], sk[ok], fl[ok]
    mi = -2.5 * np.log10(fl)
    rep = np.asarray(t['replaced_saturated'], bool)
    ff = np.asarray(t['forced_refit_frac'], float)
    have = np.where(np.isfinite(ref))[0]
    j, d, _ = dsk[have].match_to_catalog_sky(sk)
    hit = (d.arcsec < 0.08) & ~rep[j]
    mid = hit & (ref[have] >= 18.6) & (ref[have] < 21) & (d.arcsec < 0.05)
    zp = np.median(ref[have][mid] - mi[j][mid])
    dm = mi[j] + zp - ref[have]
    dra = (sk.ra.deg[j] - dsk.ra.deg[have]) * np.cos(np.deg2rad(dsk.dec.deg[have])) * 3.6e6
    dde = (sk.dec.deg[j] - dsk.dec.deg[have]) * 3.6e6
    z = hit & (ff[j] == 0)
    off[b] = (np.median(dra[z]), np.median(dde[z]))
    if b in ('115W', '150W', '200W'):
        r = np.hypot(dra - off[b][0], dde - off[b][1])
        q = hit & (ff[j] >= 0.5)
        forced[b] = (r[q], dm[q])
    print(b, off[b], flush=True)

drift = {}
for b in ('F150W', 'F162M', 'F200W'):
    out = []
    for f in sorted(glob.glob(f'{T}/{b}/{b.lower()}_nrc*_visit001_vgroup*_exp0000?_resbgsub_m7_daophot_basic.fits')):
        t = Table.read(f)
        fr = np.asarray(t['forced_refit'], bool)
        dx = np.asarray(t['x_fit'], float) - np.asarray(t['x_init'], float)
        dy = np.asarray(t['y_fit'], float) - np.asarray(t['y_init'], float)
        fl = np.asarray(t['flux_fit'], float)
        free = ~fr & np.isfinite(dx) & (fl > 0)
        br = free & (fl > np.nanpercentile(fl[free], 50))
        mod = 'A' if '_nrca' in f else 'B'
        out.append((np.median(dx[br]), np.median(dy[br]), mod))
    drift[b] = out

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
x0 = np.median([off[b][0] for b in B16 if b != '150W'])
y0 = np.median([off[b][1] for b in B16 if b != '150W'])
for b in B16:
    c = 'crimson' if b == '150W' else 'gray'
    ax[0].plot(off[b][0] - x0, off[b][1] - y0, 'o', color=c, ms=8 if b == '150W' else 5)
    if b == '150W':
        ax[0].annotate('F150W', (off[b][0] - x0, off[b][1] - y0), xytext=(6, 6), textcoords='offset points', color=c)
ax[0].annotate('other 15 bands', (1, -1), xytext=(4, -5), textcoords='data', color='gray',
               arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
ax[0].axhline(0, color='k', lw=0.5); ax[0].axvline(0, color='k', lw=0.5)
ax[0].set_xlabel('dRA* (mas), relative to median of other bands')
ax[0].set_ylabel('dDec (mas)')
ax[0].set_title('(a) per-band offset to dolphot, free-fit rows')
ax[0].set_aspect('equal')
mk = {'F150W': 'crimson', 'F162M': 'tab:blue', 'F200W': 'gray'}
for b, out in drift.items():
    for k, (dx, dy, mod) in enumerate(out):
        ax[1].plot(dx, dy, 'o' if mod == 'A' else 's', color=mk[b], ms=5, alpha=0.8,
                   label=f'{b}' if k == 0 else None)
ax[1].axhline(0, color='k', lw=0.5); ax[1].axvline(0, color='k', lw=0.5)
ax[1].set_xlabel('median x_fit - x_init (px)')
ax[1].set_ylabel('median y_fit - y_init (px)')
ax[1].set_title('(b) m7 free-fit drift from cross-band seed\nper frame (o module A, s module B)')
ax[1].legend(loc='center left')
ax[1].set_aspect('equal')
bins = np.array([0, 5, 10, 15, 20, 30, 45, 80])
for b, c in (('150W', 'crimson'), ('115W', 'tab:orange'), ('200W', 'gray')):
    r, dm = forced[b]
    ax[2].plot(r, dm, '.', color=c, alpha=0.3, ms=3)
    med = [np.median(dm[(r >= lo) & (r < hi)]) if ((r >= lo) & (r < hi)).sum() >= 5 else np.nan
           for lo, hi in zip(bins[:-1], bins[1:])]
    ax[2].plot(0.5 * (bins[:-1] + bins[1:]), med, '-o', color=c, label=f'F{b} (N={len(r)})')
ax[2].axhline(0, color='k', lw=0.5)
ax[2].set_ylim(-2, 3)
ax[2].set_xlim(0, 100)
ax[2].set_xlabel('row position - dolphot - band median offset (mas)')
ax[2].set_ylabel('dm = pipeline - dolphot (mag)')
ax[2].set_title('(c) rows with forced_refit_frac >= 0.5\n(binned medians)')
ax[2].legend()
fig.tight_layout()
fig.savefig('fig_f150w_offset.png', dpi=110)
print('wrote fig_f150w_offset.png')
