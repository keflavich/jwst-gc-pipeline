"""A/B cutouts for the LW core gap: bright dolphot stars with no row in the
mainfcbg m7 catalog, shown in the F{b} mosaic and in each arm's per-frame m7
residual, with each arm's merged m7 rows marked.
usage: python cutouts_ab.py BAND N NAME=TREE [NAME=TREE ...]
  BAND: 250M, 277W or 300M; TREE: tree root holding catalogs/ and F{b}/"""
import glob
import sys
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.visualization import simple_norm
import astropy.units as u
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

b = sys.argv[1].upper().lstrip('F')
N = int(sys.argv[2])
arms = [a.split('=', 1) for a in sys.argv[3:]]
bl = 'f' + b.lower()
G = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/f277w_gap'
an.ZPWIN.update(an.zp_windows())
A = an.Arm('mainfcbg')
dsk = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)
ref = A.ref[b]
mid = np.where(np.isfinite(ref) & (ref >= 18.6) & (ref < 21))[0]
tr = Table.read(f'{G}/trace_mainfcbg_{b}.ecsv')
rs = SkyCoord(tr['ra'] * u.deg, tr['dec'] * u.deg)


def load_rows(T):
    t = Table.read(f'{T}/catalogs/{bl}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
    sk = t['skycoord']
    fl = np.asarray(t['flux'], float)
    ok = np.isfinite(sk.ra.deg) & (fl > 0)
    t, sk, fl = t[ok], sk[ok], fl[ok]
    mi = -2.5 * np.log10(fl)
    rep = np.asarray(t['replaced_saturated'], bool) if 'replaced_saturated' in t.colnames else np.zeros(len(t), bool)
    j, d, _ = dsk[mid].match_to_catalog_sky(sk)
    sel = (d.arcsec < 0.05) & ~rep[j]
    m = mi + np.median(ref[mid][sel] - mi[j][sel])
    # classify each row: matched to a dolphot star within 0.08" with |dm|<0.3,
    # matched with |dm|>=0.3, or no dolphot source within 0.1"
    jd, dd, _ = sk.match_to_catalog_sky(dsk)
    dm = m - ref[jd]
    cls = np.where(dd.arcsec > 0.1, 2, np.where((dd.arcsec < 0.08) & (np.abs(dm) < 0.3), 0, 1))
    return sk, m, cls


rows = {name: load_rows(T) for name, T in arms}
frames = {name: sorted(glob.glob(f'{T}/F{b}/pipeline/jw03523-o005_t001_nircam_clear-{bl}-nrc*_resbgsub_m7_daophot_basic_residual.fits'))
          for name, T in arms}
for name, fl in frames.items():
    assert len(fl) == 8, (name, len(fl))
mos = glob.glob(f'{arms[0][1]}/F{b}/pipeline/jw*-o005_t001_nircam_clear-{bl}-merged_i2d.fits')[0]
mdat, mw = fits.getdata(mos, 'SCI'), WCS(fits.getheader(mos, 'SCI'))
fw = [WCS(fits.getheader(fn, 'SCI')) for fn in frames[arms[0][0]]]


def frame_for(c):
    # first frame (in sorted order) whose finite-data footprint holds the star
    for k, w in enumerate(fw):
        x, y = w.world_to_pixel(c)
        if 20 < x < 2028 and 20 < y < 2028:
            return k
    return None


order = np.argsort(tr['ref_mag'])
pick = order[np.linspace(0, len(order) - 1, N).astype(int)]
size = 2.0 * u.arcsec
ncol = 1 + len(arms)
cmap = plt.get_cmap('gray').copy()
cmap.set_bad('steelblue')
fig, axes = plt.subplots(N, ncol, figsize=(2.9 * ncol, 3.0 * N), squeeze=False)
style = {0: dict(marker='+', s=70, c='lime', lw=1.2), 1: dict(marker='+', s=70, c='orange', lw=1.2),
         2: dict(marker='x', s=45, c='magenta', lw=1.0)}
for r, k in enumerate(pick):
    c = rs[k]
    co = Cutout2D(mdat, c, size, wcs=mw, mode='partial')
    vmin, vmax = np.nanpercentile(co.data, [1, 99.5])
    kf = frame_for(c)
    panels = [(f'F{b} mosaic', co, None)]
    for name, _ in arms:
        if kf is None:
            panels.append((name, None, name))
            continue
        fn = frames[name][kf]
        w = WCS(fits.getheader(fn, 'SCI'))
        panels.append((name, Cutout2D(fits.getdata(fn, 'SCI'), c, size, wcs=w, mode='partial'), name))
    for col, (lab, cut, name) in enumerate(panels):
        ax = axes[r, col]
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(lab if name is None else f'{lab}: m7 residual', fontsize=9)
        if cut is None:
            continue
        ax.imshow(cut.data, origin='lower', cmap=cmap, norm=simple_norm(cut.data, 'asinh', vmin=vmin, vmax=vmax))
        near = dsk.separation(c) < 1.2 * u.arcsec
        x, y = cut.wcs.world_to_pixel(dsk[near])
        ax.scatter(x, y, marker='o', s=60, facecolors='none', edgecolors='red', lw=1)
        x0, y0 = cut.wcs.world_to_pixel(c)
        ax.scatter([x0], [y0], marker='o', s=200, facecolors='none', edgecolors='yellow', lw=1.2)
        if name is None:
            ax.set_ylabel(f'dolphot F{b} {tr["ref_mag"][k]:.2f}', fontsize=9)
            continue
        sk, m, cls = rows[name]
        near = sk.separation(c) < 1.2 * u.arcsec
        for cl, kw in style.items():
            s = near & (cls == cl)
            if s.any():
                x, y = cut.wcs.world_to_pixel(sk[s])
                ax.scatter(x, y, **kw)
        d = sk.separation(c)
        i = int(np.argmin(d.arcsec))
        txt = (f'dm {m[i] - tr["ref_mag"][k]:+.2f}' if d[i].arcsec < 0.08 else 'no row')
        ax.text(0.03, 0.03, txt, transform=ax.transAxes, color='yellow', fontsize=9,
                bbox=dict(facecolor='black', alpha=0.6, lw=0))
fig.suptitle(f'F{b}: bright dolphot stars with no mainfcbg m7 row (yellow); 2" cutouts\n'
             'red o dolphot; blue = NaN; arm m7 rows: lime + |dm|<0.3, orange + |dm|>=0.3, magenta x no dolphot within 0.1"',
             fontsize=9)
fig.tight_layout()
out = f'cutouts_ab_{b}.png'
fig.savefig(out, dpi=90)
print('wrote', out)
