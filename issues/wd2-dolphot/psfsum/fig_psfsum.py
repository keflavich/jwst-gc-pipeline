"""Figure for the STPSF grid transposition issue.

(a) PSF plane sum vs SIAF pixel area at the plane's assigned position, for all
    82 fovp101 grids in the wd2 PSF store loaded with stpsf.utils
    .to_griddedpsfmodel and with psf_grid_io.load_stpsf_grid, and for the
    dolphot (webbpsf 1.2.1) PSF library.  Both axes in mag relative to each
    grid's median; a plane at its correct position follows slope 1.
(b) F200W NRCA1 plane assigned to detector (x, y) = (2047, 0) by each loader,
    and their difference.
(c) main2 vs dolphot: binned median of dm - pred against pred - predT.
"""
import glob
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from stpsf.utils import to_griddedpsfmodel

from jwst_gc_pipeline.photometry.psf_grid_io import load_stpsf_grid
from siafarea import siaf_area

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402

warnings.simplefilter('ignore')
P = '/orange/adamginsburg/jwst/wd2/psfs'
C_OLD, C_NEW, C_DOL = '#8c8c8c', '#1f6fb4', '#d9730d'


def rel(grid, det):
    xy = np.asarray(grid.grid_xypos)
    s = grid.data.sum(axis=(1, 2))
    a = siaf_area(det, xy[:, 0], xy[:, 1])
    return 2.5 * np.log10(a / np.median(a)), -2.5 * np.log10(s / np.median(s))


fig = plt.figure(figsize=(15, 9.5))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1], hspace=0.35, wspace=0.3)

# (a)
ax = fig.add_subplot(gs[0, :2])
xo, yo, xn, yn = [], [], [], []
for fn in sorted(glob.glob(f'{P}/nircam_nrc*_f*_fovp101_samp2_npsf16.fits')):
    det = os.path.basename(fn).split('_')[1]
    a, s = rel(to_griddedpsfmodel(fn), det)
    xo += list(a); yo += list(s)
    a, s = rel(load_stpsf_grid(fn), det)
    xn += list(a); yn += list(s)
xd, yd = [], []
pos = np.array([0, 512, 1024, 1536, 2047.])
gy, gx = np.meshgrid(pos, pos, indexing='ij')
for fn in sorted(glob.glob('dolsrc/nircam/data/*.psf')):
    det = os.path.basename(fn).split('.')[1]
    d = np.fromfile(fn, dtype='>f4').reshape(5, 5, 7, 7, 49, 49)
    s = d[:, :, 1:6, 1:6].sum(axis=(-2, -1)).mean(axis=(-2, -1)).ravel()
    a = siaf_area(det, gx.ravel(), gy.ravel())
    xd += list(2.5 * np.log10(a / np.median(a))); yd += list(-2.5 * np.log10(s / np.median(s)))
ax.plot([-0.03, 0.03], [-0.03, 0.03], color='k', lw=1, zorder=0)
ax.text(0.024, 0.027, 'slope 1', fontsize=9, ha='right')
ax.scatter(xo, yo, s=10, color=C_OLD, alpha=0.6, label=f'stpsf to_griddedpsfmodel ({len(xo)} planes)')
ax.scatter(xn, yn, s=10, color=C_NEW, alpha=0.8, label=f'psf_grid_io.load_stpsf_grid ({len(xn)} planes)')
ax.scatter(xd, yd, s=14, marker='s', facecolor='none', edgecolor=C_DOL, lw=1,
           label=f'dolphot webbpsf 1.2.1 library ({len(xd)} positions)')
ax.set_xlabel('2.5 log10(SIAF pixel area at assigned (x, y) / median)  [mag]')
ax.set_ylabel('-2.5 log10(PSF plane sum / median)  [mag]')
ax.set_title('(a) PSF plane sum vs pixel area at the position the loader assigns', fontsize=11)
ax.legend(fontsize=9, loc='upper left', frameon=False)
ax.set_xlim(-0.03, 0.03); ax.set_ylim(-0.03, 0.03)
ax.grid(alpha=0.3)

# (b)
fn = f'{P}/nircam_nrca1_f200w_fovp101_samp2_npsf16.fits'
go, gn = to_griddedpsfmodel(fn), load_stpsf_grid(fn)


def plane(g, x, y):
    xy = [tuple(p) for p in np.asarray(g.grid_xypos)]
    return g.data[xy.index((x, y))]


po, pn = plane(go, 2047.0, 0.0), plane(gn, 2047.0, 0.0)
c = po.shape[0] // 2
h = 12
sl = (slice(c - h, c + h + 1), slice(c - h, c + h + 1))
pk = pn.max()
for k, (img, title) in enumerate([(pn, 'load_stpsf_grid'), (po, 'to_griddedpsfmodel'),
                                  ((pn - po), 'difference')]):
    axk = fig.add_subplot(gs[0, 2] if k == 0 else (gs[0, 3] if k == 1 else gs[1, 3]))
    if k < 2:
        im = axk.imshow(np.log10(np.clip(img[sl] / pk, 1e-4, None)), origin='lower',
                        cmap='magma', vmin=-3, vmax=0)
        axk.set_title(f'(b) F200W NRCA1 plane at (2047, 0)\n{title}, sum {img.sum() / 4:.4f}',
                      fontsize=9)
        plt.colorbar(im, ax=axk, fraction=0.046, label='log10(plane / peak)')
    else:
        v = np.abs(img[sl]).max() / pk
        im = axk.imshow(img[sl] / pk, origin='lower', cmap='RdBu_r', vmin=-v, vmax=v)
        axk.set_title('(b) load_stpsf_grid - to_griddedpsfmodel\n(fraction of peak)', fontsize=9)
        plt.colorbar(im, ax=axk, fraction=0.046)
    axk.set_xticks([]); axk.set_yticks([])
    axk.set_xlabel(f'{2 * h + 1} x {2 * h + 1} oversampled px (x2)', fontsize=8)

# (c)
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
PA = np.load(f'{an.Q}/photomver/areapred_main2.npz')
PT = np.load('predT_main2.npz')
ax = fig.add_subplot(gs[1, :3])
ax.plot([-0.04, 0.04], [-0.04, 0.04], color='k', lw=1, zorder=0)
ax.text(0.033, 0.036, 'slope 1', fontsize=9, ha='right')
cols = {'200W': '#1f6fb4', '212N': '#2a9d8f', '300M': '#d9730d', '323N': '#9b5de5'}
for b, col in cols.items():
    dm, pr, pt = A.dm(b), PA[b], PT[b]
    lo, hi = an.ZPWIN[b]
    ok = (A.matched & np.isfinite(dm) & np.isfinite(pr) & np.isfinite(pt) & ~A.rep[b] & ~A.sat[b]
          & (A.ref[b] >= lo) & (A.ref[b] < hi))
    x, y = (pr - pt)[ok], (dm - pr)[ok]
    y = y - np.median(y)
    edges = np.percentile(x, np.linspace(0, 100, 11))
    xc, yc, ye = [], [], []
    for e0, e1 in zip(edges[:-1], edges[1:]):
        m = (x >= e0) & (x <= e1)
        xc.append(np.median(x[m])); yc.append(np.median(y[m]))
        ye.append(1.253 * an.mad(y[m]) / np.sqrt(m.sum()))
    sl_ = np.polyfit(x, y, 1)[0]
    ax.errorbar(xc, yc, ye, color=col, marker='o', ms=5, lw=1.5, capsize=0,
                label=f'F{b}  (N={ok.sum()}, slope {sl_:+.2f})')
ax.set_xlabel('pred - predT = 2.5 log10(AREA(x, y) / AREA(y, x))  [mag]')
ax.set_ylabel('dm - pred  (ours - dolphot - ZP - pred)  [mag]')
ax.set_title('(c) main2 vs dolphot after removing dolphot\'s area double count: residual follows the transposition term',
             fontsize=11)
ax.legend(fontsize=9, frameon=False, loc='upper left')
ax.set_xlim(-0.04, 0.04); ax.set_ylim(-0.04, 0.04)
ax.grid(alpha=0.3)
fig.savefig('psfsum.png', dpi=110, bbox_inches='tight')
print('wrote psfsum.png')
