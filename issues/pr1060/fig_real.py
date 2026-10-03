"""Empirical before/after for #1055 / PR #1060: one real saturated star in
jw10678107001_02101_00001_nrcblong_destreak_o107_crf (F480M), satstar run on
main (base/) and on the #1060 branch (fixed/) with identical inputs."""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, AsinhNorm
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import cKDTree

STEM = 'jw10678107001_02101_00001_nrcblong_destreak_o107_crf'
XS, YS = float(sys.argv[1]), float(sys.argv[2])
H = 60

sci = fits.getdata(f'base/{STEM}.fits', 'SCI')
res = {k: fits.getdata(f'{k}/{STEM}_satstar_residual.fits') for k in ('base', 'fixed')}
mod = {k: fits.getdata(f'{k}/{STEM}_satstar_model.fits') for k in ('base', 'fixed')}
cat = {k: Table.read(f'{k}/{STEM}_satstar_catalog.fits') for k in ('base', 'fixed')}

b, f = cat['base'], cat['fixed']
d, i = cKDTree(np.c_[b['xcentroid'], b['ycentroid']]).query(np.c_[f['xcentroid'], f['ycentroid']])
m = d < 0.5
fb, ff = b[i[m]], f[m]
ratio = np.asarray(ff['flux_fit'] / fb['flux_fit'])
dist = np.hypot(ff['xcentroid'] - 81, ff['ycentroid'] - 81)
k = np.argmin(np.hypot(ff['xcentroid'] - XS, ff['ycentroid'] - YS))
sb, sf = fb[k], ff[k]
xc, yc = float(sf['xcentroid']), float(sf['ycentroid'])
ix, iy = int(round(xc)), int(round(yc))
sl = (slice(iy - H, iy + H + 1), slice(ix - H, ix + H + 1))
ext = [ix - H - 0.5, ix + H + 0.5, iy - H - 0.5, iy + H + 0.5]

fig, ax = plt.subplots(2, 3, figsize=(17, 10.5))
dat = sci[sl]
bg = np.nanmedian(dat)
an = AsinhNorm(linear_width=np.nanstd(res['fixed'][sl]) * 3, vmin=bg - 20, vmax=np.nanpercentile(dat, 99.5))
a = ax[0, 0].imshow(dat, origin='lower', cmap='gray_r', norm=an, extent=ext)
ax[0, 0].set_title(f'data (crf SCI); white core = saturated (NaN)\nstar at detector ({xc:.1f}, {yc:.1f}), '
                   f'{int(sf["sat_area"])} saturated px', fontsize=10)
fig.colorbar(a, ax=ax[0, 0], shrink=0.8, label='MJy/sr')

core = ~np.isfinite(dat)
yy, xx = np.mgrid[sl]
rr = np.hypot(xx - xc, yy - yc)
rb = np.where(core, np.nan, res['base'][sl])
rf = np.where(core, np.nan, res['fixed'][sl])
lim = np.nanpercentile(np.abs(rb[rr > 20]), 99)
for j, (r, s, lab) in enumerate([(rb, sb, 'before (main)'), (rf, sf, 'after (#1060)')]):
    c = ax[0, j + 1].imshow(r, origin='lower', cmap='RdBu_r', norm=TwoSlopeNorm(0, -lim, lim), extent=ext)
    ax[0, j + 1].set_title(f'{lab}: satstar residual (sat. core masked)\n'
                           f'flux_fit={s["flux_fit"]:.4g}  qfit={s["qfit"]:.3f}  red.χ²={s["reduced_chi2"]:.0f}', fontsize=10)
fig.colorbar(c, ax=ax[0, 1:], shrink=0.8, label='MJy/sr')

dm = mod['fixed'][sl] - mod['base'][sl]
dl = np.nanpercentile(np.abs(dm), 99.5)
c = ax[1, 0].imshow(dm, origin='lower', cmap='PuOr_r', norm=TwoSlopeNorm(0, -dl, dl), extent=ext)
ax[1, 0].set_title('model after − model before')
fig.colorbar(c, ax=ax[1, 0], shrink=0.8, label='MJy/sr')
for a_ in list(ax[0]) + [ax[1, 0]]:
    a_.set_xlabel('detector x [px]')
ax[0, 0].set_ylabel('detector y [px]')
ax[1, 0].set_ylabel('detector y [px]')

edges = np.arange(0, H + 1, 3)
cen = 0.5 * (edges[1:] + edges[:-1])
for r, lab, col in [(rb, 'before', 'C3'), (rf, 'after', 'C0')]:
    prof = [np.nanmedian(np.abs(r[(rr >= lo) & (rr < hi)])) for lo, hi in zip(edges[:-1], edges[1:])]
    ax[1, 1].semilogy(cen, prof, 'o-', color=col, label=lab)
ax[1, 1].set_xlabel('radius from star [px]')
ax[1, 1].set_ylabel('median |resid| [MJy/sr]')
ax[1, 1].set_title('residual amplitude vs radius, unsaturated pixels only')
ax[1, 1].legend()

ax[1, 2].scatter(dist, ratio, s=8, c='k', alpha=0.5, label=f'{m.sum()} matched satstars')
ax[1, 2].plot(dist[k], ratio[k], 'r*', ms=16, mec='k', label='star shown')
nb = np.linspace(0, dist.max(), 8)
med = [np.median(ratio[(dist >= lo) & (dist < hi)]) for lo, hi in zip(nb[:-1], nb[1:])]
ax[1, 2].step(nb[:-1], med, where='post', color='C1', lw=2, label='binned median')
ax[1, 2].axhline(1, color='gray', lw=0.8)
ax[1, 2].set_ylim(0.8, 1.2)
ax[1, 2].set_xlabel('distance from detector (81, 81) [px] (node the old fit used)')
ax[1, 2].set_ylabel('flux_fit after / before')
ax[1, 2].set_title('whole frame: flux change vs distance from (81, 81)')
ax[1, 2].legend(fontsize=8, loc='lower left')

fig.suptitle(f'#1055 empirical example: {STEM} (F480M NRCB5), same inputs, '
             f'main vs #1060 (c069dc60); flux after/before = {ratio[k]:.3f}', fontsize=11)
fig.savefig('fig_real_star.png', dpi=110, bbox_inches='tight')
print('star', xc, yc, 'ratio', ratio[k], 'qfit', sb['qfit'], sf['qfit'], 'chi2', sb['reduced_chi2'], sf['reduced_chi2'])
for lo, hi, v in zip(nb[:-1], nb[1:], med):
    print(f'{lo:.0f}-{hi:.0f}: {v:.3f}')
