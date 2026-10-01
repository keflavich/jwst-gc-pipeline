"""Figures for README.md.  Run from the working directory holding tgt_e*.npz, q3_e*.npz
(patternfit.py LOO outputs) and ../data/*.fits:   python make_figures.py <outdir>"""
import sys, os, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import ndimage
from astropy.io import fits
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exposure import Exposure
# nan-heavy medians/ratios on masked pixels; other warnings (GWCS, I/O) stay visible
warnings.filterwarnings('ignore', category=RuntimeWarning)
out = sys.argv[1]
RA, DEC = 266.5090306896523, -28.95658817641266
J1 = np.load('tgt_e1.npz')['J']
LOO = [k for k in range(1, 7) if os.path.exists(f'q3_e{k}.npz')]


def loo(k):
    ex = Exposure(f'tgt_e{k}.npz', J1); z = np.load(f'q3_e{k}.npz')
    chi = (ex.d-z['pred'])/np.sqrt(ex.var+z['varQ'])
    ok = ex.good & np.isfinite(chi) & (z['cov'] >= 3)
    yy, xx = np.mgrid[:ex.n, :ex.n]; r = np.hypot(xx-ex.xt0, yy-ex.yt0)
    return ex, z, chi, ok, r


# ---- fig 1: LOO prediction of dither 1
ex, z, chi, ok, r = loo(1)
s = np.s_[312:712, 312:712]
fig, ax = plt.subplots(1, 4, figsize=(24, 6.4))
im = ax[0].imshow(np.arcsinh(np.nan_to_num(ex.d[s]-25)/10), origin='lower', cmap='gray', vmin=-1, vmax=7); ax[0].set_title('dither 1 data (F480M, asinh)')
ax[1].imshow(np.arcsinh(np.nan_to_num(z['pred'][s]-25)/10), origin='lower', cmap='gray', vmin=-1, vmax=7); ax[1].set_title('prediction from dithers 2-6 (LOO)')
ax[2].imshow(np.where(ok, ex.d-z['pred'], np.nan)[s], origin='lower', cmap='RdBu_r', vmin=-30, vmax=30); ax[2].set_title('data - prediction [MJy/sr], +/-30')
im = ax[3].imshow(np.where(ok, chi, np.nan)[s], origin='lower', cmap='RdBu_r', vmin=-5, vmax=5); ax[3].set_title(r'$\chi$ = residual / $\sigma_{Poisson+read+model}$, +/-5')
for a in ax:
    a.set_xticks([]); a.set_yticks([])
    for rr in (80, 200):
        a.add_patch(plt.Circle((200+ex.xt0-512, 200+ex.yt0-512), rr, fill=False, color='y', lw=0.6, ls='--'))
plt.colorbar(im, ax=ax[3], fraction=0.046)
plt.tight_layout(); plt.savefig(f'{out}/fig1_loo_dither1.png', dpi=70); plt.close()

# ---- fig 2: robust sigma(chi) vs radius
edges = np.array([30, 40, 50, 65, 80, 100, 120, 150, 200, 250, 300, 400, 500, 650, 800])
mid = 0.5*(edges[1:]+edges[:-1])
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for k in LOO:
    ex, z, chi, ok, r = loo(k)
    cl = ok & ((ex.dq & 6) == 0)
    rs = [1.4826*np.median(np.abs(chi[cl & (r >= a) & (r < b)])) for a, b in zip(edges[:-1], edges[1:])]
    f5 = [np.mean(np.abs(chi[cl & (r >= a) & (r < b)]) > 5) for a, b in zip(edges[:-1], edges[1:])]
    ax[0].plot(mid, rs, 'o-', label=f'held-out dither {k}')
    ax[1].semilogy(mid, np.maximum(f5, 1e-6), 'o-', label=f'dither {k}')
ax[0].axhline(1, color='k', lw=1); ax[0].set_xscale('log'); ax[0].set_xlabel('distance from saturated star [px]'); ax[0].set_ylabel(r'robust $\sigma(\chi)$ (1 = Poisson-limited)')
ax[0].set_title('leave-one-dither-out, pixels without SAT/JUMP flags'); ax[0].legend()
ax[1].axhline(5.7e-7, color='k', lw=1); ax[1].set_xscale('log'); ax[1].set_xlabel('distance [px]'); ax[1].set_ylabel(r'fraction $|\chi|>5$ (Gaussian: 5.7e-7)')
plt.tight_layout(); plt.savefig(f'{out}/fig2_chi_vs_radius.png', dpi=80); plt.close()

# ---- fig 3: dither ratio maps (GWCS resampled)
import stdatamodels.jwst.datamodels as dm
M = {i: dm.open(f'../data/jw10678061001_02101_{i:05d}_nrcblong_cal.fits') for i in range(1, 7)}
d1 = M[1].data.astype(float); bad = ~np.isfinite(d1) | ((M[1].dq & 1) > 0)
co = ndimage.spline_filter(np.where(bad, 0, d1), order=3)
fig, ax = plt.subplots(1, 5, figsize=(30, 6.4))
H = 200
for a, j in zip(ax, range(2, 7)):
    xt, yt = M[j].meta.wcs.invert(RA, DEC)
    yy, xx = np.mgrid[int(yt)-H:int(yt)+H, int(xt)-H:int(xt)+H]; yy = yy.clip(0, 2047); xx = xx.clip(0, 2047)
    ra, dec = M[j].meta.wcs(xx.astype(float), yy.astype(float)); x1, y1 = M[1].meta.wcs.invert(ra, dec)
    v = ndimage.map_coordinates(co, [y1, x1], order=3, prefilter=False, mode='constant', cval=np.nan)
    b = ndimage.map_coordinates(bad.astype(float), [y1, x1], order=1, mode='constant', cval=1) > 0.01
    dj = M[j].data[yy, xx].astype(float); okk = ~b & ((M[j].dq[yy, xx] & 1) == 0) & np.isfinite(dj)
    im = a.imshow(np.where(okk, dj/v, np.nan), origin='lower', vmin=0.75, vmax=1.25, cmap='RdBu_r'); a.set_title(f'dither {j} / dither 1 (same sky)'); a.set_xticks([]); a.set_yticks([])
plt.colorbar(im, ax=ax[-1], fraction=0.046)
plt.tight_layout(); plt.savefig(f'{out}/fig3_dither_ratio.png', dpi=60); plt.close()

# ---- fig 4: diagnostics
fig, ax = plt.subplots(1, 3, figsize=(21, 5.5))
lev = []
for i in range(1, 7):
    xt, yt = M[i].meta.wcs.invert(RA, DEC)
    u = fits.getdata(f'../data/jw10678061001_02101_{i:05d}_nrcblong_uncal.fits', 'SCI').astype(float)[0]
    yy, xx = np.mgrid[:2048, :2048]; rr = np.hypot(xx-xt, yy-yt)
    lev.append([np.median((u[1]-u[0])[(rr >= a) & (rr < b)]) for a, b in [(45, 70), (70, 110), (110, 170), (400, 600)]])
lev = np.array(lev)
for c, lab in enumerate(['r 45-70', 'r 70-110', 'r 110-170', 'r 400-600 (scene)']):
    ax[0].plot(range(1, 7), lev[:, c]/lev[0, c], 'o-', label=lab)
ax[0].set_xlabel('dither (exposure order)'); ax[0].set_ylabel('raw _uncal group-difference / dither 1'); ax[0].set_title('F480M: raw-ramp wing flux changes between dithers'); ax[0].legend()
# int1 vs int2 correlation of the LOO residual (dither 1)
ex, z, chi, ok, r = loo(1)
X0, Y0 = ex.X0, ex.Y0; sl = np.s_[Y0:Y0+1024, X0:X0+1024]
cal = fits.open('../data/jw10678061001_02101_00001_nrcblong_cal.fits'); rate = fits.open('../data/jw10678061001_02101_00001_nrcblong_rate.fits')
ri = fits.open('../data/jw10678061001_02101_00001_nrcblong_rateints.fits')
conv = (cal['SCI'].data/rate['SCI'].data)[sl]
i1 = ri['SCI'].data[0][sl]*conv; i2 = ri['SCI'].data[1][sl]*conv
okk = np.isfinite(i1) & np.isfinite(i2) & np.isfinite(z['pred']) & (((ri['DQ'].data[0] | ri['DQ'].data[1])[sl] & 7) == 0) & ((ex.dq & 6) == 0)
cc = [np.corrcoef((i1-z['pred'])[okk & (r >= a) & (r < b)], (i2-z['pred'])[okk & (r >= a) & (r < b)])[0, 1] for a, b in zip(edges[3:-1], edges[4:])]
ax[1].plot(mid[3:], cc, 'o-'); ax[1].set_xscale('log'); ax[1].set_ylim(0, 1)
ax[1].set_xlabel('distance [px]'); ax[1].set_ylabel('corr(int1 residual, int2 residual)'); ax[1].set_title('residual pattern is identical in both integrations')
# chi histogram
for lab, m in [('r > 300 px, >10 px from field stars', None), ('r 80-150 px', (80, 150))]:
    if m is None:
        pk = (z['pred'] == ndimage.maximum_filter(np.nan_to_num(z['pred']), 7)) & (z['pred'] > 60)
        sel = ok & ((ex.dq & 6) == 0) & (r > 300) & (ndimage.distance_transform_edt(~pk) > 10)
    else:
        sel = ok & ((ex.dq & 6) == 0) & (r >= m[0]) & (r < m[1])
    ax[2].hist(np.clip(chi[sel], -8, 8), bins=160, range=(-8, 8), density=True, histtype='step', label=lab)
xg = np.linspace(-8, 8, 400); ax[2].semilogy(xg, np.exp(-xg**2/2)/np.sqrt(2*np.pi), 'k--', label='N(0,1)')
ax[2].set_ylim(1e-5, 1); ax[2].set_xlabel(r'$\chi$'); ax[2].legend(); ax[2].set_title('held-out dither 1')
plt.tight_layout(); plt.savefig(f'{out}/fig4_diagnostics.png', dpi=75); plt.close()
print('figures written to', out)
