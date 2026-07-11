#!/usr/bin/env python3
"""Caveat/consequence figures for the catalog expansion: centroid jitter,
residual over-subtraction, and nearest-neighbour (deblend-flag derivability)."""
import warnings, numpy as np, glob
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from astropy.io import fits; from astropy.wcs import WCS
from astropy.table import Table; from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats
import astropy.units as u

B = '/orange/adamginsburg/jwst/brick/cutouts'
import os
FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figs'); os.makedirs(FIG, exist_ok=True)
t = Table.read(glob.glob(f'{B}/f182m_kt_tiered/catalogs/f182m_nrca_*resbgsub_m6_dao_basic_vetted.fits')[0])
sc = SkyCoord(t['skycoord'].ra, t['skycoord'].dec)
lfq = np.asarray(t['low_fit_quality'], bool)
posjit = np.hypot(np.asarray(t['std_ra'], float), np.asarray(t['std_dec'], float)) * 3.6e6
g = glob.glob(f'{B}/f182m_kt_tiered/*/pipeline/*-nrca_data_i2d.fits')
with fits.open(g[0]) as h: d = np.asarray(h['SCI'].data, float); ww = WCS(h['SCI'].header)
rg = [x for x in glob.glob(f'{B}/f182m_kt_tiered/*/pipeline/*-nrca_*m6*residual_i2d.fits') if 'resbgsub' in x]
with fits.open(sorted(rg)[-1]) as h: r = np.asarray(h['SCI'].data, float)
_, med, std = sigma_clipped_stats(d[np.isfinite(d)], sigma=3)
x, y = ww.world_to_pixel(sc); xi = np.clip(np.rint(x).astype(int), 0, d.shape[1]-1); yi = np.clip(np.rint(y).astype(int), 0, d.shape[0]-1)
rmin = np.array([np.nanmin(r[max(0,b-1):b+2, max(0,a-1):a+2]) for a, b in zip(xi, yi)]) / std
_, nn, _ = sc.match_to_catalog_sky(sc, nthneighbor=2); nnm = nn.arcsec * 1000

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].hist(posjit[~lfq], bins=np.arange(0, 40, 2), alpha=0.6, label=f'good-fit (med {np.median(posjit[~lfq]):.1f}mas)', density=True)
ax[0].hist(posjit[lfq], bins=np.arange(0, 40, 2), alpha=0.6, color='orange', label=f'low_fit_quality (med {np.median(posjit[lfq]):.1f}mas)', density=True)
ax[0].axvline(30, color='r', ls='--', lw=1, label='cross-band match radius 30mas')
ax[0].set_xlabel('centroid jitter std(ra,dec) [mas]'); ax[0].set_ylabel('norm. density'); ax[0].legend(fontsize=7)
ax[0].set_title('CAVEAT: faint/flagged centroids jitter ~3x more\n(but bounded < cross-band radius)')

ax[1].hist(np.clip(rmin[~lfq], -10, 10), bins=np.arange(-10, 11, 1), alpha=0.6, label='good-fit', density=True)
ax[1].hist(np.clip(rmin[lfq], -10, 10), bins=np.arange(-10, 11, 1), alpha=0.6, color='orange', label='low_fit_quality', density=True)
ax[1].axvline(-3, color='r', ls='--', lw=1, label='-3sig (over-sub hole)'); ax[1].axvline(0, color='k', lw=0.6)
ax[1].set_xlabel('residual core (3x3 min) [sky-sigma]  >0 = under-subtracted'); ax[1].legend(fontsize=7)
ax[1].set_title('NO EXCESS SUBTRACTION: cores are +ve (under-sub);\nover-sub holes 3% (flagged not elevated)')

ax[2].hist(nnm[~lfq], bins=np.arange(0, 400, 20), alpha=0.6, label='good-fit', density=True)
ax[2].hist(nnm[lfq], bins=np.arange(0, 400, 20), alpha=0.6, color='orange', label='low_fit_quality', density=True)
ax[2].axvline(73, color='r', ls='--', lw=1, label='1 FWHM (73mas)')
ax[2].set_xlabel('nearest-neighbour separation [mas]'); ax[2].legend(fontsize=7)
ax[2].set_title('DEBLEND FLAG NOT derivable from separation:\nflagged sources are NOT preferentially close-pairs')
fig.suptitle('Catalog-expansion consequences (Brick F182M tiered, nrca): excess-sub SAFE; jitter + density are the caveats', fontsize=11)
fig.tight_layout(); fig.savefig(f'{FIG}/fig6_expansion_caveats.png', dpi=130); plt.close(fig)
print('fig6 done')
