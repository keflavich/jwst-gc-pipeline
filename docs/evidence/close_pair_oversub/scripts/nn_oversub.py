"""Over-subtraction of m7 vetted stars as a function of nearest-neighbour separation.

For a random sample of m7 vetted stars, measures the PSF-matched residual S/N
(phase_loss.stamp_metrics) in that product's own m7 residual mosaic, and the
separation to the nearest other vetted star in the same catalogue (in units of
the filter FWHM).  If independent (ungrouped) fits double-count the flux shared
by close pairs, the fraction with S/N < -5 rises at small separation.

usage: python nn_oversub.py <field> <FILT> <before|after> <out-prefix> [nsample]
  before: the products the pre-rerun snapshot points at (only meaningful while
          they still exist; brick's /orange copy holds the 2026-09-03 m7)
  after:  the same paths, remapped by AFTER_REMAP=old=new if set
"""
import json
import os
import sys

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(S, '..', '..', 'phaseloss'))
from phase_loss import FWHM_PIX, Mosaic, stamp_metrics  # noqa: E402

BINS = np.array([0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 5.0, np.inf])  # NN sep / FWHM

field, filt, which, outp = sys.argv[1:5]
nsample = int(sys.argv[5]) if len(sys.argv) > 5 else 60000
snap = json.load(open(os.path.join(S, '..', '..', 'prod_before', field, filt.lower(), 'snapshot.json')))
P = {k: snap['files'][k]['path'] for k in ('resbgsub_m7_vetted', 'resbgsub_m7_resid')}
if which == 'after' and os.environ.get('AFTER_REMAP'):
    old, new = os.environ['AFTER_REMAP'].split('=')
    P = {k: v.replace(old, new, 1) for k, v in P.items()}
fwhm = FWHM_PIX[filt]

t = Table.read(P['resbgsub_m7_vetted'])
mos = Mosaic(P['resbgsub_m7_resid'])
sc = SkyCoord(t['skycoord'])
x, y = mos.wcs.world_to_pixel(sc)
ok = np.isfinite(x) & np.isfinite(y)
xy = np.c_[x, y]
tree = cKDTree(xy[ok])
idx_ok = np.flatnonzero(ok)
rng = np.random.default_rng(0)
pick = rng.choice(idx_ok, size=min(nsample, len(idx_ok)), replace=False)
d, j = tree.query(xy[pick], k=2)
nn = d[:, 1] / fwhm
nbr = idx_ok[j[:, 1]]
flux = np.asarray(t['flux'], float)
ratio = flux[nbr] / flux[pick]
snr = stamp_metrics(mos, x[pick], y[pick], fwhm)['snr']
mos.close()

rows = []
for lo, hi in zip(BINS[:-1], BINS[1:]):
    m = (nn >= lo) & (nn < hi) & np.isfinite(snr)
    n = int(m.sum())
    rows.append(dict(lo=float(lo), hi=float(hi), n=n,
                     frac_lt_m5=float(np.mean(snr[m] < -5)) if n else None,
                     frac_gt5=float(np.mean(snr[m] > 5)) if n else None,
                     med_snr=float(np.median(snr[m])) if n else None))
fin = np.isfinite(snr)
summ = dict(field=field, filt=filt, which=which, vetted=P['resbgsub_m7_vetted'], resid=P['resbgsub_m7_resid'],
            n_vetted=len(t), n_sample=int(len(pick)), fwhm_pix=fwhm,
            frac_lt_m5_all=float(np.mean(snr[fin] < -5)), frac_gt5_all=float(np.mean(snr[fin] > 5)),
            frac_nn_lt_1p5fwhm=float(np.mean(nn < 1.5)),
            group_size_values=sorted(set(np.unique(np.asarray(t['group_size'])).tolist()))[:10]
            if 'group_size' in t.colnames else None,
            bins=rows)
json.dump(summ, open(outp + '.json', 'w'), indent=1)
Table(dict(x=x[pick], y=y[pick], snr=snr, nn_fwhm=nn, flux=flux[pick], nbr_flux_ratio=ratio,
           nbr_index=nbr, index=pick)).write(outp + '_sample.fits', overwrite=True)
print(json.dumps({k: v for k, v in summ.items() if k != 'bins'}, indent=1))
for r in rows:
    print(r)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
c = 0.5 * (BINS[:-1] + np.minimum(BINS[1:], 6))
ax[0].plot(c, [r['frac_lt_m5'] for r in rows], 'o-', label='S/N < -5 (over-subtracted)')
ax[0].plot(c, [r['frac_gt5'] for r in rows], 's--', label='S/N > 5 (under-subtracted)')
ax[0].set_xlabel('nearest catalogued neighbour / FWHM')
ax[0].set_ylabel('fraction of m7 vetted stars')
ax[0].set_title(f'{field} {filt} m7 ({which}), n={len(pick)}')
ax[0].legend()
m = fin & (nn < 1.5)
ax[1].scatter(np.log10(ratio[m]), np.clip(snr[m], -60, 60), s=2, alpha=0.3)
ax[1].set_xlabel('log10 neighbour flux / own flux')
ax[1].set_ylabel('residual S/N (clipped)')
ax[1].set_title('pairs closer than 1.5 FWHM')
ax[1].axhline(-5, color='r', ls=':')
fig.tight_layout()
fig.savefig(outp + '.png', dpi=100)
