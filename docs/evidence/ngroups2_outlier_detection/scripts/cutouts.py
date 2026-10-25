"""Cutouts of cloudc F212N m7 catalogued stars left with resid S/N>10 at mid flux.
Columns: data | m6 resid | m7 resid, m7 vetted sources marked (orange x)."""
import json
import os
import sys

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

S = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(S, '..', '..', 'phaseloss'))
from phase_loss import _image_hdu  # noqa: E402

field, filt = sys.argv[1:3]
fmin, fmax = float(sys.argv[3]), float(sys.argv[4])
snap = json.load(open(os.path.join(S, '..', '..', 'prod_before', field, filt.lower(), 'snapshot.json')))
P = {k: v['path'] for k, v in snap['files'].items()}
smp = Table.read(os.path.join(S, f'{field}_{filt.lower()}_ctrl_map_m7_sample.fits'))
sel = np.flatnonzero((smp['snr'] > 10) & (smp['flux'] > fmin) & (smp['flux'] < fmax))
rng = np.random.default_rng(2)
sel = rng.choice(sel, size=min(8, len(sel)), replace=False)
H = 15
hd = {k: fits.open(P[k], memmap=True) for k in ('data', 'resbgsub_m6_resid', 'resbgsub_m7_resid')}
im = {k: _image_hdu(v) for k, v in hd.items()}
w = WCS(im['resbgsub_m7_resid'].header)
t = Table.read(P['resbgsub_m7_vetted'])
sc = SkyCoord(t['skycoord'])
vx, vy = w.world_to_pixel_values(sc.ra.deg, sc.dec.deg)
wd = WCS(im['data'].header)
fig, axs = plt.subplots(len(sel), 3, figsize=(7.5, 2.5 * len(sel)))
for r, i in enumerate(sel):
    x, y = smp['x'][i], smp['y'][i]
    ra, dec = w.pixel_to_world_values(x, y)
    for c, k in enumerate(('data', 'resbgsub_m6_resid', 'resbgsub_m7_resid')):
        ww = wd if k == 'data' else WCS(im[k].header)
        xx, yy = ww.world_to_pixel_values(ra, dec)
        xi, yi = int(round(float(xx))), int(round(float(yy)))
        st = im[k].data[yi - H:yi + H + 1, xi - H:xi + H + 1]
        if c == 0:
            med = np.nanmedian(st)
            wid = max(np.nanmax(st[H - 2:H + 3, H - 2:H + 3]) - med, 1e-6)
        m = np.nanmedian(st)
        ax = axs[r, c]
        ax.imshow(st, origin='lower', cmap='gray', vmin=m - 0.3 * wid, vmax=m + 0.6 * wid, interpolation='nearest')
        ax.plot(H + (xx - xi), H + (yy - yi), 'o', mfc='none', mec='r', ms=12)
        if c == 2:
            near = (np.abs(vx - x) <= H) & (np.abs(vy - y) <= H)
            ax.plot(vx[near] - xi + H, vy[near] - yi + H, 'x', color='orange', ms=4)
        ax.set_xticks([]); ax.set_yticks([])
        if r == 0:
            ax.set_title(k.replace('resbgsub_', ''), fontsize=8)
    axs[r, 0].set_ylabel(f"flux {smp['flux'][i]:.0f}\nS/N {smp['snr'][i]:.0f}", fontsize=7)
fig.suptitle(f'{field} {filt}: m7 catalogued stars, flux {fmin:g}-{fmax:g}, m7 resid S/N>10', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(S, f'{field}_{filt.lower()}_bad_{fmin:g}_{fmax:g}.png'), dpi=100)
print('wrote', len(sel))
