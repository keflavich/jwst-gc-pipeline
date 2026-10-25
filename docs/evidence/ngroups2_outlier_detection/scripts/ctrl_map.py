"""Where are cloudc F212N m7 catalogued stars left unsubtracted?

Samples final-vetted (m7, and m6 for reference) stars, measures the PSF-matched
residual S/N at each in its own phase's residual mosaic, and maps them.
usage: python ctrl_map.py <field> <filt> <out-prefix>
"""
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
from phase_loss import FWHM_PIX, Mosaic, stamp_metrics, _image_hdu  # noqa: E402

field, filt, outp = sys.argv[1:4]
snap = json.load(open(os.path.join(S, '..', '..', 'prod_before', field, filt.lower(), 'snapshot.json')))
P = {k: v['path'] for k, v in snap['files'].items()}
fwhm = FWHM_PIX[filt]
rng = np.random.default_rng(0)
out = {}
fig, axs = plt.subplots(1, 3, figsize=(16, 5.5))
for j, ph in enumerate(('m6', 'm7')):
    t = Table.read(P[f'resbgsub_{ph}_vetted'])
    with fits.open(P[f'resbgsub_{ph}_resid'], memmap=True) as h:
        hdr = _image_hdu(h).header
        w = WCS(hdr)
        shape = _image_hdu(h).data.shape
    sc = SkyCoord(t['skycoord'])
    x, y = w.world_to_pixel_values(sc.ra.deg, sc.dec.deg)
    pick = rng.choice(len(t), size=min(len(t), 20000), replace=False)
    mos = Mosaic(P[f'resbgsub_{ph}_resid'])
    m = stamp_metrics(mos, x[pick], y[pick], fwhm)
    mos.close()
    snr = m['snr']
    fluxcol = next((c for c in ('flux_fit', 'flux', 'flux_jy') if c in t.colnames), None)
    fl = np.asarray(t[fluxcol])[pick] if fluxcol else np.full(len(pick), np.nan)
    out[ph] = dict(n=len(t), shape=list(shape), crval=list(map(float, w.wcs.crval)),
                   p50=float(np.nanmedian(snr)), p90=float(np.nanpercentile(snr, 90)),
                   p99=float(np.nanpercentile(snr, 99)), frac_gt10=float(np.nanmean(snr > 10)),
                   fluxcol=fluxcol)
    hi = snr > 10
    ax = axs[j]
    ax.scatter(x[pick][~hi], y[pick][~hi], s=1, c='0.7', rasterized=True)
    sca = ax.scatter(x[pick][hi], y[pick][hi], s=3, c=np.log10(snr[hi]), cmap='inferno', vmin=1, vmax=3)
    ax.set_title(f'{field} {filt} {ph}: {hi.sum()}/{len(pick)} catalogued stars with resid S/N>10')
    ax.set_aspect('equal')
    if ph == 'm7':
        plt.colorbar(sca, ax=ax, label='log10 resid S/N')
        good = np.isfinite(fl) & (fl > 0)
        bins = np.logspace(np.log10(np.nanpercentile(fl[good], 1)), np.log10(np.nanpercentile(fl[good], 99.9)), 25)
        idx = np.digitize(fl[good], bins)
        fr = [np.mean(snr[good][idx == k] > 10) if np.sum(idx == k) > 20 else np.nan for k in range(1, len(bins))]
        axs[2].semilogx(np.sqrt(bins[1:] * bins[:-1]), fr, 'o-')
        axs[2].set_xlabel(f'catalog {fluxcol}')
        axs[2].set_ylabel('fraction with m7 resid S/N>10')
        tt = Table(dict(x=x[pick], y=y[pick], snr=snr, flux=fl))
        tt.write(outp + '_m7_sample.fits', overwrite=True)
fig.tight_layout()
fig.savefig(outp + '.png', dpi=100)
json.dump(out, open(outp + '.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
