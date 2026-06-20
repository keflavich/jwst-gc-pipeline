#!/usr/bin/env python
"""Does the ZEROFRAME resolve merged saturated cores?

These GC exposures have only ngroup=2 and the bright cores saturate already in
group 0.  But the ramp carries a ZEROFRAME: the single first detector read at
~one frame time (much shorter effective exposure), saturating only at far higher
flux.  If the cores are NOT flat there, the ZEROFRAME centroid resolves each
star inside a merged blob -- the cleanest possible deblend signal.
"""
import os, numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from jwst.datamodels import dqflags

GC = '/orange/adamginsburg/jwst/gc2211'
RAMP = f'{GC}/F200W/pipeline/jw02211023001_02201_00001_nrca1_ramp.fits'
CAL = f'{GC}/F200W/jw02211023001_02201_00001_nrca1_cal.fits'
OUTDIR = os.path.join(os.path.dirname(__file__), 'out')

rh = fits.open(RAMP)
zf = rh['ZEROFRAME'].data[0].astype(float)   # (ny,nx)
g0 = rh['SCI'].data[0, 0].astype(float)
g1 = rh['SCI'].data[0, 1].astype(float)
satdq = (rh['PIXELDQ'].data & dqflags.pixel['SATURATED']) > 0
calsci = fits.open(CAL)['SCI'].data
# ZEROFRAME stores 0 where not needed (only filled where group1 saturated)?
print('ZEROFRAME nonzero frac:', np.mean(zf != 0),
      ' min/median/max(nonzero):',
      np.min(zf[zf != 0]) if (zf != 0).any() else None,
      np.median(zf[zf != 0]) if (zf != 0).any() else None,
      np.nanmax(zf), flush=True)

TARGETS = [('L15_double', 1020, 177), ('L138_single', 1703, 1284),
           ('L168', 1040, 1565), ('L145', 630, 1326), ('L31', 522, 421)]
PAD = 16
for name, xc, yc in TARGETS:
    y0, y1, x0, x1 = yc-PAD, yc+PAD, xc-PAD, xc+PAD
    panels = [('ZEROFRAME', zf), ('group0', g0), ('group1', g1), ('cal slope', calsci)]
    fig, axs = plt.subplots(1, len(panels), figsize=(2.4*len(panels), 2.7))
    for a, (lab, img) in zip(axs, panels):
        sub = img[y0:y1, x0:x1]
        finite = sub[np.isfinite(sub)]
        vmax = np.nanpercentile(finite, 99.5) if finite.size else 1
        vmin = np.nanpercentile(finite, 5) if finite.size else 0
        a.imshow(sub, origin='lower', vmin=vmin, vmax=vmax, cmap='gray')
        a.set_title(lab, fontsize=8); a.set_xticks([]); a.set_yticks([])
    zsub = zf[y0:y1, x0:x1]
    zpk = np.nanmax(zsub)
    fig.suptitle(f'{name} px=({xc},{yc})  ZEROFRAME peak DN={zpk:.0f}', fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUTDIR, f'zeroframe_{name}.png')
    fig.savefig(out, dpi=120); plt.close(fig)
    # how many pixels in the core window are saturated in ZEROFRAME? proxy: at the
    # ZEROFRAME hard ceiling.  Report the ZEROFRAME peak vs group0 peak.
    print(f'{name}: ZF peak={zpk:.0f}  g0 peak={np.nanmax(g0[y0:y1,x0:x1]):.0f}  '
          f'wrote {os.path.basename(out)}', flush=True)
print('DONE', flush=True)
