#!/usr/bin/env python
"""Probe whether early ramp groups carry usable signal inside saturated cores.

cal.fits is the slope fit; saturated pixels are flagged + filled from the last
unsaturated group, so cores look ~flat.  The RAW RAMP (_ramp.fits, a
[nint, ngroup, ny, nx] cube) holds every up-the-ramp read.  Bright pixels
saturate only after a few groups, so the FIRST groups before saturation should
preserve the true PSF peak of EACH star in a merged-core pair -> the deblend
signal.  This dumps, for a blended blob and a single-star blob, the group-by-
group cutout so we can see at which group saturation sets in and whether the
two cores are resolved early.
"""
import os, sys, glob
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GC = '/orange/adamginsburg/jwst/gc2211'
RAMP = f'{GC}/F200W/pipeline/jw02211023001_02201_00001_nrca1_ramp.fits'
CAL = f'{GC}/F200W/jw02211023001_02201_00001_nrca1_cal.fits'
OUTDIR = os.path.join(os.path.dirname(__file__), 'out')
os.makedirs(OUTDIR, exist_ok=True)

rh = fits.open(RAMP)
print('RAMP extensions:', [(e.name, getattr(e, 'shape', None)) for e in rh], flush=True)
cube = rh['SCI'].data        # expect (nint, ngroup, ny, nx)
gdq = rh['GROUPDQ'].data if 'GROUPDQ' in rh else None
print('cube shape', cube.shape, 'gdq', None if gdq is None else gdq.shape, flush=True)

ch = fits.open(CAL)
calsci = ch['SCI'].data
from jwst.datamodels import dqflags
SAT = dqflags.pixel['SATURATED']

# (name, x, y) of blobs identified by the diagnostic
TARGETS = [('L15_double', 1020, 177), ('L138_single', 1703, 1284),
           ('L168', 1040, 1565), ('L145', 630, 1326)]
PAD = 16
nint, ngroup = cube.shape[0], cube.shape[1]
print(f'nint={nint} ngroup={ngroup}', flush=True)

for name, xc, yc in TARGETS:
    y0, y1 = yc - PAD, yc + PAD
    x0, x1 = xc - PAD, xc + PAD
    # show up to first 8 groups of integration 0 + cal slope
    ng_show = min(8, ngroup)
    fig, axs = plt.subplots(1, ng_show + 1, figsize=(2.0 * (ng_show + 1), 2.4))
    for g in range(ng_show):
        sub = cube[0, g, y0:y1, x0:x1].astype(float)
        # show raw DN; scale to this group's own range
        vmax = np.nanpercentile(sub, 99)
        axs[g].imshow(sub, origin='lower', vmin=np.nanmin(sub), vmax=vmax, cmap='gray')
        nsat = 0
        if gdq is not None:
            nsat = int(((gdq[0, g, y0:y1, x0:x1] & SAT) > 0).sum())
        axs[g].set_title(f'grp{g}\nsat={nsat}', fontsize=7)
        axs[g].set_xticks([]); axs[g].set_yticks([])
    cs = calsci[y0:y1, x0:x1]
    vmax = np.nanpercentile(cs[np.isfinite(cs)], 99) if np.isfinite(cs).any() else 1
    axs[-1].imshow(cs, origin='lower', vmin=0, vmax=vmax, cmap='gray')
    axs[-1].set_title('cal slope', fontsize=7)
    axs[-1].set_xticks([]); axs[-1].set_yticks([])
    fig.suptitle(f'{name} px=({xc},{yc})  (raw ramp DN, integration 0)', fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUTDIR, f'ramp_{name}.png')
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f'wrote {out}', flush=True)

    # also print the group at which the central pixel first saturates
    if gdq is not None:
        cgdq = gdq[0, :, yc, xc] & SAT
        firstsat = np.argmax(cgdq > 0) if (cgdq > 0).any() else -1
        print(f'  {name}: central pix saturates at group {firstsat} '
              f'(of {ngroup}); peak grp0 DN={cube[0,0,yc,xc]}', flush=True)
print('DONE', flush=True)
