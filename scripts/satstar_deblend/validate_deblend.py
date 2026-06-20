#!/usr/bin/env python
"""Validate the ZEROFRAME deblender on gc2211 example blobs + whole-frame stats."""
import os, sys, glob
os.environ.setdefault('STPSF_PATH', '/orange/adamginsburg/jwst/stpsf-data/')
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.ndimage import find_objects, sum_labels
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-satdeblend')
sys.path.insert(0, os.path.dirname(__file__))
from jwst_gc_pipeline.reduction.saturated_star_finding import find_saturated_stars
from jwst_gc_pipeline.reduction.filtering import get_fwhm
from deblend_zeroframe import deblend_blob_zeroframe, robust_zf_ceiling

GC = '/orange/adamginsburg/jwst/gc2211'
CAL = f'{GC}/F200W/jw02211023001_02201_00001_nrca1_cal.fits'
RAMP = f'{GC}/F200W/pipeline/jw02211023001_02201_00001_nrca1_ramp.fits'
RESID = f'{GC}/F200W/pipeline/jw02211023001_02201_00001_nrca1_destreak_o023_crf_iter3_satstar_residual.fits'
OUTDIR = os.path.join(os.path.dirname(__file__), 'out')

fd = fits.open(CAL)
data = fd['SCI'].data
hdr = fd['SCI'].header
ww = WCS(hdr)
fwhm, fwhm_pix = get_fwhm(fd[0].header, instrument_replacement='NIRCam')
print(f'FWHM = {fwhm}, {fwhm_pix:.2f} px', flush=True)

zeroframe = fits.open(RAMP)['ZEROFRAME'].data[0].astype(float)
try:
    resid = fits.open(RESID)['SCI'].data
except Exception:
    resid = None

saturated, sources, coms = find_saturated_stars(fd)
nsrc = int(sources.max())
slices = find_objects(sources)
sizes = sum_labels(saturated, sources, np.arange(nsrc) + 1)

# deduped daophot is_saturated positions in-frame
dao = Table.read(f'{GC}/catalogs/f200w_merged_indivexp_merged_dao_basic.fits')
dsc = SkyCoord(dao['skycoord'])
issat = np.asarray(dao['is_saturated'], dtype=bool)
dx, dy = ww.world_to_pixel(dsc[issat])
infr = np.isfinite(dx) & np.isfinite(dy) & (dx > -5) & (dx < data.shape[1]+5) & (dy > -5) & (dy < data.shape[0]+5)
from scipy.spatial import cKDTree
pts = np.column_stack([dx[infr], dy[infr]])
tree = cKDTree(pts); parent = list(range(len(pts)))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for a, b in tree.query_pairs(2.0):
    parent[find(a)] = find(b)
g = {}
for i in range(len(pts)):
    g.setdefault(find(i), []).append(i)
daophot_xy = np.array([pts[idx].mean(axis=0) for idx in g.values()])
print(f'{len(daophot_xy)} deduped daophot sat positions', flush=True)

# FULL daophot catalog (all sources, not just is_saturated) deduped + in-frame,
# used only to CONFIRM secondary ZF peaks as cataloged stars (reject Airy/spike
# artifacts).  Unsaturated companions live here but not in the is_saturated set.
ax, ay = ww.world_to_pixel(dsc)
ainf = np.isfinite(ax) & np.isfinite(ay) & (ax > -5) & (ax < data.shape[1]+5) & (ay > -5) & (ay < data.shape[0]+5)
apts = np.column_stack([ax[ainf], ay[ainf]])
atree = cKDTree(apts); aparent = list(range(len(apts)))
def afind(a):
    while aparent[a] != a:
        aparent[a] = aparent[aparent[a]]; a = aparent[a]
    return a
for a, b in atree.query_pairs(2.0):
    aparent[afind(a)] = afind(b)
ag = {}
for i in range(len(apts)):
    ag.setdefault(afind(i), []).append(i)
confirm_xy = np.array([apts[idx].mean(axis=0) for idx in ag.values()])
print(f'{len(confirm_xy)} deduped daophot ALL positions (confirm set)', flush=True)

sat_ceiling = robust_zf_ceiling(zeroframe)

# ---- whole-frame star-count distribution ----
nstar_hist = {}
multi = []
for i in range(nsrc):
    sl = slices[i]
    if sl is None or int(sizes[i]) < 4:
        continue
    centers, info = deblend_blob_zeroframe(zeroframe, data, sources, i+1, sl,
                                           fwhm_pix, daophot_xy=daophot_xy,
                                           confirm_xy=confirm_xy, sat_ceiling=sat_ceiling)
    n = len(centers)
    nstar_hist[n] = nstar_hist.get(n, 0) + 1
    if n >= 2:
        multi.append((i+1, n, sl, centers))
print('\nstars-per-blob histogram (sizeable blobs):', dict(sorted(nstar_hist.items())), flush=True)
print(f'{len(multi)} blobs deblended into >=2 stars', flush=True)

# ---- render example blobs ----
EXAMPLES = [('L15', 15), ('L138', 138), ('L31', 31), ('L168', 168),
            ('L145', 145), ('L1', 1), ('L220', 220), ('L98', 98)]
for name, lab in EXAMPLES:
    sl = slices[lab-1]
    if sl is None:
        continue
    centers, info = deblend_blob_zeroframe(zeroframe, data, sources, lab, sl,
                                           fwhm_pix, daophot_xy=daophot_xy,
                                           confirm_xy=confirm_xy,
                                           sat_ceiling=sat_ceiling, verbose=True)
    y0, y1, x0, x1 = info['crop']
    zf = zeroframe[y0:y1, x0:x1]
    cs = data[y0:y1, x0:x1]
    panels = [('ZEROFRAME', zf, 'gray'), ('ZF smoothed', info['zf_sm'], 'magma'),
              ('cal slope', cs, 'gray')]
    if resid is not None:
        panels.append(('iter3 satstar residual', resid[y0:y1, x0:x1], 'gray'))
    fig, axs = plt.subplots(1, len(panels), figsize=(3.0*len(panels), 3.2))
    oldcy = 0.5*(sl[0].start+sl[0].stop-1); oldcx = 0.5*(sl[1].start+sl[1].stop-1)
    for a, (t, img, cm) in zip(axs, panels):
        fin = img[np.isfinite(img)]
        vmax = np.nanpercentile(fin, 99.5) if fin.size else 1
        vmin = np.nanpercentile(fin, 2) if fin.size else 0
        a.imshow(img, origin='lower', vmin=vmin, vmax=vmax, cmap=cm)
        # new centers (green), old single centroid (orange)
        for (cy, cx) in centers:
            a.plot(cx-x0, cy-y0, '+', c='lime', ms=12, mew=1.6)
        a.plot(oldcx-x0, oldcy-y0, 'x', c='orange', ms=10, mew=1.6)
        a.set_title(t, fontsize=8); a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f'{name}: deblend -> {len(centers)} star(s)  (green=new, orange=old single)',
                 fontsize=9)
    fig.tight_layout()
    out = os.path.join(OUTDIR, f'deblend_{name}.png')
    fig.savefig(out, dpi=120); plt.close(fig)
    print(f'  wrote {out}', flush=True)
print('DONE', flush=True)
