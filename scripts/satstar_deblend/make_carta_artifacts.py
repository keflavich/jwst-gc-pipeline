#!/usr/bin/env python
"""Produce CARTA-visualisable artifacts for the gc2211 satstar deblend dev frame.

Writes, next to the dev exposure's pipeline products:
  * <exp>_zeroframe.fits        ZEROFRAME (frame zero) carrying the cal SCI WCS,
                                so it overlays the crf/model/residual.
  * <exp>_deblend_catalog.fits  NEW deblended star centres (ra/dec + pixel).
  * <exp>_old_satcentroid.fits  OLD one-centroid-per-blob seeds (ra/dec + pixel).
These plus the existing crf (data), _satstar_model, _satstar_residual and
_satstar_catalog are wired into the CARTA snippet.
"""
import os, sys, glob
os.environ.setdefault('STPSF_PATH', '/orange/adamginsburg/jwst/stpsf-data/')
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from scipy.ndimage import find_objects, sum_labels
from scipy.spatial import cKDTree

sys.path.insert(0, '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-satdeblend')
sys.path.insert(0, os.path.dirname(__file__))
from jwst_gc_pipeline.reduction.saturated_star_finding import find_saturated_stars
from jwst_gc_pipeline.reduction.filtering import get_fwhm
from deblend_zeroframe import deblend_blob_zeroframe, robust_zf_ceiling

GC = '/orange/adamginsburg/jwst/gc2211'
EXP = 'jw02211023001_02201_00001_nrca1'
CAL = f'{GC}/F200W/{EXP}_cal.fits'
RAMP = f'{GC}/F200W/pipeline/{EXP}_ramp.fits'
PIPE = f'{GC}/F200W/pipeline'

fd = fits.open(CAL)
data = fd['SCI'].data
sci_hdr = fd['SCI'].header
ww = WCS(sci_hdr)
fwhm, fwhm_pix = get_fwhm(fd[0].header, instrument_replacement='NIRCam')
zeroframe = fits.open(RAMP)['ZEROFRAME'].data[0].astype(float)

# ZEROFRAME as a WCS-carrying image
zf_hdu = fits.PrimaryHDU(data=zeroframe.astype('float32'), header=sci_hdr)
zf_out = f'{PIPE}/{EXP}_zeroframe.fits'
zf_hdu.writeto(zf_out, overwrite=True)
print(f'wrote {zf_out}', flush=True)

saturated, sources, coms = find_saturated_stars(fd)
nsrc = int(sources.max())
slices = find_objects(sources)
sizes = sum_labels(saturated, sources, np.arange(nsrc) + 1)

# deduped daophot is_saturated (snap) positions
dao = Table.read(f'{GC}/catalogs/f200w_merged_indivexp_merged_dao_basic.fits')
dsc = SkyCoord(dao['skycoord'])
issat = np.asarray(dao['is_saturated'], dtype=bool)
dx, dy = ww.world_to_pixel(dsc[issat])
infr = np.isfinite(dx) & np.isfinite(dy) & (dx > -5) & (dx < data.shape[1]+5) & (dy > -5) & (dy < data.shape[0]+5)
pts = np.column_stack([dx[infr], dy[infr]])
tree = cKDTree(pts); parent = list(range(len(pts)))
def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for a, b in tree.query_pairs(2.0):
    parent[find(a)] = find(b)
gg = {}
for i in range(len(pts)):
    gg.setdefault(find(i), []).append(i)
daophot_xy = np.array([pts[idx].mean(axis=0) for idx in gg.values()])

sat_ceiling = robust_zf_ceiling(zeroframe)

new_y, new_x, new_lab, new_n = [], [], [], []
old_y, old_x, old_lab = [], [], []
for i in range(nsrc):
    sl = slices[i]
    if sl is None or int(sizes[i]) < 4:
        continue
    centers, info = deblend_blob_zeroframe(zeroframe, data, sources, i+1, sl,
                                           fwhm_pix, daophot_xy=daophot_xy,
                                           sat_ceiling=sat_ceiling)
    oldcy = 0.5*(sl[0].start+sl[0].stop-1); oldcx = 0.5*(sl[1].start+sl[1].stop-1)
    old_y.append(oldcy); old_x.append(oldcx); old_lab.append(i+1)
    for (cy, cx) in centers:
        new_y.append(cy); new_x.append(cx); new_lab.append(i+1); new_n.append(len(centers))

def write_cat(xs, ys, labs, extra, path):
    sk = ww.pixel_to_world(np.array(xs), np.array(ys))
    t = Table()
    t['x'] = np.array(xs, dtype='float64')
    t['y'] = np.array(ys, dtype='float64')
    t['ra'] = sk.ra.deg
    t['dec'] = sk.dec.deg
    t['blob_label'] = np.array(labs, dtype='int32')
    for k, v in extra.items():
        t[k] = v
    t.write(path, overwrite=True)
    print(f'wrote {path}  ({len(t)} rows)', flush=True)

write_cat(new_x, new_y, new_lab, {'n_stars_in_blob': np.array(new_n, dtype='int32')},
          f'{PIPE}/{EXP}_deblend_catalog.fits')
write_cat(old_x, old_y, old_lab, {}, f'{PIPE}/{EXP}_old_satcentroid.fits')

n_multi = sum(1 for n in new_n if n >= 2)
print(f'blobs: {len(old_lab)} ; new centres: {len(new_lab)} ; '
      f'multi-star centres: {n_multi}', flush=True)
print('DONE', flush=True)
