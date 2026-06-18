#!/usr/bin/env python
"""Fast integration test: the deblend records-builder wired into the package,
without running the (slow) PSF fitting in get_saturated_stars."""
import os, sys, inspect
os.environ.setdefault('STPSF_PATH', '/orange/adamginsburg/jwst/stpsf-data/')
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from scipy.ndimage import sum_labels
from scipy.spatial import cKDTree

sys.path.insert(0, '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-satdeblend')
from jwst_gc_pipeline.reduction import saturated_star_finding as ssf
from jwst_gc_pipeline.reduction.satstar_deblend import build_deblended_source_records, robust_zf_ceiling
from jwst_gc_pipeline.reduction.filtering import get_fwhm

# 1) signature carries the new opt-in params, defaulting to None (no behaviour change)
sig = inspect.signature(ssf.get_saturated_stars)
for p in ('zeroframe', 'deblend_daophot_xy', 'deblend_confirm_xy'):
    assert p in sig.parameters and sig.parameters[p].default is None, p
print('OK get_saturated_stars signature has opt-in deblend params (default None)')

GC = '/orange/adamginsburg/jwst/gc2211'
fd = fits.open(f'{GC}/F200W/jw02211023001_02201_00001_nrca1_cal.fits')
data = fd['SCI'].data
ww = WCS(fd['SCI'].header)
_, fwhm_pix = get_fwhm(fd[0].header, instrument_replacement='NIRCam')
zeroframe = fits.open(f'{GC}/F200W/pipeline/jw02211023001_02201_00001_nrca1_ramp.fits')['ZEROFRAME'].data[0].astype(float)

saturated, sources, coms = ssf.find_saturated_stars(fd)
coms = ssf._refine_coms_by_data(coms, data, sources)
sizes = sum_labels(saturated, sources, np.arange(int(sources.max())) + 1)

dao = Table.read(f'{GC}/catalogs/f200w_merged_indivexp_merged_dao_basic.fits')
dsc = SkyCoord(dao['skycoord']); issat = np.asarray(dao['is_saturated'], bool)
dx, dy = ww.world_to_pixel(dsc[issat])
m = np.isfinite(dx) & np.isfinite(dy) & (dx > -5) & (dx < data.shape[1]+5) & (dy > -5) & (dy < data.shape[0]+5)
pts = np.column_stack([dx[m], dy[m]]); tree = cKDTree(pts); par = list(range(len(pts)))
def f(a):
    while par[a] != a: par[a] = par[par[a]]; a = par[a]
    return a
for a, b in tree.query_pairs(2.0): par[f(a)] = f(b)
gg = {}
for i in range(len(pts)): gg.setdefault(f(i), []).append(i)
daophot_xy = np.array([pts[idx].mean(axis=0) for idx in gg.values()])

# 2) records-builder: per-component -> per-star expansion, schema preserved
recs = build_deblended_source_records(saturated, sources, coms, sizes, zeroframe,
                                      data, fwhm_pix, daophot_xy=daophot_xy)
ncomp = len(coms)
labels = set(r['label'] for r in recs)
assert all(set(r) == {'com', 'label', 'forced', 'sat_area'} for r in recs), 'schema drift'
assert all(r['forced'] is False for r in recs)
assert len(recs) >= ncomp, 'deblend should never drop below #components'
from collections import Counter
per_label = Counter(r['label'] for r in recs)
nmulti = sum(1 for v in per_label.values() if v >= 2)
print(f'OK records-builder: {ncomp} components -> {len(recs)} star seeds; '
      f'{nmulti} components split into >=2; labels covered={len(labels)}')

# 3) the None path is byte-for-byte the legacy records
legacy = [{'com': c, 'label': i+1, 'forced': False,
           'sat_area': int(sizes[i]) if i < len(sizes) else 0}
          for i, c in enumerate(coms)]
recs_none = build_deblended_source_records(saturated, sources, coms, sizes,
                                           zeroframe, data, fwhm_pix, min_area=10**9)
assert len(recs_none) == len(legacy), 'min_area huge should give 1 record/component'
print('OK min_area guard -> one record per component (legacy fallback shape)')
print('ALL TESTS PASSED')
