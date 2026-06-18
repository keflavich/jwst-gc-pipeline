#!/usr/bin/env python
"""Whole-frame deblend validation across filters / detectors / observations."""
import os, sys, glob, traceback
os.environ.setdefault('STPSF_PATH', '/orange/adamginsburg/jwst/stpsf-data/')
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from scipy.ndimage import find_objects, sum_labels
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-satdeblend')
sys.path.insert(0, os.path.dirname(__file__))
from jwst_gc_pipeline.reduction.saturated_star_finding import find_saturated_stars
from jwst_gc_pipeline.reduction.filtering import get_fwhm
from deblend_zeroframe import deblend_blob_zeroframe, robust_zf_ceiling

GC = '/orange/adamginsburg/jwst/gc2211'
OUTDIR = os.path.join(os.path.dirname(__file__), 'out_batch')
os.makedirs(OUTDIR, exist_ok=True)

# (filter, cal-glob) spanning SW/LW, A/B modules, multiple obs
FRAMES = [
    ('F150W', f'{GC}/F150W/jw02211028*_nrca1_cal.fits'),
    ('F150W', f'{GC}/F150W/jw02211028*_nrcb3_cal.fits'),
    ('F200W', f'{GC}/F200W/jw02211023001_02201_00001_nrca1_cal.fits'),
    ('F200W', f'{GC}/F200W/jw02211046*_nrca3_cal.fits'),
    ('F200W', f'{GC}/F200W/jw02211049*_nrcb2_cal.fits'),
    ('F277W', f'{GC}/F277W/jw02211023*_nrcalong_cal.fits'),
    ('F277W', f'{GC}/F277W/jw02211028*_nrcblong_cal.fits'),
]

def dedupe_inframe(sc, ww, shape, link=2.0):
    px, py = ww.world_to_pixel(sc)
    m = np.isfinite(px) & np.isfinite(py) & (px > -5) & (px < shape[1]+5) & (py > -5) & (py < shape[0]+5)
    pts = np.column_stack([px[m], py[m]])
    if len(pts) == 0:
        return np.empty((0, 2))
    tree = cKDTree(pts); parent = list(range(len(pts)))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for a, b in tree.query_pairs(link):
        parent[find(a)] = find(b)
    gg = {}
    for i in range(len(pts)):
        gg.setdefault(find(i), []).append(i)
    return np.array([pts[idx].mean(axis=0) for idx in gg.values()])

dao_cache = {}
def get_dao(filt):
    if filt not in dao_cache:
        t = Table.read(f'{GC}/catalogs/{filt.lower()}_merged_indivexp_merged_dao_basic.fits')
        dao_cache[filt] = (SkyCoord(t['skycoord']), np.asarray(t['is_saturated'], dtype=bool))
    return dao_cache[filt]

for filt, cglob in FRAMES:
    cands = sorted(glob.glob(cglob))
    if not cands:
        print(f'\n### {filt} {cglob}: NO MATCH', flush=True); continue
    cal = cands[0]
    stem = os.path.basename(cal).replace('_cal.fits', '')
    ramp = f'{GC}/{filt}/pipeline/{stem}_ramp.fits'
    print(f'\n### {filt}  {stem}', flush=True)
    try:
        fd = fits.open(cal)
        data = fd['SCI'].data
        ww = WCS(fd['SCI'].header)
        fwhm, fwhm_pix = get_fwhm(fd[0].header, instrument_replacement='NIRCam')
        rh = fits.open(ramp)
        ngroup = rh['SCI'].data.shape[1] if 'SCI' in rh else '?'
        if 'ZEROFRAME' not in rh:
            print('  NO ZEROFRAME extension -- skip', flush=True); continue
        zeroframe = rh['ZEROFRAME'].data[0].astype(float)
        sat_ceiling = robust_zf_ceiling(zeroframe)
        zf_recover = float(np.mean((zeroframe > 0) & (zeroframe < sat_ceiling)))

        saturated, sources, coms = find_saturated_stars(fd)
        nsrc = int(sources.max())
        slices = find_objects(sources)
        sizes = sum_labels(saturated, sources, np.arange(nsrc) + 1)

        sc, issat = get_dao(filt)
        daophot_xy = dedupe_inframe(sc[issat], ww, data.shape)

        hist = {}; multi = []
        for i in range(nsrc):
            sl = slices[i]
            if sl is None or int(sizes[i]) < 4:
                continue
            centers, info = deblend_blob_zeroframe(zeroframe, data, sources, i+1, sl,
                                                   fwhm_pix, daophot_xy=daophot_xy,
                                                   sat_ceiling=sat_ceiling)
            n = len(centers)
            hist[n] = hist.get(n, 0) + 1
            if n >= 2:
                multi.append((i+1, n, sl, centers))
        nbig = sum(hist.values())
        nmulti = sum(v for k, v in hist.items() if k >= 2)
        print(f'  ngroup={ngroup} fwhm_pix={fwhm_pix:.2f} ZFceiling={sat_ceiling:.0f} '
              f'ZFrecover_frac={zf_recover:.3f}', flush=True)
        print(f'  {nsrc} comps, {nbig} sizeable; hist={dict(sorted(hist.items()))}; '
              f'{nmulti} multi ({100*nmulti/max(nbig,1):.0f}%)', flush=True)

        # save the 2 most-multi example panels
        multi.sort(key=lambda m: -m[1])
        for k, (lab, n, sl, centers) in enumerate(multi[:2]):
            pad = 22
            y0 = max(0, sl[0].start-pad); y1 = min(data.shape[0], sl[0].stop+pad)
            x0 = max(0, sl[1].start-pad); x1 = min(data.shape[1], sl[1].stop+pad)
            fig, axs = plt.subplots(1, 2, figsize=(7, 3.4))
            for a, (t, img) in zip(axs, [('ZEROFRAME', zeroframe), ('cal slope', data)]):
                sub = img[y0:y1, x0:x1]; fin = sub[np.isfinite(sub)]
                vmax = np.nanpercentile(fin, 99.5) if fin.size else 1
                vmin = np.nanpercentile(fin, 2) if fin.size else 0
                a.imshow(sub, origin='lower', vmin=vmin, vmax=vmax, cmap='gray')
                for (cy, cx) in centers:
                    a.plot(cx-x0, cy-y0, '+', c='lime', ms=12, mew=1.6)
                oldcy = 0.5*(sl[0].start+sl[0].stop-1); oldcx = 0.5*(sl[1].start+sl[1].stop-1)
                a.plot(oldcx-x0, oldcy-y0, 'x', c='orange', ms=10, mew=1.6)
                a.set_title(t, fontsize=8); a.set_xticks([]); a.set_yticks([])
            fig.suptitle(f'{filt} {stem} L{lab} -> {n} stars', fontsize=8)
            fig.tight_layout()
            fig.savefig(os.path.join(OUTDIR, f'{filt}_{stem}_L{lab}.png'), dpi=110)
            plt.close(fig)
    except Exception as ex:
        print(f'  ERROR: {type(ex).__name__}: {ex}', flush=True)
        traceback.print_exc()

print('\nDONE', flush=True)
