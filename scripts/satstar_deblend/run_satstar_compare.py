#!/usr/bin/env python
"""Baseline-vs-deblend satstar comparison on ONE gc2211 frame.

The satstar accept gates (snr/qfit/ssr/sidelobe) live INSIDE get_saturated_stars,
so a bare call already uses production gates; the only thing the bare e2e test
lacked was the BASELINE (no-deblend) count on the SAME frame.  This runs one mode
(baseline | deblend) with production-equivalent kwargs and reports the accepted
row count.  The slurm log additionally carries every "Skipping source ..." line
(snr/qfit/ssr per rejected seed) for a rejection-reason histogram afterwards.

Usage:  run_satstar_compare.py {baseline|deblend} [crf_path]
"""
import os, sys, time
os.environ.setdefault('STPSF_PATH', '/orange/adamginsburg/jwst/stpsf-data/')
sys.path.insert(0, '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wt-satdeblend')
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from scipy.spatial import cKDTree
from jwst_gc_pipeline.reduction import saturated_star_finding as ssf

GC = '/orange/adamginsburg/jwst/gc2211'
mode = sys.argv[1] if len(sys.argv) > 1 else 'deblend'
assert mode in ('baseline', 'deblend')
crf = sys.argv[2] if len(sys.argv) > 2 else (
    f'{GC}/F200W/pipeline/jw02211023001_02201_00001_nrca1_destreak_'
    f'jw02211-o023_20260515t014922_image3_00001_crf.fits')
OUT = os.path.join(os.path.dirname(__file__), 'out_compare')
os.makedirs(OUT, exist_ok=True)

fh = fits.open(crf)
header = fh[0].header
if 'CRPIX1' not in header:
    from astropy import wcs as _wcs
    header.update(_wcs.WCS(fh['SCI'].header).to_header())

kwargs = dict(path_prefix=f'{GC}/psfs/', plot=False,
              use_merged_psf_for_merged=False)

if mode == 'deblend':
    zf = ssf._find_zeroframe_for(crf)
    assert zf is not None, 'no ZEROFRAME found'
    kwargs['zeroframe'] = zf
    # daophot is_saturated snap + full-catalog confirm (production would pass the
    # current per-filter merged catalog; use it here for a realistic run)
    ww = WCS(fh['SCI'].header)
    dao = Table.read(f'{GC}/catalogs/f200w_merged_indivexp_merged_dao_basic.fits')
    dsc = SkyCoord(dao['skycoord']); issat = np.asarray(dao['is_saturated'], bool)

    def dedupe(sc, link=2.0):
        px, py = ww.world_to_pixel(sc)
        m = (np.isfinite(px) & np.isfinite(py) & (px > -5) & (px < 2053) &
             (py > -5) & (py < 2053))
        pts = np.column_stack([px[m], py[m]])
        if not len(pts):
            return np.empty((0, 2))
        tr = cKDTree(pts); par = list(range(len(pts)))
        def f(a):
            while par[a] != a:
                par[a] = par[par[a]]; a = par[a]
            return a
        for a, b in tr.query_pairs(link):
            par[f(a)] = f(b)
        g = {}
        for i in range(len(pts)):
            g.setdefault(f(i), []).append(i)
        return np.array([pts[idx].mean(axis=0) for idx in g.values()])

    kwargs['deblend_daophot_xy'] = dedupe(dsc[issat])
    kwargs['deblend_confirm_xy'] = dedupe(dsc)

t0 = time.time()
tab = ssf.get_saturated_stars(fh, **kwargs)
dt = time.time() - t0
n = 0 if tab is None else len(tab)
print(f'COMPARE_RESULT mode={mode} accepted={n} seconds={dt:.0f}', flush=True)
if tab is not None:
    tab.write(os.path.join(OUT, f'compare_{mode}_satstar_catalog.fits'),
              overwrite=True)
print('COMPARE_DONE', flush=True)
