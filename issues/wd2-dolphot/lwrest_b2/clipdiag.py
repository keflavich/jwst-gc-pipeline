"""Why does sigma_clip mask all frames for some base sources? Re-run Phase 1 stacking for chosen base indices.
usage: python clipdiag.py BAND base_idx... ; also counts all-masked sources band-wide and their per-frame properties."""
import sys, glob, re, warnings
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
import astropy.units as u
warnings.filterwarnings('ignore')
T = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_main2kfpk'
b = sys.argv[1]; want = [int(x) for x in sys.argv[2:]]
fns = sorted(glob.glob(f'{T}/F{b}/f{b.lower()}_*_visit*_vgroup*_exp*_resbgsub_m7_daophot_basic.fits'))
tabs = []
for fn in fns:
    t = Table.read(fn); c = t['skycoord_centroid']; bad = ~np.isfinite(c.ra.deg) | ~np.isfinite(c.dec.deg); tabs.append(t[~bad])
base = None
for t in tabs:
    c = t['skycoord_centroid']
    if base is None: base = c
    else:
        _, sep, _ = c.match_to_catalog_sky(base, nthneighbor=1); base = SkyCoord([base, c[sep > 0.1*u.arcsec]])
n = len(base); nt = len(tabs)
ra = np.full((n, nt), np.nan); dec = ra.copy(); fl = np.full((n, nt), np.nan, dtype='float32'); fo = np.zeros((n, nt), bool); qf = fl.copy()
for ii, t in enumerate(tabs):
    c = t['skycoord_centroid']
    mi, sep, _ = c.match_to_catalog_sky(base); ri, _, _ = base.match_to_catalog_sky(c)
    keep = (sep < 0.1*u.arcsec) & (ri[mi] == np.arange(len(mi)))
    m = mi[keep]
    ra[m, ii] = c.ra.deg[keep]; dec[m, ii] = c.dec.deg[keep]; fl[m, ii] = t['flux_fit'][keep]; qf[m, ii] = t['qfit'][keep]
    fo[m, ii] = np.asarray(t['forced_refit'], bool)[keep]
cf = sigma_clip(fl, stdfunc='mad_std', axis=1); cr = sigma_clip(ra, stdfunc='mad_std', axis=1); cd = sigma_clip(dec, stdfunc='mad_std', axis=1)
keepm = ~(cf.mask | cr.mask | cd.mask)
ng = keepm.sum(1); nm = np.isfinite(fl).sum(1)
allm = (ng == 0) & (nm > 0)
print('base', n, 'all-masked (nmatch>0, nmatch_good==0):', allm.sum(), ' by nmatch', np.bincount(nm[allm]))
print('frac with nmatch=4 all masked', allm[nm == 4].sum(), 'of', (nm == 4).sum(), '; nmatch>=3', allm[nm >= 3].sum(), 'of', (nm >= 3).sum())
print('masked by: flux-only-all', (cf.mask.all(1) & allm).sum(), 'ra-all', (cr.mask.all(1) & allm).sum(), 'dec-all', (cd.mask.all(1) & allm).sum())
for w in want:
    print('base', w, 'nmatch', nm[w])
    for k in range(nt):
        if np.isfinite(fl[w, k]): print(f'  fr{k} ra={ra[w,k]:.9f} dec={dec[w,k]:.9f} flux={fl[w,k]:.2f} forced={fo[w,k]} qfit={qf[w,k]:.3f} masked(f,r,d)={cf.mask[w,k]},{cr.mask[w,k]},{cd.mask[w,k]}')
np.save('/blue/adamginsburg/adamginsburg/tmp/claude-3663/allm_%s.npy' % b, np.where(allm)[0])
