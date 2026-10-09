"""Distribution of nmatch_good vs nmatch under the Phase-1 union clip (replica of clipdiag.py stacking),
plus how many all-masked rows would be rescued by a fallback to all finite frames, and the mag
offset vs dolphot for partially-clipped rows (clipped mean vs unclipped mean)."""
import sys, glob, warnings
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
import astropy.units as u
warnings.filterwarnings('ignore')
T = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_main2kfpk'
b = sys.argv[1]
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
ra = np.full((n, nt), np.nan); dec = ra.copy(); fl = np.full((n, nt), np.nan, dtype='float32'); fo = np.zeros((n, nt), bool)
for ii, t in enumerate(tabs):
    c = t['skycoord_centroid']
    mi, sep, _ = c.match_to_catalog_sky(base); ri, _, _ = base.match_to_catalog_sky(c)
    keep = (sep < 0.1*u.arcsec) & (ri[mi] == np.arange(len(mi)))
    m = mi[keep]
    ra[m, ii] = c.ra.deg[keep]; dec[m, ii] = c.dec.deg[keep]; fl[m, ii] = t['flux_fit'][keep]
    fo[m, ii] = np.asarray(t['forced_refit'], bool)[keep]
cf = sigma_clip(fl, stdfunc='mad_std', axis=1); cr = sigma_clip(ra, stdfunc='mad_std', axis=1); cd = sigma_clip(dec, stdfunc='mad_std', axis=1)
km = ~(cf.mask | cr.mask | cd.mask)
ng = km.sum(1); nm = np.isfinite(fl).sum(1)
print(f'F{b}: {nt} frames, {n} base sources')
for k in range(1, nt + 1):
    s = nm == k
    if s.sum():
        print(f'  nmatch={k}: {s.sum():6d} rows; nmatch_good histogram', np.bincount(ng[s], minlength=k + 1).tolist())
allm = (ng == 0) & (nm > 0)
anyforced = (fo & np.isfinite(fl)).any(1)
print('  all-masked', allm.sum(), '; with >=1 forced_refit frame', (allm & anyforced).sum(),
      '; masked-by-flux-only-all', (cf.mask.all(1) & allm).sum(), 'ra-all', (cr.mask.all(1) & allm).sum(), 'dec-all', (cd.mask.all(1) & allm).sum())
# identical positions (forced) -> MAD 0 on that axis
def mad0(a):
    med = np.nanmedian(a, 1)[:, None]
    return np.nanmedian(np.abs(a - med), 1) == 0
z = (mad0(ra) | mad0(dec)) & (nm >= 3)
print('  nmatch>=3 rows with a zero position MAD', z.sum(), '; of these all-masked', (z & allm).sum())
np.savez('/blue/adamginsburg/adamginsburg/tmp/claude-3663/overclip_%s.npz' % b, nm=nm, ng=ng, ra=ra, dec=dec, fl=fl)
