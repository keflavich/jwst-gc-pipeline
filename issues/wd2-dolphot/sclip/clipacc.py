"""Accuracy against dolphot of the Phase-1 union clip vs alternatives, nmatch>=3 rows (replica stacking from overclip.py).
Variants: union (production), fallback (union, all finite frames when union masks all), none (no clip),
fluxonly (flux clip only), floor (union clip with the clip scale floored: flux at 3% of median, position at 5 mas)."""
import sys, warnings
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip, mad_std
import astropy.units as u
warnings.filterwarnings('ignore')
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sys.path.insert(0, Q)
import analyze as an
b = sys.argv[1]
d = np.load('/blue/adamginsburg/adamginsburg/tmp/claude-3663/overclip_%s.npz' % b)
nm, ra, dec, fl = d['nm'], d['ra'], d['dec'], d['fl'].astype(float)
sel = nm >= 3
ra, dec, fl, nm = ra[sel], dec[sel], fl[sel], nm[sel]
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2kfpk')
rs = SkyCoord(np.asarray(A.m['RA'], float)*u.deg, np.asarray(A.m['DEC'], float)*u.deg)
ref = np.asarray(A.ref[b], float); okr = np.isfinite(ref); rsk = rs[okr]; refk = ref[okr]
fin = np.isfinite(fl)
cf = sigma_clip(fl, stdfunc='mad_std', axis=1).mask
cr = sigma_clip(ra, stdfunc='mad_std', axis=1).mask
cd = sigma_clip(dec, stdfunc='mad_std', axis=1).mask
def floored(x, floor):
    def f(a, axis=None, **kw):
        return np.maximum(mad_std(a, axis=axis, ignore_nan=True), floor)
    return sigma_clip(x, stdfunc=f, axis=1).mask
medf = np.nanmedian(fl, 1)
ffl = sigma_clip(fl / medf[:, None], stdfunc=lambda a, axis=None, **kw: np.maximum(mad_std(a, axis=axis, ignore_nan=True), 0.03), axis=1).mask
cosd = np.cos(np.deg2rad(np.nanmedian(dec, 1)))[:, None]
fra = floored((ra - np.nanmedian(ra, 1)[:, None]) * cosd * 3.6e6, 5.0)   # mas
fde = floored((dec - np.nanmedian(dec, 1)[:, None]) * 3.6e6, 5.0)
union = fin & ~(cf | cr | cd)
allm = ~union.any(1)
fb = union.copy(); fb[allm] = fin[allm]
V = {'union': union, 'fallback': fb, 'none': fin.copy(), 'fluxonly': fin & ~cf, 'floor': fin & ~(ffl | fra | fde)}
# zero point from unclipped rows, 18.6-21 mag
def comb(k):
    w = k.astype(float)
    with np.errstate(invalid='ignore', divide='ignore'):
        f = np.nansum(np.where(k, fl, 0), 1) / w.sum(1)
        r = np.nansum(np.where(k, ra, 0), 1) / w.sum(1)
        de = np.nansum(np.where(k, dec, 0), 1) / w.sum(1)
    return f, r, de
f0, r0, d0 = comb(V['none'])
c0 = SkyCoord(r0*u.deg, d0*u.deg)
j, dd, _ = c0.match_to_catalog_sky(rsk)
m0 = -2.5*np.log10(f0)
s = (dd.arcsec < 0.05) & (refk[j] >= 18.6) & (refk[j] < 21)
zp = np.median(refk[j][s] - m0[s])
print(f'F{b}: nmatch>=3 rows {len(nm)}, zp {zp:.4f}, all-masked under union {allm.sum()}')
print('  variant    rows_with_pos  good(<0.3,<0.08")  good_18-22  robust_std_dm_18-22  median_dm_18-22  median_nkept')
for name, k in V.items():
    f, r, de = comb(k)
    ok = np.isfinite(r) & np.isfinite(de) & (f > 0)
    c = SkyCoord(np.where(ok, r, 0)*u.deg, np.where(ok, de, 0)*u.deg)
    jj, sep, _ = c.match_to_catalog_sky(rsk)
    dm = -2.5*np.log10(np.where(ok, f, np.nan)) + zp - refk[jj]
    near = ok & (sep.arcsec < 0.08)
    good = near & (np.abs(dm) < 0.3)
    mid = near & (refk[jj] >= 18) & (refk[jj] < 22)
    print(f'  {name:9s} {ok.sum():8d} {good.sum():10d} {(good & mid).sum():12d} {mad_std(dm[mid]):14.4f} {np.median(dm[mid]):14.4f} {np.median(k.sum(1)):8.1f}')
