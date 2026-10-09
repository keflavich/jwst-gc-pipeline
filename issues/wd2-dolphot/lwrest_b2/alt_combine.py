"""Offline alternatives for the merge flux/position combine (combine_singleframe Phase 1). Scores against dolphot.
usage: python alt_combine.py BAND. Writes allmasked_BAND.ecsv (rows with nmatch>0 & nmatch_good==0 and what an unclipped mean gives) and prints a scoring table."""
import sys, glob, warnings
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip
import astropy.units as u
warnings.filterwarnings('ignore')
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sys.path.insert(0, Q)
import analyze as an
b = sys.argv[1]
T = f'{Q}/tree_main2kfpk'
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2kfpk')
rs = SkyCoord(np.asarray(A.m['RA'], float)*u.deg, np.asarray(A.m['DEC'], float)*u.deg)
ref = np.asarray(A.ref[b], float)
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
ra = np.full((n, nt), np.nan); dec = ra.copy(); fl = np.full((n, nt), np.nan, dtype='float32'); fe = fl.copy()
for ii, t in enumerate(tabs):
    c = t['skycoord_centroid']
    mi, sep, _ = c.match_to_catalog_sky(base); ri, _, _ = base.match_to_catalog_sky(c)
    keep = (sep < 0.1*u.arcsec) & (ri[mi] == np.arange(len(mi))); m = mi[keep]
    ra[m, ii] = c.ra.deg[keep]; dec[m, ii] = c.dec.deg[keep]; fl[m, ii] = t['flux_fit'][keep]; fe[m, ii] = t['flux_err'][keep]
cf = sigma_clip(fl, stdfunc='mad_std', axis=1); cr = sigma_clip(ra, stdfunc='mad_std', axis=1); cd = sigma_clip(dec, stdfunc='mad_std', axis=1)
keepm = ~(cf.mask | cr.mask | cd.mask)
nm = np.isfinite(fl).sum(1); ng = keepm.sum(1)
pre = Table.read(f'/blue/adamginsburg/adamginsburg/tmp/claude-3663/pre_{b}.fits')
assert (pre['nmatch'] == nm).all() and (pre['nmatch_good'] == ng).all()
# ZP from the final catalog recipe (unreplaced rows, 18.6<ref<21, sep<0.05)
fin = Table.read(f'{T}/catalogs/f{b.lower()}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
ff = np.asarray(fin['flux'], float); sk = fin['skycoord']; ok = np.isfinite(sk.ra.deg) & (ff > 0) & ~np.asarray(fin['replaced_saturated'], bool)
mid = np.where(np.isfinite(ref) & (ref >= 18.6) & (ref < 21))[0]
j, d, _ = rs[mid].match_to_catalog_sky(sk[ok]); sel = d.arcsec < 0.05
zp = np.median(ref[mid][sel] - (-2.5*np.log10(ff[ok][j])[sel])); print(b, 'zp', zp)
with np.errstate(all='ignore'):
    fl64 = fl.astype(float)
    O = {}
    O['current'] = np.asarray(pre['flux_fit_avg'], float)
    O['median_unclipped'] = np.nanmedian(fl64, axis=1)
    O['max_frame'] = np.nanmax(fl64, axis=1)
    srt = -np.sort(-np.where(np.isfinite(fl64), fl64, -np.inf), axis=1)
    kk = np.maximum(1, np.ceil(nm/2).astype(int))
    O['mean_brightest_half'] = np.array([srt[i, :kk[i]].mean() if nm[i] > 0 else np.nan for i in range(n)])
    w = 1/fe.astype(float)**2
    O['ivar_mean_unclipped'] = np.nansum(fl64*w, axis=1)/np.nansum(np.where(np.isfinite(fl64), w, 0), axis=1)
    O['flux_mean_unclipped'] = np.nanmean(fl64, axis=1)
# unclipped mean position for all-masked rows
mra = np.nanmean(ra, axis=1); mdec = np.nanmean(dec, axis=1)
allm = (ng == 0) & (nm > 0)
P = SkyCoord(mra[allm]*u.deg, mdec[allm]*u.deg)
ii, dd, _ = P.match_to_catalog_sky(rs)
tab = Table({'base_idx': np.where(allm)[0], 'nmatch': nm[allm], 'ra': mra[allm], 'dec': mdec[allm], 'mag_mean_unclipped': -2.5*np.log10(O['flux_mean_unclipped'][allm]) + zp,
             'mag_ivar_unclipped': -2.5*np.log10(O['ivar_mean_unclipped'][allm]) + zp, 'dol_idx': ii, 'dol_sep': dd.arcsec, 'dol_ref': ref[ii]})
tab['dm'] = tab['mag_ivar_unclipped'] - tab['dol_ref']
tab.write(f'allmasked_{b}.ecsv', overwrite=True)
g = (tab['dol_sep'] < 0.08) & np.isfinite(tab['dol_ref'])
print(f'all-masked rows {allm.sum()}; with a dolphot star <0.08" {int(g.sum())}; of those |dm|<0.3 with ivar mean {int((np.abs(tab["dm"][g]) < 0.3).sum())}; bright (ref<18.5): {int((g & (tab["dol_ref"] < 18.5)).sum())}')
# population scoring of flux combine options vs dolphot: pre rows (non-replaced final rows are unaffected by satstar replacement) with nmatch>=2 and finite pos
pos = pre['skycoord_avg']; okp = np.isfinite(pos.ra.deg)
ip, dp, _ = pos[okp].match_to_catalog_sky(rs)
idx = np.where(okp)[0]
good = (dp.arcsec < 0.08) & np.isfinite(ref[ip]) & (nm[idx] >= 2)
idx, ip = idx[good], ip[good]
print('rows compared', len(idx), '(nmatch>=2 within 0.08" of a dolphot star with band mag)')
res = []
for k, f in O.items():
    m = -2.5*np.log10(f[idx]) + zp
    dm = m - ref[ip]
    row = dict(band=b, option=k)
    for lo, hi, lab in ((10, 16, '<16'), (16, 18, '16-18'), (18, 19.5, '18-19.5'), (10, 19.5, 'all')):
        s = (ref[ip] >= lo) & (ref[ip] < hi)
        row['good_' + lab] = int((np.abs(dm[s]) < 0.3).sum()); row['n_' + lab] = int(s.sum())
    row['n_changed_gt0.1_vs_current'] = int((np.abs(-2.5*np.log10(f[idx]/O['current'][idx])) > 0.1).sum()) if k != 'current' else 0
    row['n_changed_gt0.3_vs_current'] = int((np.abs(-2.5*np.log10(f[idx]/O['current'][idx])) > 0.3).sum()) if k != 'current' else 0
    res.append(row)
Rt = Table(rows=res); Rt.write(f'alt_combine_{b}.ecsv', overwrite=True); Rt.pprint_all()
# the c stars
C = Table.read('c_detail.ecsv'); C = C[C['band'] == b]
for r in C:
    bi = int(r['base_idx'])
    if bi < 0: continue
    print(r['dolphot_idx'], f"ref={r['ref']:.2f}", ' '.join(f"{k}:{(-2.5*np.log10(O[k][bi])+zp)-r['ref']:+.2f}" for k in O))
