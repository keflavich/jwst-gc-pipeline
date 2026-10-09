"""Count split stars: pairs of merged (pre-replacement) rows < R arcsec apart whose contributing frame sets are disjoint
(the same star seen in different frames but assigned to two base sources by the greedy Loop 1 of combine_singleframe).
Scores the effect on dolphot: for each pair compare merged-flux-combined mag with dolphot at the pair's mean position. usage: python fragments.py BAND"""
import sys, glob, warnings
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sys.path.insert(0, Q)
import analyze as an
b = sys.argv[1]
T = f'{Q}/tree_main2kfpk'
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2kfpk')
rs = SkyCoord(np.asarray(A.m['RA'], float)*u.deg, np.asarray(A.m['DEC'], float)*u.deg); ref = np.asarray(A.ref[b], float)
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
fl = np.full((n, nt), np.nan); ra = fl.copy(); dec = fl.copy()
for ii, t in enumerate(tabs):
    c = t['skycoord_centroid']
    mi, sep, _ = c.match_to_catalog_sky(base); ri, _, _ = base.match_to_catalog_sky(c)
    keep = (sep < 0.1*u.arcsec) & (ri[mi] == np.arange(len(mi))); m = mi[keep]
    fl[m, ii] = t['flux_fit'][keep]; ra[m, ii] = c.ra.deg[keep]; dec[m, ii] = c.dec.deg[keep]
fin_ = np.isfinite(fl)
pre = Table.read(f'/blue/adamginsburg/adamginsburg/tmp/claude-3663/pre_{b}.fits')
pos = pre['skycoord_avg']; okp = np.isfinite(pos.ra.deg)
# flat tangent coordinates in arcsec
ra0, dec0 = np.nanmedian(pos.ra.deg), np.nanmedian(pos.dec.deg)
x = (pos.ra.deg - ra0)*np.cos(np.radians(dec0))*3600; y = (pos.dec.deg - dec0)*3600
idx = np.where(okp)[0]; tree = cKDTree(np.column_stack([x[idx], y[idx]]))
pairs = tree.query_pairs(0.12, output_type='ndarray'); pi, pj = idx[pairs[:, 0]], idx[pairs[:, 1]]
disjoint = (fin_[pi] & fin_[pj]).sum(1) == 0
d = np.hypot(x[pi]-x[pj], y[pi]-y[pj])
print(b, 'rows', n, 'pairs<0.12"', len(pi), 'disjoint-frame pairs', int(disjoint.sum()), ' <0.1":', int((disjoint & (d < 0.1)).sum()))
nm = fin_.sum(1)
# also does a frame set union stay <=4 (one module)? count pairs where union size<=nt//2+... report distribution of union
un = (fin_[pi] | fin_[pj]).sum(1)
print('  disjoint-pair union sizes', np.bincount(un[disjoint], minlength=9))
# fix estimate: merge disjoint pairs; compare combined flux (ivar-free mean of the union) vs dolphot at the merged position
zpmid = None
fin = Table.read(f'{T}/catalogs/f{b.lower()}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
ff = np.asarray(fin['flux'], float); sk = fin['skycoord']; ok = np.isfinite(sk.ra.deg) & (ff > 0) & ~np.asarray(fin['replaced_saturated'], bool)
mid = np.where(np.isfinite(ref) & (ref >= 18.6) & (ref < 21))[0]
j, dd, _ = rs[mid].match_to_catalog_sky(sk[ok]); sel = dd.arcsec < 0.05
zp = np.median(ref[mid][sel] - (-2.5*np.log10(ff[ok][j])[sel]))
sel_p = disjoint & (d < 0.1)
mflux = np.nanmean(np.where(fin_[pi[sel_p]] | fin_[pj[sel_p]], np.where(fin_[pi[sel_p]], fl[pi[sel_p]], fl[pj[sel_p]]), np.nan), axis=1)
mra = np.nanmean(np.where(fin_[pi[sel_p]], ra[pi[sel_p]], ra[pj[sel_p]]), axis=1); mdec = np.nanmean(np.where(fin_[pi[sel_p]], dec[pi[sel_p]], dec[pj[sel_p]]), axis=1)
P = SkyCoord(mra*u.deg, mdec*u.deg); ii, d2, _ = P.match_to_catalog_sky(rs)
mm = -2.5*np.log10(mflux) + zp
# current: each fragment compared separately
fi_ = pre['flux_fit_avg']
cur_i = -2.5*np.log10(np.asarray(fi_[pi[sel_p]], float)) + zp; cur_j = -2.5*np.log10(np.asarray(fi_[pj[sel_p]], float)) + zp
g = (d2.arcsec < 0.08) & np.isfinite(ref[ii])
print('  split pairs matched to a dolphot star:', int(g.sum()), '| merged-union |dm|<0.3:', int((np.abs(mm[g]-ref[ii][g]) < 0.3).sum()),
      '| both fragments |dm|<0.3:', int(((np.abs(cur_i-ref[ii]) < 0.3) & (np.abs(cur_j-ref[ii]) < 0.3))[g].sum()),
      '| only one fragment good:', int(((np.abs(cur_i-ref[ii]) < 0.3) ^ (np.abs(cur_j-ref[ii]) < 0.3))[g].sum()),
      '| neither fragment good:', int(((np.abs(cur_i-ref[ii]) >= 0.3) & (np.abs(cur_j-ref[ii]) >= 0.3))[g].sum()))
Table({'pre_i': pi[sel_p], 'pre_j': pj[sel_p], 'sep': d[sel_p], 'nm_i': nm[pi[sel_p]], 'nm_j': nm[pj[sel_p]], 'dol_idx': ii, 'dol_sep': d2.arcsec, 'ref': ref[ii], 'mag_i': cur_i, 'mag_j': cur_j, 'mag_union': mm}).write(f'fragments_{b}.ecsv', overwrite=True)
