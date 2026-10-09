"""Trace b2 and c stars through the merge. usage: python trace.py BAND -> trace_BAND.ecsv (per-star), rows_BAND.ecsv (per-row evidence)."""
import sys, glob, re, warnings
import numpy as np
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u
from collections import defaultdict
warnings.filterwarnings('ignore')
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sys.path.insert(0, Q)
import analyze as an
b = sys.argv[1].upper().lstrip('F')
T = f'{Q}/tree_main2kfpk'
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2kfpk')
rs = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)
ref = A.ref[b]
pre = Table.read(f'/blue/adamginsburg/adamginsburg/tmp/claude-3663/pre_{b}.fits')
fin = Table.read(f'{T}/catalogs/f{b.lower()}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
sat = Table.read(f'{T}/catalogs/f{b.lower()}_consolidated_satstar_catalog.fits')
# ZP exactly as gather.py
sk = fin['skycoord']; fl = np.asarray(fin['flux'], float)
ok = np.isfinite(sk.ra.deg) & (fl > 0)
t, sk2, fl2 = fin[ok], sk[ok], fl[ok]
mi = -2.5 * np.log10(fl2); rep = np.asarray(t['replaced_saturated'], bool)
mid = np.where(np.isfinite(ref) & (ref >= 18.6) & (ref < 21))[0]
j, d, _ = rs[mid].match_to_catalog_sky(sk2)
sel = (d.arcsec < 0.05) & ~rep[j]
zp = np.median(ref[mid][sel] - mi[j][sel]); print('zp', zp)
def mag(f): 
    f = np.asarray(f, float)
    with np.errstate(divide='ignore', invalid='ignore'): return -2.5*np.log10(f)+zp
# map pre -> fin
def key(q,l,f,n):
    q=np.ma.filled(np.ma.asarray(q,float),np.nan); l=np.ma.filled(np.ma.asarray(l,float),np.nan); f=np.ma.filled(np.ma.asarray(f,float),np.nan)
    return [(f'{a:.7g}',f'{c:.7g}',f'{d:.5g}',int(e)) for a,c,d,e in zip(q,l,f,np.asarray(n))]
dd = defaultdict(list)
for i,k in enumerate(key(fin['qfit'],fin['local_bkg'],fin['flux_init'],fin['nmatch'])): dd[k].append(i)
pm = np.full(len(pre), -1)
print('pre rows mapped (set below)')
for i,k in enumerate(key(pre['qfit_avg'],pre['local_bkg_avg'],pre['flux_init_avg'],pre['nmatch'])):
    if k in dd and len(dd[k])==1: pm[i] = dd[k][0]
pre_sc = pre['skycoord_avg']; fin_sc = fin['skycoord']
fin_ok = np.isfinite(fin_sc.ra.deg)
pre_ok = np.isfinite(pre_sc.ra.deg)
# per-frame tables
fns = sorted(glob.glob(f'{T}/F{b}/f{b.lower()}_*_visit*_vgroup*_exp*_resbgsub_m7_daophot_basic.fits'))
pf = []
for fn in fns:
    mm = re.search(r'_(nrc[ab]long)_.*_exp(\d{5})_', fn)
    tb = Table.read(fn); tb['frame'] = f'{mm.group(1)}_{mm.group(2)}'
    pf.append(tb)
cls = Table.read(f'{Q}/lwrest/class_{b}.ecsv')
rows_out = []; star_out = []
for s in cls:
    c = s['cat'][:2]
    if c not in ('b2', 'c:'): continue
    idx = int(s['dolphot_idx']); p = rs[idx]; rm = float(ref[idx])
    rec = dict(band=b, dolphot_idx=idx, cat=c.rstrip(':'), ref_mag=rm)
    # per-frame rows
    nfr = 0
    for tb in pf:
        sc = tb['skycoord_centroid']
        ok_ = np.isfinite(sc.ra.deg)
        sep = np.full(len(tb), np.inf); sep[ok_] = sc[ok_].separation(p).arcsec
        for k in np.where(sep < 0.6)[0]:
            rows_out.append(dict(band=b, dolphot_idx=idx, kind='frame', where=tb['frame'][0], sep=sep[k], mag=float(mag(tb['flux_fit'][k])),
                                 flux=float(tb['flux_fit'][k]), nmatch=-1, flags=int(tb['flags'][k]), qfit=float(tb['qfit'][k]), forced=bool(tb['forced_refit'][k]) if 'forced_refit' in tb.colnames else False,
                                 gsize=int(tb['group_size'][k]), ra=sc.ra.deg[k], dec=sc.dec.deg[k], rep=False, x=float(tb['x_fit'][k]), y=float(tb['y_fit'][k])))
        nfr += int((sep < 0.5*0.063*1.48/0.063*0.063).sum() > 0) if False else int((sep < 0.063).sum() > 0)
    rec['n_frames_row_1px'] = nfr
    # pre rows
    sep = np.full(len(pre), np.inf); sep[pre_ok] = pre_sc[pre_ok].separation(p).arcsec
    for k in np.where(sep < 0.6)[0]:
        f_ = pm[k]
        rows_out.append(dict(band=b, dolphot_idx=idx, kind='pre', where=f'pre{k}', sep=sep[k], mag=float(mag(pre['flux_fit_avg'][k])), flux=float(pre['flux_fit_avg'][k]),
                             nmatch=int(pre['nmatch'][k]), flags=float(pre['flags_avg'][k]) if 'flags_avg' in pre.colnames else np.nan, qfit=float(pre['qfit_avg'][k]), forced=False, gsize=-1,
                             ra=pre_sc.ra.deg[k], dec=pre_sc.dec.deg[k], rep=bool(fin['replaced_saturated'][f_]) if f_>=0 else False,
                             x=float(fin_sc[f_].separation(pre_sc[k]).arcsec) if f_>=0 else np.nan, y=float(fin['satstar_match_sep'][f_]) if f_>=0 else np.nan))
    sepf = np.full(len(fin), np.inf); sepf[fin_ok] = fin_sc[fin_ok].separation(p).arcsec
    for k in np.where(sepf < 0.6)[0]:
        rows_out.append(dict(band=b, dolphot_idx=idx, kind='final', where=f'fin{k}', sep=sepf[k], mag=float(mag(fin['flux'][k])), flux=float(fin['flux'][k]), nmatch=int(fin['nmatch'][k]),
                             flags=float(fin['flags'][k]), qfit=float(fin['qfit'][k]), forced=False, gsize=-1, ra=fin_sc.ra.deg[k], dec=fin_sc.dec.deg[k], rep=bool(fin['replaced_saturated'][k]),
                             x=np.nan, y=float(fin['satstar_match_sep'][k])))
    ss = sat['skycoord_fit']; oks = np.isfinite(ss.ra.deg)
    sps = np.full(len(sat), np.inf); sps[oks] = ss[oks].separation(p).arcsec
    for k in np.where(sps < 0.6)[0]:
        rows_out.append(dict(band=b, dolphot_idx=idx, kind='satstar', where=f'sat{k}', sep=sps[k], mag=float(mag(sat['flux_fit'][k])), flux=float(sat['flux_fit'][k]), nmatch=int(sat['n_frames_fit'][k]) if 'n_frames_fit' in sat.colnames else -1,
                             flags=np.nan, qfit=float(sat['qfit'][k]) if 'qfit' in sat.colnames else np.nan, forced=False, gsize=-1, ra=ss.ra.deg[k], dec=ss.dec.deg[k], rep=False, x=np.nan, y=np.nan))
    star_out.append(rec)
R = Table(rows_out); R.write(f'rows_{b}.ecsv', overwrite=True)
Table(star_out).write(f'stars_{b}.ecsv', overwrite=True)
print(len(star_out), len(R), 'zp', zp, 'mapped', int((pm>=0).sum()), 'of', len(pm))
