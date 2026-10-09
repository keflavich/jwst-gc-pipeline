"""Population study of replace_saturated second-pass (mutual-nearest, 0.1"<sep<0.5") pairs with dolphot as truth.
For every final row with replaced_saturated & satstar_match_sep>=radius, compare the dolphot star at the OLD (pre-replacement) position and at the satstar position.
Writes second_pass_BAND.ecsv and prints a summary."""
import sys, warnings
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
from collections import defaultdict
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
okr = np.isfinite(ref); rsk = rs[okr]; refk = ref[okr]
pre = Table.read(f'/blue/adamginsburg/adamginsburg/tmp/claude-3663/pre_{b}.fits')
fin = Table.read(f'{T}/catalogs/f{b.lower()}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
def key(q,l,f,n):
    q=np.ma.filled(np.ma.asarray(q,float),np.nan); l=np.ma.filled(np.ma.asarray(l,float),np.nan); f=np.ma.filled(np.ma.asarray(f,float),np.nan)
    return [(f'{a:.7g}',f'{c:.7g}',f'{d:.5g}',int(e)) for a,c,d,e in zip(q,l,f,np.asarray(n))]
dd = defaultdict(list)
for i,k in enumerate(key(fin['qfit'],fin['local_bkg'],fin['flux_init'],fin['nmatch'])): dd[k].append(i)
pm = np.full(len(pre), -1)
for i,k in enumerate(key(pre['qfit_avg'],pre['local_bkg_avg'],pre['flux_init_avg'],pre['nmatch'])):
    if k in dd and len(dd[k]) == 1: pm[i] = dd[k][0]
# zp
fl = np.asarray(fin['flux'], float); sk = fin['skycoord']
ok = np.isfinite(sk.ra.deg) & (fl > 0) & ~np.asarray(fin['replaced_saturated'], bool)
mi = -2.5*np.log10(fl[ok]); mid = np.where(np.isfinite(ref) & (ref >= 18.6) & (ref < 21))[0]
j, d, _ = rs[mid].match_to_catalog_sky(sk[ok]); sel = d.arcsec < 0.05
zp = np.median(ref[mid][sel] - mi[j][sel])
rows = []
sel_pre = np.where((pm >= 0))[0]
fi = pm[sel_pre]
rep = np.asarray(fin['replaced_saturated'], bool)[fi]
msep = np.asarray(fin['satstar_match_sep'], float)[fi]
use = rep & np.isfinite(msep)
sel_pre = sel_pre[use]; fi = fi[use]; msep = msep[use]
oldpos = pre['skycoord_avg'][sel_pre]; newpos = fin['skycoord'][fi]
okp = np.isfinite(oldpos.ra.deg)
sel_pre, fi, msep, oldpos, newpos = sel_pre[okp], fi[okp], msep[okp], oldpos[okp], newpos[okp]
io, do, _ = oldpos.match_to_catalog_sky(rsk); isat, dsat, _ = newpos.match_to_catalog_sky(rsk)
oldmag = -2.5*np.log10(np.asarray(pre['flux_fit_avg'], float)[sel_pre]) + zp
satmag = -2.5*np.log10(np.asarray(fin['flux'], float)[fi]) + zp
T_ = Table({'pre_idx': sel_pre, 'fin_idx': fi, 'satsep': msep, 'oldmag': oldmag, 'satmag': satmag, 'dold': do.arcsec, 'dsat': dsat.arcsec,
            'old_ref': refk[io], 'sat_ref': refk[isat], 'old_dol_idx': np.where(okr)[0][io], 'sat_dol_idx': np.where(okr)[0][isat], 'qfit': np.ma.filled(np.ma.asarray(pre['qfit_avg'],float)[sel_pre],np.nan), 'nmatch': np.asarray(pre['nmatch'])[sel_pre], 'old_flags': np.ma.filled(np.ma.asarray(pre['flags_avg'],float)[sel_pre],np.nan), 'pass': np.where(msep < {'277W':0.1,'250M':0.08,'300M':0.1}[b], 1, 2)})
T_['same_dol'] = T_['old_dol_idx'] == T_['sat_dol_idx']
T_.write(f'second_pass_{b}.ecsv', overwrite=True)
p2 = T_[T_['pass'] == 2]
print(b, 'replaced rows', len(T_), 'second-pass', len(p2), 'first-pass', len(T_) - len(p2), 'zp', zp)
close_o = p2['dold'] < 0.08; close_s = p2['dsat'] < 0.08
print('  second-pass: dolphot at old pos & at sat pos (distinct):', int((close_o & close_s & ~p2['same_dol']).sum()),
      '| only at sat pos:', int((~close_o & close_s).sum()), '| only at old pos:', int((close_o & ~close_s).sum()),
      '| same dolphot star within 0.08 of both:', int((close_o & close_s & p2['same_dol']).sum()), '| neither:', int((~close_o & ~close_s).sum()))
