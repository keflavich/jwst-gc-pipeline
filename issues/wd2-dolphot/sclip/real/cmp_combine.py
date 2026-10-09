"""Compare combine_singleframe outputs (main vs fix) against dolphot, matching as clipacc.py.
usage: python cmp_combine.py <band>"""
import sys, warnings
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std
import astropy.units as u
warnings.filterwarnings('ignore')
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sys.path.insert(0, Q)
import analyze as an
IN = '/blue/adamginsburg/adamginsburg/tmp/claude-3663/sclip_real'
b = sys.argv[1]
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2kfpk')
rs = SkyCoord(np.asarray(A.m['RA'], float)*u.deg, np.asarray(A.m['DEC'], float)*u.deg)
ref = np.asarray(A.ref[b], float); okr = np.isfinite(ref); rsk = rs[okr]; refk = ref[okr]
R = {k: np.load(f'{IN}/comb_{k}_{b}.npz') for k in ('main', 'fix')}
assert len(R['main']['ra']) == len(R['fix']['ra'])


def score(d, zp=None):
    ok = np.isfinite(d['ra']) & np.isfinite(d['dec']) & (d['flux'] > 0)
    c = SkyCoord(np.where(ok, d['ra'], 0)*u.deg, np.where(ok, d['dec'], 0)*u.deg)
    jj, sep, _ = c.match_to_catalog_sky(rsk)
    m = -2.5*np.log10(np.where(ok, d['flux'], np.nan))
    if zp is None:
        s = ok & (sep.arcsec < 0.05) & (refk[jj] >= 18.6) & (refk[jj] < 21)
        zp = np.median(refk[jj][s] - m[s])
    dm = m + zp - refk[jj]
    near = ok & (sep.arcsec < 0.08)
    good = near & (np.abs(dm) < 0.3)
    return ok, jj, sep.arcsec, dm, good, zp


okM, jM, sM, dmM, gM, zp = score(R['main'])
okF, jF, sF, dmF, gF, _ = score(R['fix'], zp)
same = okM & okF
moved = same & ((np.abs(R['main']['ra'] - R['fix']['ra']) > 1e-9) | (np.abs(R['main']['flux'] - R['fix']['flux']) > 1e-6 * np.abs(R['main']['flux'])))
new = okF & ~okM
# dolphot stars with a good row (unique)
starsM = set(jM[gM]); starsF = set(jF[gF])
# rescued rows near an existing finite main row (possible duplicate)
cM = SkyCoord(R['main']['ra'][okM]*u.deg, R['main']['dec'][okM]*u.deg)
cN = SkyCoord(R['fix']['ra'][new]*u.deg, R['fix']['dec'][new]*u.deg)
_, dN, _ = cN.match_to_catalog_sky(cM)
print(f'F{b}: rows {len(okM)}; finite main {okM.sum()} fix {okF.sum()}; rescued {new.sum()}; '
      f'rows changed among finite-in-both {moved.sum()}; zp {zp:.4f}')
print(f'  good rows main {gM.sum()} fix {gF.sum()} (+{gF.sum()-gM.sum()}); dolphot stars with a good row '
      f'main {len(starsM)} fix {len(starsF)} (+{len(starsF)-len(starsM)}, lost {len(starsM-starsF)})')
print(f'  rescued rows: within 0.08" of dolphot {(new & (sF < 0.08)).sum()}, good {(new & gF).sum()}; '
      f'nearest other finite row < 0.1": {(dN.arcsec < 0.1).sum()}, median sep {np.median(dN.arcsec):.3f}"; '
      f'nmatch_good of rescued: {np.bincount(R["fix"]["nmatch_good"][new]).tolist()}')
newgood = new & gF
print(f'  rescued good rows whose dolphot star has no good main row: {len(set(jF[newgood]) - starsM)}')
mid = lambda ok, s, j: ok & (s < 0.08) & (refk[j] >= 18) & (refk[j] < 22)
print(f'  robust std dm 18-22: main {mad_std(dmM[mid(okM, sM, jM)]):.4f} fix {mad_std(dmF[mid(okF, sF, jF)]):.4f}')
np.save(f'{IN}/rescued_{b}.npy', np.flatnonzero(new))
