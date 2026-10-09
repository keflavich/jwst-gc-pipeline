"""Per band: combine_singleframe main vs fix on the main2kfpk per-frame m7 tables (run_combine.py outputs),
scored against dolphot (as clipacc.py), and the m8 value the main2kfpk catalog already gives each rescued star.
usage: python allband.py [bands...]   -> markdown table on stdout"""
import sys, os, warnings
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std
import astropy.units as u
warnings.filterwarnings('ignore')
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sys.path.insert(0, Q)
import analyze as an
IN = '/blue/adamginsburg/adamginsburg/tmp/claude-3663/sclip_real'
BANDS = sys.argv[1:] or ['115W', '150W', '162M', '182M', '200W', '250M', '277W', '300M', '335M', '410M',
                         '164N', '187N', '212N', '323N', '405N', '466N']
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2kfpk')
m8 = Table.read(f'{Q}/tree_main2kfpk/catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits')
rs = SkyCoord(np.asarray(A.m['RA'], float)*u.deg, np.asarray(A.m['DEC'], float)*u.deg)
oi = np.asarray(A.m['our_idx']); mt = np.asarray(A.matched, bool)

print('| band | rows | rescued | rescued within 0.08″ / good | other rows changed | stars with a good combine row, main → fix (lost) '
      '| rescued good stars: m8 value / forced-filled / good in m8 | net good stars at m8 | median \\|dm\\| rescued vs m8 (same stars) |')
print('|---|---|---|---|---|---|---|---|---|')
tot = np.zeros(4, int)
for b in BANDS:
    fM, fF = f'{IN}/comb_main_{b}.npz', f'{IN}/comb_fix_{b}.npz'
    if not (os.path.exists(fM) and os.path.exists(fF)):
        print(f'| F{b} | missing |'); continue
    R = {'main': np.load(fM), 'fix': np.load(fF)}
    ref = A.ref[b]; okr = np.flatnonzero(np.isfinite(ref)); rsk = rs[okr]; refk = ref[okr]

    def score(d, zp=None):
        ok = np.isfinite(d['ra']) & np.isfinite(d['dec']) & (d['flux'] > 0)
        c = SkyCoord(np.where(ok, d['ra'], 0)*u.deg, np.where(ok, d['dec'], 0)*u.deg)
        jj, sep, _ = c.match_to_catalog_sky(rsk)
        m = -2.5*np.log10(np.where(ok, d['flux'], np.nan))
        if zp is None:
            s = ok & (sep.arcsec < 0.05) & (refk[jj] >= 18.6) & (refk[jj] < 21)
            zp = np.median(refk[jj][s] - m[s])
        dm = m + zp - refk[jj]
        good = ok & (sep.arcsec < 0.08) & (np.abs(dm) < 0.3)
        return ok, jj, sep.arcsec, dm, good, zp
    okM, jM, sM, dmM, gM, zp = score(R['main'])
    okF, jF, sF, dmF, gF, _ = score(R['fix'], zp)
    same = okM & okF
    moved = same & ((R['main']['ra'] != R['fix']['ra']) | (R['main']['flux'] != R['fix']['flux']))
    new = okF & ~okM
    starsM, starsF = set(jM[gM]), set(jF[gF])
    ng = new & gF
    idx = okr[jF[ng]]; dmr = dmF[ng]
    dm8 = A.dm(b)[idx]; has = np.isfinite(dm8)
    ff = np.zeros(len(idx), bool); sel = mt[idx] & (oi[idx] >= 0)
    col = f'forced_filled_f{b.lower()}'
    if col in m8.colnames:
        ff[sel] = np.ma.filled(m8[col][oi[idx][sel]], False).astype(bool)
    good8 = has & (np.abs(dm8) < 0.3)
    net = len(idx) - good8.sum()
    tot += [new.sum(), ng.sum(), good8.sum(), net]
    med = (f'{np.median(np.abs(dmr[has])):.3f} vs {np.median(np.abs(dm8[has])):.3f}' if has.any() else '—')
    print(f'| F{b} | {len(okM)} | {new.sum()} | {(new & (sF < 0.08)).sum()} / {ng.sum()} | {moved.sum()} | '
          f'{len(starsM)} → {len(starsF)} ({len(starsM - starsF)}) | {has.sum()} / {(ff & has).sum()} / {good8.sum()} | '
          f'{net:+d} | {med} |')
print(f'\ntotals: rescued {tot[0]}, rescued good {tot[1]}, already good in m8 {tot[2]}, net good at m8 {tot[3]:+d}')
