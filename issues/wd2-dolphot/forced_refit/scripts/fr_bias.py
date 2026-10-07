"""dm against dolphot split by forced_refit_frac, per band, on per-band m7
merged catalogs (unvetted) of one arm.  ZP as in score_lw.py: median over
dolphot 18.6-21 stars matched within 0.05" and not replaced_saturated.
usage: python fr_bias.py ARM [BAND ...]"""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an
arm = sys.argv[1]
bands = sys.argv[2:] or an.BANDS
an.ZPWIN.update(an.zp_windows())
A = an.Arm('mainfcbg')
dsk = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)
core = SkyCoord(156.0 * u.deg, -57.757 * u.deg)
print(f'## {arm}: dm = pipeline - dolphot for dolphot stars with a row within 0.08" (median / fraction |dm|>0.3 / N)')
print('| band | region | mag | forced_refit_frac = 0 | 0 < frac < 0.5 | frac >= 0.5 |')
print('|---|---|---|---|---|---|')
for b in bands:
    bl = 'f' + b.lower()
    t = Table.read(f'{an.Q}/tree_{arm}/catalogs/{bl}_merged_indivexp_merged_resbgsub_m7_dao_basic.fits')
    sk = t['skycoord']
    fl = np.asarray(t['flux'], float)
    ok = np.isfinite(sk.ra.deg) & (fl > 0)
    t, sk, fl = t[ok], sk[ok], fl[ok]
    mi = -2.5 * np.log10(fl)
    rep = np.asarray(t['replaced_saturated'], bool)
    ff = np.asarray(t['forced_refit_frac'], float)
    ref = A.ref[b]
    mid = np.where(np.isfinite(ref) & (ref >= 18.6) & (ref < 21))[0]
    j, d, _ = dsk[mid].match_to_catalog_sky(sk)
    sel = (d.arcsec < 0.05) & ~rep[j]
    zp = np.median(ref[mid][sel] - mi[j][sel])
    have = np.where(np.isfinite(ref))[0]
    j, d, _ = dsk[have].match_to_catalog_sky(sk)
    hit = (d.arcsec < 0.08) & ~rep[j]
    dm = mi[j] + zp - ref[have]
    incore = dsk[have].separation(core).arcsec < 20
    for reg, rs in (('core', incore), ('outside', ~incore)):
        for lo, hi in ((10, 18.6), (18.6, 21), (21, 24)):
            s = hit & rs & (ref[have] >= lo) & (ref[have] < hi)
            cells = []
            for fs in (ff[j] == 0, (ff[j] > 0) & (ff[j] < 0.5), ff[j] >= 0.5):
                x = dm[s & fs]
                cells.append(f'{np.median(x):+.3f} / {np.mean(np.abs(x) > 0.3):.2f} / {len(x)}' if len(x) > 4 else f'- / - / {len(x)}')
            print(f'| F{b} | {reg} | {lo}-{hi} | ' + ' | '.join(cells) + ' |')
