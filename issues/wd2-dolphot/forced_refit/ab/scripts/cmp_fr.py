"""Forced-refit A/B (#932) scoring against dolphot for one band.
Per arm (per-band m7 merged catalog, before vetting): ZP as in fr_bias.py,
dm = pipeline - dolphot for dolphot stars with a row within 0.08" (not
replaced_saturated), split by the row's forced_refit_frac; good fractions;
spurious rows (no dolphot source within 0.1") in and out of the core.
usage: python cmp_fr.py BAND NAME=CATALOG [NAME=CATALOG ...]"""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

b = sys.argv[1].upper().lstrip('F')
arms = [a.split('=', 1) for a in sys.argv[2:]]
an.ZPWIN.update(an.zp_windows())
A = an.Arm('mainfcbg')
dsk = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)
core = SkyCoord(156.0 * u.deg, -57.757 * u.deg)
ref = A.ref[b]
have = np.where(np.isfinite(ref))[0]
incore = dsk[have].separation(core).arcsec < 20
EX = SkyCoord(156.00605 * u.deg, -57.75952 * u.deg)


def cell(x):
    return (f'{np.median(x):+.3f} / {np.mean(np.abs(x) > 0.3):.2f} / {len(x)}'
            if len(x) > 4 else f'- / - / {len(x)}')


print(f'## F{b}')
rows = {}
for name, path in arms:
    t = Table.read(path)
    sk = t['skycoord']
    fl = np.asarray(t['flux'], float)
    ok = np.isfinite(sk.ra.deg) & (fl > 0)
    t, sk, fl = t[ok], sk[ok], fl[ok]
    mi = -2.5 * np.log10(fl)
    rep = np.asarray(t['replaced_saturated'], bool)
    ff = np.asarray(t['forced_refit_frac'], float)
    mid = np.where(np.isfinite(ref) & (ref >= 18.6) & (ref < 21))[0]
    j, d, _ = dsk[mid].match_to_catalog_sky(sk)
    sel = (d.arcsec < 0.05) & ~rep[j]
    zp = np.median(ref[mid][sel] - mi[j][sel])
    j, d, _ = dsk[have].match_to_catalog_sky(sk)
    hit = (d.arcsec < 0.08) & ~rep[j]
    dm = mi[j] + zp - ref[have]
    m = mi + zp
    jj, dd, _ = sk.match_to_catalog_sky(dsk)
    nod = dd.arcsec > 0.1
    rc = sk.separation(core).arcsec < 20
    ie, de, _ = EX.match_to_catalog_sky(sk)
    print(f'### {name}: {len(t)} rows, ZP {zp:.3f}, forced rows (frac>0) {int((ff > 0).sum())}, '
          f'frac>=0.5 {int((ff >= 0.5).sum())}')
    print(f'  spurious (no dolphot within 0.1"): core <18.6 {int((nod & rc & (m < 18.6)).sum())}, '
          f'core all {int((nod & rc).sum())}, outside <18.6 {int((nod & ~rc & (m < 18.6)).sum())}, '
          f'outside all {int((nod & ~rc).sum())}')
    print(f'  example 156.00605 -57.75952: nearest row {de.arcsec[0]:.3f}" mag {m[int(ie)]:.2f} '
          f'frac {ff[int(ie)]:.2f}')
    print('| region | mag | all good | frac = 0 | 0 < frac < 0.5 | frac >= 0.5 |')
    print('|---|---|---|---|---|---|')
    for reg, rs in (('core', incore), ('outside', ~incore)):
        for lo, hi in ((10, 18.6), (18.6, 21), (21, 24)):
            inb = rs & (ref[have] >= lo) & (ref[have] < hi)
            s = hit & inb
            good = int((s & (np.abs(dm) < 0.3)).sum())
            cells = [cell(dm[s & fs]) for fs in
                     (ff[j] == 0, (ff[j] > 0) & (ff[j] < 0.5), ff[j] >= 0.5)]
            print(f'| {reg} | {lo}-{hi} | {good}/{int(inb.sum())} | ' + ' | '.join(cells) + ' |')
    rows[name] = (t, sk, m, ff)
