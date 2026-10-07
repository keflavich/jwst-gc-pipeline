"""m7 satstar rows, KEEP_FINITE on arm vs off arm (default mainfcbgkf vs mainfcbg; KF_OFF / KF_ON env pick others), per band.
Rows are matched per frame on xcentroid/ycentroid within 0.5 px.  Reports row counts, rows present in one arm only,
non-positive / non-finite flux_fit per arm, and the on/off flux_fit ratio (median, 2.5-97.5 %, sign flips,
fraction > 20 % and > 5 % off unity), overall and by sat_area.
usage: python kf_satcheck.py [BAND ...]"""
import os
import sys
import glob
import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
OFF, ON = os.environ.get('KF_OFF', 'mainfcbg'), os.environ.get('KF_ON', 'mainfcbgkf')
MARK = f'{Q}/guard/manifest_{OFF}_{ON}.json'
BANDS = sys.argv[1:] or ['F115W', 'F150W', 'F162M', 'F164N', 'F182M', 'F187N', 'F200W', 'F212N',
                         'F250M', 'F277W', 'F300M', 'F323N', 'F335M', 'F405N', 'F410M', 'F466N']
SAB = [(0, 30), (30, 300), (300, 3000), (3000, 1e9)]
t0 = os.path.getmtime(MARK)


def load(fn):
    t = Table.read(fn)
    f = np.ma.filled(t['flux_fit'], np.nan).astype(float)
    return (np.c_[np.ma.filled(t['xcentroid'], np.nan), np.ma.filled(t['ycentroid'], np.nan)].astype(float),
            f, np.ma.filled(t['sat_area'], 0).astype(float))


def stats(r):
    if len(r) == 0:
        return 'N=0'
    lo, hi = np.percentile(r, [2.5, 97.5])
    return (f'N={len(r)} med={np.median(r):.4f} [{lo:.3f}, {hi:.3f}] flips={(r < 0).sum()} '
            f'>20%={(np.abs(r - 1) > 0.2).sum()} >5%={(np.abs(r - 1) > 0.05).sum()}')


print('| band | frames | rows off / on | off-only / on-only | flux<=0 or NaN off / on | on/off flux_fit ratio |')
print('|---|---|---|---|---|---|')
detail = []
for b in BANDS:
    offs = sorted(f for f in glob.glob(f'{Q}/tree_{OFF}/{b}/pipeline/*_m7_satstar_catalog.fits') if os.path.getmtime(f) > t0)
    nfr = noff = non = oo = no = negoff = negon = 0
    R, S = [], []
    for fo in offs:
        fn = fo.replace(f'/tree_{OFF}/', f'/tree_{ON}/')
        if not (os.path.exists(fn) and os.path.getmtime(fn) > t0):
            continue
        nfr += 1
        po, fo_, so = load(fo)
        pn, fn_, sn = load(fn)
        noff += len(fo_); non += len(fn_)
        negoff += int((~(fo_ > 0)).sum()); negon += int((~(fn_ > 0)).sum())
        if len(po) == 0 or len(pn) == 0:
            oo += len(po); no += len(pn)
            continue
        d, i = cKDTree(pn).query(po, distance_upper_bound=0.5)
        ok = np.isfinite(d)
        oo += int((~ok).sum())
        no += len(pn) - len(np.unique(i[ok]))
        a, c = fo_[ok], fn_[i[ok]]
        good = np.isfinite(a) & np.isfinite(c) & (a != 0)
        R.append(c[good] / a[good]); S.append(so[ok][good])
    if nfr == 0:
        print(f'| {b} | 0 | | | | |')
        continue
    R = np.concatenate(R); S = np.concatenate(S)
    print(f'| {b} | {nfr} | {noff} / {non} | {oo} / {no} | {negoff} / {negon} | {stats(R)} |')
    for lo, hi in SAB:
        s = (S >= lo) & (S < hi)
        detail.append(f'{b} sat_area {lo:g}-{hi:g}: {stats(R[s])}')
print()
print('\n'.join(detail))
