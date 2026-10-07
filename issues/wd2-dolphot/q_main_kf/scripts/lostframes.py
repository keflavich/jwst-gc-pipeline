"""For each lost star (matched in A, not in B): the frames of its detection band where A has a per-frame m7 daophot
row within 0.1", and the m7 satstar row counts of that frame in A and B.  Frames with B/A satstar count > 2 are the
ones where the A satstar fit dropped most saturated stars (R(g0) collapse, #1096).
usage: lostframes.py A B"""
import sys, glob, os, re
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
A, B = sys.argv[1:3]
tr = Table.read(f'{Q}/kf_lost/trace_{A}_{B}.ecsv')
lo = Table.read(f'{Q}/kf_lost/lost_{A}_{B}.ecsv')
pos = {int(r['i']): SkyCoord(r['RA'] * u.deg, r['DEC'] * u.deg) for r in lo}
rx = re.compile(r'_(nrc[ab][0-9a-z]+)_visit\d+_vgroup\d+_exp0*(\d+)_')


def nsat(arm, band, det, exp):
    f = glob.glob(f'{Q}/tree_{arm}/{band.upper()}/pipeline/jw*_0000{exp}_{det}_*crf_resbgsub_m7_satstar_catalog.fits')
    return len(Table.read(f[0])) if f else -1


frames = {}
for band in sorted(set(tr['band'])):
    for f in sorted(glob.glob(f'{Q}/tree_{A}/{band.upper()}/{band}_*_resbgsub_m7_daophot_basic.fits')):
        det, exp = rx.search(os.path.basename(f)).groups()
        t = Table.read(f)
        frames[(band, det, exp)] = SkyCoord(t['skycoord_centroid'])
cnt = {}
rows = []
for r in tr:
    b = str(r['band']); p = pos[int(r['i'])]
    for (band, det, exp), c in frames.items():
        if band != b:
            continue
        if p.separation(c).arcsec.min() < 0.1:
            k = (band, det, exp)
            if k not in cnt:
                cnt[k] = (nsat(A, band, det, exp), nsat(B, band, det, exp))
            rows.append((int(r['i']), band, det, exp, *cnt[k]))
t = Table(rows=rows, names=['i', 'band', 'det', 'exp', 'nsatA', 'nsatB'])
t.write(f'{Q}/kf_lost/lostframes_{A}_{B}.ecsv', overwrite=True)
hi = t['nsatB'] > 2 * np.maximum(t['nsatA'], 1)
print(f'{len(t)} (star, frame) A detections of {len(set(t["i"]))} lost stars; on frames with B/A satstar count > 2: {hi.sum()}')
per = {}
for r in t:
    per.setdefault(r['i'], []).append(r['nsatB'] > 2 * max(r['nsatA'], 1))
print(f'lost stars with every A detection on such frames: {sum(all(v) for v in per.values())}; with any: {sum(any(v) for v in per.values())}; none: {sum(not any(v) for v in per.values())}')
print('frames involved (band det exp nsatA nsatB n_lost_detections):')
for k, (na, nb) in sorted(cnt.items()):
    print(f'  {k[0]} {k[1]} exp{k[2]}: {na} {nb} {((t["band"] == k[0]) & (t["det"] == k[1]) & (t["exp"] == k[2])).sum()}')
