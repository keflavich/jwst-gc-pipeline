"""Lost stars (A only): sep, sep/lambda and PA to the nearest saturated A row in the band of the A row's
reference position; no trace file needed.  usage: sb_ab.py A B"""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
M8 = 'catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits'
A, B = sys.argv[1:3]
lo = Table.read(f'{Q}/kf_lost/lost_{A}_{B}.ecsv')
ca = Table.read(f'{Q}/tree_{A}/{M8}')
sk = SkyCoord(ca['skycoord_ref'])
out = []
for r in lo:
    b = ca['skycoord_ref_filtername'][int(r['Arow'])]
    b = (b.decode() if isinstance(b, bytes) else str(b)).strip()
    s = np.ma.filled(ca[f'replaced_saturated_{b}'], 0).astype(bool)
    p = sk[int(r['Arow'])]
    d = p.separation(sk[s]).arcsec
    j = int(np.argmin(d))
    lam = int(b[1:4]) / 100
    out.append((int(r['i']), b, d[j], d[j] / lam, sk[s][j].position_angle(p).deg % 360))
t = Table(rows=out, names=['i', 'band', 'sep', 'sep_over_lam', 'pa'])
t.write(f'{Q}/kf_lost/sb_{A}_{B}.ecsv', overwrite=True)
print('sep/lambda histogram:')
h, e = np.histogram(t['sep_over_lam'], bins=np.arange(0, 2.01, 0.05))
for n, l in zip(h, e):
    if n:
        print(f'  {l:.2f}-{l + 0.05:.2f}: {n:3d} ' + '#' * n)
m = t['sep'] < 3
print('PA mod 60 (sep<3"):')
h, e = np.histogram(np.asarray(t['pa'][m]) % 60, bins=np.arange(0, 61, 5))
for n, l in zip(h, e):
    print(f'  {l:2.0f}-{l + 5:2.0f}: {n:3d} ' + '#' * n)
print(f'N={len(t)}; sep<3": {m.sum()}')
print('bands:', dict(zip(*np.unique(t['band'], return_counts=True))))
