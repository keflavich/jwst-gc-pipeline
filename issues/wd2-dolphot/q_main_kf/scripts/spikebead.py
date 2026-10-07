"""Lost stars (matched in A, not in B): separation and position angle to the nearest saturated m8 row of the
same band in A, scaled by wavelength.  Tests whether lost stars sit at fixed sep/lambda (diffraction-spike beads).
usage: spikebead.py A B"""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
M8 = 'catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits'
A, B = sys.argv[1:3]
tr = Table.read(f'{Q}/kf_lost/trace_{A}_{B}.ecsv')
lo = Table.read(f'{Q}/kf_lost/lost_{A}_{B}.ecsv')
ca = Table.read(f'{Q}/tree_{A}/{M8}')
pos = {int(r['i']): SkyCoord(r['RA'] * u.deg, r['DEC'] * u.deg) for r in lo}
out = []
for r in tr:
    b = str(r['band'])
    col = f'replaced_saturated_{b}'
    s = np.ma.filled(ca[col], 0).astype(bool) if col in ca.colnames else np.zeros(len(ca), bool)
    cs = SkyCoord(ca['skycoord_ref'][s])
    p = pos[int(r['i'])]
    d = p.separation(cs).arcsec
    j = int(np.nanargmin(d))
    lam = int(b[1:4]) / 100
    out.append((int(r['i']), b, lam, d[j], d[j] / lam, cs[j].position_angle(p).deg % 360, int(r['nframesB'])))
t = Table(rows=out, names=['i', 'band', 'lam', 'sep', 'sep_over_lam', 'pa', 'nframesB'])
t.write(f'{Q}/kf_lost/spikebead_{A}_{B}.ecsv', overwrite=True)
h, e = np.histogram(t['sep_over_lam'], bins=np.arange(0, 2.01, 0.05))
print('sep/lambda ["/um] histogram (all lost):')
for n, lo_ in zip(h, e):
    if n:
        print(f'  {lo_:.2f}-{lo_ + 0.05:.2f}: {n} ' + '#' * n)
print('PA mod 60 histogram (sep < 3"):')
m = t['sep'] < 3
h, e = np.histogram(np.asarray(t['pa'][m]) % 60, bins=np.arange(0, 61, 5))
for n, lo_ in zip(h, e):
    print(f'  {lo_:2.0f}-{lo_ + 5:2.0f}: {n} ' + '#' * n)
print(f'N={len(t)}; sep<3": {m.sum()}')
