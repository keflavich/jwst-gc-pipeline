"""Summaries from per_frame.ecsv / per_star.ecsv (run trace_pk3.py first)."""
import json
import numpy as np
from astropy.table import Table
O = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/pk_deblend_trace'
pf = Table.read(f'{O}/per_frame.ecsv')
ps = Table.read(f'{O}/per_star.ecsv')
ZP = json.load(open(f'{O}/zp.json'))['ZP']
print('ZP', ZP)
names = {1: 'satstar row <0.08"', 2: 'satstar row 0.08-0.5"', 3: 'rejected row <0.5"', 4: 'daophot row <0.08"', 5: 'nothing <0.5"'}
for arm in ('pk2', 'pk3'):
    s = pf[pf['arm'] == arm]
    print(f'\n== {arm}: per (star,frame) classes, nrcb frames (322 stars x 4) + nrca (5 x 4)')
    for c in range(1, 6):
        print(names[c], np.sum(s['cls'] == c))
    print('-- per-star best class over frames')
    b = ps[f'{arm}_best']
    for c in range(0, 6):
        print(c, np.sum(b == c))
    print('-- merged <0.08:', ps[f'{arm}_mhit'].sum(), 'replaced among them', np.sum(ps[f'{arm}_mhit'] & ps[f'{arm}_mrep']))
    print('-- merged hit by best class:')
    for c in range(1, 6):
        m = b == c
        print(c, m.sum(), 'merged hit', np.sum(m & ps[f'{arm}_mhit']))
    # sat flux dm
    m1 = s['cls'] == 1
    dm = -2.5 * np.log10(s['sat_flux'][m1]) + ZP - ps['ref_mag'][s['idx'][m1]]
    print('class1 dm: n', m1.sum(), 'pctl 5,16,50,84,95', np.nanpercentile(dm, [5, 16, 50, 84, 95]).round(2), 'neg flux', np.sum(s['sat_flux'][m1] <= 0), '|dm|<0.3', np.sum(np.abs(dm) < 0.3))
    m2 = s['cls'] == 2
    print('class2 sep pctl 5,25,50,75,95', np.percentile(s['sat_sep'][m2], [5, 25, 50, 75, 95]).round(3) if m2.any() else '')
    m3 = s['cls'] == 3
    if m3.any():
        print('class3 reasons', np.unique(s['rej_reason'][m3], return_counts=True), 'sep pctl', np.percentile(s['rej_sep'][m3], [5, 50, 95]).round(3))
        print('class3 rej flux/dolphot flux pctl', np.nanpercentile(s['rej_flux'][m3] / ps['dol_flux'][s['idx'][m3]], [5, 25, 50, 75, 95]).round(3))
