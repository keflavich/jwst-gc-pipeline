"""Check whether a floored cap gives up useful protection: bin stars by cap trim depth and compare dm under cap, floor and no cap."""
import sys
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capbind')
from cb_lib import Band, mad
from an4 import bind_info, REFV

TRIM = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.25), (0.25, 1.0)]
for band in ('250M', '300M', '150W', '200W'):
    B = Band(band)
    cap0, _ = bind_info(B, 'cutH0')
    a = B.a_H_h0_bgfree
    c = cap0 * B.rcor
    c = np.where(np.isfinite(c), c, np.inf)
    acap = np.minimum(a, c)
    with np.errstate(invalid='ignore', divide='ignore'):
        trim_row = 1 - acap / a
    trim = B.med_per_star(trim_row)
    v = {'uncap': B.dm_of(a), 'cap': B.dm_of(acap),
         'floor.96': B.dm_of(np.maximum(acap, 0.96 * a)),
         'floor.90': B.dm_of(np.maximum(acap, 0.90 * a)),
         'floor.85': B.dm_of(np.maximum(acap, 0.85 * a))}
    ref = REFV[band]
    have = B.have0 & np.isfinite(trim)
    for d in v.values():
        have &= np.isfinite(d)
    print(f'\n### F{band}: {int(have.sum())} stars; per-star median trim = 1 - min(a,cap)/a (H+h0+bgfree); entries: median dm (MAD) [frac |dm-ref|<0.05]')
    print('| trim | N | mag range | ' + ' | '.join(v) + ' |')
    print('|---|---|---|' + '---|' * len(v))
    for lo, hi in TRIM:
        s = have & (trim >= lo) & (trim < hi)
        if s.sum() < 3:
            print(f'| {lo:.2f}-{hi:.2f} | {int(s.sum())} | | ' + ' | '.join('-' for _ in v) + ' |')
            continue
        mr = f'{np.percentile(B.ref[s], 10):.1f}-{np.percentile(B.ref[s], 90):.1f}'
        cells = [f'{np.median(d[s]):+.3f} ({mad(d[s]):.3f}) [{np.mean(np.abs(d[s] - ref) < 0.05):.2f}]' for d in v.values()]
        print(f'| {lo:.2f}-{hi:.2f} | {int(s.sum())} | {mr} | ' + ' | '.join(cells) + ' |')
    # deep-trim stars listed individually
    s = have & (trim >= 0.15)
    if s.sum():
        print(f'\nF{band} stars with trim >= 0.15: idx, ref mag, trim, dm cap, dm floor.96, dm uncap')
        for k in np.where(s)[0]:
            print(f'  {k} {B.ref[k]:.2f} {trim[k]:.2f} {v["cap"][k]:+.3f} {v["floor.96"][k]:+.3f} {v["uncap"][k]:+.3f}')
