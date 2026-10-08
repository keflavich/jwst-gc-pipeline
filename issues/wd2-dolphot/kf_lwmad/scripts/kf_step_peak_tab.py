import sys
import numpy as np
from astropy.table import Table, join
band = sys.argv[1]
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/kf_lwmad'
p = Table.read(f'{Q}/kf_step_peak_{band}.fits'); s = Table.read(f'{Q}/kf_step_stars_{band}_withref.fits')
dec = lambda c: np.array([q.decode() if isinstance(q, bytes) else str(q) for q in c])
p['fr'] = dec(p['frame']); s['fr'] = dec(s['frame'])
p['key'] = [f'{a}_{x:.3f}_{y:.3f}' for a, x, y in zip(p['fr'], p['x'], p['y'])]
s['key'] = [f'{a}_{x:.3f}_{y:.3f}' for a, x, y in zip(s['fr'], s['x'], s['y'])]
j = join(p, s['key', 'dmag_ref'], keys='key')
m = j['dmag_ref']
print(f'## {band} peak-pixel comparison (component max of fit data); N={len(j)}')
print('| dolphot mag | N | peak on/off | flux on/off | precap on/off | med nuse at off-peak px | frac peak px nuse<=1 |')
print('|---|---|---|---|---|---|---|')
for lo, hi in [(12, 13), (13, 14), (14, 15), (15, 15.5), (15.5, 16), (16, 17)]:
    k = (m >= lo) & (m < hi) & (j['peak_off'] > 0) & np.isfinite(j['peak_on'])
    if not k.any(): continue
    r = j[k]
    print(f"| {lo}-{hi} | {k.sum()} | {np.median(r['peak_on']/r['peak_off']):.3f} | {np.median(r['f_on']/r['f_off']):.3f} | {np.median(r['pre_on']/r['pre_off']):.3f} | {np.median(r['nuse_pk_off']):.0f} | {np.mean(r['nuse_pk_off']<=1):.2f} |")
