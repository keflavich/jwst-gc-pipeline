"""Per-frame comparison of fr0/fr1 m7 catalogs: seeds identical (x_init,y_init)?
For fr1 forced rows: is the same seed present in fr0, forced?, flux ratio fr0/fr1.
usage: python perframe.py BAND"""
import sys, glob, os
import numpy as np
from astropy.table import Table
H = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/forced_refit/ab'
b = sys.argv[1].upper()
tot = dict(fr0_rows=0, fr1_rows=0, fr1_forced=0, fr0_forced=0, f1_present_f0_forced=0, f1_present_f0_notforced=0, f1_absent_f0=0,
           f0_only=0, n_unmatched_seed=0)
ratios = []
absent_examples = []
for f in sorted(glob.glob(f'{H}/tree_fr1/{b}/*_m7_daophot_basic.fits')):
    g = f.replace('tree_fr1', 'tree_fr0')
    if not os.path.exists(g):
        continue
    t1 = Table.read(f); t0 = Table.read(g)
    k0 = {(round(float(x), 2), round(float(y), 2)): i for i, (x, y) in enumerate(zip(t0['x_init'], t0['y_init']))}
    tot['fr0_rows'] += len(t0); tot['fr1_rows'] += len(t1)
    f1 = np.asarray(t1['forced_refit'], bool); f0 = np.asarray(t0['forced_refit'], bool) if 'forced_refit' in t0.colnames else np.zeros(len(t0), bool)
    tot['fr1_forced'] += f1.sum(); tot['fr0_forced'] += f0.sum()
    seen0 = set()
    for i in np.where(f1)[0]:
        k = (round(float(t1['x_init'][i]), 2), round(float(t1['y_init'][i]), 2))
        j = k0.get(k)
        if j is None:
            tot['f1_absent_f0'] += 1
            if len(absent_examples) < 40:
                absent_examples.append((os.path.basename(f), k, float(t1['flux_fit'][i]), float(t1['local_bkg'][i]), float(t1['model_data_peak_ratio'][i])))
        else:
            seen0.add(j)
            if f0[j]:
                tot['f1_present_f0_forced'] += 1
                ratios.append(float(t0['flux_fit'][j]) / float(t1['flux_fit'][i]))
            else:
                tot['f1_present_f0_notforced'] += 1
    k1 = set((round(float(x), 2), round(float(y), 2)) for x, y in zip(t1['x_init'], t1['y_init']))
    tot['f0_only'] += sum(1 for k in k0 if k not in k1)
print(b, tot)
r = np.array(ratios)
print('flux ratio fr0/fr1 for forced in both: median %.2f, 16/84 %.2f %.2f, frac>2 %.2f' % (np.median(r), *np.percentile(r, [16, 84]), np.mean(r > 2)))
for e in absent_examples[:12]:
    print(e)
