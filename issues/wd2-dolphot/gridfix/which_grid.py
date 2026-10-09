"""Which grid orientation did each main2 frame fit with?

For every closure row with a matched per-frame catalogue source (src == 1),
dmain = -2.5 log10(flux_fit / fold5): main2's own per-frame flux against the
forced fit with the transposed (old-loader) grid at the same position.
A frame fitted with the transposed grid gives slope(dmain vs lA) ~ 0; a frame
fitted with the in-memory (correct) grid gives slope ~ -1, like dfix5.
"""
import glob
import os
import sys

import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402

D = f'{an.Q}/gridfix'
BANDS = sys.argv[1:] or ['200W', '212N', '182M', '150W', '300M', '250M', '323N', '410M']
print('| band | frame | det | N | slope dmain | slope dfix5 | rstd dmain |')
print('|---|---|---|---|---|---|---|')
summ = []
for b in BANDS:
    F = Table.read(f'{D}/frames_{b}.ecsv')
    F = F[F['src'] == 1]
    tree_dir = f'{an.Q}/tree_main2/F{b}'
    cats = {}
    for fn in glob.glob(f'{tree_dir}/f{b.lower()}_*_daophot_basic.fits'):
        t = Table.read(fn)
        cats[os.path.basename(t.meta['FILENAME'])] = t
    sl = []
    for fr in np.unique(F['frame']):
        G = F[F['frame'] == fr]
        cat = cats[fr]
        ct = cKDTree(np.c_[np.asarray(cat['x_fit'], float), np.asarray(cat['y_fit'], float)])
        d, j = ct.query(np.c_[G['x'], G['y']])
        fl = np.asarray(cat['flux_fit'], float)[j]
        dmain = -2.5 * np.log10(fl / G['fold5'])
        ok = (d < 1e-3) & np.isfinite(dmain)
        dmain, lA, dfx = dmain[ok], np.asarray(G['lA'])[ok], np.asarray(G['dfix5'])[ok]
        med = np.median(dmain)
        k = np.abs(dmain - med) < 5 * 1.4826 * np.median(np.abs(dmain - med))
        s1 = np.polyfit(lA[k], dmain[k], 1)[0]
        s2 = np.polyfit(lA[k], dfx[k], 1)[0]
        sl.append(s1)
        print(f'| F{b} | {fr[:30]} | {G["det"][0]} | {k.sum()} | {s1:+.2f} | {s2:+.2f} | '
              f'{1.4826 * np.median(np.abs(dmain[k] - np.median(dmain[k]))):.4f} |')
    sl = np.array(sl)
    summ.append(f'F{b}: frames {len(sl)}, slope dmain median {np.median(sl):+.2f}, '
                f'frames with slope < -0.5: {(sl < -0.5).sum()}')
print('\n' + '\n'.join(summ))
