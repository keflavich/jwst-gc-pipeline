"""Radial data/model ratio for capped (amp=min(a_H,cap_H)) and uncapped (a_H) models, per star-frame, then median over rows."""
import glob, pickle, numpy as np
from astropy.table import Table, vstack
res = {}
edges = [0, 2, 3, 4, 5, 6, 8, 10, 14, 20]
for band in ('250M', '300M'):
    s2 = vstack([Table.read(f) for f in sorted(glob.glob(f's2_{band}_*.fits'))])
    cuts = {}
    for f in sorted(glob.glob(f'cut_{band}_*.pkl')):
        cuts[f] = pickle.load(open(f, 'rb'))
    R = {'cap': [], 'unc': [], 'rimcap': [], 'rimunc': [], 'capped': []}
    for r in s2:
        if not np.isfinite(r['cap_H']):
            continue
        c = cuts[f"cut_{band}_{r['pixfile']}.pkl"][f"{r['istar']}_{r['row']}"]
        d, m = c['data'].astype(float), c['model'].astype(float)
        yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]]
        rr = np.hypot(xx - r['x_fit'], yy - r['y_fit'])
        k = r['a_H'] / r['amp']
        rowc, rowu = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (rr >= lo) & (rr < hi) & (d > 0) & (m > 0)
            if s.sum() >= 3:
                q = np.median(d[s] / m[s])
            else:
                q = np.nan
            rowc.append(q); rowu.append(q / k)
        R['cap'].append(rowc); R['unc'].append(rowu); R['capped'].append(k > 1.001)
    for key in ('cap', 'unc'):
        R[key] = np.array(R[key])
    cp = np.array(R['capped'])
    print(f'\nF{band}: median over {len(cp)} star-frames (of which cap binding: {cp.sum()}) of median(data/model) by radius from fit position')
    print('radius px      : ' + ' | '.join(f'{lo}-{hi}' for lo, hi in zip(edges[:-1], edges[1:])))
    for key, nm in (('cap', 'capped model (amp=min(a_H,cap_H))'), ('unc', 'uncapped model (amp=a_H)')):
        print(f'{nm:38s}: ' + ' | '.join(f'{np.nanmedian(R[key][:, i]):.3f}' for i in range(len(edges) - 1)))
    print('  cap binding rows only, uncapped model: ' + ' | '.join(f'{np.nanmedian(R["unc"][cp][:, i]):.3f}' for i in range(len(edges) - 1)))
