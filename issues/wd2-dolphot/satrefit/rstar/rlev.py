"""rstar: PSF-fit and peak-pixel (R_header g0 / flat) / cal by peak group-0 level, per band and detector."""
import glob, pickle
import numpy as np
EDGES = [0, 0.01, 0.02, 0.03, 0.045, 0.06, 0.08, 0.1, 0.15]
for band in ('150W', '200W', '250M', '300M'):
    dets = ('nrcblong',) if band in ('250M', '300M') else ('nrcb1', 'nrcb3')
    print(f'\n### F{band}: median [N] of PSF fit / peak pixel / flat_w in peak g0 / ceiling bins')
    for det in dets:
        R = []
        for fn in sorted(glob.glob(f'rstar_{band}_{det}_*.pkl')):
            d = pickle.load(open(fn, 'rb'))
            for r in d['rows']:
                r['Rh'] = d['meta']['Rh']; r['ceil'] = d['meta']['ceiling']
            R += d['rows']
        g = lambda k: np.array([r[k] for r in R], float)
        ok = (g('a_cal') > 0) & (g('a_gh') / g('ae_gh') > 30)
        fr = g('g0frac')
        fit = g('a_ghf') / g('a_cal')
        pk = g('Rh') * g('peak_g0') / g('flat_pk') / g('peak_cal')
        cells = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            s = ok & (fr >= lo) & (fr < hi)
            if s.sum() >= 10:
                cells.append(f'{lo:.3f}-{hi:.3f} [{s.sum()}]: {np.median(fit[s]):.3f} / {np.median(pk[s]):.3f} / {np.median(g("flat_w")[s]):.3f}')
        print(f'{det} (ceiling {np.median(g("ceil")):.0f} DN):')
        for c in cells:
            print('   ', c)
