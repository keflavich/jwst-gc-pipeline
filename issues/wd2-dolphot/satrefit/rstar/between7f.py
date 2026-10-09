"""Between-star test: per-star dm (hard cap, bgfree) against the star's median row flat, H+h0 and Hf+h0f, by detector."""
import sys, re
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capbind')
from cb_lib import Band, mad
from scipy.stats import theilslopes
from an4 import REFV
from score7f import load, caps, BINS

for band in sys.argv[1:] or ['150W', '200W', '250M', '300M']:
    B = Band(band)
    col, G = load(B)
    lf = 2.5 * np.log10(col('flat_fitw'))
    sflat = B.med_per_star(lf)
    det_r = np.array([re.search(r'_(nrc\w+?)_align', B.map['files'][k]).group(1) for k in B.frame_of])
    isb3 = B.med_per_star((det_r == 'nrcb3').astype(float))
    D = {}
    for nm in ('H+h0', 'Hf+h0f'):
        a = col('a_' + nm + '+bgfree')
        c = caps(B, G, nm) * B.rcor
        c = np.where(np.isfinite(c), c, np.inf)
        D[nm] = B.dm_of(np.minimum(a, c)) - REFV[band]
    print(f'\n### F{band}: per-star dm against 2.5 log10(star median flat)')
    print('| bin | N | flat sd (mag) | slope H+h0 | slope Hf+h0f | slope H+h0, only-nrcb1 stars | only-nrcb3 stars |')
    print('|---|---|---|---|---|---|---|')
    for lo, hi in BINS[band]:
        s = B.have0 & np.isfinite(D['H+h0']) & np.isfinite(D['Hf+h0f']) & np.isfinite(sflat) & (B.ref >= lo) & (B.ref < hi)
        if s.sum() < 20:
            continue
        sl = [theilslopes(D[k][s], sflat[s])[0] for k in D]
        s1, s3 = s & (isb3 == 0), s & (isb3 == 1)
        f = lambda q: f'{theilslopes(D["H+h0"][q], sflat[q])[0]:+.2f} [{q.sum()}]' if q.sum() >= 15 else '-'
        print(f'| {lo}-{hi} | {s.sum()} | {np.std(sflat[s]):.3f} | {sl[0]:+.2f} | {sl[1]:+.2f} | {f(s1)} | {f(s3)} |')
