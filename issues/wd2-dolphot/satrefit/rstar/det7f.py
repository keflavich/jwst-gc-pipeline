"""Per-detector per-star medians (stars seen only on nrcb1 or only on nrcb3), H+h0 and Hf+h0f, hard cap, bgfree."""
import sys, re
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capbind')
from cb_lib import Band, mad
from an4 import REFV
from score7f import load, caps, BINS

for band in sys.argv[1:] or ['150W', '200W']:
    B = Band(band)
    col, G = load(B)
    det_r = np.array([re.search(r'_(nrc\w+?)_align', B.map['files'][k]).group(1) for k in B.frame_of])
    isb3 = B.med_per_star((det_r == 'nrcb3').astype(float))
    D = {}
    for nm in ('H+h0', 'Hf+h0f'):
        a = col('a_' + nm + '+bgfree')
        c = caps(B, G, nm) * B.rcor
        c = np.where(np.isfinite(c), c, np.inf)
        D[nm] = B.dm_of(np.minimum(a, c)) - REFV[band]
    print(f'\n### F{band}: median dm - ref / MAD [N] for stars seen on one detector only')
    print('| bin | H+h0 nrcb1 | H+h0 nrcb3 | Hf+h0f nrcb1 | Hf+h0f nrcb3 | b3 - b1 (H) | b3 - b1 (Hf) |')
    print('|---|---|---|---|---|---|---|')
    for lo, hi in BINS[band]:
        s = B.have0 & np.isfinite(D['H+h0']) & np.isfinite(D['Hf+h0f']) & (B.ref >= lo) & (B.ref < hi)
        s1, s3 = s & (isb3 == 0), s & (isb3 == 1)
        if s1.sum() < 8 or s3.sum() < 8:
            continue
        f = lambda d, q: f'{np.median(d[q]):+.3f} / {mad(d[q]):.3f} [{q.sum()}]'
        print(f"| {lo}-{hi} | {f(D['H+h0'], s1)} | {f(D['H+h0'], s3)} | {f(D['Hf+h0f'], s1)} | {f(D['Hf+h0f'], s3)} | "
              f"{np.median(D['H+h0'][s3]) - np.median(D['H+h0'][s1]):+.3f} | {np.median(D['Hf+h0f'][s3]) - np.median(D['Hf+h0f'][s1]):+.3f} |")
