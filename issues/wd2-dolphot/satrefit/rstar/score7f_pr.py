"""PR-relevant subset of score7f: rim rewrite R_header x g0 (H) against R_header x g0 / flat (Hf), no h0, with the
production background (fit {}) and with bgfree; hard recovered-core cap.  Also split by detector for SW.
usage: python score7f_pr.py [BAND ...]"""
import sys, re
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capbind')
from cb_lib import Band, mad
from an4 import REFV
from score7f import load, caps, BINS

for band in sys.argv[1:] or ['150W', '200W', '250M', '300M']:
    B = Band(band)
    col, G = load(B)
    D = {}
    for nm in ('H', 'Hf'):
        c = caps(B, G, nm) * B.rcor
        c = np.where(np.isfinite(c), c, np.inf)
        for bg in ('', '+bgfree'):
            D[nm + bg] = B.dm_of(np.minimum(col('a_' + nm + bg), c)) - REFV[band]
    have = B.have0.copy()
    for d in D.values():
        have &= np.isfinite(d)
    lo0, hi0 = BINS[band][0][0], BINS[band][-1][1]
    print(f'\n### F{band}: median dm - ref / MAD ({int(have.sum())} stars, hard cap)')
    print('| bin | N | ' + ' | '.join(D) + ' |')
    print('|---|---|' + '---|' * len(D))
    for lo, hi in BINS[band] + [(lo0, hi0)]:
        st = have & (B.ref >= lo) & (B.ref < hi)
        if st.sum() < 5:
            continue
        lab = 'all' if (lo, hi) == (lo0, hi0) else f'{lo}-{hi}'
        print(f'| {lab} | {int(st.sum())} | ' + ' | '.join(f'{np.median(d[st]):+.3f} / {mad(d[st]):.3f}' for d in D.values()) + ' |')
    if band in ('150W', '200W'):
        det_r = np.array([re.search(r'_(nrc\w+?)_align', B.map['files'][k]).group(1) for k in B.frame_of])
        isb3 = B.med_per_star((det_r == 'nrcb3').astype(float))
        st = have & (B.ref >= lo0) & (B.ref < hi0)
        print('| detector | N | ' + ' | '.join(D) + ' |')
        print('|---|---|' + '---|' * len(D))
        for nm_, q in (('nrcb1 only', st & (isb3 == 0)), ('nrcb3 only', st & (isb3 == 1))):
            print(f'| {nm_} | {int(q.sum())} | ' + ' | '.join(f'{np.median(d[q]):+.3f} / {mad(d[q]):.3f}' for d in D.values()) + ' |')
