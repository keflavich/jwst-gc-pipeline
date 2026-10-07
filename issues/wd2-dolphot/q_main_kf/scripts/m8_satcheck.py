"""m8 satstar-replaced values per band and arm (PR #1095 review: m7 -> m8 negative / outlier check).
For each arm: catalog rows with replaced_saturated set, how many of them have flux <= 0 or a non-finite
mag; for dolphot-matched replaced values, N, median dm (after band ZP), fraction |dm| > 0.3 and > 1.
usage: python m8_satcheck.py ARM [ARM ...]"""
import sys
import numpy as np
import analyze as an

arms_ = sys.argv[1:]
an.ZPWIN.update(an.zp_windows())
arms = [an.Arm(a) for a in arms_]
print('| band | ' + ' | '.join(f'{A.name}: replaced rows / flux<=0 or NaN mag / matched N / med dm / f>0.3 / f>1' for A in arms) + ' |')
print('|---|' + '---|' * len(arms))
for b in an.BANDS:
    row = f'| F{b} |'
    for A in arms:
        c = b.lower()
        rep = an.fl(A.cat[f'replaced_saturated_f{c}']) == 1 if f'replaced_saturated_f{c}' in A.cat.colnames else np.zeros(len(A.cat), bool)
        fx = an.fl(A.cat[f'flux_f{c}']); mg = an.fl(A.cat[f'mag_vega_f{c}'])
        bad = rep & (~(fx > 0) | ~np.isfinite(mg))
        s = A.matched & A.rep[b] & np.isfinite(A.our[b]) & np.isfinite(A.ref[b])
        d = A.dm(b)[s]
        if len(d):
            row += f' {rep.sum()} / {bad.sum()} / {len(d)} / {np.median(d):+.3f} / {np.mean(np.abs(d) > 0.3):.3f} / {np.mean(np.abs(d) > 1):.3f} |'
        else:
            row += f' {rep.sum()} / {bad.sum()} / 0 / -- / -- / -- |'
    print(row)
