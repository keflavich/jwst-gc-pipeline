"""Split the satstar-replaced dm statistics of main2 (off) and main2kf (on) by which arm replaced the star."""
import numpy as np
import analyze as an
an.ZPWIN.update(an.zp_windows())
A, B = an.Arm('main2'), an.Arm('main2kf')
def st(x):
    x = x[np.isfinite(x)]
    return f'{len(x)} / {np.median(x):+.3f} / {an.mad(x):.3f} / {np.mean(np.abs(x) > 0.3):.3f}' if len(x) else '0'
print('| band | group | off: N / med / MAD / f>0.3 | on: N / med / MAD / f>0.3 | median dolphot mag |')
print('|---|---|---|---|---|')
for b in ['250M', '277W', '300M', '323N', '335M', '410M']:
    dA, dB = A.dm(b), B.dm(b)
    ok = A.matched & B.matched
    for name, s in [('replaced in both', ok & A.rep[b] & B.rep[b]), ('replaced on only', ok & ~A.rep[b] & B.rep[b]),
                    ('replaced off only', ok & A.rep[b] & ~B.rep[b])]:
        r = A.ref[b][s & np.isfinite(A.ref[b])]
        print(f'| F{b} | {name} | {st(dA[s])} | {st(dB[s])} | {np.median(r) if len(r) else np.nan:.2f} |')
    for lo, hi in [(10, 13), (13, 14), (14, 15), (15, 16), (16, 18)]:
        s = ok & A.rep[b] & B.rep[b] & (A.ref[b] >= lo) & (A.ref[b] < hi)
        print(f'| F{b} | both, dolphot {lo}-{hi} | {st(dA[s])} | {st(dB[s])} | |')
