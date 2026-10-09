"""main2kfpk vs main2kf: quality of band values gained (main2kf NaN, main2kfpk finite) against dolphot,
plus dm of all matched values by dolphot magnitude bin.  Writes gained.md."""
import sys
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an

an.ZPWIN.update(an.zp_windows())
K, P = an.Arm('main2kf'), an.Arm('main2kfpk')
BANDS = ['150W', '200W', '250M', '277W', '300M', '335M', '410M', '323N', '405N', '466N']


def mad(x):
    return 1.4826 * np.median(np.abs(x - np.median(x))) if len(x) else np.nan


L = ['| band | gained N | gained med dm / MAD | gained f(\\|dm\\|<0.3) | gained replaced-sat N | lost N | dolphot mag p10/50/90 of gained |',
     '|---|---|---|---|---|---|---|']
for b in BANDS:
    both = K.matched & P.matched & (K.idx >= 0)
    kd, pd = K.dm(b), P.dm(b)
    g = both & ~np.isfinite(kd) & np.isfinite(pd) & np.isfinite(P.ref[b])
    lo = both & np.isfinite(kd) & ~np.isfinite(pd) & np.isfinite(K.ref[b])
    x = pd[g]
    r = P.ref[b][g]
    L.append(f'| F{b} | {g.sum()} | {np.median(x) if len(x) else np.nan:+.3f} / {mad(x):.3f} | {(np.abs(x) < 0.3).mean() if len(x) else np.nan:.2f} | '
             f'{(P.rep[b] & g).sum()} | {lo.sum()} | ' + ('/'.join(f'{v:.1f}' for v in np.percentile(r, [10, 50, 90])) if len(r) else '-') + ' |')
L += ['', '| band | dolphot bin | main2kf N / med / MAD / f(\\|dm\\|<0.3) | main2kfpk N / med / MAD / f(\\|dm\\|<0.3) |', '|---|---|---|---|']
for b in BANDS:
    for lo_, hi_ in ((0, 15), (15, 17), (17, 19), (19, 21)):
        cells = []
        for A in (K, P):
            s = A.matched & np.isfinite(A.dm(b)) & (A.ref[b] >= lo_) & (A.ref[b] < hi_)
            x = A.dm(b)[s]
            cells.append(f'{s.sum()} / {np.median(x):+.3f} / {mad(x):.3f} / {(np.abs(x) < 0.3).mean():.3f}' if s.sum() else '-')
        L.append(f'| F{b} | {lo_}-{hi_} | ' + ' | '.join(cells) + ' |')
open('/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/kfpkdet/gained.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
