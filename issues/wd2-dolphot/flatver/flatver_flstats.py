"""Spatial statistics of fl (flat change) and its correlation with pred for LW bands."""
import sys
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
Pa = np.load(f'{an.Q}/photomver/areapred_main2.npz')
F = np.load(f'{an.Q}/flatver/flatver_jwst_1298.npz')
print('| band | std(pred) | std(fl) | pk-pk (1-99%) fl | corr(fl, pred) | std(ph) |')
print('|---|---|---|---|---|---|')
for b in ['250M', '277W', '300M', '335M', '410M', '323N', '405N', '466N', '200W']:
    m = A.matched & np.isfinite(F[f'fl_{b}']) & np.isfinite(Pa[b])
    fl, p = F[f'fl_{b}'][m], Pa[b][m]
    print(f'| F{b} | {np.std(p):.4f} | {np.std(fl):.4f} | {np.percentile(fl, 99) - np.percentile(fl, 1):.4f} | {np.corrcoef(fl, p)[0, 1]:+.2f} | {np.std(F[f"ph_{b}"][m]):.4f} |')
