"""For moved vs control faint stars: change in m8 flux and in the model-subtracted background terms, A -> B.  usage: bkg_check.py A B"""
import sys
import numpy as np
from astropy.table import Table
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/faint_sw')
from common import *
A_, B_ = sys.argv[1:3]
an.ZPWIN.update(an.zp_windows())
A, B = an.Arm(A_), an.Arm(B_)
S = Table.read(f'{FS}/stars_{A_}_{B_}.ecsv')
L = ['## m8 flux and background terms, A -> B (medians over stars)',
     '| band | group | N | flux A | flux B | median (fB-fA)/fA | (fB-fA)/flux_err_A | d mean_modelsub_bkg (B-A) | d modelsub_bkg / fluxA | d local_bkg (B-A) | d forced_refit_frac | frac forced_filled A | frac forced_filled B | flux_err_prop A / B | flux A < 0 frac |', '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
def g(arm, col, ix):
    return np.asarray(an.fl(np.ma.asarray(arm.cat[col]).astype(float)))[ix]
for b in BANDS4:
    lo = 'f' + b.lower()
    for nm, flag in (('moved', True), ('control', False)):
        s = S[(S['band'] == b) & (S['moved'] == flag)]
        ids = np.asarray(s['i'])
        ia, ib = A.idx[ids], B.idx[ids]
        fA, fB = g(A, f'flux_{lo}', ia), g(B, f'flux_{lo}', ib)
        eA = g(A, f'flux_err_{lo}', ia)
        mA, mB = g(A, f'mean_modelsub_bkg_{lo}', ia), g(B, f'mean_modelsub_bkg_{lo}', ib)
        lA, lB = g(A, f'local_bkg_{lo}', ia), g(B, f'local_bkg_{lo}', ib)
        rA, rB = g(A, f'forced_refit_frac_{lo}', ia), g(B, f'forced_refit_frac_{lo}', ib)
        ffA, ffB = g(A, f'forced_filled_{lo}', ia), g(B, f'forced_filled_{lo}', ib)
        pA, pB = g(A, f'flux_err_prop_{lo}', ia), g(B, f'flux_err_prop_{lo}', ib)
        md = lambda x: np.nanmedian(x)
        L.append(f'| F{b} | {nm} | {len(s)} | {md(fA):.3g} | {md(fB):.3g} | {md((fB - fA) / fA):+.3f} | {md((fB - fA) / eA):+.2f} | {md(mB - mA):+.3g} | {md((mB - mA) / fA):+.3f} | {md(lB - lA):+.3g} | {md(rB - rA):+.3f} | {np.nanmean(ffA):.3f} | {np.nanmean(ffB):.3f} | {md(pA):.3g} / {md(pB):.3g} | {np.nanmean(fA < 0):.3f} |')
open(f'{FS}/bkg_check_{A_}_{B_}.md', 'w').write('\n'.join(L))
print('\n'.join(L))
