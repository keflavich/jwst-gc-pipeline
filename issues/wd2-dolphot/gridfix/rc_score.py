"""Score the recentre_ab.py variants against dolphot.

Per star (closure stars, stars_<band>.ecsv):
  d_X     = -2.5 log10(mean over frames of f_X / f_main2)
  score_X = dm + d_X - pred        (dm = main2 - dolphot - ZP; pred removes the dolphot
                                    area double count, as in closure.md)
Reports the robust std of score_X over stars, per band and per detector, for all stars
and for stars without forced-refit frames, plus the init-to-fit offsets per detector.

usage: python rc_score.py BAND [BAND ...]
"""
import glob
import sys

import numpy as np
from astropy.table import Table, vstack

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402

D = f'{an.Q}/gridfix'
V = ['oi', 'ni', 'of', 'nf']
LAB = {'oi': 'old grid, seed box (main2)', 'ni': 'new grid, seed box',
       'of': 'old grid, recentred box', 'nf': 'new grid, recentred box'}


def mad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


for b in sys.argv[1:]:
    R = vstack([Table.read(f) for f in sorted(glob.glob(f'{D}/recentre/rc_{b}_*.ecsv'))])
    S = Table.read(f'{D}/stars_{b}.ecsv')
    S = S[np.isin(S['i'], R['i'])]
    idx = {k: n for n, k in enumerate(S['i'])}
    row = np.array([idx[k] for k in R['i']])
    sc = {}
    for v in V:
        num = np.bincount(row, weights=R[f'f_{v}'] / R['fmain'], minlength=len(S))
        cnt = np.bincount(row, minlength=len(S))
        sc[v] = np.asarray(S['dm']) + (-2.5 * np.log10(num / cnt)) - np.asarray(S['pred'])
    forced = np.bincount(row, weights=R['forced'].astype(float), minlength=len(S)) > 0
    print(f'F{b}: N stars {len(S)} ({forced.sum()} with forced-refit frames), frame rows {len(R)}')
    print(f'  {"variant":28s} rstd all   rstd no-forced   median')
    for v in V:
        print(f'  {LAB[v]:28s} {mad(sc[v]):.4f}     {mad(sc[v][~forced]):.4f}        {np.nanmedian(sc[v]):+.4f}')
    print(f'  closure dm + dfix5 - pred    {mad(S["dm"] + S["dfix5"] - S["pred"]):.4f}')
    print('  per detector rstd (oi ni of nf), init - fit offset dx dy [px] (main2)')
    for det in sorted(np.unique(R['det'])):
        k = np.asarray(S['i'])[np.unique(row[R['det'] == det])]
        m = np.isin(S['i'], k)
        r = R[R['det'] == det]
        dx = np.nanmedian(r['x_init'] - r['x_fit'])
        dy = np.nanmedian(r['y_init'] - r['y_fit'])
        print(f'  {det}  ' + '  '.join(f'{mad(sc[v][m]):.4f}' for v in V) + f'   {dx:+.3f} {dy:+.3f}   N={m.sum()}')
    for v in ('ni', 'of', 'nf'):
        k = np.isfinite(R['lA']) & np.isfinite(R[f'f_{v}'])
        dd = -2.5 * np.log10(R[f'f_{v}'][k] / R['fmain'][k])
        g = np.abs(dd - np.median(dd)) < 5 * mad(dd)
        print(f'  per-frame slope of -2.5 log10(f_{v}/f_main2) on lA: {np.polyfit(R["lA"][k][g], dd[g], 1)[0]:+.3f}')
