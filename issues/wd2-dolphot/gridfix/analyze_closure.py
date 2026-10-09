"""Summarize the grid-loader closure test (closure.py outputs).

Per frame: dfix = -2.5 log10(f_new / f_old) against lA = 2.5 log10(A(x,y)/A(y,x));
the transposition model predicts dfix = -lA.
Per star: robust std (1.4826 MAD) of
  dm                     ours - dolphot - ZP, main2 as run (old loader)
  dm - pred              dolphot area double count removed
  dm + dfix              ours with the new loader
  dm + dfix - pred       new loader, dolphot corrected  (the benchmark score)
  dm - (2 pred - predT)  the model prediction for the previous line
"""
import sys

import numpy as np
from astropy.table import Table

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
from analyze import mad  # noqa: E402

BANDS = ['200W', '212N', '182M', '150W', '300M', '250M', '323N', '410M']
D = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/gridfix'

out = []
out.append('## Per frame: dfix vs lA = 2.5 log10(A(x,y)/A(y,x)) (model: slope -1)\n')
out.append('| band | N frame rows | slope 5x5 | corr 5x5 | rstd(dfix + lA) 5x5 | slope 11x11 | rstd(dfix + lA) 11x11 |')
out.append('|---|---|---|---|---|---|---|')
star_rows, det_rows = [], []
for b in BANDS:
    F = Table.read(f'{D}/frames_{b}.ecsv')
    S = Table.read(f'{D}/stars_{b}.ecsv')
    g = np.isfinite(F['dfix5']) & np.isfinite(F['dfix11']) & np.isfinite(F['lA'])
    s5 = np.polyfit(F['lA'][g], F['dfix5'][g], 1)[0]
    c5 = np.corrcoef(F['lA'][g], F['dfix5'][g])[0, 1]
    s11 = np.polyfit(F['lA'][g], F['dfix11'][g], 1)[0]
    out.append(f'| F{b} | {g.sum()} | {s5:+.3f} | {c5:+.3f} | {mad(F["dfix5"][g] + F["lA"][g]):.4f} '
               f'| {s11:+.3f} | {mad(F["dfix11"][g] + F["lA"][g]):.4f} |')
    dm, pr, pt = np.asarray(S['dm']), np.asarray(S['pred']), np.asarray(S['predT'])
    for box in (5, 11):
        df = np.asarray(S[f'dfix{box}'])
        ok = np.isfinite(dm) & np.isfinite(df) & np.isfinite(pr) & np.isfinite(pt)
        star_rows.append((b, box, ok.sum(), mad(dm[ok]), mad((dm - pr)[ok]), mad((dm + df)[ok]),
                          mad((dm + df - pr)[ok]), mad((dm - 2 * pr + pt)[ok]),
                          np.polyfit((pr - pt)[ok], df[ok], 1)[0]))
        if box == 5:
            for det in np.unique(S['det'][ok]):
                m = ok & (np.asarray(S['det']) == det)
                if m.sum() < 50:
                    continue
                det_rows.append((b, det, m.sum(), mad(dm[m]), mad((dm - pr)[m]), mad((dm + df - pr)[m]),
                                 np.median((dm + df - pr)[m])))

out.append('\n## Per star (main2 stars in the ZP window with >= 1 frame fit)\n')
out.append('| band | box | N | rstd dm | dm - pred | dm + dfix | **dm + dfix - pred** | dm - (2 pred - predT) | slope dfix vs (pred - predT) |')
out.append('|---|---|---|---|---|---|---|---|---|')
for r in star_rows:
    out.append(f'| F{r[0]} | {r[1]} | {r[2]} | {r[3]:.4f} | {r[4]:.4f} | {r[5]:.4f} | **{r[6]:.4f}** | {r[7]:.4f} | {r[8]:+.2f} |')
out.append('\n## Per detector, 5x5 box (median offset of dm + dfix - pred is relative to the band ZP)\n')
out.append('| band | det | N | rstd dm | dm - pred | dm + dfix - pred | median dm + dfix - pred |')
out.append('|---|---|---|---|---|---|---|')
for r in det_rows:
    out.append(f'| F{r[0]} | {r[1]} | {r[2]} | {r[3]:.4f} | {r[4]:.4f} | {r[5]:.4f} | {r[6]:+.4f} |')
open(f'{D}/closure.md', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
