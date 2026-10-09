"""Per-detector median offsets after the grid fix against the PHOTMJSR change.

ph = -2.5 log10(PHOTMJSR_ours / PHOTMJSR_1298) per (band, detector), from
flatver/flatver_jwst_1298_dets.json.  If ours and dolphot both convert with the
PHOTMJSR of their own context, ours - dolphot carries +ph per detector.
Medians are of dm + dfix - pred (closure.py, 5x5 box) minus the band median.
"""
import json
import sys

import numpy as np
from astropy.table import Table

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
from analyze import mad  # noqa: E402

D = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/gridfix'
J = json.load(open('/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/flatver/flatver_jwst_1298_dets.json'))
BANDS = ['200W', '212N', '182M', '150W', '300M', '250M', '323N', '410M']

rows, allm, allp = [], [], []
for b in BANDS:
    S = Table.read(f'{D}/stars_{b}.ecsv')
    dm, pr, df = (np.asarray(S[c]) for c in ('dm', 'pred', 'dfix5'))
    v = dm + df - pr
    ok = np.isfinite(v)
    band_med = np.median(v[ok])
    meds, phs = [], []
    for det in np.unique(S['det'][ok]):
        m = ok & (np.asarray(S['det']) == det)
        if m.sum() < 50:
            continue
        key = f'{b}|{det.upper()}'
        if key not in J:
            continue
        ph = -2.5 * np.log10(J[key]['photmjsr_ours'] / J[key]['photmjsr_old'])
        meds.append(np.median(v[m]) - band_med)
        phs.append(ph)
        rows.append((b, det, m.sum(), meds[-1], ph, mad(v[m])))
    meds, phs = np.array(meds), np.array(phs) - np.mean(phs)
    allm += list(meds); allp += list(phs)
    c = np.corrcoef(meds, phs)[0, 1] if len(meds) > 2 else np.nan
    print(f'F{b}: N det {len(meds)}  rms med {np.std(meds):.4f}  rms(med - ph) {np.std(meds - phs):.4f}  '
          f'rms(med + ph) {np.std(meds + phs):.4f}  corr {c:+.2f}')
allm, allp = np.array(allm), np.array(allp)
print(f'all: corr {np.corrcoef(allm, allp)[0, 1]:+.2f}  slope {np.polyfit(allp, allm, 1)[0]:+.2f}  '
      f'rms med {np.std(allm):.4f}  rms(med - ph) {np.std(allm - allp):.4f}')
print('\n| band | det | N | median (rel. band) | ph (rel. 1298) | rstd |')
print('|---|---|---|---|---|---|')
for r in rows:
    print(f'| F{r[0]} | {r[1]} | {r[2]} | {r[3]:+.4f} | {r[4]:+.4f} | {r[5]:.4f} |')
