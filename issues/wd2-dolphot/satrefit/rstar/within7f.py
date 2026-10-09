"""Within-star test: row amplitude deviation from the star's median row against the row's flat deviation, H+h0 and Hf+h0f."""
import sys
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capbind')
from cb_lib import Band
from scipy.stats import theilslopes
from score7f import load, BINS

for band in sys.argv[1:] or ['150W', '200W', '250M', '300M']:
    B = Band(band)
    col, G = load(B)
    fl = col('flat_fitw')
    print(f'\n### F{band}: slope of (row - star median) of -2.5 log10(a) against (row - star median) of 2.5 log10(flat); uncapped bgfree')
    print('| bin | rows | H+h0 | Hf+h0f | rms row dev H+h0 | Hf+h0f |')
    print('|---|---|---|---|---|---|')
    T = {}
    for nm in ('H+h0', 'Hf+h0f'):
        a = col('a_' + nm + '+bgfree')
        with np.errstate(invalid='ignore', divide='ignore'):
            T[nm] = -2.5 * np.log10(a)
    lf = 2.5 * np.log10(fl)
    i, j = B.i, B.j
    ok = B.good[j] & np.isfinite(T['H+h0'][j]) & np.isfinite(T['Hf+h0f'][j]) & np.isfinite(lf[j])
    i, j = i[ok], j[ok]
    dev = {}
    for key, v in list(T.items()) + [('flat', lf)]:
        x = v[j]
        med = np.full(B.n, np.nan)
        for s in np.unique(i):
            med[s] = np.median(x[i == s])
        dev[key] = x - med[i]
    nrow = np.bincount(i, minlength=B.n)[i]
    for lo, hi in BINS[band]:
        s = (B.ref[i] >= lo) & (B.ref < hi)[i] & (nrow >= 3)
        if s.sum() < 20:
            continue
        r = [theilslopes(dev[k][s], dev['flat'][s])[0] for k in ('H+h0', 'Hf+h0f')]
        rm = [1.4826 * np.median(np.abs(dev[k][s])) for k in ('H+h0', 'Hf+h0f')]
        print(f'| {lo}-{hi} | {int(s.sum())} | {r[0]:+.2f} | {r[1]:+.2f} | {rm[0]:.3f} | {rm[1]:.3f} |')
