"""Cap-rule variants scored by magnitude bin and by trim depth.

floor:   a_eff = max(min(a, cap), tau a)
gfloor:  the floor applies only when the binding pixel's source DN exceeds f x full well
partial: a_eff = a (min(a, cap) / a)^beta
"""
import sys
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capbind')
from cb_lib import Band, mad
from an4 import bind_info, BINS, REFV

TRIM = [(0.02, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 1.0)]
out = []
for band in ('250M', '300M', '150W', '200W'):
    B = Band(band)
    cap0, sf0 = bind_info(B, 'cutH0')
    a = B.a_H_h0_bgfree
    c = cap0 * B.rcor
    c = np.where(np.isfinite(c), c, np.inf)
    acap = np.minimum(a, c)
    with np.errstate(invalid='ignore', divide='ignore'):
        trim = B.med_per_star(1 - acap / a)
    sfb = np.where(np.isfinite(sf0), sf0, 0.0)
    V = {'cap': acap, 'uncap': a}
    for tau in (0.96,):
        V[f'floor {tau}'] = np.maximum(acap, tau * a)
        for f in (0.4, 0.5, 0.6, 0.7):
            V[f'gfloor {tau} f>{f}'] = np.where(sfb > f, np.maximum(acap, tau * a), acap)
    for tau in (0.90, 0.93):
        for f in (0.5, 0.6):
            V[f'gfloor {tau} f>{f}'] = np.where(sfb > f, np.maximum(acap, tau * a), acap)
    for beta in (0.3, 0.5):
        with np.errstate(invalid='ignore', divide='ignore'):
            V[f'partial {beta}'] = a * (acap / a) ** beta
    D = {k: B.dm_of(v) for k, v in V.items()}
    ref = REFV[band]
    have = B.have0 & np.isfinite(trim)
    for d in D.values():
        have &= np.isfinite(d)
    bins = BINS[band]
    out.append(f'\n#### F{band} ({int(have.sum())} stars, reference {ref:+.3f})\n')
    hdr = '| variant | ' + ' | '.join(f'{lo}-{hi}' for lo, hi in bins) + ' | max dev | ' + ' | '.join(f'trim {lo:.2f}-{hi:.2f}' for lo, hi in TRIM) + ' | med abs(dm-ref), trim>0.02 | frac within 0.05, trim>0.02 |'
    out.append(hdr)
    out.append('|---|' + '---|' * (len(bins) + len(TRIM) + 3))
    tr = have & (trim >= 0.02)
    for k, d in D.items():
        cells, dev = [], []
        for lo, hi in bins:
            s = have & (B.ref >= lo) & (B.ref < hi)
            if s.sum() >= 5:
                m_ = np.median(d[s])
                cells.append(f'{m_:+.3f}')
                dev.append(abs(m_ - ref))
            else:
                cells.append('-')
        tcells = []
        for lo, hi in TRIM:
            s = have & (trim >= lo) & (trim < hi)
            tcells.append(f'{np.median(d[s]):+.3f} (N={int(s.sum())})' if s.sum() >= 3 else '-')
        out.append(f'| {k} | ' + ' | '.join(cells) + f' | {max(dev):.3f} | ' + ' | '.join(tcells)
                   + f' | {np.median(np.abs(d[tr] - ref)):.3f} | {np.mean(np.abs(d[tr] - ref) < 0.05):.2f} |')
print('\n'.join(out))
