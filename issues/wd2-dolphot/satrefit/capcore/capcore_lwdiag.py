"""LW cap diagnosis from out/rows_<band>.fits: dolphot-implied peak vs recovered peak as a function of first-frame level."""
import numpy as np
from astropy.table import Table
HERE = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit/capcore'
L = []
for band in ('250M', '300M', '150W', '200W'):
    t = Table.read(f'{HERE}/out/rows_{band}.fits')
    g = lambda c: np.asarray(t[c], float)  # noqa: E731
    ok = np.isfinite(g('pk_val')) & np.isfinite(g('pk_ff')) & (g('pk_ff') > 0) & np.isfinite(g('lostf')) & (g('lostf') < 0.2) & (g('pk_g0sat') == 1)
    q = g('pk_val') / (g('f_dol') * g('ppk'))
    x = g('pk_ff') / g('m_ffmax')
    xg = g('pk_g0') / g('m_g0sat99')
    L.append(f'#### F{band}: recovered peak pixel / (dolphot-implied flux x PSF peak), rows whose peak pixel has a flagged group 0 (scaled first-frame value), lost < 0.2; N={ok.sum()}')
    L.append('')
    L.append('| first frame / frame max | N | peak/(f_dol*ppk) median (MAD) | cap/f_dol | core_c/f_dol | precap/f_dol | peak value / f_dol-implied peak vs value/first-frame (MJy/sr per DN) | median ff DN |')
    L.append('|---|---|---|---|---|---|---|---|')
    edges = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    for a, b in zip(edges[:-1], edges[1:]):
        s = ok & (x >= a) & (x < b)
        if s.sum() < 5:
            continue
        mm = lambda v: np.median(v[s & np.isfinite(v)])  # noqa: E731
        mad = 1.4826 * np.median(np.abs(q[s] - np.median(q[s])))
        L.append(f'| {a:.1f}-{b:.1f} | {s.sum()} | {np.median(q[s]):.3f} ({mad:.3f}) | {mm(g("cap") / g("f_dol")):.3f} | {mm(g("a_core_c") / g("f_dol")):.3f} | {mm(g("c_flux_fit_precap") / g("f_dol")):.3f} | {mm(g("pk_val") / g("pk_ff")):.4f} | {mm(g("pk_ff")):.0f} |')
    L.append('')
    s = ok
    L.append(f'median pk_val/pk_ff (MJy/sr per first-frame DN) overall {np.median(g("pk_val")[s] / g("pk_ff")[s]):.4f}; header R {np.median(g("m_rhdr")):.4f}; implied k*R_used: header R x (4+1)/... see text. Frame max first frame {np.median(g("m_ffmax")):.0f} DN.')
    L.append('')
    # rows whose peak is not g0-flagged (g0 path)
    ok2 = np.isfinite(g('pk_val')) & np.isfinite(g('lostf')) & (g('lostf') < 0.2) & (g('pk_g0sat') == 0) & np.isfinite(g('pk_g0'))
    xg2 = g('pk_g0') / g('m_g0sat99')
    L.append(f'Rows whose peak pixel reads group 0 directly (not flagged): N={ok2.sum()}; by g0 / (99th pct of g0 at SATURATED px):')
    L.append('')
    L.append('| g0 / g0sat99 | N | peak/(f_dol*ppk) | median g0 DN |')
    L.append('|---|---|---|---|')
    for a, b in zip([0, .2, .4, .6, .8, 1.0], [.2, .4, .6, .8, 1.0, 9]):
        s = ok2 & (xg2 >= a) & (xg2 < b)
        if s.sum() >= 5:
            L.append(f'| {a:.1f}-{b:.1f} | {s.sum()} | {np.median(q[s]):.3f} | {np.median(g("pk_g0")[s]):.0f} |')
    L.append('')
open(f'{HERE}/capcore_lwdiag.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
