"""Stage 3: join per-row stats to stars, correlations, tables (printed markdown)."""
import glob, numpy as np
from astropy.table import Table, vstack
from scipy.stats import spearmanr
H = '.'
out = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); out.append(s)
allst = {}
for band in ('250M', '300M'):
    st = Table.read(f'star_{band}.fits')
    s2 = vstack([Table.read(f) for f in sorted(glob.glob(f's2_{band}_*.fits'))])
    s2['sat_frac_g0'] = s2['n_g0sat'] / np.maximum(s2['nsat'], 1)
    s2['capbind'] = s2['a_raw'] / s2['a_cat']
    s2['capH_ratio'] = s2['cap_H'] / s2['a_H']
    s2['rimabove_frac'] = s2['n_rim_above_ceil'] / np.maximum(s2['n_rim_near'], 1)
    s2['core_above_frac'] = s2['n_core_above_ceil'] / np.maximum(s2['nsat'], 1)
    s2['fitsep_pix'] = np.hypot(s2['x_fit'] - s2['x_fit'], 0)  # placeholder
    agg = {}
    for c in ('sat_area', 'nsat', 'sat_frac_g0', 'n_zfdeep', 'n_rim_near', 'rimabove_frac', 'core_above_frac', 'g0max_core', 'capbind', 'capH_ratio', 'nfit', 'nrim_fit', 'qfit', 'amp', 'model_peak', 'data_rimmax'):
        col = np.array(s2[c], dtype=float); ii = np.array(s2['istar']); agg[c] = np.array([np.nanmedian(col[ii == i]) for i in st['istar']])
    for c, v in agg.items():
        st[c] = v
    allst[band] = (st, s2)
    P(f'\n### F{band}: {len(st)} stars, {len(s2)} frame rows\n')
    P('Per-star table (median over the frame rows): ref, dm(final), dm(H+cap), sat_area, nsat (own SATURATED px), g0-sat fraction, rim px near, a_raw/a_cat (cap binding), cap_H/a_H, nfit, nrim_fit, rim px with g0>ceiling')
    P('\n| istar | ref | dm final | dm H+cap | sat_area | nsat | f_g0sat | n_rim | a_raw/a_cat | cap_H/a_H | nfit | nrim_fit | frac rim g0>ceil | qfit | nrows |')
    P('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for r in sorted(st, key=lambda r: -abs(r['dm_H+cap'])):
        P(f"| {r['istar']} | {r['ref']:.3f} | {r['dm_final']:+.3f} | {r['dm_H+cap']:+.3f} | {r['sat_area']:.0f} | {r['nsat']:.0f} | {r['sat_frac_g0']:.2f} | {r['n_rim_near']:.0f} | {r['capbind']:.3f} | {r['capH_ratio']:.3f} | {r['nfit']:.0f} | {r['nrim_fit']:.0f} | {r['rimabove_frac']:.2f} | {r['qfit']:.3f} | {r['nrows']} |")
    P('\nSpearman rho (p) of dm vs property, excluding stars with |dm_H+cap| > 0.5 as outliers noted separately:')
    good = np.abs(st['dm_H+cap']) < 0.5
    P(f'(n = {int(good.sum())} of {len(st)}; outliers removed: {[ (int(i), round(float(d),2)) for i, d in zip(st["istar"][~good], st["dm_H+cap"][~good])]})')
    P('\n| property | rho vs dm_H+cap (p) | rho vs dm_final (p) | rho vs dm_uncapped (p) |')
    P('|---|---|---|---|')
    for c in ('ref', 'sat_area', 'nsat', 'sat_frac_g0', 'n_zfdeep', 'n_rim_near', 'rimabove_frac', 'core_above_frac', 'g0max_core', 'capbind', 'capH_ratio', 'nfit', 'nrim_fit', 'qfit', 'our_qfit', 'ecsv_sep_mas', 'amp'):
        x = np.asarray(st[c], float)
        cells = []
        for d in ('dm_H+cap', 'dm_final', 'dm_uncapped'):
            y = np.asarray(st[d], float); m = good & np.isfinite(x) & np.isfinite(y)
            if m.sum() > 5 and np.ptp(x[m]) > 0:
                r, p = spearmanr(x[m], y[m]); cells.append(f'{r:+.2f} ({p:.2f})')
            else:
                cells.append('-')
        P(f'| {c} | ' + ' | '.join(cells) + ' |')
    P('\nMedians: ' + ', '.join(f'{c}={np.nanmedian(st[c][good]):.3f}' for c in ('dm_final', 'dm_H+cap', 'dm_H+bgfree+cap', 'dm_uncapped')))
    P('Medians of frame-level diagnostics: ' + ', '.join(f'{c}={np.nanmedian(s2[c]):.3g}' for c in ('sat_area', 'nsat', 'sat_frac_g0', 'n_zfdeep', 'n_rim_near', 'capbind', 'rimabove_frac', 'core_above_frac', 'g0max_core', 'ceiling', 'curve_top')))
open('analysis_out.txt', 'w').write('\n'.join(out) + '\n')
import pickle; pickle.dump({k: (v[0], v[1]) for k, v in allst.items()}, open('joined.pkl', 'wb'))
