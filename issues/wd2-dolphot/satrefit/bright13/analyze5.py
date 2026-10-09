import glob, numpy as np
from astropy.table import Table, vstack, join
lines = []
def P(*a):
    s = ' '.join(str(x) for x in a); print(s); lines.append(s)
for band in ('250M', '300M'):
    st = Table.read(f'allstar_{band}.fits'); rw = Table.read(f'allrows_{band}.fits')
    s5 = vstack([Table.read(f) for f in sorted(glob.glob(f's5_{band}_*.fits'))])
    t = join(rw['istar', 'pixfile', 'row', 'a_cat', 'a_raw', 'a_H', 'cap_H', 'cap_base'], s5, keys=['istar', 'pixfile', 'row'], join_type='inner', table_names=['r', 's'])
    aH = np.array(t['a_H_r'], float) if 'a_H_r' in t.colnames else np.array(t['a_H'], float)
    capr = np.array(t['cap_H'], float) / aH
    ref = np.array([st['ref'][list(st['istar']).index(i)] for i in t['istar']])
    t['ref'] = ref; t['capr'] = capr; t['acr'] = np.array(t['a_raw'], float) / np.array(t['a_cat'], float)
    t['g0rel'] = np.array(t['g0pk3'], float) / np.array(t['ceiling'], float)
    t['g0satfrac8'] = np.array(t['n_g0sat8'], float) / np.maximum(np.array(t['nsat8'], float), 1)
    P(f'\n### F{band}: per-frame-row medians by dolphot mag (all {len(t)} mapped satstar rows)\n')
    P('| ref bin | N rows | cap_H/a_H | a_raw/a_cat | data/model(a_H) r<2 | 2-3 | 3-4 | 4-5 | g0 peak (r<3) / ceiling | frac sat px (r<=8) with g0 sat | n sat px r<=8 |')
    P('|---|---|---|---|---|---|---|---|---|---|---|')
    for lo, hi in [(12, 13), (13, 13.5), (13.5, 14), (14, 15), (15, 16), (16, 17), (17, 18)]:
        s = (ref >= lo) & (ref < hi)
        if s.sum() < 3: continue
        f = lambda c: np.nanmedian(np.array(t[c], float)[s])
        P(f'| {lo}-{hi} | {s.sum()} | {np.nanmedian(capr[s]):.3f} | {f("acr"):.3f} | {f("q0_2"):.3f} | {f("q2_3"):.3f} | {f("q3_4"):.3f} | {f("q4_5"):.3f} | {f("g0rel"):.2f} | {f("g0satfrac8"):.2f} | {f("nsat8"):.0f} |')
    P(f'\nF{band}: same medians binned by group-0 peak / ceiling (all rows)\n')
    P('| g0pk/ceiling | N | cap_H/a_H | data/model(a_H) r<2 | 3-4 | median ref mag |')
    P('|---|---|---|---|---|---|')
    g = np.array(t['g0rel'], float)
    for lo, hi in [(0, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.05), (1.05, 1.2), (1.2, 3)]:
        s = (g >= lo) & (g < hi)
        if s.sum() < 5: continue
        P(f'| {lo}-{hi} | {s.sum()} | {np.nanmedian(capr[s]):.3f} | {np.nanmedian(np.array(t["q0_2"],float)[s]):.3f} | {np.nanmedian(np.array(t["q3_4"],float)[s]):.3f} | {np.median(ref[s]):.2f} |')
    # peak-valid-g0 (unsaturated g0 pixels within r<3)
    s = (ref >= 12) & (ref < 13)
    P(f'F{band} 12-13 mag rows: with any group-0-saturated pixel within r<=3 of the fit position: {np.mean(np.array(t["g0pk3_sat"] if "g0pk3_sat" in t.colnames else np.zeros(len(t)))[s]) if False else "n/a"}')
    P(f'F{band} 12-13 mag rows: g0 at brightest recovered pixel / ceiling median {np.nanmedian((np.array(t["g0_at_dmax"],float)/np.array(t["ceiling"],float))[s]):.2f}; its data/model(a_H) median {np.nanmedian(np.array(t["q_dmax"],float)[s]):.3f}; r_dmax median {np.nanmedian(np.array(t["r_dmax"],float)[s]):.2f} px')
    s2 = (ref >= 14) & (ref < 15)
    P(f'F{band} 14-15 mag rows: g0 at brightest recovered pixel / ceiling median {np.nanmedian((np.array(t["g0_at_dmax"],float)/np.array(t["ceiling"],float))[s2]):.2f}; its data/model(a_H) median {np.nanmedian(np.array(t["q_dmax"],float)[s2]):.3f}')
    from scipy.stats import spearmanr
    ok = np.isfinite(capr) & np.isfinite(g) & (ref < 17)
    P(f'F{band} Spearman cap_H/a_H vs g0pk/ceiling over all rows: {spearmanr(capr[ok], g[ok])[0]:+.2f}; vs ref mag {spearmanr(capr[ok], ref[ok])[0]:+.2f}')
open('analysis5_out.txt', 'w').write('\n'.join(lines) + '\n')
