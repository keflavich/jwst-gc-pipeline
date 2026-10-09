import numpy as np
from astropy.table import Table, vstack
FW = {'277W': 1.48, '250M': 1.33, '300M': 1.58}
PIX = 0.063
out = []
allst = []
for b in ('277W', '250M', '300M'):
    RAD = max(1.0, 0.5 * FW[b])
    s = Table.read(f'star_{b}.ecsv'); f = Table.read(f'sf_{b}.ecsv')
    good = np.asarray(s['hit']) & (np.abs(np.asarray(s['dm'])) < 0.3)
    s['good'] = good
    cat = np.array([''] * len(s), dtype='U40')
    ex = {k: np.zeros(len(s)) for k in ('nfr', 'min_dacc', 'min_drej', 'max_comp', 'any_onsat', 'any_nan', 'min_dallpk', 'min_dcur', 'min_drow')}
    why = np.array([''] * len(s), dtype='U60')
    for i, row in enumerate(s):
        if good[i]:
            cat[i] = 'good'; continue
        g = f[f['dolphot_idx'] == row['dolphot_idx']]
        ex['nfr'][i] = len(g)
        if len(g):
            ex['min_dacc'][i] = g['d_acc'].min(); ex['min_drej'][i] = g['d_rej'].min()
            ex['max_comp'][i] = g['comp'].max(); ex['any_onsat'][i] = g['on_sat'].any()
            ex['any_nan'][i] = (g['pix_nan'] | g['pix_dnu']).any()
            ex['min_dallpk'][i] = g['d_allpk'].min(); ex['min_dcur'][i] = g['d_cur'].min(); ex['min_drow'][i] = g['d_row'].min()
        if row['hit']:
            cat[i] = 'c: row, |dm|>=0.3'
            why[i] = ('faint(dm>0)' if row['dm'] > 0 else 'bright(dm<0)') + (' big' if abs(row['dm']) > 1 else '')
            continue
        if len(g) == 0:
            cat[i] = 'x: no frame covers'; continue
        if (g['d_acc'] <= RAD).any():
            cat[i] = 'd1: accepted satstar within 1 px, no merged row'
        elif (g['d_rej'] <= RAD).any():
            cat[i] = 'd2: rejected satstar within 1 px'
            why[i] = ','.join(sorted(set(g['rej_reason'][g['d_rej'] <= RAD])))
        elif (g['d_row'] <= RAD).any():
            cat[i] = 'b2: per-frame fit row, not in merged'
        elif (g['d_cur'] <= RAD).any():
            cat[i] = 'b1: handed off, no per-frame fit'
            why[i] = 'pix restored' if g['rest_cur'][g['d_cur'] <= RAD].any() else 'pix not restored'
        elif not g['on_sat'].any():
            cat[i] = 'a0: not on SATURATED pixel'
            why[i] = 'sat<=1px' if (g['d_sat'] <= 1).any() else 'sat>1px'
        else:
            gs = g[g['on_sat']]
            if (gs['comp'] < 50).all():
                cat[i] = 'a1: small SAT comp (<50 px), COM miss'
            elif (gs['d_allpk'] <= RAD).any():
                cat[i] = 'a2: peak exists, dropped by 1.5 FWHM satstar exclusion'
            elif (gs['pix_nan'] | gs['pix_dnu']).any():
                cat[i] = 'a3: star pixel NaN/DO_NOT_USE (lost core), no peak'
            else:
                cat[i] = 'a4: big comp, star not a local max (neighbour wing/flat)'
    s['cat'] = cat; s['why'] = why
    for k, v in ex.items(): s[k] = v
    s['band'] = b
    s.write(f'class_{b}.ecsv', overwrite=True)
    allst.append(s)
    lost = s[~good]
    out.append(f'## F{b}: lost {len(lost)} of {len(s)} (RAD {RAD:g} px)')
    out.append('| category | N | ref mag p10/med/p90 | min d to accepted satstar (arcsec) med | stars with F-band mag<16 |')
    out.append('|---|---|---|---|---|')
    for c in sorted(set(lost['cat'])):
        q = lost[lost['cat'] == c]
        mg = np.asarray(q['ref_mag'], float)
        da = np.asarray(q['min_dacc'], float) * PIX
        da = da[da > 0]
        out.append(f'| {c} | {len(q)} | {np.nanpercentile(mg,10):.1f}/{np.nanmedian(mg):.1f}/{np.nanpercentile(mg,90):.1f} | '
                   f'{np.median(da) if len(da) else np.nan:.2f} | {(mg<16).sum()} |')
    for c in sorted(set(lost['cat'])):
        q = lost[lost['cat'] == c]
        w = Table(q)['why']
        from collections import Counter
        cc = Counter([x for x in q['why'] if x])
        if cc: out.append(f'  - {c}: {dict(cc)}')
    out.append('')
open('categories.md', 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
vstack(allst).write('class_all.ecsv', overwrite=True)
