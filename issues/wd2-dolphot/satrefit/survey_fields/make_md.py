import pickle, numpy as np, os
res = pickle.load(open('survey_fields.pkl', 'rb'))
L = ['# Non-wd2 NIRCam survey: measured rim R vs R_header (g0 = 2440 DN)', '',
     'Ratios are R(2440)/R_header from the pipeline measured path; N0 = no distance cut, N25 = edt >= 25. Far-field = median cal/g0/R_header for unsaturated pixels with edt >= 25 (count in parentheses).', '',
     '| field | band | det | R_hdr | N0/H | N25/H | ngood | rebuilt | fallback | far 200-500 | far 500-1000 | far 1000-2000 | far 2000-4000 | H/N0-1 |', '|' + '---|' * 14]
for r in res:
    f = r['far']
    fs = ' | '.join('%.3f (%d)' % (f[k][1], f[k][0]) for k in ('200-500', '500-1000', '1000-2000', '2000-4000'))
    n0 = r[0]['R2440'] / r['Rhdr']; n25 = r[25]['R2440'] / r['Rhdr']
    r['n0'] = n0
    L.append('| %s | %s | %s | %.4f | %.4f | %.4f | %d | %s | %s | %s | %+.1f%% |' % (r['field'], r['band'], r['det'], r['Rhdr'], n0, n25, r[0]['ngood'], r[0]['rebuilt'], r[0]['fallback'], fs, 100 * (1 / n0 - 1)))
L += ['', '## Summary (measured)', '', '| field | channel | n | median N0/H | min | max | median far 2000-4000 | min | max |', '|' + '---|' * 9]
up = dn = 0; flags = []
for field in ('brick', 'sgrb2', 'sgrc'):
    for ch in ('SW', 'LW'):
        s = [r for r in res if r['field'] == field and (r['det'].endswith('long') == (ch == 'LW'))]
        if not s: continue
        a = np.array([r['n0'] for r in s]); b = np.array([r['far']['2000-4000'][1] for r in s])
        L.append('| %s | %s | %d | %.4f | %.4f | %.4f | %.3f | %.3f | %.3f |' % (field, ch, len(s), np.median(a), a.min(), a.max(), np.nanmedian(b), np.nanmin(b), np.nanmax(b)))
for r in res:
    d = 1 / r['n0'] - 1
    if abs(d) > 0.02:
        if d > 0: up += 1
        else: dn += 1
        flags.append('- %s %s %s: header/N0 - 1 = %+.1f%% (header moves rim %s)%s' % (r['field'], r['band'], r['det'], 100 * d, 'UP' if d > 0 else 'DOWN', ' [rebuilt/fallback]' if r[0]['rebuilt'] or r[0]['fallback'] else ''))
L += ['', 'Frames with |R_header/R_N0 - 1| > 2%%: %d up, %d down, of %d.' % (up, dn, len(res)), ''] + flags
nrf = sum(r[0]['rebuilt'] or r[0]['fallback'] for r in res)
L += ['', 'Frames with rebuilt or fallback flag set at N0: %d.' % nrf]
open('survey_fields.md', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L[L.index('## Summary (measured)'):]))
L2 = ['', '## Far-field caveat', '', 'Far-field bin values below 0.5 (or above 1.15 at 200-500 DN) mark frames where cal/g0 departs strongly from R_header (inferred: low-DN bins mixing in sky/background, or a g0 vs cal mismatch). Frames with far 2000-4000 ratio < 0.5 are excluded from the table below.', '',
      '| field | channel | n used | median far 2000-4000 | min | max | list of excluded |', '|' + '---|' * 7]
for field in ('brick', 'sgrb2', 'sgrc'):
    for ch in ('SW', 'LW'):
        s = [r for r in res if r['field'] == field and (r['det'].endswith('long') == (ch == 'LW'))]
        ok = [r for r in s if r['far']['2000-4000'][1] >= 0.5]
        bad = ['%s %s' % (r['band'], r['det']) for r in s if not r['far']['2000-4000'][1] >= 0.5]
        if ok:
            b = np.array([r['far']['2000-4000'][1] for r in ok])
            L2.append('| %s | %s | %d | %.3f | %.3f | %.3f | %s |' % (field, ch, len(ok), np.median(b), b.min(), b.max(), '; '.join(bad)))
open('survey_fields.md', 'a').write('\n'.join(L2) + '\n')
print('\n'.join(L2))
