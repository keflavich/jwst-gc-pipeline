"""Task 2: LW wing migration q = cal/(R_h g0) vs distance d to DQ SATURATED, by dolphot magnitude, with unsaturated controls."""
import pickle
import numpy as np
from cb_lib import Band
rng = np.random.default_rng(1)
EDG = ['1', '2', '3', '4-5', '6-8', '9-12', '13-20']
MAGC = [(12.3, 13.0), (13.0, 14.0), (14.0, 15.0), (15.0, 17.0)]
CTLC = [(1500, 4000), (4000, 10000), (10000, 1e9)]
U = []
for band in ('250M', '300M'):
    B = Band(band, s1=False)
    rowref = np.full(B.nrow, np.nan)
    rowref[B.j] = B.ref[B.i]
    offs = np.concatenate([[0], np.cumsum(B.map['lens'])])
    for k in range(4):
        d = pickle.load(open(f's2_{band}_nrcblong_{k+1}.pkl', 'rb'))
        for u in d['units']:
            u['band'] = band
            u['ref'] = rowref[offs[k] + u['idx']] if u['kind'] == 'sat' else np.nan
            u['frame'] = k
            U.append(u)
print('units', len(U))


def stat(sel, key='q'):
    x = np.array([u[key] for u in sel])
    x = x[np.isfinite(x)]
    if len(x) < 5:
        return None
    bs = [np.median(rng.choice(x, len(x))) for _ in range(300)]
    return np.median(x), np.std(bs), len(x)


def table(title, groups, key='q'):
    L = [title, '', '| group | ' + ' | '.join(EDG) + ' |', '|---|' + '---|' * len(EDG)]
    for nm, sel in groups:
        cells = []
        for k in range(len(EDG)):
            s = stat([u for u in sel if u['dbin'] == k], key)
            cells.append(f'{s[0]:.3f}+-{s[1]:.3f} ({s[2]})' if s else '-')
        L.append(f'| {nm} | ' + ' | '.join(cells) + ' |')
    return L + ['']


L = []
for band in ('250M', '300M', 'both'):
    sub = [u for u in U if band == 'both' or u['band'] == band]
    groups = [(f'sat {lo}-{hi} mag', [u for u in sub if u['kind'] == 'sat' and lo <= u['ref'] < hi]) for lo, hi in MAGC]
    groups += [('sat unmatched/other', [u for u in sub if u['kind'] == 'sat' and not (12.3 <= (u['ref'] if np.isfinite(u['ref']) else 0) < 17)])]
    groups += [(f'control peak g0 {lo}-{hi:.0f}' if hi < 1e8 else f'control peak g0 >{lo}', [u for u in sub if u['kind'] == 'ctl' and lo <= u['flux'] < hi]) for lo, hi in CTLC]
    L += table(f'#### F{band}: q_h = cal / (R_header g0), median over units (bootstrap err, N units); d bins in px', groups)
    L += table(f'#### F{band}: q_f = cal / (R_field(g0) g0) (R_field measured >= 25 px from saturation in the same frame)', groups, 'qf')
# matched g0
GE = [(200, 400), (400, 800), (800, 1600), (1600, 3200), (3200, 1e9)]
L += ['#### Matched g0: q_h at fixed pixel-group g0 (median g0 of the unit) for satstars 12.3-14 mag, satstars 14-17 mag and controls, both bands', '',
      '| d bin | g0 bin | sat 12.3-14 | sat 14-17 | control |', '|---|---|---|---|---|']
for k in (0, 1, 2, 3, 4, 5, 6):
    for glo, ghi in GE:
        cells = []
        for sel in ([u for u in U if u['kind'] == 'sat' and 12.3 <= u['ref'] < 14], [u for u in U if u['kind'] == 'sat' and 14 <= u['ref'] < 17], [u for u in U if u['kind'] == 'ctl']):
            s = stat([u for u in sel if u['dbin'] == k and glo <= u['g0'] < ghi])
            cells.append(f'{s[0]:.3f}+-{s[1]:.3f} ({s[2]})' if s else '-')
        if any(c != '-' for c in cells):
            L.append(f'| {EDG[k]} | {glo}-{ghi:.0f} | ' + ' | '.join(cells) + ' |')
open('an2_tables.md', 'w').write('\n'.join(L) + '\n')
pickle.dump(U, open('an2_units.pkl', 'wb'))
print('\n'.join(L))
