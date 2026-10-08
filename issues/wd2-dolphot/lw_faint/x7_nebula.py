"""dm (ours - dolphot) and NB-MB colour vs local background level (nebular brightness), per band.
Background proxy: our_local_bkg in the band, split into quartiles. Mag window 17-20 (ref) to stay above limits."""
import sys
import numpy as np
from astropy.table import Table

tag = sys.argv[1] if len(sys.argv) > 1 else 'mainfcbg'
t = Table.read(f'/orange/adamginsburg/jwst/wd2/dolphot_benchmark/matched_Q_{tag}.fits')

def arr(c):
    return np.asarray(t[c], float)

def okb(b):
    r, o = arr(f'ref_{b}'), arr(f'our_{b}')
    sat = np.asarray(t[f'our_is_saturated_{b}']).astype(bool) | np.asarray(t[f'our_replaced_saturated_{b}']).astype(bool)
    return np.isfinite(r) & np.isfinite(o) & ~sat & (r < 90) & (o < 90)

print(f'tag={tag}')
for b, lo, hi in (('200W', 18, 21), ('212N', 17, 20), ('277W', 17, 20), ('300M', 17, 20), ('335M', 17, 20),
                  ('410M', 17, 20), ('405N', 17, 19.5), ('466N', 17, 19.5)):
    r, o, lb = arr(f'ref_{b}'), arr(f'our_{b}'), arr(f'our_local_bkg_{b}')
    ok = okb(b) & np.isfinite(lb)
    q = np.nanpercentile(lb[ok], [25, 50, 75])
    s0 = ok & (r >= lo) & (r < hi)
    cells = []
    for k, (a, z) in enumerate(zip([-np.inf, *q], [*q, np.inf])):
        s = s0 & (lb >= a) & (lb < z)
        cells.append(f'Q{k+1}: dm={np.median(o[s]-r[s]):+.3f} (lb~{np.median(lb[s]):.3g}, N={s.sum()})')
    print(f'{b} [{lo},{hi}) ' + ' | '.join(cells))

print('NB-MB colour (ours / dolphot), MB mean mag 17-19.5, by quartile of NB local bkg')
for nb, mb in (('405N', '410M'), ('466N', '410M'), ('187N', '182M'), ('212N', '200W')):
    ok = okb(nb) & okb(mb)
    lb = arr(f'our_local_bkg_{nb}')
    ok &= np.isfinite(lb)
    x = 0.5 * (arr(f'our_{mb}') + arr(f'ref_{mb}'))
    s0 = ok & (x >= 17) & (x < 19.5)
    q = np.nanpercentile(lb[s0], [25, 50, 75])
    cells = []
    for k, (a, z) in enumerate(zip([-np.inf, *q], [*q, np.inf])):
        s = s0 & (lb >= a) & (lb < z)
        co = np.median(arr(f'our_{nb}')[s] - arr(f'our_{mb}')[s])
        cr = np.median(arr(f'ref_{nb}')[s] - arr(f'ref_{mb}')[s])
        cells.append(f'Q{k+1}: {co:+.3f}/{cr:+.3f} (N={s.sum()})')
    print(f'{nb}-{mb} ' + ' | '.join(cells))
