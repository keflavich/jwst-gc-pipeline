"""Row-for-row comparison of the #1021 replay (fix8r) with the #1016 replay (snrp2).

Prints kept counts, the sources added and lost, and whether every addition
lies in the exemption region (qfit <= 0.2, prominence 2-3, S/N_prop =
flux / flux_err_prop >= 40).  Also counts the region under the S/N_prop proxy
flux / flux_err * sqrt(nmatch) that anal_exempt_bins.py and peak_check.py use.

usage: python compare_replay.py <field> <band>   (run where out/ lives)
"""
import sys

import numpy as np
from astropy.table import Table

field, band = sys.argv[1], sys.argv[2].lower()
a = Table.read(f'out/{field}_{band}_snrp2.fits')
c = Table.read(f'out/{field}_{band}_fix8r.fits')
assert np.array_equal(np.asarray(a['rowid']), np.asarray(c['rowid']))
ka, kc = np.asarray(a['kept'], bool), np.asarray(c['kept'], bool)
add, lost = kc & ~ka, ka & ~kc
print(f'{field} {band}: snrp2 kept {ka.sum()}, fix8r kept {kc.sum()}, added {add.sum()}, lost {lost.sum()}')
qf = np.asarray(c['qfit'], float)
pr = np.asarray(c['prominence'], float)
f = np.asarray(c['flux'], float)
snrp = f / np.asarray(c['flux_err_prop'], float)
proxy = f / np.asarray(c['flux_err'], float) * np.sqrt(np.clip(np.asarray(c['nmatch'], float), 1, None))
reg = (qf <= 0.2) & (pr >= 2) & (pr < 3)
if add.sum():
    print(f'  additions: qfit max {qf[add].max():.3f}, prominence {pr[add].min():.2f}-{pr[add].max():.2f}, '
          f'S/N_prop min {snrp[add].min():.1f}; all in region: {bool((reg & (snrp >= 40))[add].all())}')
r = reg & (snrp >= 40)
print(f'  region at S/N_prop >= 40: {r.sum()} ({(r & kc).sum()} kept, {(r & ka).sum()} already kept by #1016)')
print(f'  region at proxy >= 40: {(reg & (proxy >= 40)).sum()} '
      f'(of which S/N_prop >= 40: {(reg & (proxy >= 40) & (snrp >= 40)).sum()})')
g = reg & np.isfinite(proxy / snrp)
if g.sum():
    print('  proxy / S/N_prop in the region: median %.3f, 16-84%% %.3f-%.3f'
          % tuple(np.percentile((proxy / snrp)[g], [50, 16, 84])))
