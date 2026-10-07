"""Dolphot stars in the saturated-star offset zones (east/west group, north group) that the arm does not match,
against the mirrored control zones.  Union over bands, counted once per dolphot star.
usage: python zone_count.py"""
import sys
sys.path.insert(0, '/blue/adamginsburg/adamginsburg/tmp/claude-3663/s')
import numpy as np
from an import offs, ma

# zone centres: east/west group from the per-band excess peak (beadmap), in "/um; north group in arcsec
EW = {'150W': (0.32, 0.0), '162M': (0.32, 0.0), '164N': (0.32, 0.0), '182M': (-0.28, -0.04), '187N': (-0.32, 0.0),
      '200W': (-0.24, -0.04)}
NORTH = (0.05, 0.69)
R_EW, R_N = 0.08, 0.12          # "/um, arcsec
matched = np.asarray(ma['matched'], bool)
N = len(ma)
zone = {'ew': np.zeros(N, bool), 'ew_mirror': np.zeros(N, bool), 'north': np.zeros(N, bool),
        'north_mirror': np.zeros(N, bool)}
print('| band | east/west zone | mirror | north zone | mirror |')
print('|---|---|---|---|---|')
for b in ['150W', '162M', '164N', '182M', '187N', '200W']:
    lam = int(b[:3]) / 100
    has, si, x, y, A, B = offs(b)
    um = ~matched[has]
    px, py = EW[b]
    e = np.hypot(x / lam - px, y / lam - py) < R_EW
    em = np.hypot(x / lam + px, y / lam + py) < R_EW
    n = np.hypot(x - NORTH[0], y - NORTH[1]) < R_N
    nm = np.hypot(x + NORTH[0], y + NORTH[1]) < R_N
    for k, v in (('ew', e), ('ew_mirror', em), ('north', n), ('north_mirror', nm)):
        zone[k][has[v & um]] = True
    print(f'| F{b} | {(e & um).sum()} | {(em & um).sum()} | {(n & um).sum()} | {(nm & um).sum()} |')
print(f'dolphot stars: {N}; unmatched: {(~matched).sum()}')
for k, v in zone.items():
    print(f'{k}: {v.sum()} distinct unmatched dolphot stars')
both = zone['ew'] | zone['north']
ctrl = zone['ew_mirror'] | zone['north_mirror']
print(f'east/west or north: {both.sum()} (mirror {ctrl.sum()}); excess {both.sum() - ctrl.sum()}')
