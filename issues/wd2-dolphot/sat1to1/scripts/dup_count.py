"""Per-band exact-duplicate row pairs (within 1 mas) in the m7 vetted per-band tables of an arm tree, and how many
of them are satstar-replaced in both rows.  usage: python dup_count.py ARM"""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import search_around_sky
import astropy.units as u
arm = sys.argv[1]
T = f'/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_{arm}/catalogs'
BANDS = ['f115w', 'f150w', 'f162m', 'f164n', 'f182m', 'f187n', 'f200w', 'f212n', 'f250m', 'f277w', 'f300m',
         'f323n', 'f335m', 'f405n', 'f410m', 'f466n']
print('| band | rows | pairs within 1 mas | both replaced | sep of farther row, median (mas) |')
print('|---|---|---|---|---|')
tot = 0
for f in BANDS:
    fn = f'{T}/{f}_merged_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits'
    t = Table.read(fn)
    c = t['skycoord']
    i1, i2, s, _ = search_around_sky(c, c, 1 * u.mas)
    k = i1 < i2
    i1, i2 = i1[k], i2[k]
    rep = np.ma.filled(t['replaced_saturated'], False).astype(bool)
    both = rep[i1] & rep[i2]
    ms = np.ma.filled(t['satstar_match_sep'], np.nan).astype(float)
    far = np.maximum(ms[i1], ms[i2]) * 1e3 if len(i1) else np.array([np.nan])
    tot += len(i1)
    print(f'| {f.upper()} | {len(t)} | {len(i1)} | {both.sum()} | {np.nanmedian(far):.0f} |')
print(f'total pairs {tot}')
