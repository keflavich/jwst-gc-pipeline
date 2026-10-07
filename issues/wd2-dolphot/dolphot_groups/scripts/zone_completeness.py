"""Completeness with and without the dolphot east/west and north groups around saturated stars.
Parents: the replaced_saturated rows of each band in tree_mainfcbg (one fixed list for every arm).
A dolphot star is in a zone when, in any of the listed bands, it has a dolphot magnitude and lies in the zone of its
nearest parent.  usage: python zone_completeness.py ARM [ARM ...]"""
import sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord

D = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark'
Q = f'{D}/Q_integ'
M8 = 'catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits'
EW = {'150W': (0.32, 0.0), '162M': (0.32, 0.0), '164N': (0.32, 0.0), '182M': (-0.28, -0.04), '187N': (-0.32, 0.0),
      '200W': (-0.24, -0.04)}
NORTH = (0.05, 0.69)
R_EW, R_N = 0.08, 0.12          # "/um, arcsec


def fl(x):
    return np.ma.filled(x, np.nan).astype(float) if np.ma.isMaskedArray(x) else np.asarray(x, float)


par = Table.read(f'{Q}/tree_mainfcbg/{M8}')


def zones(m):
    ref = SkyCoord(m['RA'], m['DEC'], unit='deg')
    z = np.zeros(len(m), bool)
    zm = np.zeros(len(m), bool)
    for b, (px, py) in EW.items():
        lam = int(b[:3]) / 100
        s = np.ma.filled(par[f'replaced_saturated_f{b.lower()}'], False).astype(bool)
        sat = SkyCoord(par['skycoord_ref'][s])
        has = np.where(np.isfinite(fl(m[f'ref_{b}'])))[0]
        i, _, _ = ref[has].match_to_catalog_sky(sat)
        dra, dd = sat[i].spherical_offsets_to(ref[has])
        x, y = dra.arcsec, dd.arcsec
        inz = (np.hypot(x / lam - px, y / lam - py) < R_EW) | (np.hypot(x - NORTH[0], y - NORTH[1]) < R_N)
        inm = (np.hypot(x / lam + px, y / lam + py) < R_EW) | (np.hypot(x + NORTH[0], y + NORTH[1]) < R_N)
        z[has[inz]] = True
        zm[has[inm]] = True
    return z, zm


print('| arm | dolphot stars | matched | completeness | in zones (matched) | mirror zones (matched) '
      '| completeness outside zones |')
print('|---|---|---|---|---|---|---|')
for arm in sys.argv[1:]:
    m = Table.read(f'{D}/matched_Q_{arm}.fits')
    ok = np.asarray(m['matched'], bool)
    z, zm = zones(m)
    out = ~z
    print(f'| {arm} | {len(m)} | {ok.sum()} | {ok.mean():.3f} | {z.sum()} ({(z & ok).sum()}) | '
          f'{zm.sum()} ({(zm & ok).sum()}) | {ok[out].mean():.3f} |')
