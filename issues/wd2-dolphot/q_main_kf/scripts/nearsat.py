"""Lost stars with no per-frame B rows: nearest m7 satstar row in each arm (same band), and whether
that satstar row exists only in B (KEEP_FINITE on)."""
import glob, os
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
D = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark'
M8 = 'catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits'
tr = Table.read(f'{Q}/kf_lost/trace_mainfcbg_mainfcbgkf.ecsv')
ma = Table.read(f'{D}/matched_Q_mainfcbg.fits')
ca = Table.read(f'{Q}/tree_mainfcbg/{M8}')
ia = np.asarray(ma['our_idx'])
cache = {}
def sat(arm, band):
    if (arm, band) not in cache:
        out = []
        for f in sorted(glob.glob(f'{Q}/tree_{arm}/{band.upper()}/pipeline/*_resbgsub_m7_satstar_catalog.fits')):
            t = Table.read(f)
            if len(t):
                out.append(SkyCoord(t['skycoord_fit']))
        cache[(arm, band)] = SkyCoord(np.concatenate([c.ra.deg for c in out]) * u.deg,
                                      np.concatenate([c.dec.deg for c in out]) * u.deg)
    return cache[(arm, band)]
sel = tr[tr['nframesB'] == 0]
n_only = 0
rows = []
for r in sel:
    b = str(r['band'])
    p = SkyCoord(ca['skycoord_ref'][ia[r['i']]])
    sa, sb = sat('mainfcbg', b), sat('mainfcbgkf', b)
    da = p.separation(sa).arcsec.min()
    db = p.separation(sb).arcsec.min()
    # is the nearest B satstar present in A (within 0.1")?
    jb = np.argmin(p.separation(sb).arcsec)
    inA = sb[jb].separation(sa).arcsec.min() < 0.1
    n_only += (not inA) and db < 3
    rows.append((r['i'], b, da, db, inA))
    print(f'{r["i"]:6d} {b} nearest satstar A {da:6.2f}"  B {db:6.2f}"  nearest-B satstar also in A: {inA}')
t = Table(rows=rows, names=['i', 'band', 'dsatA', 'dsatB', 'Bsat_inA'])
print(f'\n{len(t)} lost stars with no B per-frame rows; nearest B satstar < 3" and absent from A: {n_only}; '
      f'median nearest satstar A {np.median(t["dsatA"]):.2f}" B {np.median(t["dsatB"]):.2f}"')
print('bands:', dict(zip(*np.unique(t['band'], return_counts=True))))
t.write(f'{Q}/kf_lost/nearsat.ecsv', overwrite=True)
