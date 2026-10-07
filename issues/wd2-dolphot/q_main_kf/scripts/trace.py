"""For dolphot stars matched only in arm A: count per-frame m7 daophot rows within 0.1" in A and B
(detection band of the A row), and the nearest satstar row (m7) distance and flux ratio B/A.
Tells whether B loses the star at per-frame detection or later at merge."""
import glob, os, sys
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
D = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark'
Q = f'{D}/Q_integ'
M8 = 'catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits'
A, B = sys.argv[1:3]
ma = Table.read(f'{D}/matched_Q_{A}.fits')
ca = Table.read(f'{Q}/tree_{A}/{M8}')
L = np.load(f'{Q}/kf_lost/lost_{A}_{B}.npy')
ia = np.asarray(ma['our_idx'])
rows = ca[ia[L]]
pos = SkyCoord(rows['skycoord_ref'])
fl = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in rows['skycoord_ref_filtername']])
cache = {}
def cats(arm, band):
    k = (arm, band)
    if k not in cache:
        out = []
        for f in sorted(glob.glob(f'{Q}/tree_{arm}/{band.upper()}/{band}_*_resbgsub_m7_daophot_basic.fits')):
            t = Table.read(f)
            out.append((os.path.basename(f), SkyCoord(t['skycoord_centroid']), t))
        cache[k] = out
    return cache[k]
res = []
for j in range(len(L)):
    b = fl[j]
    na = nb = 0
    fa = []
    fb = []
    for (fn, sa, ta), (fnb, sb, tb) in zip(cats(A, b), cats(B, b)):
        assert fn == fnb
        da = pos[j].separation(sa)
        db = pos[j].separation(sb)
        if (da < 0.1 * u.arcsec).any():
            na += 1; fa.append(float(ta['flux_fit'][np.argmin(da)]))
        if (db < 0.1 * u.arcsec).any():
            nb += 1; fb.append(float(tb['flux_fit'][np.argmin(db)]))
    res.append((L[j], b, na, nb, np.median(fa) if fa else np.nan, np.median(fb) if fb else np.nan,
                int(rows['nmatch_' + b][j]) if ('nmatch_' + b) in rows.colnames else -1))
    print(f'{L[j]:6d} {b} {pos[j].ra.deg:.6f} {pos[j].dec.deg:.6f} frames-with-row A {na} B {nb}  '
          f'flux A {res[-1][4]:.4g} B {res[-1][5]:.4g}  m8 nmatch {res[-1][6]}', flush=True)
r = np.array([(x[2], x[3]) for x in res])
print(f'\nlost {len(res)}: B has per-frame rows in >= as many frames as A: {(r[:, 1] >= r[:, 0]).sum()}; '
      f'B has none: {(r[:, 1] == 0).sum()}; A frames median {np.median(r[:, 0])}, B {np.median(r[:, 1])}')
Table(rows=res, names=['i', 'band', 'nframesA', 'nframesB', 'fluxA', 'fluxB', 'nmatchA']).write(
    f'{Q}/kf_lost/trace_{A}_{B}.ecsv', overwrite=True)
