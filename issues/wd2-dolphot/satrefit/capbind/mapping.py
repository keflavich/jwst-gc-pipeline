"""Star <-> frame-row mapping exactly as score7.py (0.1 arcsec match against the main2 Arm).  usage: python mapping.py BAND  -> map_<band>.pkl"""
import glob, os, pickle, sys
import numpy as np
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an
from astropy.table import Table, vstack
from astropy.coordinates import SkyCoord
import astropy.units as u

Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
band = sys.argv[1]
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
ok = A.matched & A.rep[band] & np.isfinite(A.ref[band])
tgt = A.sky[A.idx[ok]]
ref = np.asarray(A.ref[band][ok], float)
dm_final = A.dm(band)[ok]
files, ras, decs, labs, lens = [], [], [], [], []
for fn2 in sorted(glob.glob(f'{Q}/satrefit/out2/{band}_*_satrefit.fits')):
    t6 = Table.read(fn2.replace('/out2/', '/out7/').replace('_satrefit.fits', '_satrefit7.fits'))
    files.append(os.path.basename(fn2))
    ras.append(np.asarray(t6['ra'], float)); decs.append(np.asarray(t6['dec'], float)); labs.append(np.asarray(t6['label']))
    lens.append(len(t6))
ra, dec, lab = np.concatenate(ras), np.concatenate(decs), np.concatenate(labs)
sk = SkyCoord(ra * u.deg, dec * u.deg)
i, j, _, _ = sk.search_around_sky(tgt, 0.1 * u.arcsec)
edge = float(np.percentile(ref, 95))
zpwin_ok = A.matched & ~A.rep[band] & np.isfinite(A.dm(band)) & np.isfinite(A.ref[band])
unsat = zpwin_ok & (A.ref[band] >= edge) & (A.ref[band] < edge + 1)
unsat_dm = float(np.nanmedian(A.dm(band)[unsat]))
pickle.dump(dict(ref=ref, dm_final=dm_final, i=i, j=j, files=files, lens=lens, label=lab, unsat_dm=unsat_dm, edge=edge), open(f'map_{band}.pkl', 'wb'))
print(band, len(ref), len(i), unsat_dm)
