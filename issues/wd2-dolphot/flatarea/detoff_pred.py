"""Satstar nrcb3 - nrcb1 median offset with pred subtracted (as for the unsaturated main2gt per-detector table),
fixed-grid refit (out7g) and transposed (out7f), H and Hf, production background, hard cap.  usage: python detoff_pred.py"""
import re
import sys
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sys.path.insert(0, f'{Q}/satrefit/capbind')
sys.path.insert(0, f'{Q}/satrefit/rstar')
sys.path.insert(0, Q)
from cb_lib import Band, mad
from an4 import REFV
import score7f
import analyze as an

P = np.load(f'{Q}/photomver/areapred_main2.npz')
M = an.Table.read(an.PATH['main2'][1])
dsky = SkyCoord(np.asarray(M['RA'], float) * u.deg, np.asarray(M['DEC'], float) * u.deg)
print('| band | run | var | nrcb1 median dm / dm-pred | nrcb3 median dm / dm-pred | nrcb3 - nrcb1: dm / dm-pred |')
print('|---|---|---|---|---|---|')
for band in ('150W', '200W'):
    for run, d in (('transposed', f'{Q}/satrefit/rstar/out7f'), ('fixed', f'{Q}/flatarea/out7g')):
        score7f.D = d
        B = Band(band)
        col, G = score7f.load(B)
        det_r = np.array([re.search(r'_(nrc\w+?)_align', B.map['files'][k]).group(1) for k in B.frame_of])
        isb3 = B.med_per_star((det_r == 'nrcb3').astype(float))
        ra, dec = B.med_per_star(B.col('ra')), B.med_per_star(B.col('dec'))
        okp = np.isfinite(ra) & np.isfinite(dec)
        jj = np.full(B.n, -1)
        idx, sep, _ = SkyCoord(ra[okp] * u.deg, dec[okp] * u.deg).match_to_catalog_sky(dsky)
        jj[np.flatnonzero(okp)[sep.arcsec < 0.06]] = idx[sep.arcsec < 0.06]
        p = np.where(jj >= 0, P[band][np.maximum(jj, 0)], np.nan)
        lo0, hi0 = score7f.BINS[band][0][0], score7f.BINS[band][-1][1]
        for nm in ('H', 'Hf'):
            c = score7f.caps(B, G, nm) * B.rcor
            c = np.where(np.isfinite(c), c, np.inf)
            dm = B.dm_of(np.minimum(col('a_' + nm), c)) - REFV[band]
            st = B.have0 & np.isfinite(dm) & np.isfinite(p) & (B.ref >= lo0) & (B.ref < hi0)
            q1, q3 = st & (isb3 == 0), st & (isb3 == 1)
            m1, m3 = np.median(dm[q1]), np.median(dm[q3])
            n1, n3 = np.median((dm - p)[q1]), np.median((dm - p)[q3])
            print(f'| F{band} | {run} | {nm} | {m1:+.3f} / {n1:+.3f} ({q1.sum()}) | {m3:+.3f} / {n3:+.3f} ({q3.sum()}) | '
                  f'{m3 - m1:+.3f} / {n3 - n1:+.3f} |')
