"""#1150 review item: the F200W nrcb3 within-detector slope of satstar dm against 2.5 log10(flat) is +0.31 with
the flat (Hf) and -0.56 without (H).  The satrefit fits use the cached STPSF grid through the pre-#1154 loader
(transposed), so their dm carries 2 pred - predT (#1153: ours faint by pred - predT, dolphot bright by pred).
This repeats rstar/satflat_det.py (same stars, caps, bin-median removal, Theil-Sen) and adds:
  (1) the correlation of 2 pred - predT (and pred) with 2.5 log10(flat) over the same stars;
  (2) the slope after subtracting 2 pred - predT (the transposed-grid fit) and after subtracting pred only;
  (3) a joint least-squares fit y = s_flat x + s_area (2 pred - predT) + c.
pred / predT per dolphot star from photomver/areapred_main2.npz, psfsum/predT_main2.npz (main2 frames)."""
import os
import re
import sys
import numpy as np
from scipy.stats import theilslopes
from astropy.coordinates import SkyCoord
import astropy.units as u
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sys.path.insert(0, f'{Q}/satrefit/capbind')
sys.path.insert(0, f'{Q}/satrefit/rstar')
sys.path.insert(0, Q)
from cb_lib import Band
from an4 import REFV
import score7f
from score7f import load, caps, BINS
# OUTD=<dir>: score a refit written elsewhere (flatarea/out7g, the fixed-grid run_frames7g.py)
score7f.D = os.environ.get('OUTD', score7f.D)
import analyze as an

P = np.load(f'{Q}/photomver/areapred_main2.npz')
PT = np.load(f'{Q}/psfsum/predT_main2.npz')
M = an.Table.read(an.PATH['main2'][1])
dsky = SkyCoord(np.asarray(M['RA'], float) * u.deg, np.asarray(M['DEC'], float) * u.deg)


def ts(y, x):
    s, _, l1, l2 = theilslopes(y, x)
    return f'{s:+.2f} [{l1:+.2f},{l2:+.2f}]'


for band in sys.argv[1:] or ('150W', '200W'):
    B = Band(band)
    col, G = load(B)
    det_r = np.array([re.search(r'_(nrc\w+?)_align', B.map['files'][k]).group(1) for k in B.frame_of])
    fl = col('flat_rimw')
    D = {}
    for nm in ('H', 'Hf'):
        c = caps(B, G, nm) * B.rcor
        c = np.where(np.isfinite(c), c, np.inf)
        D[nm] = B.dm_of(np.minimum(col('a_' + nm), c)) - REFV[band]
    isb3 = B.med_per_star((det_r == 'nrcb3').astype(float))
    sfl = B.med_per_star(fl)
    ra, dec = B.med_per_star(B.col('ra')), B.med_per_star(B.col('dec'))
    okp = np.isfinite(ra) & np.isfinite(dec)
    jj = np.full(B.n, -1)
    idx, sep, _ = SkyCoord(ra[okp] * u.deg, dec[okp] * u.deg).match_to_catalog_sky(dsky)
    jj[np.flatnonzero(okp)[sep.arcsec < 0.06]] = idx[sep.arcsec < 0.06]
    p = np.where(jj >= 0, P[band][np.maximum(jj, 0)], np.nan)
    pt = np.where(jj >= 0, PT[band][np.maximum(jj, 0)], np.nan)
    tgrid = 2 * p - pt
    print(f'\n### F{band}: {(jj >= 0).sum()} of {B.n} satstars matched to dolphot within 0.06"')
    print('| subset | stars | corr(flat, 2pred-predT) | corr(flat, pred) | slope of 2pred-predT on flat '
          '| H: slope / minus (2pred-predT) / minus pred | Hf: slope / minus (2pred-predT) / minus pred '
          '| Hf joint fit: s_flat, s_area |')
    print('|---|---|---|---|---|---|---|---|')
    lo0 = BINS[band][0][0]
    for nmq, v in (('nrcb1 only', 0), ('nrcb3 only', 1)):
        s = (B.have0 & (isb3 == v) & np.isfinite(sfl) & np.isfinite(D['H']) & np.isfinite(D['Hf'])
             & (B.ref >= lo0) & (B.ref < 17) & np.isfinite(tgrid))
        x = 2.5 * np.log10(sfl[s])
        Y = {}
        for nm in ('H', 'Hf'):
            y = D[nm][s].copy()
            for lo, hi in BINS[band]:
                b = (B.ref[s] >= lo) & (B.ref[s] < hi)
                if b.any():
                    y[b] -= np.median(y[b])
            Y[nm] = y
        tg, pp = tgrid[s] - np.median(tgrid[s]), p[s] - np.median(p[s])
        cells = []
        for nm in ('H', 'Hf'):
            cells.append(f'{ts(Y[nm], x)} / {ts(Y[nm] - tg, x)} / {ts(Y[nm] - pp, x)}')
        X = np.column_stack([x, tg, np.ones_like(x)])
        k = np.ones(len(x), bool)
        for _ in range(3):
            cf = np.linalg.lstsq(X[k], Y['Hf'][k], rcond=None)[0]
            r = Y['Hf'] - X @ cf
            k = np.abs(r) < 3 * 1.4826 * np.median(np.abs(r[k]))
        print(f'| {nmq} | {s.sum()} | {np.corrcoef(x, tg)[0, 1]:+.2f} | {np.corrcoef(x, pp)[0, 1]:+.2f} | '
              f'{ts(tg, x)} | {cells[0]} | {cells[1]} | {cf[0]:+.2f}, {cf[1]:+.2f} |')
