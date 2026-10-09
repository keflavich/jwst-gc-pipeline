"""Is the F150W star-level residual spatially coherent, or tied to magnitude or crowding?

Per closure star, score = O - D' (new loader m7 replay 'ni', detector medians removed)
for F150W and F200W.  Reports:
  - binned median and rstd of the score against dolphot magnitude and distance to the
    Wd2 core (156.0, -57.757);
  - the correlation between a star's score and the mean score of its k = 8 nearest closure
    neighbours (spatial coherence), and the rstd left after subtracting that neighbour mean;
  - the correlation of the F150W and F200W scores for stars in both (a shared, star-level term).
usage: python spatial.py -> spatial.txt
"""
import glob
import sys

import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.table import Table, vstack
from scipy.spatial import cKDTree

sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ')
import analyze as an  # noqa: E402

Q = an.Q
GF = f'{Q}/gridfix'


def rs(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return 1.4826 * np.median(np.abs(x - np.median(x)))


def demed(x, det):
    out = np.array(x, float)
    for d in np.unique(det):
        m = det == d
        out[m] -= np.nanmedian(out[m])
    return out


an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2')
ra = np.asarray(A.m['RA'], float)
dec = np.asarray(A.m['DEC'], float)
core = SkyCoord(156.0 * u.deg, -57.757 * u.deg)
L = []
scores = {}
for b in ['150W', '200W']:
    S = Table.read(f'{GF}/stars_{b}.ecsv')
    i = np.asarray(S['i'])
    R = vstack([Table.read(f) for f in sorted(glob.glob(f'{GF}/recentre/rc_{b}_*.ecsv'))])
    R = R[np.isin(R['i'], i)]
    idx = {k: n for n, k in enumerate(i)}
    row = np.array([idx[k] for k in R['i']])
    num = np.bincount(row, weights=R['f_ni'] / R['fmain'], minlength=len(S))
    cnt = np.bincount(row, minlength=len(S))
    forced = np.bincount(row, weights=R['forced'].astype(float), minlength=len(S)) > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        d = -2.5 * np.log10(num / cnt)
    sc = np.asarray(S['dm']) + d - np.asarray(S['pred'])
    ok = np.isfinite(sc) & ~forced
    i, sc, det, mag = i[ok], sc[ok], np.asarray(S['det'])[ok], np.asarray(S['ref'])[ok]
    sc = demed(sc, det)
    g = np.abs(sc) < 5 * rs(sc)
    i, sc, mag = i[g], sc[g], mag[g]
    scores[b] = dict(zip(i, sc))
    sk = SkyCoord(ra[i] * u.deg, dec[i] * u.deg)
    rc = sk.separation(core).arcsec
    L.append(f'## F{b}: {len(i)} stars (no forced-refit frames, 5-sigma clipped), rstd {rs(sc):.4f}')
    for name, x, edges in (('dolphot mag', mag, np.nanpercentile(mag, [0, 20, 40, 60, 80, 100])),
                           ('core distance ["]', rc, np.nanpercentile(rc, [0, 20, 40, 60, 80, 100]))):
        cells = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (x >= lo) & (x <= hi)
            cells.append(f'{lo:.1f}-{hi:.1f}: med {np.median(sc[m]):+.4f} rstd {rs(sc[m]):.4f}')
        L.append(f'  vs {name}: ' + ' | '.join(cells))
    xy = np.c_[(ra[i] - 156.0) * np.cos(np.deg2rad(-57.757)) * 3600, (dec[i] + 57.757) * 3600]
    tr = cKDTree(xy)
    dd, jj = tr.query(xy, k=9)
    nbr = sc[jj[:, 1:]].mean(axis=1)
    L.append(f'  neighbour mean (k=8, median sep {np.median(dd[:, 1:]):.1f}"): corr {np.corrcoef(sc, nbr)[0, 1]:+.3f}, '
             f'rstd(score - nbr mean) {rs(sc - nbr):.4f}, rstd(nbr mean) {rs(nbr):.4f}')
    rng = np.random.default_rng(0)
    perm = np.array([np.corrcoef(sc, rng.permutation(sc)[jj[:, 1:]].mean(axis=1))[0, 1] for _ in range(200)])
    L.append(f'  permutation null for corr: mean {perm.mean():+.3f}, 99th pct {np.percentile(perm, 99):+.3f}')
common = sorted(set(scores['150W']) & set(scores['200W']))
a = np.array([scores['150W'][k] for k in common])
c = np.array([scores['200W'][k] for k in common])
L.append(f'## F150W vs F200W score, {len(common)} common stars: corr {np.corrcoef(a, c)[0, 1]:+.3f}, '
         f'rstd F150W {rs(a):.4f}, F200W {rs(c):.4f}, F150W - F200W {rs(a - c):.4f}')
open(f'{Q}/f150x/spatial.txt', 'w').write('\n'.join(L) + '\n')
print('\n'.join(L))
