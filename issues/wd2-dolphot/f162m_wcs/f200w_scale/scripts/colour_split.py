"""Experiment B: per-detector J refit (SIGN=-1 rotation correction applied in rc_scale.py)
in colour terciles, colour = -2.5 log10(F115W/F212N flux) of the anchor star."""
import glob
import pickle
import sys

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from stdatamodels.jwst import datamodels

R = '/orange/adamginsburg/jwst/wd2'
CAT = R + '/catalogs/{b}_merged_indivexp_merged_resbgsub_m6_dao_basic_vetted.fits'
NB = 200
NMIN = 30
rng = np.random.default_rng(1)
P = pickle.load(open('pairs_wd2.pkl', 'rb'))
ra, dec = P['anchor']

# anchor fluxes with the same selection as rc_scale.load_anchor
t = Table.read(CAT.format(b='f212n'))
q = np.asarray(t['qfit'], float)
f = np.asarray(t['flux'], float)
e = np.asarray(t['flux_err'], float)
sc = SkyCoord(t['skycoord'])
with np.errstate(divide='ignore', invalid='ignore'):
    k = np.isfinite(sc.ra.deg) & np.isfinite(q) & (q <= 0.1) & np.isfinite(f / e) & (f / e >= 20)
k &= ~np.asarray(t['replaced_saturated'], bool)
assert k.sum() == len(ra) and np.allclose(sc.ra.deg[k], ra)
f212 = f[k]
anc = sc[k]
tb = Table.read(CAT.format(b='f115w'))
sb = SkyCoord(tb['skycoord'])
fb = np.asarray(tb['flux'], float)
eb = np.asarray(tb['flux_err'], float)
idx, sep, _ = anc.match_to_catalog_sky(sb)
good = (sep.arcsec < 0.1) & (fb[idx] > 0) & (fb[idx] / eb[idx] > 5) & (f212 > 0)
colour = np.full(len(anc), np.nan)
colour[good] = -2.5 * np.log10(fb[idx][good] / f212[good])
print(f'anchor stars {len(anc)}, with F115W colour {good.sum()}; colour pctl 5/50/95 = {np.nanpercentile(colour, [5, 50, 95]).round(2)}')


def fitJ(px, py, rx, ry):
    A = np.vstack([np.ones_like(px), px, py]).T
    m = np.ones(len(px), bool)
    for _ in range(5):
        cx, *_ = np.linalg.lstsq(A[m], rx[m], rcond=None)
        cy, *_ = np.linalg.lstsq(A[m], ry[m], rcond=None)
        ex, ey = rx - A @ cx, ry - A @ cy
        s = 1.4826 * np.median(np.abs(np.r_[ex[m], ey[m]]))
        m = (np.abs(ex) < 3 * s) & (np.abs(ey) < 3 * s)
    return np.array([[cx[1], cx[2]], [cy[1], cy[2]]])


def Cmat(band, det):
    crf = sorted(glob.glob(f'{R}/{band.upper()}/pipeline/jw*_00001_{det}_*_o00?_crf.fits'))[0]
    with datamodels.open(crf) as dm:
        w = dm.meta.wcs
        r0, d0 = w(1023.5, 1023.5)
        rx1, dx1 = w(1024.5, 1023.5)
        ry1, dy1 = w(1023.5, 1024.5)
    c = np.cos(np.deg2rad(d0))
    return np.array([[(rx1 - r0) * c, (ry1 - r0) * c], [dx1 - d0, dy1 - d0]]) * 3.6e6


def scale_rot(J, C):
    M = np.linalg.solve(C, J)
    return (M[0, 0] + M[1, 1]) / 2 * 1e6, (M[1, 0] - M[0, 1]) / 2 * 206265


out = {}
lines = []
for band in ['f200w', 'f150w']:
    for det in [f'nrc{m}{i}' for m in 'ab' for i in range(1, 5)]:
        fr = P[(band, det)]
        d = {key: np.concatenate([x[key] for x in fr]) for key in ('x', 'y', 'rx', 'ry', 'ib')}
        C = Cmat(band, det)
        col = colour[d['ib']]
        ok = np.isfinite(col)
        d = {key: v[ok] for key, v in d.items()}
        col = col[ok]
        edges = np.quantile(col, [0, 1 / 3, 2 / 3, 1])
        groups = [('all', np.ones(len(col), bool))] + [(f't{j + 1}', (col >= edges[j]) & ((col < edges[j + 1]) if j < 2 else (col <= edges[j + 1]))) for j in range(3)]
        res = {}
        for name, m in groups:
            n = int(m.sum())
            if n < NMIN:
                continue
            args = [d[key][m] for key in ('x', 'y', 'rx', 'ry')]
            s0, r0 = scale_rot(fitJ(*args), C)
            bs = []
            for _ in range(NB):
                ii = rng.integers(0, n, n)
                bs.append(scale_rot(fitJ(*[a[ii] for a in args]), C))
            bs = np.array(bs)
            res[name] = dict(n=n, scale=s0, escale=bs[:, 0].std(), rot=r0, erot=bs[:, 1].std(), medcol=float(np.median(col[m])))
        # slope from bootstrap-consistent fit of tercile scales vs median colour
        if all(k in res for k in ('t1', 't2', 't3')):
            xs = np.array([res[k]['medcol'] for k in ('t1', 't2', 't3')])
            ys = np.array([res[k]['scale'] for k in ('t1', 't2', 't3')])
            ws = 1 / np.array([res[k]['escale'] for k in ('t1', 't2', 't3')]) ** 2
            A = np.vstack([np.ones(3), xs]).T
            cov = np.linalg.inv(A.T @ (A * ws[:, None]))
            sl = (cov @ (A.T @ (ws * ys)))[1]
            res['slope'] = (sl, np.sqrt(cov[1, 1]))
        out[(band, det)] = res
        print(band, det, {k: (round(v['n']), round(v['scale'], 1), round(v['escale'], 1), round(v['medcol'], 2)) if isinstance(v, dict) else v for k, v in res.items()}, flush=True)
pickle.dump(out, open('colour_split.pkl', 'wb'))
