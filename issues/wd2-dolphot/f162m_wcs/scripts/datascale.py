"""Within-detector linear residual field of each band's m6 per-frame catalog
against the F212N m6 merged vetted catalog.

Pairs come from ``local_residual_map(return_pairs=True)`` at one giant cell
after the verified ``measure_offset`` tie (the solver's estimator).  The
per-pair (frame - anchor) residual is fit, per detector, as
    r = c0 + J (p - p_centre),  p = (x_fit, y_fit) in detector pixels,
with iterative 3-sigma clipping.  J (2x2, mas/pix) is converted to a scale
and rotation relative to the 31 mas/pix SW pixel; c0 is the residual at the
detector centre.  Exposures 00001-00004, per-(exposure, module) median of c0
removed as in the solver."""
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from jwst_gc_pipeline.photometry.astrometry_offsets import local_residual_map, measure_offset

R = '/orange/adamginsburg/jwst/wd2'
STAGE = 'resbgsub_m6'
MATCH = 0.15
BANDS = sys.argv[1:] or ['f162m', 'f164n', 'f150w', 'f182m', 'f200w']


def load_anchor():
    t = Table.read(f'{R}/catalogs/f212n_merged_indivexp_merged_{STAGE}_dao_basic_vetted.fits')
    sc = SkyCoord(t['skycoord_centroid' if 'skycoord_centroid' in t.colnames else 'skycoord'])
    q = np.asarray(t['qfit'], float)
    f = np.asarray(t['flux_fit' if 'flux_fit' in t.colnames else 'flux'], float)
    e = np.asarray(t['flux_err'], float)
    with np.errstate(divide='ignore', invalid='ignore'):
        k = np.isfinite(sc.ra.deg) & np.isfinite(q) & (q <= 0.1) & np.isfinite(f / e) & (f / e >= 20)
    if 'replaced_saturated' in t.colnames:
        k &= ~np.asarray(t['replaced_saturated'], bool)
    return sc[k]


def frame_fit(path, anc):
    t = Table.read(path)
    q = np.asarray(t['qfit'], float)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.asarray(t['flux_fit'], float) / np.asarray(t['flux_err'], float)
    k = np.isfinite(q) & (q <= 0.1) & np.isfinite(snr) & (snr >= 20)
    sc = SkyCoord(t['skycoord_centroid'][k])
    x = np.asarray(t['x_fit'], float)[k]
    y = np.asarray(t['y_fit'], float)[k]
    ok = np.isfinite(sc.ra.deg) & np.isfinite(x)
    sc, x, y = sc[ok], x[ok], y[ok]
    if len(sc) < 100:
        return None
    g = measure_offset(sc, anc, sweep=True, context=os.path.basename(path))
    if g is None or not g.get('ok') or g.get('swept') or g['off'] > MATCH * 1000 / 3:
        return None
    lrm = local_residual_map(sc, anc, g, cell_arcsec=1e9, match_radius=MATCH * u.arcsec,
                             min_stars=50, tol_mas=np.inf, return_pairs=True,
                             context=os.path.basename(path))
    pr = lrm.get('pairs')
    if not pr or len(pr['ia']) < 50:
        return None
    ia = np.asarray(pr['ia'])
    ib = np.asarray(pr['ib'])
    # (frame - anchor) per sanctioned pair, computed explicitly from the pair indices
    cosd = np.cos(np.deg2rad(anc.dec.deg[ib]))
    rx = (sc.ra.deg[ia] - anc.ra.deg[ib]) * cosd * 3.6e6
    ry = (sc.dec.deg[ia] - anc.dec.deg[ib]) * 3.6e6
    # solver-style total for the same frame: g + cell, where measure_offset(a, b) is (b - a)
    c = max(lrm['cells'], key=lambda cc: cc['n'])
    solver_style = (g['dra'] + c['dra_mas'], g['ddec'] + c['ddec_mas'])
    print(f"  {os.path.basename(path)[:40]}  median(frame-anchor)=({np.median(rx):+.2f},{np.median(ry):+.2f})  "
          f"solver g+cell=({solver_style[0]:+.2f},{solver_style[1]:+.2f})", flush=True)
    px, py = x[ia] - 1023.5, y[ia] - 1023.5
    A = np.vstack([np.ones_like(px), px, py]).T
    m = np.ones(len(px), bool)
    for _ in range(5):
        cx, *_ = np.linalg.lstsq(A[m], rx[m], rcond=None)
        cy, *_ = np.linalg.lstsq(A[m], ry[m], rcond=None)
        ex, ey = rx - A @ cx, ry - A @ cy
        s = 1.4826 * np.median(np.abs(np.r_[ex[m], ey[m]]))
        m = (np.abs(ex) < 3 * s) & (np.abs(ey) < 3 * s)
    return cx, cy, int(m.sum()), s


anc = load_anchor()
for band in BANDS:
    per = defaultdict(dict)
    for p in sorted(glob.glob(f'{R}/{band.upper()}/{band}_nrc[ab][1-4]_visit001_vgroup*_exp0000[1-4]_{STAGE}_daophot_basic.fits')):
        mm = re.search(r'_(nrc[ab][1-4])_visit001_vgroup\w+_exp(\d+)_', os.path.basename(p))
        r = frame_fit(p, anc)
        if r is not None:
            per[mm.group(2)][mm.group(1)] = r
    print(f'\n### {band} - f212n (m6): per-detector linear fit, exposures {sorted(per)}')
    print('det    n_exp  c0 dRA*  c0 dDec (mas, module-median removed)  scale(ppm)  rot(arcsec)  pairs  sigma(mas)')
    acc = defaultdict(list)
    for e, rows in per.items():
        for mod in ('nrca', 'nrcb'):
            ds = [d for d in rows if d.startswith(mod)]
            if len(ds) < 2:
                continue
            mx = np.median([rows[d][0][0] for d in ds])
            my = np.median([rows[d][1][0] for d in ds])
            for d in ds:
                cx, cy, n, s = rows[d]
                acc[d].append((cx[0] - mx, cy[0] - my, cx[1], cx[2], cy[1], cy[2], n, s))
    for d in sorted(acc):
        a = np.array(acc[d])
        c0x, c0y = np.median(a[:, 0]), np.median(a[:, 1])
        # J in mas/pix: [[dRA/dx, dRA/dy],[dDec/dx, dDec/dy]]; similarity part relative to 31 mas/pix
        J = np.median(a[:, 2:6], axis=0).reshape(2, 2)
        # a similarity J = k*[[cos,-sin],[sin,cos]] composed with the pixel->sky rotation; use invariants
        scale = 0.5 * np.sqrt(abs(np.linalg.det(J + np.eye(2) * 0)))  # placeholder, replaced below
        # divergence-like (isotropic scale) and curl-like (rotation) terms are rotation-invariant
        # once J is expressed in any orthonormal frame with matching parity; sky frame has parity -1
        # relative to pixels, so use trace of J @ P for both parities and report the larger-|.| pair
        tr, cu = J[0, 0] + J[1, 1], J[1, 0] - J[0, 1]
        tr2, cu2 = J[0, 0] - J[1, 1], J[1, 0] + J[0, 1]
        print(f'{d}   {len(a)}   {c0x:+8.2f} {c0y:+8.2f}       '
              f'J=[{J[0,0]:+.4f} {J[0,1]:+.4f}; {J[1,0]:+.4f} {J[1,1]:+.4f}] mas/pix   '
              f'|J|-invariants: (tr {tr:+.4f}, curl {cu:+.4f}) (tr- {tr2:+.4f}, curl+ {cu2:+.4f})  '
              f'{int(np.median(a[:, 6]))}  {np.median(a[:, 7]):.1f}')
