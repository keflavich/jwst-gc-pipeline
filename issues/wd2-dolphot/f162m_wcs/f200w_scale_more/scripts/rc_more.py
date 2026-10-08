"""Prototype of a per-detector rotation correction for the F150W-group
distortion references (#1135).

For each band and detector, the rotation r of the band's CRDS distortion
reference relative to F212N's (similarity fit in V2/V3, as in refscale.py)
is removed by composing the detector->v2v3 transform with a rotation by
SIGN*r about the detector centre in V2/V3.  The m6 per-frame catalog
positions are shifted by (corrected WCS - original WCS) at (x_fit, y_fit),
the pairs of datascale_field.py are kept fixed, and the per-detector linear
term J of (band - F212N) is refit.  SIGN = -1 is the expected correction;
SIGN = +1 doubles the term if the sign convention is right.

A second implementation through tweakwcs.JWSTWCSCorrector (tangent plane at
the detector centre) is compared against the direct composition on a grid."""
import glob
import os
import re
import sys
from collections import defaultdict
from copy import deepcopy

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.modeling.models import Rotation2D, Shift
from astropy.table import Table
from stdatamodels.jwst import datamodels
from tweakwcs.correctors import JWSTWCSCorrector
from jwst_gc_pipeline.photometry.astrometry_offsets import local_residual_map, measure_offset

TAG = os.environ.get('TAG', 'wd2')
PAIRS = defaultdict(list)
R = os.environ.get('FIELD_ROOT', '/orange/adamginsburg/jwst/wd2')
STAGE = 'resbgsub_m6'
MATCH = 0.15
MAXEXP = os.environ.get('MAXEXP', '4')
PROG_UNTAGGED = os.environ.get('PROG_UNTAGGED', '0') == '1'
BANDS = sys.argv[1:] or ['f200w']
ANCHOR_CAT = os.environ['ANCHOR_CAT']
PROG = os.environ.get('PROG', '')       # restrict per-frame files to this program tag (e.g. j6778) / crf program number
ROTFILE = '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-drot/jwst_gc_pipeline/reduction/data/distortion_rotations.ecsv'
SIGNS = (0, -1)

import asdf  # noqa: E402
from astropy.io import fits  # noqa: E402

CRDS = '/orange/adamginsburg/jwst/crds/references/jwst/nircam/'
g = np.linspace(100, 1947, 9)
GX, GY = np.meshgrid(g, g)
GXr, GYr = GX.ravel(), GY.ravel()


def dist_model(path):
    ref = fits.getheader(path, 0)['R_DISTOR'].replace('crds://', '')
    with asdf.open(CRDS + os.path.basename(ref), lazy_load=False, memmap=False) as af:
        return af.tree['model']


_rot_tab = Table.read(ROTFILE)


def ref_rotation(band_crf, det):
    """Rotation r (rad) such that SIGN=-1 correction is rotated_wcs(w, -r); r = -correction_arcsec
    from distortion_rotations.ecsv (band CLEAR/F200W rows; other bands: no correction, r=0)."""
    hdr = fits.getheader(band_crf, 0)
    filt = hdr['FILTER']
    if filt != 'F200W':
        return 0.0, 0.0
    ref = os.path.basename(hdr['R_DISTOR'])
    row = _rot_tab[(_rot_tab['detector'] == det.upper()) & (_rot_tab['filter'] == filt)]
    if len(row) != 1 or row['distortion_ref'][0] != ref:
        raise RuntimeError(f'no matching rotation row for {det} {filt} {ref}')
    return -np.deg2rad(row['correction_arcsec'][0] / 3600.0), 0.0


def rotated_wcs(w, angle_rad):
    """Compose detector->v2v3 with a rotation by angle about the detector centre."""
    t = w.get_transform('detector', 'v2v3')
    c2, c3 = t(1023.5, 1023.5)
    rot = (Shift(-c2) & Shift(-c3)) | Rotation2D(np.rad2deg(angle_rad)) | (Shift(c2) & Shift(c3))
    w2 = deepcopy(w)
    w2.set_transform('detector', 'v2v3', t | rot)
    return w2


def corrector_wcs(w, angle_rad):
    """Same rotation through JWSTWCSCorrector with its tangent plane at the detector centre."""
    t = w.get_transform('detector', 'v2v3')
    c2, c3 = t(1023.5, 1023.5)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    corr = JWSTWCSCorrector(deepcopy(w), {'v2_ref': c2, 'v3_ref': c3, 'roll_ref': 0.0})
    corr.set_correction(matrix=[[c, -s], [s, c]])
    return corr.wcs


def sky_delta(w0, w1, x, y):
    r0, d0 = w0(x, y, with_bounding_box=False)
    r1, d1 = w1(x, y, with_bounding_box=False)
    cosd = np.cos(np.deg2rad(d0))
    return (r1 - r0) * cosd * 3.6e6, (d1 - d0) * 3.6e6


def load_anchor():
    t = Table.read(ANCHOR_CAT)
    sc = SkyCoord(t['skycoord_centroid' if 'skycoord_centroid' in t.colnames else 'skycoord'])
    q = np.asarray(t['qfit'], float)
    f = np.asarray(t['flux_fit' if 'flux_fit' in t.colnames else 'flux'], float)
    e = np.asarray(t['flux_err'], float)
    with np.errstate(divide='ignore', invalid='ignore'):
        k = np.isfinite(sc.ra.deg) & np.isfinite(q) & (q <= 0.1) & np.isfinite(f / e) & (f / e >= 20)
    if 'replaced_saturated' in t.colnames:
        k &= ~np.asarray(t['replaced_saturated'], bool)
    return sc[k]


def fitJ(px, py, rx, ry):
    A = np.vstack([np.ones_like(px), px, py]).T
    m = np.ones(len(px), bool)
    for _ in range(5):
        cx, *_ = np.linalg.lstsq(A[m], rx[m], rcond=None)
        cy, *_ = np.linalg.lstsq(A[m], ry[m], rcond=None)
        ex, ey = rx - A @ cx, ry - A @ cy
        s = 1.4826 * np.median(np.abs(np.r_[ex[m], ey[m]]))
        m = (np.abs(ex) < 3 * s) & (np.abs(ey) < 3 * s)
    return np.array([[cx[1], cx[2]], [cy[1], cy[2]]]), int(m.sum())


def m_arcsec(J):
    return np.hypot(J[0, 0] - J[1, 1], J[1, 0] + J[0, 1]) / 62 * 206265


def frame(path, anc, band, det):
    mm = re.search(r'visit(\d+)_vgroup([0-9a-z]+)_exp(\d+)_', os.path.basename(path))
    pr_ = re.search(r'_j(\d+)_', os.path.basename(path))
    prog = pr_.group(1).zfill(5) if pr_ else '*'
    crfs = glob.glob(f'{R}/{band.upper()}/pipeline/jw{prog}*{mm.group(1)}_{mm.group(2)}_{mm.group(3)}_{det}_*_o00?_crf.fits')
    crfs = [c for c in crfs if not c.endswith(('_satstar_catalog.fits',))]
    if not crfs:
        return None
    crf = crfs[0]
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
    gg = measure_offset(sc, anc, sweep=True, context=os.path.basename(path))
    if gg is None or not gg.get('ok') or gg.get('swept') or gg['off'] > MATCH * 1000 / 3:
        return None
    lrm = local_residual_map(sc, anc, gg, cell_arcsec=1e9, match_radius=MATCH * u.arcsec,
                             min_stars=50, tol_mas=np.inf, return_pairs=True,
                             context=os.path.basename(path))
    pr = lrm.get('pairs')
    if not pr or len(pr['ia']) < 50:
        return None
    ia, ib = np.asarray(pr['ia']), np.asarray(pr['ib'])
    cosd = np.cos(np.deg2rad(anc.dec.deg[ib]))
    rx = (sc.ra.deg[ia] - anc.ra.deg[ib]) * cosd * 3.6e6
    ry = (sc.dec.deg[ia] - anc.dec.deg[ib]) * 3.6e6
    xs, ys = x[ia], y[ia]
    r, s = ref_rotation(crf, det)
    with datamodels.open(crf) as dm:
        w = dm.meta.wcs
        out = {}
        for sign in SIGNS:
            if sign == 0:
                dx = dy = 0.0
            else:
                dx, dy = sky_delta(w, rotated_wcs(w, sign * r), xs, ys)
            out[sign] = fitJ(xs - 1023.5, ys - 1023.5, rx + dx, ry + dy)
            if sign == -1:
                PAIRS[(band, det)].append(dict(x=xs - 1023.5, y=ys - 1023.5, rx=rx + dx, ry=ry + dy, ib=ib, name=os.path.basename(path)))
        # implementation check: corrector vs direct composition on a grid, sign -1
        wd = rotated_wcs(w, -r)
        wc = corrector_wcs(w, -r)
        ddx, ddy = sky_delta(wd, wc, GXr, GYr)
        dmax = np.max(np.hypot(ddx, ddy))
        wcp = corrector_wcs(w, +r)
        ddx, ddy = sky_delta(wd, wcp, GXr, GYr)
        dmax_flip = np.max(np.hypot(ddx, ddy))
        cx, cy = sky_delta(w, wd, GXr, GYr)
        corner = np.max(np.hypot(cx, cy))
        # local pixel->sky Jacobian (mas/pix) at the detector centre
        r0, d0 = w(1023.5, 1023.5)
        rx1, dx1 = w(1024.5, 1023.5)
        ry1, dy1 = w(1023.5, 1024.5)
        cosd0 = np.cos(np.deg2rad(d0))
        C = np.array([[(rx1 - r0) * cosd0, (ry1 - r0) * cosd0], [dx1 - d0, dy1 - d0]]) * 3.6e6
    print(f'  {os.path.basename(path)[:44]}  r={np.rad2deg(r) * 3600:+7.2f}"  '
          f'm(0,-,+)=({m_arcsec(out[0][0]):5.1f},{m_arcsec(out[-1][0]):5.1f},{m_arcsec(out[-1][0]):5.1f})  '
          f'max|corr|={corner:.2f} mas  corrector-direct={dmax:.4f} mas (flip {dmax_flip:.2f})', flush=True)
    return r, out, C


anc = load_anchor()
dump = {}
for band in BANDS:
    per = defaultdict(list)
    allf = sorted(f for f in glob.glob(f'{R}/{band.upper()}/{band}_nrc[ab][1-4]_*{STAGE}_daophot_basic.fits') if 'stale' not in f)
    keyed = {}
    for f in allf:
        b = os.path.basename(f)
        mm_ = re.search(r'_(nrc[ab][1-4])_(j\d+_)?visit(\d+)_vgroup([0-9a-z]+)_exp(\d+)_', b)
        if mm_ is None:
            continue
        tag = mm_.group(2)
        if PROG and (tag or '').strip('_') != PROG:
            if not (tag is None and PROG_UNTAGGED):
                continue
        keyed[(mm_.group(1), mm_.group(3), mm_.group(4), int(mm_.group(5)))] = f   # tagged overwrites? sorted order: j-tag after
    sel = [f for k, f in sorted(keyed.items()) if k[3] <= int(MAXEXP)]
    DETS = os.environ.get('DETS')
    for p in sel:
        if DETS and re.search(r'_(nrc[ab][1-4])_', os.path.basename(p)).group(1) not in DETS.split(','):
            continue
        det = re.search(r'_(nrc[ab][1-4])_', os.path.basename(p)).group(1)
        res = frame(p, anc, band, det)
        if res is not None:
            per[det].append(res)
    print(f'\n### {band} - f212n (m6), exposures 1-{MAXEXP}: m (arcsec) of median J per detector')
    print('det    n  ref_rot(")  m_uncorr  m_sign-  m_sign+   (tr-,curl+) uncorr -> sign-  (mas/pix)')
    for det in sorted(per):
        rr = per[det]
        Js = {sg: np.median([o[sg][0] for _, o, _ in rr], axis=0) for sg in SIGNS}
        inv = {sg: (Js[sg][0, 0] - Js[sg][1, 1], Js[sg][1, 0] + Js[sg][0, 1]) for sg in SIGNS}
        C = rr[0][2]
        for sg in SIGNS:
            M = np.linalg.solve(C, Js[sg])
            dump[f'{band}_{det}_{sg}'] = M
        Mu, Mm = np.linalg.solve(C, Js[0]), np.linalg.solve(C, Js[-1])
        print(f'    pixel frame: scale uncorr {(Mu[0,0]+Mu[1,1])/2*1e6:+7.1f} ppm rot {(Mu[1,0]-Mu[0,1])/2*206265:+7.2f}"  '
              f'-> corrected scale {(Mm[0,0]+Mm[1,1])/2*1e6:+7.1f} ppm rot {(Mm[1,0]-Mm[0,1])/2*206265:+7.2f}"', flush=True)
        print(f'{det}  {len(rr)}  {np.rad2deg(rr[0][0]) * 3600:+8.2f}   {m_arcsec(Js[0]):7.1f}  '
              f'{m_arcsec(Js[-1]):7.1f}  {m_arcsec(Js[-1]):7.1f}   '
              f'({inv[0][0]:+.4f},{inv[0][1]:+.4f}) -> ({inv[-1][0]:+.4f},{inv[-1][1]:+.4f})', flush=True)

np.savez(f'rcm_J_{TAG}.npz', **dump)
import pickle
PAIRS['anchor'] = (anc.ra.deg, anc.dec.deg)
pickle.dump(dict(PAIRS), open(f'pairsm_{TAG}.pkl', 'wb'))
