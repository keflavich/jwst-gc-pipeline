"""Validation of the F200W per-detector scale correction (#1137, distortion_scale_correction.py).

Per frame, the WCS is corrected with the shipped modules (rotation from
distortion_rotations.ecsv, scale from distortion_scales.ecsv, both looked up by
R_DISTOR), the m6 per-frame positions are shifted by (corrected - original) at
(x_fit, y_fit), and the per-detector linear term J of (F200W - anchor) is refit.
Variants: 0 = rotation only, -1 = rotation + scale (shipped sign),
+1 = rotation + opposite scale (should double the residual scale).
Derived from f200w_scale_more/rc_more.py."""
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table
from stdatamodels.jwst import datamodels
from jwst_gc_pipeline.photometry.astrometry_offsets import local_residual_map, measure_offset

TAG = os.environ.get('TAG', 'wd2')
R = os.environ.get('FIELD_ROOT', '/orange/adamginsburg/jwst/wd2')
STAGE = 'resbgsub_m6'
MATCH = 0.15
MAXEXP = os.environ.get('MAXEXP', '4')
PROG_UNTAGGED = os.environ.get('PROG_UNTAGGED', '0') == '1'
BANDS = sys.argv[1:] or ['f200w']
ANCHOR_CAT = os.environ['ANCHOR_CAT']
VISIT = os.environ.get('VISIT', '')     # restrict to this visit number (e.g. 001)
PROG = os.environ.get('PROG', '')       # restrict per-frame files to this program tag (e.g. j6778) / crf program number
from jwst_gc_pipeline.reduction import distortion_rotation_correction as drc  # noqa: E402
from jwst_gc_pipeline.reduction import distortion_scale_correction as dsc  # noqa: E402
assert 'jwst-gc-pipeline-dscl' in dsc.__file__, dsc.__file__
SIGNS = (0, -1, +1)

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


def variants(crf, w, det):
    """WCS variants by sign: 0 rotation only, -1 rotation + scale, +1 rotation - scale."""
    h = fits.getheader(crf, 0)
    rot = drc.lookup_correction_arcsec(det, h['FILTER'], h['PUPIL'], h['R_DISTOR'])
    ppm = dsc.lookup_correction_ppm(det, h['FILTER'], h['PUPIL'], h['R_DISTOR'])
    if rot is None or ppm is None:
        raise RuntimeError(f'no table row for {crf}')
    wr, _ = drc.rotated_wcs(w, rot)
    return rot, ppm, {0: wr, -1: dsc.scaled_wcs(wr, ppm)[0], +1: dsc.scaled_wcs(wr, -ppm)[0]}


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
    with datamodels.open(crf) as dm:
        w = dm.meta.wcs
        rot, ppm, ws = variants(crf, w, det)
        out = {}
        for sign in SIGNS:
            dx, dy = sky_delta(w, ws[sign], xs, ys)
            out[sign] = fitJ(xs - 1023.5, ys - 1023.5, rx + dx, ry + dy)
        r0, d0 = w(1023.5, 1023.5)
        rx1, dx1 = w(1024.5, 1023.5)
        ry1, dy1 = w(1023.5, 1024.5)
        cosd0 = np.cos(np.deg2rad(d0))
        C = np.array([[(rx1 - r0) * cosd0, (ry1 - r0) * cosd0], [dx1 - d0, dy1 - d0]]) * 3.6e6
        cx, cy = sky_delta(ws[0], ws[-1], GXr, GYr)
        corner = np.max(np.hypot(cx, cy))
    print(f'  {os.path.basename(path)[:44]}  rot={rot:+7.2f}"  scale={ppm:+6.1f} ppm  '
          f'max|scale shift|={corner:.2f} mas  pairs={out[-1][1]}', flush=True)
    return (rot, ppm), out, C


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
        if VISIT and mm_.group(3) != VISIT:
            continue
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
    print(f'\n### {band} - anchor, exposures 1-{MAXEXP}: pixel-frame scale (ppm) / rotation (") of median J')
    print('det    n  table_ppm   rot-only          rot+scale         rot-scale(flip)')
    for det in sorted(per):
        rr = per[det]
        Js = {sg: np.median([o[sg][0] for _, o, _ in rr], axis=0) for sg in SIGNS}
        C = rr[0][2]
        row = []
        for sg in SIGNS:
            M = np.linalg.solve(C, Js[sg])
            dump[f'{band}_{det}_{sg}'] = M
            row.append(((M[0, 0] + M[1, 1]) / 2 * 1e6, (M[1, 0] - M[0, 1]) / 2 * 206265))
        print(f'{det}  {len(rr):2d}  {rr[0][0][1]:+7.1f}   ' + '   '.join(
            f'{s:+7.1f} {r:+6.2f}"' for s, r in row), flush=True)

np.savez(f'val_J_{TAG}.npz', **dump)

