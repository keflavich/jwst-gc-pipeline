"""Restore-rule and hand-off-variant reach per star-frame for the no-row stars.
usage: python gather2.py BAND -> reach_BAND.ecsv"""
import glob, re, sys, warnings, io, contextlib
import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
import astropy.units as u
from scipy import ndimage
from scipy.spatial import cKDTree
warnings.filterwarnings('ignore')
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
sys.path.insert(0, Q)
import analyze as an
REPO = '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-wd2main2kfpk'
sys.path.insert(0, REPO)
from jwst_gc_pipeline.photometry import cataloging as C
assert C.__file__.startswith(REPO)
b = sys.argv[1].upper().lstrip('F')
FWHM = {'277W': 1.48, '250M': 1.33, '300M': 1.58}[b]
RAD = max(1.0, 0.5 * FWHM)
P = f'{Q}/tree_main2kfpk/F{b}/pipeline'
an.ZPWIN.update(an.zp_windows())
A = an.Arm('main2kfpk')
rs = SkyCoord(np.asarray(A.m['RA'], float) * u.deg, np.asarray(A.m['DEC'], float) * u.deg)
noro = np.asarray(Table.read(f'star_{b}.ecsv')['dolphot_idx'])


def restore_r1(dq, sci, bad, hxy, acc_xy, excl_fwhm=1.5, radius_fwhm=3.0):
    """Like _handoff_restore_pixels but components holding an accepted centre are kept;
    instead pixels within excl_fwhm FWHM of an accepted centre are left masked."""
    sat = (dq & 2) != 0
    out = np.zeros(sat.shape, bool)
    if not len(hxy) or not sat.any():
        return out
    lab, _ = ndimage.label(sat)
    ny, nx = sat.shape
    labs = set()
    for xc, yc in zip(np.rint(hxy[:, 0]).astype(int), np.rint(hxy[:, 1]).astype(int)):
        y0, y1, x0, x1 = max(yc - 1, 0), min(yc + 2, ny), max(xc - 1, 0), min(xc + 2, nx)
        if y1 > y0 and x1 > x0:
            labs.update(np.unique(lab[y0:y1, x0:x1]).tolist())
    labs.discard(0)
    if not labs:
        return out
    comp = np.isin(lab, sorted(labs))
    yy, xx = np.nonzero(comp)
    near = C._L._protect_mask(xx, yy, hxy, radius_fwhm * FWHM)
    if len(acc_xy) and excl_fwhm > 0:
        nacc = C._L._protect_mask(xx, yy, acc_xy, excl_fwhm * FWHM)
        near &= ~nacc
    out[yy[near], xx[near]] = True
    valid = np.isfinite(sci) & ~bad & ((dq & 1) == 0)
    return out & valid


rows = []
for f in sorted(glob.glob(f'{P}/jw03523005001_*_nrc?long_align_o005_crf.fits')):
    mm = re.search(r'_(\d{5})_(nrc[ab]long)_', f)
    exp, det = int(mm.group(1)), mm.group(2)
    sci = fits.getdata(f, 'SCI').astype(float)
    dq = fits.getdata(f, 'DQ')
    err = fits.getdata(f, 'ERR').astype(float)
    with np.errstate(all='ignore'):
        bad = ~np.isfinite(1.0 / err) | (sci == 0) | ~np.isfinite(sci) | (err == 0)
    w = WCS(fits.getheader(f, 'SCI'))
    acc = Table.read(f.replace('.fits', '_resbgsub_m7_satstar_catalog.fits'))
    rej = Table.read(f.replace('.fits', '_resbgsub_m7_satstar_rejected.fits'))
    axy = np.column_stack([np.asarray(acc['xcentroid'], float), np.asarray(acc['ycentroid'], float)])
    axy = axy[np.isfinite(axy).all(1)]
    gx = np.empty((0, 2))
    if len(rej):
        rx = np.column_stack([np.asarray(rej['xcentroid'], float), np.asarray(rej['ycentroid'], float)])
        rr = np.asarray(rej['reject_reason']).astype(str)
        gx = rx[(rr == 'implied_peak_gate') & np.isfinite(rx).all(1)]
    sat = (dq & 2) != 0
    lab, n = ndimage.label(sat)
    acc_lab = set(lab[np.rint(axy[:, 1]).astype(int).clip(0, sat.shape[0]-1), np.rint(axy[:, 0]).astype(int).clip(0, sat.shape[1]-1)].tolist()) - {0}
    cfg = {'cur': (50, 1.5), 'a10': (10, 1.5), 'a1': (1, 1.5), 'a0': (0, 1.5),
           'e1.0': (50, 1.0), 'e0.5': (50, 0.5), 'e0': (50, 0.0),
           'a10e0.5': (10, 0.5), 'a1e0.5': (1, 0.5), 'a1e0': (1, 0.0)}
    H = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for k, (area, ex) in cfg.items():
            xy = C._unaccepted_sat_component_xy(dq, acc, FWHM, sci=sci, data_floor=0.0, label='x',
                                                accepted_excl_fwhm=ex, peak_min_area=area)
            xy = np.empty((0, 2)) if xy is None else xy
            H[k] = np.vstack([xy, gx]) if len(gx) else xy
    R = {}
    for k, v in H.items():
        R[k + '|R0'] = C._handoff_restore_pixels(dq, sci, bad, v, acc, FWHM) if len(v) else np.zeros(sat.shape, bool)
        R[k + '|R1'] = restore_r1(dq, sci, bad, v, axy)
    T = {k: (cKDTree(v) if len(v) else None) for k, v in H.items()}
    x, y = w.world_to_pixel(rs[noro])
    ix, iy = np.rint(x).astype(int), np.rint(y).astype(int)
    inside = (ix >= 4) & (ix < sat.shape[1] - 4) & (iy >= 4) & (iy < sat.shape[0] - 4)
    for k in np.where(inside)[0]:
        r = dict(dolphot_idx=int(noro[k]), frame=f'{det}_{exp}', on_sat=bool(sat[iy[k], ix[k]]),
                 comp_has_acc=bool(lab[iy[k], ix[k]] in acc_lab), bad_pix=bool(bad[iy[k], ix[k]]))
        for kk, tr in T.items():
            d = float(tr.query([x[k], y[k]])[0]) if tr else np.inf
            r['h_' + kk] = d <= RAD
            r['d_' + kk] = d
        for kk, m in R.items():
            r['r_' + kk] = bool(m[iy[k], ix[k]])
        rows.append(r)
    print(f, len(rows), flush=True)
Table(rows).write(f'reach_{b}.ecsv', overwrite=True)
