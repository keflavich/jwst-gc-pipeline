"""Aperture closure of the pipeline PSF fluxes (flux_fit) on the crf frames.
usage: detclose.py BAND [nproc] [maxstars]
Per frame: select isolated unsaturated stars from the resbgsub_m7 per-frame
catalogue, measure aperture fluxes at r = 2,3,4,6 px with an annulus background,
render the fovp101 PSF grid at the fitted position (flux = flux_fit) and form
EE_model(r) with the same annulus background removed from the model.
"""
import sys, glob, os, warnings
import numpy as np
from multiprocessing import Pool
from astropy.io import fits
from astropy.table import Table, vstack
from astropy.stats import sigma_clipped_stats
from scipy.spatial import cKDTree
from photutils.aperture import CircularAperture
from stpsf.utils import to_griddedpsfmodel

warnings.filterwarnings('ignore')
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/tree_main2'
RADII = [2, 3, 4, 6]
LW_BANDS = ('F250M', 'F277W', 'F300M', 'F335M', 'F410M', 'F323N', 'F405N', 'F466N')


def run_frame(args):
    band, crf, cat, maxstars = args
    lw = band in LW_BANDS
    ann = (8, 12) if lw else (10, 15)
    H = 15 if lw else 18
    hdr = fits.getheader(crf)
    det = hdr['DETECTOR'].lower()
    gdet = {'nrcalong': 'nrca5', 'nrcblong': 'nrcb5'}.get(det, det)
    g = to_griddedpsfmodel(f'{Q}/psfs/nircam_{gdet}_{band.lower()}_fovp101_samp2_npsf16.fits')
    if isinstance(g, list):
        g = g[0]
    with fits.open(crf) as fh:
        sci = np.array(fh['SCI'].data, float)
        dq = np.array(fh['DQ'].data)
        pixar = fh['SCI'].header['PIXAR_SR']
    t = Table.read(cat)
    pixscale = t.meta['PIXSCALE']
    x = np.array(t['x_fit'], float); y = np.array(t['y_fit'], float)
    f = np.array(t['flux_fit'], float); fe = np.array(t['flux_err'], float)
    ok = (t['flags'] == 0) & np.isfinite(f) & (f > 0) & (f / fe > 30)
    ok &= (x > H + 1) & (y > H + 1) & (x < sci.shape[1] - H - 2) & (y < sci.shape[0] - H - 2)
    tree = cKDTree(np.c_[x, y])
    rn = 1.5 / pixscale
    sel = []
    for i in np.where(ok)[0]:
        nb = tree.query_ball_point([x[i], y[i]], rn)
        if any((j != i) and (f[j] > 0.01 * f[i]) for j in nb):
            continue
        sel.append(i)
    rng = np.random.default_rng(1)
    if len(sel) > maxstars:
        sel = sorted(rng.choice(sel, maxstars, replace=False))
    rows = []
    yy, xx = np.mgrid[0:2 * H + 1, 0:2 * H + 1]
    for i in sel:
        x0 = int(round(x[i])) - H; y0 = int(round(y[i])) - H
        cut = sci[y0:y0 + 2 * H + 1, x0:x0 + 2 * H + 1]
        xc = x[i] - x0; yc = y[i] - y0
        r = np.hypot(xx - xc, yy - yc)
        reg = r <= max(RADII) + 1
        dqc = dq[y0:y0 + 2 * H + 1, x0:x0 + 2 * H + 1]
        badpix = (~np.isfinite(cut)) | ((dqc & 1) != 0)
        if badpix[reg].any() or ((dqc & 2) != 0)[r <= ann[1] + 1].any():
            continue
        mod = g.evaluate(xx + x0, yy + y0, f[i], x[i], y[i])
        am = (r >= ann[0]) & (r < ann[1]) & ~badpix
        if am.sum() < 0.5 * np.pi * (ann[1] ** 2 - ann[0] ** 2):
            continue
        bd = sigma_clipped_stats(cut[am], sigma=3, maxiters=10)[1]
        bm = sigma_clipped_stats(mod[am], sigma=3, maxiters=10)[1]
        row = dict(frame=os.path.basename(crf), det=det, i=int(i), x=x[i], y=y[i], flux_fit=f[i],
                   flux_err=fe[i], mag_ab=-2.5 * np.log10(f[i] * pixar * 1e6 / 3631.0),
                   peak=cut[int(round(yc)), int(round(xc))] - bd, bkg=bd, bkg_model=bm,
                   msum=mod.sum())
        for R in RADII:
            ap = CircularAperture((xc, yc), R)
            area = ap.area
            row[f'ap{R}'] = ap.do_photometry(cut)[0][0] - area * bd
            row[f'mod{R}'] = ap.do_photometry(mod)[0][0] - area * bm
        rows.append(row)
    print(band, det, os.path.basename(crf)[-30:], len(sel), len(rows), flush=True)
    return rows


if __name__ == '__main__':
    band = sys.argv[1]
    nproc = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    maxstars = int(sys.argv[3]) if len(sys.argv) > 3 else 600
    crfs = sorted(glob.glob(f'{Q}/{band}/pipeline/jw03523005001_*_align_o005_crf.fits'))
    jobs = []
    for c in crfs:
        b = os.path.basename(c).split('_')
        vg, ex, det = b[1], b[2], b[3]
        cat = f'{Q}/{band}/{band.lower()}_{det}_visit001_vgroup{vg}_exp{ex}_resbgsub_m7_daophot_basic.fits'
        if os.path.exists(cat):
            jobs.append((band, c, cat, maxstars))
        else:
            print('no catalogue', cat)
    print(len(jobs), 'frames', flush=True)
    with Pool(nproc) as p:
        res = p.map(run_frame, jobs, chunksize=1)
    rows = [r for rr in res for r in rr]
    Table(rows).write(f'detclose_{band}.ecsv', overwrite=True)
