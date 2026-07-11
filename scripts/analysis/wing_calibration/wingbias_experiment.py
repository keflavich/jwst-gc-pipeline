"""Measure systematic bias of wing-only (core-masked) PSF fitting on real data.

Self-calibration experiment: bright UNSATURATED stars are fit (a) normally
(small fit_shape -> TRUE flux) and (b) with the core masked to radius r_mask
and a large fit_shape + LocalBackground, mimicking the satstar fit in
jwst_gc_pipeline/reduction/saturated_star_finding.py.

bias(r_mask) = median(flux_masked / flux_true).
"""
import os
import sys
import json
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats, mad_std
from astropy.table import Table
from photutils.detection import DAOStarFinder
from photutils.psf import PSFPhotometry
from photutils.background import LocalBackground
from stpsf.utils import to_griddedpsfmodel

SCRATCH = os.path.dirname(os.path.abspath(__file__))

R_MASKS = [2, 3, 4, 6, 8, 10, 12]
PEAK_LO, PEAK_HI = 500.0, 3500.0
N_STARS_MAX = 40


def odd(n):
    n = int(n)
    return n if n % 2 == 1 else n + 1


def load_frame(fn):
    with fits.open(fn) as hdul:
        data = hdul['SCI'].data.astype(float)
        err = hdul['ERR'].data.astype(float)
        dq = hdul['DQ'].data
    bad = (dq & 1).astype(bool) | ~np.isfinite(data) | ~np.isfinite(err) | (err <= 0)
    sat = (dq & 2).astype(bool)
    return data, err, bad, sat


def select_stars(data, bad, sat, fwhm_pix=2.0, edge=110):
    mean, med, std = sigma_clipped_stats(data[~bad], sigma=3.0, maxiters=5)
    dao = DAOStarFinder(threshold=10 * std, fwhm=fwhm_pix, exclude_border=True)
    cat = dao(data - med, mask=bad)
    if cat is None:
        raise RuntimeError('no detections')
    x = np.asarray(cat['xcentroid'])
    y = np.asarray(cat['ycentroid'])
    peak = np.asarray(cat['peak'])

    ny, nx = data.shape
    # candidate bright-but-unsaturated stars
    cand = np.where((peak > PEAK_LO) & (peak < PEAK_HI)
                    & (x > edge) & (x < nx - edge)
                    & (y > edge) & (y < ny - edge))[0]
    # sat-pixel positions for proximity veto
    sy, sx = np.nonzero(sat)

    selected = []
    for i in cand:
        d2 = (x - x[i]) ** 2 + (y - y[i]) ** 2
        d2[i] = np.inf
        # isolation: no comparable neighbor near, no significant neighbor
        # inside the wing-fit region
        if np.any((d2 < 20 ** 2) & (peak > 0.05 * peak[i])):
            continue
        if np.any((d2 < 60 ** 2) & (peak > 0.3 * peak[i])):
            continue
        # no saturated pixels within 40 px (their halos corrupt the wings)
        if len(sx) and np.any((sx - x[i]) ** 2 + (sy - y[i]) ** 2 < 40 ** 2):
            continue
        selected.append(i)
    selected = np.array(selected)
    # spread over brightness: sort by peak desc, take up to N
    order = np.argsort(peak[selected])[::-1]
    selected = selected[order][:N_STARS_MAX]
    return x[selected], y[selected], peak[selected], med


def fit_one(data, err, mask, psf, x0, y0, flux0, fit_shape, localbkg,
            fixed_pos):
    psf = psf.copy()
    psf.x_0.fixed = fixed_pos
    psf.y_0.fixed = fixed_pos
    phot = PSFPhotometry(psf_model=psf, fit_shape=(fit_shape, fit_shape),
                         localbkg_estimator=localbkg, aperture_radius=5)
    init = Table({'x_init': [x0], 'y_init': [y0], 'flux_init': [flux0]})
    res = phot(data, error=err, mask=mask, init_params=init)
    row = res[0]
    flags = int(row['flags'])
    return (float(row['flux_fit']), float(row['x_fit']), float(row['y_fit']),
            float(row['flux_err']) if np.isfinite(row['flux_err']) else np.nan,
            flags)


def run_frame(frame_fn, psf_fn, label, outer_test=True, annulus_test=True):
    print(f'=== {label}: {os.path.basename(frame_fn)}', flush=True)
    data, err, bad, sat = load_frame(frame_fn)
    psf = to_griddedpsfmodel(psf_fn)
    if isinstance(psf, list):
        psf = psf[0]
    xs, ys, peaks, skymed = select_stars(data, bad, sat)
    print(f'  selected {len(xs)} stars, peaks {peaks.min():.0f}-{peaks.max():.0f} MJy/sr',
          flush=True)

    yy, xx = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    lb_ref = LocalBackground(25, 50)

    rows = []
    for k, (x0, y0, pk) in enumerate(zip(xs, ys, peaks)):
        # cut a working box to keep the distance computation cheap
        # (masks are built full-frame but distance only near the star)
        # (a) TRUE flux: small fit_shape, free position
        try:
            f_true, xf, yf, fe_true, fl_true = fit_one(
                data, err, bad, psf, x0, y0, pk * 20.0, 11, lb_ref, False)
        except Exception as e:  # fit failure on one star: skip it
            print(f'  star {k}: true-fit failed: {e}', flush=True)
            continue
        if not np.isfinite(f_true) or f_true <= 0 or \
                np.hypot(xf - x0, yf - y0) > 1.5:
            print(f'  star {k}: bad true fit (f={f_true}, dpos='
                  f'{np.hypot(xf - x0, yf - y0):.2f})', flush=True)
            continue
        rec = dict(x=xf, y=yf, peak=pk, flux_true=f_true,
                   flux_err_true=fe_true, flags_true=fl_true)

        d2 = (xx - xf) ** 2 + (yy - yf) ** 2
        ok = True
        for rm in R_MASKS:
            m = bad | (d2 < rm ** 2)
            fs = min(odd(4 * rm + 21), 101)
            try:
                f_m, _, _, fe_m, fl_m = fit_one(
                    data, err, m, psf, xf, yf, f_true, fs, lb_ref, True)
            except Exception as e:
                print(f'  star {k} rmask={rm}: fit failed: {e}', flush=True)
                f_m, fe_m, fl_m = np.nan, np.nan, -1
            rec[f'ratio_r{rm}'] = f_m / f_true
            rec[f'flags_r{rm}'] = fl_m

        if outer_test:
            # fixed r_mask=6, vary the fit outer radius via fit_shape
            for fs in [25, 41, 61, 81, 101]:
                m = bad | (d2 < 6 ** 2)
                try:
                    f_m, _, _, _, _ = fit_one(
                        data, err, m, psf, xf, yf, f_true, fs, lb_ref, True)
                except Exception:
                    f_m = np.nan
                rec[f'ratio_fs{fs}'] = f_m / f_true

        if annulus_test:
            # fixed r_mask=6, fit_shape=45, vary the LocalBackground annulus
            m = bad | (d2 < 6 ** 2)
            for (bi, bo) in [(10, 20), (25, 50), (50, 80)]:
                try:
                    f_m, _, _, _, _ = fit_one(
                        data, err, m, psf, xf, yf, f_true, 45,
                        LocalBackground(bi, bo), True)
                except Exception:
                    f_m = np.nan
                rec[f'ratio_ann{bi}_{bo}'] = f_m / f_true
            # and no local background at all
            try:
                f_m, _, _, _, _ = fit_one(
                    data, err, m, psf, xf, yf, f_true, 45, None, True)
            except Exception:
                f_m = np.nan
            rec['ratio_annNone'] = f_m / f_true

        rows.append(rec)
        print(f'  star {k}: peak={pk:.0f} f_true={f_true:.1f} '
              + ' '.join(f'r{rm}={rec[f"ratio_r{rm}"]:.3f}' for rm in R_MASKS),
              flush=True)

    tbl = Table(rows)
    tbl.write(os.path.join(SCRATCH, f'wingbias_{label}.ecsv'),
              format='ascii.ecsv', overwrite=True)

    summary = {'label': label, 'frame': frame_fn, 'n_stars': len(tbl)}
    for rm in R_MASKS:
        r = np.asarray(tbl[f'ratio_r{rm}'])
        r = r[np.isfinite(r)]
        summary[f'r{rm}'] = dict(median=float(np.median(r)),
                                 madstd=float(mad_std(r)), n=len(r))
    for col in tbl.colnames:
        if col.startswith('ratio_fs') or col.startswith('ratio_ann'):
            r = np.asarray(tbl[col], dtype=float)
            r = r[np.isfinite(r)]
            summary[col[6:]] = dict(median=float(np.median(r)),
                                    madstd=float(mad_std(r)), n=len(r))
    with open(os.path.join(SCRATCH, f'summary_{label}.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return tbl, summary


if __name__ == '__main__':
    which = sys.argv[1] if len(sys.argv) > 1 else 'nrca1'
    base = '/blue/adamginsburg/adamginsburg/jwst/brick'
    cfg = {
        'nrca1': (f'{base}/F182M/pipeline/jw02221001001_07101_00001_nrca1_destreak_o001_crf.fits',
                  f'{base}/psfs/nircam_nrca1_f182m_fovp101_samp2_npsf16.fits'),
        'nrcb1': (f'{base}/F182M/pipeline/jw02221001001_07101_00001_nrcb1_destreak_o001_crf.fits',
                  f'{base}/psfs/nircam_nrcb1_f182m_fovp101_samp2_npsf16.fits'),
        'f187n_nrca1': (f'{base}/F187N/pipeline/jw02221001001_03101_00001_nrca1_destreak_o001_crf.fits',
                        f'{base}/psfs/nircam_nrca1_f187n_fovp101_samp2_npsf16.fits'),
    }
    fn, pfn = cfg[which]
    run_frame(fn, pfn, which)
