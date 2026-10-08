"""In-frame Delta with a per-bin pooled fit sum(d-bg) = (1+S) M + dbar N over bright isolated unsat stars,
split by peak brightness; compare to offline raw Delta_unsat (v7b) and radprof offset-corrected sat/unsat S."""
import sys, time, warnings
import numpy as np
sys.path.insert(0, '/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline-satwing')
sys.path.insert(0, '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ/satrefit')
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.modeling.fitting import LevMarLSQFitter
from photutils.background import LocalBackground
import jwst_gc_pipeline.reduction.satstar_psf_radial as R
import jwst_gc_pipeline.reduction.saturated_star_finding as S
from jwst_gc_pipeline.photometry.psf_fitting import _make_psfphotometry
warnings.filterwarnings('ignore', category=RuntimeWarning)
Q = '/orange/adamginsburg/jwst/wd2/dolphot_benchmark/Q_integ'
band, det, exp = sys.argv[1], sys.argv[2], int(sys.argv[3])
lw = band in ('250M', '300M')
vg = {'150W': '10101', '200W': '12101', '250M': '04101', '300M': '12101'}[band]
stem = f'jw03523005001_{vg}_{exp:05d}_{det}_align_o005_crf'
fh = fits.open(f'{Q}/tree_main2/F{band}/pipeline/{stem}.fits')
hdr = fh[0].header
data = np.array(fh['SCI'].data, float)
data[np.isnan(fh['VAR_POISSON'].data)] = 0
err = np.array(fh['ERR'].data, float)
dq = fh['DQ'].data
sat = (dq & 2) > 0
bad = sat | ((dq & 1) > 0) | ~np.isfinite(data) | (data == 0) | ~np.isfinite(err) | (err <= 0)
good = ~bad
pix = float(np.sqrt(fh['SCI'].header['PIXAR_A2']))
fwhm, fwhm_pix = S.get_fwhm(hdr, instrument_replacement='NIRCam'); fwhm_pix = float(fwhm_pix)
import satrefit_core as C
grid, gf = C.load_grid(Q + '/tree_main2/psfs', hdr, lw)
rmax_as = 0.5 if lw else 0.35
rmax = rmax_as / pix
edges = R.default_bin_edges(rmax)
nb = len(edges) - 1
rc = 0.5 * (edges[1:] + edges[:-1])
r_bg_in, r_bg_out = 1.8 / pix, 2.4 / pix
half = int(np.ceil(r_bg_out)) + 1
yc, xc = R._select_calibrators(data, good, sat, fwhm_pix=fwhm_pix, edge=half + 1, iso_pix=(1.0 if lw else 0.6) / pix,
                               iso_frac=0.05, sat_iso_pix=1.0 / pix, n_max=1000)
ap = 2.0 * fwhm_pix
lb_in = max(6, int(round(ap + 0.5 * fwhm_pix))); lb_out = lb_in + max(4, int(round(fwhm_pix)))
phot = _make_psfphotometry(localbkg_estimator=LocalBackground(lb_in, lb_out), psf_model=grid, fitter=LevMarLSQFitter(),
                           fit_shape=(5, 5), aperture_radius=ap, progress_bar=False)
res = phot(np.where(good, data, 0.0), mask=bad, error=np.where(good, err, 1e10),
           init_params=Table({'x': xc.astype(float), 'y': yc.astype(float)}))
rin = int(np.ceil(rmax)) + 1
D, M, N, F, LB = [], [], [], [], []
for x, y, f, lbk in zip(res['x_fit'], res['y_fit'], res['flux_fit'], res['local_bkg']):
    if not (np.isfinite(x) and np.isfinite(y) and f > 0):
        continue
    ix, iy = int(round(x)), int(round(y))
    if not (half <= ix < data.shape[1] - half and half <= iy < data.shape[0] - half):
        continue
    sub = data[iy - half:iy + half + 1, ix - half:ix + half + 1]
    gsub = good[iy - half:iy + half + 1, ix - half:ix + half + 1]
    gy, gx = np.mgrid[iy - half:iy + half + 1, ix - half:ix + half + 1]
    rr = np.hypot(gx - x, gy - y)
    bsel = gsub & (rr >= r_bg_in) & (rr < r_bg_out)
    if bsel.sum() < 50:
        continue
    _, bg, _ = sigma_clipped_stats(sub[bsel], sigma=3.0, maxiters=5)
    c = slice(half - rin, half + rin + 1)
    model = grid.evaluate(gx[c, c].astype(float), gy[c, c].astype(float), f, x, y)
    d = sub[c, c] - bg; g = gsub[c, c]; r = rr[c, c]
    dr, mr, nr = np.full(nb, np.nan), np.full(nb, np.nan), np.full(nb, np.nan)
    for k in range(nb):
        inb = (r >= edges[k]) & (r < edges[k + 1]); use = inb & g
        if use.sum() < 0.9 * inb.sum() or use.sum() == 0:
            continue
        dr[k], mr[k], nr[k] = d[use].sum(), model[use].sum(), use.sum()
    D.append(dr); M.append(mr); N.append(nr); F.append(f); LB.append(lbk - bg)
D, M, N, F, LB = map(np.array, (D, M, N, F, LB))
mag = -2.5 * np.log10(F)
print(f'{band} {det} exp{exp}: n={len(F)} flux range mag {mag.min():.2f}..{mag.max():.2f} (inst); daophot local_bkg - far bg median {np.median(LB):+.4f} MJy/sr', flush=True)


def pooled(sel):
    S_, B_, n_ = np.full(nb, np.nan), np.full(nb, np.nan), np.zeros(nb, int)
    for k in range(nb):
        ok = sel & np.isfinite(D[:, k]) & np.isfinite(M[:, k])
        for it in range(6):
            if ok.sum() < 10:
                break
            y = D[ok, k] / N[ok, k]; A = np.column_stack([M[ok, k] / N[ok, k], np.ones(ok.sum())])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            resid = D[:, k] / N[:, k] - (coef[0] * M[:, k] / N[:, k] + coef[1])
            s = 1.4826 * np.nanmedian(np.abs(resid[ok] - np.nanmedian(resid[ok])))
            new = sel & np.isfinite(resid) & (np.abs(resid) < 4 * s)
            if (new == ok).all():
                break
            ok = new
        if ok.sum() >= 10:
            S_[k], B_[k], n_[k] = coef[0] - 1, coef[1], ok.sum()
    return S_, B_, n_


def ratio_med(sel):
    with np.errstate(all='ignore'):
        return np.nanmedian(np.where(sel[:, None], D / M - 1, np.nan), axis=0)


all_ = np.ones(len(F), bool)
q = np.nanpercentile(mag, [33, 67])
bright = mag < q[0]; mid = (mag >= q[0]) & (mag < q[1]); faint = mag >= q[1]
Sa, Ba, na = pooled(all_)
Sb, _, _ = pooled(bright); Sm, _, _ = pooled(mid); Sf, _, _ = pooled(faint)
rmed = ratio_med(all_)
print(' r_px  r_as   rawmed   S_all   dbar_all   n   S_bright  S_mid  S_faint')
for k in range(nb):
    print(f'{rc[k]:5.2f} {rc[k]*pix:5.3f} {rmed[k]:+7.3f} {Sa[k]:+7.3f} {Ba[k]:+9.4f} {na[k]:4d} {Sb[k]:+7.3f} {Sm[k]:+7.3f} {Sf[k]:+7.3f}')
