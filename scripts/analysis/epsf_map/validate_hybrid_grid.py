"""Validate the pipeline hybrid PSF grid (epsf_hybrid) with the production fitter.

    python validate_hybrid_grid.py <stpsf grid .fits> <core dir> <o081_<det>.npz> <paired npz> <filter> <out prefix>

Fits the same isolated catalog stars of a held-out frame (outputs of
fullfield_residual.py / fullfield_paired.py) with photutils ``PSFPhotometry`` the
way cataloging does (``fit_shape=(5, 5)``, ``LevMarLSQFitter``), once with the
plain STPSF grid and once with ``maybe_apply_epsf_core`` applied, then renders
each fitted model over r <= 10 px and compares:

* chi^2 / pixel in annuli around the stars (residual after subtraction);
* the mean stacked residual / flux of the brightest 20%;
* flux ratio hybrid / STPSF (normalisation must be preserved);
* position differences hybrid - STPSF (the astrometric convention must be);
* data / model summed inside r <= 8 px: which model's fitted flux matches the
  star's aperture flux (1 = right; isolated stars, so neighbours are small);
* pixel-phase flux modulation: each model's native-pixel flux inside 10 px for
  a unit-flux star at 4 x 4 sub-pixel phases (peak-to-peak), and the fitted
  flux ratio hybrid / STPSF binned in x and y pixel phase.

Writes <out prefix>.json and <out prefix>.png.  The background is the
fullfield_residual STPSF-fit bilinear background; neighbours are those of
the isolated selection (none brighter than 10% of the star within 8 px).
"""
import json
import sys

import numpy as np
from astropy.table import Table
from astropy.modeling.fitting import LevMarLSQFitter
from photutils.psf import PSFPhotometry
from jwst_gc_pipeline.photometry.psf_grid_io import load_stpsf_grid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from jwst_gc_pipeline.photometry.epsf_hybrid import maybe_apply_epsf_core, grid_detector, EPSF_CORE_DIR_ENV

A = [(0, 1.5), (1.5, 3), (3, 6), (6, 10)]
H = 10
NPHASE = 4


def phase_flux(g, xc=1024, yc=1024, r=10, h=50):
    """Native-pixel flux inside r of a unit-flux star at NPHASE x NPHASE sub-pixel phases."""
    k = np.arange(-h, h + 1)
    X, Y = np.meshgrid(k, k)
    ph = np.arange(NPHASE) / NPHASE
    return np.array([[g.evaluate(X + xc, Y + yc, 1.0, xc + px, yc + py)[np.hypot(X - px, Y - py) <= r].sum()
                      for px in ph] for py in ph])


def fit(grid, img, err, mask, x, y, f):
    phot = PSFPhotometry(grid, fit_shape=(5, 5), fitter=LevMarLSQFitter(), aperture_radius=3, progress_bar=False)
    init = Table(dict(x=x, y=y, flux=f))
    t = phot(img, error=err, mask=mask, init_params=init)
    return np.asarray(t['x_fit']), np.asarray(t['y_fit']), np.asarray(t['flux_fit'])


def main():
    gfn, cdir, zfn, pfn, filt, outp = sys.argv[1:7]
    st = load_stpsf_grid(gfn)
    st = st[0] if isinstance(st, list) else st
    hyb = maybe_apply_epsf_core(st, grid_detector(st), filt, environ={EPSF_CORE_DIR_ENV: cdir})
    assert hyb is not st, 'no core file for this grid'
    z = np.load(zfn); p = np.load(pfn)
    sel, f0 = p['sel'], p['f']
    img = z['sci'] - z['stpsf_bkg']; err = z['err']; good = z['good']
    ny, nx = img.shape
    x0, y0 = z['stpsf_x'][sel], z['stpsf_y'][sel]
    ok = (x0 > H + 3) & (x0 < nx - H - 4) & (y0 > H + 3) & (y0 < ny - H - 4) & (f0 > 0)
    sel, f0, x0, y0 = sel[ok], f0[ok], x0[ok], y0[ok]
    mask = ~good | ~np.isfinite(img) | ~np.isfinite(err) | (err <= 0)
    res = {}
    k = np.arange(-H, H + 1); UX, UY = np.meshgrid(k, k); R = np.hypot(UX, UY)
    for name, g in (('stpsf', st), ('hybrid', hyb)):
        xf, yf, ff = fit(g, np.where(mask, 0, img), np.where(mask, 1, err), mask, x0, y0, f0)
        C = np.full((len(sel), len(A)), np.nan); stack = []; apr = np.full(len(sel), np.nan)
        for i in range(len(sel)):
            if not (np.isfinite(xf[i]) and ff[i] > 0):
                continue
            ix, iy = int(round(xf[i])), int(round(yf[i]))
            s = np.s_[iy - H:iy + H + 1, ix - H:ix + H + 1]
            m = g.evaluate(UX + ix, UY + iy, ff[i], xf[i], yf[i])
            chi = np.where(good[s], (img[s] - m) / err[s], np.nan)
            C[i] = [np.nanmean(chi[(R >= a) & (R < b)] ** 2) for a, b in A]
            stack.append(np.where(good[s], (img[s] - m) / ff[i], np.nan))
            # flux check: data / model summed over the same unmasked pixels inside r <= 8
            ap = good[s] & (R <= 8)
            if ap.mean() > 0.9 * (R <= 8).mean():
                apr[i] = img[s][ap].sum() / m[ap].sum()
        res[name] = dict(x=xf, y=yf, f=ff, C=C, stack=np.asarray(stack), apr=apr)
    top = f0 >= np.percentile(f0, 80)
    out = dict(frame=zfn, grid=gfn, n_stars=int(len(sel)), oversampling=np.atleast_1d(st.oversampling).tolist(),
               centroid_shift_max_mpix=1e3 * hyb.meta['epsf_centroid_shift_max_px'])
    pf = {name: phase_flux(g) for name, g in (('stpsf', st), ('hybrid', hyb))}
    out['phase_flux_r10_peak_to_peak_pct'] = {name: float(100 * np.ptp(v) / v.mean()) for name, v in pf.items()}
    for name in res:
        out[name] = dict(annular_chi2_median_all=np.nanmedian(res[name]['C'], 0).tolist(),
                         annular_chi2_median_top20=np.nanmedian(res[name]['C'][top], 0).tolist(),
                         stack_rms_r8_top20=float(np.sqrt(np.nanmean(
                             np.nanmean(res[name]['stack'][top[:len(res[name]['stack'])]], 0)[R <= 8] ** 2))),
                         aperture_data_over_model_r8_median_top20=float(np.nanmedian(res[name]['apr'][top])),
                         aperture_data_over_model_r8_median_all=float(np.nanmedian(res[name]['apr'])))
    fr = res['hybrid']['f'] / res['stpsf']['f']
    dx = res['hybrid']['x'] - res['stpsf']['x']; dy = res['hybrid']['y'] - res['stpsf']['y']
    out['flux_ratio_hybrid_over_stpsf'] = dict(median=float(np.nanmedian(fr)), median_top20=float(np.nanmedian(fr[top])),
                                               p16_84=np.nanpercentile(fr, [16, 84]).tolist())
    out['dpos_hybrid_minus_stpsf_mpix'] = dict(dx_median=1e3 * float(np.nanmedian(dx)), dy_median=1e3 * float(np.nanmedian(dy)),
                                               dx_median_top20=1e3 * float(np.nanmedian(dx[top])),
                                               dy_median_top20=1e3 * float(np.nanmedian(dy[top])))
    pbins = np.linspace(0, 1, NPHASE * 2 + 1)
    pc = 0.5 * (pbins[1:] + pbins[:-1])
    xph = np.mod(res['stpsf']['x'] + 0.5, 1)      # 0 = pixel edge, 0.5 = pixel centre
    yph = np.mod(res['stpsf']['y'] + 0.5, 1)
    good_fr = np.isfinite(fr) & top
    fr_ph = {}
    for ax_name, ph in (('x', xph), ('y', yph)):
        idx = np.digitize(ph, pbins) - 1
        fr_ph[ax_name] = [float(np.nanmedian(fr[good_fr & (idx == b)])) if np.any(good_fr & (idx == b)) else None
                          for b in range(len(pc))]
    out['flux_ratio_vs_phase_top20'] = dict(phase_bin_centre=pc.tolist(), **fr_ph)
    # which model's fitted flux is phase-dependent: aperture data / model (r <= 8) vs phase, each model
    rph = np.hypot(xph - 0.5, yph - 0.5)          # 0 = star on a pixel centre, 0.71 = on a corner
    rb = np.linspace(0, 0.71, 6)
    out['aperture_ratio_vs_radial_phase_top20'] = dict(bin_centre=(0.5 * (rb[1:] + rb[:-1])).tolist(), **{
        name: [float(np.nanmedian(res[name]['apr'][top & (rph >= a) & (rph < b)])) for a, b in zip(rb[:-1], rb[1:])]
        for name in res})
    json.dump(out, open(outp + '.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))

    fig, ax = plt.subplots(1, 5, figsize=(25, 4.8))
    rc = [0.5 * (a + b) for a, b in A]
    for name, col in (('stpsf', 'k'), ('hybrid', 'C3')):
        ax[0].plot(rc, out[name]['annular_chi2_median_top20'], 'o-', color=col, label=f'{name}, brightest 20%')
        ax[0].plot(rc, out[name]['annular_chi2_median_all'], 's--', color=col, mfc='none', label=f'{name}, all')
    ax[0].set_yscale('log'); ax[0].set_xlabel('radius [px]'); ax[0].set_ylabel('median chi^2 / pixel'); ax[0].legend()
    ax[0].set_title(f'{filt}: PSFPhotometry fit_shape=(5,5), N={len(sel)}')
    vv = np.nanpercentile(np.abs(np.nanmean(res['stpsf']['stack'][top], 0)), 99)
    for q, name in enumerate(('stpsf', 'hybrid')):
        im = ax[1 + q].imshow(np.nanmean(res[name]['stack'][top], 0), origin='lower', cmap='RdBu_r', vmin=-vv, vmax=vv,
                              extent=[-H - .5, H + .5, -H - .5, H + .5])
        ax[1 + q].set_title(f'{name}: mean residual / flux, brightest 20%\nrms(r<=8) = {out[name]["stack_rms_r8_top20"]:.2e}')
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    ax[3].scatter(1e3 * dx, 1e3 * dy, s=2, alpha=0.3, c=np.log10(np.clip(f0, 1e-3, None)))
    ax[3].set_xlabel('x hybrid - STPSF [mpix]'); ax[3].set_ylabel('y hybrid - STPSF [mpix]')
    ax[3].set_xlim(-60, 60); ax[3].set_ylim(-60, 60); ax[3].axhline(0, color='k', lw=.5); ax[3].axvline(0, color='k', lw=.5)
    ax[3].set_title(f'fitted position difference\nmedian ({out["dpos_hybrid_minus_stpsf_mpix"]["dx_median"]:.1f}, '
                    f'{out["dpos_hybrid_minus_stpsf_mpix"]["dy_median"]:.1f}) mpix; flux ratio {out["flux_ratio_hybrid_over_stpsf"]["median"]:.4f}')
    for ax_name, mk in (('x', 'o-'), ('y', 's--')):
        ax[4].plot(pc, [np.nan if v is None else v for v in fr_ph[ax_name]], mk, label=f'{ax_name} phase')
    ax[4].set_xlabel('pixel phase (0.5 = pixel centre)'); ax[4].set_ylabel('fitted flux hybrid / STPSF, brightest 20%')
    ax[4].legend(loc='upper left')
    ap = out['aperture_ratio_vs_radial_phase_top20']
    a4 = ax[4].twiny()
    for name, col in (('stpsf', 'k'), ('hybrid', 'C3')):
        v = np.array(ap[name]); a4.plot(ap['bin_centre'], v / np.nanmean(v) * np.nanmean(fr[top]), ':', color=col, marker='d',
                                       label=f'{name}: aperture data/model (scaled)')
    a4.set_xlabel('distance of star from pixel centre [px] (dotted)'); a4.legend(loc='lower right', fontsize=8)
    pp = out['phase_flux_r10_peak_to_peak_pct']
    ax[4].set_title(f'model flux r<=10 px over {NPHASE}x{NPHASE} phases, p2p:\n'
                    f'STPSF {pp["stpsf"]:.2f}%, hybrid {pp["hybrid"]:.2f}%')
    fig.tight_layout(); fig.savefig(outp + '.png', dpi=150)


if __name__ == '__main__':
    main()
