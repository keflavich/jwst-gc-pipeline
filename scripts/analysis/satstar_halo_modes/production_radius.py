"""Does the halo-mode flux help at the PRODUCTION fit radius (#1013 follow-up)?

    python production_radius.py psf <psf dir> [detector [WSS OPD file]]
    python production_radius.py fit <psf dir> <out prefix> cal_glob

The 7-star evidence of ``halo_photometry.py`` used stars whose saturated
core reaches r ~ 100 px, fitted to 100-200 px.  The production hook
(``SATSTAR_HALO_MODES=1`` in ``saturated_star_finding``) fits inside the
photutils box, r <= 40.5 px, so it only ever sees moderately saturated
stars.  This measures those: every SATURATED region of 5-1000 px (core
radius ~1-18 px) in one star-visit's ``_cal`` frames, carried between the
dithers through the GWCS (``frame_wcs``; no catalog matching), its position
refined on the standard fit's chi2, and fitted on the
same pixels with ``F P + B`` and with the halo modes (knots from
``satstar_halo_knots``, as the hook does) at rmax = 40.5 and 60 px.  Each
star is constant, so the per-star dither scatter of F is the figure of merit.

``psf`` writes STPSF F480M ``OVERDIST`` images (oversample 4, 161 px) on a
4x4 grid of detector positions; ``fit`` uses the node nearest each star
(identical for both models).
"""
import glob
import json
import os
import sys

import numpy as np
from astropy.io import fits
from scipy import ndimage, optimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from jwst_gc_pipeline.frame_wcs import frame_wcs
from jwst_gc_pipeline.photometry.satstar_halo_modes import fit_flux_with_halo_modes, satstar_halo_knots

OS = 4
FOV = 161
HALF = 70                         # native half-size of the cutout and the PSF stamp
NODES = (256.0, 768.0, 1280.0, 1792.0)
RMAXES = (40.5, 60.0)
NSAT = (5, 1000)
LINK_ARCSEC = 0.5


def make_psfs(psfdir, det='NRCB5', opd=None):
    import stpsf
    from jwst_gc_pipeline.photometry.psf_channel import nircam_channel_safe_psf_kwargs
    os.makedirs(psfdir, exist_ok=True)
    for x in NODES:
        for y in NODES:
            out = os.path.join(psfdir, f'psf_{int(x)}_{int(y)}.npy')
            if os.path.exists(out):
                continue
            nc = stpsf.NIRCam(); nc.filter = 'F480M'; nc.detector = det; nc.detector_position = (x, y)
            if opd:
                nc.load_wss_opd(opd, plot=False, verbose=False, use_exact_wss_target_phase=False)
            p = nc.calc_psf(fov_pixels=FOV, oversample=OS, **nircam_channel_safe_psf_kwargs(nc))
            np.save(out, p['OVERDIST'].data.astype(np.float32))
            print('psf', out, flush=True)


def native_psf(P4, xt, yt, n):
    """Pixel-integrated PSF on an n x n grid (n = 2 HALF + 1) for a star at
    (xt, yt) in that grid; normalised to unit sum of the stamp."""
    ix, iy = int(np.floor(xt + 0.5)), int(np.floor(yt + 0.5))
    S = ndimage.shift(P4.astype(float), (OS * (yt - iy), OS * (xt - ix)), order=1, mode='constant')
    c = (P4.shape[0] - OS * n) // 2
    S = S[c:c + OS * n, c:c + OS * n].reshape(n, OS, n, OS).sum((1, 3))
    S /= S.sum()
    out = np.zeros((n, n))
    dx, dy = ix - HALF, iy - HALF
    ys, xs = slice(max(dy, 0), min(n + dy, n)), slice(max(dx, 0), min(n + dx, n))
    out[ys, xs] = S[ys.start - dy:ys.stop - dy, xs.start - dx:xs.stop - dx]
    return out


def exposure(fn, psfs):
    with fits.open(fn) as h:
        d = h['SCI'].data.astype(float)
        dq = h['DQ'].data.astype(np.int64)
        var = (h['VAR_POISSON'].data + h['VAR_RNOISE'].data).astype(float)
    sat = (dq & 2) > 0
    lab, nl = ndimage.label(sat, structure=np.ones((3, 3)))
    sizes = ndimage.sum(np.ones_like(lab), lab, np.arange(1, nl + 1))
    keep = np.flatnonzero((sizes >= NSAT[0]) & (sizes <= NSAT[1])) + 1
    big = np.flatnonzero(sizes >= NSAT[0]) + 1
    com = np.array(ndimage.center_of_mass(np.ones_like(lab), lab, big))
    cz = dict(zip(big, com))
    n = 2 * HALF + 1
    rows = []
    for L in keep:
        y0, x0 = cz[L]
        if not (HALF + 2 < x0 < 2045 - HALF and HALF + 2 < y0 < 2045 - HALF):
            continue
        # isolation: no saturated region at least half as large within 45 px
        others = [(cz[m], sizes[m - 1]) for m in big if m != L]
        if any(np.hypot(c[1] - x0, c[0] - y0) < 45 and s >= 0.5 * sizes[L - 1] for c, s in others):
            continue
        X0, Y0 = int(round(x0)) - HALF, int(round(y0)) - HALF
        sl = (slice(Y0, Y0 + n), slice(X0, X0 + n))
        xt, yt = x0 - X0, y0 - Y0
        node = min(psfs, key=lambda k: np.hypot(k[0] - x0, k[1] - y0))
        P = native_psf(psfs[node], xt, yt, n)
        bad = ndimage.binary_dilation((dq[sl] & 3) > 0, iterations=3) | ~np.isfinite(d[sl]) | ~(var[sl] > 0)
        err = np.sqrt(np.where(var[sl] > 0, var[sl], np.nan))
        r_core = float(np.sqrt(sizes[L - 1] / np.pi)) + 3.0
        # the production fit solves for the position; the DQ centroid of a
        # saturated core is only good to ~0.5 px, so refine it on the
        # standard fit's chi2 (the same position is then used by both models)
        def chi2(p):
            if np.hypot(*p) > 2:
                return np.inf
            Pp = native_psf(psfs[node], xt + p[0], yt + p[1], n)
            return fit_flux_with_halo_modes(d[sl], err, bad, Pp, xt + p[0], yt + p[1], knots=None,
                                            rmax=RMAXES[0], niter=1).chi2_dof
        opt = optimize.minimize(chi2, [0.0, 0.0], method='Nelder-Mead',
                                options=dict(xatol=0.02, fatol=1e-3, maxiter=80))
        if np.hypot(*opt.x) > 2:
            continue
        xt, yt = xt + opt.x[0], yt + opt.x[1]
        P = native_psf(psfs[node], xt, yt, n)
        res = dict(x=X0 + xt, y=Y0 + yt, nsat=int(sizes[L - 1]), r_core=r_core)
        for rmax in RMAXES:
            knots = satstar_halo_knots(r_core, rmax)
            s = fit_flux_with_halo_modes(d[sl], err, bad, P, xt, yt, knots=None, rmax=rmax)
            h = (fit_flux_with_halo_modes(d[sl], err, bad, P, xt, yt, knots=knots, rmax=rmax)
                 if knots is not None else None)
            res[rmax] = (s.flux, h.flux if h is not None else np.nan)
        rows.append(res)
    return rows, frame_wcs(fn)


def fit(psfdir, outp, pat):
    psfs = {}
    for f in glob.glob(os.path.join(psfdir, 'psf_*.npy')):
        x, y = os.path.basename(f)[4:-4].split('_')
        psfs[(float(x), float(y))] = np.load(f)
    fns = sorted(glob.glob(pat))
    allrows = []
    for e, fn in enumerate(fns):
        rows, w = exposure(fn, psfs)
        sky = w.pixel_to_world(np.array([r['x'] for r in rows]), np.array([r['y'] for r in rows]))
        for r, c in zip(rows, sky):
            r['exp'], r['sky'] = e, c
        allrows += rows
        print(os.path.basename(fn)[:36], len(rows), flush=True)
    clusters = []
    for r in allrows:
        for k in clusters:
            if k[0]['sky'].separation(r['sky']).arcsec < LINK_ARCSEC and r['exp'] not in {q['exp'] for q in k}:
                k.append(r)
                break
        else:
            clusters.append([r])
    res = dict(n_frames=len(fns))
    figdat = {}
    for rmax in RMAXES:
        sc_S, sc_H, ratio, nsat = [], [], [], []
        for k in clusters:
            if len(k) < 4:
                continue
            FS = np.array([q[rmax][0] for q in k]); FH = np.array([q[rmax][1] for q in k])
            if not (np.all(FS > 0) and np.all(FH > 0)):
                continue
            sc_S.append(np.std(FS / FS.mean(), ddof=1)); sc_H.append(np.std(FH / FH.mean(), ddof=1))
            ratio.append(float(np.median(FH / FS))); nsat.append(float(np.median([q['nsat'] for q in k])))
        sc_S, sc_H, ratio, nsat = map(np.array, (sc_S, sc_H, ratio, nsat))
        res[f'rmax{rmax:g}'] = dict(
            n_stars=int(sc_S.size),
            dither_rms_S_median=float(np.median(sc_S)), dither_rms_H_median=float(np.median(sc_H)),
            frac_stars_H_better=float(np.mean(sc_H < sc_S)),
            ratio_H_over_S_median=float(np.median(ratio)),
            ratio_H_over_S_p16_p84=[float(np.percentile(ratio, 16)), float(np.percentile(ratio, 84))])
        figdat[rmax] = (sc_S, sc_H, ratio, nsat)
    json.dump(res, open(outp + '.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
    for i, rmax in enumerate(RMAXES):
        sc_S, sc_H, ratio, nsat = figdat[rmax]
        ax[0].scatter(sc_S, sc_H, s=10, color=f'C{i}', label=f'r ≤ {rmax:g} px ({sc_S.size} stars)')
        ax[1].scatter(nsat, ratio, s=10, color=f'C{i}', label=f'r ≤ {rmax:g} px')
    lim = [1e-3, 0.5]
    ax[0].plot(lim, lim, 'k-', lw=0.5); ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_xlabel('dither rms of F, standard F P + B'); ax[0].set_ylabel('dither rms of F, halo modes')
    ax[0].legend(fontsize=8)
    ax[1].axhline(1, color='k', lw=0.5); ax[1].set_xscale('log')
    ax[1].set_xlabel('saturated pixels'); ax[1].set_ylabel('F halo modes / F standard (median over dithers)')
    ax[1].legend(fontsize=8)
    fig.suptitle('halo-mode flux at the production fit radius: moderately saturated F480M stars', fontsize=9)
    fig.tight_layout(); fig.savefig(outp + '.png', dpi=int(os.environ.get('FIG_DPI', 130)))


if __name__ == '__main__':
    if sys.argv[1] == 'psf':
        make_psfs(sys.argv[2], *sys.argv[3:5])
    else:
        fit(sys.argv[2], sys.argv[3], sys.argv[4])
