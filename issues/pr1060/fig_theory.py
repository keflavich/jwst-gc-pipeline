"""Theoretical before/after for #1055 / PR #1060.

Noise-free stars rendered from the production F480M NRCB5 grid at their
detector position, fitted the way the satstar path fits them (162 px cutout,
r < 6 px core masked), once with the bare grid at cutout coordinates (the bug)
and once with psf_in_cutout_coords (the fix).
"""
import os
os.environ.setdefault('STPSF_PATH', '/orange/adamginsburg/jwst/stpsf-data/')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from stpsf.utils import to_griddedpsfmodel
from photutils.psf import PSFPhotometry, GriddedPSFModel
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.table import QTable
from jwst_gc_pipeline.reduction import saturated_star_finding as ssf

GRID = '/orange/adamginsburg/jwst/gc-treasury/psfs/nircam_nrcb5_f480m_fovp1024_samp2_npsf16.fits'
g = to_griddedpsfmodel(GRID)
g = g[0] if isinstance(g, list) else g
ev = GriddedPSFModel.evaluate
PAD, F, RMASK = 81, 1e6, 6.0


def fit(X, Y, fixed):
    x0, y0 = int(round(X)) - PAD, int(round(Y)) - PAD
    yy, xx = np.mgrid[y0:y0 + 2 * PAD, x0:x0 + 2 * PAD].astype(float)
    cut = ev(g, xx, yy, F, X, Y)
    mask = (xx - X)**2 + (yy - Y)**2 < RMASK**2
    init = QTable()
    init['x'] = [X - x0]
    init['y'] = [Y - y0]
    model = ssf.psf_in_cutout_coords(g, x0, y0) if fixed else g
    r = PSFPhotometry(psf_model=model, fit_shape=81, aperture_radius=10,
                      fitter=LevMarLSQFitter())(cut, init_params=init, mask=mask)
    xf, yf, ff = (float(r[k][0]) for k in ('x_fit', 'y_fit', 'flux_fit'))
    cy, cx = np.mgrid[0:2 * PAD, 0:2 * PAD].astype(float)
    mod = model.evaluate(cx, cy, ff, xf, yf)
    return dict(cut=cut, mask=mask, resid=cut - mod, ratio=ff / F,
                dx=xf + x0 - X, dy=yf + y0 - Y)


# flux/true map over the detector, bug vs fix
pos = np.linspace(90, 1958, 5)


def _ratio(XY):
    return fit(XY[0] + 0.3, XY[1] + 0.6, False)['ratio']


if __name__ == '__main__':
    import multiprocessing as mp
    if os.path.exists('rmap.npy'):
        rmap = np.load('rmap.npy')
    else:
        with mp.get_context('fork').Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 2))) as pool:
            vals = pool.map(_ratio, [(X, Y) for Y in pos for X in pos], chunksize=1)
        rmap = np.array(vals).reshape(len(pos), len(pos))
    print(np.round(rmap, 4), flush=True)

    corners = {(i, j): rmap[j, i] for i in (0, -1) for j in (0, -1) if (i, j) != (0, 0)}
    (ci, cj) = max(corners, key=lambda k: abs(corners[k] - 1))
    XW, YW = pos[ci] + 0.3, pos[cj] + 0.6
    print('worst corner', XW, YW, corners[(ci, cj)])

    bad = fit(XW, YW, False)
    good = fit(XW, YW, True)

    # PSFs: true one at the star vs the one the buggy fit used (node near 81,81)
    yy, xx = np.mgrid[-40:41, -40:41].astype(float)
    p_true = ev(g, xx + XW, yy + YW, 1.0, XW, YW)
    p_bug = ev(g, xx + 81 + (XW % 1), yy + 81 + (YW % 1), 1.0, 81 + (XW % 1), 81 + (YW % 1))
    pk = p_true.max()

    fig, ax = plt.subplots(2, 3, figsize=(18, 11))
    ln = LogNorm(vmin=pk * 1e-5, vmax=pk)
    ext = [-40.5, 40.5, -40.5, 40.5]
    a = ax[0, 0].imshow(p_true, norm=ln, origin='lower', cmap='magma', extent=ext)
    ax[0, 0].set_title(f'PSF at the star, detector ({XW:.0f}, {YW:.0f})\n(what the fix uses)', fontsize=10)
    ax[0, 1].imshow(p_bug, norm=ln, origin='lower', cmap='magma', extent=ext)
    ax[0, 1].set_title('PSF the old fit used, detector (81, 81)\n(cutout coordinates read as detector)', fontsize=10)
    fig.colorbar(a, ax=ax[0, :2], shrink=0.8, label='PSF / peak (log)')
    d = (p_bug - p_true) / pk
    lim = np.nanmax(np.abs(d))
    b = ax[0, 2].imshow(d, norm=TwoSlopeNorm(0, -lim, lim), origin='lower', cmap='RdBu_r', extent=ext)
    ax[0, 2].set_title('old PSF − correct PSF (fraction of peak)\ndashed circle = r<6 px core masked in the fit', fontsize=10)
    ax[0, 2].add_patch(plt.Circle((0, 0), RMASK, fill=False, ls='--', color='k'))
    fig.colorbar(b, ax=ax[0, 2], shrink=0.8)
    for k in range(3):
        ax[0, k].set_xlabel('Δx [px]')
    ax[0, 0].set_ylabel('Δy [px]')

    rl = np.nanpercentile(np.abs(bad['resid'][~bad['mask']]), 99.5)
    for k, (res, lab) in enumerate([(bad, 'before (main)'), (good, 'after (#1060)')]):
        im = np.where(res['mask'], np.nan, res['resid'])
        c = ax[1, k].imshow(im, norm=TwoSlopeNorm(0, -rl, rl), origin='lower', cmap='RdBu_r')
        ax[1, k].set_title(f'{lab}: fit residual, core r<{RMASK:.0f} px masked\n'
                           f'flux/true = {res["ratio"]:.4f}, Δpos = ({res["dx"]:+.3f}, {res["dy"]:+.3f}) px', fontsize=10)
        ax[1, k].set_xlabel('cutout x [px]')
    fig.colorbar(c, ax=ax[1, :2], shrink=0.8, label='residual [same units as F = 1e6 total flux]')
    ax[1, 0].set_ylabel('cutout y [px]')

    m = ax[1, 2].imshow((rmap - 1) * 100, origin='lower', cmap='viridis',
                        extent=[pos[0] - 233, pos[-1] + 233] * 2)
    ax[1, 2].plot(XW, YW, 'r*', ms=14, mec='w')
    ax[1, 2].plot(81, 81, 'wx', ms=10, mew=2)
    for nd in (0.5, 682.5, 1365.5, 2047.5):
        ax[1, 2].axvline(nd, color='w', lw=0.4, alpha=0.5)
        ax[1, 2].axhline(nd, color='w', lw=0.4, alpha=0.5)
    ax[1, 2].set_title('before: fitted flux / true − 1 [%] vs detector position\n'
                       '(× = node the old fit used; ★ = star shown; lines = grid nodes)', fontsize=10)
    ax[1, 2].set_xlabel('detector x [px]')
    ax[1, 2].set_ylabel('detector y [px]')
    fig.colorbar(m, ax=ax[1, 2], shrink=0.8, label='%')
    fig.suptitle('#1055 theoretical worst case: noise-free star, NRCB5 F480M STPSF grid '
                 '(nircam_nrcb5_f480m_fovp1024_samp2_npsf16)', fontsize=11)
    fig.savefig('fig_theory_corner.png', dpi=110, bbox_inches='tight')
    print('after-fix ratio', good['ratio'], 'before', bad['ratio'])
    print('map range', rmap.min(), rmap.max())
    np.save('rmap.npy', rmap)
