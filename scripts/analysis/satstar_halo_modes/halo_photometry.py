"""How does saturated-star photometry change when the halo is allowed to vary
per exposure?  (#1013 follow-up)

    python halo_photometry.py <cutout dir> <psf dir> <out prefix> star [star ...]

For every exposure of each star (``<cutout dir>/<star>_e<k>.npz`` from
``satstar_poisson/cutout.py``; ``<psf dir>/psf_<star>_e<k>.npy`` an STPSF
``OVERDIST`` image, oversample 4, F480M, at that exposure's detector position)
the flux is fitted twice on the same pixels, weights and mask:

* ``S``: ``F P + B`` -- what a masked-core PSF fit does (production
  ``get_saturated_stars``);
* ``H``: ``F P + F sum_k c_k b_k(r) Pbar(r) + B``, free log-r rescalings of
  the model's smooth halo ``Pbar`` (``jwst_gc_pipeline.photometry.satstar_halo_modes``),
  so inside the halo range the flux is anchored by the diffraction spikes only.

Each star is constant, so the right method gives the same flux in every
dither: the per-star dither scatter of F is the figure of merit.  The fit radius
``rmax`` and the knots are swept.  Outputs ``<out prefix>.json`` and
``<out prefix>.png``.

Pixels: DO_NOT_USE or SATURATED (DQ bits 0, 1), dilated by 3 px, are masked;
the error is sqrt(VAR_POISSON + VAR_RNOISE); field-star pixels are clipped at
5 sigma (3 iterations, identical for both models).
"""
import glob
import json
import os
import sys

import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from jwst_gc_pipeline.photometry.satstar_halo_modes import fit_flux_with_halo_modes, azimuthal_median, log_hats

OS = 4                     # STPSF oversampling of the OVERDIST image
HALF = 200                 # native half-size of the PSF stamp (401 px)
CONFIGS = {                # name: (rmax, knots)
    # halo free over the whole fit range: F is anchored by the spikes alone
    'r100_spk': (100., (15., 35., 60., 100.)),
    'r200_spk': (200., (15., 35., 80., 150., 200.)),
    # halo free to 80 / 150 px: the off-spike pixels beyond also anchor F
    'r100': (100., (15., 35., 80.)),
    'r200': (200., (15., 35., 80., 150.)),
}


def native_psf(P4, xt, yt, shape):
    """Pixel-integrated native-pixel PSF on a ``shape`` grid for a star at
    (xt, yt): shift the oversampled image by the sub-pixel phase, then sum
    OS x OS blocks.  Unit total flux (the stamp's own sum)."""
    ix, iy = int(np.floor(xt + 0.5)), int(np.floor(yt + 0.5))
    fx, fy = xt - ix, yt - iy
    S = ndimage.shift(P4.astype(float), (OS * fy, OS * fx), order=1, mode='constant')
    n = 2 * HALF + 1
    c = (P4.shape[0] - OS * n) // 2
    S = S[c:c + OS * n, c:c + OS * n].reshape(n, OS, n, OS).sum((1, 3))
    S /= S.sum()
    out = np.zeros(shape)
    y0, x0 = iy - HALF, ix - HALF
    ys, xs = slice(max(y0, 0), min(y0 + n, shape[0])), slice(max(x0, 0), min(x0 + n, shape[1]))
    out[ys, xs] = S[ys.start - y0:ys.stop - y0, xs.start - x0:xs.stop - x0]
    return out


def fit_exposure(fn, psffn):
    z = np.load(fn)
    d, dq, var = z['d'], z['dq'].astype(np.int64), z['var']
    xt, yt = float(z['xt']), float(z['yt'])
    P = native_psf(np.load(psffn), xt, yt, d.shape)
    bad = ndimage.binary_dilation((dq & 3) > 0, iterations=3) | ~np.isfinite(d) | ~np.isfinite(var)
    err = np.sqrt(np.where(var > 0, var, np.nan))
    out = dict(x=int(z['X0']) + xt, y=int(z['Y0']) + yt, nsat=int(((dq & 2) > 0).sum()))
    yy, xx = np.indices(d.shape)
    r = np.hypot(xx - xt, yy - yt)
    pbar = azimuthal_median(P, r)
    for name, (rmax, knots) in CONFIGS.items():
        s = fit_flux_with_halo_modes(d, err, bad, P, xt, yt, knots=None, rmax=rmax)
        h = fit_flux_with_halo_modes(d, err, bad, P, xt, yt, knots=knots, rmax=rmax)
        # light in the fitted halo change, as a fraction of F: sum_k c_k sum Pbar b_k over the
        # FITTED pixels only.  A hat lying mostly under the masked core is constrained by its
        # few unmasked pixels; integrating it over the masked annulus too would extrapolate.
        hats = log_hats(r, knots)
        used = ~bad & (r <= rmax)
        halo_light = float(np.sum([h.halo[k] * np.sum((pbar * hats[..., k])[used]) for k in range(len(knots))
                                   if np.isfinite(h.halo[k])]))
        # how much of the star's (model) light the halo range holds
        frac_range = float(np.sum(P * (hats.sum(-1) > 0)))
        out[name] = dict(F_S=s.flux, F_S_err=s.flux_err, chi2_S=s.chi2_dof, F_H=h.flux, F_H_err=h.flux_err,
                         chi2_H=h.chi2_dof, c=h.halo.tolist(), c_err=h.halo_err.tolist(), nspike=h.nspike,
                         npix=h.npix, halo_light=halo_light, frac_halo_range=frac_range)
    return out


def main():
    cutdir, psfdir, outp = sys.argv[1:4]
    stars = sys.argv[4:]
    res = {}
    for st in stars:
        res[st] = {}
        for fn in sorted(glob.glob(f'{cutdir}/{st}_e*.npz')):
            e = os.path.basename(fn)[len(st) + 2:-4]
            pf = f'{psfdir}/psf_{st}_e{e}.npy'
            if not os.path.exists(pf):
                print('no psf for', fn, flush=True)
                continue
            res[st][e] = fit_exposure(fn, pf)
            print(st, e, {k: (round(v['F_H'] / v['F_S'], 4), [round(c, 3) for c in v['c']])
                          for k, v in res[st][e].items() if isinstance(v, dict)}, flush=True)
    summ = {}
    for st, ex in res.items():
        summ[st] = {}
        for name in CONFIGS:
            FS = np.array([v[name]['F_S'] for v in ex.values()])
            FH = np.array([v[name]['F_H'] for v in ex.values()])
            summ[st][name] = dict(scatter_S=float(np.std(FS / FS.mean())), scatter_H=float(np.std(FH / FH.mean())),
                                  mean_H_over_S=float(np.mean(FH / FS)),
                                  halo_light=[ex[e][name]['halo_light'] for e in ex])
    json.dump(dict(per_exposure=res, summary=summ, configs={k: [v[0], list(v[1])] for k, v in CONFIGS.items()}),
              open(outp + '.json', 'w'), separators=(',', ':'))
    for st, v in summ.items():
        print(st, {k: {kk: (round(vv, 4) if isinstance(vv, float) else [round(x, 4) for x in vv]) for kk, vv in s.items()}
                   for k, s in v.items()})
    plot(res, summ, outp)


def replot(outp):
    J = json.load(open(outp + '.json'))
    plot(J['per_exposure'], J['summary'], outp)


def plot(res, summ, outp):
    fig, ax = plt.subplots(1, 4, figsize=(24, 5.6))
    sts = list(res)
    for i, st in enumerate(sts):
        ex = res[st]
        es = sorted(ex, key=int)
        x = np.array([int(e) for e in es]) + 0.06 * i
        FS = np.array([ex[e]['r200']['F_S'] for e in es])
        FH = np.array([ex[e]['r200']['F_H'] for e in es])
        ax[0].plot(x, FS / FS.mean(), 'o-', color=f'C{i}', ms=4, label=st)
        ax[1].plot(x, FH / FH.mean(), 's-', color=f'C{i}', ms=4, label=st)
        # absolute level vs fit radius, normalised to the halo-mode flux at r <= 200
        ref = FH.mean()
        for name, xo in (('r100', 0), ('r200', 1)):
            s = np.mean([ex[e][name]['F_S'] for e in es]) / ref
            h = np.mean([ex[e][name]['F_H'] for e in es]) / ref
            ax[3].plot([xo + 0.03 * i], [s], 'o', color=f'C{i}', mfc='none', ms=7)
            ax[3].plot([xo + 0.03 * i], [h], 's', color=f'C{i}', ms=6)
    for a, t in ((ax[0], 'standard F·P + B (masked core), r ≤ 200 px'),
                 (ax[1], 'halo modes: F·P + F·Σc_k b_k(r)·P̄(r) + B, r ≤ 200 px')):
        a.axhline(1, color='k', lw=0.5); a.set_ylim(0.8, 1.2); a.set_xlabel('dither'); a.set_title(t)
        a.set_ylabel("fitted flux / the star's mean"); a.legend(fontsize=7, ncol=2)
    xx = np.arange(len(sts))
    for j, name in enumerate(('r100', 'r200')):
        ax[2].bar(xx + 0.4 * j - 0.3, [summ[s][name]['scatter_S'] * 100 for s in sts], 0.18, color=f'C{j}',
                  alpha=0.4, label=f'standard, r ≤ {name[1:]} px')
        ax[2].bar(xx + 0.4 * j - 0.1, [summ[s][name]['scatter_H'] * 100 for s in sts], 0.18, color=f'C{j}',
                  label=f'halo modes, r ≤ {name[1:]} px')
    ax[2].set_xticks(xx); ax[2].set_xticklabels(sts, rotation=30, fontsize=8)
    ax[2].set_ylabel('rms of flux across dithers [%]'); ax[2].legend(fontsize=8)
    ax[2].set_title('dither-to-dither scatter (constant star: 0 is right)')
    ax[3].plot([], [], 'ko', mfc='none', label='standard'); ax[3].plot([], [], 'ks', label='halo modes')
    ax[3].axhline(1, color='k', lw=0.5); ax[3].set_xticks([0.1, 1.1]); ax[3].set_xticklabels(['r ≤ 100 px', 'r ≤ 200 px'])
    ax[3].set_xlim(-0.4, 1.6); ax[3].set_ylabel('mean flux / halo-mode flux at r ≤ 200 px'); ax[3].legend(fontsize=8)
    ax[3].set_title('absolute flux vs fit radius (one colour per star)')
    fig.tight_layout(); fig.savefig(outp + '.png', dpi=int(os.environ.get('FIG_DPI', 150)))


if __name__ == '__main__':
    if sys.argv[1] == '--replot':
        replot(sys.argv[2])
    else:
        main()
