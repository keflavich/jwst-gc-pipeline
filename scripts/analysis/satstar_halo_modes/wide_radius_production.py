"""Would a WIDE-radius halo-mode fit improve production fluxes of heavily
saturated stars (#1013 / #1023 follow-up)?

    python wide_radius_production.py fit <catalog dir> <psf grid fits> <out npz> <cal key>...
    python wide_radius_production.py analyze <out npz>... <out prefix>

The production hook (``SATSTAR_HALO_MODES``) is confined to the photutils box
(r <= 40.5 px at pad=81), where the halo modes do not help
(``production_radius.py``).  The 7-star evidence showed a gain only out to
r ~ 200 px.  This measures that on PRODUCTION catalogs: for every row of the
``<root>_satstar_catalog.fits`` in ``<catalog dir>`` (from
``satstar_perexp_halo/production_flux_vs_x.py``) it cuts a 2*RMAX+1 stamp of
the ``_cal`` frame around (x_0, y_0) -- the fitted DETECTOR position;
``x_fit``/``y_fit`` are local to the fit box -- evaluates the production PSF grid
(``to_griddedpsfmodel``) there at unit flux, masks DQ DO_NOT_USE/SATURATED
dilated by 3 px, and fits ``F P + B`` and the halo modes
(``satstar_halo_knots(r_core, RMAX)``, r_core = sqrt(sat_area/pi) + 3) on the
same pixels.  Other stars are left to the MAD clipping of the fit, as in the
hook.

``analyze`` links rows between the dithers of a star-visit by
``skycoord_fit`` (0.3") and compares the per-star dither scatter of
``flux_fit`` (production), ``F_S`` and ``F_H`` (wide radius), binned by
sat_area.
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from jwst_gc_pipeline.photometry.satstar_halo_modes import fit_flux_with_halo_modes, satstar_halo_knots

BASE = 'https://stpubdata.s3.amazonaws.com/'
RMAX = float(os.environ.get('WIDE_RMAX', 200))
AREA_MIN = 50
LINK_ARCSEC = 0.3
AREA_BINS = (50, 300, 1000, 3000, 1e6)
BAND, NULL_BAND = (250, 550), (1300, 1600)    # the NRCBLONG column deficit (#1013) and a control


def fit(catdir, gridfn, outfn, keys):
    from jwst_gc_pipeline.reduction.saturated_star_finding import to_griddedpsfmodel
    grid = to_griddedpsfmodel(gridfn)
    if isinstance(grid, list):
        grid = grid[0]
    H = int(RMAX) + 2
    out = []
    for key in keys:
        root = os.path.basename(key).replace('.fits', '')
        cat = os.path.join(catdir, root + '_satstar_catalog.fits')
        if not os.path.exists(cat):
            continue
        t = Table.read(cat)
        local = os.path.exists(key)
        fn = key if local else os.path.join(os.path.dirname(outfn), os.path.basename(key))
        if not local:
            subprocess.run(['curl', '-sS', '--retry', '4', '-o', fn, BASE + key], check=True)
        try:
            with fits.open(fn) as h:
                d = h['SCI'].data.astype(float)
                dq = h['DQ'].data.astype(np.int64)
                var = (h['VAR_POISSON'].data + h['VAR_RNOISE'].data).astype(float)
        finally:
            if not local:
                os.remove(fn)
        bad_full = ndimage.binary_dilation((dq & 3) > 0, iterations=3) | ~np.isfinite(d) | ~(var > 0)
        for i in range(len(t)):
            area = int(t['sat_area'][i]) if np.isfinite(t['sat_area'][i]) else 0
            xf, yf, ff = float(t['x_0'][i]), float(t['y_0'][i]), float(t['flux_fit'][i])
            if area < AREA_MIN or not (np.isfinite(xf) and np.isfinite(yf) and ff > 0):
                continue
            ix, iy = int(round(xf)), int(round(yf))
            y1, y2, x1, x2 = max(iy - H, 0), min(iy + H + 1, 2048), max(ix - H, 0), min(ix + H + 1, 2048)
            yy, xx = np.mgrid[y1:y2, x1:x2]
            P = grid.evaluate(xx, yy, 1.0, xf, yf)
            s = P.sum()
            if not s > 0:
                continue
            sl = (slice(y1, y2), slice(x1, x2))
            err = np.sqrt(np.where(var[sl] > 0, var[sl], np.nan))
            r_core = float(np.sqrt(area / np.pi)) + 3.0
            knots = satstar_halo_knots(r_core, RMAX)
            fs = fit_flux_with_halo_modes(d[sl], err, bad_full[sl], P, xf - x1, yf - y1, knots=None, rmax=RMAX)
            fh = (fit_flux_with_halo_modes(d[sl], err, bad_full[sl], P, xf - x1, yf - y1, knots=knots, rmax=RMAX)
                  if knots is not None else None)
            sc = t['skycoord_fit'][i]
            out.append((root[:19], int(root[20:25]), xf, yf, area, ff, fs.flux, fh.flux if fh is not None else np.nan,
                        float(sc.ra.deg), float(sc.dec.deg)))
        print('done', root, len(out), flush=True)
    a = np.array(out, dtype=object)
    np.savez(outfn, visit=a[:, 0].astype(str), exp=a[:, 1].astype(int), x=a[:, 2].astype(float),
             y=a[:, 3].astype(float), area=a[:, 4].astype(int), flux_fit=a[:, 5].astype(float),
             F_S=a[:, 6].astype(float), F_H=a[:, 7].astype(float), ra=a[:, 8].astype(float),
             dec=a[:, 9].astype(float), rmax=RMAX)


def analyze(infns, outp):
    z = [np.load(f) for f in infns]
    cols = {k: np.concatenate([q[k] for q in z]) for k in ('visit', 'exp', 'x', 'area', 'flux_fit', 'F_S', 'F_H', 'ra', 'dec')}
    n = cols['ra'].size
    stars = []
    for v in np.unique(cols['visit']):
        idx = np.flatnonzero(cols['visit'] == v)
        cl, cra, cdec = [], [], []
        for i in idx:
            j = None
            if cl:
                dra = (np.array(cra) - cols['ra'][i]) * np.cos(np.deg2rad(cols['dec'][i]))
                sep = 3600 * np.hypot(dra, np.array(cdec) - cols['dec'][i])
                for k in np.argsort(sep):
                    if sep[k] > LINK_ARCSEC:
                        break
                    if cols['exp'][i] not in {cols['exp'][q] for q in cl[k]}:
                        j = k
                        break
            if j is None:
                cl.append([i]); cra.append(cols['ra'][i]); cdec.append(cols['dec'][i])
            else:
                cl[j].append(i)
        stars += [c for c in cl if len(c) >= 4]
    res = dict(rmax=float(z[0]['rmax']), n_rows=int(n), n_stars=len(stars), bins={})
    sc = {k: [] for k in ('flux_fit', 'F_S', 'F_H')}
    area_med, ratio = [], []
    for c in stars:
        c = np.array(c)
        ok = all(np.all(np.isfinite(cols[k][c])) and np.all(cols[k][c] > 0) for k in sc)
        if not ok:
            continue
        for k in sc:
            v = cols[k][c]
            sc[k].append(np.std(v / v.mean(), ddof=1))
        area_med.append(np.median(cols['area'][c]))
        ratio.append(np.median(cols['F_H'][c] / cols['flux_fit'][c]))
    sc = {k: np.array(v) for k, v in sc.items()}
    area_med, ratio = np.array(area_med), np.array(ratio)
    for lo, hi in zip(AREA_BINS[:-1], AREA_BINS[1:]):
        m = (area_med >= lo) & (area_med < hi)
        if m.sum() < 3:
            continue
        res['bins'][f'{lo:g}-{hi:g}'] = dict(
            n_stars=int(m.sum()),
            dither_rms_flux_fit=float(np.median(sc['flux_fit'][m])),
            dither_rms_F_S=float(np.median(sc['F_S'][m])),
            dither_rms_F_H=float(np.median(sc['F_H'][m])),
            frac_F_H_better_than_flux_fit=float(np.mean(sc['F_H'][m] < sc['flux_fit'][m])),
            F_H_over_flux_fit=float(np.median(ratio[m])))
    # does each flux carry the column deficit?  per star, median in band /
    # median outside (outside excludes both regions +-50 px), split at 300 px
    rng = np.random.default_rng(0)
    res['column'] = {}
    for lo, hi in ((AREA_MIN, 300), (300, 1e6)):
        for name, (a, b) in (('band', BAND), ('null', NULL_BAND)):
            q = {k: [] for k in sc}
            for c in stars:
                c = np.array(c); x = cols['x'][c]
                if not lo <= np.median(cols['area'][c]) < hi:
                    continue
                inb = (x > a) & (x < b)
                out = np.ones(x.size, bool)
                for A, B in (BAND, NULL_BAND):
                    out &= (x < A - 50) | (x > B + 50)
                if inb.sum() < 1 or out.sum() < 2:
                    continue
                for k in q:
                    f = cols[k][c]
                    if np.all(np.isfinite(f)) and np.all(f > 0):
                        q[k].append(np.median(f[inb]) / np.median(f[out]))
            res['column'][f'{name} area {lo:g}-{hi:g}' if hi < 1e6 else f'{name} area >={lo:g}'] = {
                k: dict(n_stars=len(v), ratio=float(np.median(v)) if v else None,
                        err=float(np.std([np.median(rng.choice(v, len(v))) for _ in range(1000)])) if v else None)
                for k, v in q.items()}
    json.dump(res, open(outp + '.json', 'w'), indent=1)
    print(json.dumps(res, indent=1))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    for k, mk in (('flux_fit', 'o'), ('F_S', 's'), ('F_H', '^')):
        ax[0].scatter(area_med, sc[k], s=8, marker=mk, label=k, alpha=0.7)
    ax[0].set_xscale('log'); ax[0].set_yscale('log')
    ax[0].set_xlabel('saturated pixels'); ax[0].set_ylabel('per-star dither rms')
    ax[0].legend(fontsize=8)
    ax[0].set_title(f'production flux_fit vs wide-radius fits (r ≤ {res["rmax"]:g} px)', fontsize=9)
    ax[1].scatter(area_med, ratio, s=8)
    ax[1].axhline(1, color='k', lw=0.5); ax[1].set_xscale('log')
    ax[1].set_xlabel('saturated pixels'); ax[1].set_ylabel('F_H (wide) / production flux_fit')
    labels = list(res['column'])
    for j, k in enumerate(('flux_fit', 'F_S', 'F_H')):
        v = [res['column'][L][k] for L in labels]
        ax[2].errorbar(np.arange(len(labels)) + 0.15 * (j - 1), [np.nan if q['ratio'] is None else q['ratio'] for q in v],
                       yerr=[np.nan if q['err'] is None else q['err'] for q in v], fmt='os^'[j], label=k)
    ax[2].axhline(1, color='k', lw=0.5); ax[2].set_xticks(range(len(labels)))
    ax[2].set_xticklabels([L.replace(' area ', '\n') + ' px' for L in labels], fontsize=8)
    ax[2].set_ylabel('in x-region / outside (per-star median)'); ax[2].legend(fontsize=8)
    ax[2].set_title(f'column deficit: band x = {BAND[0]}–{BAND[1]}, null x = {NULL_BAND[0]}–{NULL_BAND[1]}', fontsize=9)
    fig.tight_layout(); fig.savefig(outp + '.png', dpi=int(os.environ.get('FIG_DPI', 130)))


if __name__ == '__main__':
    if sys.argv[1] == 'fit':
        fit(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:])
    else:
        analyze(sys.argv[2:-1], sys.argv[-1])
