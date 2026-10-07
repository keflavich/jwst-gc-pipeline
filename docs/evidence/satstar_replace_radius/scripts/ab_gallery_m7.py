"""Stars the final catalog of variant B keeps and variant A leaves in its residual.

For each (field, band, seed): every m7 vetted source of variant B inside the
evaluated inner box with no m7 vetted source of variant A within 1 mosaic
pixel.  The PSF-matched S/N left at its position in each variant's final
(m7) residual is measured with ``phase_loss.stamp_metrics``; the rows with
the largest S/N left in A are shown.  Both variants differ from m3 on, so
the m6 residual is not shared (unlike ``m7_companion_ratio/ab_gallery.py``).

Per row: data mosaic | A m7 residual + A m7 vetted | B m7 residual + B m7
vetted.  Red circle: the star; orange x: that column's m7 vetted sources;
yellow star: saturated-star fits (the band's consolidated satstar catalog of
the column's variant; data column: variant B); magenta diamond: the
brightest B m7 vetted source within 2.5 FWHM that is brighter than the star;
cyan square: injected stars (seeds >= 1).  All panels of a row share one
linear stretch width, set by the star's peak above the local median in the
data panel (floored at 3x the stamp's MAD sigma); each panel is centred on
its own local median.

usage: python ab_gallery_m7.py <out.png> <varA> <varB> <field:band:seed[:n[:sel]]> ...
       n rows per spec (default 3; 0 = counts only, no figure).  sel:
       'inj' keeps injected stars only, 'sat' keeps stars within
       ``SAT_R`` arcsec of a variant-B satstar, 'all' (default) keeps all.
       Writes <out>.json with the per-spec counts and the per-row numbers.
"""
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy import units as u  # noqa: E402
from astropy.coordinates import SkyCoord  # noqa: E402
from astropy.stats import mad_std  # noqa: E402
from astropy.table import Table  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, '..', '..', 'm7_companion_ratio', 'scripts'))
from jwst_gc_pipeline.photometry import reference_fields as RF  # noqa: E402
from phase_loss import FWHM_PIX, Mosaic, phase_files, read_cat, stamp_metrics, _in_box  # noqa: E402

M7 = 'resbgsub_m7'
COMP = 2.5
SAT_R = 0.5


def stamp(mos, x, y, half):
    xi, yi = int(round(x)), int(round(y))
    ny, nx = mos.shape
    out = np.full((2 * half + 1, 2 * half + 1), np.nan)
    y0, y1 = max(yi - half, 0), min(yi + half + 1, ny)
    x0, x1 = max(xi - half, 0), min(xi + half + 1, nx)
    out[y0 - (yi - half):y1 - (yi - half), x0 - (xi - half):x1 - (xi - half)] = mos.data[y0:y1, x0:x1]
    return out, xi - half, yi - half


def _files(spec, variant, seed, filt):
    d = RF.run_dir(spec, variant, seed)
    f = filt.lower()
    hits = glob.glob(f'{d}/catalogs/{f}_merged*_indivexp_merged_{M7}_dao_basic_vetted.fits')
    if len(hits) != 1:
        raise FileNotFoundError(f'{d} {filt}: {hits}')
    obstok = re.match(rf'{f}_merged(.*)_indivexp', os.path.basename(hits[0])).group(1)
    out = phase_files(f'{d}/catalogs', f'{d}/{filt}/pipeline', filt, 'merged', obstok)
    sat = glob.glob(f'{d}/catalogs/{f}_consolidated_satstar_catalog.fits')
    out['satstar'] = sat[0] if sat else None
    return out


def _satstar_xy(path, mos):
    if path is None:
        return np.zeros((0, 2))
    t = Table.read(path)
    if not len(t):
        return np.zeros((0, 2))
    x, y = mos.xy(SkyCoord(t['skycoord_fit']))
    return np.c_[x, y]


def candidates(spec, name, filt, seed, var_a, var_b):
    fa, fb = _files(spec, var_a, seed, filt), _files(spec, var_b, seed, filt)
    data = Mosaic(fa['data'])
    va, vb = read_cat(fa[M7]['vetted'], data), read_cat(fb[M7]['vetted'], data)
    half_as = spec['size_arcsec'] / 2 - spec['inner_margin_arcsec']
    inner = _in_box(SkyCoord(vb['skycoord']), (spec['ra'], spec['dec'], half_as))
    dd, _ = cKDTree(np.c_[va['_x'], va['_y']]).query(np.c_[vb['_x'], vb['_y']])
    new = np.flatnonzero(inner & (dd > 1.0))
    fw = FWHM_PIX[filt]
    x, y = np.asarray(vb['_x'])[new], np.asarray(vb['_y'])[new]
    out = dict(x=x, y=y, flux=np.asarray(vb['flux'], float)[new])
    for key, path in (('a', fa[M7]['resid']), ('b', fb[M7]['resid'])):
        m = Mosaic(path)
        out[f'snr_{key}'] = stamp_metrics(m, x, y, fw)['snr']
        m.close()
    out['snr_data'] = stamp_metrics(data, x, y, fw)['snr']
    tb = cKDTree(np.c_[vb['_x'], vb['_y']])
    fl = np.asarray(vb['flux'], float)
    nb_x, nb_y, ratio = (np.full(len(new), np.nan) for _ in range(3))
    for k, i in enumerate(new):
        js = [j for j in tb.query_ball_point([vb['_x'][i], vb['_y'][i]], COMP * fw) if fl[j] > fl[i]]
        if js:
            j = js[int(np.argmax(fl[js]))]
            nb_x[k], nb_y[k], ratio[k] = vb['_x'][j], vb['_y'][j], fl[i] / fl[j]
    out.update(nb_x=nb_x, nb_y=nb_y, ratio=ratio)
    sat_a, sat_b = _satstar_xy(fa['satstar'], data), _satstar_xy(fb['satstar'], data)
    if len(sat_b):
        ds, _ = cKDTree(sat_b).query(np.c_[x, y])
        out['sat_dist_as'] = ds * data.pix_as
    else:
        out['sat_dist_as'] = np.full(len(new), np.inf)
    inj = None
    if seed > 0:
        t = Table.read(RF.injection_table_path(name, seed))
        isc = SkyCoord(np.asarray(t['ra']) * u.deg, np.asarray(t['dec']) * u.deg)
        inj = np.c_[data.xy(isc)]
        di, _ = cKDTree(inj).query(np.c_[x, y])
        out['injected'] = di <= 1.0
    else:
        out['injected'] = np.zeros(len(new), bool)
    return dict(fa=fa, fb=fb, data=data, va=va, vb=vb, inj=inj, sat_a=sat_a, sat_b=sat_b,
                n_inner=int(inner.sum()), **out)


def main(outpng, var_a, var_b, *specs, half_arcsec=0.5):
    _, fields = RF.load_config()
    rows, summary = [], []
    for s in specs:
        parts = s.split(':')
        name, filt, seed = parts[0], parts[1].upper(), int(parts[2])
        n = int(parts[3]) if len(parts) > 3 else 3
        sel = parts[4] if len(parts) > 4 else 'all'
        c = candidates(fields[name], name, filt, seed, var_a, var_b)
        order = np.argsort(-np.nan_to_num(c['snr_a'], nan=-np.inf))
        if sel == 'inj':
            order = order[c['injected'][order]]
        elif sel == 'sat':
            order = order[c['sat_dist_as'][order] <= SAT_R]
        left_a = np.isfinite(c['snr_a']) & (c['snr_a'] >= 5)
        left_b = np.isfinite(c['snr_b']) & (c['snr_b'] >= 5)
        near = c['sat_dist_as'] <= SAT_R
        summary.append(dict(field=name, filt=filt, seed=seed, n_inner_b=c['n_inner'],
                            n_b_only=int(len(c['x'])),
                            n_left_in_a_snr5=int(left_a.sum()), n_left_in_b_snr5=int(left_b.sum()),
                            n_near_satstar=int(near.sum()),
                            n_near_satstar_left_in_a_snr5=int((near & left_a).sum()),
                            n_injected=int(c['injected'].sum()),
                            n_brighter_nb=int(np.isfinite(c['ratio']).sum())))
        for k in order[:n]:
            rows.append((name, filt, seed, c, k))
    if rows:
        _figure(outpng, var_a, var_b, rows, half_arcsec)
    json.dump(dict(var_a=var_a, var_b=var_b, summary=summary,
                   rows=[dict(field=nm, filt=fl, seed=sd, x=float(c['x'][k]), y=float(c['y'][k]),
                              ratio=float(c['ratio'][k]), injected=bool(c['injected'][k]),
                              sat_dist_as=float(c['sat_dist_as'][k]),
                              snr_data=float(c['snr_data'][k]),
                              snr_a=float(c['snr_a'][k]), snr_b=float(c['snr_b'][k]))
                         for nm, fl, sd, c, k in rows]),
              open(os.path.splitext(outpng)[0] + '.json', 'w'), indent=1)
    for row in summary:
        print(row)


def _figure(outpng, var_a, var_b, rows, half_arcsec):
    fig, axes = plt.subplots(len(rows), 3, figsize=(7.4, 2.5 * len(rows)), squeeze=False)
    for r, (name, filt, seed, c, k) in enumerate(rows):
        data = c['data']
        half = int(round(half_arcsec / data.pix_as))
        x, y = c['x'][k], c['y'][k]
        ds, x0, y0 = stamp(data, x, y, half)
        yy, xx = np.mgrid[0:ds.shape[0], 0:ds.shape[1]]
        core = np.hypot(xx - (x - x0), yy - (y - y0)) <= 2
        # stretch on the star's core contrast, floored at 3 sigma of the stamp so
        # a position with no star in the data (a fake) is not stretched on noise
        span = max(np.nanmax(ds[core]) - np.nanmedian(ds), 3 * mad_std(ds, ignore_nan=True), 1e-6)
        panels = [('data', data, None, c['sat_b']),
                  (f'{var_a} m7 resid + cat', c['fa'][M7]['resid'], c['va'], c['sat_a']),
                  (f'{var_b} m7 resid + cat', c['fb'][M7]['resid'], c['vb'], c['sat_b'])]
        for j, (title, src, cat, sat) in enumerate(panels):
            ax = axes[r, j]
            m = src if isinstance(src, Mosaic) else Mosaic(src)
            st, _, _ = stamp(m, x, y, half)
            bg = np.nanmedian(st)
            ax.imshow(st, origin='lower', cmap='gray', vmin=bg - 0.5 * span, vmax=bg + 1.2 * span,
                      interpolation='nearest')
            ax.add_patch(plt.Circle((x - x0, y - y0), 3.5, fill=False, color='r', lw=1.2))
            if cat is not None:
                sel = (np.abs(cat['_x'] - x) < half) & (np.abs(cat['_y'] - y) < half)
                ax.plot(cat['_x'][sel] - x0, cat['_y'][sel] - y0, 'x', color='orange', ms=5, mew=1.0)
            if len(sat):
                sel = (np.abs(sat[:, 0] - x) < half) & (np.abs(sat[:, 1] - y) < half)
                ax.plot(sat[sel, 0] - x0, sat[sel, 1] - y0, '*', mfc='none', mec='yellow', ms=13, mew=1.2)
            if c['inj'] is not None:
                sel = (np.abs(c['inj'][:, 0] - x) < half) & (np.abs(c['inj'][:, 1] - y) < half)
                ax.plot(c['inj'][sel, 0] - x0, c['inj'][sel, 1] - y0, 's', mfc='none', mec='cyan',
                        ms=9, mew=1.0)
            if np.isfinite(c['nb_x'][k]):
                ax.plot(c['nb_x'][k] - x0, c['nb_y'][k] - y0, 'D', mfc='none', mec='magenta', ms=8, mew=1.2)
            ax.set_xlim(-0.5, 2 * half + 0.5)
            ax.set_ylim(-0.5, 2 * half + 0.5)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title, fontsize=8)
            if m is not data:
                m.close()
        ratio = c['ratio'][k]
        nbtxt = f"flux/neighbour {ratio:.2f}" if np.isfinite(ratio) else "no brighter neighbour"
        sd = c['sat_dist_as'][k]
        sattxt = f"; satstar {sd:.2f}\"" if np.isfinite(sd) and sd <= 1.0 else ""
        axes[r, 0].set_ylabel(f"{name} {filt} seed {seed}\n" + nbtxt + sattxt
                              + (" (injected)" if c['injected'][k] else "")
                              + f"\nresid S/N {var_a} {c['snr_a'][k]:.0f} -> {var_b} {c['snr_b'][k]:.0f}",
                              fontsize=7)
    fig.tight_layout()
    fig.savefig(outpng, dpi=110)
    plt.close(fig)


if __name__ == '__main__':
    main(*sys.argv[1:])
