"""Grouped vs ungrouped m7 fits on the reference fields, by neighbour separation.

For every run (field, seed 0-10) of two variants and both filters of each
field, scores each m7 vetted source inside the inner box with the
reference-field over-subtraction test (3x3 px minimum of the residual
matched-filter S/N map below -oversub_snr, evaluate.oversubtracted_mask) and
its nearest-neighbour separation in the same catalogue (FWHM units).  Pools
the seeds per (field, filter, variant).

Writes <out>.json and <out>.md (counts and fractions per separation bin) and
<out>_gallery_<field>_<filt>.png: data | A residual | B residual at random
A sources that are over-subtracted and have a neighbour within 2 FWHM, with
each variant's own catalogue marked.

usage: python ref_nn_oversub.py <variantA> <variantB> <out-prefix>
"""
import json
import os
import sys

import numpy as np
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from jwst_gc_pipeline.photometry import reference_fields as RF  # noqa: E402
from jwst_gc_pipeline.photometry.reference_fields import evaluate as EV  # noqa: E402

BINS = [0, 0.75, 1.25, 1.5, 2.0, 3.0, 5.0, np.inf]
SEEDS = range(11)
HALF = 8


def score_run(spec, variant, seed, filt):
    rdir = RF.run_dir(spec, variant, seed)
    prod = EV.find_products(rdir, filt, allow_missing_bg=True)
    res, err, wcs, _ = EV._image(prod['residual'])
    if prod['smoothed_bg'] is not None:
        res = res - EV._image(prod['smoothed_bg'])[0]
    fr = EV.Frame(spec, wcs, res.shape)
    fw = EV.fwhm_pix(filt)
    snr, _ = EV.matched_filter_snr(res, err, fw, fr.mask())
    sc, flux, _ = EV.load_catalog(prod['catalog'])
    x, y = fr.xy(sc)
    ok = np.isfinite(x) & np.isfinite(y)
    nn = np.full(len(x), np.inf)
    if ok.sum() > 1:
        d, _ = cKDTree(np.c_[x[ok], y[ok]]).query(np.c_[x[ok], y[ok]], k=2)
        nn[ok] = d[:, 1] / fw
    inside = fr.inside(x, y) & ok
    osub = EV.oversubtracted_mask(snr, x, y, thresh=spec['oversub_snr'])
    dimg, _, dwcs, _ = EV._image(prod['data'])
    return dict(x=x, y=y, sc=sc, nn=nn, inside=inside, osub=osub, snr=snr, res=res,
                data=dimg, dwcs=dwcs, wcs=wcs, area=fr.area_as2, fwhm=fw)


def tally(runs):
    nn = np.concatenate([r['nn'][r['inside']] for r in runs])
    os_ = np.concatenate([r['osub'][r['inside']] for r in runs])
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        m = (nn >= lo) & (nn < hi)
        rows.append(dict(lo=lo, hi=hi, n=int(m.sum()), n_osub=int(os_[m].sum())))
    return dict(n=int(len(nn)), n_osub=int(os_.sum()), area_as2=float(sum(r['area'] for r in runs)),
                n_nn_lt_2=int(np.sum(nn < 2.0)), n_osub_nn_lt_2=int(np.sum(os_ & (nn < 2.0))),
                bins=rows)


def stamp(img, x, y):
    out = np.full((2 * HALF + 1, 2 * HALF + 1), np.nan)
    xi, yi = int(round(x)), int(round(y))
    ny, nx = img.shape
    y0, y1 = max(yi - HALF, 0), min(yi + HALF + 1, ny)
    x0, x1 = max(xi - HALF, 0), min(xi + HALF + 1, nx)
    if y1 > y0 and x1 > x0:
        out[y0 - (yi - HALF):y1 - (yi - HALF), x0 - (xi - HALF):x1 - (xi - HALF)] = img[y0:y1, x0:x1]
    return out


def gallery(runsA, runsB, va, vb, title, path, ncol=8, rng=None):
    rng = rng or np.random.default_rng(3)
    cand = [(k, i) for k, r in enumerate(runsA)
            for i in np.flatnonzero(r['inside'] & r['osub'] & (r['nn'] < 2.0))]
    if not cand:
        return 0
    pick = [cand[j] for j in rng.choice(len(cand), size=min(ncol, len(cand)), replace=False)]
    fig, axs = plt.subplots(3, len(pick), figsize=(1.8 * len(pick), 6.0), squeeze=False)
    for c, (k, i) in enumerate(pick):
        A, B = runsA[k], runsB[k]
        xa, ya = A['x'][i], A['y'][i]
        xb, yb = (float(v) for v in B['wcs'].world_to_pixel(A['sc'][i]))
        xd, yd = (float(v) for v in A['dwcs'].world_to_pixel(A['sc'][i]))
        d = stamp(A['data'], xd, yd)
        med = np.nanmedian(d)
        w = max(np.nanmax(d[HALF - 2:HALF + 3, HALF - 2:HALF + 3]) - med, 1e-6)
        panels = ((d, None, None, 'data'), (stamp(A['res'], xa, ya), A, (xa, ya), f'{va} resid'),
                  (stamp(B['res'], xb, yb), B, (xb, yb), f'{vb} resid'))
        for rr, (st, R, cen, lab) in enumerate(panels):
            ax = axs[rr, c]
            ax.imshow(st, origin='lower', cmap='gray', vmin=np.nanmedian(st) - 0.5 * w,
                      vmax=np.nanmedian(st) + w, interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])
            if R is not None:
                cx, cy = cen
                near = (np.abs(R['x'] - cx) <= HALF) & (np.abs(R['y'] - cy) <= HALF)
                ax.plot(R['x'][near] - round(cx) + HALF, R['y'][near] - round(cy) + HALF,
                        'x', color='orange', ms=5, mew=1.2)
                mn = np.nanmin(stamp(R['snr'], cx, cy)[HALF - 1:HALF + 2, HALF - 1:HALF + 2])
                ax.text(0.03, 0.03, f'min S/N {mn:.0f}', transform=ax.transAxes, color='yellow', fontsize=7)
            if c == 0:
                ax.set_ylabel(lab, fontsize=8)
        axs[0, c].set_title(f's{k} nn {A["nn"][i]:.2f}', fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return len(cand)


def main(va, vb, outp):
    _, fields = RF.load_config()
    out = {}
    md = ['| field filter | variant | n src | n osub | osub / as2 | n NN<2 FWHM | osub NN<2 | '
          + ' | '.join(f'{lo:g}-{hi:g}' for lo, hi in zip(BINS[:-1], BINS[1:])) + ' |',
          '|' + '---|' * (7 + len(BINS) - 1)]
    for name, spec in fields.items():
        for filt in spec['filters']:
            runs = {}
            for v in (va, vb):
                runs[v] = []
                for s in SEEDS:
                    try:
                        runs[v].append(score_run(spec, v, s, filt))
                    except FileNotFoundError as e:
                        print('missing', name, filt, v, s, e, flush=True)
            n = min(len(runs[va]), len(runs[vb]))
            key = f'{name}_{filt}'
            out[key] = {}
            for v in (va, vb):
                t = tally(runs[v][:n])
                out[key][v] = t
                cells = ' | '.join(f'{b["n_osub"]}/{b["n"]}' for b in t['bins'])
                md.append(f'| {name} {filt} | {v} | {t["n"]} | {t["n_osub"]} | '
                          f'{t["n_osub"] / t["area_as2"]:.2f} | {t["n_nn_lt_2"]} | {t["n_osub_nn_lt_2"]} | {cells} |')
            out[key]['n_runs'] = n
            ncand = gallery(runs[va][:n], runs[vb][:n], va, vb,
                            f'{name} {filt}: random {va} m7 sources over-subtracted (3x3 min S/N < '
                            f'-{spec["oversub_snr"]:g}) with a neighbour < 2 FWHM;\n'
                            f'same sky position in {vb}; orange x = that variant\'s m7 vetted sources',
                            f'{outp}_gallery_{name}_{filt.lower()}.png')
            out[key]['n_gallery_candidates'] = ncand
            print(key, json.dumps(out[key]), flush=True)
    json.dump(out, open(outp + '.json', 'w'), indent=1, default=float)
    open(outp + '.md', 'w').write('\n'.join(md) + '\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main(*sys.argv[1:4])
