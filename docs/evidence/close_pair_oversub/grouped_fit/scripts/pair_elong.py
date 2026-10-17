"""Are the close pairs that grouped fitting adds real blends or split single stars?

Runs of seeds 0-10 (0 = clean; 1-10 add 24 injected stars each but hold the
same field stars, so the pairs largely repeat from seed to seed and the pooled
counts are not independent).  Takes variant-B m7 vetted pairs with mutual separation
< SEP_MAX FWHM and classifies each by how many of its two members variant A
also has within MATCH FWHM ('both', 'one' = B added a member, 'none').  For
each pair, measures the data stamp's Gaussian-windowed second moments about
the flux-weighted pair centroid, and the variance ratio along / across the
pair axis.  The null is the same ratio at isolated (NN > ISO FWHM) B sources
of similar flux, along a random axis.  A single star split in two shows the
null ratio; a real pair at separation d with flux ratio q is stretched along
its axis by q/(1+q)^2 d^2 (predicted ratio listed alongside).

usage: python pair_elong.py <variantA> <variantB> <out-prefix>
"""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from jwst_gc_pipeline.photometry import reference_fields as RF  # noqa: E402
from jwst_gc_pipeline.photometry.reference_fields import evaluate as EV  # noqa: E402

SEP_MAX = 1.25
MATCH = 0.4
ISO = 2.5
HALF = 8
SEEDS = range(11)


def load(spec, v, filt, seed):
    prod = EV.find_products(RF.run_dir(spec, v, seed), filt, allow_missing_bg=True)
    img, _, wcs, _ = EV._image(prod['data'])
    fr = EV.Frame(spec, wcs, img.shape)
    sc, flux, _ = EV.load_catalog(prod['catalog'])
    x, y = fr.xy(sc)
    ok = np.isfinite(x) & np.isfinite(y)
    return img, fr, x[ok], y[ok], np.asarray(flux, float)[ok]


def moments(img, xc, yc, sig_w, theta):
    """variance along / across axis theta in a Gaussian window; local
    background = median of the stamp's outer ring."""
    xi, yi = int(round(xc)), int(round(yc))
    ny, nx = img.shape
    if not (HALF <= xi < nx - HALF and HALF <= yi < ny - HALF):
        return np.nan
    st = img[yi - HALF:yi + HALF + 1, xi - HALF:xi + HALF + 1].astype(float)
    yy, xx = np.mgrid[-HALF:HALF + 1, -HALF:HALF + 1].astype(float)
    xx -= xc - xi
    yy -= yc - yi
    r2 = xx ** 2 + yy ** 2
    st = st - np.nanmedian(st[r2 > (HALF - 1.5) ** 2])
    w = np.exp(-0.5 * r2 / sig_w ** 2) * st
    if not np.isfinite(w).all() or w.sum() <= 0:
        return np.nan
    u = xx * np.cos(theta) + yy * np.sin(theta)
    p = -xx * np.sin(theta) + yy * np.cos(theta)
    vu = (w * u ** 2).sum() / w.sum()
    vp = (w * p ** 2).sum() / w.sum()
    return vu / vp if vp > 0 else np.nan


def main(va, vb, outp):
    _, fields = RF.load_config()
    rng = np.random.default_rng(0)
    out = {}
    panels = []
    for name, spec in fields.items():
        for filt in spec['filters']:
            fw = EV.fwhm_pix(filt)
            sig = fw / 2.355
            rows, null, nf = [], [], []
            for seed in SEEDS:
                img, fr, xa, ya, fa = load(spec, va, filt, seed)
                _, _, xb, yb, fb = load(spec, vb, filt, seed)
                ta, tb = cKDTree(np.c_[xa, ya]), cKDTree(np.c_[xb, yb])
                d, j = tb.query(np.c_[xb, yb], k=2)
                nn, jn = d[:, 1] / fw, j[:, 1]
                ins = fr.inside(xb, yb)
                pairs = sorted({(min(i, jn[i]), max(i, jn[i])) for i in np.flatnonzero((nn < SEP_MAX) & ins)})
                da, _ = ta.query(np.c_[xb, yb])
                inA = da / fw < MATCH
                for i, k in pairs:
                    cls = {2: 'both', 1: 'one', 0: 'none'}[int(inA[i]) + int(inA[k])]
                    f1, f2 = fb[i], fb[k]
                    xc = (f1 * xb[i] + f2 * xb[k]) / (f1 + f2)
                    yc = (f1 * yb[i] + f2 * yb[k]) / (f1 + f2)
                    th = np.arctan2(yb[k] - yb[i], xb[k] - xb[i])
                    sep = np.hypot(xb[k] - xb[i], yb[k] - yb[i])
                    q = min(f1, f2) / max(f1, f2)
                    # predicted window-free stretch for two Gaussians
                    pred = (sig ** 2 + q / (1 + q) ** 2 * sep ** 2) / sig ** 2
                    rows.append(dict(seed=seed, xc=float(xc), yc=float(yc), cls=cls, sep_fwhm=float(sep / fw), q=float(q), ftot=float(f1 + f2),
                                     ratio=float(moments(img, xc, yc, 1.5 * sig, th)), pred=float(pred)))
                # null: isolated B sources along a random axis
                iso = np.flatnonzero((nn > ISO) & ins)
                null += [moments(img, xb[i], yb[i], 1.5 * sig, rng.uniform(0, np.pi)) for i in iso]
                nf += list(fb[iso])
            null, nf = np.array(null), np.array(nf)
            res = dict(n_pairs=len(rows), n_iso=len(null))
            for cls in ('both', 'one', 'none'):
                r = [x for x in rows if x['cls'] == cls and np.isfinite(x['ratio'])]
                if not r:
                    continue
                ft = np.array([x['ftot'] for x in r])
                # null: isolated stars at least as bright as the pairs' 10th-percentile total flux
                nm = null[np.isfinite(null) & (nf >= np.percentile(ft, 10))]
                rat = np.array([x['ratio'] for x in r])
                # unique sky positions: pair centroids merged within 1 px across seeds
                cen = np.array([(x['xc'], x['yc']) for x in r])
                grp = cKDTree(cen).query_pairs(1.0)
                lab = np.arange(len(cen))
                for a, b in sorted(grp):
                    lab[lab == lab[b]] = lab[a]
                res_u = len(set(lab))
                res[cls] = dict(n=len(r), n_unique=int(res_u), ratio_med=float(np.median(rat)),
                                ratio_q16_84=[float(v) for v in np.percentile(rat, [16, 84])],
                                pred_med=float(np.median([x['pred'] for x in r])),
                                q_med=float(np.median([x['q'] for x in r])),
                                sep_med=float(np.median([x['sep_fwhm'] for x in r])),
                                null_n=int(len(nm)),
                                null_med=float(np.median(nm)) if len(nm) else None,
                                null_q16_84=[float(v) for v in np.percentile(nm, [16, 84])] if len(nm) else None,
                                frac_above_null84=float(np.mean(rat > np.percentile(nm, 84))) if len(nm) else None)
            res['rows'] = rows
            out[f'{name}_{filt}'] = res
            print(name, filt, json.dumps({k: v for k, v in res.items() if k != 'rows'}), flush=True)
            panels.append((f'{name} {filt}', rows, null[np.isfinite(null)]))
    json.dump(out, open(outp + '.json', 'w'), indent=1)
    fig, axs = plt.subplots(2, 4, figsize=(16, 7))
    for ax, (lab, rows, null) in zip(axs.flat, panels):
        bins = np.linspace(0.6, 3.0, 25)
        ax.hist(np.clip(null, 0.6, 3), bins, density=True, histtype='step', color='k', label='isolated (null)')
        for cls, col in (('both', 'C0'), ('one', 'C3')):
            r = np.array([x['ratio'] for x in rows if x['cls'] == cls])
            r = r[np.isfinite(r)]
            if len(r):
                ax.hist(np.clip(r, 0.6, 3), bins, density=True, histtype='step', color=col, lw=1.5,
                        label=f'{vb} pairs, {cls} in {va} (n={len(r)})')
        ax.set_title(lab, fontsize=9)
        ax.set_xlabel('data variance along / across pair axis')
        ax.legend(fontsize=6)
    fig.suptitle(f'{vb} m7 vetted pairs closer than {SEP_MAX} FWHM (seeds 0-10 pooled): elongation of the data along the '
                 f'pair axis vs isolated stars', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(outp + '.png', dpi=100)


if __name__ == '__main__':
    main(*sys.argv[1:4])
