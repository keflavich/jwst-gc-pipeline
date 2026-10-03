"""Does the PRODUCTION saturated-star flux depend on detector x (#1013 follow-up)?

    python production_flux_vs_x.py fit <workdir> <psf dir> <cal key>...
    python production_flux_vs_x.py analyze <out prefix> <workdir>

The NRCBLONG halo of saturated stars is ~20% fainter at x = 250-550 px
(``perexp_ramp.py``: already in the raw ramp).  The production masked-core
fit takes its amplitude from that halo, so the same star should read fainter
in the dithers that put it there.

``fit`` runs ``saturated_star_finding.remove_saturated_stars`` unchanged
(production gates, the fovp1024 LW grid found in ``<psf dir>``) on each
``_cal`` frame -- an S3 key of the public mirror or a local path -- keeps the
``_satstar_catalog.fits`` and deletes the frame and its model/residual
images.  No Detector1 ``_ramp.fits`` is available here, so the group-0
anchoring (SATSTAR_ZEROFRAME_FIT) is not used.

``analyze`` links each catalog row between the dithers of its star-visit by
its ``skycoord_fit`` (0.3"; no catalog matching against a reference), and
reports, per star, median flux_fit over the in-band dithers / median over the
out-of-band ones, for the band and for a null band (x = 1300-1600).
"""
import glob
import json
import os
import subprocess
import sys

import numpy as np
from astropy.table import Table
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = 'https://stpubdata.s3.amazonaws.com/'
BAND = (250, 550)
NULL_BAND = (1300, 1600)
LINK_ARCSEC = 0.3
AREA_BINS = (0, 50, 150, 500, 1e6)     # saturated pixels (catalog sat_area)


def fit(workdir, psfdir, keys):
    from jwst_gc_pipeline.reduction.saturated_star_finding import remove_saturated_stars
    os.makedirs(workdir, exist_ok=True)
    for key in keys:
        root = os.path.basename(key).replace('.fits', '')
        cat = os.path.join(workdir, root + '_satstar_catalog.fits')
        if os.path.exists(cat):
            continue
        local = os.path.exists(key)
        fn = key if local else os.path.join(workdir, os.path.basename(key))
        if not local:
            subprocess.run(['curl', '-sS', '--retry', '4', '-o', fn, BASE + key], check=True)
        try:
            remove_saturated_stars(fn, path_prefix=psfdir, plot=False)
            made = fn.replace('.fits', '_satstar_catalog.fits')
            if os.path.exists(made) and made != cat:
                os.replace(made, cat)
        finally:
            for f in glob.glob(fn.replace('.fits', '_satstar_*.fits')) + \
                    glob.glob(fn.replace('.fits', '_unsatstar*.fits')) + \
                    glob.glob(fn.replace('.fits', '_wingcal_*.fits')):
                if not f.endswith('_satstar_catalog.fits'):
                    os.remove(f)
            if not local:
                os.remove(fn)
        print('done', root, flush=True)


def analyze(outp, workdir):
    rows = []
    for fn in sorted(glob.glob(os.path.join(workdir, '*_satstar_catalog.fits'))):
        t = Table.read(fn)
        b = os.path.basename(fn)
        visit, exp = b[:19], int(b[20:25])
        sc = t['skycoord_fit']
        for i in range(len(t)):
            f = float(t['flux_fit'][i])
            if not np.isfinite(f) or f <= 0:
                continue
            area = float(t['sat_area'][i])
            rows.append(dict(visit=visit, exp=exp, x=float(t['x_0'][i]), flux=f, area=area if np.isfinite(area) else 0.0,
                             ra=float(sc[i].ra.deg), dec=float(sc[i].dec.deg)))
    clusters = []
    for v in sorted({r['visit'] for r in rows}):
        cl, cra, cdec = [], [], []
        for r in (r for r in rows if r['visit'] == v):
            j = None
            if cl:
                dra = (np.array(cra) - r['ra']) * np.cos(np.deg2rad(r['dec']))
                sep = 3600 * np.hypot(dra, np.array(cdec) - r['dec'])
                for i in np.argsort(sep):
                    if sep[i] > LINK_ARCSEC:
                        break
                    if r['exp'] not in cl[i]['exps']:
                        j = i
                        break
            if j is None:
                cl.append(dict(rows=[r], exps={r['exp']})); cra.append(r['ra']); cdec.append(r['dec'])
            else:
                cl[j]['rows'].append(r); cl[j]['exps'].add(r['exp'])
        clusters += cl
    rng = np.random.default_rng(0)
    res = dict(n_catalog_rows=len(rows), band=BAND, null_band=NULL_BAND)
    per = {}
    for name, (lo, hi) in (('band', BAND), ('null', NULL_BAND)):
        q, fmed, amed, ain = [], [], [], []
        for k in clusters:
            x = np.array([r['x'] for r in k['rows']]); F = np.array([r['flux'] for r in k['rows']])
            inb = (x > lo) & (x < hi)
            # "outside" excludes BOTH regions (+-50 px), so neither the band's
            # low dithers nor the null's enter the other's denominator
            out = np.ones(x.size, bool)
            for a, b in (BAND, NULL_BAND):
                out &= (x < a - 50) | (x > b + 50)
            if inb.sum() < 1 or out.sum() < 2:
                continue
            q.append(np.median(F[inb]) / np.median(F[out])); fmed.append(np.median(F))
            A = np.array([r['area'] for r in k['rows']])
            amed.append(np.median(A)); ain.append(np.median(A[inb]))
        q, fmed, amed, ain = np.array(q), np.array(fmed), np.array(amed), np.array(ain)
        bs = [np.median(rng.choice(q, q.size)) for _ in range(2000)] if q.size else [np.nan]
        res[name] = dict(n_stars=int(q.size), median_ratio=float(np.median(q)) if q.size else None,
                         err=float(np.std(bs)), p16_p84=[float(np.percentile(q, 16)), float(np.percentile(q, 84))]
                         if q.size else None)
        res[name]['by_sat_area'] = {}
        for lo, hi in zip(AREA_BINS[:-1], AREA_BINS[1:]):
            m = (amed >= lo) & (amed < hi)
            if m.sum() < 5:
                continue
            bsm = [np.median(rng.choice(q[m], m.sum())) for _ in range(2000)]
            res[name]['by_sat_area'][f'{lo:g}-{hi:g}'] = dict(n_stars=int(m.sum()), median_ratio=float(np.median(q[m])),
                                                             err=float(np.std(bsm)))
        per[name] = (q, fmed, amed, ain)
    # the column term: band / null per area bin (a size trend common to both cancels)
    res['band_over_null_by_sat_area'] = {
        k: dict(ratio=v['median_ratio'] / res['null']['by_sat_area'][k]['median_ratio'],
                err=float(np.hypot(v['err'], res['null']['by_sat_area'][k]['err'])))
        for k, v in res['band']['by_sat_area'].items() if k in res['null']['by_sat_area']}
    # proposed correction: in-band depth = k log10(area / A0) above A0, with
    # area the core area OF THE IN-BAND DETECTION (what a per-detection
    # correction has); robust (L1) grid fit over the band stars
    q, _, _, ain = per['band']
    la = np.log10(np.maximum(ain, 1))
    best = None
    for la0 in np.arange(0.5, 2.5, 0.01):
        z = np.maximum(la - la0, 0)
        if not np.any(z > 0):
            continue
        ks = np.linspace(0, 0.4, 401)
        cost = np.abs((1 - q)[None] - ks[:, None] * z[None]).sum(1)
        i = np.argmin(cost)
        if best is None or cost[i] < best[0]:
            best = (cost[i], ks[i], la0)
    res['depth_model'] = dict(form='flux_in_band / flux_outside = 1 - k * max(0, log10(sat_area / A0))',
                              area='core area of the in-band detection', k=float(best[1]), A0=float(10 ** best[2]))
    # its x shape, for stars whose in-band core has >= A0 pixels: 32-px profile of
    # flux / the star's median over its out-of-band dithers
    edges32 = np.arange(0, 2049, 32)
    xs2, rel2 = [], []
    for k in clusters:
        x = np.array([r['x'] for r in k['rows']]); F = np.array([r['flux'] for r in k['rows']])
        A = np.array([r['area'] for r in k['rows']])
        out = (x < BAND[0] - 50) | (x > BAND[1] + 50)
        if out.sum() < 2 or np.median(A) < 2 * res['depth_model']['A0']:
            continue
        xs2 += list(x); rel2 += list(F / np.median(F[out]))
    xs2, rel2 = np.array(xs2), np.array(rel2)
    kb2 = np.digitize(xs2, edges32) - 1
    res['xshape32'] = dict(edges=edges32.tolist(), min_area=2 * res['depth_model']['A0'],
                           median_rel_flux=[float(np.median(rel2[kb2 == i])) if (kb2 == i).sum() >= 10 else None
                                            for i in range(len(edges32) - 1)],
                           n=[int((kb2 == i).sum()) for i in range(len(edges32) - 1)])
    # flux_fit relative to the star's median over its dithers, binned in x
    xs, rel = [], []
    for k in clusters:
        if len(k['rows']) < 4:
            continue
        F = np.array([r['flux'] for r in k['rows']])
        xs += [r['x'] for r in k['rows']]; rel += list(F / np.median(F))
    xs, rel = np.array(xs), np.array(rel)
    edges = np.arange(0, 2049, 128)
    kb = np.digitize(xs, edges) - 1
    prof = [float(np.median(rel[kb == i])) if (kb == i).sum() >= 20 else None for i in range(len(edges) - 1)]
    nprof = [int((kb == i).sum()) for i in range(len(edges) - 1)]
    res['xprofile'] = dict(edges=edges.tolist(), median_rel_flux=prof, n=nprof)
    json.dump(res, open(outp + '.json', 'w'), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != 'xprofile'}, indent=1))
    print('xprofile', [None if p is None else round(p, 3) for p in prof])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
    xc = 0.5 * (edges[1:] + edges[:-1])
    ax[0].plot(xc, [np.nan if p is None else p for p in prof], 'o-', label='all saturated stars (128 px)')
    xc32 = 0.5 * (edges32[1:] + edges32[:-1])
    ax[0].plot(xc32, [np.nan if p is None else p for p in res['xshape32']['median_rel_flux']], '.-',
               label=f"cores ≥ {res['xshape32']['min_area']:.0f} px, / out-of-band median (32 px)")
    ax[0].legend(fontsize=8)
    ax[0].axhline(1, color='k', lw=0.5); ax[0].axvspan(*BAND, color='C1', alpha=0.15)
    ax[0].set_xlabel('NRCBLONG detector x [px]'); ax[0].set_ylabel("flux_fit / the star's median over its dithers")
    ax[0].set_title(f'production satstar flux vs x ({rel.size} measurements)', fontsize=9)
    for j, name in ((1, 'null'), (0, 'band')):
        q, fmed, amed, ain = per[name]
        ax[1].scatter(ain, q, s=8, alpha=0.6, color=f'C{j}',
                      label=f'{name}: median {res[name]["median_ratio"]:.3f} ± {res[name]["err"]:.3f} ({q.size} stars)')
    dm = res['depth_model']
    aa = np.logspace(0.7, 3.5, 100)
    ax[1].plot(aa, 1 - dm['k'] * np.maximum(np.log10(aa / dm['A0']), 0), 'k--',
               label=f"1 − {dm['k']:.3f} log10(area / {dm['A0']:.0f})")
    ax[1].axhline(1, color='k', lw=0.5); ax[1].set_xscale('log'); ax[1].set_ylim(0.5, 1.4)
    ax[1].set_xlabel('saturated pixels of the in-band detection'); ax[1].set_ylabel('flux_fit in band / outside')
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(outp + '.png', dpi=int(os.environ.get('FIG_DPI', 130)))


if __name__ == '__main__':
    if sys.argv[1] == 'fit':
        fit(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        analyze(sys.argv[2], sys.argv[3])
