#!/usr/bin/env python
"""Systematically generate color-magnitude (CMD) and color-color (CCD) diagrams
for the m7 merged photometry catalogs (brick 1182/2221 and Cloud C 2221).

Design notes
------------
* Magnitudes default to the **Vega** system (``mag_vega_*``); errors are
  magnitude-system independent so ``emag_ab_*`` is always used.
* A source is a *firm* detection in a band when unmasked, finite, not saturated /
  near-saturated, and (optionally) with ``emag_ab < --firm-emag``.
* **Upper limits** (``--upper-limits``, on by default): a CMD point whose y-band
  (f1) is firmly detected but whose color-band (f2) is a genuine non-detection is
  drawn at ``mag(f1) - limit(f2)`` -- an upper bound on the color -- in a distinct
  low-alpha color with a left-pointing marker.  CCDs handle each axis the same
  way.  Limiting magnitudes come from the SNR=``--limit-nsigma`` crossing of the
  per-band error-vs-magnitude envelope (:func:`plot_tools.compute_band_limits`).
* The "merged m7" catalog currently contains only the 6 narrow/medium 2221 bands
  (f182m f187n f212n f405n f410m f466n) for *both* fields; the 1182 broadband
  bands (f115w/f200w/f356w/f444w) are not in this merge, so broadband-spanning
  CMDs are skipped automatically (the curated pair lists are filtered to the
  bands actually present).  When a cross-band merged catalog exists, this driver
  will pick the extra bands up with no code change.

Usage
-----
    python -m jwst_gc_pipeline.plotting.make_all_cmds_m7 --field all
    python -m jwst_gc_pipeline.plotting.make_all_cmds_m7 --field brick --magsystem vega
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.table import Table

from jwst_gc_pipeline.plotting.plot_tools import (
    cmd, ccd, cmds, ccds, compute_band_limits, band_detected, band_nondetected,
)
from dust_extinction.averages import CT06_MWGC

FIELDS = {
    'brick':  '/orange/adamginsburg/jwst/brick',
    'cloudc': '/orange/adamginsburg/jwst/cloudc',
}

# Synthetic line-subtracted "bands" present in the merged catalog; these have a
# magnitude but no error/mask columns, so they are valid plot axes but cannot be
# detection-vetted or used as limit targets.
SYNTHETIC = {'410m405', '405m410', '182m187', '187m182'}

# Curated CMD color pairs (f1 = magnitude axis = bluer band).  Filtered to the
# bands actually present in each catalog.
CMD_PAIRS = [
    ('f182m', 'f212n'),
    ('f187n', 'f405n'),   # signature GC CMD
    ('f212n', 'f405n'),
    ('f212n', 'f410m'),
    ('f405n', 'f410m'),
    ('f410m', 'f466n'),
    ('f187n', 'f212n'),
    ('f182m', 'f405n'),
    ('410m405', 'f405n'),  # line-subtracted continuum
    ('182m187', 'f187n'),
    # broadband (only fire if a cross-band merged catalog is supplied)
    ('f115w', 'f200w'),
    ('f200w', 'f444w'),
    ('f115w', 'f444w'),
    ('f200w', 'f405n'),
    ('f356w', 'f444w'),
]

# Curated CCD pairs: (x-color, y-color).
CCD_PAIRS = [
    (('f182m', 'f212n'), ('f212n', 'f405n')),
    (('f187n', 'f405n'), ('f405n', 'f410m')),
    (('f405n', 'f410m'), ('f410m', 'f466n')),
    (('f182m', 'f187n'), ('f405n', 'f410m')),
    (('f212n', 'f405n'), ('f410m', 'f466n')),
    (('f115w', 'f200w'), ('f200w', 'f444w')),
    (('f212n', 'f405n'), ('f356w', 'f444w')),
]


def catalog_path(base, vetting, phase='m7'):
    stem = f'basic_merged_indivexp_photometry_tables_merged_resbgsub_{phase}'
    if vetting == 'qualcuts':
        fn = f'{base}/catalogs/{stem}_qualcuts_oksep2221.fits'
        if os.path.exists(fn):
            return fn
    return f'{base}/catalogs/{stem}.fits'


def real_filters(tbl):
    """Real (non-synthetic) photometric bands present, by mag_vega/mag_ab cols."""
    found = set()
    for c in tbl.colnames:
        for pre in ('mag_vega_', 'mag_ab_'):
            if c.startswith(pre):
                f = c[len(pre):]
                if f not in SYNTHETIC:
                    found.add(f)
    return found


def band_present(tbl, f, magprefix):
    return f in SYNTHETIC and f'{magprefix}_{f}' in tbl.colnames or f'{magprefix}_{f}' in tbl.colnames


def saturated_any(tbl, filters):
    """Boolean: saturated / near-saturated / replaced in ANY of ``filters``."""
    from jwst_gc_pipeline.plotting.plot_tools import _col
    excl = np.zeros(len(tbl), dtype=bool)
    for f in filters:
        for sc in (f'is_saturated_{f}', f'near_saturated_{f}_{f}',
                   f'replaced_saturated_{f}'):
            scol = _col(tbl, sc, dtype=bool, fill=False)
            if scol is not None:
                excl |= scol
    return excl


def _axlims_cmd(colorvals, magvals, limit_colorvals=None):
    cv = colorvals[np.isfinite(colorvals)]
    mv = magvals[np.isfinite(magvals)]
    if cv.size < 10 or mv.size < 10:
        return None
    xlo, xhi = np.nanpercentile(cv, [1, 99])
    ybr, yfa = np.nanpercentile(mv, [1, 99])
    # widen x to keep the upper-limit locus in frame (it sits blueward, since the
    # color-band is replaced by a fixed limit) -- otherwise the limits fall off
    # the left edge.  Use their 2nd percentile, but cap how far we zoom out.
    if limit_colorvals is not None:
        lc = limit_colorvals[np.isfinite(limit_colorvals)]
        if lc.size > 10:
            xlo = min(xlo, max(np.nanpercentile(lc, 2), xlo - 6))
    pad = 0.05 * (xhi - xlo + 1e-3)
    # inverted y: faint (large mag) at bottom, bright (small mag) at top
    return (xlo - pad, xhi + pad, yfa + 0.3, ybr - 0.3)


def _axlims_ccd(xv, yv):
    xv = xv[np.isfinite(xv)]
    yv = yv[np.isfinite(yv)]
    if xv.size < 10 or yv.size < 10:
        return (-1, 5, -1, 5)
    xlo, xhi = np.nanpercentile(xv, [1, 99])
    ylo, yhi = np.nanpercentile(yv, [1, 99])
    return (xlo - 0.3, xhi + 0.3, ylo - 0.3, yhi + 0.3)


def make_field(field, magsystem='vega', vetting='qualcuts', do_limits=True,
               limit_nsigma=3, firm_emag=0.1, ext_model=None, dpi=150, phase='m7'):
    base = FIELDS[field]
    magprefix = f'mag_{magsystem}'
    errprefix = 'emag_ab'
    path = catalog_path(base, vetting, phase=phase)
    print(f'\n=== {field}: {os.path.basename(path)} ({magsystem}) ===')
    tbl = Table.read(path)
    rf = real_filters(tbl)
    print(f'  rows={len(tbl)}  real filters present: {sorted(rf)}')

    excl_sat = saturated_any(tbl, rf)
    print(f'  saturated/near-saturated (any band): {excl_sat.sum()}')

    limits = None
    if do_limits:
        limits = compute_band_limits(tbl, sorted(rf), nsigma=limit_nsigma,
                                     errprefix=errprefix, magprefix=magprefix,
                                     verbose=True)

    outdir = f'{base}/catalogs/cmds_{phase}/{magsystem}'
    os.makedirs(outdir, exist_ok=True)
    ext = ext_model if ext_model is not None else CT06_MWGC()
    manifest = [f'# {field} {magsystem} vetting={vetting} firm_emag={firm_emag} '
                f'limit_nsigma={limit_nsigma}',
                f'# catalog: {path}', f'# rows: {len(tbl)}']
    if limits:
        manifest.append('# limiting mags: ' +
                        ', '.join(f'{k}={v:.2f}' for k, v in sorted(limits.items())))

    # ---- CMDs --------------------------------------------------------------
    n_cmd = 0
    for f1, f2 in CMD_PAIRS:
        if f'{magprefix}_{f1}' not in tbl.colnames or f'{magprefix}_{f2}' not in tbl.colnames:
            continue
        det1 = band_detected(tbl, f1, errprefix=errprefix, max_emag=firm_emag,
                             magprefix=magprefix) if f1 not in SYNTHETIC else \
            np.isfinite(np.asarray(tbl[f'{magprefix}_{f1}'], dtype=float))
        det2 = band_detected(tbl, f2, errprefix=errprefix, max_emag=firm_emag,
                             magprefix=magprefix) if f2 not in SYNTHETIC else \
            np.isfinite(np.asarray(tbl[f'{magprefix}_{f2}'], dtype=float))
        include = det1 & det2 & (~excl_sat)
        if include.sum() < 10:
            continue
        from jwst_gc_pipeline.plotting.plot_tools import _col
        m1 = _col(tbl, f'{magprefix}_{f1}')
        colorvals = (m1 - _col(tbl, f'{magprefix}_{f2}'))[include]

        n_lim = 0
        lim_colorvals = None
        if limits is not None and f2 in limits and np.isfinite(limits[f2]) and f2 not in SYNTHETIC:
            lim_set = det1 & band_nondetected(tbl, f2)
            n_lim = int(lim_set.sum())
            if n_lim:
                lim_colorvals = m1[lim_set] - limits[f2]
        axlims = _axlims_cmd(colorvals, m1[include], lim_colorvals)

        fig, ax = plt.subplots(figsize=(6, 6))
        cmd(ax=ax, basetable=tbl, f1=f1, f2=f2, include=include, sel=None,
            axlims=axlims, ext=ext, extvec_scale=2, head_width=0.05,
            markersize=3, alpha=0.4, rasterized=True, color='k',
            magprefix=magprefix, errprefix=errprefix,
            limits=limits if do_limits else None, det_max_emag=firm_emag,
            limit_alpha=0.18, limit_color='#1f77b4', extvec_start=(axlims[0] + 0.3, axlims[3] - 1.5) if axlims else None)
        ttl = f'{field} {f1} vs {f1}-{f2}  (N={int(include.sum())}'
        ttl += f', lim={n_lim})' if n_lim else ')'
        ax.set_title(ttl, fontsize=10)
        ax.set_ylabel(f'{f1} [{magsystem}]')
        out = f'{outdir}/cmd_{f1}_{f1}-{f2}'
        out += '_ulim' if n_lim else ''
        fig.savefig(out + '.png', dpi=dpi, bbox_inches='tight')
        fig.savefig(out + '.pdf', bbox_inches='tight')
        plt.close(fig)
        manifest.append(f'CMD {f1} vs {f1}-{f2}: N_det={int(include.sum())} N_ulim={n_lim}')
        n_cmd += 1
    print(f'  wrote {n_cmd} CMDs')

    # ---- CCDs --------------------------------------------------------------
    n_ccd = 0
    from jwst_gc_pipeline.plotting.plot_tools import _col
    for color1, color2 in CCD_PAIRS:
        bands = list(color1) + list(color2)
        if any(f'{magprefix}_{b}' not in tbl.colnames for b in bands):
            continue
        x = _col(tbl, f'{magprefix}_{color1[0]}') - _col(tbl, f'{magprefix}_{color1[1]}')
        y = _col(tbl, f'{magprefix}_{color2[0]}') - _col(tbl, f'{magprefix}_{color2[1]}')
        firm = np.isfinite(x) & np.isfinite(y) & (~excl_sat)
        if firm.sum() < 10:
            continue
        axlims = _axlims_ccd(x[firm], y[firm])
        fig, ax = plt.subplots(figsize=(6, 6))
        ccd(tbl, ax=ax, color1=color1, color2=color2, sel=False, axlims=axlims,
            ext=ext, extvec_scale=30, exclude=excl_sat, max_uncertainty=firm_emag,
            markersize=3, alpha=0.4, color='k', selcolor=None, rasterized=True,
            magprefix=magprefix, errprefix=errprefix,
            limits=limits if do_limits else None, det_max_emag=firm_emag,
            limit_alpha=0.15, limit_color='#1f77b4', allow_missing=True)
        ax.set_title(f'{field}  ({color1[0]}-{color1[1]}) vs ({color2[0]}-{color2[1]})',
                     fontsize=10)
        out = (f'{outdir}/ccd_{color1[0]}-{color1[1]}_{color2[0]}-{color2[1]}')
        fig.savefig(out + '.png', dpi=dpi, bbox_inches='tight')
        fig.savefig(out + '.pdf', bbox_inches='tight')
        plt.close(fig)
        manifest.append(f'CCD ({color1[0]}-{color1[1]}) vs ({color2[0]}-{color2[1]}): N_firm={int(firm.sum())}')
        n_ccd += 1
    print(f'  wrote {n_ccd} CCDs')

    # ---- contact sheets ----------------------------------------------------
    cmd_colors = [(a, b) for (a, b) in CMD_PAIRS
                  if f'{magprefix}_{a}' in tbl.colnames and f'{magprefix}_{b}' in tbl.colnames][:9]
    if cmd_colors:
        fig = plt.figure(figsize=(16, 16))
        cmds(tbl, colors=cmd_colors, fig=fig, exclude=excl_sat,
             max_uncertainty=firm_emag, ext=ext, extvec_scale=2, markersize=2,
             alpha=0.4, selcolor=None, magprefix=magprefix, errprefix=errprefix,
             limits=limits if do_limits else None, det_max_emag=firm_emag,
             limit_alpha=0.15, axlims=None)
        fig.savefig(f'{outdir}/_contactsheet_cmds.png', dpi=110, bbox_inches='tight')
        plt.close(fig)

    with open(f'{outdir}/_manifest.txt', 'w') as fh:
        fh.write('\n'.join(manifest) + '\n')
    print(f'  manifest + figures -> {outdir}')
    return outdir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--field', default='all', choices=['brick', 'cloudc', 'all'])
    ap.add_argument('--magsystem', default='vega', choices=['vega', 'ab'])
    ap.add_argument('--vetting', default='qualcuts',
                    choices=['qualcuts', 'basic'])
    ap.add_argument('--phase', default='m7', choices=['m7', 'm8', 'm8_dedup'],
                    help='photometry phase catalog to plot (m8 = forced cross-band '
                         'fill; m8_dedup = + post-merge de-duplication)')
    ap.add_argument('--upper-limits', dest='do_limits', action='store_true', default=True)
    ap.add_argument('--no-upper-limits', dest='do_limits', action='store_false')
    ap.add_argument('--limit-nsigma', type=float, default=3.0)
    ap.add_argument('--firm-emag', type=float, default=0.1)
    ap.add_argument('--dpi', type=int, default=150)
    args = ap.parse_args()

    fields = ['brick', 'cloudc'] if args.field == 'all' else [args.field]
    for field in fields:
        make_field(field, magsystem=args.magsystem, vetting=args.vetting,
                   do_limits=args.do_limits, limit_nsigma=args.limit_nsigma,
                   firm_emag=args.firm_emag, dpi=args.dpi, phase=args.phase)


if __name__ == '__main__':
    main()
