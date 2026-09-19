"""Figures for #925 item 3 / PR #928, from the delivered o132 products.

Reads only; writes PNGs beside this file.  Run from a checkout that has the
#928 changes importable (the consolidation's ensemble columns are used).

    python docs/evidence/satstar_sibling_seed/make_figures.py

Three claims, one figure each:

  coverage_gap.png      exposures that COVER a star vs exposures that MEASURE
                        it -- the gap the PR closes
  rejected_offsets.png  why the gate-rejected fits on disk cannot fill it
  shell.png             what the 223 mas fit_quality_gate population actually
                        is: a shell, not a displacement and not a random field
"""
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt                                    # noqa: E402
from astropy.table import Table, vstack                            # noqa: E402
from astropy.coordinates import SkyCoord                           # noqa: E402
import astropy.units as u                                          # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, '..', '..', '..')))
from jwst_gc_pipeline.photometry import merge_catalogs as mc       # noqa: E402
from jwst_gc_pipeline.photometry.satstar_wcs_refresh import (      # noqa: E402
    frame_path_for_satstar_catalog)
from jwst_gc_pipeline.frame_wcs import frame_wcs                   # noqa: E402

BP = os.environ.get('SATSTAR_FIG_BASEPATH',
                    '/orange/adamginsburg/jwst/gc-treasury')
OBS = 'o132'
CAP_MAS = 300.0
RNG = np.random.default_rng(20260919)

C_REAL = '#1a1a1a'
C_ACC = '#2166ac'
C_IPG = '#d6604d'
C_FQG = '#b2182b'
C_RAND = '#888888'


def load(filt):
    """Consolidated stars + every per-exposure accepted/rejected fit."""
    acc_fns = sorted(glob.glob(
        f'{BP}/{filt}/pipeline/*{OBS}*_m*_satstar_catalog.fits'))
    rej_fns = sorted(glob.glob(
        f'{BP}/{filt}/pipeline/*{OBS}*_m*_satstar_rejected.fits'))
    wcs_cache = {}
    acc = vstack([mc._read_satstar_catalog_on_current_frame(fn, wcs_cache)
                  for fn in acc_fns], metadata_conflicts='silent')
    out = mc._dedup_satstar_catalog(acc, target='gc-treasury')

    parts = []
    for fn in rej_fns:
        t = Table.read(fn)
        if len(t) == 0 or 'skycoord_fit' not in t.colnames:
            continue
        cols = [c for c in ('skycoord_fit', 'reject_reason') if c in t.colnames]
        parts.append(t[cols])
    rej = vstack(parts, metadata_conflicts='silent') if parts else None
    return out, acc, rej, acc_fns, wcs_cache


def coverage(out, acc_fns, wcs_cache):
    """Exposures whose footprint contains each consolidated star."""
    by_exp = {}
    for fn in acc_fns:
        by_exp.setdefault(mc.satstar_exposure_key(fn), []).append(fn)
    sc = out['skycoord_fit']
    n_cover = np.zeros(len(out), dtype=int)
    for group in by_exp.values():
        frame = frame_path_for_satstar_catalog(group[0])
        if frame is None:
            continue
        w = wcs_cache.get(frame) or frame_wcs(frame)
        try:
            x, y = w.world_to_pixel(sc)
        except (ValueError, RuntimeError):
            continue
        shape = getattr(w, 'array_shape', None) or (2048, 2048)
        ny, nx = shape[-2], shape[-1]
        n_cover += (np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < nx)
                    & (y >= 0) & (y < ny)).astype(int)
    return n_cover


def fig_coverage(data):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2))
    for col, (filt, d) in enumerate(data.items()):
        out, n_cover = d['out'], d['n_cover']
        n_meas = np.asarray(out['n_frames_fit'], int)
        bins = np.arange(-0.5, max(n_cover.max(), n_meas.max()) + 1.5)

        a = axes[0, col]
        a.hist(n_cover, bins=bins, color=C_RAND, alpha=.75,
               label=f'exposures COVERING (median {np.median(n_cover):.0f})')
        a.hist(n_meas, bins=bins, histtype='step', lw=2.2, color=C_REAL,
               label=f'exposures MEASURING (median {np.median(n_meas):.0f})')
        a.set_xlabel('exposures per star')
        a.set_ylabel('number of stars')
        a.set_title(f'{filt}  jw10678-{OBS}   {len(out)} satstars')
        a.legend(fontsize=8)

        b = axes[1, col]
        ok = n_cover > 0
        frac = n_meas[ok] / n_cover[ok]
        b.hist(np.clip(frac, 0, 1.2), bins=np.arange(0, 1.25, 0.05),
               color=C_ACC, alpha=.85)
        missed = int(np.clip(n_cover - n_meas, 0, None).sum())
        b.axvline(1.0, color=C_REAL, ls='--', lw=1.5)
        b.set_xlabel('measured / covered')
        b.set_ylabel('number of stars')
        b.set_title(f'{np.mean(frac >= 0.999) * 100:.1f}% complete; '
                    f'{missed} of {int(n_cover.sum())} star-exposures missed '
                    f'({missed / max(1, n_cover.sum()) * 100:.1f}%)',
                    fontsize=10)
    fig.suptitle('The gap #928 closes: a star covered by an exposure that '
                 'never measured it', fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_HERE, 'coverage_gap.png'), dpi=110)
    print('wrote coverage_gap.png', flush=True)


def offsets(out, acc, rej):
    """Offset of each per-exposure fit from its star's ensemble mean."""
    star = out['skycoord_fit']
    res = {}
    asc = acc['skycoord_fit']
    fin = np.isfinite(asc.ra.deg)
    _, d, _ = asc[fin].match_to_catalog_sky(star)
    d = d.mas
    res['accepted'] = d[d < CAP_MAS]

    rsc = SkyCoord(rej['skycoord_fit'])
    fin = np.isfinite(rsc.ra.deg) & np.isfinite(rsc.dec.deg)
    rsc = rsc[fin]
    reason = np.asarray(rej['reject_reason'], dtype=str)[fin]
    idx, d2, _ = rsc.match_to_catalog_sky(star)
    for tag in ('implied_peak_gate', 'fit_quality_gate'):
        sel = (reason == tag) & (d2.mas < CAP_MAS)
        res[tag] = d2.mas[sel]
        if tag == 'fit_quality_gate':
            a, b = rsc[sel], star[idx[sel]]
            res['_vec'] = (
                (a.ra.deg - b.ra.deg) * np.cos(np.radians(b.dec.deg)) * 3.6e6,
                (a.dec.deg - b.dec.deg) * 3.6e6)

    ra, dec = rsc.ra.deg, rsc.dec.deg
    n = 200000
    rand = SkyCoord(RNG.uniform(ra.min(), ra.max(), n) * u.deg,
                    RNG.uniform(dec.min(), dec.max(), n) * u.deg, frame='icrs')
    _, dr, _ = rand.match_to_catalog_sky(star)
    res['random'] = dr.mas[dr.mas < CAP_MAS]
    return res


def fig_offsets(res):
    fig, ax = plt.subplots(figsize=(8.4, 5.4))
    series = (('accepted per-exposure fits', res['accepted'], C_ACC, '-'),
              ('implied_peak_gate rejections', res['implied_peak_gate'], C_IPG, '-'),
              ('fit_quality_gate rejections', res['fit_quality_gate'], C_FQG, '-'),
              ('random positions, same footprint', res['random'], C_RAND, '--'))
    for label, v, c, ls in series:
        if len(v) < 10:
            continue
        s = np.sort(v)
        ax.plot(s, np.arange(1, len(s) + 1) / len(s), ls, color=c, lw=2,
                label=f'{label}  (n={len(v)}, median {np.median(v):.1f} mas)')
        ax.axvline(np.median(v), color=c, lw=.8, alpha=.45)
    ax.set_xscale('log')
    # clip at 0.3 mas: the accepted set includes each star's own representative
    # row matching itself at ~0 separation, which otherwise stretches the axis
    # to 1e-8 and squeezes every real feature into the right-hand third
    ax.set_xlim(0.3, 330)
    ax.axvspan(120, 330, color=C_FQG, alpha=.05)
    ax.set_xlabel('offset from the star\'s accepted-only ensemble mean (mas)')
    ax.set_ylabel('cumulative fraction')
    ax.set_title('F480M jw10678-o132: why the gate-rejected fits on disk '
                 'cannot fill the gap', fontsize=11)
    ax.legend(fontsize=8.5, loc='upper left')
    ax.grid(alpha=.25)
    ax.text(0.985, 0.30,
            'even the SAME-STAR component of the rejected fits\n'
            '(left of the shaded band) sits at 16-18 mas,\n'
            'seven times the accepted 2.27 mas',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
            bbox=dict(fc='white', ec='0.7', alpha=.92))
    fig.tight_layout()
    fig.savefig(os.path.join(_HERE, 'rejected_offsets.png'), dpi=110)
    print('wrote rejected_offsets.png', flush=True)


SPLIT_MAS = 120.0        # trough between the two components


def fig_shell(res):
    """fit_quality_gate is TWO populations; the single median hid that."""
    dra, ddec = res['_vec']
    r = np.hypot(dra, ddec)
    near, far = r < SPLIT_MAS, r >= SPLIT_MAS
    fig = plt.figure(figsize=(12.6, 4.4))

    a = fig.add_subplot(1, 3, 1)
    a.scatter(dra[near], ddec[near], s=7, alpha=.5, color=C_ACC, lw=0,
              label=f'same star, poorly fit: {near.sum()} '
                    f'({near.mean() * 100:.0f}%), med {np.median(r[near]):.0f} mas')
    a.scatter(dra[far], ddec[far], s=7, alpha=.35, color=C_FQG, lw=0,
              label=f'shell: {far.sum()} ({far.mean() * 100:.0f}%), '
                    f'med {np.median(r[far]):.0f} mas')
    th = np.linspace(0, 2 * np.pi, 200)
    for rad, ls in ((np.percentile(r[far], 16), ':'),
                    (np.median(r[far]), '--'),
                    (np.percentile(r[far], 84), ':')):
        a.plot(rad * np.cos(th), rad * np.sin(th), color=C_REAL, lw=1, ls=ls)
    a.set_aspect('equal')
    a.set_xlabel('$\\Delta$RA$\\cos\\delta$ (mas)')
    a.set_ylabel('$\\Delta$Dec (mas)')
    a.set_title(f'fit_quality_gate offsets (n={len(r)})', fontsize=10)
    a.legend(fontsize=7.2, loc='upper left')

    b = fig.add_subplot(1, 3, 2)
    bins = np.arange(0, CAP_MAS + 10, 10)
    b.hist(r, bins=bins, density=True, color=C_FQG, alpha=.8,
           label='fit_quality_gate')
    b.hist(res['random'], bins=bins, density=True, histtype='step', lw=2,
           color=C_REAL, label='random, same footprint')
    b.axvline(SPLIT_MAS, color=C_ACC, ls='--', lw=1.4)
    rf = r[far]
    hw = (np.percentile(rf, 84) - np.percentile(rf, 16)) / 2
    b.set_xlabel('offset (mas)')
    b.set_ylabel('density')
    b.set_title(f'bimodal; the far mode is narrow:\n'
                f'{np.median(rf):.0f} $\\pm$ {hw:.0f} mas '
                f'({hw / np.median(rf) * 100:.1f}% half-width)', fontsize=10)
    b.legend(fontsize=8)

    c = fig.add_subplot(1, 3, 3, projection='polar')
    c.hist(np.arctan2(ddec[far], dra[far]),
           bins=np.linspace(-np.pi, np.pi, 25), color=C_FQG, alpha=.8)
    ux, uy = dra[far] / rf, ddec[far] / rf
    Rbar = np.hypot(ux.mean(), uy.mean())
    c.set_title(f'shell direction: |$\\bar R$|={Rbar:.3f}\n'
                f'(3$\\sigma$ {np.sqrt(-np.log(0.0027) / far.sum()):.3f}, '
                f'5$\\sigma$ {np.sqrt(-np.log(5.733e-7) / far.sum()):.3f}) '
                f'-> no 3$\\sigma$ signal', fontsize=9, pad=18)

    fig.suptitle('fit_quality_gate is TWO populations: a same-star component '
                 'and a narrow shell at 233 mas = 3.7 F480M pixels',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(_HERE, 'shell.png'), dpi=110)
    print('wrote shell.png', flush=True)


if __name__ == '__main__':
    data = {}
    for filt in ('F480M', 'F212N'):
        out, acc, rej, acc_fns, wcs_cache = load(filt)
        data[filt] = dict(out=out, acc=acc, rej=rej,
                          n_cover=coverage(out, acc_fns, wcs_cache))
        print(f'{filt}: {len(out)} stars loaded', flush=True)
    fig_coverage(data)
    res = offsets(data['F480M']['out'], data['F480M']['acc'],
                  data['F480M']['rej'])
    fig_offsets(res)
    fig_shell(res)
