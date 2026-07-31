#!/usr/bin/env python
"""A/B: adaptive per-star fit_shape vs the flat size=81 baseline.

The fit-footprint sweep showed a GLOBAL small fit box biases satstar flux low
(the amplitude leverage is in the extended wings), but the leverage scales with
the saturated-core size -- so a box scaled to each star's core should keep the
bright (large-core) stars accurate while shrinking the many faint/small-core
satstars (cheaper + less faint-neighbour contamination). This runs the finder
ONCE with adaptive_fit_shape=True on the same gc2211 F200W frame and compares to
the saved flat-81 catalog (out_footprint/satstar_size081.fits) star-by-star.

Success criterion: flux agrees with flat-81 across ALL cores (especially the
faint/small ones that got the smallest boxes), at a wall-clock win.

Usage: adaptive_fit_shape_ab.py [crf_path]
"""
import os
import sys
import time

os.environ.setdefault('STPSF_PATH', '/orange/adamginsburg/jwst/stpsf-data/')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u

from jwst_gc_pipeline.reduction import saturated_star_finding as ssf

GC = '/orange/adamginsburg/jwst/gc2211'
CRF = sys.argv[1] if len(sys.argv) > 1 else (
    f'{GC}/F200W/pipeline/jw02211023001_02201_00001_nrca1_destreak_'
    f'jw02211-o023_20260515t014922_image3_00001_crf.fits')
OUT = os.path.join(os.path.dirname(__file__), 'out_footprint')
BASELINE = os.path.join(OUT, 'satstar_size081.fits')   # flat size=81 (from the sweep)
BASELINE_SECONDS = 585.0                                # flat-81 wall-clock (sweep)
MATCH_ARCSEC = 0.3


def run_adaptive():
    fh = fits.open(CRF)
    header = fh[0].header
    if 'CRPIX1' not in header:
        from astropy import wcs as _wcs
        header.update(_wcs.WCS(fh['SCI'].header, relax=True).to_header(relax=True))
    kwargs = dict(path_prefix=f'{GC}/psfs/', plot=False,
                  use_merged_psf_for_merged=False, pad=81,
                  adaptive_fit_shape=True)
    zf = ssf._find_zeroframe_for(CRF)
    if zf is not None:
        kwargs['zeroframe'] = zf
    t0 = time.time()
    tab = ssf.get_saturated_stars(fh, **kwargs)
    dt = time.time() - t0
    fh.close()
    print(f'[ab] adaptive accepted={0 if tab is None else len(tab)} seconds={dt:.0f}',
          flush=True)
    if tab is not None:
        tab.write(os.path.join(OUT, 'satstar_adaptive.fits'), overwrite=True)
    return tab, dt


def main():
    adap, dt = run_adaptive()
    base = Table.read(BASELINE)
    if adap is None or not len(adap):
        raise SystemExit('[ab] adaptive run produced no satstars')

    # match adaptive -> baseline by sky position
    sc_b = SkyCoord(base['skycoord_fit'])
    sc_a = SkyCoord(adap['skycoord_fit'])
    idx, sep, _ = sc_a.match_to_catalog_sky(sc_b)
    ok = sep.arcsec <= MATCH_ARCSEC
    ia = np.where(ok)[0]
    ib = idx[ok]
    print(f'[ab] matched {len(ia)}/{len(adap)} adaptive rows to baseline '
          f'(<= {MATCH_ARCSEC}")', flush=True)

    fb = np.asarray(base['flux_fit'][ib], float)
    fa = np.asarray(adap['flux_fit'][ia], float)
    sa = np.asarray(base['sat_area'][ib], float) if 'sat_area' in base.colnames else \
        np.full(len(ib), np.nan)
    # if sat_area not stored, derive core radius from n_pixels? fall back to flux rank
    dflux = 100 * (fa - fb) / fb
    dpos = sc_a[ia].separation(sc_b[ib]).to(u.mas).value
    npa = np.asarray(adap['n_pixels_fit'][ia], float)
    npb = np.asarray(base['n_pixels_fit'][ib], float)

    finite = np.isfinite(dflux) & np.isfinite(fb) & (fb > 0)
    print('\n' + '=' * 78)
    print('ADAPTIVE vs FLAT-81 (matched satstars)')
    print('=' * 78)
    print(f'  wall-clock: adaptive {dt:.0f}s vs flat-81 {BASELINE_SECONDS:.0f}s '
          f'-> {dt / BASELINE_SECONDS:.2f}x ({100 * (1 - dt / BASELINE_SECONDS):.0f}% faster)')
    print(f'  n matched: {int(finite.sum())}')
    print(f'  |dflux| median {np.nanmedian(np.abs(dflux[finite])):.2f}%  '
          f'90th {np.nanpercentile(np.abs(dflux[finite]), 90):.2f}%  '
          f'max {np.nanmax(np.abs(dflux[finite])):.2f}%')
    print(f'  dpos   median {np.nanmedian(dpos[finite]):.2f} mas  '
          f'90th {np.nanpercentile(dpos[finite], 90):.2f} mas')
    print(f'  n_pixels_fit median: adaptive {np.nanmedian(npa):.0f} vs flat {np.nanmedian(npb):.0f} '
          f'({100 * np.nanmedian(npa) / max(1, np.nanmedian(npb)):.0f}%)')

    # split by brightness: are the FAINT (small-core, smallest boxes) still accurate?
    print('\n  |dflux|% by baseline-flux quartile (faint stars got the smallest boxes):')
    q = np.nanpercentile(fb[finite], [25, 50, 75])
    edges = [-np.inf] + list(q) + [np.inf]
    labs = ['faintest 25%', 'Q2', 'Q3', 'brightest 25%']
    for lo, hi, lab in zip(edges[:-1], edges[1:], labs):
        m = finite & (fb > lo) & (fb <= hi)
        if m.sum():
            print(f'    {lab:16s} N={int(m.sum()):4d}  |dflux| med {np.nanmedian(np.abs(dflux[m])):6.2f}%  '
                  f'dpos med {np.nanmedian(dpos[m]):5.1f} mas  '
                  f'n_pix med {np.nanmedian(npa[m]):5.0f}')

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    a = ax[0]
    a.scatter(fb[finite], dflux[finite], s=12, alpha=0.5)
    a.axhline(0, color='0.5'); a.axhline(2, color='r', ls=':'); a.axhline(-2, color='r', ls=':')
    a.set_xscale('log'); a.set_xlabel('baseline flux_fit'); a.set_ylabel('adaptive − flat-81 flux (%)')
    a.set_ylim(-20, 20); a.set_title('(a) flux match vs brightness\n(faint = smallest boxes)')

    a = ax[1]
    a.scatter(npa[finite], npb[finite], s=12, alpha=0.5)
    lim = [1, max(np.nanmax(npb), np.nanmax(npa)) * 1.1]
    a.plot(lim, lim, 'k--', lw=1)
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xlabel('adaptive n_pixels_fit'); a.set_ylabel('flat-81 n_pixels_fit')
    a.set_title('(b) fit pixels used: adaptive shrinks the box')

    a = ax[2]
    a.scatter(fb[finite], np.abs(dflux[finite]), s=12, alpha=0.5, label='|dflux|')
    a.set_xscale('log'); a.set_yscale('log')
    a.set_xlabel('baseline flux_fit'); a.set_ylabel('|adaptive − flat-81| flux (%)')
    a.axhline(2, color='r', ls=':', label='2%'); a.legend()
    a.set_title(f'(c) accuracy vs brightness\nadaptive {dt:.0f}s vs 585s baseline')

    fig.suptitle('Adaptive per-star fit_shape vs flat size=81 — gc2211 F200W nrca1', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = os.path.join(OUT, 'adaptive_fit_shape_ab.png')
    fig.savefig(p, dpi=120)
    print(f'\n[ab] wrote {p}')
    print('[ab] DONE')


if __name__ == '__main__':
    main()
