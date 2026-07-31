#!/usr/bin/env python
"""How much do satstar fits change with a SMALLER fitting footprint?

Motivation (Matt's PEPPAR discussion + the coupling found in the code map): in
``saturated_star_finding.get_saturated_stars`` the fit footprint ``size``
(PSFPhotometry ``fit_shape``, the ONLY place ``size`` is used -- line ~2987)
defaults to ``pad`` (the cutout half-size, 81). Everything else -- the cutout,
the PSF model grid (``fov_pixels`` 512/1024/2048), the saturated-core mask, the
local-background annulus, the rendered model image used for wing rejection and
background maps -- is driven by ``pad``/``fov_pixels``, NOT ``size``. So fitting
can be decoupled from the model/mask TODAY by passing a small ``size`` with
``pad`` unchanged; no code change needed.

This experiment quantifies the trade. For a sweep of ``size`` (holding pad=81
and the PSF grid fixed) it re-runs the finder on one gc2211 F200W frame and, for
the top ~20 saturated stars (by flux), tracks how flux / position / fit quality
and the wall-clock move as the fit footprint shrinks:

  * flux_fit fractional change vs the large-footprint reference (contamination /
    wing-leverage bias),
  * position shift (mas) vs reference,
  * qfit, reduced_chi2, n_pixels_fit (does a small box starve the fit -- fall
    inside the masked saturated core?),
  * per-run wall-clock and the accepted-count (does a small box change the gated
    population?).

Reads production frames read-only; writes catalogs + a summary + a figure to
``out_footprint/``.

Usage: fit_footprint_sweep.py [sizes_csv] [crf_path]
       sizes default "11,17,21,31,51,81"  (81 = current production default)
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
SIZES = [int(s) for s in (sys.argv[1] if len(sys.argv) > 1
                          else os.environ.get('SWEEP_SIZES', '11,17,21,31,51,81')).split(',')]
CRF = sys.argv[2] if len(sys.argv) > 2 else os.environ.get('SWEEP_CRF', (
    f'{GC}/F200W/pipeline/jw02211023001_02201_00001_nrca1_destreak_'
    f'jw02211-o023_20260515t014922_image3_00001_crf.fits'))
# PSF path_prefix: production uses {basepath}/psfs; override per field via env.
PATH_PREFIX = os.environ.get('SWEEP_PSFS', f'{GC}/psfs/')
OUT = os.environ.get('SWEEP_OUT', os.path.join(os.path.dirname(__file__), 'out_footprint'))
os.makedirs(OUT, exist_ok=True)
PAD = 81                 # held FIXED across the sweep (cutout/mask/model-eval)
REF_SIZE = max(SIZES)    # reference footprint (production default = 81)
NTOP = 20                # top saturated stars to track
MATCH_ARCSEC = 0.3


def run_one(size):
    """Run the finder at fit_shape=size (pad fixed), return (table, seconds)."""
    fh = fits.open(CRF)
    header = fh[0].header
    if 'CRPIX1' not in header:
        # The GWCS, not the SCI header's SIP approximation (astrometry rule #2:
        # SIP is a fit of the GWCS and carries 5-8 mas of its own error).
        from jwst_gc_pipeline.frame_wcs import frame_wcs
        from jwst_gc_pipeline.reduction.fits_wcs_sync import sync_header_to_gwcs
        sync_header_to_gwcs(header, frame_wcs(fh).gwcs, fh['SCI'].data.shape,
                            label='fit-shape-sweep')
    # zeroframe amplitude anchoring: match production (SATSTAR_ZEROFRAME_FIT on)
    kwargs = dict(path_prefix=PATH_PREFIX, plot=False,
                  use_merged_psf_for_merged=False, pad=PAD, size=size)
    zf = ssf._find_zeroframe_for(CRF)
    if zf is not None:
        kwargs['zeroframe'] = zf
    t0 = time.time()
    tab = ssf.get_saturated_stars(fh, **kwargs)
    dt = time.time() - t0
    fh.close()
    n = 0 if tab is None else len(tab)
    print(f'[sweep] size={size:3d} accepted={n} seconds={dt:.0f}', flush=True)
    if tab is not None:
        tab.write(os.path.join(OUT, f'satstar_size{size:03d}.fits'), overwrite=True)
    return tab, dt


def match_to_ref(ref_top, tab):
    """Nearest-sky match of tab rows to the top reference stars (<=MATCH_ARCSEC)."""
    if tab is None or not len(tab):
        return [None] * len(ref_top)
    sc_ref = SkyCoord(ref_top['skycoord_fit'])
    sc = SkyCoord(tab['skycoord_fit'])
    idx, sep, _ = sc_ref.match_to_catalog_sky(sc)
    out = []
    for k in range(len(ref_top)):
        out.append(int(idx[k]) if sep[k].arcsec <= MATCH_ARCSEC else None)
    return out


def main():
    print(f'[sweep] frame {os.path.basename(CRF)}  sizes {SIZES}  pad(FIXED)={PAD}',
          flush=True)
    results = {}
    times = {}
    counts = {}
    for size in SIZES:
        tab, dt = run_one(size)
        results[size] = tab
        times[size] = dt
        counts[size] = 0 if tab is None else len(tab)

    ref = results[REF_SIZE]
    if ref is None or not len(ref):
        raise SystemExit('[sweep] reference run produced no satstars')
    # top-N by flux, restricted to in-FOV fits (finite position, real cores)
    infov = ~np.asarray(ref['outside_fov_seed'], bool) if 'outside_fov_seed' in ref.colnames \
        else np.ones(len(ref), bool)
    order = np.argsort(np.where(infov, ref['flux_fit'].filled(np.nan)
                                if hasattr(ref['flux_fit'], 'filled') else ref['flux_fit'],
                                -np.inf))[::-1]
    top = ref[order[:NTOP]]
    sc_ref = SkyCoord(top['skycoord_fit'])
    print(f'\n[sweep] tracking top {len(top)} stars by flux '
          f'(flux {top["flux_fit"].min():.3g}..{top["flux_fit"].max():.3g})', flush=True)

    # assemble per-star tracks
    cols = ['flux_fit', 'qfit', 'reduced_chi2', 'n_pixels_fit', 'flux_err']
    track = {c: np.full((len(top), len(SIZES)), np.nan) for c in cols}
    dra = np.full((len(top), len(SIZES)), np.nan)   # position shift vs ref, mas
    for j, size in enumerate(SIZES):
        m = match_to_ref(top, results[size])
        tab = results[size]
        for k, mi in enumerate(m):
            if mi is None:
                continue
            for c in cols:
                v = tab[c][mi]
                track[c][k, j] = float(v) if v is not None and np.isfinite(
                    float(v) if v is not np.ma.masked else np.nan) else np.nan
            sc = SkyCoord(tab['skycoord_fit'][mi])
            dra[k, j] = sc_ref[k].separation(sc).to(u.mas).value

    jref = SIZES.index(REF_SIZE)
    flux_ref = track['flux_fit'][:, jref]
    dflux = 100 * (track['flux_fit'] - flux_ref[:, None]) / flux_ref[:, None]

    # ---- text summary ----
    print('\n' + '=' * 90)
    print('WALL-CLOCK and ACCEPTED-COUNT vs fit footprint (whole-frame run, 485 satstars)')
    print('=' * 90)
    print(f'  {"size":>5} {"seconds":>9} {"rel_time":>9} {"n_accepted":>11}')
    for size in SIZES:
        print(f'  {size:5d} {times[size]:9.0f} {times[size]/times[REF_SIZE]:9.2f} {counts[size]:11d}')

    print('\n' + '=' * 90)
    print(f'TOP-{len(top)} STAR FITS vs footprint (medians over the tracked stars)')
    print('=' * 90)
    print(f'  {"size":>5} {"|dflux|% med":>12} {"|dflux|% 90p":>12} '
          f'{"dpos mas med":>12} {"qfit med":>9} {"n_pix med":>9} {"n_matched":>9}')
    for j, size in enumerate(SIZES):
        adf = np.abs(dflux[:, j])
        nmatch = int(np.isfinite(track['flux_fit'][:, j]).sum())
        print(f'  {size:5d} {np.nanmedian(adf):12.2f} {np.nanpercentile(adf, 90):12.2f} '
              f'{np.nanmedian(dra[:, j]):12.1f} {np.nanmedian(track["qfit"][:, j]):9.3f} '
              f'{np.nanmedian(track["n_pixels_fit"][:, j]):9.0f} {nmatch:9d}')

    print('\n  per-star flux %-change vs ref (rows=stars by flux desc, cols=sizes):')
    print('     flux_ref    ' + ' '.join(f'{s:>7d}' for s in SIZES))
    for k in range(len(top)):
        print(f'   {flux_ref[k]:10.3g}  ' + ' '.join(
            f'{dflux[k, j]:7.1f}' for j in range(len(SIZES))))

    # ---- figure ----
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    a = ax[0, 0]
    for k in range(len(top)):
        a.plot(SIZES, dflux[k], '-', alpha=0.35, color='tab:blue')
    a.plot(SIZES, np.nanmedian(dflux, axis=0), 'k-o', lw=2, label='median')
    a.axhline(0, color='0.5', lw=1); a.axhline(2, color='r', ls=':'); a.axhline(-2, color='r', ls=':')
    a.set_xlabel('fit footprint size (px)'); a.set_ylabel('flux change vs ref (%)')
    a.set_title('(a) flux vs fit footprint (each line = one top star)')
    a.set_ylim(-15, 15); a.legend()

    a = ax[0, 1]
    for k in range(len(top)):
        a.plot(SIZES, dra[k], '-', alpha=0.35, color='tab:green')
    a.plot(SIZES, np.nanmedian(dra, axis=0), 'k-o', lw=2, label='median')
    a.set_xlabel('fit footprint size (px)'); a.set_ylabel('position shift vs ref (mas)')
    a.set_title('(b) centroid stability'); a.legend()

    a = ax[1, 0]
    a.plot(SIZES, [np.nanmedian(track['qfit'][:, j]) for j in range(len(SIZES))],
           'k-o', label='qfit (median)')
    a.plot(SIZES, [np.nanmedian(track['reduced_chi2'][:, j]) for j in range(len(SIZES))],
           'r-s', label='reduced_chi2 (median)')
    a.set_xlabel('fit footprint size (px)'); a.set_ylabel('fit quality')
    a.set_title('(c) fit-quality metrics'); a.legend()

    a = ax[1, 1]
    a.plot(SIZES, [times[s] for s in SIZES], 'k-o', label='wall-clock (s)')
    a.set_xlabel('fit footprint size (px)'); a.set_ylabel('seconds (whole frame)')
    a.set_title('(d) cost vs footprint (485-satstar frame)')
    a2 = a.twinx()
    a2.plot(SIZES, [counts[s] for s in SIZES], 'b--^', label='n accepted')
    a2.set_ylabel('n accepted', color='b')
    a.legend(loc='upper left')

    fig.suptitle(f'Satstar fit vs FITTING footprint (pad/model/mask fixed) — '
                 f'{os.path.basename(CRF)[:32]}', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    p = os.path.join(OUT, 'fit_footprint_sweep.png')
    fig.savefig(p, dpi=120)
    print(f'\n[sweep] wrote {p}')

    Table({'size': SIZES,
           'seconds': [times[s] for s in SIZES],
           'n_accepted': [counts[s] for s in SIZES]}).write(
        os.path.join(OUT, 'sweep_summary.ecsv'), overwrite=True)
    print('[sweep] DONE')


if __name__ == '__main__':
    main()
