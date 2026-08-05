"""Contamination-free aperture correction (isolated unsaturated reference stars),
compared to (a) the satstar-derived apcorr, (b) the THEORETICAL PSF-grid curve of
growth, and (c) Rob Gutermuth's synthetic-PSF apcorr values.

Rob's method (rguter_jwst_photometry manuscript): IDL PhotVis + aper, aperture
radius = FWHM, sky annulus ~1.1-2.5x FWHM, aperture corrections from SYNTHETIC
WebbPSF/JDocs PSFs (not empirical), Vega mags.  He keeps saturated pixels NaN
(WCSmosaic) and REJECTS Stage-3 mosaics because S3 interpolates saturated cores
(false Airy-ring sources + a 20-30% flux-deficit ring around bright stars).  Our
i2d ARE Stage-3, so the empirical-vs-theoretical curve-of-growth comparison here
directly tests whether that S3 artifact biases our aperture photometry.

Key measured fact (see report): our GC/CMZ/globular fields are so crowded that
essentially no stars are isolated beyond ~0.3-0.5", so an empirical curve of
growth to "total" is impossible from the data -- the isolation-limited clean COG
reaches only ~0.25", and the outer correction must come from a theoretical PSF
(exactly Rob's choice).  We therefore compare EMPIRICAL vs THEORETICAL COG in the
clean inner range.
"""
import os
import glob
import warnings

os.environ.setdefault('CRDS_PATH', '/orange/adamginsburg/jwst/crds')
os.environ.setdefault('STPSF_PATH', '/orange/adamginsburg/jwst/stpsf-data/')
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
import astropy.units as u
from photutils.aperture import CircularAperture, aperture_photometry

from jwst_gc_pipeline.photometry import aperture_photometry as apm

OUT = os.environ.get(
    'APREF_OUT',
    '/blue/adamginsburg/adamginsburg/tmp/claude-3663/-orange-adamginsburg-jwst/'
    '81778923-7a53-4903-85de-4e1a21cfef0f/scratchpad/wt-apphot/docs/reports/apphot')

# Rob's Table (aper radius = FWHM; AperCorr = flux fraction within it from
# synthetic PSFs, i.e. EE(FWHM)/EE(infinity)).
ROB = {
    'f162m': dict(aper_as=0.074, apcorr_frac=0.534),
    'f210m': dict(aper_as=0.085, apcorr_frac=0.513),
    'f360m': dict(aper_as=0.157, apcorr_frac=0.541),
    'f480m': dict(aper_as=0.195, apcorr_frac=0.550),
}
REQUIRED = ('skycoord', 'flux', 'flux_err')
COG_RADII = (0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25)


def find_main_catalog(target, filtername):
    cdir = f'/orange/adamginsburg/jwst/{target}/catalogs'
    pats = [f'{cdir}/{filtername.lower()}_merged_indivexp_merged*iterative*.fits',
            f'{cdir}/{filtername.lower()}_merged_indivexp_merged*allcols*.fits',
            f'{cdir}/{filtername.lower()}_merged_indivexp_merged*dao*.fits']
    for p in pats:
        for fn in sorted(glob.glob(p), key=os.path.getsize, reverse=True):
            if any(t in fn for t in ('satstar', 'reference', 'apcorr')):
                continue
            try:
                cols = Table.read(fn, memmap=True).colnames
            except (OSError, ValueError):
                continue
            if all(c in cols for c in REQUIRED):
                return fn
    return None


def theoretical_cog(target, filtername, radii_arcsec, pixscale_as):
    """Enclosed-energy curve of growth from the field's STPSF PSF grid
    (oversample1 = detector sampling), normalised to the grid-box total.
    Returns dict radius->EE fraction, or None if no grid."""
    pdir = f'/orange/adamginsburg/jwst/{target}/psfs'
    g = sorted(glob.glob(f'{pdir}/{filtername.upper()}_*PSFgrid_oversample1.fits'))
    if not g:
        return None
    with fits.open(g[0]) as h:
        cube = np.asarray(h[0].data, float)      # (Ngrid, ny, nx)
    psf = np.nanmean(cube, axis=0)
    psf = psf / np.nansum(psf)
    ny, nx = psf.shape
    cy, cx = (ny - 1) / 2, (nx - 1) / 2
    pos = [(cx, cy)]
    tot = np.nansum(psf)
    ee = {}
    for r in radii_arcsec:
        rp = r / pixscale_as
        if rp > min(cx, cy):
            ee[r] = np.nan
            continue
        ap = CircularAperture(pos, r=rp)
        ee[r] = float(aperture_photometry(psf, ap)['aperture_sum'][0] / tot)
    return ee


def run(target, filtername):
    base = f'/orange/adamginsburg/jwst/{target}/'
    i2ds = apm.find_i2d_mosaics(filtername, target, base)
    if not i2ds:
        print(f'[skip] {target}/{filtername}: no i2d'); return None
    mcat = find_main_catalog(target, filtername)
    if mcat is None:
        print(f'[skip] {target}/{filtername}: no main catalog'); return None
    cat = Table.read(mcat)
    tbl = apm.build_reference_apcorr(cat, i2ds, filtername, snr_min=50.0,
                                     radii_arcsec=COG_RADII, min_ref_stars=200)
    path = f'{base}/catalogs/{filtername.lower()}_satstar_apcorr_refstars.ecsv'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tbl.write(path, overwrite=True, format='ascii.ecsv')
    with fits.open(i2ds[0]) as h:
        pixscale_as = np.sqrt(WCS(h['SCI'].header).proj_plane_pixel_area()
                              .to(u.arcsec**2).value)
    theo = theoretical_cog(target, filtername, COG_RADII, pixscale_as)
    print(f"[ok] {target}/{filtername}: N_ref={tbl.meta['n_reference_stars']} "
          f"iso={tbl.meta['isolation_arcsec']}\" "
          f"reliable<={tbl.meta['reliable_max_radius_arcsec']:.2f}\" "
          f"theo_grid={'yes' if theo else 'no'}")
    return dict(target=target, filt=filtername, tbl=tbl, theo=theo,
                pixscale=pixscale_as)


def _interp(tbl, r):
    return float(np.interp(r, np.asarray(tbl['radius_arcsec'], float),
                           np.asarray(tbl['flux_ratio_to_total'], float)))


def main():
    os.makedirs(OUT, exist_ok=True)
    results = []
    for f in ('f162m', 'f210m', 'f360m', 'f480m'):     # Rob's field/filters
        r = run('cloudef', f)
        if r:
            results.append(r)
    for f in ('f200w', 'f356w'):                        # brick wide (was contaminated)
        r = run('brick', f)
        if r:
            results.append(r)

    rows = []
    for r in results:
        f, t = r['filt'], r['tbl']
        rmax = t.meta['reliable_max_radius_arcsec']
        row = dict(target=r['target'], filt=f, n_ref=t.meta['n_reference_stars'],
                   iso_as=t.meta['isolation_arcsec'], reliable_rmax=round(rmax, 3))
        # empirical vs theoretical EE, both normalised to rmax (clean range)
        if r['theo'] and np.isfinite(r['theo'].get(rmax, np.nan)):
            theo_rmax = r['theo'][rmax]
            for rr in (0.10, 0.15):
                if rr <= rmax:
                    emp = _interp(t, rr)                       # already /rmax
                    th = r['theo'][rr] / theo_rmax            # /rmax
                    row[f'emp_EE_{apm._rtag(rr)}'] = round(emp, 3)
                    row[f'theo_EE_{apm._rtag(rr)}'] = round(th, 3)
                    row[f'emp_over_theo_{apm._rtag(rr)}'] = round(emp / th, 3)
        # satstar-derived (contaminated) apcorr at 0.15" for contrast
        sp = apm.apcorr_table_path(f, r['target'], f"/orange/adamginsburg/jwst/{r['target']}/")
        if os.path.exists(sp):
            st = Table.read(sp)
            row['satstar_EE_0p15_norm1p0'] = round(_interp(st, 0.15), 3)
        if f in ROB:
            row['rob_aper_as'] = ROB[f]['aper_as']
            row['rob_frac_inf'] = ROB[f]['apcorr_frac']
        rows.append(row)
    comp = Table(rows)
    comp.write(f'{OUT}/apcorr_reference_vs_theory_vs_rob.ecsv', overwrite=True,
               format='ascii.ecsv')
    comp.pprint_all()

    # figure: empirical vs theoretical COG per field/filter
    ncol = 3
    nrow = int(np.ceil(len(results) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.6 * nrow),
                             squeeze=False)
    for i, r in enumerate(results):
        ax = axes[i // ncol][i % ncol]
        t = r['tbl']
        rmax = t.meta['reliable_max_radius_arcsec']
        ax.plot(t['radius_arcsec'], t['flux_ratio_to_total'], 'o-',
                color='tab:green', label='empirical (i2d, clean)')
        if r['theo']:
            th_rmax = r['theo'].get(rmax, np.nan)
            rr = sorted([k for k in r['theo'] if np.isfinite(r['theo'][k])])
            yy = [r['theo'][k] / th_rmax for k in rr]
            ax.plot(rr, yy, 's--', color='tab:blue', label='theoretical PSF grid')
        if r['filt'] in ROB:
            ax.axvline(ROB[r['filt']]['aper_as'], color='0.7', ls=':', lw=0.8)
        ax.set_title(f"{r['target']} {r['filt']}  N={t.meta['n_reference_stars']} "
                     f"iso={t.meta['isolation_arcsec']}\"", fontsize=8)
        ax.set_xlabel('radius (arcsec)'); ax.set_ylabel(f'EE / EE({rmax:.2f}")')
        ax.axhline(1.0, color='0.8', lw=0.7); ax.legend(fontsize=6)
    for j in range(len(results), nrow * ncol):
        axes[j // ncol][j % ncol].axis('off')
    fig.suptitle('Contamination-free empirical curve of growth vs theoretical PSF '
                 '(normalised in the clean inner range)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(f'{OUT}/apcorr_reference_vs_theory.png', dpi=120)
    print(f'\nwrote {OUT}/apcorr_reference_vs_theory{{.png,_vs_rob.ecsv}}')


if __name__ == '__main__':
    main()
