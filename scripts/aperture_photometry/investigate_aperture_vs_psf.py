"""Investigation: i2d aperture photometry vs PSF flux for saturated stars.

For a matrix of fields/filters spanning the saturation/crowding/background axes:
  * read the existing consolidated satstar catalog (READ-ONLY -- does not touch
    the production cache),
  * measure aperture photometry from the i2d mosaic (aperture_photometry module),
  * build + WRITE the aperture-correction table (separate file, catalogs/),
  * compare aperture flux (corrected to the reference radius) to the PSF flux_fit,
  * emit a per-field diagnostic figure + a master summary table/figure.

Run:  python investigate_aperture_vs_psf.py
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

from jwst_gc_pipeline.photometry import aperture_photometry as apm

OUT = os.environ.get(
    'APINV_OUT',
    '/blue/adamginsburg/adamginsburg/tmp/claude-3663/-orange-adamginsburg-jwst/'
    '81778923-7a53-4903-85de-4e1a21cfef0f/scratchpad/wt-apphot/docs/reports/apphot')

# (target, filter, short-regime label)
MATRIX = [
    ('brick', 'f200w', 'GC crowd SW-wide NG7'),
    ('brick', 'f212n', 'GC crowd SW-narrow NG7'),
    ('brick', 'f356w', 'GC crowd LW-wide'),
    ('brick', 'f405n', 'GC crowd LW-narrow'),
    ('gc2211', 'f200w', 'super-sat SW-wide NG2'),
    ('gc2211', 'f277w', 'super-sat LW-wide NG2'),
    ('ngc6334', 'f200w', 'ext-emission SW-wide'),
    ('ngc6334', 'f187n', 'ext-emission SW-narrow'),
    ('sgra', 'f405n', 'GC LW-narrow'),
    ('m4', 'f322w2', 'globular no-bkg LW'),
    ('ngc6397', 'f322w2', 'globular no-bkg LW'),
]


def _frame_pixel_area_deg2(target, filtername, i2d_pix_area):
    """crf/frame pixel solid angle for flux_fit->Jy.  The satstar PSF fits run
    on the per-exposure frames; use one frame's WCS.  Falls back to the i2d
    pixel area (native detector scale mosaics match the frame scale closely)."""
    pipe = f'/orange/adamginsburg/jwst/{target}/{filtername.upper()}/pipeline'
    cands = sorted(glob.glob(f'{pipe}/*_crf.fits')
                   + glob.glob(f'{pipe}/*_cal.fits'))
    for fn in cands[:1]:
        try:
            with fits.open(fn) as h:
                return WCS(h['SCI'].header).proj_plane_pixel_area().to(u.deg**2)
        except (OSError, KeyError):
            continue
    return i2d_pix_area


def run_field(target, filtername, label):
    base = f'/orange/adamginsburg/jwst/{target}/'
    cpath = (f'{base}/catalogs/{filtername.lower()}'
             f'_consolidated_satstar_catalog.fits')
    if not os.path.exists(cpath):
        print(f'[skip] {target}/{filtername}: no consolidated cat')
        return None
    cat = Table.read(cpath)
    cat.meta.setdefault('filter', filtername)
    i2ds = apm.find_i2d_mosaics(filtername, target, base)
    if not i2ds:
        print(f'[skip] {target}/{filtername}: no i2d mosaic')
        return None
    out = apm.measure_aperture_photometry(cat, i2ds, filtername=filtername)
    if not apm.has_aperture_columns(out):
        print(f'[skip] {target}/{filtername}: aperture measure failed')
        return None

    # aperture-correction table (separate deliverable) -> catalogs/
    try:
        tbl = apm.build_aperture_correction_table(out, filtername=filtername)
        apm.write_aperture_correction_table(tbl, filtername, target, base)
    except ValueError as ex:
        print(f'[warn] {target}/{filtername}: apcorr table failed: {ex}')
        tbl = None

    # i2d pixel area (aperture already in Jy); frame pixel area for PSF->Jy
    with fits.open(i2ds[0]) as h:
        i2d_pa = WCS(h['SCI'].header).proj_plane_pixel_area().to(u.deg**2)
    frame_pa = _frame_pixel_area_deg2(target, filtername, i2d_pa)
    psf_jy = (np.asarray(cat['flux_fit'], float) * u.MJy / u.sr
              * frame_pa).to(u.Jy).value

    aper = np.asarray(out['aper_flux_jy'], float)          # primary radius, Jy
    cov = np.asarray(out['aper_area_frac'], float)
    core = np.asarray(out['aper_core_saturated'], bool)
    # correct the primary aperture to the reference radius using the field's own
    # curve of growth (ratio_to_total at the primary radius)
    prim = out.meta['APER_PRIM']
    ratio_prim = 1.0
    if tbl is not None:
        m = np.isclose(np.asarray(tbl['radius_arcsec'], float), prim)
        if m.any():
            ratio_prim = float(tbl['flux_ratio_to_total'][m][0])
    aper_tot = aper / ratio_prim if ratio_prim else aper

    good = (np.isfinite(aper) & np.isfinite(psf_jy) & (aper > 0) & (psf_jy > 0)
            & (cov > 0.9))
    good_clean = good & ~core                       # aperture not core-saturated
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = aper_tot / psf_jy
    summ = dict(
        target=target, filt=filtername, label=label, n=len(cat),
        n_measured=int(np.isfinite(aper).sum()),
        frac_core_sat=float(core.mean()),
        n_compare=int(good_clean.sum()),
        apcorr_prim_mag=(-2.5 * np.log10(ratio_prim) if ratio_prim > 0 else np.nan),
        med_aper_over_psf=float(np.nanmedian(ratio[good_clean]))
        if good_clean.any() else np.nan,
        scatter_dex=float(np.nanstd(np.log10(ratio[good_clean])))
        if good_clean.any() else np.nan,
    )

    # per-field figure: (a) aper_tot vs PSF, (b) curve of growth
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    a = ax[0]
    psf_mag = -2.5 * np.log10(psf_jy)
    a.scatter(psf_mag[good_clean], (ratio[good_clean]), s=5, alpha=0.2,
              color='0.4', label='non-core-sat')
    gcs = good & core
    a.scatter(psf_mag[gcs], ratio[gcs], s=5, alpha=0.3, color='tab:red',
              label='core-saturated')
    a.axhline(1.0, color='k', lw=0.8)
    a.set_xlabel('PSF instrumental mag (-2.5log10 flux_jy)')
    a.set_ylabel('aper(corrected-to-total) / PSF')
    a.set_ylim(0, 2)
    a.set_title(f'(a) {target} {filtername}: aperture/PSF')
    a.legend(fontsize=8)
    a = ax[1]
    if tbl is not None:
        a.plot(tbl['radius_arcsec'], tbl['flux_ratio_to_total'], 'o-')
        a.set_xlabel('aperture radius (arcsec)')
        a.set_ylabel('enclosed flux / total (ref radius)')
        a.set_title(f"(b) curve of growth (N={tbl.meta['n_clean_stars']} clean)")
        a.axhline(1.0, color='0.6', lw=0.8)
    fig.suptitle(f'{target} {filtername} — {label}', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(OUT, exist_ok=True)
    p = f'{OUT}/apphot_{target}_{filtername}.png'
    fig.savefig(p, dpi=110)
    plt.close(fig)
    print(f'[ok] {target}/{filtername}: N={summ["n"]} '
          f"core-sat={summ['frac_core_sat']:.1%} "
          f"aper/PSF med={summ['med_aper_over_psf']:.3f} "
          f"scatter={summ['scatter_dex']:.2f}dex apcorr={summ['apcorr_prim_mag']:.2f}mag")
    return summ


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for t, f, lbl in MATRIX:
        r = run_field(t, f, lbl)
        if r:
            rows.append(r)
    if not rows:
        print('no fields succeeded')
        return
    st = Table(rows)
    st.write(f'{OUT}/aperture_vs_psf_summary.ecsv', overwrite=True,
             format='ascii.ecsv')
    st.pprint_all()

    # master figure: aper/PSF and scatter per field/filter
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    labels = [f"{r['target']}\n{r['filt']}" for r in rows]
    x = np.arange(len(rows))
    ax[0].bar(x, [r['med_aper_over_psf'] for r in rows], color='tab:blue')
    ax[0].axhline(1.0, color='k', lw=0.8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax[0].set_ylabel('median aper(corr)/PSF'); ax[0].set_title('(a) aperture vs PSF')
    ax[1].bar(x, [r['frac_core_sat'] * 100 for r in rows], color='tab:red')
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax[1].set_ylabel('% satstars core-saturated in i2d')
    ax[1].set_title('(b) mosaic core-saturation fraction')
    fig.tight_layout()
    fig.savefig(f'{OUT}/aperture_vs_psf_master.png', dpi=120)
    print(f'\nwrote {OUT}/aperture_vs_psf_summary.ecsv + master figure')


if __name__ == '__main__':
    main()
