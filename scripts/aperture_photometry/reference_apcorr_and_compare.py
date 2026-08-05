"""Contamination-free aperture correction (isolated unsaturated reference stars)
and comparison to (a) the satstar-derived apcorr and (b) Rob Gutermuth's
synthetic-PSF apcorr values.

Rob's method (rguter_jwst_photometry, manuscript): IDL PhotVis + aper, aperture
radius = FWHM per band, sky annulus ~1.1-2.5x FWHM, aperture corrections from
SYNTHETIC WebbPSF/JDocs PSFs (NOT an empirical curve of growth), Vega mags.  He
KEEPS saturated pixels NaN (WCSmosaic) and rejects Stage-3 mosaics because S3
interpolates saturated cores.  Our i2d ARE Stage-3 -> a caveat we test here.

For Rob's field (Cloud E&F = our `cloudef`, prop 2092) in his 4 NIRCam filters we
build the clean empirical COG and compare enclosed-energy at HIS aperture radius
to his AperCorr.  We also fix the brick wide-band apcorr that was crowding-
contaminated when derived from the satstar catalog.
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
from astropy.table import Table

from jwst_gc_pipeline.photometry import aperture_photometry as apm

OUT = os.environ.get(
    'APREF_OUT',
    '/blue/adamginsburg/adamginsburg/tmp/claude-3663/-orange-adamginsburg-jwst/'
    '81778923-7a53-4903-85de-4e1a21cfef0f/scratchpad/wt-apphot/docs/reports/apphot')

# Rob Gutermuth's Table (aperture radius = FWHM(mas); AperCorr = flux fraction
# within that aperture from synthetic PSFs; apcorr_mag = -2.5log10(AperCorr)).
ROB = {
    'f162m': dict(fwhm_as=0.074, aper_as=0.074, sky_in=0.081, sky_out=0.185, apcorr_frac=0.534),
    'f210m': dict(fwhm_as=0.085, aper_as=0.085, sky_in=0.093, sky_out=0.212, apcorr_frac=0.513),
    'f360m': dict(fwhm_as=0.157, aper_as=0.157, sky_in=0.173, sky_out=0.393, apcorr_frac=0.541),
    'f480m': dict(fwhm_as=0.195, aper_as=0.195, sky_in=0.215, sky_out=0.488, apcorr_frac=0.550),
}

REQUIRED = ('skycoord', 'flux', 'flux_err')


def find_main_catalog(target, filtername):
    """Pick a merged photometry catalog carrying skycoord+flux(+is_saturated)."""
    cdir = f'/orange/adamginsburg/jwst/{target}/catalogs'
    pats = [
        f'{cdir}/{filtername.lower()}_merged_indivexp_merged*iterative*.fits',
        f'{cdir}/{filtername.lower()}_merged_indivexp_merged*allcols*.fits',
        f'{cdir}/{filtername.lower()}_merged_indivexp_merged*dao*.fits',
        f'{cdir}/*{filtername.lower()}*dao*.fits',
    ]
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


def run(target, filtername, extra_radii=()):
    base = f'/orange/adamginsburg/jwst/{target}/'
    i2ds = apm.find_i2d_mosaics(filtername, target, base)
    if not i2ds:
        print(f'[skip] {target}/{filtername}: no i2d'); return None
    mcat = find_main_catalog(target, filtername)
    if mcat is None:
        print(f'[skip] {target}/{filtername}: no main catalog'); return None
    cat = Table.read(mcat)
    radii = tuple(sorted(set(apm.RADII_ARCSEC) | set(extra_radii) | {1.5, 2.0}))
    ref_tbl = apm.build_reference_apcorr(cat, i2ds, filtername,
                                         snr_min=50.0, isolation_arcsec=3.0,
                                         radii_arcsec=radii)
    # write the contamination-free table separately (distinct suffix)
    path = (f'{base}/catalogs/{filtername.lower()}_satstar_apcorr_refstars.ecsv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ref_tbl.write(path, overwrite=True, format='ascii.ecsv')
    print(f'[ok] {target}/{filtername}: refstars={ref_tbl.meta["n_reference_stars"]} '
          f'clean={ref_tbl.meta["n_clean_stars"]} -> {os.path.basename(path)}')
    return dict(target=target, filt=filtername, tbl=ref_tbl, path=path,
                mcat=os.path.basename(mcat))


def _ratio_at(tbl, r_as):
    """flux_ratio_to_total interpolated at radius r_as."""
    r = np.asarray(tbl['radius_arcsec'], float)
    y = np.asarray(tbl['flux_ratio_to_total'], float)
    return float(np.interp(r_as, r, y))


def main():
    os.makedirs(OUT, exist_ok=True)
    results = []
    # Rob's field/filters (apples-to-apples): cloudef NIRCam medium bands
    for f in ('f162m', 'f210m', 'f360m', 'f480m'):
        r = run('cloudef', f, extra_radii=(ROB[f]['aper_as'],))
        if r:
            results.append(r)
    # brick wide bands that were crowding-contaminated from the satstar cat
    for f in ('f200w', 'f356w'):
        r = run('brick', f)
        if r:
            results.append(r)

    # comparison table
    rows = []
    for r in results:
        f = r['filt']
        row = dict(target=r['target'], filt=f,
                   n_ref=r['tbl'].meta['n_reference_stars'],
                   n_clean=r['tbl'].meta['n_clean_stars'],
                   ee_0p3_ref=_ratio_at(r['tbl'], 0.3))
        # satstar-derived apcorr (contaminated) for contrast
        sp = apm.apcorr_table_path(f, r['target'], f"/orange/adamginsburg/jwst/{r['target']}/")
        if os.path.exists(sp):
            st = Table.read(sp)
            row['ee_0p3_satstar'] = _ratio_at(st, 0.3)
        if f in ROB:
            ra = ROB[f]['aper_as']
            row['rob_aper_as'] = ra
            row['ee_robrad_ours'] = _ratio_at(r['tbl'], ra)   # rel to our 2.0" ref
            row['rob_apcorr_frac'] = ROB[f]['apcorr_frac']    # rel to infinite PSF
        rows.append(row)
    comp = Table(rows)
    comp.write(f'{OUT}/apcorr_reference_vs_satstar_vs_rob.ecsv', overwrite=True,
               format='ascii.ecsv')
    comp.pprint_all()

    # figure: clean COG per field/filter + Rob's point
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for r in results:
        t = r['tbl']
        ax[0].plot(t['radius_arcsec'], t['flux_ratio_to_total'], 'o-',
                   label=f"{r['target']} {r['filt']} (N={t.meta['n_reference_stars']})")
    for f in ('f162m', 'f210m', 'f360m', 'f480m'):
        ax[0].scatter(ROB[f]['aper_as'], ROB[f]['apcorr_frac'], marker='*',
                      s=180, edgecolor='k', zorder=5,
                      label=f'Rob {f} (synthetic PSF)' if f == 'f162m' else None)
    ax[0].set_xlabel('aperture radius (arcsec)')
    ax[0].set_ylabel('enclosed flux / total (2.0" ref)')
    ax[0].set_title('(a) clean empirical curve of growth vs Rob (synthetic)')
    ax[0].axhline(1.0, color='0.7', lw=0.8); ax[0].legend(fontsize=7)
    # panel b: reference vs satstar apcorr at 0.3"
    labels, ee_ref, ee_sat = [], [], []
    for row in rows:
        labels.append(f"{row['target']}\n{row['filt']}")
        ee_ref.append(row.get('ee_0p3_ref', np.nan))
        ee_sat.append(row.get('ee_0p3_satstar', np.nan))
    x = np.arange(len(labels))
    ax[1].bar(x - 0.2, ee_ref, 0.4, label='reference (clean)', color='tab:green')
    ax[1].bar(x + 0.2, ee_sat, 0.4, label='satstar (contaminated)', color='tab:red')
    ax[1].set_xticks(x); ax[1].set_xticklabels(labels, fontsize=7)
    ax[1].set_ylabel('EE(0.3") / total'); ax[1].legend(fontsize=8)
    ax[1].set_title('(b) clean vs satstar-derived apcorr @0.3"')
    fig.tight_layout()
    fig.savefig(f'{OUT}/apcorr_reference_vs_satstar_vs_rob.png', dpi=120)
    print(f'\nwrote {OUT}/apcorr_reference_vs_satstar_vs_rob.{{ecsv,png}}')


if __name__ == '__main__':
    main()
