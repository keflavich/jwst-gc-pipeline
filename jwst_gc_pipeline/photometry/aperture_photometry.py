"""Aperture photometry of (saturated) stars from the i2d mosaics.

The saturated-star catalogs carry a PSF-wing-fit flux (``flux_fit``) measured on
the per-exposure crf frames with the saturated core masked.  This module adds an
INDEPENDENT circular-aperture flux measured directly on the resampled mosaic
(i2d, ``MJy/sr``), so the two can be compared.  It is deliberately kept separate
from the PSF flux: aperture columns are *added* to the catalog, but the aperture
correction (curve-of-growth) tables are written to their own files.

Design notes
------------
* Saturated cores are frequently NaN/masked in the mosaic too, so every sum is
  NaN-aware (``ApertureStats``/``aperture_photometry`` with a finite-pixel mask)
  and each measurement carries a coverage fraction plus a core-saturated flag.
* Radii are defined in ARCSEC and converted to the mosaic's own pixel grid, so
  SW/LW and different resamplings are directly comparable.
* Flux is converted MJy/sr -> Jy with the mosaic's pixel solid angle
  (``ww.proj_plane_pixel_area()``), the same conversion the merge uses for
  ``flux_fit`` (merge_catalogs), so aperture-Jy and PSF-Jy are on one system.
* Vega/AB zeropoints follow merge_catalogs: SVO ZeroPoint (Jy) and
  ``ABMAG_OFFSET = 8.90``.
"""
import os
import re
import glob

import numpy as np
from astropy import log
from astropy import units as u
from astropy.io import fits
from astropy.table import Table, Column
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clipped_stats

from photutils.aperture import CircularAperture, CircularAnnulus, ApertureStats

ABMAG_OFFSET = 8.90

# Curve-of-growth radii (arcsec).  Spans the PSF core (~0.05-0.1") out past the
# first Airy rings so the aperture correction can be measured.  The PRIMARY
# aperture (the one promoted to ``aper_flux_jy``) is 0.3".
RADII_ARCSEC = (0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00)
PRIMARY_RADIUS_ARCSEC = 0.30
# Local-sky annulus (arcsec).  Well outside the primary aperture; wide enough to
# beat down crowding noise but not so wide it wanders off small mosaics.
ANNULUS_IN_ARCSEC = 1.20
ANNULUS_OUT_ARCSEC = 1.80


def _rtag(r_arcsec):
    """Column-name tag for a radius in arcsec: 0.30 -> '0p30'."""
    return f"{r_arcsec:.2f}".replace('.', 'p')


def has_aperture_columns(catalog):
    return 'aper_flux_jy' in getattr(catalog, 'colnames', [])


_JFILTS = None


def _vega_zeropoint_jy(filtername):
    """SVO Vega ZeroPoint (Jy) for a JWST filter, matching merge_catalogs."""
    global _JFILTS
    if _JFILTS is None:
        from astroquery.svo_fps import SvoFps
        _JFILTS = SvoFps.get_filter_list('JWST')
        _JFILTS.add_index('filterID')
    from jwst_gc_pipeline.photometry.naming import _svo_filter_id
    return u.Quantity(_JFILTS.loc[_svo_filter_id(filtername)]['ZeroPoint'], u.Jy)


def find_i2d_mosaics(filtername, target, basepath):
    """Return the merged i2d mosaic(s) for a target/filter.

    NIRCam single-obs targets have one ``..._clear-<filt>-merged_i2d.fits``;
    multi-obs targets (gc2211) have one merged i2d per observation.  Returns a
    sorted list (possibly empty).
    """
    from jwst_gc_pipeline.photometry.merge_catalogs import (
        project_obsnum, _project_for_target_filter)
    from jwst_gc_pipeline.photometry.naming import _inst_token
    filt_l = filtername.lower()
    inst = _inst_token(filtername)
    pipe = f'{basepath}/{filtername.upper()}/pipeline'
    pats = []
    try:
        proj = _project_for_target_filter(target, filtername)
    except KeyError:
        proj = None
    if inst == 'nircam' and proj is not None and target in project_obsnum \
            and proj in project_obsnum[target]:
        obs = project_obsnum[target][proj]
        pats.append(f'{pipe}/jw0{proj}-o{obs}_t001_nircam_clear-{filt_l}-merged_i2d.fits')
    # generic fallbacks (covers MIRI, non-registry targets, plain naming)
    pats += [
        f'{pipe}/jw*-o*_t001_{inst}_clear-{filt_l}-merged_i2d.fits',
        f'{pipe}/jw*-o*_t001_{inst}_*{filt_l}*_i2d.fits',
    ]
    # Reject derived/intermediate i2d products that share the mosaic naming but
    # are NOT the science mosaic: pipeline residual/model images, downselected
    # catalogs, filtered/reprojected variants, and per-iteration products.
    _reject = ('residual', 'model', 'mergedcat', 'smoothed', '_data_',
               'medfilt', 'reprj', 'resbgsub', 'downsel', 'daophot', '_cat')
    seen, out = set(), []
    for p in pats:
        for m in sorted(glob.glob(p)):
            b = os.path.basename(m)
            # keep only observation-level mosaics (jw<prop>-o<obs>_t001...)
            if '_t001_' not in b:
                continue
            if any(tok in b for tok in _reject):
                continue
            # the segment between _t001_ and _i2d must not carry an iteration
            # token (_m2/_m3...) -- those are catalog-residual products
            mid = b.split('_t001_', 1)[1].rsplit('_i2d', 1)[0]
            if re.search(r'_m\d+(_|$)', mid):
                continue
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out


def _measure_one_mosaic(sky, i2d_path, radii_arcsec, primary_radius_arcsec,
                        ann_in_arcsec, ann_out_arcsec):
    """Measure aperture photometry for all positions on one mosaic.

    Returns a dict of per-star arrays (length == len(sky)); stars off this
    mosaic have NaN flux and coverage 0.
    """
    # Keep native dtype (i2d SCI/ERR are float32) -- do NOT upcast to float64;
    # these mosaics are multi-GB and this runs inside the memory-bound merge.
    with fits.open(i2d_path, memmap=False) as hdul:
        sci = np.asarray(hdul['SCI'].data, dtype=np.float32)
        try:
            err = np.asarray(hdul['ERR'].data, dtype=np.float32)
        except KeyError:
            err = None
        ww = WCS(hdul['SCI'].header)
        pix_area = ww.proj_plane_pixel_area().to(u.deg**2)
        pixscale_as = np.sqrt(pix_area.to(u.arcsec**2).value)  # arcsec/pix
    n = len(sky)
    x, y = ww.world_to_pixel(sky)
    ny, nx = sci.shape
    r_out_pix = ann_out_arcsec / pixscale_as
    inb = (np.isfinite(x) & np.isfinite(y)
           & (x > r_out_pix) & (x < nx - r_out_pix)
           & (y > r_out_pix) & (y < ny - r_out_pix))

    res = {'flux_jy': np.full(n, np.nan), 'flux_err_jy': np.full(n, np.nan),
           'bkg': np.full(n, np.nan), 'cov': np.zeros(n),
           'core_sat': np.zeros(n, bool), 'inb': inb}
    for r in radii_arcsec:
        res[f'cog_{_rtag(r)}'] = np.full(n, np.nan)
        res[f'cov_{_rtag(r)}'] = np.zeros(n)
    if inb.sum() == 0:
        return res, pix_area

    pos = np.column_stack([x[inb], y[inb]])
    finite = np.isfinite(sci)
    mjy_to_jy = (1 * u.MJy / u.sr * pix_area).to(u.Jy).value  # per (MJy/sr)*pix

    # core-saturated: central pixel non-finite
    xi = np.clip(np.round(x[inb]).astype(int), 0, nx - 1)
    yi = np.clip(np.round(y[inb]).astype(int), 0, ny - 1)
    res['core_sat'][inb] = ~finite[yi, xi]

    # local background from annulus (plain median, no sigma clip; per pixel, MJy/sr)
    ann = CircularAnnulus(pos, r_in=ann_in_arcsec / pixscale_as,
                          r_out=ann_out_arcsec / pixscale_as)
    annst = ApertureStats(sci, ann, mask=~finite, sigma_clip=None)
    bkg = np.asarray(annst.median, float)
    res['bkg'][inb] = bkg

    for r in radii_arcsec:
        rp = r / pixscale_as
        ap = CircularAperture(pos, r=rp)
        apst = ApertureStats(sci, ap, error=err, mask=~finite, sigma_clip=None)
        area = np.asarray(apst.sum_aper_area.value, float)
        cov = area / ap.area
        raw = np.asarray(apst.sum, float)
        flux = (raw - bkg * area) * mjy_to_jy
        tag = _rtag(r)
        res[f'cog_{tag}'][inb] = flux
        res[f'cov_{tag}'][inb] = cov
        if abs(r - primary_radius_arcsec) < 1e-6:
            res['flux_jy'][inb] = flux
            res['cov'][inb] = cov
            if err is not None:
                se = getattr(apst, 'sum_err', None)
                if se is not None:
                    res['flux_err_jy'][inb] = np.asarray(se, float) * mjy_to_jy
    return res, pix_area


def measure_aperture_photometry(catalog, i2d_paths, filtername=None,
                                radii_arcsec=RADII_ARCSEC,
                                primary_radius_arcsec=PRIMARY_RADIUS_ARCSEC,
                                ann_in_arcsec=ANNULUS_IN_ARCSEC,
                                ann_out_arcsec=ANNULUS_OUT_ARCSEC,
                                skycoord_col='skycoord_fit'):
    """Add i2d aperture-photometry columns to ``catalog`` (returns a copy).

    For each star the measurement from the mosaic with the highest primary-radius
    coverage is kept (handles multi-obs targets where mosaics overlap).
    """
    cat = catalog.copy()
    if skycoord_col not in cat.colnames:
        log.warning(f"aperture_photometry: no {skycoord_col!r} column; skipping")
        return cat
    sky = SkyCoord(cat[skycoord_col])
    n = len(cat)
    if primary_radius_arcsec not in radii_arcsec:
        radii_arcsec = tuple(sorted(set(radii_arcsec) | {primary_radius_arcsec}))

    best = None
    best_i2d = np.array(['' for _ in range(n)], dtype=object)
    pix_area = None
    for path in i2d_paths:
        res, pix_area = _measure_one_mosaic(
            sky, path, radii_arcsec, primary_radius_arcsec,
            ann_in_arcsec, ann_out_arcsec)
        if best is None:
            best = res
            best_i2d[res['inb']] = os.path.basename(path)
            continue
        # keep whichever mosaic gives higher primary-aperture coverage per star
        take = res['cov'] > best['cov']
        take |= (~np.isfinite(best['flux_jy']) & np.isfinite(res['flux_jy']))
        for k, v in res.items():
            if k == 'inb':
                continue
            best[k][take] = v[take]
        best_i2d[take] = os.path.basename(path)
    if best is None:
        log.warning("aperture_photometry: no usable i2d mosaic; skipping")
        return cat

    # zeropoint / magnitudes
    zp = None
    if filtername is None:
        filtername = cat.meta.get('filter', None)
    if filtername is not None:
        zp = _vega_zeropoint_jy(filtername)
    flux = best['flux_jy']
    with np.errstate(invalid='ignore', divide='ignore'):
        mag_ab = -2.5 * np.log10(flux) + ABMAG_OFFSET
        mag_vega = (-2.5 * np.log10(flux / zp.to(u.Jy).value)
                    if zp is not None else np.full(n, np.nan))

    cat['aper_flux_jy'] = Column(flux, unit=u.Jy,
                                 description='i2d circular-aperture flux (bkg-sub, primary radius)')
    cat['aper_flux_err_jy'] = Column(best['flux_err_jy'], unit=u.Jy)
    cat['aper_mag_vega'] = Column(mag_vega, unit=u.mag)
    cat['aper_mag_ab'] = Column(mag_ab, unit=u.mag)
    cat['aper_bkg'] = Column(best['bkg'], unit=u.MJy / u.sr,
                             description='annulus local sky (per pixel)')
    cat['aper_area_frac'] = Column(best['cov'],
                                   description='finite-pixel fraction in primary aperture')
    cat['aper_core_saturated'] = Column(best['core_sat'],
                                        description='central pixel NaN/masked in mosaic')
    cat['aper_i2d'] = Column(np.array([str(s) for s in best_i2d]))
    for r in radii_arcsec:
        tag = _rtag(r)
        cat[f'aper_flux_jy_r{tag}'] = Column(best[f'cog_{tag}'], unit=u.Jy)
        cat[f'aper_area_frac_r{tag}'] = Column(best[f'cov_{tag}'])

    cat.meta['APER_RAD'] = str(list(radii_arcsec))
    cat.meta['APER_PRIM'] = float(primary_radius_arcsec)
    cat.meta['APER_ANN'] = f'{ann_in_arcsec},{ann_out_arcsec}'
    cat.meta['APER_UNIT'] = 'arcsec'
    if zp is not None:
        cat.meta['APER_ZP'] = float(zp.to(u.Jy).value)
    return cat


def add_aperture_photometry(catalog, filtername, target, basepath, **kwargs):
    """Convenience: locate the i2d mosaic(s) and add aperture columns.

    Returns the catalog unchanged (with a log message) if disabled by
    ``SATSTAR_APERTURE_PHOT=0`` or if no mosaic is found.
    """
    if os.environ.get('SATSTAR_APERTURE_PHOT', '1') == '0':
        return catalog
    i2ds = find_i2d_mosaics(filtername, target, basepath)
    if not i2ds:
        log.warning(f"aperture_photometry: no i2d mosaic for {target}/{filtername} "
                    f"under {basepath}; leaving catalog without aperture columns")
        return catalog
    log.info(f"aperture_photometry: measuring {filtername} from "
             f"{len(i2ds)} mosaic(s)")
    return measure_aperture_photometry(catalog, i2ds, filtername=filtername,
                                       **kwargs)


def build_aperture_correction_table(catalog, filtername=None,
                                    radii_arcsec=RADII_ARCSEC,
                                    min_coverage=0.98, min_snr=20.0,
                                    isolation_arcsec=1.0,
                                    skycoord_col='skycoord_fit'):
    """Curve-of-growth aperture corrections from clean isolated stars.

    A star qualifies if, at every radius, its aperture is fully covered
    (coverage >= ``min_coverage``, not core-saturated) and it is isolated (no
    other catalog star within ``isolation_arcsec``) with adequate SNR at the
    primary radius.  The reference "total" is the largest radius.  Returns a
    small Table (one row per radius): median mag correction to total, ratio,
    scatter (MAD) and N -- to be written SEPARATELY from the photometry table.
    """
    if not has_aperture_columns(catalog):
        raise ValueError("catalog has no aperture columns; run "
                         "measure_aperture_photometry first")
    tags = [_rtag(r) for r in radii_arcsec]
    flux_cols = [f'aper_flux_jy_r{t}' for t in tags]
    cov_cols = [f'aper_area_frac_r{t}' for t in tags]
    missing = [c for c in flux_cols + cov_cols if c not in catalog.colnames]
    if missing:
        raise ValueError(f"catalog missing curve-of-growth columns: {missing}")

    fluxes = np.column_stack([np.asarray(catalog[c], float) for c in flux_cols])
    covs = np.column_stack([np.asarray(catalog[c], float) for c in cov_cols])
    ref = fluxes[:, -1]  # largest radius = "total"
    core_sat = np.asarray(catalog['aper_core_saturated'], bool) \
        if 'aper_core_saturated' in catalog.colnames else np.zeros(len(catalog), bool)

    clean = (np.all(covs >= min_coverage, axis=1)
             & np.all(np.isfinite(fluxes), axis=1)
             & (ref > 0) & ~core_sat)
    # SNR at primary radius (if available)
    if 'aper_flux_jy' in catalog.colnames and 'aper_flux_err_jy' in catalog.colnames:
        f0 = np.asarray(catalog['aper_flux_jy'], float)
        e0 = np.asarray(catalog['aper_flux_err_jy'], float)
        with np.errstate(invalid='ignore', divide='ignore'):
            snr = f0 / e0
        clean &= np.isfinite(snr) & (snr >= min_snr)

    # isolation: drop stars with a neighbour within isolation_arcsec
    if skycoord_col in catalog.colnames and isolation_arcsec and len(catalog) > 1:
        sky = SkyCoord(catalog[skycoord_col])
        idx, sep2d, _ = sky.match_to_catalog_sky(sky, nthneighbor=2)
        clean &= sep2d.arcsec > isolation_arcsec

    rows = []
    fc = fluxes[clean]
    for i, r in enumerate(radii_arcsec):
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = fc[:, i] / fc[:, -1]
        good = np.isfinite(ratio) & (ratio > 0)
        rr = ratio[good]
        if rr.size:
            med = np.median(rr)
            mad = np.median(np.abs(rr - med)) * 1.4826
            apcorr_mag = -2.5 * np.log10(med)
        else:
            med = mad = apcorr_mag = np.nan
        rows.append((r, med, apcorr_mag, mad, int(rr.size)))
    tbl = Table(rows=rows, names=('radius_arcsec', 'flux_ratio_to_total',
                                  'apcorr_mag', 'ratio_mad', 'n_stars'))
    tbl.meta['filter'] = filtername or catalog.meta.get('filter', '')
    tbl.meta['ref_radius_arcsec'] = float(radii_arcsec[-1])
    tbl.meta['min_coverage'] = float(min_coverage)
    tbl.meta['min_snr'] = float(min_snr)
    tbl.meta['isolation_arcsec'] = float(isolation_arcsec)
    tbl.meta['n_clean_stars'] = int(clean.sum())
    return tbl


def select_reference_stars(catalog, snr_min=50.0, isolation_arcsec=3.0,
                           max_qfit=None, max_group_size=1, min_nmatch=2,
                           skycoord_col='skycoord', flux_col='flux',
                           fluxerr_col='flux_err'):
    """Boolean mask of clean, isolated, UNSATURATED reference stars for a
    contamination-free curve of growth.

    Selects on: not saturated / not saturation-replaced; high SNR; good fit
    (``qfit`` below ``max_qfit`` if given, or the 60th percentile of the SNR-cut
    sample otherwise); un-blended (``group_size <= max_group_size``); detected in
    ``>= min_nmatch`` frames; and geometrically isolated (nearest neighbour in the
    FULL catalog farther than ``isolation_arcsec`` -- so the aperture AND its sky
    annulus are free of other sources).  Missing columns are simply not applied.
    """
    n = len(catalog)
    keep = np.ones(n, bool)
    for satcol in ('is_saturated', 'replaced_saturated'):
        if satcol in catalog.colnames:
            keep &= ~np.asarray(catalog[satcol], bool)
    if flux_col in catalog.colnames and fluxerr_col in catalog.colnames:
        f = np.asarray(catalog[flux_col], float)
        e = np.asarray(catalog[fluxerr_col], float)
        with np.errstate(invalid='ignore', divide='ignore'):
            snr = f / e
        keep &= np.isfinite(snr) & (snr >= snr_min) & (f > 0)
    if 'group_size' in catalog.colnames:
        keep &= np.asarray(catalog['group_size'], float) <= max_group_size
    if 'nmatch' in catalog.colnames:
        keep &= np.asarray(catalog['nmatch'], float) >= min_nmatch
    if 'qfit' in catalog.colnames:
        q = np.asarray(catalog['qfit'], float)
        if max_qfit is None:
            qthr = np.nanpercentile(q[keep], 60) if keep.any() else np.inf
        else:
            qthr = max_qfit
        keep &= np.isfinite(q) & (q <= qthr)
    if isolation_arcsec and skycoord_col in catalog.colnames and keep.any() \
            and len(catalog) > 1:
        sky = SkyCoord(catalog[skycoord_col])
        # nearest neighbour in the FULL catalog (not just the kept subset)
        idx, sep2d, _ = sky.match_to_catalog_sky(sky, nthneighbor=2)
        keep &= sep2d.arcsec > isolation_arcsec
    return keep


def build_reference_apcorr(ref_catalog, i2d_paths, filtername,
                           snr_min=50.0, isolation_ladder=(2.0, 1.0, 0.6, 0.4, 0.3),
                           min_ref_stars=200, ann_in_arcsec=None,
                           ann_out_arcsec=None, radii_arcsec=RADII_ARCSEC,
                           **sel_kw):
    """Contamination-free curve-of-growth apcorr from isolated unsaturated stars.

    ``ref_catalog`` is a photometry catalog (e.g. the merged daophot catalog)
    with ``skycoord`` + flux/quality columns.  In the very crowded GC/globular
    fields there are essentially no stars isolated beyond ~0.3-0.5", so the
    isolation radius is stepped DOWN through ``isolation_ladder`` until at least
    ``min_ref_stars`` clean unsaturated stars survive; the achieved isolation and
    the radius range it makes trustworthy (aperture + sky annulus inside the
    isolation radius) are recorded in the table meta.  The curve of growth is
    therefore reliable only out to ~ the achieved isolation radius -- beyond that
    (the "total") no isolated empirical measurement is possible in these fields
    and a theoretical PSF must be used.
    """
    scol = 'skycoord' if 'skycoord' in ref_catalog.colnames else 'skycoord_fit'
    achieved_iso, keep = None, None
    for iso in isolation_ladder:
        k = select_reference_stars(ref_catalog, snr_min=snr_min,
                                   isolation_arcsec=iso, skycoord_col=scol,
                                   **sel_kw)
        if int(k.sum()) >= min_ref_stars:
            achieved_iso, keep = iso, k
            break
    if keep is None:                       # even the loosest rung is too sparse
        achieved_iso = isolation_ladder[-1]
        keep = select_reference_stars(ref_catalog, snr_min=snr_min,
                                      isolation_arcsec=achieved_iso,
                                      skycoord_col=scol, **sel_kw)
    # keep the annulus INSIDE the isolation radius so neighbours never enter the
    # sky annulus; aperture stays well inside the annulus
    if ann_out_arcsec is None:
        ann_out_arcsec = max(0.20, achieved_iso - 0.02)
    if ann_in_arcsec is None:
        ann_in_arcsec = max(0.15, ann_out_arcsec - 0.10)
    # only trust radii inside the sky annulus (contamination-free); normalise the
    # curve of growth to the largest such radius, NOT the crowding-contaminated
    # outer radii
    radii_reliable = tuple(r for r in sorted(radii_arcsec) if r <= ann_in_arcsec)
    if len(radii_reliable) < 2:
        radii_reliable = tuple(sorted(radii_arcsec)[:2])
    ref = ref_catalog[keep].copy()
    ref.meta.setdefault('filter', filtername)
    ref = measure_aperture_photometry(ref, i2d_paths, filtername=filtername,
                                      radii_arcsec=radii_reliable,
                                      ann_in_arcsec=ann_in_arcsec,
                                      ann_out_arcsec=ann_out_arcsec,
                                      skycoord_col=scol)
    # already geometrically isolated + unsaturated -> don't re-cut on isolation
    tbl = build_aperture_correction_table(ref, filtername=filtername,
                                          radii_arcsec=radii_reliable,
                                          isolation_arcsec=0.0,
                                          skycoord_col=scol)
    tbl.meta['source'] = 'reference_unsaturated_isolated'
    tbl.meta['snr_min'] = float(snr_min)
    tbl.meta['isolation_arcsec'] = float(achieved_iso)
    tbl.meta['annulus_arcsec'] = f'{ann_in_arcsec:.3f},{ann_out_arcsec:.3f}'
    tbl.meta['reliable_max_radius_arcsec'] = float(ann_in_arcsec)
    tbl.meta['n_reference_stars'] = int(keep.sum())
    return tbl


def apcorr_table_path(filtername, target, basepath, module='', obs_token=''):
    """Path for the SEPARATE aperture-correction table (not the photometry table)."""
    mod = f'_{module}' if module else ''
    tok = obs_token or ''
    return (f'{basepath}/catalogs/'
            f'{filtername.lower()}{mod}{tok}_satstar_apcorr.ecsv')


def write_aperture_correction_table(tbl, filtername, target, basepath,
                                    module='', obs_token=''):
    path = apcorr_table_path(filtername, target, basepath, module, obs_token)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tbl.write(path, overwrite=True, format='ascii.ecsv')
    log.info(f"Wrote aperture-correction table {path}")
    return path


def main():
    import argparse
    ap = argparse.ArgumentParser(
        description='Measure i2d aperture photometry for a saturated-star '
                    'catalog and (optionally) write an aperture-correction table.')
    ap.add_argument('--target', required=True)
    ap.add_argument('--filter', dest='filtername', required=True)
    ap.add_argument('--basepath', default=None)
    ap.add_argument('--catalog', default=None,
                    help='satstar catalog (default: consolidated for target/filter)')
    ap.add_argument('--out', default=None, help='write augmented catalog here')
    ap.add_argument('--apcorr', action='store_true',
                    help='also build+write the aperture-correction table')
    args = ap.parse_args()

    basepath = args.basepath or f'/orange/adamginsburg/jwst/{args.target}/'
    if args.catalog:
        cat = Table.read(args.catalog)
    else:
        cat = Table.read(f'{basepath}/catalogs/'
                         f'{args.filtername.lower()}_consolidated_satstar_catalog.fits')
    cat = add_aperture_photometry(cat, args.filtername, args.target, basepath)
    if args.out:
        cat.write(args.out, overwrite=True)
        print(f'wrote {args.out} (n={len(cat)})')
    if args.apcorr and has_aperture_columns(cat):
        tbl = build_aperture_correction_table(cat, filtername=args.filtername)
        write_aperture_correction_table(tbl, args.filtername, args.target, basepath)
        tbl.pprint_all()


if __name__ == '__main__':
    main()
