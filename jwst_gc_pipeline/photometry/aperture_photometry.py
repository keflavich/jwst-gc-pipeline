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

from jwst_gc_pipeline.mast_names import jw_prefix

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


_FILTER_TOKEN = re.compile(r'^f\d{3,4}[a-z]?$')       # e.g. f200w, f405n, f2550w


def _is_science_mosaic(basename, filt_l, inst):
    """True iff ``basename`` is a science i2d mosaic for ``filt_l`` -- an
    anchored positive test (see find_i2d_mosaics).  Rejects every derived product
    (residual/model/mergedcat/reproject/resbgsub/_m<N>/downsel/...) by requiring
    the filter-descriptor segment to contain no non-filter tokens."""
    m = re.match(rf'^jw\d+-o\d+_t001_{inst}_(.+)_i2d\.fits$', basename)
    if not m:
        return False
    seg = m.group(1)
    if seg.startswith('clear-'):
        seg = seg[len('clear-'):]
    toks = seg.split('-')
    if filt_l not in toks:
        return False
    partner_filters = []
    for t in toks:
        if t == filt_l or t in ('merged', 'nrca', 'nrcb'):
            continue
        if _FILTER_TOKEN.match(t):        # LW dual-filter partner (e.g. f405n-f444w)
            partner_filters.append(t)
            continue
        return False                       # any other token -> derived product
    # A dual-filter mosaic (`<narrow>-<wide>`, e.g. f405n-f444w) carries the
    # NARROW/MEDIUM band's flux; it is the science mosaic for that band only.  The
    # WIDE partner has its own `clear-<wide>-merged`, so do NOT claim the dual
    # mosaic for a wide filter (that would apply the wide zeropoint to narrow-band
    # flux).  This does not rely on the per-filter directory scoping to be safe.
    if partner_filters and not (filt_l.endswith('n') or filt_l.endswith('m')):
        return False
    return True


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
    except (KeyError, AttributeError):
        # KeyError: filter not in the target's map; AttributeError: target not in
        # the registry at all (returns None).  Fall back to the generic globs.
        proj = None
    if inst == 'nircam' and proj is not None and target in project_obsnum \
            and proj in project_obsnum[target]:
        obs = project_obsnum[target][proj]
        pats.append(f'{pipe}/{jw_prefix(proj)}-o{obs}_t001_nircam_clear-{filt_l}-merged_i2d.fits')
    # generic fallbacks (covers MIRI, non-registry targets, plain naming)
    pats += [
        f'{pipe}/jw*-o*_t001_{inst}_clear-{filt_l}-merged_i2d.fits',
        f'{pipe}/jw*-o*_t001_{inst}_*{filt_l}*_i2d.fits',
    ]
    # POSITIVE match for the science mosaic name (anchored), rather than a
    # substring reject-list: the segment between `_t001_<inst>_` and `_i2d` must
    # be exactly a filter descriptor -- `[clear-]<filt>` optionally suffixed with
    # `-merged` / `-nrca` / `-nrcb`, or an LW dual-filter pair `<filt>-<other>` /
    # `<other>-<filt>` where <other> is itself a filter token.  Any extra token
    # (residual, model, mergedcat, reproject, resbgsub, medfilt, _m<N>, downsel,
    # blur, seed, ...) breaks the match and is rejected -- no substring can slip
    # through (the `merged-reproject` class that beat the old reject list is now
    # rejected because `reproject` is not a filter token).
    seen, out = set(), []
    for p in pats:
        for m in sorted(glob.glob(p)):
            if m not in seen and _is_science_mosaic(os.path.basename(m),
                                                    filt_l, inst):
                seen.add(m)
                out.append(m)
    # Prefer the module-merged mosaic over per-module (-nrca/-nrcb) mosaics of the
    # SAME observation: they are the same data, and the merged one avoids
    # measuring each star on two half-depth tiles.  Keep per-module mosaics only
    # for observations that have no merged product (e.g. gc2211 o050 nrcb-only).
    def _obs(b):
        mm = re.search(r'(jw\d+-o\d+)_t001', b)
        return mm.group(1) if mm else b
    merged_obs = {_obs(os.path.basename(m)) for m in out
                  if f'-{filt_l}-merged_i2d' in os.path.basename(m)}
    pruned = [m for m in out
              if not (re.search(rf'-{filt_l}-nrc[ab]_i2d', os.path.basename(m))
                      and _obs(os.path.basename(m)) in merged_obs)]
    return pruned


def _measure_one_mosaic(sky, i2d_path, radii_arcsec, primary_radius_arcsec,
                        ann_in_arcsec, ann_out_arcsec, recenter_box=0):
    """Measure aperture photometry for all positions on one mosaic.

    Returns a dict of per-star arrays (length == len(sky)); stars off this
    mosaic have NaN flux and coverage 0.  If ``recenter_box`` > 0, positions are
    re-centroided on the mosaic within that box (px) before aperture photometry
    -- essential for a curve of growth, since catalog positions come from the crf
    grid and are offset from the i2d grid by ~1 px, which bleeds small-aperture
    flux and makes the empirical PSF look spuriously broad.
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

    if recenter_box and recenter_box > 0:
        from photutils.centroids import centroid_sources, centroid_com
        # NaN-safe centroiding: zero the non-finite pixels for the fit
        sci_c = np.where(finite, sci, 0.0)
        xc, yc = centroid_sources(sci_c, pos[:, 0], pos[:, 1],
                                  box_size=int(recenter_box),
                                  centroid_func=centroid_com)
        ok = np.isfinite(xc) & np.isfinite(yc)
        pos[ok, 0] = xc[ok]
        pos[ok, 1] = yc[ok]

    # core-saturated: central pixel non-finite
    xi = np.clip(np.round(pos[:, 0]).astype(int), 0, nx - 1)
    yi = np.clip(np.round(pos[:, 1]).astype(int), 0, ny - 1)
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
                                skycoord_col='skycoord_fit', recenter_box=0):
    """Add i2d aperture-photometry columns to ``catalog`` (returns a copy).

    For each star the measurement from the mosaic with the highest primary-radius
    coverage is kept (handles multi-obs targets where mosaics overlap).
    ``recenter_box`` (px) > 0 re-centroids each source on the mosaic before
    measuring (needed for curve-of-growth accuracy; off by default so the shipped
    satstar photometry stays at the catalog position).
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
            ann_in_arcsec, ann_out_arcsec, recenter_box=recenter_box)
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

    # zeropoint / magnitudes.  A missing/unknown filter or an SVO outage must
    # only cost the Vega magnitude, not the whole measurement.
    zp = None
    if filtername is None:
        filtername = cat.meta.get('filter', None)
    if filtername is not None:
        from requests.exceptions import RequestException
        from astroquery.exceptions import InvalidQueryError, TimeoutError \
            as AstroqueryTimeoutError
        try:
            zp = _vega_zeropoint_jy(filtername)
        except (KeyError, IndexError, ValueError, ConnectionError,
                RequestException, InvalidQueryError, AstroqueryTimeoutError) as ex:
            # unknown filter (KeyError/IndexError), or SVO down/slow (builtin
            # ConnectionError / RequestException / astroquery InvalidQueryError /
            # TimeoutError): cost only the Vega magnitude, keep the aperture flux.
            # (ConnectionError is caught HERE, not by the outer guard, so an SVO
            # outage keeps the flux regardless of which error flavour it raises.)
            log.warning(f"aperture_photometry: Vega zeropoint lookup failed for "
                        f"{filtername} ({type(ex).__name__}); mag_vega will be NaN")
    flux = best['flux_jy']
    with np.errstate(invalid='ignore', divide='ignore'):
        mag_ab = -2.5 * np.log10(flux) + ABMAG_OFFSET
        mag_vega = (-2.5 * np.log10(flux / zp.to(u.Jy).value)
                    if zp is not None else np.full(n, np.nan))

    # A saturated/masked core, or an aperture that clipped a mosaic edge/NaN, makes
    # the aperture flux a LOWER LIMIT (the PSF fit's flux_fit is the trustworthy
    # value for those).  Require ~COMPLETE coverage: even a handful of masked
    # pixels in the aperture bias the flux low (measured: cov in [0.98,0.998]
    # carries a ~1 mag deficit), so the validity threshold is 0.999, i.e. "no
    # masked pixels", not merely "mostly covered".
    _COV_VALID = 0.999
    valid = ((~best['core_sat']) & (best['cov'] >= _COV_VALID)
             & np.isfinite(flux) & (flux > 0))
    _vnote = (f'aperture flux is a trustworthy measurement (not core-saturated, '
              f'coverage>={_COV_VALID} i.e. no masked pixels, finite positive); '
              f'False = LOWER LIMIT (use PSF flux_fit instead)')

    cat['aper_flux_jy'] = Column(
        flux, unit=u.Jy,
        description='i2d circular-aperture flux (bkg-sub, primary radius); RAW, '
                    'NOT aperture-corrected; a LOWER LIMIT where aper_flux_valid '
                    'is False (masked/saturated core or incomplete coverage)')
    cat['aper_flux_valid'] = Column(valid, description=_vnote)
    cat['aper_flux_err_jy'] = Column(
        best['flux_err_jy'], unit=u.Jy,
        description='1-sigma error on aper_flux_jy from the mosaic ERR extension')
    cat['aper_mag_vega'] = Column(
        mag_vega, unit=u.mag,
        description='Vega mag from aper_flux_jy (SVO ZeroPoint); RAW, not '
                    'aperture-corrected; LOWER LIMIT / brighter-biased where '
                    'aper_flux_valid is False')
    cat['aper_mag_ab'] = Column(
        mag_ab, unit=u.mag,
        description='AB mag from aper_flux_jy (ABMAG_OFFSET=8.90); RAW, not '
                    'aperture-corrected; follows aper_flux_valid (see aper_flux_jy)')
    cat['aper_bkg'] = Column(best['bkg'], unit=u.MJy / u.sr,
                             description='annulus local sky (per pixel)')
    cat['aper_area_frac'] = Column(
        best['cov'],
        description='finite-pixel fraction in the primary aperture (1.0 = no '
                    'masked pixels); drives aper_flux_valid')
    cat['aper_core_saturated'] = Column(best['core_sat'],
                                        description='central pixel NaN/masked in mosaic')
    cat['aper_i2d'] = Column(np.array([str(s) for s in best_i2d]),
                             description='mosaic the measurement was taken from')
    for r in radii_arcsec:
        tag = _rtag(r)
        cat[f'aper_flux_jy_r{tag}'] = Column(
            best[f'cog_{tag}'], unit=u.Jy,
            description=f'curve-of-growth aperture flux at r={r}" (bkg-sub, RAW); '
                        f'trustworthy only where aper_area_frac_r{tag}>=0.999 and '
                        f'not aper_core_saturated (no per-radius validity flag)')
        cat[f'aper_area_frac_r{tag}'] = Column(
            best[f'cov_{tag}'],
            description=f'finite-pixel fraction in the r={r}" aperture')

    cat.meta['APER_RAD'] = str(list(radii_arcsec))
    cat.meta['APER_PRIM'] = float(primary_radius_arcsec)
    cat.meta['APER_ANN'] = f'{ann_in_arcsec},{ann_out_arcsec}'
    cat.meta['APER_UNIT'] = 'arcsec'
    if zp is not None:
        cat.meta['APER_ZP'] = float(zp.to(u.Jy).value)
    return cat


def aperture_photometry_enabled():
    """Whether aperture photometry is on (default yes).  Any of 0/false/no/off
    (case-insensitive) via ``SATSTAR_APERTURE_PHOT`` disables it."""
    return os.environ.get('SATSTAR_APERTURE_PHOT', '1').strip().lower() \
        not in ('0', 'false', 'no', 'off')


def add_aperture_photometry(catalog, filtername, target, basepath, **kwargs):
    """Convenience: locate the i2d mosaic(s) and add aperture columns.

    Returns the catalog unchanged (with a log message) if disabled by
    ``SATSTAR_APERTURE_PHOT`` or if no mosaic is found.  A missing/corrupt mosaic
    or an SVO outage is caught and logged (aperture photometry is a diagnostic
    add-on and must never abort the merge).
    """
    if not aperture_photometry_enabled():
        return catalog
    i2ds = find_i2d_mosaics(filtername, target, basepath)
    if not i2ds:
        log.warning(f"aperture_photometry: no i2d mosaic for {target}/{filtername} "
                    f"under {basepath}; leaving catalog without aperture columns")
        return catalog
    log.info(f"aperture_photometry: measuring {filtername} from "
             f"{len(i2ds)} mosaic(s)")
    from requests.exceptions import RequestException
    try:
        return measure_aperture_photometry(catalog, i2ds, filtername=filtername,
                                           **kwargs)
    except (OSError, KeyError, ValueError, RequestException) as ex:
        # OSError: truncated/unreadable i2d; KeyError: missing SCI extension or
        # unknown SVO filter; ValueError: bad WCS/shape; RequestException: SVO
        # zeropoint service outage.  Aperture photometry is a diagnostic add-on
        # and must never abort the merge.
        log.warning(f"aperture_photometry: measurement failed for "
                    f"{target}/{filtername} ({type(ex).__name__}: {ex}); "
                    f"leaving catalog without aperture columns")
        return catalog


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
    tbl.meta['provenance'] = 'satstar_catalog_curve_of_growth'
    # The "clean" stars here are the least-saturated members of the SATURATED-STAR
    # catalog, and in crowded fields the largest-radius reference is neighbour-
    # contaminated -> this table is a DIAGNOSTIC, not the aperture correction of
    # record.  Use build_reference_apcorr (isolated unsaturated main-catalog stars)
    # for a usable apcorr.  ratio_mad does NOT rank reliability across bands here.
    tbl.meta['warning'] = ('DIAGNOSTIC ONLY: crowding-contaminated at large '
                           'radius; use *_apcorr_refstars for the aperture '
                           'correction of record')
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
                           snr_min=150.0, isolation_ladder=(2.0, 1.0, 0.6, 0.4, 0.3),
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
    # isolation is controlled by isolation_ladder here; forbid the collision with
    # select_reference_stars' own isolation_arcsec (older calling convention).
    sel_kw.pop('isolation_arcsec', None)
    scol = 'skycoord' if 'skycoord' in ref_catalog.colnames else 'skycoord_fit'
    achieved_iso, keep = None, None
    last_iso, last_k = None, None
    for iso in isolation_ladder:
        k = select_reference_stars(ref_catalog, snr_min=snr_min,
                                   isolation_arcsec=iso, skycoord_col=scol,
                                   **sel_kw)
        last_iso, last_k = iso, k
        if int(k.sum()) >= min_ref_stars:
            achieved_iso, keep = iso, k
            break
    if keep is None:                       # even the loosest rung is too sparse
        achieved_iso, keep = last_iso, last_k   # reuse last computed (no recompute)
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
    # pin the primary radius to the largest reliable radius so
    # measure_aperture_photometry does not inject its default 0.30" aperture
    # (which would fall inside/beyond the sky annulus and skew normalization)
    ref = measure_aperture_photometry(ref, i2d_paths, filtername=filtername,
                                      radii_arcsec=radii_reliable,
                                      primary_radius_arcsec=max(radii_reliable),
                                      ann_in_arcsec=ann_in_arcsec,
                                      ann_out_arcsec=ann_out_arcsec,
                                      skycoord_col=scol, recenter_box=5)
    # already geometrically isolated + unsaturated -> don't re-cut on isolation
    tbl = build_aperture_correction_table(ref, filtername=filtername,
                                          radii_arcsec=radii_reliable,
                                          isolation_arcsec=0.0,
                                          skycoord_col=scol)
    # this IS the apcorr of record (isolated unsaturated stars) -- override the
    # diagnostic provenance/warning that build_aperture_correction_table stamps
    tbl.meta['source'] = 'reference_unsaturated_isolated'
    tbl.meta['provenance'] = 'reference_unsaturated_isolated'
    tbl.meta.pop('warning', None)
    tbl.meta['snr_min'] = float(snr_min)
    tbl.meta['isolation_arcsec'] = float(achieved_iso)
    tbl.meta['annulus_arcsec'] = f'{ann_in_arcsec:.3f},{ann_out_arcsec:.3f}'
    # the curve of growth is normalised to (and reliable out to) the largest
    # MEASURED radius inside the annulus -- NOT ann_in itself, which is not one of
    # the measured radii
    tbl.meta['reliable_max_radius_arcsec'] = float(max(radii_reliable))
    tbl.meta['n_reference_stars'] = int(keep.sum())
    # provenance: which mosaic(s) the curve of growth was measured on (the source
    # catalog is recorded by the caller, which knows its path)
    tbl.meta['i2d_mosaics'] = ','.join(os.path.basename(p) for p in i2d_paths)
    tbl.meta['recentered'] = True
    tbl.meta.pop('min_snr', None)     # inherited-but-unused knob; snr_min is the live one
    return tbl


def apcorr_table_path(filtername, target, basepath, module='', obs_token='',
                      kind='refstars'):
    """Path for the SEPARATE aperture-correction table (not the photometry table).

    ``kind='refstars'`` (default) is the aperture correction of record, derived
    from isolated unsaturated reference stars (``build_reference_apcorr``).
    ``kind='diagnostic'`` is the satstar-catalog curve of growth, which is
    crowding-contaminated at large radius and is NOT a usable correction.  The
    bare ``*_satstar_apcorr.ecsv`` name is retired so nothing reads the
    contaminated table as if it were the correction.
    """
    mod = f'_{module}' if module else ''
    tok = obs_token or ''
    suffix = {'refstars': 'apcorr_refstars',
              'diagnostic': 'apcorr_diagnostic'}[kind]
    return (f'{basepath}/catalogs/'
            f'{filtername.lower()}{mod}{tok}_satstar_{suffix}.ecsv')


def write_aperture_correction_table(tbl, filtername, target, basepath,
                                    module='', obs_token='', kind=None):
    if kind is None:
        kind = ('refstars'
                if tbl.meta.get('provenance') == 'reference_unsaturated_isolated'
                else 'diagnostic')
    path = apcorr_table_path(filtername, target, basepath, module, obs_token, kind)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tbl.write(path, overwrite=True, format='ascii.ecsv')
    log.info(f"Wrote {kind} aperture-correction table {path}")
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
