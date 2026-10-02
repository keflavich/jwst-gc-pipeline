"""Score reference-field runs against their fixed thresholds.

::

    python -m jwst_gc_pipeline.photometry.reference_fields.evaluate --variant main
    python -m jwst_gc_pipeline.photometry.reference_fields.evaluate --variant main --json out.json
    python -m jwst_gc_pipeline.photometry.reference_fields.evaluate --variant new --baseline out.json
    python -m jwst_gc_pipeline.photometry.reference_fields.evaluate --calibrate out.json --commit <sha>

Metrics (all computed inside the field's inner box, the cutout minus
``inner_margin_arcsec`` per side, where the cutout's own edge does not bias
detection or the residual):

``completeness``  (injection runs)
    Fraction of injected primary-band stars recovered, per ``snr_bins`` bin of
    the injected ERR-based S/N.  A recovery is a ONE-TO-ONE match within
    ``match_radius_pix`` with ``|dmag| < match_dmag``.

``flux_bias_mag``  (injection runs)
    Median ``-2.5 log10(F_out / F_in)`` of recovered stars with
    ``snr_true >= bias_min_snr``.

``residual_excess``  (clean run)
    Unsubtracted point sources left in the final residual mosaic, per arcsec^2:
    positive minus negative local extrema of the PSF-matched-filter S/N map
    (``|S/N| > excess_snr``, renormalised by its own robust scatter) farther than
    ``excess_excl_pix`` from every catalog source.  Noise and symmetric
    structure cancel between the signs; faint stars left in the residual do not.

``oversubtracted``  (clean run)
    Catalog sources whose residual core is over-subtracted, per arcsec^2: the
    minimum of the matched-filter S/N map in the 3x3 px box at the source is
    below ``-oversub_snr``.  ``residual_excess`` excludes the region around
    every catalog source, so a fit that takes flux from a bright neighbour's
    PSF wing (or from extended emission) lowers the positive count and leaves
    its negative core unseen; this metric counts those fits.  Blended real
    stars and PSF-model mismatch at bright stars also contribute, so the
    threshold is set per field from a baseline.

``seed_scalars``  (injection runs)
    ``residual_excess`` and ``oversubtracted`` of each injection run, scored on
    the field's own sources (injected positions excluded from the excess,
    catalog sources on an injected star left out of the over-subtracted
    count): median, standard deviation and number of seeds.  The injected
    stars perturb the fits of their neighbours, so the spread over seeds
    measures how much a neutral change moves the clean-run value.

``ring_ratio``  (clean run)
    Faint sources found ``ring_pix`` (1.5-4.5 px) from a bright star, over the
    number expected there from the faint sources' mean density.  Real faint
    stars there are HARDER to find than elsewhere (ratio <= 1); a ratio well
    above 1 means the catalog fits PSF-model mismatch as companions.

``emission_purity``  (clean + injection runs)
    Chance-corrected cross-band confirmation of the faint accepted primary
    sources (catalog S/N in ``faint_snr``).  Each source's companion-band
    forced matched-filter S/N is measured in the run's companion data mosaic;
    ``r_on`` is the fraction above ``confirm_snr``, ``r_off`` the same at
    positions offset ``offset_pix`` in random directions, and ``r_real`` the
    rate for injected stars of the same primary S/N (which carry the field's
    median colour).  ``purity = (r_on - r_off) / (r_real - r_off)``.  A source
    built from emission structure of the primary band has no companion-band
    counterpart at the star's colour, so fakes pull the purity down.  It is
    reported as NaN when ``r_real - r_off < 0.2`` (the companion band cannot
    tell stars from chance there).

``labels_recovered``  (clean run, fields with a ``labels`` region file)
    Fraction of hand-marked stars matched within ``label_radius_arcsec``.

``emission_labels_cataloged``  (clean run, fields with ``emission_labels``)
    Number of hand-marked nebular structures (knots, filament ridges) inside
    the inner box with a catalog source within ``emission_label_radius_arcsec``:
    sources fit to extended emission.  The labels are positions (RA, Dec) that
    show the same extended shape in the field's continuum bands and no point
    source in any band.

Threshold calibration: ``--calibrate`` prints, per field, the ``thresholds:``
and ``calibration:`` blocks for ``fields.yaml`` (with ``calibrate_main``) from
the ``--json`` output of a calibration variant.  Each threshold sits ``nsigma`` (default 2) standard errors on
the passing side of the calibration value, so a code change that leaves the
true value where the calibration branch has it passes with probability
~0.98 per threshold:

* completeness floor = p - 2 sigma, ``p = k/n`` pooled over the seeds and
  ``sigma = sqrt(p~(1 - p~)/n)`` with ``p~ = (k + 1)/(n + 2)`` (so a bin at 0/n
  or n/n still has a margin), rounded down to 0.01.  A bin whose floor would be
  below ``min_floor`` (0.05) carries none: a floor there cannot fail.
* ``residual_excess_max`` / ``oversubtracted_max`` = clean-run value + 2 sigma,
  sigma the standard deviation of the same metric over the injection seeds
  (``seed_scalars``), rounded up to 0.05.
* ``flux_bias_max_mag`` = |median dmag| + 2 sigma (its standard error),
  rounded up to 0.01, and at least ``bias_floor`` (0.1 mag).
* ``labels_recovered_min`` = clean-run fraction - 2 sigma (binomial over the
  labelled stars), rounded down to 0.01.
* ``emission_labels_cataloged_max`` = calibration count (a count of four fixed
  structures has no seed replication; any further knot fails).
"""
import argparse
import glob
import json
import math
import os

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import mad_std
from astropy.table import Table
from astropy.wcs import WCS
from scipy import ndimage
from scipy.spatial import cKDTree

from jwst_gc_pipeline.photometry import reference_fields as RF
from jwst_gc_pipeline.photometry.injection import flux_column

FWHM_TABLE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), 'reduction', 'fwhm_table.ecsv')


def fwhm_pix(filt):
    """PSF FWHM (detector pixels) from the pipeline's FWHM table."""
    tbl = Table.read(FWHM_TABLE)
    return float(tbl[tbl['Filter'] == filt.upper()]['PSF FWHM (pixel)'][0])


# ---------------------------------------------------------------------------
# products of one run
# ---------------------------------------------------------------------------

def find_products(rdir, filt, phase=None):
    """Final catalog and mosaics of one filter of one run.

    ``phase`` defaults to the last phase present (m7, else m6).  Returns a
    dict with ``catalog``, ``residual``, ``smoothed_bg``, ``data`` paths and
    ``phase``; a missing product raises ``FileNotFoundError``.
    """
    fl = filt.lower()
    phases = [phase] if phase else ['m7', 'm6']
    for ph in phases:
        cats = sorted(glob.glob(os.path.join(
            rdir, 'catalogs', f'{fl}_*_resbgsub_{ph}_dao_basic_vetted.fits')))
        if cats:
            break
    else:
        raise FileNotFoundError(f'{rdir}: no {fl} {"/".join(phases)} vetted catalog')
    pipe = os.path.join(rdir, filt.upper(), 'pipeline')
    res = sorted(glob.glob(os.path.join(
        pipe, f'*{fl}-*_{ph}_*mergedcat_residual_i2d.fits')))
    data = sorted(glob.glob(os.path.join(pipe, f'*{fl}-*_data_i2d.fits')))
    if not res or not data:
        raise FileNotFoundError(f'{rdir}: {fl} {ph} residual ({len(res)}) '
                                f'or data ({len(data)}) mosaic missing')
    bg = res[0].replace('_residual_i2d.fits', '_residual_smoothed_bg_i2d.fits')
    return dict(catalog=cats[0], residual=res[0], data=data[0], phase=ph,
                smoothed_bg=bg if os.path.exists(bg) else None)


def _image(path):
    with fits.open(path) as h:
        ext = 'SCI' if 'SCI' in h else 0
        img = np.asarray(h[ext].data, float)
        hdr = h[ext].header
        err = np.asarray(h['ERR'].data, float) if 'ERR' in h else None
    return img, err, WCS(hdr), hdr


def load_catalog(path):
    """Catalog as ``(SkyCoord, flux, flux_err)`` (image units)."""
    t = Table.read(path)
    sc = SkyCoord(t['skycoord'])
    return sc, np.asarray(t['flux'], float), np.asarray(t['flux_err'], float)


class Frame:
    """A mosaic's pixel grid and the field's inner box on it."""

    def __init__(self, spec, wcs, shape):
        self.wcs = wcs
        self.shape = shape
        self.pixas = float(np.sqrt(abs(np.linalg.det(wcs.pixel_scale_matrix))) * 3600)
        cx, cy = wcs.world_to_pixel(SkyCoord(spec['ra'] * u.deg, spec['dec'] * u.deg))
        self.cx, self.cy = float(cx), float(cy)
        self.half = (spec['size_arcsec'] / 2 - spec['inner_margin_arcsec']) / self.pixas
        self.area_as2 = (2 * self.half * self.pixas) ** 2

    def xy(self, sc):
        x, y = self.wcs.world_to_pixel(sc)
        return np.atleast_1d(np.asarray(x, float)), np.atleast_1d(np.asarray(y, float))

    def inside(self, x, y):
        return (np.abs(np.asarray(x) - self.cx) <= self.half) & (np.abs(np.asarray(y) - self.cy) <= self.half)

    def mask(self):
        yy, xx = np.mgrid[0:self.shape[0], 0:self.shape[1]]
        return self.inside(xx, yy)


# ---------------------------------------------------------------------------
# metrics (pure functions of arrays; unit-tested in test_reference_fields.py)
# ---------------------------------------------------------------------------

def match_one_to_one(x_true, y_true, f_true, x_cat, y_cat, f_cat, radius, max_dmag):
    """Greedy one-to-one match by separation.  Returns the catalog index per
    truth row (-1 = unmatched)."""
    out = -np.ones(len(x_true), int)
    if len(x_cat) == 0 or len(x_true) == 0:
        return out
    tree = cKDTree(np.c_[x_cat, y_cat])
    pairs = []
    for i, nbrs in enumerate(tree.query_ball_point(np.c_[x_true, y_true], radius)):
        for j in nbrs:
            if not (f_cat[j] > 0 and f_true[i] > 0):
                continue
            dmag = -2.5 * np.log10(f_cat[j] / f_true[i])
            if abs(dmag) < max_dmag:
                pairs.append((np.hypot(x_cat[j] - x_true[i], y_cat[j] - y_true[i]), i, j))
    used = set()
    for _, i, j in sorted(pairs):
        if out[i] < 0 and j not in used:
            out[i] = j
            used.add(j)
    return out


def completeness_by_bin(snr_true, recovered, bins):
    """``{'lo-hi': (n_injected, n_recovered, fraction)}`` per S/N bin."""
    res = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (snr_true >= lo) & (snr_true < hi)
        n = int(sel.sum())
        k = int(recovered[sel].sum())
        res[f'{lo:g}-{hi:g}'] = (n, k, k / n if n else float('nan'))
    return res


def matched_filter_snr(image, err, fwhm, renormalise_mask=None):
    """PSF(Gaussian)-matched-filter S/N map of ``image`` with noise ``err``,
    renormalised to unit robust scatter over ``renormalise_mask``.  Returns
    ``(snr_map, empirical_scale)``."""
    sig = fwhm / 2.3548
    r = int(np.ceil(3 * sig))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    k = np.exp(-(xx ** 2 + yy ** 2) / (2 * sig ** 2))
    img = np.nan_to_num(image)
    e2 = np.where(np.isfinite(err) & (err > 0), err ** 2, np.inf)
    num = ndimage.convolve(img, k, mode='nearest')
    den = np.sqrt(ndimage.convolve(np.where(np.isfinite(e2), e2, 1e30), k ** 2, mode='nearest'))
    smf = num / den
    sel = np.isfinite(smf) & (np.abs(smf) < 20)
    if renormalise_mask is not None:
        sel &= renormalise_mask
    scale = float(mad_std(smf[sel])) if sel.sum() > 50 else 1.0
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return smf / scale, scale


def residual_excess(snr_map, inner_mask, src_x, src_y, area_as2, *, thresh, excl_pix):
    """Positive-minus-negative matched-filter extrema beyond ``excl_pix`` of
    every source, per arcsec^2.  Returns ``(excess_per_as2, n_pos, n_neg)``."""
    mx = ndimage.maximum_filter(snr_map, 5)
    mn = ndimage.minimum_filter(snr_map, 5)
    pos = (snr_map == mx) & (snr_map > thresh) & inner_mask
    neg = (snr_map == mn) & (snr_map < -thresh) & inner_mask
    tree = cKDTree(np.c_[src_x, src_y]) if len(src_x) else None

    def far(mask):
        yy, xx = np.nonzero(mask)
        if tree is None or not len(xx):
            return len(xx)
        return int((tree.query(np.c_[xx, yy])[0] > excl_pix).sum())

    npos, nneg = far(pos), far(neg)
    return (npos - nneg) / area_as2, npos, nneg


def oversubtracted_mask(snr_map, src_x, src_y, *, thresh):
    """Per source: is the 3x3 px residual matched-filter S/N minimum at its
    position below ``-thresh``?  False off the map."""
    mn = ndimage.minimum_filter(snr_map, 3)
    x, y = np.asarray(src_x, float), np.asarray(src_y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    ix = np.where(ok, np.rint(np.where(ok, x, 0)), -1).astype(int)
    iy = np.where(ok, np.rint(np.where(ok, y, 0)), -1).astype(int)
    ok &= (ix >= 0) & (iy >= 0) & (ix < snr_map.shape[1]) & (iy < snr_map.shape[0])
    out = np.zeros(len(x), bool)
    out[ok] = mn[iy[ok], ix[ok]] < -thresh
    return out


def oversubtracted(snr_map, src_x, src_y, inside, area_as2, *, thresh):
    """Sources inside the inner box whose 3x3 px residual matched-filter S/N
    minimum is below ``-thresh``, per arcsec^2.  Returns ``(per_as2, n)``."""
    n = int(np.sum(oversubtracted_mask(snr_map, src_x, src_y, thresh=thresh)
                   & np.asarray(inside, bool)))
    return n / area_as2, n


def injected_run_scalars(snr_map, inner_mask, area_as2, src_x, src_y, inside,
                         inj_x, inj_y, *, excess_snr, excl_pix, oversub_snr,
                         match_pix):
    """``residual_excess`` and ``oversubtracted`` of an injection run's residual,
    scored on the field's own sources: injected positions join the excess
    exclusion list (an unrecovered injected star is not a missed field star),
    and catalog sources within ``match_pix`` of an injected star leave the
    over-subtracted count.  Returns ``(excess, npos, nneg, osub, n_osub)``."""
    src_x, src_y = np.asarray(src_x, float), np.asarray(src_y, float)
    inj_x, inj_y = np.asarray(inj_x, float), np.asarray(inj_y, float)
    exc, npos, nneg = residual_excess(snr_map, inner_mask, np.r_[src_x, inj_x],
                                      np.r_[src_y, inj_y], area_as2,
                                      thresh=excess_snr, excl_pix=excl_pix)
    field = np.ones(len(src_x), bool)
    if len(inj_x) and len(src_x):
        field = cKDTree(np.c_[inj_x, inj_y]).query(np.c_[src_x, src_y])[0] > match_pix
    osub, n_osub = oversubtracted(snr_map, src_x[field], src_y[field],
                                  np.asarray(inside, bool)[field], area_as2,
                                  thresh=oversub_snr)
    return exc, npos, nneg, osub, n_osub


def ring_ratio(x, y, flux, snr, inside, *, bright_snr, faint_ratio, ring, area_pix):
    """Faint-near-bright pair count over its uniform-density expectation.

    Bright: catalog S/N >= ``bright_snr``.  Faint: flux < ``faint_ratio`` x the
    faintest bright source's flux.  Returns ``(ratio, n_obs, n_exp)``.
    """
    bright = inside & (snr >= bright_snr)
    if not bright.any():
        return float('nan'), 0, 0.0
    faint = inside & (flux < faint_ratio * np.min(flux[bright]))
    nf = int(faint.sum())
    if nf == 0:
        return float('nan'), 0, 0.0
    n_obs = 0
    for bx, by in zip(x[bright], y[bright]):
        d = np.hypot(x[faint] - bx, y[faint] - by)
        n_obs += int(((d >= ring[0]) & (d <= ring[1])).sum())
    n_exp = nf / area_pix * np.pi * (ring[1] ** 2 - ring[0] ** 2) * int(bright.sum())
    return (n_obs / n_exp if n_exp > 0 else float('nan')), n_obs, float(n_exp)


def n_within(x_lab, y_lab, x_cat, y_cat, radius):
    """Number of label positions with a catalog source within ``radius``."""
    if len(x_lab) == 0:
        return 0
    if len(x_cat) == 0:
        return 0
    d = cKDTree(np.c_[x_cat, y_cat]).query(np.c_[x_lab, y_lab])[0]
    return int(np.sum(d < radius))


def forced_snr_at(snr_map, x, y):
    """Matched-filter S/N at the nearest pixel (NaN off-map)."""
    ix = np.rint(x).astype(int)
    iy = np.rint(y).astype(int)
    ok = (ix >= 0) & (iy >= 0) & (iy < snr_map.shape[0]) & (ix < snr_map.shape[1])
    out = np.full(len(x), np.nan)
    out[ok] = snr_map[iy[ok], ix[ok]]
    return out


def purity_estimate(r_on, r_off, r_real, min_contrast=0.2):
    """``(r_on - r_off) / (r_real - r_off)``; NaN when the companion band
    cannot separate stars from chance (contrast < ``min_contrast``)."""
    if not np.isfinite(r_real) or (r_real - r_off) < min_contrast:
        return float('nan')
    return float((r_on - r_off) / (r_real - r_off))


# ---------------------------------------------------------------------------
# evaluation of one (field, variant)
# ---------------------------------------------------------------------------

def _companion_snr_map(rdir, filt, spec, phase=None):
    prod = find_products(rdir, filt, phase)
    img, err, wcs, _ = _image(prod['data'])
    if prod['smoothed_bg'] is not None:
        bg = _image(prod['smoothed_bg'])[0]
        img = img - bg
    else:
        img = img - ndimage.median_filter(np.nan_to_num(img), int(6 * fwhm_pix(filt)) | 1)
    fr = Frame(spec, wcs, img.shape)
    snr, _ = matched_filter_snr(img, err, fwhm_pix(filt), fr.mask())
    return snr, fr


def _residual_snr(spec, prod):
    """Matched-filter S/N map of a run's primary-band residual minus its
    smoothed background, on the residual mosaic's grid.  Returns
    ``(frame, inner_mask, snr_map, scale)``."""
    res, err, wcs, _ = _image(prod['residual'])
    if prod['smoothed_bg'] is not None:
        res = res - _image(prod['smoothed_bg'])[0]
    fr = Frame(spec, wcs, res.shape)
    inner = fr.mask()
    snr_map, scale = matched_filter_snr(res, err, fwhm_pix(spec['filters'][0]), inner)
    return fr, inner, snr_map, scale


def evaluate_clean(spec, variant, *, rng=None, phase=None):
    """Residual excess, ring ratio, faint-source confirmation rates and label
    recovery of the clean run of ``spec`` (primary band) at ``phase`` (default:
    the last phase present)."""
    rng = rng or np.random.default_rng(0)
    rdir = RF.run_dir(spec, variant, 0)
    prim = spec['filters'][0]
    prod = find_products(rdir, prim, phase)
    fr, inner, snr_map, scale = _residual_snr(spec, prod)
    sc, flux, ferr = load_catalog(prod['catalog'])
    x, y = fr.xy(sc)
    inside = fr.inside(x, y)
    snr_cat = flux / ferr
    exc, npos, nneg = residual_excess(snr_map, inner, x, y, fr.area_as2,
                                      thresh=spec['excess_snr'], excl_pix=spec['excess_excl_pix'])
    osub, n_osub = oversubtracted(snr_map, x, y, inside, fr.area_as2,
                                  thresh=spec['oversub_snr'])
    rr, n_obs, n_exp = ring_ratio(x, y, flux, snr_cat, inside,
                                  bright_snr=spec['ring_bright_snr'],
                                  faint_ratio=spec['ring_faint_ratio'],
                                  ring=spec['ring_pix'], area_pix=float(inner.sum()))
    out = dict(phase=prod['phase'], n_sources=int(inside.sum()),
               density_per_as2=float(inside.sum() / fr.area_as2),
               residual_excess=float(exc), residual_npos=npos, residual_nneg=nneg,
               residual_mf_scale=scale, oversubtracted=float(osub), n_oversubtracted=n_osub,
               ring_ratio=rr, ring_n_obs=n_obs, ring_n_exp=n_exp)

    if len(spec['filters']) > 1:
        comp = spec['filters'][1]
        cmap, cfr = _companion_snr_map(rdir, comp, spec, phase)
        lo, hi = spec['faint_snr']
        faint = inside & (snr_cat >= lo) & (snr_cat < hi)
        cx, cy = cfr.xy(sc[faint])
        on = forced_snr_at(cmap, cx, cy)
        ang = rng.uniform(0, 2 * np.pi, (len(cx), 8))
        rad = rng.uniform(*spec['offset_pix'], (len(cx), 8))
        off = forced_snr_at(cmap, (cx[:, None] + rad * np.cos(ang)).ravel(),
                            (cy[:, None] + rad * np.sin(ang)).ravel())
        thr = spec['confirm_snr']
        out.update(n_faint=int(faint.sum()),
                   r_on=float(np.nanmean(on > thr)) if len(on) else float('nan'),
                   r_off=float(np.nanmean(off > thr)) if len(off) else float('nan'))

    if spec.get('labels'):
        from jwst_gc_pipeline.photometry.evaluate_region_recovery import load_region_targets
        lab, kinds = load_region_targets(os.path.join(RF.basepath(spec), spec['labels']),
                                         include_boxes=False)
        lx, ly = fr.xy(lab)
        lin = fr.inside(lx, ly)
        tree = cKDTree(np.c_[x, y])
        d = tree.query(np.c_[lx[lin], ly[lin]])[0] * fr.pixas
        out.update(n_labels=int(lin.sum()),
                   labels_recovered=float(np.mean(d < spec['label_radius_arcsec'])) if lin.any() else float('nan'))
    if spec.get('emission_labels'):
        ek = np.asarray(spec['emission_labels'], float)
        ex, ey = fr.xy(SkyCoord(ek[:, 0] * u.deg, ek[:, 1] * u.deg))
        ein = fr.inside(ex, ey)
        out.update(n_emission_labels=int(ein.sum()),
                   emission_labels_cataloged=n_within(
                       ex[ein], ey[ein], x, y, spec['emission_label_radius_arcsec'] / fr.pixas))
    return out


def evaluate_injected(spec, variant, seed, *, phase=None):
    """Completeness, flux bias and the real-star companion confirmation rate
    of injection run ``seed``."""
    rdir = RF.run_dir(spec, variant, seed)
    prim = spec['filters'][0]
    inj = Table.read(RF.injection_table_path(spec['name'], seed))
    prod = find_products(rdir, prim, phase)
    img, _, wcs, hdr = _image(prod['data'])
    pixar = float(hdr['PIXAR_SR'])
    fr = Frame(spec, wcs, img.shape)
    sc, flux, _ = load_catalog(prod['catalog'])
    x, y = fr.xy(sc)
    isc = SkyCoord(np.asarray(inj['ra']) * u.deg, np.asarray(inj['dec']) * u.deg)
    ix, iy = fr.xy(isc)
    keep = fr.inside(ix, iy)
    f_true = np.asarray(inj[flux_column(prim)], float) / (1e6 * pixar)
    snr_true = np.asarray(inj[f'snr_true_{prim}'], float)
    idx = match_one_to_one(ix[keep], iy[keep], f_true[keep], x, y, flux,
                           spec['match_radius_pix'], spec['match_dmag'])
    rec = idx >= 0
    out = dict(seed=int(seed), phase=prod['phase'], n_injected=int(keep.sum()),
               snr_true=snr_true[keep].tolist(), recovered=rec.tolist())
    dm = np.full(keep.sum(), np.nan)
    dm[rec] = -2.5 * np.log10(flux[idx[rec]] / f_true[keep][rec])
    out['dmag'] = dm.tolist()
    # the clean run's residual scalars on this run, scored on the field's own
    # sources: their spread over seeds is the run-to-run noise of the
    # clean-run value the thresholds check
    rfr, inner, snr_map, _ = _residual_snr(spec, prod)
    rx, ry = rfr.xy(sc)
    jx, jy = rfr.xy(isc[keep])
    exc, npos, nneg, osub, n_osub = injected_run_scalars(
        snr_map, inner, rfr.area_as2, rx, ry, rfr.inside(rx, ry), jx, jy,
        excess_snr=spec['excess_snr'], excl_pix=spec['excess_excl_pix'],
        oversub_snr=spec['oversub_snr'], match_pix=spec['match_radius_pix'])
    out.update(residual_excess=float(exc), residual_npos=npos, residual_nneg=nneg,
               oversubtracted=float(osub), n_oversubtracted=n_osub)
    if len(spec['filters']) > 1:
        comp = spec['filters'][1]
        cmap, cfr = _companion_snr_map(rdir, comp, spec, phase)
        cx, cy = cfr.xy(isc[keep])
        out['companion_snr'] = forced_snr_at(cmap, cx, cy).tolist()
    return out


def evaluate_field(spec, variant, *, phase=None):
    """All metrics of one field for one variant (missing runs are skipped)."""
    res = dict(field=spec['name'], variant=variant, environment=spec['environment'])
    try:
        res['clean'] = evaluate_clean(spec, variant, phase=phase)
    except FileNotFoundError as ex:
        res['clean_missing'] = str(ex)
    injected = []
    for seed in spec['seeds']:
        try:
            injected.append(evaluate_injected(spec, variant, seed, phase=phase))
        except FileNotFoundError as ex:
            res.setdefault('injected_missing', []).append(str(ex))
    if injected:
        snr = np.concatenate([np.asarray(r['snr_true']) for r in injected])
        rec = np.concatenate([np.asarray(r['recovered'], bool) for r in injected])
        dm = np.concatenate([np.asarray(r['dmag']) for r in injected])
        res['completeness'] = completeness_by_bin(snr, rec, spec['snr_bins'])
        # per injected star, in seed then table order (the same order for
        # every variant): pairs two variants star by star (paired_comparison)
        res['injected_seed'] = [r['seed'] for r in injected for _ in r['recovered']]
        res['injected_snr'] = snr.tolist()
        res['injected_recovered'] = rec.tolist()
        bright = rec & (snr >= spec['bias_min_snr'])
        res['flux_bias_mag'] = float(np.nanmedian(dm[bright])) if bright.any() else float('nan')
        # standard error of that median (1.253 sigma/sqrt(n), sigma robust)
        nb = int(np.isfinite(dm[bright]).sum())
        res['flux_bias_n'] = nb
        res['flux_bias_err_mag'] = (float(1.2533 * mad_std(dm[bright], ignore_nan=True) / np.sqrt(nb))
                                    if nb > 1 else float('nan'))
        res['n_injected'] = int(len(snr))
        res['seed_scalars'] = {
            key: dict(median=float(np.median(v)),
                      std=float(np.std(v, ddof=1)) if len(v) > 1 else float('nan'),
                      n=int(len(v)))
            for key in ('residual_excess', 'oversubtracted')
            for v in [np.array([r[key] for r in injected], float)]}
        if 'companion_snr' in injected[0]:
            csnr = np.concatenate([np.asarray(r['companion_snr']) for r in injected])
            lo, hi = spec['faint_snr']
            sel = (snr >= lo) & (snr < hi) & np.isfinite(csnr)
            res['r_real'] = float(np.mean(csnr[sel] > spec['confirm_snr'])) if sel.any() else float('nan')
    if 'clean' in res and 'r_on' in res['clean'] and 'r_real' in res:
        c = res['clean']
        res['emission_purity'] = purity_estimate(c['r_on'], c['r_off'], res['r_real'])
    return res


def paired_comparison(res, base, bins):
    """Star-by-star recovery of two variants on the same injected stars, per
    injected-S/N bin: ``{bin: (gained, lost, p)}`` with ``gained`` the stars
    ``res`` recovers and ``base`` does not, ``lost`` the reverse, and ``p`` the
    exact two-sided McNemar p-value (binomial test of gained vs lost at 1/2).
    Stars both recover or both miss carry no information on the difference."""
    from scipy.stats import binomtest
    for key in ('injected_seed', 'injected_snr'):
        if res[key] != base[key]:
            raise ValueError(f'{key} differs: the two results are not the same injections')
    snr = np.asarray(res['injected_snr'], float)
    a = np.asarray(res['injected_recovered'], bool)
    b = np.asarray(base['injected_recovered'], bool)
    out = {}
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (snr >= lo) & (snr < hi)
        gained, lost = int((a & ~b & sel).sum()), int((b & ~a & sel).sum())
        p = float(binomtest(gained, gained + lost, 0.5).pvalue) if gained + lost else 1.0
        out[f'{lo:g}-{hi:g}'] = (gained, lost, p)
    return out


def check(res, thresholds):
    """Threshold failures of one field's result (empty list = pass).  A
    threshold whose metric is missing or NaN counts as a failure."""
    fails = []
    comp = res.get('completeness', {})
    for b, fmin in (thresholds.get('completeness_min') or {}).items():
        n, k, frac = comp.get(b, (0, 0, float('nan')))
        if not (frac >= fmin):
            fails.append(f'completeness[{b}] = {frac:.2f} ({k}/{n}) < {fmin}')
    scalar = [('flux_bias_max_mag', lambda r: abs(r.get('flux_bias_mag', np.nan)), 'max'),
              ('residual_excess_max', lambda r: r.get('clean', {}).get('residual_excess', np.nan), 'max'),
              ('oversubtracted_max', lambda r: r.get('clean', {}).get('oversubtracted', np.nan), 'max'),
              ('ring_ratio_max', lambda r: r.get('clean', {}).get('ring_ratio', np.nan), 'max'),
              ('emission_purity_min', lambda r: r.get('emission_purity', np.nan), 'min'),
              ('labels_recovered_min', lambda r: r.get('clean', {}).get('labels_recovered', np.nan), 'min'),
              ('emission_labels_cataloged_max',
               lambda r: r.get('clean', {}).get('emission_labels_cataloged', np.nan), 'max')]
    for key, get, kind in scalar:
        if key not in thresholds:
            continue
        v = get(res)
        ok = (v <= thresholds[key]) if kind == 'max' else (v >= thresholds[key])
        if not ok:
            fails.append(f'{key}: {v:.3g} vs {thresholds[key]}')
    return fails


# ---------------------------------------------------------------------------
# threshold calibration
# ---------------------------------------------------------------------------
def floor_down(x, step=0.01):
    return math.floor(round(x / step, 6)) * step


def ceil_up(x, step=0.05):
    return math.ceil(round(x / step, 6)) * step


def completeness_floor(k, n, *, nsigma=2.0, min_floor=0.05):
    """The floor of a bin recovered ``k`` of ``n`` in calibration, or None."""
    if n <= 0:
        return None
    p = k / n
    pt = (k + 1) / (n + 2)
    f = floor_down(p - nsigma * math.sqrt(pt * (1 - pt) / n))
    return round(f, 2) if f >= min_floor else None


def thresholds_from(res, spec, *, nsigma=2.0, min_floor=0.05, bias_floor=0.1):
    """``(thresholds, calibration)`` for one field from its ``evaluate_field``
    result ``res`` (JSON form).  Metrics absent from ``res`` get no threshold."""
    thr, cal = {}, {}
    comp = res.get('completeness') or {}
    floors = {}
    for b, (n, k, _) in comp.items():
        f = completeness_floor(k, n, nsigma=nsigma, min_floor=min_floor)
        if f is not None:
            floors[b] = f
    cal['completeness'] = {b: [int(k), int(n)] for b, (n, k, _) in comp.items()}
    if floors:
        thr['completeness_min'] = floors
    clean = res.get('clean') or {}
    ss = res.get('seed_scalars') or {}
    for key in ('residual_excess', 'oversubtracted'):
        v, sd = clean.get(key), (ss.get(key) or {}).get('std')
        if v is None or sd is None or not np.isfinite(sd):
            continue
        thr[f'{key}_max'] = round(ceil_up(v + nsigma * sd), 2)
        cal[key] = dict(clean=round(v, 3), seed_std=round(sd, 3),
                        seed_median=round(ss[key]['median'], 3), n_seeds=ss[key]['n'])
    fb, fe = res.get('flux_bias_mag'), res.get('flux_bias_err_mag')
    if fb is not None and fe is not None and np.isfinite(fe):
        thr['flux_bias_max_mag'] = round(max(bias_floor, ceil_up(abs(fb) + nsigma * fe, 0.01)), 2)
        cal['flux_bias_mag'] = dict(median=round(fb, 3), err=round(fe, 3), n=res.get('flux_bias_n'))
    if 'labels_recovered' in clean and clean['labels_recovered'] is not None:
        n = clean['n_labels']
        k = int(round(clean['labels_recovered'] * n))
        f = completeness_floor(k, n, nsigma=nsigma, min_floor=min_floor)
        if f is not None:
            thr['labels_recovered_min'] = f
        cal['labels_recovered'] = [k, n]
    if 'emission_labels_cataloged' in clean and clean['emission_labels_cataloged'] is not None:
        thr['emission_labels_cataloged_max'] = int(clean['emission_labels_cataloged'])
        cal['emission_labels_cataloged'] = [int(clean['emission_labels_cataloged']),
                                            int(clean['n_emission_labels'])]
    return thr, cal


def pass_probability(thr, cal, *, nsim=20000, seed=0):
    """Probability that a NEUTRAL change passes every threshold of one field:
    one that keeps each metric's true value at the calibration value but draws
    it afresh (each completeness bin and the labels a new binomial sample at
    the calibration fraction; the residual scalars normal with the seed
    standard deviation; the bias normal with its standard error).  A pessimistic
    model: a real change re-draws only the marginal stars."""
    rng = np.random.default_rng(seed)
    ok = np.ones(nsim, bool)
    for b, f in (thr.get('completeness_min') or {}).items():
        k, n = cal['completeness'][b]
        ok &= rng.binomial(n, k / n, nsim) / n >= f
    for key in ('residual_excess', 'oversubtracted'):
        if f'{key}_max' in thr:
            c = cal[key]
            ok &= rng.normal(c['clean'], c['seed_std'], nsim) <= thr[f'{key}_max']
    if 'flux_bias_max_mag' in thr:
        c = cal['flux_bias_mag']
        ok &= np.abs(rng.normal(c['median'], c['err'], nsim)) <= thr['flux_bias_max_mag']
    if 'labels_recovered_min' in thr:
        k, n = cal['labels_recovered']
        ok &= rng.binomial(n, k / n, nsim) / n >= thr['labels_recovered_min']
    return float(ok.mean())


def _yaml_block(name, thr, cal, variant, commit, nsigma):
    lines = [f'  # {name}', '    thresholds:']
    for key, v in thr.items():
        if isinstance(v, dict):
            inner = ', '.join(f'{b}: {f}' for b, f in v.items())
            lines.append(f'      {key}: {{{inner}}}')
        else:
            lines.append(f'      {key}: {v}')
    lines.append('    calibration:')
    lines.append(f'      variant: {variant}')
    if commit:
        lines.append(f"      commit: '{commit}'")
    lines.append(f'      nsigma: {nsigma}')
    for key, v in cal.items():
        if isinstance(v, dict) and key == 'completeness':
            inner = ', '.join(f'{b}: [{k}, {n}]' for b, (k, n) in v.items())
            lines.append(f'      completeness: {{{inner}}}   # [recovered, injected]')
        elif isinstance(v, dict):
            inner = ', '.join(f'{a}: {b}' for a, b in v.items())
            lines.append(f'      {key}: {{{inner}}}')
        else:
            lines.append(f'      {key}: {list(v)}')
    return '\n'.join(lines)


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


def calibrate_main(path, fields, *, commit='', nsigma=2.0):
    """Print the ``thresholds:``/``calibration:`` blocks of every field in the
    ``--json`` output ``path``, each field's neutral-change pass probability
    and their product."""
    with open(path) as fh:
        allres = json.load(fh)
    joint = 1.0
    for name, res in allres.items():
        thr, cal = thresholds_from(res, fields[name], nsigma=nsigma)
        pp = pass_probability(thr, cal)
        joint *= pp
        print(_yaml_block(name, thr, cal, res.get('variant'), commit, nsigma))
        print(f'    # neutral-change pass probability: {pp:.3f}')
    print(f'# all fields: {joint:.3f}')
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--variant', help='run label to score')
    p.add_argument('--fields', default='')
    p.add_argument('--json', default='')
    p.add_argument('--baseline', default='',
                   help='evaluate --json output of a baseline variant: also print the '
                        'star-by-star gained/lost counts against it (paired_comparison)')
    p.add_argument('--phase', default=None,
                   help='score this phase (default: the last present, m7 else m6)')
    p.add_argument('--calibrate', default='', metavar='JSON',
                   help='print fields.yaml thresholds + calibration blocks from this '
                        '--json output of the calibration variant, then exit')
    p.add_argument('--commit', default='', help='with --calibrate: code commit of the runs')
    p.add_argument('--nsigma', type=float, default=2.0, help='with --calibrate: margin')
    a = p.parse_args(argv)
    _, fields = RF.load_config()
    if a.calibrate:
        return calibrate_main(a.calibrate, fields, commit=a.commit, nsigma=a.nsigma)
    if not a.variant:
        p.error('--variant is required unless --calibrate is given')
    names = [n for n in a.fields.split(',') if n] or list(fields)
    base = None
    if a.baseline:
        with open(a.baseline) as fh:
            base = json.load(fh)
    allres = {}
    nfail = 0
    for name in names:
        spec = fields[name]
        res = evaluate_field(spec, a.variant, phase=a.phase)
        res['failures'] = check(res, spec['thresholds'])
        nfail += len(res['failures'])
        allres[name] = res
        c = res.get('clean', {})
        ss = res.get('seed_scalars', {})
        spread = {k: '{median:.2f}+-{std:.2f}'.format(**ss[k]) if k in ss else '-'
                  for k in ('residual_excess', 'oversubtracted')}
        comp = ' '.join(f'{b}:{v[1]}/{v[0]}' for b, v in res.get('completeness', {}).items())
        print(f"{name:12s} {a.variant:10s} complete[{comp}] "
              f"bias={res.get('flux_bias_mag', np.nan):+.3f} "
              f"excess={c.get('residual_excess', np.nan):.2f}/as2 (seeds {spread['residual_excess']}) "
              f"oversub={c.get('oversubtracted', np.nan):.2f}/as2 (seeds {spread['oversubtracted']}) "
              f"ring={c.get('ring_ratio', np.nan):.2f} "
              f"purity={res.get('emission_purity', np.nan):.2f} "
              f"labels={c.get('labels_recovered', np.nan):.2f} "
              f"knots={c.get('emission_labels_cataloged', '-')}/{c.get('n_emission_labels', '-')} "
              f"-> {'PASS' if not res['failures'] else 'FAIL: ' + '; '.join(res['failures'])}")
        if base and name in base and 'injected_recovered' in res:
            pc = paired_comparison(res, base[name], spec['snr_bins'])
            res['paired_vs_baseline'] = pc
            print(f"{'':12s} vs {base[name].get('variant')}: " + ' '.join(
                f'{b}:+{g}/-{l} (p={p:.2g})' for b, (g, l, p) in pc.items()))
    if a.json:
        with open(a.json, 'w') as fh:
            json.dump(_jsonable(allres), fh, indent=1)
    return 1 if nfail else 0


if __name__ == '__main__':
    raise SystemExit(main())
