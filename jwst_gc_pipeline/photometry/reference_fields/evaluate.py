"""Score reference-field runs against their fixed thresholds.

::

    python -m jwst_gc_pipeline.photometry.reference_fields.evaluate --variant main
    python -m jwst_gc_pipeline.photometry.reference_fields.evaluate --variant main --json out.json

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
"""
import argparse
import glob
import json
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


def evaluate_clean(spec, variant, *, rng=None, phase=None):
    """Residual excess, ring ratio, faint-source confirmation rates and label
    recovery of the clean run of ``spec`` (primary band) at ``phase`` (default:
    the last phase present)."""
    rng = rng or np.random.default_rng(0)
    rdir = RF.run_dir(spec, variant, 0)
    prim = spec['filters'][0]
    prod = find_products(rdir, prim, phase)
    res, err, wcs, _ = _image(prod['residual'])
    if prod['smoothed_bg'] is not None:
        res = res - _image(prod['smoothed_bg'])[0]
    fr = Frame(spec, wcs, res.shape)
    inner = fr.mask()
    sc, flux, ferr = load_catalog(prod['catalog'])
    x, y = fr.xy(sc)
    inside = fr.inside(x, y)
    snr_cat = flux / ferr
    fw = fwhm_pix(prim)
    snr_map, scale = matched_filter_snr(res, err, fw, inner)
    exc, npos, nneg = residual_excess(snr_map, inner, x, y, fr.area_as2,
                                      thresh=spec['excess_snr'], excl_pix=spec['excess_excl_pix'])
    rr, n_obs, n_exp = ring_ratio(x, y, flux, snr_cat, inside,
                                  bright_snr=spec['ring_bright_snr'],
                                  faint_ratio=spec['ring_faint_ratio'],
                                  ring=spec['ring_pix'], area_pix=float(inner.sum()))
    out = dict(phase=prod['phase'], n_sources=int(inside.sum()),
               density_per_as2=float(inside.sum() / fr.area_as2),
               residual_excess=float(exc), residual_npos=npos, residual_nneg=nneg,
               residual_mf_scale=scale, ring_ratio=rr, ring_n_obs=n_obs, ring_n_exp=n_exp)

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
        bright = rec & (snr >= spec['bias_min_snr'])
        res['flux_bias_mag'] = float(np.nanmedian(dm[bright])) if bright.any() else float('nan')
        res['n_injected'] = int(len(snr))
        if 'companion_snr' in injected[0]:
            csnr = np.concatenate([np.asarray(r['companion_snr']) for r in injected])
            lo, hi = spec['faint_snr']
            sel = (snr >= lo) & (snr < hi) & np.isfinite(csnr)
            res['r_real'] = float(np.mean(csnr[sel] > spec['confirm_snr'])) if sel.any() else float('nan')
    if 'clean' in res and 'r_on' in res['clean'] and 'r_real' in res:
        c = res['clean']
        res['emission_purity'] = purity_estimate(c['r_on'], c['r_off'], res['r_real'])
    return res


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
              ('ring_ratio_max', lambda r: r.get('clean', {}).get('ring_ratio', np.nan), 'max'),
              ('emission_purity_min', lambda r: r.get('emission_purity', np.nan), 'min'),
              ('labels_recovered_min', lambda r: r.get('clean', {}).get('labels_recovered', np.nan), 'min')]
    for key, get, kind in scalar:
        if key not in thresholds:
            continue
        v = get(res)
        ok = (v <= thresholds[key]) if kind == 'max' else (v >= thresholds[key])
        if not ok:
            fails.append(f'{key}: {v:.3g} vs {thresholds[key]}')
    return fails


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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--variant', required=True)
    p.add_argument('--fields', default='')
    p.add_argument('--json', default='')
    p.add_argument('--phase', default=None,
                   help='score this phase (default: the last present, m7 else m6)')
    a = p.parse_args(argv)
    _, fields = RF.load_config()
    names = [n for n in a.fields.split(',') if n] or list(fields)
    allres = {}
    nfail = 0
    for name in names:
        spec = fields[name]
        res = evaluate_field(spec, a.variant, phase=a.phase)
        res['failures'] = check(res, spec['thresholds'])
        nfail += len(res['failures'])
        allres[name] = res
        c = res.get('clean', {})
        comp = ' '.join(f'{b}:{v[1]}/{v[0]}' for b, v in res.get('completeness', {}).items())
        print(f"{name:12s} {a.variant:10s} complete[{comp}] "
              f"bias={res.get('flux_bias_mag', np.nan):+.3f} "
              f"excess={c.get('residual_excess', np.nan):.2f}/as2 "
              f"ring={c.get('ring_ratio', np.nan):.2f} "
              f"purity={res.get('emission_purity', np.nan):.2f} "
              f"labels={c.get('labels_recovered', np.nan):.2f} "
              f"-> {'PASS' if not res['failures'] else 'FAIL: ' + '; '.join(res['failures'])}")
    if a.json:
        with open(a.json, 'w') as fh:
            json.dump(_jsonable(allres), fh, indent=1)
    return 1 if nfail else 0


if __name__ == '__main__':
    raise SystemExit(main())
