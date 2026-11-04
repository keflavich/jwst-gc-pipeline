"""Cross-exposure consistency of the per-frame products at the m2 merge.

Every m12 frame runs alone: it finds the stars it can see and fits them without
knowing what the other exposures of the same sky contain.  Nothing afterwards
compared them, so a bright star that one exposure carries and its neighbours do
not (a NaN core beside it, a masked neighbourhood, a cosmic-ray snowball taken
for a star) went unrecorded.

After the per-frame merge the merged catalog says which stars exist.  For every
merged row and every frame of the merge this module asks

* does the frame COVER the position?  The position is inside the array, clear
  of the border margin, and not inside a masked region larger than
  ``max_hole_pix``.  A saturated or NaN core is a hole inside a frame that
  holds the star, so it counts as covered; a masked detector strip does not.
                                                                 -> ``ncover``
* does the frame carry a COUNTERPART?  A daophot row within ``match_arcsec``
  whose flux is within ``flux_factor``, or a satstar row within
  ``sat_match_arcsec``.                                          -> ``nfound``

and flags (``single_exposure``) the bright or saturated rows that two or more
frames cover and exactly one carries.  Nothing is added, removed, refitted or
reordered: the columns are for vetting and QA.

Why an audit and no refit of the missing stars.  ``build_mergedcat_residuals``
recovers each frame's data as ``raw residual + raw model`` and subtracts a
render of the vetted merged catalog, so a star in the merged catalog is removed
from every frame that covers it, whether or not that frame fitted it.  The rows
this audit can flag are the ones where the merge's own evidence is thin.

The module is I/O-light on purpose: the WCS loader is passed in, so the numerics
run without the pipeline.
"""
import os
import re
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy import ndimage
from scipy.spatial import cKDTree

SCHEMA = 1

# DO_NOT_USE | SATURATED.  The hook passes the instrument's own mask.
DEFAULT_BAD_DQ = 3


class XexpError(RuntimeError):
    """A frame's products cannot be read or paired; that frame is skipped."""


@dataclass(frozen=True)
class XexpParams:
    # counterpart
    match_arcsec: float = 0.10
    sat_match_arcsec: float = 0.30
    flux_factor: float = 3.0
    # coverage: border margin, and the largest masked region that is a hole in
    # a frame (a saturated core is a few hundred pixels at most in gc-treasury
    # o078; a detector strip is tens of thousands)
    cover_margin_pix: float = 6.0
    max_hole_pix: int = 2000
    # `single_exposure` is for BRIGHT rows: saturated, or merged flux with S/N of
    # at least `single_snr_min` AND among the brightest `single_top_pct` percent
    # of the rows that two or more frames cover (100 switches the rank cut off).
    # The S/N alone does not hold across bands: the catalogs' flux errors leave
    # most of the F480M rows of o078 above 50 (70% of the multi-covered ones),
    # with a median pixel peak S/N of 4 in the frame that carries them.
    single_snr_min: float = 50.0
    single_top_pct: float = 5.0

    def __post_init__(self):
        if not 0.0 < self.single_top_pct <= 100.0:
            raise ValueError('single_top_pct must be in (0, 100], got '
                             f'{self.single_top_pct}')


def params_from_options(options):
    """``XexpParams`` from the ``manual_xexp_*`` options."""
    from jwst_gc_pipeline.photometry.manual_defaults import mopt
    return XexpParams(
        single_snr_min=float(mopt(options, 'manual_xexp_single_snr')),
        single_top_pct=float(mopt(options, 'manual_xexp_single_top_pct')))


# ---------------------------------------------------------------------------
# Naming: merged-catalog header -> per-frame products
# ---------------------------------------------------------------------------

_FN_KEY = re.compile(r'fn(\d+)', re.IGNORECASE)


def frame_catalog_names(meta):
    """Per-frame catalog basenames a merge recorded (``fn0``, ``fn1``, ...), in
    merge order.  Keys are matched case-insensitively: the FITS header spells
    them ``FN0`` and the in-memory merge ``fn0``."""
    named = {}
    for key, value in meta.items():
        m = _FN_KEY.fullmatch(str(key))
        if m:
            named[int(m.group(1))] = os.path.basename(str(value))
    return [named[i] for i in sorted(named)]


def split_catalog_name(basename, filt):
    """``(detector, obs, tail)`` of ``{filt}_{det}_o{obs}{tail}.fits``, or None.

    ``tail`` starts at the first token after the observation
    (``_visit001_vgroup02101_exp00001_hybpsf_m2_daophot_basic``).
    """
    m = re.fullmatch(
        rf'{re.escape(str(filt).lower())}_([a-z0-9]+)_o([0-9a-z]+)(_.*)\.fits',
        basename, flags=re.IGNORECASE)
    if m is None:
        return None
    return m.group(1).lower(), m.group(2), m.group(3)


@dataclass(frozen=True)
class FrameFiles:
    name: str            # per-frame catalog basename, the merge's fn<i>
    detector: str
    catalog: str
    crf: str
    satstar: str         # '' when the frame has no satstar catalog
    error: str = ''      # why the frame could not be resolved; '' when it could


def _catalog_crf(catalog):
    with fits.open(catalog, memmap=True) as hdul:
        hdr = hdul[1].header if len(hdul) > 1 else hdul[0].header
        crf = hdr.get('FILENAME') or hdul[0].header.get('FILENAME')
    if not crf:
        raise XexpError(f'{catalog}: no FILENAME in the header; cannot find the crf')
    return str(crf)


def resolve_frame_files(name, *, catalog_dirs, filt, satstar_suffix):
    """Catalog, crf and satstar catalog of one frame.

    The catalog is looked up by name in ``catalog_dirs``; the crf is the
    ``FILENAME`` its header records; the satstar catalog is the crf's name with
    ``satstar_suffix`` + ``_satstar_catalog.fits``.  A frame with no satstar
    catalog had no saturated stars.
    """
    parsed = split_catalog_name(name, filt)
    if parsed is None:
        raise XexpError(f'{name}: not a {filt} per-frame catalog name')
    catalog = next((os.path.join(d, name) for d in catalog_dirs
                    if os.path.exists(os.path.join(d, name))), None)
    if catalog is None:
        raise XexpError(f'{name}: not found in {list(catalog_dirs)}')
    crf = _catalog_crf(catalog)
    if not os.path.exists(crf):
        raise XexpError(f'{name}: crf {crf} does not exist')
    satstar = crf.replace('.fits', f'{satstar_suffix}_satstar_catalog.fits')
    return FrameFiles(name=name, detector=parsed[0], catalog=catalog, crf=crf,
                      satstar=satstar if os.path.exists(satstar) else '')


def resolve_frames(names, *, catalog_dirs, filt, satstar_suffix):
    """``resolve_frame_files`` for every name; a frame that cannot be resolved
    comes back with ``error`` set, in the merge's order, and is skipped later."""
    out = []
    for name in names:
        try:
            out.append(resolve_frame_files(
                name, catalog_dirs=catalog_dirs, filt=filt,
                satstar_suffix=satstar_suffix))
        except (XexpError, OSError, ValueError, KeyError) as exc:
            out.append(FrameFiles(name=name, detector='', catalog='', crf='',
                                  satstar='',
                                  error=f'{type(exc).__name__}: {exc}'))
    return out


def frames_newer_than(frames, reference_path):
    """How many frames have a daophot or satstar catalog written after
    ``reference_path`` (the merged catalog under audit).

    The audit compares the merged catalog with the per-frame catalogs now on
    disk.  When a rerun has replaced them since the merge, the counts compare
    two runs of the pipeline: ``none`` (merged rows no covering frame carries)
    reads far higher than it does against the frames the merge was made from,
    and ``one`` and ``single_exposure`` shift.  A frame that could not be
    resolved has no files and is not counted.
    """
    reference = os.path.getmtime(reference_path)
    return sum(1 for ff in frames
               if any(os.path.getmtime(p) > reference
                      for p in (ff.catalog, ff.satstar) if p))


# ---------------------------------------------------------------------------
# Coverage and counterparts (pure numpy)
# ---------------------------------------------------------------------------

def bad_dq_mask_for(filt):
    """The pipeline's bad-pixel DQ bitmask for the instrument of ``filt``
    (NIRCam: DO_NOT_USE | SATURATED).  Imports the long catalog module."""
    from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as _L
    return _L._bad_dq_bitmask(_L._instrument_from_filter(filt))


def usable_pixels(dq, bad_dq_mask=DEFAULT_BAD_DQ):
    """True where the DQ plane holds no bad-pixel bit."""
    return (np.asarray(dq).astype(np.int64) & int(bad_dq_mask)) == 0


def covered_positions(x, y, usable, params):
    """Which positions (pixels of this frame) the frame covers.

    Inside the array, at least ``cover_margin_pix`` from every edge, and not in
    a masked region (connected unusable pixels) of more than ``max_hole_pix``
    pixels.  The margin ring is cleared before the regions are labelled, so a
    saturated core near an edge does not join the border into one giant region.
    """
    ny, nx = usable.shape
    m = float(params.cover_margin_pix)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    inside = (np.isfinite(x) & np.isfinite(y)
              & (x >= m) & (x <= nx - 1 - m) & (y >= m) & (y <= ny - 1 - m))
    covered = inside.copy()
    if not inside.any() or usable.all():
        return covered
    bad = ~usable
    ring = int(np.ceil(m))
    bad[:ring, :] = False
    bad[ny - ring:, :] = False
    bad[:, :ring] = False
    bad[:, nx - ring:] = False
    labels, n_regions = ndimage.label(bad)
    if n_regions == 0:
        return covered
    # sizes[0] counts the usable pixels (label 0); never a "big region"
    big = np.bincount(labels.ravel()) > int(params.max_hole_pix)
    big[0] = False
    ix = np.rint(x[inside]).astype(int)
    iy = np.rint(y[inside]).astype(int)
    covered[inside] = ~big[labels[iy, ix]]
    return covered


def frame_cover_and_match(x, y, usable, cat_x, cat_y, cat_flux, sat_x, sat_y,
                          flux, is_sat, scale_arcsec, params):
    """Which merged rows a frame covers and which of them it carries.

    ``x``, ``y`` are the merged rows in this frame's pixel frame (NaN for rows
    the WCS cannot place).  ``cat_*`` are the frame's daophot rows, ``sat_*`` its
    satstar rows (pixels).  A merged saturated row may match a daophot row out
    to ``sat_match_arcsec`` with no upper flux bound: the same star unsaturated
    in this frame is a daophot row whose position scatters more and whose flux
    can sit below the saturated fit.

    Returns ``(covered, found)``; ``found`` implies ``covered``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    found = np.zeros(len(x), bool)
    covered = covered_positions(x, y, usable, params)
    ci = np.flatnonzero(covered)
    if ci.size == 0:
        return covered, found
    q = np.column_stack([x[ci], y[ci]])
    r_dao = params.match_arcsec / scale_arcsec
    r_sat = params.sat_match_arcsec / scale_arcsec
    fac = float(params.flux_factor)
    hit = np.zeros(ci.size, bool)
    if len(cat_x):
        k = min(6, len(cat_x))
        d, idx = cKDTree(np.column_stack([cat_x, cat_y])).query(
            q, k=k, distance_upper_bound=max(r_dao, r_sat))
        d = d.reshape(ci.size, k)
        idx = idx.reshape(ci.size, k)
        have = np.isfinite(d)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = np.asarray(cat_flux)[np.where(have, idx, 0)] / flux[ci][:, None]
        sat_row = is_sat[ci][:, None]
        in_r = np.where(sat_row, d <= r_sat, d <= r_dao)
        in_flux = np.where(sat_row, ratio > 1.0 / fac,
                           (ratio > 1.0 / fac) & (ratio < fac))
        hit |= (have & in_r & in_flux).any(axis=1)
    if len(sat_x):
        ds, _ = cKDTree(np.column_stack([sat_x, sat_y])).query(
            q, distance_upper_bound=r_sat)
        hit |= np.isfinite(ds)
    found[ci] = hit
    return covered, found


def footprint_candidates(wcs, shape, ra, dec):
    """Rows (deg) that can fall on a frame: the sky bounding box of the array's
    edges, padded by 10%.  A cheap guard, so the exact world-to-pixel transform
    runs only on the rows near this frame (one detector of a module mosaic
    holds an eighth of a merged catalog)."""
    ny, nx = shape
    t = np.linspace(0.0, 1.0, 17)
    ex = np.r_[t * (nx - 1), t * (nx - 1), np.zeros_like(t), np.full_like(t, nx - 1)]
    ey = np.r_[np.zeros_like(t), np.full_like(t, ny - 1), t * (ny - 1), t * (ny - 1)]
    era, edec = wcs.pixel_to_world_values(ex, ey)
    era = np.asarray(era, dtype=float)
    edec = np.asarray(edec, dtype=float)
    # circular mean: a frame that straddles RA = 0 has edge RAs near 0 and 360
    ra_rad = np.deg2rad(era)
    ra_c = float(np.rad2deg(np.arctan2(np.sin(ra_rad).mean(),
                                       np.cos(ra_rad).mean())))
    dec_c = float(edec.mean())
    cosd = np.cos(np.deg2rad(dec_c))

    def offsets(a, d):
        dra = ((np.asarray(a, dtype=float) - ra_c + 180.0) % 360.0) - 180.0
        return dra * cosd, np.asarray(d, dtype=float) - dec_c

    ex_off, ey_off = offsets(era, edec)
    pad = 0.1 * max(np.ptp(ex_off), np.ptp(ey_off))
    rx, ry = offsets(ra, dec)
    return (np.isfinite(rx) & np.isfinite(ry)
            & (rx >= ex_off.min() - pad) & (rx <= ex_off.max() + pad)
            & (ry >= ey_off.min() - pad) & (ry <= ey_off.max() + pad))


# ---------------------------------------------------------------------------
# Per-frame inputs
# ---------------------------------------------------------------------------

def _read_xyf(path, flux_col='flux_fit', detector_frame=False):
    """Detector pixel positions and fluxes of the finite, positive-flux rows.

    A daophot catalog's ``x_fit``/``y_fit`` are detector pixels.  A satstar
    catalog's are cutout-frame coordinates (the fit runs in a pad-centred
    stamp, so most rows read about (pad, pad)); its detector pixels are
    ``xcentroid``/``ycentroid``.  ``detector_frame=True`` reads those when both
    are present and falls back to ``x_fit``/``y_fit`` otherwise, the rule
    ``merge_catalogs.replace_saturated`` applies.
    """
    try:
        with fits.open(path, memmap=True) as hdul:
            data = hdul[1].data if len(hdul) > 1 else None
            if data is None or len(data) == 0:
                return np.zeros(0), np.zeros(0), np.zeros(0)
            names = set(data.columns.names)
            if detector_frame and {'xcentroid', 'ycentroid'} <= names:
                xcol, ycol = 'xcentroid', 'ycentroid'
            else:
                xcol, ycol = 'x_fit', 'y_fit'
            x = np.array(data[xcol], dtype=float)
            y = np.array(data[ycol], dtype=float)
            f = np.array(data[flux_col], dtype=float)
    except TypeError as exc:
        # a table cut short in its data raises "buffer is too small" here
        raise XexpError(f'{path}: truncated table ({exc})') from exc
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(f) & (f > 0)
    return x[ok], y[ok], f[ok]


def _pixel_scale_arcsec(catalog, wcs, shape):
    with fits.open(catalog, memmap=True) as hdul:
        hdr = hdul[1].header if len(hdul) > 1 else hdul[0].header
        scale = float(hdr.get('PIXSCALE', 0.0) or 0.0)
    if scale > 0.0:
        return scale
    cx, cy = shape[1] / 2.0, shape[0] / 2.0
    ra, dec = wcs.pixel_to_world_values([cx, cx + 1.0], [cy, cy])
    dra = (ra[1] - ra[0]) * np.cos(np.deg2rad(dec[0]))
    return float(np.hypot(dra, dec[1] - dec[0]) * 3600.0)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

@dataclass
class XexpResult:
    ncover: np.ndarray
    nfound: np.ndarray
    single_exposure: np.ndarray
    frames: Table


def bright_rows(flux, flux_err, is_sat, snr_min, top_pct=100.0, multi=None):
    """The rows ``single_exposure`` may name: saturated rows, and rows whose
    merged flux has S/N of at least ``snr_min`` and lies in the brightest
    ``top_pct`` percent of the rows in ``multi`` (those two or more frames
    cover; every row when ``multi`` is None).  The cut is the flux quantile of
    the finite-flux rows of ``multi`` and a row at the cut is kept.  ``top_pct``
    of 100 or more applies no rank cut.
    """
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.where(flux_err > 0, flux / flux_err, 0.0)
    keep = np.isfinite(snr) & (snr >= snr_min)
    if top_pct < 100.0:
        pool = np.isfinite(flux)
        if multi is not None:
            pool &= np.asarray(multi, dtype=bool)
        if not pool.any():
            return np.asarray(is_sat, dtype=bool)
        cut = np.quantile(flux[pool], 1.0 - top_pct / 100.0)
        keep &= np.isfinite(flux) & (flux >= cut)
    return np.asarray(is_sat, dtype=bool) | keep


def reconcile(ra, dec, flux, flux_err, is_sat, frames, params, *, wcs_loader,
              bad_dq_mask=DEFAULT_BAD_DQ, log=print):
    """Coverage / counterpart accounting of a merge.

    ``ra``, ``dec`` (deg), ``flux``, ``flux_err``, ``is_sat`` describe the
    merged rows.  ``frames`` is a list of ``FrameFiles``.  ``wcs_loader(crf)``
    gives an object with ``world_to_pixel_values`` and ``pixel_to_world_values``.

    A frame whose products or GWCS cannot be read is skipped and listed in
    ``frames`` with its reason; ``ncover`` then counts only the frames that were
    read.
    """
    ra = np.asarray(ra, dtype=float)
    dec = np.asarray(dec, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)
    is_sat = np.asarray(is_sat, dtype=bool)
    n = len(ra)
    ncover = np.zeros(n, np.int16)
    nfound = np.zeros(n, np.int16)
    frame_rows = []

    for i, ff in enumerate(frames):
        row = {'frame': i, 'name': ff.name, 'status': 'ok', 'n_cover': 0,
               'n_found': 0, 'note': ''}
        frame_rows.append(row)
        if ff.error:
            row['status'], row['note'] = 'skipped', ff.error
            log(f'xexp: WARNING frame {i} {ff.name} skipped ({ff.error})')
            continue
        try:
            wcs = wcs_loader(ff.crf)
            dq = fits.getdata(ff.crf, 'DQ', memmap=False)
            usable = usable_pixels(dq, bad_dq_mask)
            cat_x, cat_y, cat_f = _read_xyf(ff.catalog)
            if ff.satstar:
                sat_x, sat_y, _ = _read_xyf(ff.satstar, detector_frame=True)
            else:
                sat_x, sat_y = np.zeros(0), np.zeros(0)
            scale = _pixel_scale_arcsec(ff.catalog, wcs, usable.shape)
            near = footprint_candidates(wcs, usable.shape, ra, dec)
            x = np.full(n, np.nan)
            y = np.full(n, np.nan)
            if near.any():
                xs, ys = wcs.world_to_pixel_values(ra[near], dec[near])
                x[near] = np.asarray(xs, dtype=float)
                y[near] = np.asarray(ys, dtype=float)
        except (XexpError, OSError, ValueError, KeyError, RuntimeError) as exc:
            # RuntimeError: frame_wcs.GwcsReadError, a GWCS that cannot be read
            row['status'], row['note'] = 'skipped', f'{type(exc).__name__}: {exc}'
            log(f'xexp: WARNING frame {i} {ff.name} skipped ({row["note"]})')
            continue
        covered, found = frame_cover_and_match(
            x, y, usable, cat_x, cat_y, cat_f, sat_x, sat_y, flux, is_sat,
            scale, params)
        ncover += covered
        nfound += found
        row['n_cover'], row['n_found'] = int(covered.sum()), int(found.sum())
        log(f'xexp: frame {i} {ff.name}: covers {row["n_cover"]}, carries '
            f'{row["n_found"]}')

    bright = bright_rows(flux, flux_err, is_sat, params.single_snr_min,
                         params.single_top_pct, multi=ncover >= 2)
    single = (ncover >= 2) & (nfound == 1) & bright
    frames_tab = Table(rows=frame_rows) if frame_rows else Table()
    return XexpResult(ncover, nfound, single, frames_tab)


def add_audit_columns(table, result, params):
    """Add ``ncover``, ``nfound`` and ``single_exposure`` to ``table`` in place
    and record the run in its header."""
    table['ncover'] = result.ncover
    table['nfound'] = result.nfound
    table['single_exposure'] = result.single_exposure
    ok = (result.frames['status'] != 'skipped').sum() if len(result.frames) else 0
    table.meta['XEXPVER'] = SCHEMA
    table.meta['XEXPFRM'] = int(ok)
    table.meta['XEXPSKP'] = int(len(result.frames) - ok)
    table.meta['XEXPSNR'] = float(params.single_snr_min)
    table.meta['XEXPTOP'] = float(params.single_top_pct)
    return table


def _column(table, name, default, dtype=float):
    """``table[name]`` as an unmasked array, ``default`` where masked or absent."""
    if name not in table.colnames:
        return np.full(len(table), default, dtype=dtype)
    col = table[name]
    if hasattr(col, 'filled'):
        col = col.filled(default)
    return np.asarray(col, dtype=dtype)


def saturated_rows(table):
    """Rows the merge fitted from a satstar model (``is_saturated`` or
    ``replaced_saturated``)."""
    return (_column(table, 'is_saturated', False, bool)
            | _column(table, 'replaced_saturated', False, bool))


def audit_merged_catalog(merged, *, filt, catalog_dirs, satstar_suffix, params,
                         wcs_loader, bad_dq_mask, log=print):
    """Add ``ncover``, ``nfound`` and ``single_exposure`` to a merged m2 table.

    The per-frame catalogs are the ``fn<i>`` cards the merge recorded in the
    table's header.  Returns the ``XexpResult``, or None when the table records
    no per-frame catalogs.  The table is modified in place.
    """
    from astropy.coordinates import SkyCoord
    names = frame_catalog_names(merged.meta)
    if not names:
        log('xexp: the merged table records no per-frame catalogs (no fn cards); '
            'audit skipped')
        return None
    sc = SkyCoord(merged['skycoord'])
    is_sat = saturated_rows(merged)
    frames = resolve_frames(names, catalog_dirs=catalog_dirs, filt=filt,
                            satstar_suffix=satstar_suffix)
    result = reconcile(np.asarray(sc.ra.deg), np.asarray(sc.dec.deg),
                       _column(merged, 'flux', np.nan),
                       _column(merged, 'flux_err', np.nan), is_sat, frames,
                       params, wcs_loader=wcs_loader, bad_dq_mask=bad_dq_mask,
                       log=log)
    add_audit_columns(merged, result, params)
    n_skip = int(merged.meta['XEXPSKP'])
    log(f'xexp: {filt}: {len(merged)} rows, {len(frames) - n_skip} of '
        f'{len(frames)} frames read; covered by >=2 frames '
        f'{int((result.ncover >= 2).sum())}, carried by exactly one '
        f'{int(((result.ncover >= 2) & (result.nfound == 1)).sum())}, flagged '
        f'single_exposure {int(result.single_exposure.sum())}')
    return result


def audit_summary(ncover, nfound, single_exposure, bright):
    """Counts of the audit columns, for the QA report.

    Rows covered by two or more frames are split by how many of their covering
    frames carry them: ``all`` (every one), ``some`` (more than one, not all),
    ``one`` (exactly one), ``none`` (the catalogs hold no counterpart).  The
    split is given for all rows and for the bright ones.
    """
    ncover = np.asarray(ncover)
    nfound = np.asarray(nfound)
    bright = np.asarray(bright, dtype=bool)
    multi = ncover >= 2
    out = {'rows': int(len(ncover)),
           'single_exposure': int(np.asarray(single_exposure).sum())}
    for tag, sel in (('', multi), ('bright_', multi & bright)):
        out[tag + 'multi'] = int(sel.sum())
        out[tag + 'all'] = int((sel & (nfound == ncover)).sum())
        out[tag + 'one'] = int((sel & (nfound == 1)).sum())
        out[tag + 'none'] = int((sel & (nfound == 0)).sum())
        out[tag + 'some'] = int((sel & (nfound > 1) & (nfound < ncover)).sum())
    return out
