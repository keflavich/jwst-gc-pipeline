"""Per-detector F200W plate-scale correction for NIRCam SW (opt-in).

WHAT IS BEING CORRECTED
-----------------------
After the in-detector rotation correction (``distortion_rotation_correction``,
issue #1135), F200W still differs in scale from the other SW bands by a
per-detector term (issue #1137): about +14 to +34 ppm on NRCA1-A4 and about
-8 to -14 ppm on NRCB1-B4, i.e. 0.3-1.1 mas at the detector edge midpoints
and up to 1.5 mas at the corners.

The term is measured on m6 per-frame catalogs as the pixel-frame scale of
(F200W - anchor), ``trace(C^-1 J) / 2``, where ``J`` is the linear fit of the
sky residual (F200W - anchor) against pixel position and ``C`` the local
pixel -> sky Jacobian.  Evidence (#1137):

* F200W - F150W agrees between wd2 and wd1 to 2.5 ppm rms, with no colour
  dependence (slopes +0.4 +- 0.4 ppm/mag).
* NGC 6334 F200W - F187N (same program) matches the wd2/wd1 mean to 9 ppm rms
  (r = 0.96); F200W - F182M on NGC 6334 and both anchors on the Brick show
  the same A-positive, B-negative contrast with 12-30 ppm scatter.
* F115W, F150W, F162M, F164N, F182M and F187N stay within about +-10 ppm of
  F212N on wd2, and the same-program F182M/F187N controls on the Brick show no
  A-minus-B contrast.  F200W is the outlier.
* The CRDS and STDGDC F200W references agree on the F200W/F212N scale to
  5 ppm, so neither library carries the term.

The origin is not known.  The correction puts F200W on the common scale of
the other SW bands, measured empirically; it says nothing about which of the
two is closer to the true plate scale.

WHAT THIS APPLIES
-----------------
An isotropic scale of the ``detector -> v2v3`` transform about the V2/V3
position of the full-frame detector centre (pixel 1023.5, 1023.5), by the
factor ``1 + correction_ppm * 1e-6`` from ``data/distortion_scales.ecsv``.
The pivot is the one ``distortion_rotation_correction`` uses, and a rotation
and an isotropic scale about the same point commute, so the two corrections
can run in either order.  The detector centre maps to the same sky position
before and after, so the reference shift, the module ties and the
array-centre fiducial used by ``fix_alignment`` are unaffected.

SIGN
----
``correction_ppm`` is the CORRECTION, already negated: minus the measured
scale of (F200W - F150W), the mean of wd2 and wd1.  A positive measured scale
means F200W positions sit too far from the detector centre, so the stored
value is negative there and shrinks the transform.

KEYING
------
Rows are keyed on ``(detector, filter, pupil, distortion_ref)``.  A frame whose
filter has rows but whose ``R_DISTOR`` matches none of them is refused: the
stored scale was measured with one reference file, and a new delivery would
need a new measurement (#1137).  A frame whose ``(detector, filter, pupil)``
has no rows is left unchanged.

Opt-in: ``DISTORTION_SCALE_CORRECTION=1``.  Applied in ``fix_alignment``
BEFORE the reference shift, idempotent via the ``DSCLCORR`` marker, with a
``DSCLPEND`` pending flag so a crash between the GWCS write and the marker
write fails loud instead of double-scaling.
"""
import copy
import os

import numpy as np
from astropy.io import fits
from astropy.modeling.models import Scale, Shift
from astropy.table import Table

from jwst_gc_pipeline.reduction.distortion_rotation_correction import (
    _norm, _pivot_pixel, _ref_basename)
from jwst_gc_pipeline.reduction.fits_wcs_sync import sync_header_to_gwcs

__all__ = ['TABLE', 'MARKER', 'PENDING', 'DistortionScaleError', 'correction_enabled',
           'load_scale_table', 'lookup_correction_ppm', 'scale_about_pivot_model',
           'scaled_wcs', 'verify_scale', 'check_apply_preconditions',
           'apply_distortion_scale_correction']

MARKER = 'DSCLCORR'
PENDING = 'DSCLPEND'
TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                     'distortion_scales.ecsv')

TABLE_COLUMNS = ('detector', 'filter', 'pupil', 'distortion_ref', 'correction_ppm',
                 'wd2_ppm', 'wd1_ppm')


class DistortionScaleError(RuntimeError):
    """The correction cannot be applied safely to this frame."""


def correction_enabled():
    # same test as the fix_alignment hook (and its sibling corrections): exactly '1'
    return os.environ.get('DISTORTION_SCALE_CORRECTION', '0') == '1'


def load_scale_table(path=TABLE):
    tbl = Table.read(path)
    missing = [c for c in TABLE_COLUMNS if c not in tbl.colnames]
    if missing:
        raise DistortionScaleError(f"{path}: missing column(s) {missing}; expected {TABLE_COLUMNS}")
    return tbl


def lookup_correction_ppm(detector, filtername, pupil, distortion_ref, table=None):
    """``correction_ppm`` for this frame, or None if nothing is to be done.

    Returns None when the table has no rows for ``(detector, filter, pupil)``.
    Raises ``DistortionScaleError`` when it has rows for them but none for
    this ``distortion_ref``.
    """
    tbl = load_scale_table() if table is None else table
    det, filt, pup = _norm(detector), _norm(filtername), _norm(pupil, 'CLEAR')
    rows = [r for r in tbl if _norm(r['detector']) == det and _norm(r['filter']) == filt
            and _norm(r['pupil'], 'CLEAR') == pup]
    if not rows:
        return None
    ref = _ref_basename(distortion_ref)
    match = [r for r in rows if _ref_basename(r['distortion_ref']) == ref]
    if not match:
        raise DistortionScaleError(
            f"distortion-scale table has rows for ({det}, {filt}, {pup}) with "
            f"reference(s) {sorted({str(r['distortion_ref']) for r in rows})}, but this frame "
            f"used {ref!r}.  The stored scale was measured with one reference file; "
            f"re-measure it for the new reference (issue #1137) or unset "
            f"DISTORTION_SCALE_CORRECTION.")
    return float(match[0]['correction_ppm'])


def scale_about_pivot_model(v2c, v3c, correction_ppm):
    """(V2, V3) -> (V2, V3) isotropic scale by ``1 + correction_ppm * 1e-6``
    about ``(v2c, v3c)``."""
    f = 1.0 + correction_ppm * 1e-6
    m = (Shift(-v2c) & Shift(-v3c)) | (Scale(f) & Scale(f)) | (Shift(v2c) & Shift(v3c))
    m.name = 'distortion_scale_correction'
    return m


def scaled_wcs(wcs, correction_ppm, xstart=1, ystart=1):
    """Copy of ``wcs`` with ``detector -> v2v3`` scaled by ``correction_ppm``
    about the full-frame detector centre.  Returns ``(new_wcs, (v2c, v3c))``."""
    t = wcs.get_transform('detector', 'v2v3')
    v2c, v3c = (float(v) for v in t(*_pivot_pixel(xstart, ystart)))
    new = t | scale_about_pivot_model(v2c, v3c, correction_ppm)
    try:
        new.bounding_box = t.bounding_box
    except NotImplementedError:
        pass
    out = copy.deepcopy(wcs)
    out.set_transform('detector', 'v2v3', new)
    return out, (v2c, v3c)


def verify_scale(wcs_before, wcs_after, pivot, correction_ppm, shape, tol_mas=0.05):
    """Max deviation (mas) of the V2/V3 change from a pure isotropic scale
    about ``pivot``, on a pixel grid.  Raises above ``tol_mas``."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny:max(ny // 8, 1), 0:nx:max(nx // 8, 1)]
    t0 = wcs_before.get_transform('detector', 'v2v3')
    t1 = wcs_after.get_transform('detector', 'v2v3')
    v20, v30 = t0(xx.ravel(), yy.ravel())
    v21, v31 = t1(xx.ravel(), yy.ravel())
    e2, e3 = scale_about_pivot_model(pivot[0], pivot[1], correction_ppm)(v20, v30)
    dev = np.hypot(v21 - e2, v31 - e3) * 1e3
    good = np.isfinite(dev)
    worst = float(np.max(dev[good])) if good.any() else np.inf
    if not worst <= tol_mas:
        raise DistortionScaleError(
            f"distortion-scale verification failed: V2/V3 change deviates from a "
            f"{correction_ppm:+.2f} ppm scale about the detector centre by up to {worst:.4f} mas "
            f"(tolerance {tol_mas} mas); NOT writing.")
    return worst


def check_apply_preconditions(hdr, label=""):
    """``'skip'`` if already applied, ``'go'`` if safe, raise otherwise.  Pure."""
    if MARKER in hdr:
        return 'skip'
    if hdr.get(PENDING):
        raise DistortionScaleError(
            f"{label}: pending distortion-scale marker without completion marker -- a "
            f"previous apply crashed mid-write.  Re-create this frame from its _cal; refusing "
            f"to guess the GWCS state.")
    if 'RAOFFSET' in hdr:
        raise DistortionScaleError(
            f"{label}: frame already carries a reference shift (RAOFFSET={hdr['RAOFFSET']}) "
            f"but no {MARKER}; the scale must be applied before the shift.  Re-create this "
            f"frame from its _cal and re-run fix_alignment with "
            f"DISTORTION_SCALE_CORRECTION=1.")
    return 'go'


def apply_distortion_scale_correction(fn, table=None, verbose=True):
    """Apply the scale for ``fn``'s (detector, filter, pupil, R_DISTOR) in
    place, updating the ASDF GWCS and the FITS SCI-header WCS, idempotently.
    Returns the applied ``correction_ppm``, or None when skipped."""
    hdr0 = fits.getheader(fn, ext=0)
    hdr = fits.getheader(fn, ext=('SCI', 1))
    if check_apply_preconditions(hdr, label=fn) == 'skip':
        if verbose:
            print(f"distortion-scale correction skipped for {fn}: already applied "
                  f"({hdr.get('DSCLPPM')} ppm)")
        return None
    det = hdr0.get('DETECTOR', hdr.get('DETECTOR'))
    filt = hdr0.get('FILTER', hdr.get('FILTER'))
    pupil = hdr0.get('PUPIL', hdr.get('PUPIL'))
    rdist = hdr0.get('R_DISTOR', hdr.get('R_DISTOR'))
    if not rdist:
        raise DistortionScaleError(f"{fn}: no R_DISTOR keyword; cannot identify the "
                                   f"distortion reference the scale applies to")
    corr = lookup_correction_ppm(det, filt, pupil, rdist, table=table)
    if corr is None:
        if verbose:
            print(f"distortion-scale correction not needed for {fn}: no row for "
                  f"({det}, {filt}, {pupil})")
        return None

    with fits.open(fn, mode='update') as hdul:
        hdul['SCI'].header[PENDING] = (True, 'distortion-scale apply in progress')

    from jwst.datamodels import ImageModel
    fa = ImageModel(fn)
    wcsobj = fa.meta.wcs
    ww, pivot = scaled_wcs(wcsobj, corr, xstart=fa.meta.subarray.xstart,
                           ystart=fa.meta.subarray.ystart)
    worst = verify_scale(wcsobj, ww, pivot, corr, fa.data.shape)
    fa.meta.oldwcs = copy.copy(wcsobj)
    fa.meta.wcs = ww
    fa.save(fn, overwrite=True)

    with fits.open(fn) as hdul:
        h = hdul['SCI'].header
        _sip_max, _sip_med = sync_header_to_gwcs(h, ww, fa.data.shape, label=os.path.basename(fn))
        h['SIPGWMAX'] = (_sip_max, '[mas] max FITS/SIP vs GWCS disagreement')
        h[MARKER] = (True, 'F200W per-detector scale applied (#1137)')
        h['DSCLPPM'] = (corr, '[ppm] V2V3 scale about detector centre')
        h['DSCLREF'] = (_ref_basename(rdist), 'distortion ref of DSCLPPM')
        h['DSCLPV2'] = (pivot[0], '[arcsec] scale pivot V2 (detector centre)')
        h['DSCLPV3'] = (pivot[1], '[arcsec] scale pivot V3 (detector centre)')
        h['DSCLVMAS'] = (worst, '[mas] max deviation from pure scale')
        h[PENDING] = (False, 'distortion-scale apply completed')
        hdul.writeto(fn, overwrite=True)
    if verbose:
        print(f"distortion-scale correction applied to {fn} ({det} {filt}/{pupil}): "
              f"{corr:+.2f} ppm about V2V3 ({pivot[0]:.3f}, {pivot[1]:.3f}); verified to "
              f"{worst:.4f} mas")
    return corr
