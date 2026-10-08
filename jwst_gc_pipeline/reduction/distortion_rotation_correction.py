"""In-detector rotation correction for NIRCam SW distortion references (opt-in).

WHAT IS BEING CORRECTED
-----------------------
The CRDS NIRCam SW distortion references fall into two groups that differ by a
rotation inside each detector (issue #1135):

* F182M, F187N, F210M and F212N agree with each other to < 1".
* F115W, F140M, F150W, F162M, F164N and F200W carry an extra rotation relative
  to them: 19-31" on NRCA2/NRCA3, 6-11" on NRCA1/NRCA4 and <= 2.5" on module B
  (F200W: -8/+19/+31/+1" on A1-A4).

All references share one ``(V2Ref, V3Ref, V3IdlYAngle)``, so the rotation sits
in the linear terms of each band's polynomial.  A 25" rotation moves positions
by 3.9 mas at the detector edge midpoints and 5.5 mas at the corners, and by
zero at the detector centre.

WHICH GROUP IS RIGHT
--------------------
Against Gaia DR3 (unsaturated stars, wd2 and wd1), F182M/F187N/F212N show an
in-detector term of about 1-9" on NRCA2/NRCA3, while F115W/F164N/F200W show
the full 20-31" reference-to-reference rotation.  Jay Anderson's STDGDC
solutions put every SW filter within 5.5" of F212N on module A.  The F212N
group is therefore taken as correct, and the other references are rotated
onto it.  A smaller term of 5-20", common to all bands, remains against Gaia
on several detectors (mainly module B); this module does not address it.

WHAT THIS APPLIES
-----------------
A rotation of the ``detector -> v2v3`` transform about the V2/V3 position of
the full-frame detector centre (pixel 1023.5, 1023.5), by the angle stored in
``data/distortion_rotations.ecsv``.  The angle is a property of the reference
pair alone: it is the rotation term of a similarity fit of the band's
pixel -> V2/V3 map against the anchor (F212N) map of the same detector
(``reference_rotation``).  On wd2 m6 per-frame catalogs, the pixel-frame
rotation of (band - F212N) left after the correction is <= 2.5" on every
detector for F115W, F150W, F162M, F164N and F200W, so no field solve is
needed.  Only the rotation is corrected.  The references also differ in scale
(F150W +15..+55 ppm, F162M/F164N about +210..+240 ppm against F212N); the data
follow those differences to about 10 ppm, so they are left alone.  F200W is
the exception: its data scale differs from the reference by +16..+42 ppm on
module A and about -17 ppm on NRCB2-4 (1-2 mas at the detector edge), while
STDGDC agrees with the CRDS F200W scale to 5 ppm.  That term is not corrected
here.

The detector centre maps to the same sky position before and after the
correction, so the per-exposure reference shift, the module ties and the
array-centre fiducial used by ``fix_alignment`` are unaffected to first order.

SIGN
----
``correction_arcsec`` is the CORRECTION, already negated: the rotation to
apply to the band's V2/V3 positions, in the ``v2 + i*v3`` sense (positive
rotates +V2 toward +V3).  It equals minus the rotation of the band reference
relative to the anchor reference.  Applying the measured rotation instead
doubles the error (measured on wd2: NRCA3 F150W 25.7" -> 50.3").

KEYING
------
Rows are keyed on ``(detector, filter, pupil, distortion_ref)``.  A frame whose
filter has rows but whose ``R_DISTOR`` matches none of them is refused: the
stored angle describes one reference file, and a new delivery may or may not
keep the rotation.  Regenerate the table with
``scripts/analysis/solve_distortion_rotations.py``.  A frame whose
``(detector, filter, pupil)`` has no rows (the anchor group, LW, or a filter
not validated against data) is left unchanged.

Opt-in: ``DISTORTION_ROTATION_CORRECTION=1``.  Applied in ``fix_alignment``
BEFORE the reference shift, idempotent via the ``DROTCORR`` marker, with a
``DROTPEND`` pending flag so a crash between the GWCS write and the marker
write fails loud instead of double-rotating.
"""
import copy
import os

import numpy as np
from astropy.io import fits
from astropy.modeling.models import Rotation2D, Shift
from astropy.table import Table

from jwst_gc_pipeline.reduction.fits_wcs_sync import sync_header_to_gwcs

__all__ = ['TABLE', 'MARKER', 'PENDING', 'DistortionRotationError', 'correction_enabled',
           'reference_rotation', 'load_rotation_table', 'lookup_correction_arcsec',
           'rotation_about_pivot_model', 'rotated_wcs', 'verify_rotation',
           'check_apply_preconditions', 'apply_distortion_rotation_correction']

MARKER = 'DROTCORR'
PENDING = 'DROTPEND'
TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data',
                     'distortion_rotations.ecsv')

#: Full-frame SW detector centre (0-based pixels), the rotation pivot.
FULL_FRAME_CENTRE = (1023.5, 1023.5)

TABLE_COLUMNS = ('detector', 'filter', 'pupil', 'distortion_ref', 'anchor_filter',
                 'anchor_ref', 'correction_arcsec', 'ref_scale_ppm', 'fit_rms_mas')


class DistortionRotationError(RuntimeError):
    """The correction cannot be applied safely to this frame."""


def correction_enabled():
    return str(os.environ.get('DISTORTION_ROTATION_CORRECTION', '0')).strip().lower() in (
        '1', 'true', 'yes', 'on')


def reference_rotation(band_model, anchor_model, centre=FULL_FRAME_CENTRE, n=9, margin=100):
    """Rotation and scale of ``band_model`` relative to ``anchor_model``.

    Both are pixel -> (V2, V3) [arcsec] transforms of the same detector.  A
    similarity ``z_b = k z_a + c`` (``z = v2 + i*v3`` about the anchor's
    centre) is fit on an ``n x n`` pixel grid.  Returns
    ``(rotation_arcsec, scale_ppm, rms_mas)``, where ``rotation_arcsec`` is
    ``arg(k)`` (positive rotates +V2 toward +V3) and ``rms_mas`` the residual
    after the similarity.  The correction is ``-rotation_arcsec``.
    """
    g = np.linspace(margin, 2 * centre[0] - margin, n)
    gx, gy = np.meshgrid(g, g)
    x, y = gx.ravel(), gy.ravel()
    v2a, v3a = anchor_model(x, y)
    v2b, v3b = band_model(x, y)
    c2, c3 = anchor_model(*centre)
    za = (v2a - c2) + 1j * (v3a - c3)
    zb = (v2b - c2) + 1j * (v3b - c3)
    A = np.vstack([za, np.ones_like(za)]).T
    coef, *_ = np.linalg.lstsq(A, zb, rcond=None)
    res = zb - A @ coef
    return (float(np.degrees(np.angle(coef[0])) * 3600.0),
            float((abs(coef[0]) - 1.0) * 1e6),
            float(np.sqrt(np.mean(np.abs(res) ** 2)) * 1e3))


def load_rotation_table(path=TABLE):
    tbl = Table.read(path)
    missing = [c for c in TABLE_COLUMNS if c not in tbl.colnames]
    if missing:
        raise DistortionRotationError(f"{path}: missing column(s) {missing}; expected {TABLE_COLUMNS}")
    return tbl


def _norm(v, default=''):
    s = str(v if v is not None else default).strip().upper()
    return s or default


def _ref_basename(ref):
    return os.path.basename(str(ref or '').replace('crds://', '').strip())


def lookup_correction_arcsec(detector, filtername, pupil, distortion_ref, table=None):
    """``correction_arcsec`` for this frame, or None if nothing is to be done.

    Returns None when the table has no rows for ``(detector, filter, pupil)``.
    Raises ``DistortionRotationError`` when it has rows for them but none for
    this ``distortion_ref``.
    """
    tbl = load_rotation_table() if table is None else table
    det, filt, pup = _norm(detector), _norm(filtername), _norm(pupil, 'CLEAR')
    rows = [r for r in tbl if _norm(r['detector']) == det and _norm(r['filter']) == filt
            and _norm(r['pupil'], 'CLEAR') == pup]
    if not rows:
        return None
    ref = _ref_basename(distortion_ref)
    match = [r for r in rows if _ref_basename(r['distortion_ref']) == ref]
    if not match:
        raise DistortionRotationError(
            f"distortion-rotation table has rows for ({det}, {filt}, {pup}) with "
            f"reference(s) {sorted({str(r['distortion_ref']) for r in rows})}, but this frame "
            f"used {ref!r}.  The stored angle describes one reference file; regenerate the "
            f"table with scripts/analysis/solve_distortion_rotations.py or unset "
            f"DISTORTION_ROTATION_CORRECTION.")
    return float(match[0]['correction_arcsec'])


def rotation_about_pivot_model(v2c, v3c, angle_arcsec):
    """(V2, V3) -> (V2, V3) rotation by ``angle_arcsec`` about ``(v2c, v3c)``."""
    m = ((Shift(-v2c) & Shift(-v3c)) | Rotation2D(angle_arcsec / 3600.0)
         | (Shift(v2c) & Shift(v3c)))
    m.name = 'distortion_rotation_correction'
    return m


def _pivot_pixel(xstart=1, ystart=1):
    # the GWCS detector frame is in subarray pixels (assign_wcs prepends the
    # subarray -> full-frame shift), so the full-frame centre moves with SUBSTRT
    return (FULL_FRAME_CENTRE[0] - (int(xstart or 1) - 1),
            FULL_FRAME_CENTRE[1] - (int(ystart or 1) - 1))


def rotated_wcs(wcs, angle_arcsec, xstart=1, ystart=1):
    """Copy of ``wcs`` with ``detector -> v2v3`` rotated by ``angle_arcsec``
    about the full-frame detector centre.  Returns ``(new_wcs, (v2c, v3c))``."""
    t = wcs.get_transform('detector', 'v2v3')
    v2c, v3c = (float(v) for v in t(*_pivot_pixel(xstart, ystart)))
    new = t | rotation_about_pivot_model(v2c, v3c, angle_arcsec)
    try:
        new.bounding_box = t.bounding_box
    except NotImplementedError:
        pass
    out = copy.deepcopy(wcs)
    out.set_transform('detector', 'v2v3', new)
    return out, (v2c, v3c)


def verify_rotation(wcs_before, wcs_after, pivot, angle_arcsec, shape, tol_mas=0.05):
    """Max deviation (mas) of the V2/V3 change from a pure rotation about
    ``pivot``, on a pixel grid.  Raises above ``tol_mas``."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny:max(ny // 8, 1), 0:nx:max(nx // 8, 1)]
    t0 = wcs_before.get_transform('detector', 'v2v3')
    t1 = wcs_after.get_transform('detector', 'v2v3')
    v20, v30 = t0(xx.ravel(), yy.ravel())
    v21, v31 = t1(xx.ravel(), yy.ravel())
    e2, e3 = rotation_about_pivot_model(pivot[0], pivot[1], angle_arcsec)(v20, v30)
    dev = np.hypot(v21 - e2, v31 - e3) * 1e3
    good = np.isfinite(dev)
    worst = float(np.max(dev[good])) if good.any() else np.inf
    if not worst <= tol_mas:
        raise DistortionRotationError(
            f"distortion-rotation verification failed: V2/V3 change deviates from a "
            f"{angle_arcsec:+.3f}\" rotation about the detector centre by up to {worst:.4f} mas "
            f"(tolerance {tol_mas} mas); NOT writing.")
    return worst


def check_apply_preconditions(hdr, label=""):
    """``'skip'`` if already applied, ``'go'`` if safe, raise otherwise.  Pure."""
    if MARKER in hdr:
        return 'skip'
    if hdr.get(PENDING):
        raise DistortionRotationError(
            f"{label}: pending distortion-rotation marker without completion marker -- a "
            f"previous apply crashed mid-write.  Re-create this frame from its _cal; refusing "
            f"to guess the GWCS state.")
    if 'RAOFFSET' in hdr:
        raise DistortionRotationError(
            f"{label}: frame already carries a reference shift (RAOFFSET={hdr['RAOFFSET']}) "
            f"but no {MARKER}; the rotation must be applied before the shift.  Re-create this "
            f"frame from its _cal and re-run fix_alignment with "
            f"DISTORTION_ROTATION_CORRECTION=1.")
    return 'go'


def apply_distortion_rotation_correction(fn, table=None, verbose=True):
    """Apply the rotation for ``fn``'s (detector, filter, pupil, R_DISTOR) in
    place, updating the ASDF GWCS and the FITS SCI-header WCS, idempotently.
    Returns the applied ``correction_arcsec``, or None when skipped."""
    hdr0 = fits.getheader(fn, ext=0)
    hdr = fits.getheader(fn, ext=('SCI', 1))
    if check_apply_preconditions(hdr, label=fn) == 'skip':
        if verbose:
            print(f"distortion-rotation correction skipped for {fn}: already applied "
                  f"({hdr.get('DROTARC')} arcsec)")
        return None
    det = hdr0.get('DETECTOR', hdr.get('DETECTOR'))
    filt = hdr0.get('FILTER', hdr.get('FILTER'))
    pupil = hdr0.get('PUPIL', hdr.get('PUPIL'))
    rdist = hdr0.get('R_DISTOR', hdr.get('R_DISTOR'))
    if not rdist:
        raise DistortionRotationError(f"{fn}: no R_DISTOR keyword; cannot identify the "
                                      f"distortion reference the rotation applies to")
    corr = lookup_correction_arcsec(det, filt, pupil, rdist, table=table)
    if corr is None:
        if verbose:
            print(f"distortion-rotation correction not needed for {fn}: no row for "
                  f"({det}, {filt}, {pupil})")
        return None

    with fits.open(fn, mode='update') as hdul:
        hdul['SCI'].header[PENDING] = (True, 'distortion-rotation apply in progress')

    from jwst.datamodels import ImageModel
    fa = ImageModel(fn)
    wcsobj = fa.meta.wcs
    ww, pivot = rotated_wcs(wcsobj, corr, xstart=fa.meta.subarray.xstart,
                            ystart=fa.meta.subarray.ystart)
    worst = verify_rotation(wcsobj, ww, pivot, corr, fa.data.shape)
    fa.meta.oldwcs = copy.copy(wcsobj)
    fa.meta.wcs = ww
    fa.save(fn, overwrite=True)

    with fits.open(fn) as hdul:
        h = hdul['SCI'].header
        _sip_max, _sip_med = sync_header_to_gwcs(h, ww, fa.data.shape, label=os.path.basename(fn))
        h['SIPGWMAX'] = (_sip_max, '[mas] max FITS/SIP vs GWCS disagreement')
        h[MARKER] = (True, 'distortion-ref rotation applied (#1135)')
        h['DROTARC'] = (corr, '[arcsec] V2V3 rotation, +V2 toward +V3')
        h['DROTREF'] = (_ref_basename(rdist), 'distortion ref of DROTARC')
        h['DROTPV2'] = (pivot[0], '[arcsec] rotation pivot V2 (detector centre)')
        h['DROTPV3'] = (pivot[1], '[arcsec] rotation pivot V3 (detector centre)')
        h['DROTVMAS'] = (worst, '[mas] max deviation from pure rotation')
        h[PENDING] = (False, 'distortion-rotation apply completed')
        hdul.writeto(fn, overwrite=True)
    if verbose:
        print(f"distortion-rotation correction applied to {fn} ({det} {filt}/{pupil}): "
              f"{corr:+.3f}\" about V2V3 ({pivot[0]:.3f}, {pivot[1]:.3f}); verified to "
              f"{worst:.4f} mas")
    return corr
