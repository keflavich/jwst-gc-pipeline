"""Write a FITS (SIP) representation of a GWCS at *verified* fidelity.

Background
----------
The authoritative WCS of every JWST detector-frame product is the **GWCS** in
the ASDF extension: a full distortion model (SIAF polynomial + velocity
aberration + tangent-plane projection) that FITS keywords cannot express
exactly.  A FITS ``RA---TAN-SIP`` header is only ever a *fitted low-order
approximation* of it, kept in the SCI header so that plain-``astropy.wcs``
consumers (DS9/CARTA, ``reproject``, legacy code paths) can read something.

The failure this module exists to prevent
-----------------------------------------
``gwcs.WCS.to_fits()`` / ``to_fits_sip()`` default to ``max_pix_error=0.25``
**pixels**.  Every place the pipeline re-stamped a corrected GWCS into the SCI
header with a bare ``header.update(ww.to_fits()[0])`` therefore replaced the
delivered STScI SIP fit (``jwst.assign_wcs.util.update_fits_wcsinfo``, which
uses ``max_pix_error=0.01``) with one an order of magnitude *worse*.

Measured on real products (2026-07-29):

===========================================  =======  ===========  ============
product                                      A_ORDER  median err   max err
===========================================  =======  ===========  ============
brick F182M nrca1 ``_cal`` (MAST, 0.01 px)    4        0.17 mas     0.49 mas
brick F182M nrca1 ``_destreak``/``_crf``      3        0.83 mas     **5.5 mas**
brick F410M nrcalong ``_crf``                 3        0.63 mas     **6.6 mas**
sickle F770W ``_align``/``_crf`` (MIRI)       3        1.18 mas     **8.0 mas**
same GWCS refit at ``max_pix_error=0.01``     4-5      0.000 mas    0.000 mas
===========================================  =======  ===========  ============

A 5-8 mas *position-dependent* error in the WCS that per-frame catalogs are
built from sits right on top of the pipeline's own astrometric tolerances (2 mas
m2 per-exposure consensus, 5 mas m7 cross-filter gate, 30 mas inter-frame
overlap gate).  It is not a constant offset that a bulk tie removes: it is a
different low-order surface per detector *and per filter*, so it injects
spurious cross-filter and cross-detector structure exactly where the
astrometric gates look.

**The whole 5-8 mas is attributable to ``max_pix_error``** -- to the degree-3
refit itself, and to nothing else.  See the note on orphan coefficients below
before assuming otherwise.

Two further defects the bare ``header.update()`` had:

1. ``Header.update`` **merges**.  Overwriting a degree-4 header with a degree-3
   fit leaves orphan ``A_0_4``/``A_1_3``/``A_2_2``/``A_3_1``/``A_4_0`` cards
   behind, contradicting the ``A_ORDER=3`` written beside them.  Observed on
   every ``_destreak.fits`` in the archive.

   **These orphans are inert in astropy and contribute 0.000 mas** -- astropy
   allocates the SIP coefficient matrix from ``A_ORDER`` and never reads terms
   above it (measured: deleting all ten orphan cards changes positions by
   0.000000 mas over an 8x8 grid on the full array).  They are stripped anyway,
   because a header that contradicts itself is a trap for any reader that
   infers the order from the cards present rather than from ``A_ORDER``, and
   for non-astropy consumers -- but they are *not* part of the measured error.

2. Nothing ever *checked* that the written FITS WCS reproduced the GWCS.
   ``check_wcs`` compared only the array **centre**, where the distortion
   residual is identically zero by construction.  A 25x accuracy regression
   survived because the check was placed where the error vanishes.

So: fit tight, strip stale coefficients, and **verify against the GWCS before
returning**.

Rule of thumb
-------------
Read the GWCS (``jwst_gc_pipeline.frame_wcs.frame_wcs``) whenever it exists.
Use this module only at the few points where a FITS header genuinely must be
written for an external consumer.
"""
import os
import re
import warnings

import numpy as np
from astropy.io import fits

__all__ = ['SIP_MAX_PIX_ERROR', 'FITS_GWCS_TOL_MAS', 'FitsGwcsMismatchError',
           'sip_header_from_gwcs', 'strip_sip_keywords',
           'fits_gwcs_discrepancy_mas', 'sync_header_to_gwcs']

#: Requested SIP fit accuracy, in *pixels*.  STScI's ``update_fits_wcsinfo``
#: uses 0.01; gwcs's own default (0.25) is what this module exists to avoid.
#: 0.01 px = 0.31 mas (NIRCam SW), 0.63 mas (LW), 1.1 mas (MIRI) worst case --
#: in practice the degree that meets it overshoots and lands at ~0.
SIP_MAX_PIX_ERROR = float(os.environ.get('SIP_MAX_PIX_ERROR', 0.01))

#: Hard ceiling on the measured FITS-vs-GWCS disagreement, in mas.  Below the
#: tightest astrometric tolerance in the pipeline (the 2 mas m2 per-exposure
#: consensus) by a comfortable margin.
FITS_GWCS_TOL_MAS = float(os.environ.get('FITS_GWCS_TOL_MAS', 0.5))

#: Fallback ladder if the tight fit cannot be achieved (degree caps out at 9).
#:
#: The ladder cannot silently degrade a header, because
#: :func:`sync_header_to_gwcs` gates on the **measured** disagreement, not on
#: the requested ``max_pix_error``.  The warning is a diagnostic; the
#: measurement is the guard.
#:
#: Do not reason from the rung to an error budget -- ``max_pix_error`` is an
#: upper bound on a fit whose DEGREE is discrete, so the achieved error is
#: usually far better than requested and the relationship is not monotonic in
#: any useful way.  Measured (requested px -> achieved mas):
#:
#:   brick SW nrca1    0.01->0.000  0.02->0.594  0.05->0.594  0.1->0.594  0.25->5.487
#:   brick LW nrcalong 0.01->0.000  0.02->0.000  0.05->0.000  0.1->4.394  0.25->6.586
#:   sickle MIRI       0.01->0.000  0.02->0.000  0.05->0.000  0.1->8.392  0.25->8.392
#:
#: In practice no real frame has needed the ladder at all: 0.01 px converges on
#: the first try for NIRCam SW/LW and for MIRI (8 MIRI frames across
#: sickle/brick/cloudc/w51, all degree 4, all achieving <1e-4 mas) -- so MIRI,
#: despite auditing worst at 8.39 mas *as previously written*, is not a case
#: that exercises the ladder.
_FALLBACK_PIX_ERRORS = (0.01, 0.02, 0.05, 0.1)

_SIP_KEY_RE = re.compile(r'^(A|B|AP|BP)_(\d+_\d+|ORDER)$')

#: The two mutually-exclusive FITS linear-WCS representations.  A header that
#: ends up with both is invalid, and astropy resolves it silently ("cdelt will
#: be ignored since cd is present") -- so whichever set the new fit does not
#: write must be deleted, not left behind.
_LINEAR_WCS_KEYS = ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                    'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
                    'CDELT1', 'CDELT2')


class FitsGwcsMismatchError(RuntimeError):
    """The FITS/SIP header does not reproduce the GWCS to the required tolerance."""


def strip_sip_keywords(header):
    """Delete every SIP coefficient card from ``header`` (in place).

    ``Header.update()`` merges, so writing a lower-degree SIP fit over a
    higher-degree one leaves orphan high-order coefficients that disagree with
    the new ``A_ORDER``.  Always strip before updating.
    """
    for key in [k for k in header if _SIP_KEY_RE.match(str(k))]:
        del header[key]
    return header


def sip_header_from_gwcs(gwcs_obj, *, max_pix_error=None, npoints=32,
                         bounding_box=None, verbose=False):
    """FITS ``RA---TAN-SIP`` header approximating ``gwcs_obj``.

    Unlike ``gwcs_obj.to_fits()[0]`` this requests ``max_pix_error``
    (default :data:`SIP_MAX_PIX_ERROR` = 0.01 px) rather than gwcs's 0.25 px
    default, and walks a fallback ladder if that accuracy is unreachable
    instead of failing outright.

    Returns an ``astropy.io.fits.Header``.
    """
    requested = SIP_MAX_PIX_ERROR if max_pix_error is None else float(max_pix_error)
    ladder = [requested] + [e for e in _FALLBACK_PIX_ERRORS if e > requested]
    last_exc = None
    for err in ladder:
        try:
            hdr = gwcs_obj.to_fits_sip(max_pix_error=err, max_inv_pix_error=err,
                                       npoints=npoints, bounding_box=bounding_box,
                                       verbose=verbose)
        except ValueError as ex:
            # gwcs raises ValueError when the requested accuracy is not
            # reachable within its maximum SIP degree (9).
            last_exc = ex
            continue
        if err > requested:
            warnings.warn(
                f"SIP fit could not reach the requested {requested} px accuracy; "
                f"fell back to {err} px. The FITS header is a coarser "
                f"approximation of the GWCS than intended -- read the GWCS "
                f"(frame_wcs.open_frame_wcs) for astrometry on this frame.")
        return fits.Header(hdr)
    raise FitsGwcsMismatchError(
        f"could not fit a SIP approximation to the GWCS at any of {ladder} px "
        f"accuracy") from last_exc


def fits_gwcs_discrepancy_mas(header, gwcs_obj, shape, npoints=25):
    """Max on-sky separation (mas) between the header WCS and ``gwcs_obj``.

    Samples an ``npoints x npoints`` grid over ``shape = (ny, nx)``.  Returns
    ``(max_mas, median_mas)``; non-finite samples (outside the GWCS bounding
    box) are ignored.
    """
    from astropy import wcs as awcs

    ny, nx = int(shape[0]), int(shape[1])
    yy, xx = np.meshgrid(np.linspace(0, ny - 1, npoints),
                         np.linspace(0, nx - 1, npoints), indexing='ij')
    xf, yf = xx.ravel().astype(float), yy.ravel().astype(float)

    ra_g, dec_g = gwcs_obj(xf, yf)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        # relax=True so the '-SIP' CTYPE suffix and the A_*/B_* terms are
        # honoured; a plain WCS() on a header whose CTYPE lost the suffix
        # silently drops the distortion.
        ra_f, dec_f = awcs.WCS(header, relax=True).all_pix2world(xf, yf, 0)

    ok = np.isfinite(ra_g) & np.isfinite(dec_g) & np.isfinite(ra_f) & np.isfinite(dec_f)
    if not ok.any():
        raise FitsGwcsMismatchError(
            "no finite samples when comparing the FITS header WCS to the GWCS")
    cosd = np.cos(np.radians(dec_g[ok]))
    sep = np.hypot((ra_g[ok] - ra_f[ok]) * cosd, dec_g[ok] - dec_f[ok]) * 3.6e6
    return float(np.max(sep)), float(np.median(sep))


def sync_header_to_gwcs(header, gwcs_obj, shape, *, max_pix_error=None,
                        tol_mas=None, npoints=32, label=''):
    """Replace the WCS in ``header`` with a verified SIP fit of ``gwcs_obj``.

    Strips stale SIP coefficients, writes a tight fit, then *measures* the
    residual against the GWCS over ``shape`` and raises
    :class:`FitsGwcsMismatchError` if it exceeds ``tol_mas``
    (default :data:`FITS_GWCS_TOL_MAS`).

    Returns ``(max_mas, median_mas)`` of the achieved agreement.
    """
    tol = FITS_GWCS_TOL_MAS if tol_mas is None else float(tol_mas)
    sip = sip_header_from_gwcs(gwcs_obj, max_pix_error=max_pix_error,
                               npoints=npoints)
    strip_sip_keywords(header)
    # Drop whichever linear representation the new fit does NOT write, so the
    # header never carries CD *and* PC/CDELT at once (astropy then silently
    # ignores CDELT, giving a header that reads differently from what was
    # intended).  Same defect class as #181, one level up.
    for key in _LINEAR_WCS_KEYS:
        if key in header and key not in sip:
            del header[key]
    header.update(sip)
    max_mas, med_mas = fits_gwcs_discrepancy_mas(header, gwcs_obj, shape)
    if max_mas > tol:
        raise FitsGwcsMismatchError(
            f"FITS/SIP header{' for ' + label if label else ''} disagrees with "
            f"the GWCS by up to {max_mas:.3f} mas (median {med_mas:.3f}), over "
            f"the {tol:.3f} mas tolerance. The SIP approximation is not good "
            f"enough for this frame; astrometry must read the GWCS.")
    return max_mas, med_mas
