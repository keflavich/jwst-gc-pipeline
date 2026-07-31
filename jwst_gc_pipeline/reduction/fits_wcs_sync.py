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
overlap gate).  It is a different low-order surface per detector *and per
filter*, so no bulk tie removes it and it injects spurious cross-filter and
cross-detector structure exactly where the astrometric gates look.

**The whole 5-8 mas comes from ``max_pix_error``** -- from the degree-3 refit
itself.  The orphan coefficients described below contribute 0.000 mas.

Two further defects the bare ``header.update()`` had:

1. ``Header.update`` **merges**.  Overwriting a degree-4 header with a degree-3
   fit leaves orphan ``A_0_4``/``A_1_3``/``A_2_2``/``A_3_1``/``A_4_0`` cards
   behind, contradicting the ``A_ORDER=3`` written beside them.  Observed on
   NEARLY every ``_destreak.fits`` in the archive (55 of 60 sampled at random
   from the 7,012 on disk; 5 carry no orphans).

   **These orphans are inert in astropy and contribute 0.000 mas** -- astropy
   allocates the SIP coefficient matrix from ``A_ORDER`` and never reads terms
   above it (measured: deleting all ten orphan cards changes positions by
   0.000000 mas over an 8x8 grid on the full array).  They are stripped anyway,
   because a header that contradicts itself is a trap for any reader that
   infers the order from the cards present rather than from ``A_ORDER``, and
   for non-astropy consumers.

2. Nothing ever *checked* that the written FITS WCS reproduced the GWCS.
   ``check_wcs`` compared only the array **centre**, where the distortion
   residual is identically zero by construction (measured on brick F182M nrca1:
   centre 0.0000 mas, (0,0) 5.117 mas, (2047,2047) 5.289 mas).  A 25x loosening
   of the REQUESTED bound (0.25 px vs STScI's 0.01) survived because the check sat
   exactly where the error vanishes: a gate blind to the failure it is named for.
   Measured, that loosening takes brick SW nrca1 from 0.000 to 5.487 mas.

So: fit tight, strip stale coefficients, and **verify against the GWCS before
returning**.

Rule of thumb
-------------
Read the GWCS whenever it exists -- ``model.meta.wcs`` via
``stdatamodels.jwst.datamodels``, or the ``jwst_gc_pipeline.frame_wcs.frame_wcs``
helper.  Use this module only at the few points where a FITS header genuinely
must be written for an external consumer.
"""
import os
import re
import warnings

import numpy as np
from astropy.io import fits

__all__ = ['SIP_MAX_PIX_ERROR', 'FITS_GWCS_TOL_MAS', 'FitsGwcsMismatchError',
           'SipAccuracyWarning',
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

#: TIGHTENING retry ladder, tried in order when the MEASURED FITS-vs-GWCS
#: disagreement exceeds :data:`FITS_GWCS_TOL_MAS`.  A tighter request raises the
#: SIP degree, which genuinely improves the fit -- measured on the reference
#: fixture: 0.01 px -> degree 2 -> 0.107 mas; 0.001 -> degree 3 -> 0.019 mas;
#: 1e-4 -> degree 5 -> 0.000 mas.
#:
#: There is deliberately no LOOSENING ladder.  The previous version had one, on
#: the belief that ``to_fits_sip`` raises ``ValueError`` when the requested
#: accuracy is unreachable.  gwcs 1.0.3 does not: ``fit_2D_poly`` issues a
#: ``scipy.linalg.LinAlgWarning`` and returns a header regardless, so that ladder
#: never advanced, its "fell back to N px" warning was unreachable, and the
#: ``except ValueError`` guarding it was dead code.
#:
#: Do not reason from a rung to an error budget either: ``max_pix_error`` bounds
#: a fit whose DEGREE is discrete, so achieved error is usually far better than
#: requested and the relationship is not monotonic.  Measured (requested px ->
#: achieved mas) with the AS-WRITTEN headers:
#:
#:   brick SW nrca1    0.01->0.000  0.02->0.594  0.05->0.594  0.1->0.594  0.25->5.487
#:   brick LW nrcalong 0.01->0.000  0.02->0.000  0.05->0.000  0.1->4.394  0.25->6.586
#:   sickle MIRI       0.01->0.000  0.02->0.000  0.05->0.000  0.1->8.392  0.25->8.392
#:
#: In practice no real frame has needed the retry: 0.01 px converges on the first
#: try for NIRCam SW/LW and for MIRI -- including FULL-FRAME 1024x1032 MIRI
#: (cloudc F2550W/F770W, w51 F560W/F2100W, sgrb2 F1280W), not only the 512x512
#: sickle subarrays, all degree 4 at <=7e-5 mas.
_TIGHTENING_PIX_ERRORS = (1e-3, 1e-4, 1e-5)

_SIP_KEY_RE = re.compile(r'^(A|B|AP|BP)_(\d+_\d+|ORDER)$')

#: The two mutually-exclusive FITS linear-WCS representations.  A header that
#: ends up with both is invalid, and astropy resolves it silently ("cdelt will
#: be ignored since cd is present") -- so whichever set the new fit does not
#: write must be deleted, not left behind.
_LINEAR_WCS_KEYS = ('CD1_1', 'CD1_2', 'CD2_1', 'CD2_2',
                    'PC1_1', 'PC1_2', 'PC2_1', 'PC2_2',
                    'CDELT1', 'CDELT2')


class SipAccuracyWarning(UserWarning):
    """gwcs could not reach the requested SIP accuracy, or a tighter retry was needed."""


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

    A single fit, requesting ``max_pix_error`` (default
    :data:`SIP_MAX_PIX_ERROR` = 0.01 px) instead of gwcs's 0.25 px default.

    **This function does not judge its own output.**  ``max_pix_error`` is a
    *request*, and gwcs 1.0.3 does not enforce it: when the accuracy is
    unreachable ``fit_2D_poly`` issues a ``scipy.linalg.LinAlgWarning``
    ("Failed to achieve requested SIP approximation accuracy") and **returns a
    header anyway** -- it does not raise.  Its own ``SIPMXERR`` card is not a
    substitute either: at 5000x the reference distortion it reported 5.5e-5 px
    while simultaneously emitting that warning.

    So the only trustworthy check is to measure the written header against the
    GWCS, which is what :func:`sync_header_to_gwcs` does -- use that unless you
    genuinely want an unvalidated fit.

    Returns an ``astropy.io.fits.Header``.
    """
    requested = SIP_MAX_PIX_ERROR if max_pix_error is None else float(max_pix_error)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        hdr = gwcs_obj.to_fits_sip(max_pix_error=requested,
                                   max_inv_pix_error=requested,
                                   npoints=npoints, bounding_box=bounding_box,
                                   verbose=verbose)
    for w in caught:
        if 'SIP approximation accuracy' in str(w.message):
            warnings.warn(
                f"gwcs could not reach the requested {requested} px SIP accuracy "
                f"({w.message}). The header may be a coarser approximation than "
                f"intended; sync_header_to_gwcs will measure it and refuse it if "
                f"so. Read the GWCS (model.meta.wcs) for astrometry on this "
                f"frame.", SipAccuracyWarning)
    return fits.Header(hdr)


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

    Strips stale SIP coefficients, fits, then **measures** the residual against
    the GWCS over ``shape``.  If the measurement exceeds ``tol_mas`` (default
    :data:`FITS_GWCS_TOL_MAS`) the fit is retried at successively TIGHTER
    ``max_pix_error`` -- which raises the SIP degree and does genuinely help
    (measured on the reference fixture: 0.01 px -> degree 2 -> 0.107 mas;
    0.001 px -> degree 3 -> 0.019 mas; 1e-4 px -> degree 5 -> 0.000 mas).
    :class:`FitsGwcsMismatchError` is raised only if no rung gets inside the
    tolerance.

    Note the direction: tightening is the retry that can help.  There is no
    point loosening, and gwcs's own accuracy signal cannot be trusted to decide
    (see :func:`sip_header_from_gwcs`), so the loop is driven entirely by the
    measured error.

    Returns ``(max_mas, median_mas)`` of the achieved agreement.
    """
    tol = FITS_GWCS_TOL_MAS if tol_mas is None else float(tol_mas)
    requested = SIP_MAX_PIX_ERROR if max_pix_error is None else float(max_pix_error)
    ladder = [requested] + [e for e in _TIGHTENING_PIX_ERRORS if e < requested]

    best = None
    for attempt, err in enumerate(ladder):
        sip = sip_header_from_gwcs(gwcs_obj, max_pix_error=err, npoints=npoints)
        candidate = header.copy()
        strip_sip_keywords(candidate)
        # Drop whichever linear representation the new fit does NOT write, so the
        # header never carries CD *and* PC/CDELT at once (astropy then silently
        # ignores CDELT, giving a header that reads differently from what was
        # intended).  Same defect class as #181, one level up.
        for key in _LINEAR_WCS_KEYS:
            if key in candidate and key not in sip:
                del candidate[key]
        candidate.update(sip)
        max_mas, med_mas = fits_gwcs_discrepancy_mas(candidate, gwcs_obj, shape)
        if best is None or max_mas < best[0]:
            best = (max_mas, med_mas, candidate, err)
        if max_mas <= tol:
            if attempt:
                warnings.warn(
                    f"SIP fit{' for ' + label if label else ''} needed a tighter "
                    f"request than {requested} px: {err} px achieved "
                    f"{max_mas:.4f} mas.", SipAccuracyWarning)
            _replace_header_wcs(header, candidate)
            return max_mas, med_mas

    max_mas, med_mas, candidate, err = best
    raise FitsGwcsMismatchError(
        f"FITS/SIP header{' for ' + label if label else ''} disagrees with "
        f"the GWCS by up to {max_mas:.3f} mas (median {med_mas:.3f}), over "
        f"the {tol:.3f} mas tolerance, at every requested accuracy in "
        f"{ladder} px (best was {err} px). The SIP approximation is not good "
        f"enough for this frame; astrometry must read the GWCS.")


def _replace_header_wcs(header, candidate):
    """Adopt ``candidate``'s WCS cards into ``header`` in place."""
    strip_sip_keywords(header)
    for key in _LINEAR_WCS_KEYS:
        if key in header and key not in candidate:
            del header[key]
    header.update(candidate)
    return header
