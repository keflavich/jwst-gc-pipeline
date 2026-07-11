"""Inter-detector differential velocity aberration (DVA) correction.

WHY THIS EXISTS
---------------
Velocity aberration: the spacecraft's velocity (~30 km/s, dominated by
Earth's orbit) displaces every apparent star position toward the velocity
apex by up to ~20.5''.  The bulk of that displacement is absorbed by the
attitude solution (FGS guides on the *apparent* star field), so what matters
for the WCS is only the *differential* part across the field of view: to
first order a plate-scale change by the header factor ``VA_SCALE``
(:math:`|1 - \\mathrm{VA\\_SCALE}| \\approx 10^{-4}`).

``assign_wcs`` corrects DVA per detector by scaling V2V3 about *each
detector's own* reference point (``pointing.dva_corr_model(va_scale, v2_ref,
v3_ref)``).  That removes the differential aberration WITHIN each detector
but leaves the INTER-detector component -- the aberration of the detector
*separations* -- in the delivered WCS.  The resulting internal inconsistency
is a pure per-detector rigid shift of

.. math::  \\Delta = -(1 - \\mathrm{VA\\_SCALE}) \\times (\\mathrm{ref}_d - C)

for any field point :math:`C` common to the exposure (the choice of
:math:`C` only moves a common rigid shift, which the downstream reference
tie absorbs).  At NIRCam module lever arms this is +-9-13 mas -- the
dominant term of the apparent "module A/B offset" -- and it is
EPOCH-DEPENDENT (VA_SCALE swings by ~+-1e-4 over the year, so uncorrected
module separations move by up to ~25 mas between epochs).

Measured empirically by network self-calibration on the Brick (2026-07-11):
the fitted inter-detector scale tracks each program's own ``1 - VA_SCALE``
(2221: 9.7-9.9e-5 vs 9.18e-5 predicted; 1182: 1.05-1.06e-4 vs 1.00e-4), and
after removing it the static SIAF detector placements are good to 1-2.5 mas.
See the astrometry paper section ``siaf_accuracy.tex`` and
``ASTROMETRY_WCS_CORRECTION_FLOW.md``.

WHAT THIS MODULE DOES
---------------------
Applies the missing inter-detector term as a per-detector rigid WCS shift
(exactly constant across a detector, because the intra-detector part is
already handled by ``assign_wcs``).  The common point :math:`C` is the V1
(telescope boresight) axis position, read from the same header
(``RA_V1``/``DEC_V1``) -- common to all detectors of an exposure by
construction, so no sibling-file gathering is needed.

The shift is applied with the same GWCS + FITS-header mechanism as
``fix_alignment`` and is idempotent (``DVACORR`` marker keyword).

Opt-in: nothing calls this by default.  Enable in the reduction with
``APPLY_DVA_CORRECTION=1`` (hook in ``fix_alignment``) or call
:func:`apply_dva_correction` directly.  Apply BEFORE measuring offsets
tables so the tie sees DVA-consistent frames; applying after an existing
tie only re-introduces a small common shift, which the next tie absorbs.
"""
import copy

import numpy as np
from astropy import units as u
from astropy.io import fits

__all__ = ['interdetector_dva_shift_deg', 'apply_dva_correction',
           'dva_shift_needed']

# Header keyword written after application (idempotency marker).
DVA_MARKER = 'DVACORR'


def interdetector_dva_shift_deg(va_scale, ra_ref, dec_ref, ra_v1, dec_v1):
    """Rigid shift (dRA, dDec) in COORDINATE degrees that restores
    inter-detector DVA consistency for one detector.

    Parameters
    ----------
    va_scale : float or None
        Header ``VA_SCALE`` (apparent/true plate-scale ratio, ~0.9999).
        ``None`` or 1 -> no shift.
    ra_ref, dec_ref : float
        The detector aperture reference sky position (header ``RA_REF``,
        ``DEC_REF``), degrees.
    ra_v1, dec_v1 : float
        The telescope boresight position (header ``RA_V1``, ``DEC_V1``),
        degrees -- the common field point C shared by all detectors of the
        exposure.

    Returns
    -------
    (dra_deg, ddec_deg) : tuple of float
        COORDINATE offsets (RA offset is a coordinate delta, not on-sky;
        the cos(dec) factors of the separation and of the shift cancel to
        first order, so no cos term appears).

    Notes
    -----
    Sign: ``assign_wcs`` leaves computed positions displaced OUTWARD from
    the common point by ``(1 - va_scale) * (ref_d - C)``; the correction is
    the negative of that.  Empirical check: after applying this shift the
    network-self-cal inter-detector scale term fits to ~0 (was ~+1e-4).
    """
    if va_scale is None or va_scale == 1:
        return 0.0, 0.0
    if not np.isfinite(va_scale) or va_scale <= 0:
        raise ValueError(f"invalid VA_SCALE {va_scale}")
    s = 1.0 - float(va_scale)
    return (-s * (float(ra_ref) - float(ra_v1)),
            -s * (float(dec_ref) - float(dec_v1)))


def dva_shift_needed(header):
    """True if the file has the needed keywords and no DVA marker yet.

    ``header`` is the SCI-extension header (where jwst keeps WCS/pointing
    keywords).  Missing VA_SCALE / RA_REF / RA_V1 -> False (nothing to do;
    e.g. non-jwst or synthetic data).
    """
    if DVA_MARKER in header:
        return False
    for k in ('VA_SCALE', 'RA_REF', 'DEC_REF', 'RA_V1', 'DEC_V1'):
        if k not in header:
            return False
    return header['VA_SCALE'] not in (None, 1)


def apply_dva_correction(fn, verbose=True):
    """Apply the inter-detector DVA shift to ``fn`` (a jwst cal-like file),
    updating BOTH the ASDF GWCS and the FITS SCI-header WCS, idempotently.

    Returns the (dra_deg, ddec_deg) applied, or None if skipped
    (already applied / keywords absent / VA_SCALE trivial).
    """
    hdr = fits.getheader(fn, ext=('SCI', 1))
    if not dva_shift_needed(hdr):
        if verbose:
            why = 'already applied' if DVA_MARKER in hdr else 'keywords absent/trivial'
            print(f"DVA correction skipped for {fn}: {why}")
        return None
    dra, ddec = interdetector_dva_shift_deg(
        hdr['VA_SCALE'], hdr['RA_REF'], hdr['DEC_REF'], hdr['RA_V1'], hdr['DEC_V1'])

    # GWCS (ASDF) side -- same mechanism as fix_alignment.
    from jwst.datamodels import ImageModel
    from jwst.tweakreg.utils import adjust_wcs
    fa = ImageModel(fn)
    wcsobj = fa.meta.wcs
    fa.meta.oldwcs = copy.copy(wcsobj)
    ww = adjust_wcs(wcsobj, delta_ra=dra * u.deg, delta_dec=ddec * u.deg)
    fa.meta.wcs = ww
    fa.save(fn, overwrite=True)

    # FITS-header side (SIP WCS read by the cataloging code).
    with fits.open(fn) as hdul:
        h = hdul['SCI'].header
        h['OLCRVAL1'] = h['CRVAL1']
        h['OLCRVAL2'] = h['CRVAL2']
        h.update(ww.to_fits()[0])
        h[DVA_MARKER] = (True, 'inter-detector DVA shift applied')
        h['DVASHRA'] = (dra, '[deg] DVA inter-detector RA coord shift')
        h['DVASHDE'] = (ddec, '[deg] DVA inter-detector Dec shift')
        hdul.writeto(fn, overwrite=True)
    if verbose:
        print(f"DVA correction applied to {fn}: "
              f"({dra * 3.6e6:+.2f}, {ddec * 3.6e6:+.2f}) mas "
              f"(VA_SCALE={hdr['VA_SCALE']})")
    return dra, ddec
