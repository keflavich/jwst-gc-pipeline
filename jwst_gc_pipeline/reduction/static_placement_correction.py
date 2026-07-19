"""Static per-detector SIAF placement-field correction (opt-in, default OFF).

The network self-calibration of the Brick/1182 exposure sets (astrometry
paper, siaf_accuracy section; analysis/decompose_selfcal.py) measured, after
removing per-exposure attitude and the deterministic inter-detector DVA scale,
a STATIC per-detector placement error of 1-2.5 mas that repeats across five
independent solves (three 2221 SW filters, two 1182 SW filters; different
epochs, rolls, guide stars, proposals) with rms-about-the-mean of 0.2-0.9 mas
per detector.  This module subtracts that measured mean placement field.

Scope and caveats (why this is OPT-IN, unlike the DVA correction):
- The field was measured in SKY coordinates on Galactic-center pointings; at
  strongly different position angles the sky-frame values rotate out of
  validity.  A proper V2/V3-frame version requires the attitude matrix; until
  then this correction is only appropriate for GC-survey-like orientations.
- Only the eight SW detectors were solved; LW detectors get no correction.
- Amplitude is 1-2.5 mas -- below the tie scatter for most uses.

Enable with STATIC_PLACEMENT_CORRECTION=1 in the environment.  Applied (like
the DVA correction) BEFORE the reference tie so the tie absorbs any
common-mode part.  Idempotent via the SPLACORR marker, with a SPLAPEND
pending flag making a mid-write crash fail loud instead of double-shifting.

WARNING: enabling this on an already-processed field invalidates any
consensus/VIRAC2locked offsets table solved against the uncorrected frame
(the corrections would stack); rebuild the offsets tables after enabling.
"""
import copy

import numpy as np
import astropy.units as u
from astropy.io import fits

MARKER = 'SPLACORR'
PENDING = 'SPLAPEND'

# Measured mean placement error per detector, SKY frame, mas (dRA*, dDec):
# the "mean" column of the siaf_accuracy placement table (five-solve average).
# The correction applied is the NEGATIVE of these values.
PLACEMENT_MAS = {
    'NRCA1': (+0.1, +1.3),
    'NRCA2': (+0.7, -1.6),
    'NRCA3': (-0.4, +2.4),
    'NRCA4': (+1.0, +0.1),
    'NRCB1': (+0.9, -0.7),
    'NRCB2': (-1.9, +0.7),
    'NRCB3': (+0.7, -1.6),
    'NRCB4': (-1.1, -0.8),
}


def placement_shift_deg(detector, dec_deg):
    """Correction (dra_deg, ddec_deg) for ``detector``, or None if the
    detector was not solved (LW, or unknown).  ``dec_deg`` converts the
    RA* (on-sky) offset to a coordinate offset."""
    key = str(detector).upper().strip()
    if key not in PLACEMENT_MAS:
        return None
    dra_star, ddec = PLACEMENT_MAS[key]
    cosd = np.cos(np.radians(float(dec_deg)))
    return (-dra_star / 3.6e6 / cosd, -ddec / 3.6e6)


def placement_shift_needed(header):
    """True if ``header`` (SCI ext) names a solved detector and carries no
    marker yet."""
    if MARKER in header:
        return False
    det = header.get('DETECTOR')
    return det is not None and str(det).upper().strip() in PLACEMENT_MAS


def apply_placement_correction(fn, verbose=True):
    """Apply the static placement correction to ``fn`` (jwst cal-like file),
    updating both the ASDF GWCS and the FITS SCI-header WCS, idempotently.
    Returns (dra_deg, ddec_deg) applied, or None if skipped."""
    hdr0 = fits.getheader(fn, ext=0)
    hdr = fits.getheader(fn, ext=('SCI', 1))
    det = hdr.get('DETECTOR', hdr0.get('DETECTOR'))
    if MARKER in hdr:
        if verbose:
            print(f"placement correction skipped for {fn}: already applied")
        return None
    shift = placement_shift_deg(det, hdr.get('CRVAL2', 0.0))
    if shift is None:
        if verbose:
            print(f"placement correction skipped for {fn}: detector {det} "
                  f"not in the solved set")
        return None
    if hdr.get(PENDING) and MARKER not in hdr:
        # A previous run died between the GWCS write and the marker write:
        # the GWCS shift state is ambiguous, and guessing risks the silent
        # 1-2.5 mas double-correction this guard exists to prevent.
        raise RuntimeError(
            f"{fn}: pending placement-correction marker without completion "
            f"marker -- a previous apply crashed mid-write. Re-fetch or "
            f"re-reduce this file; refusing to guess the GWCS state.")
    dra, ddec = shift

    # Crash-safety protocol (review #129 item 1): set a PENDING flag in a
    # cheap in-place header update BEFORE the GWCS rewrite; clear it in the
    # same write that sets the completion marker.  A crash between the two
    # writes then fails LOUD on re-run (above) instead of double-shifting.
    with fits.open(fn, mode='update') as hdul:
        hdul['SCI'].header[PENDING] = (True,
                                       'placement-field apply in progress')

    from jwst.datamodels import ImageModel
    from jwst.tweakreg.utils import adjust_wcs
    fa = ImageModel(fn)
    wcsobj = fa.meta.wcs
    fa.meta.oldwcs = copy.copy(wcsobj)
    ww = adjust_wcs(wcsobj, delta_ra=dra * u.deg, delta_dec=ddec * u.deg)
    fa.meta.wcs = ww
    fa.save(fn, overwrite=True)

    with fits.open(fn) as hdul:
        h = hdul['SCI'].header
        h.update(ww.to_fits()[0])
        h[MARKER] = (True, 'static SIAF placement-field shift applied')
        h['SPLASHRA'] = (dra, '[deg] placement-field RA coord shift')
        h['SPLASHDE'] = (ddec, '[deg] placement-field Dec shift')
        h[PENDING] = (False, 'placement-field apply completed')
        hdul.writeto(fn, overwrite=True)
    if verbose:
        print(f"placement correction applied to {fn} ({det}): "
              f"({dra * 3.6e6:+.2f}, {ddec * 3.6e6:+.2f}) mas coord")
        print("NOTE: consensus/VIRAC2locked offsets tables solved against the "
              "UNcorrected frame are invalidated by this shift -- rebuild them "
              "for this field before use.")
    return dra, ddec
