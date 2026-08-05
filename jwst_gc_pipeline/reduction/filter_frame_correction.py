"""Per-(detector, filter) placement correction onto the JWST 2-micron frame.

WHAT IS BEING CORRECTED
-----------------------
The true filter-dependent placement of a NIRCam detector varies DETECTOR TO
DETECTOR inside a module, and neither of the two things that could express that
is keyed finely enough:

* CRDS ``jwst_nircam_filteroffset`` is keyed on ``(META.INSTRUMENT.CHANNEL,
  META.INSTRUMENT.MODULE)`` -- four reference files for ten detectors, each
  holding ``{filter, pupil, column_offset, row_offset}`` rows with no detector
  key.  A filter offset that differs between the four SW detectors of a module
  has nowhere to go.
* the pipeline offsets table is keyed on ``(visit, exposure, MODULE)`` and the
  tie it applies is a pure translation, so it cannot remove it either.

Jay Anderson's per-(detector, filter) STDGDC solutions state the size of what
CRDS drops.  Taking ``STDGDC_f.forward - STDGDC_F212N.forward`` per detector,
removing the per-MODULE mean (the part CRDS *does* apply), the residual is:

    F115W - F212N   1.18 mas        F182M - F212N   1.94 mas
    F150W - F212N   1.65 mas        F200W - F212N   2.11 mas
    F210M - F212N   1.93 mas

F200W is the instructive one: its raw per-detector offsets are -81/-136 mas in
module A and +39/+48 in module B -- the large F200W filteroffset with its A/B
split, which CRDS applies correctly -- and 2.11 mas is what is left once that
per-module constant is taken out.

WHY IT SHOWS UP AS A POSITION-DEPENDENT FIELD
---------------------------------------------
In a mosaic, different sky positions are covered by different mixes of the four
detectors of a module, so a per-detector CONSTANT becomes a position-dependent
sky-frame error.  Measured on the Brick m7 catalogs (issue #296): 1.40 mas rms
per component (1.97 mas as a 2-D vector) between two SW filters, 2.50 / 3.54
across the SW/LW split, at a per-cell standard error of 0.03-0.13 mas.  Both
conventions are quoted because the two get crossed easily -- see #297.
Per-star precision is untouched, which is why the per-star (~1 mas) and
field-wide (2-4 mas) numbers disagree.

WHAT THIS CANNOT REACH
----------------------
LW.  The module-mean gauge (below) returns exactly zero for a module with one
detector, so NRCALONG/NRCBLONG can never be corrected by this mechanism --
structurally, not for want of a solve.  The SW/LW split is #296's largest
single term, and it belongs to the module-level machinery, not here.  On the
SW side, applying the shipped F182M table to the Brick takes the F182M-F212N
field from 1.971 to 0.647 mas 2-D (67% of the amplitude, 89% of the variance)
and F187N from 0.743 to 0.371 -- real, and still above the 0.16-0.41 mas
per-detector shape floor, so this is not the last word on the SW side either.

SIGN
----
The measurement is a RESIDUAL: ``(filter f position) - (anchor position)``.
The correction is its NEGATIVE.  Everything public in this module -- the ECSV
``dx_mas``/``dy_mas``, ``gdc_filter_offsets``, the dict passed to
``apply_filter_frame_correction`` -- is **the correction, already negated**,
i.e. the shift to ADD to filter f's positions.  ``solve_filter_frame_offsets``
measures the residual and negates it when it writes the table; if you feed a
measured residual straight in, the correction doubles the error rather than
removing it.

WHAT THIS APPLIES
-----------------
A rigid per-detector shift -- the same shape as ``dva_correction`` and
``static_placement_correction``, so it goes through
``jwst.tweakreg.utils.adjust_wcs`` and needs no GWCS surgery.  The
position-dependent *shape* within a detector is NOT corrected here and does not
need to be: differencing the anchor filter's own field out of the measurement
leaves only 0.16-0.41 mas of genuinely filter-specific shape, an order below
the constant.

GAUGE, AND WHY THIS ONE CAN BE APPLIED LAST
-------------------------------------------
The correction is defined with the per-MODULE mean removed, so it is orthogonal
to everything the offsets table can express.  Applying it moves no visit bulk,
no module tie and no reference tie -- it does not invalidate an offsets table
and does not require a re-tie.  (Contrast ``static_placement_correction``, whose
docstring warns that enabling it invalidates any table solved against the
uncorrected frame.)  That is what makes it safe as the last term applied.

COEFFICIENT SOURCES
-------------------
``gdc``     -- derived from Jay Anderson's STDGDC library.  Covers SW F070W,
               F090W, F115W, F140M, F150W, F182M, F200W, F210M, F212N.
``table``   -- an ECSV solved from our own data, for everything the library does
               not cover: F187N and every LW GC band (F323N ... F480M, since the
               library's only LW solution is F277W).  Format is documented in
               ``filter_frame_table_schema``.

Opt-in: ``FILTER_FRAME_CORRECTION=1``.  Default off, byte-identical behaviour.
"""
import copy
import os

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.reduction.fits_wcs_sync import sync_header_to_gwcs

MARKER = 'FFRAMCOR'
PENDING = 'FFRAMPND'

#: The frame every filter is brought onto: the observed filter closest to this.
ANCHOR_TARGET_UM = 2.0

#: NIRCam SW/LW plate scales (arcsec/pixel), for converting STDGDC pixel
#: differences to mas.  LW is only used if a LW STDGDC solution ever exists.
PIXSCALE_ARCSEC = {'SW': 0.031, 'LW': 0.063}

SW_DETECTORS = ('NRCA1', 'NRCA2', 'NRCA3', 'NRCA4',
                'NRCB1', 'NRCB2', 'NRCB3', 'NRCB4')


class FilterFrameError(RuntimeError):
    """No usable per-(detector, filter) coefficients for this request."""


def filter_wavelength_um(filtername):
    """Wavelength in micron parsed from a JWST filter name.

    The digits are hundredths of a micron in BOTH conventions, so the whole
    digit run must be parsed, not the first three: NIRCam ``F212N`` -> 2.12,
    MIRI ``F1000W`` -> 10.00.  Taking three digits reads ``F1000W`` as 1.0
    micron, which makes a MIRI filter look like the best 2-micron anchor.
    """
    name = str(filtername).upper().strip()
    digits = name[1:].rstrip('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    if not name.startswith('F') or not digits.isdigit():
        raise ValueError(f"cannot parse wavelength from filter name {filtername!r}")
    return int(digits) / 100.0


def anchor_filter(filternames, target_um=ANCHOR_TARGET_UM):
    """The observed filter closest to ``target_um`` -- the frame everything else
    is tied onto.  Ties broken by name so the choice is reproducible."""
    names = sorted({str(f).upper().strip() for f in filternames if f})
    if not names:
        raise FilterFrameError("no filters given")
    return min(names, key=lambda f: (abs(filter_wavelength_um(f) - target_um), f))


def _module_of(detector):
    d = str(detector).upper().strip()
    return 'NRCA' if d.startswith('NRCA') else ('NRCB' if d.startswith('NRCB') else None)


def _remove_module_means(offsets):
    """Subtract, per module, the mean over that module's detectors.

    This is the gauge that makes the correction orthogonal to the offsets table
    (which carries one row per (visit, exposure, module)): whatever the module
    mean is, the table already owns it, and re-applying it here would either
    double-correct or force a re-tie.

    **Consequence for LW, which is structural and not a scheduling gap:** a LW
    module has ONE detector, so its module mean is that detector's own value
    and the gauge returns exactly zero.  This mechanism can therefore never
    correct NRCALONG/NRCBLONG -- correctly, because any LW per-detector shift
    IS a module-level shift and the offsets table already owns it.  #296's
    largest measured term is the 2.5 mas SW/LW split, and that term is out of
    reach here; it needs the module-level machinery, not this.
    """
    out = {}
    for mod in ('NRCA', 'NRCB'):
        members = {d: v for d, v in offsets.items() if _module_of(d) == mod}
        if not members:
            continue
        arr = np.array(list(members.values()), dtype=float)
        mean = arr.mean(axis=0)
        for d, v in members.items():
            out[d] = (float(v[0] - mean[0]), float(v[1] - mean[1]))
    return out


def gdc_filter_offsets(filtername, anchor, detectors=SW_DETECTORS, nsamp=24,
                       pixscale_arcsec=None):
    """Per-detector correction from Jay's STDGDC library, in the DETECTOR frame.

    ``-(STDGDC_f.forward - STDGDC_anchor.forward)`` averaged over the detector
    (negated, so it is a correction and not a residual -- see SIGN above), then
    module-mean-removed.  Returns ``{detector: (dx_mas, dy_mas)}``.

    **This output is NOT applicable as it stands.**  It is in the detector
    array frame, and no validated detector -> instrument converter exists here:
    a Procrustes fit of these vectors onto the data-solved instrument-frame
    table needs a REFLECTION (det(R) = -1, 93.4% explained, residual 0.43 mas),
    so the two frames differ by a parity as well as a rotation, and feeding
    this into ``apply_filter_frame_correction`` as ``frame='instrument'``
    applies a mirrored vector.  ``apply_filter_frame_correction`` therefore
    refuses ``frame='detector'`` outright.

    Use this for what it is good for: an INDEPENDENT SIZING of the term, from
    an external solution, against which the data-solved amplitudes can be
    checked (1.94 mas here vs 1.51-1.89 measured for F182M).  Making it a
    coefficient source needs the converter, and the converter needs validating
    against SIAF detector orientations, which is not done.

    Raises ``FilterFrameError`` if either filter has no solution for any of the
    requested detectors, rather than silently returning a subset (a partial
    set would break the module-mean gauge).
    """
    from jwst_gc_pipeline.astrometry_gdc.stdgdc import STDGDC, GDCFileNotFoundError

    scale = (pixscale_arcsec or PIXSCALE_ARCSEC['SW']) * 1000.0
    grid = np.linspace(64, 1984, nsamp)
    gx, gy = np.meshgrid(grid, grid)
    x, y = gx.ravel(), gy.ravel()

    raw = {}
    for det in detectors:
        try:
            a = STDGDC.load(det.upper(), str(filtername).upper())
            b = STDGDC.load(det.upper(), str(anchor).upper())
        except (GDCFileNotFoundError, FileNotFoundError) as exc:
            raise FilterFrameError(
                f"no STDGDC solution for ({det}, {filtername}) or "
                f"({det}, {anchor}): {exc}.  The library covers SW F070W, "
                f"F090W, F115W, F140M, F150W, F182M, F200W, F210M, F212N and "
                f"LW F277W only -- use a solved table for anything else.") from exc
        xa, ya = a.forward(x, y)
        xb, yb = b.forward(x, y)
        ok = np.isfinite(xa) & np.isfinite(ya) & np.isfinite(xb) & np.isfinite(yb)
        if not ok.any():
            raise FilterFrameError(f"STDGDC forward map is empty for {det}")
        # negated: the library states the residual, we return the correction
        raw[det.upper()] = (float(-np.mean(xa[ok] - xb[ok]) * scale),
                            float(-np.mean(ya[ok] - yb[ok]) * scale))
    return _remove_module_means(raw)


def filter_frame_table_schema():
    """Column contract for a solved coefficient table (ECSV).

    ``detector``  NRCA1..NRCB4, NRCALONG, NRCBLONG
    ``filter``    the filter this row corrects
    ``anchor``    the filter it is being brought onto
    ``frame``     ``'instrument'`` (portable) or ``'sky'`` (roll-specific)
    ``dx_mas``    first component of the CORRECTION to ADD, mas,
                  module-mean-removed.  This is the NEGATIVE of the measured
                  (filter - anchor) residual; see SIGN in the module docstring
    ``dy_mas``    second component
    ``n``         number of OBSERVATIONS averaged into the row (not exposures)
    ``sem_mas``   standard error of the row across those observations

    ``'sky'`` rows are only valid at the roll they were solved at.  The
    portable convention is ``'instrument'``: the measured on-sky vector rotated
    by ``+ROLL_REF``, which is what makes one table serve every observation.
    Verified on three observations (brick 2221-o001 roll 89.1, sgrc 4147-o012
    roll 91.5, wd2 3523-o005 roll 141.0 -- two fields, two years, 52 degrees of
    roll): de-rotating collapses them onto one vector set with 94-98% of the
    variance explained and a per-detector scatter of 0.19 mas on a 1.69 mas
    signal.
    """
    return ('detector', 'filter', 'anchor', 'frame', 'dx_mas', 'dy_mas',
            'n', 'sem_mas')


def instrument_to_sky(offsets_mas, roll_ref_deg):
    """Rotate instrument-frame rows to on-sky (dRA*, dDec) for one exposure.

    Inverse of the de-rotation used when the table is solved: stored vectors
    are the sky vectors rotated by ``+ROLL_REF``, so applying them to a frame
    means rotating back by ``-ROLL_REF``.
    """
    t = np.radians(-float(roll_ref_deg))
    c, s = np.cos(t), np.sin(t)
    return {d: (float(c * v[0] - s * v[1]), float(s * v[0] + c * v[1]))
            for d, v in offsets_mas.items()}


def load_filter_frame_table(path):
    """Read a solved table and validate it against the schema."""
    tbl = Table.read(path)
    missing = [c for c in filter_frame_table_schema() if c not in tbl.colnames]
    if missing:
        raise FilterFrameError(f"{path}: missing column(s) {missing}; expected "
                               f"{filter_frame_table_schema()}")
    return tbl


def table_filter_offsets(tbl, filtername, anchor):
    """``{detector: (dra_mas, ddec_mas)}`` from a solved table, re-gauged.

    The stored rows are re-gauged here rather than trusted, so a table written
    with a different (or no) gauge cannot silently inject a module-level shift
    that the offsets table already owns.
    """
    f = str(filtername).upper().strip()
    a = str(anchor).upper().strip()
    sel = [r for r in tbl
           if str(r['filter']).upper() == f and str(r['anchor']).upper() == a]
    if not sel:
        raise FilterFrameError(
            f"solved table has no rows for filter {f} anchored on {a}")
    frames = {str(r['frame']).lower() for r in sel}
    if len(frames) > 1:
        raise FilterFrameError(
            f"rows for {f}/{a} mix frames {sorted(frames)}; a table must be "
            f"all 'instrument' or all 'sky'")
    return frames.pop(), _remove_module_means(
        {str(r['detector']).upper(): (float(r['dx_mas']), float(r['dy_mas']))
         for r in sel})


def _env_flag(name):
    return str(os.environ.get(name, '0')).strip().lower() in ('1', 'true', 'yes', 'on')


def correction_enabled():
    return _env_flag('FILTER_FRAME_CORRECTION')


def check_apply_preconditions(hdr, det, offsets_mas, anchor=None, label=""):
    """Decide whether ``hdr``'s frame may be corrected.  Pure, so it is testable.

    Returns ``'skip'`` when the file is already corrected onto the same anchor,
    ``'go'`` otherwise, and raises when continuing would be unsafe.
    """
    if MARKER in hdr:
        prior = str(hdr.get('FFRAMANC', '') or '').upper().strip()
        want = str(anchor or '').upper().strip()
        if want and prior and prior != want:
            # Idempotency by marker alone would silently accept a file already
            # tied onto a DIFFERENT anchor; the image and the catalog would
            # then sit on different frames with nothing recording it.
            raise FilterFrameError(
                f"{label}: already corrected onto anchor {prior!r}, now asked "
                f"for {want!r}.  Re-derive from _cal rather than stacking.")
        return 'skip'
    if hdr.get(PENDING):
        raise FilterFrameError(
            f"{label}: pending filter-frame marker without a completion marker "
            f"-- a previous apply crashed mid-write.  Re-fetch or re-reduce "
            f"this file; refusing to guess the GWCS state.")
    if det not in offsets_mas:
        # Refuse rather than skip.  gdc_filter_offsets already refuses a
        # partial coefficient set because a partial correction breaks the
        # module-mean gauge; silently skipping one exposure of a filter
        # produces exactly that, one frame at a time.
        raise FilterFrameError(
            f"{label}: no coefficients for detector {det!r} (have "
            f"{sorted(offsets_mas)}).  Correcting only some detectors of a "
            f"module breaks the module-mean gauge -- supply a full set or "
            f"correct nothing.")
    return 'go'


def sky_shift_deg(offsets_mas, det, frame, roll_ref_deg, dec_deg):
    """The coordinate shift to apply, in degrees.  Pure, so it is testable.

    ``offsets_mas`` holds CORRECTIONS (see SIGN in the module docstring), so
    the returned shift is added to the WCS as-is.  ``frame='instrument'``
    rotates by ``-roll_ref_deg`` first.  The RA component is divided by
    ``cos(dec)`` because the stored value is an on-sky offset and the WCS takes
    a coordinate offset.
    """
    f = str(frame).lower()
    if f == 'instrument':
        if roll_ref_deg is None:
            raise FilterFrameError(
                "instrument-frame coefficients need ROLL_REF to rotate onto "
                "sky, and the header has none")
        dra_mas, ddec_mas = instrument_to_sky(offsets_mas, roll_ref_deg)[det]
    elif f == 'sky':
        dra_mas, ddec_mas = offsets_mas[det]
    elif f == 'detector':
        raise FilterFrameError(
            "frame='detector' is not applicable: the detector array frame "
            "differs from the instrument frame by a parity as well as a "
            "rotation (see gdc_filter_offsets), and no validated converter "
            "exists.  Solve an instrument-frame table instead.")
    else:
        raise FilterFrameError(f"unknown frame {frame!r}; expected "
                               f"'instrument' or 'sky'")
    if dec_deg is None:
        raise FilterFrameError(
            "no declination, so the on-sky mas -> RA-coordinate conversion is "
            "undefined.  Defaulting to cos(0)=1 would be a 14% error at "
            "Galactic-centre declinations.")
    cosd = max(np.cos(np.radians(float(dec_deg))), 1e-6)
    return dra_mas / 3.6e6 / cosd, ddec_mas / 3.6e6


def apply_filter_frame_correction(fn, offsets_mas, frame='instrument',
                                  anchor=None, source=None, verbose=True):
    """Apply the per-detector shift for ``fn``'s detector, idempotently.

    ``offsets_mas`` is ``{detector: (dx_mas, dy_mas)}``, module-mean-removed
    CORRECTIONS.  ``frame='instrument'`` (the portable convention) rotates them
    onto sky using this exposure's own ``ROLL_REF``; ``frame='sky'`` takes them
    as on-sky already and is only valid at the roll they were solved at.
    Returns the applied ``(dra_deg, ddec_deg)``, or None when skipped.

    Mirrors ``static_placement_correction.apply_placement_correction``: a
    PENDING flag is written before the GWCS rewrite and cleared in the same
    write that sets the completion marker, so a crash between the two fails
    loud on re-run instead of double-shifting.

    The two decisions this makes are in ``check_apply_preconditions`` and
    ``sky_shift_deg``, which are pure and directly tested; everything below is
    FITS/GWCS plumbing.
    """
    hdr0 = fits.getheader(fn, ext=0)
    hdr = fits.getheader(fn, ext=('SCI', 1))
    det = str(hdr.get('DETECTOR', hdr0.get('DETECTOR')) or '').upper().strip()
    if check_apply_preconditions(hdr, det, offsets_mas, anchor=anchor,
                                 label=fn) == 'skip':
        if verbose:
            print(f"filter-frame correction skipped for {fn}: already applied")
        return None
    dra, ddec = sky_shift_deg(offsets_mas, det, frame,
                              hdr.get('ROLL_REF', hdr0.get('ROLL_REF')),
                              hdr.get('CRVAL2', hdr0.get('CRVAL2')))
    dra_mas = dra * 3.6e6 * max(np.cos(np.radians(float(hdr['CRVAL2']))), 1e-6)
    ddec_mas = ddec * 3.6e6

    with fits.open(fn, mode='update') as hdul:
        hdul['SCI'].header[PENDING] = (True, 'filter-frame apply in progress')

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
        _sip_max, _sip_med = sync_header_to_gwcs(h, ww, fa.data.shape,
                                                 label=os.path.basename(fn))
        h['SIPGWMAX'] = (_sip_max, '[mas] max FITS/SIP vs GWCS disagreement')
        h[MARKER] = (True, 'per-(detector,filter) frame shift applied')
        h['FFRAMRA'] = (dra, '[deg] filter-frame RA coord shift')
        h['FFRAMDE'] = (ddec, '[deg] filter-frame Dec shift')
        h['FFRAMANC'] = (str(anchor or ''), 'anchor filter of the frame tied onto')
        h['FFRAMSRC'] = (str(source or ''), 'coefficient source (table path / gdc)')
        h['FFRAMFRM'] = (str(frame), 'coefficient frame as supplied')
        h[PENDING] = (False, 'filter-frame apply completed')
        hdul.writeto(fn, overwrite=True)
    if verbose:
        print(f"filter-frame correction applied to {fn} ({det}): "
              f"({dra_mas:+.2f}, {ddec_mas:+.2f}) mas on sky.  Module-mean "
              f"removed, so no offsets table is invalidated.")
    return dra, ddec
