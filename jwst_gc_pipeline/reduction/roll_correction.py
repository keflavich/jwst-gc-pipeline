"""Field-rotation (telescope roll) correction for NIRCam frames (default ON).

WHY THIS EXISTS
---------------
``fix_alignment`` ties every frame to the Gaia-frame reference (VIRAC2 / Gaia DR3)
with a SHIFT only: the offsets tables carry (dRA, dDec) per exposure and
``adjust_wcs`` is called without ``delta_roll``.  Any error in the attitude
roll that ``assign_wcs`` received from the telescope pointing therefore
survives into the delivered WCS as a rigid rotation of the whole focal plane.

JWST-GC/data-qa#346 measured that rotation on every NIRCam observation with
per-exposure catalogs, by fitting a similarity transform of the per-exposure
PSF positions onto VIRAC2 and onto Gaia DR3 (both PM-propagated to the
exposure epoch).  The rotation appears against both references with the same
sign, the two references agree with each other in orientation to <~1", and
both NIRCam modules rotate by the same amount -- so it sits in the JWST
attitude, and a per-module shift cannot absorb it.

WHAT THIS MODULE DOES
---------------------
Rotates each frame's WCS by ``delta_roll`` about a PIVOT common to all ten
NIRCam detectors of an exposure: the NRCALL_FULL reference point (the centre
of the NIRCam field), mapped to the sky through the frame's own
``v2v3vacorr -> world`` transform.  ``jwst.tweakreg.utils.adjust_wcs`` rolls
a frame about the frame's OWN reference point, so a bare ``delta_roll``
would leave every detector centre in place and could not remove the
inter-detector part of the rotation.  The rotation about the pivot is
therefore written as a roll about the detector reference point PLUS a
shift of that reference point along the rotation arc about the pivot:

    ref' = pivot + R(delta_roll) (ref - pivot)
    adjust_wcs(wcs, delta_ra=ref'.ra - ref.ra, delta_dec=ref'.dec - ref.dec,
               delta_roll=delta_roll)

Pivoting on the field centre keeps the bulk (field-centre) position
unchanged to first order, so the existing shift tables stay close to valid;
they must still be re-measured after enabling this, because per-exposure
shifts were solved against the unrotated frames (see WARNING).

Sign convention (same as data-qa#346): ``delta_roll > 0`` rotates the frame
from North toward East (increasing position angle).  ``adjust_wcs``
``delta_roll`` uses the same sense (verified: a +20" delta_roll moves
positions by a +20.000" similarity rotation in this convention; see
``test_roll_correction.py``).

WHERE THE VALUE COMES FROM
--------------------------
``resolve_roll_arcsec(program, observation, visit)`` reads
``roll_corrections.csv`` next to this module (columns: program, observation,
visit, delta_roll_arcsec, delta_roll_err_arcsec, reference, source).  Rows
match exact values or ``*``; the most specific row wins.
``ROLL_CORRECTION_ARCSEC`` in the environment overrides the table with one
value for every frame (for tests / a global-constant run).

``fix_alignment`` calls :func:`ensure_roll_correction` on every frame unless
``ROLL_CORRECTION=0``.  It makes the frame carry the CURRENT table value,
whatever the frame already holds (:func:`roll_apply_plan`):

* no roll and no table row: nothing to do;
* no roll and a row: rotate by the row, about the NRCALL centre mapped
  through the frame's current WCS.  On a frame that already carries its
  reference shift ``t`` that pivot is ``P + t``, and
  ``R_(P+t) o S_t = S_t o R_P``, so the result is the frame a pre-shift
  roll would have produced;
* a roll equal to the row: nothing to do (the idempotent case);
* a roll that differs from the row: rotate by the difference about the
  pivot the first roll recorded (``ROLLPVRA``/``ROLLPVDE``).  Two rotations
  about one point add exactly, so the frame ends up carrying the row's value
  whichever tool rolled it first (this module, or ``image_roll_wcs.py``
  with its visit pivot);
* a roll with no row: refused.  A table that lost a row is more likely a
  regression than a decision to un-roll.

Fields whose offsets table is ``TABLE_LOCKED`` are deferred until their
``alignment_config`` entry sets ``roll_ready`` (see the note there): their
per-module shifts already absorbed part of the roll.

A ``ROLLPEND`` flag makes a crash between the GWCS write and the marker write
fail loud instead of double-rotating.  :func:`apply_roll_correction` is the
older single-shot entry point; it applies only to a virgin frame and refuses
a frame that already carries ``RAOFFSET``.

WARNING: rolling a field for the first time moves each exposure's detectors
relative to shifts solved on the unrotated frames.  On a consensus-channel
field the residual is a per-exposure translation of theta x (pivot offset),
which the next m2 checkpoint measures and writes back; re-drizzle and
re-catalog from m2.  On a locked field the per-module shifts absorbed part of
the roll, so the table must be rebuilt (``roll_ready`` above).
"""
import copy
import csv
import os
import time

import numpy as np
import astropy.units as u
from astropy.io import fits

from jwst_gc_pipeline.reduction.fits_wcs_sync import sync_header_to_gwcs

__all__ = ['resolve_roll_arcsec', 'rotation_about_pivot', 'roll_correction_needed',
           'apply_roll_correction', 'nircam_pivot_sky', 'roll_correction_enabled',
           'roll_apply_plan', 'ensure_roll_correction', 'RollStateError',
           'ROLL_APPLY_FULL', 'ROLL_APPLY_DELTA', 'ROLL_SKIP_CURRENT',
           'ROLL_SKIP_NO_ROW', 'ROLL_DEFER_LOCKED']

MARKER = 'ROLLCORR'
PENDING = 'ROLLPEND'
TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roll_corrections.csv')

# NRCALL_FULL reference point (SIAF PRD, arcsec).  Used only as the rotation
# pivot.  Moving the pivot by d adds a common rigid shift of d x delta_roll(rad)
# to every detector of the exposure: ~0.1 mas per arcsec of pivot offset at a
# 20" roll (1" x 20/206265).  The per-exposure reference shift absorbs that
# rigid term, so the pivot choice leaves the delivered astrometry unchanged
# once the offsets tables are rebuilt.
NRCALL_V2_ARCSEC = -1.957302
NRCALL_V3_ARCSEC = -492.89529


def _norm_obs(v):
    s = str(v).strip()
    return s if s == '*' else f"{int(s):03d}"


def resolve_roll_arcsec(program, observation, visit='*', table=TABLE):
    """Roll correction in arcsec for (program, observation, visit), or None.

    ``ROLL_CORRECTION_ARCSEC`` (env) overrides the table.  Table rows match
    exact values or ``*``; among matching rows the one with the fewest
    wildcards wins (visit-specific > observation-specific > program > global).
    """
    env = os.environ.get('ROLL_CORRECTION_ARCSEC')
    if env not in (None, ''):
        return float(env)
    if not os.path.exists(table):
        return None
    if program is None or observation is None or str(program).strip() in ('', '*'):
        return None
    prog = str(int(str(program).strip().lstrip('jw') or 0))
    obs = _norm_obs(observation)
    if visit is None or str(visit).strip() == '':
        visit = '*'
    vis = _norm_obs(visit) if str(visit).strip() != '*' else '*'
    best, best_rank = None, None
    with open(table) as fh:
        for row in csv.DictReader(line for line in fh if not line.lstrip().startswith('#')):
            rp = row['program'].strip()
            ro = _norm_obs(row['observation'])
            rv = _norm_obs(row['visit'])
            if rp not in ('*', prog) or ro not in ('*', obs) or rv not in ('*', vis):
                continue
            rank = (rp == '*') + (ro == '*') + (rv == '*')
            if best_rank is None or rank < best_rank:
                best, best_rank = float(row['delta_roll_arcsec']), rank
    return best


def _tangent(ra, dec, ra0, dec0):
    ra, dec, ra0, dec0 = map(np.radians, (ra, dec, ra0, dec0))
    cosc = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    xi = np.cos(dec) * np.sin(ra - ra0) / cosc
    eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cosc
    return xi, eta


def _untangent(xi, eta, ra0, dec0):
    # inverse gnomonic projection (vectorised; exact at xi = eta = 0)
    ra0, dec0 = np.radians(ra0), np.radians(dec0)
    xi, eta = np.asarray(xi, float), np.asarray(eta, float)
    den = np.cos(dec0) - eta * np.sin(dec0)
    ra = ra0 + np.arctan2(xi, den)
    dec = np.arctan2(np.sin(dec0) + eta * np.cos(dec0), np.hypot(xi, den))
    return np.degrees(ra) % 360.0, np.degrees(dec)


def rotation_about_pivot(ra, dec, ra_p, dec_p, delta_roll_arcsec):
    """Sky position (ra, dec) rotated about (ra_p, dec_p) by ``delta_roll_arcsec``,
    positive from North toward East.  Returns (ra', dec') in degrees."""
    d = np.radians(delta_roll_arcsec / 3600.0)
    x, y = _tangent(ra, dec, ra_p, dec_p)
    # (x east, y north): +d moves a point at PA phi to PA phi+d
    xr = x * np.cos(d) + y * np.sin(d)
    yr = -x * np.sin(d) + y * np.cos(d)
    return _untangent(xr, yr, ra_p, dec_p)


def _v23tosky_step(wcs):
    for step in wcs.pipeline[::-1]:
        if (step.frame.name and step.frame.name.startswith('v2v3')
                and step.transform is not None and step.transform.name == 'v23tosky'):
            return step
    raise ValueError("WCS has no v2v3 -> sky (v23tosky) step")


def nircam_pivot_sky(wcs):
    """NRCALL_FULL reference point on the sky through this frame's own attitude."""
    step = _v23tosky_step(wcs)
    T = wcs.get_transform(step.frame.name, 'world')
    ra, dec = T(NRCALL_V2_ARCSEC, NRCALL_V3_ARCSEC)
    return float(ra), float(dec)


def _reference_sky(wcs):
    """(ra_ref, dec_ref) the v23tosky step pins the frame's reference point to."""
    v2, v3, roll, dec, ra = _v23tosky_step(wcs).transform.parameters[-5:]
    return float(-ra) % 360.0, float(dec)


def roll_adjusted_wcs(wcs, delta_roll_arcsec, pivot=None):
    """New GWCS rotated by ``delta_roll_arcsec`` about ``pivot`` (default: NRCALL
    centre).  Returns (new_wcs, (dra_deg, ddec_deg), pivot)."""
    from jwst.tweakreg.utils import adjust_wcs
    if pivot is None:
        pivot = nircam_pivot_sky(wcs)
    ra_ref, dec_ref = _reference_sky(wcs)
    ra_new, dec_new = rotation_about_pivot(ra_ref, dec_ref, pivot[0], pivot[1], delta_roll_arcsec)
    dra = ((ra_new - ra_ref + 180.0) % 360.0) - 180.0
    ddec = dec_new - dec_ref
    ww = adjust_wcs(wcs, delta_ra=dra * u.deg, delta_dec=ddec * u.deg,
                    delta_roll=(delta_roll_arcsec / 3600.0) * u.deg)
    return ww, (dra, ddec), pivot


def verify_rotation(wcs_before, wcs_after, shape, pivot, delta_roll_arcsec, tol_mas=0.5):
    """Max deviation (mas) of the applied change from a pure rotation about ``pivot``,
    evaluated on a pixel grid.  Raises if above ``tol_mas``."""
    ny, nx = shape
    yy, xx = np.mgrid[0:ny:max(ny // 8, 1), 0:nx:max(nx // 8, 1)]
    ra0, dec0 = wcs_before(xx.ravel(), yy.ravel())
    ra1, dec1 = wcs_after(xx.ravel(), yy.ravel())
    good = np.isfinite(ra0) & np.isfinite(ra1)
    rap, decp = rotation_about_pivot(ra0[good], dec0[good], pivot[0], pivot[1], delta_roll_arcsec)
    dev = np.hypot((ra1[good] - rap) * np.cos(np.radians(dec1[good])), dec1[good] - decp) * 3.6e6
    worst = float(np.max(dev)) if dev.size else np.inf
    if not worst <= tol_mas:
        raise RuntimeError(f"roll correction verification failed: applied WCS change deviates "
                           f"from a {delta_roll_arcsec:+.2f}\" rotation about the pivot by up to "
                           f"{worst:.3f} mas (tolerance {tol_mas} mas); NOT writing.")
    return worst


def roll_correction_needed(header):
    return MARKER not in header


def apply_roll_correction(fn, delta_roll_arcsec=None, verbose=True):
    """Apply the roll correction to ``fn`` (cal/crf-like NIRCam file) in place,
    updating the ASDF GWCS and the FITS SCI-header WCS, idempotently.

    ``delta_roll_arcsec`` defaults to ``resolve_roll_arcsec`` for the frame's
    PROGRAM / OBSERVTN / VISIT.  Returns the applied roll (arcsec) or None.
    """
    hdr0 = fits.getheader(fn, ext=0)
    hdr = fits.getheader(fn, ext=('SCI', 1))
    if MARKER in hdr:
        if verbose:
            print(f"roll correction skipped for {fn}: already applied "
                  f"({hdr.get('ROLLARC')} arcsec)")
        return None
    if 'RAOFFSET' in hdr:
        # fix_alignment's own already-aligned guard runs AFTER this hook.  Rolling a
        # frame that already carries a baked reference shift would change its base
        # fiducial under that shift; the stale-base check would then refuse the
        # shift and leave the file half-processed.  Refuse before writing anything.
        raise RuntimeError(
            f"{fn}: frame already carries a reference shift (RAOFFSET="
            f"{hdr['RAOFFSET']}) but no {MARKER}; the roll must be applied before "
            f"the shift. Re-create this frame from its _cal and re-run fix_alignment "
            f"with ROLL_CORRECTION=1.")
    if hdr.get(PENDING):
        raise RuntimeError(
            f"{fn}: pending roll-correction marker without completion marker -- a "
            f"previous apply crashed mid-write. Re-create this frame from its _cal; "
            f"refusing to guess the GWCS state.")
    if delta_roll_arcsec is None:
        delta_roll_arcsec = resolve_roll_arcsec(hdr0.get('PROGRAM'), hdr0.get('OBSERVTN'),
                                                hdr0.get('VISIT', '*'))
    if delta_roll_arcsec is None or delta_roll_arcsec == 0:
        if verbose:
            print(f"roll correction skipped for {fn}: no roll configured for "
                  f"program {hdr0.get('PROGRAM')} obs {hdr0.get('OBSERVTN')}")
        return None

    pivot, dra, ddec, worst = _write_roll(fn, delta_roll_arcsec, delta_roll_arcsec)
    if verbose:
        print(f"roll correction applied to {fn}: {delta_roll_arcsec:+.2f}\" about "
              f"({pivot[0]:.6f}, {pivot[1]:.6f}); ref-point shift "
              f"({dra * 3.6e6:+.2f}, {ddec * 3.6e6:+.2f}) mas coord; verified to {worst:.3f} mas")
        print("NOTE: offsets tables solved against the unrotated frames are invalidated -- "
              "rebuild them for this field before use.")
    return float(delta_roll_arcsec)


def _write_roll(fn, delta_arcsec, total_arcsec, pivot=None, mode=None, previous_arcsec=None):
    """Rotate ``fn``'s GWCS by ``delta_arcsec`` about ``pivot`` (default: the
    NRCALL centre through the frame's current WCS), re-sync the SIP header,
    and stamp the roll keywords with ``total_arcsec`` as the roll the frame
    now carries.  Returns (pivot, dra_deg, ddec_deg, verify_mas)."""
    with fits.open(fn, mode='update') as hdul:
        hdul['SCI'].header[PENDING] = (True, 'roll correction apply in progress')

    from jwst.datamodels import ImageModel
    fa = ImageModel(fn)
    wcsobj = fa.meta.wcs
    ww, (dra, ddec), pivot = roll_adjusted_wcs(wcsobj, delta_arcsec, pivot=pivot)
    worst = verify_rotation(wcsobj, ww, fa.data.shape, pivot, delta_arcsec)
    fa.meta.oldwcs = copy.copy(wcsobj)
    fa.meta.wcs = ww
    fa.save(fn, overwrite=True)

    stamp = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    with fits.open(fn) as hdul:
        h = hdul['SCI'].header
        _sip_max, _sip_med = sync_header_to_gwcs(h, ww, fa.data.shape, label=os.path.basename(fn))
        h['SIPGWMAX'] = (_sip_max, '[mas] max FITS/SIP vs GWCS disagreement')
        h[MARKER] = (True, 'field roll correction applied (data-qa#346)')
        h['ROLLARC'] = (float(total_arcsec), '[arcsec] roll applied, +N->E')
        if previous_arcsec is None:
            h['ROLLPVRA'] = (pivot[0], '[deg] roll pivot RA (NRCALL_FULL ref)')
            h['ROLLPVDE'] = (pivot[1], '[deg] roll pivot Dec (NRCALL_FULL ref)')
            # the pivot is a sky position in the frame as it stood: a later
            # alignment shift moves it, and a delta must follow (see _delta_pivot)
            h['ROLLPVOR'] = (float(h.get('RAOFFSET', 0.0)), '[arcsec] RAOFFSET when pivot taken')
            h['ROLLPVOD'] = (float(h.get('DEOFFSET', 0.0)), '[arcsec] DEOFFSET when pivot taken')
            h['ROLLSHRA'] = (dra, '[deg] ref-point RA coord shift of the roll')
            h['ROLLSHDE'] = (ddec, '[deg] ref-point Dec shift of the roll')
            h['ROLLVMAS'] = (worst, '[mas] max deviation from pure rotation')
            if mode is not None:
                h['ROLLMODE'] = (mode, 'how the roll was applied')
                h['ROLLDATE'] = (stamp, 'UTC the roll was applied')
        else:
            # A re-correction keeps the first roll's pivot, mode and date (the
            # date is when the frame first moved, which catalog_roll_correction
            # reads) and records the step it took.
            h['ROLLPREV'] = (float(previous_arcsec), '[arcsec] roll carried before the delta')
            h['ROLLDVMA'] = (worst, '[mas] delta step: max deviation from rotation')
            h['ROLLDDAT'] = (stamp, 'UTC of the last delta re-correction')
            h['ROLLNDLT'] = (int(h.get('ROLLNDLT', 0)) + 1, 'delta re-corrections applied')
        h[PENDING] = (False, 'roll correction apply completed')
        hdul.writeto(fn, overwrite=True)
    return pivot, dra, ddec, worst


def roll_correction_enabled():
    """``fix_alignment`` applies the roll unless ``ROLL_CORRECTION=0``."""
    return os.environ.get('ROLL_CORRECTION', '1') != '0'


class RollStateError(RuntimeError):
    """The frame's roll keywords and the table cannot be reconciled."""


#: ``roll_apply_plan`` verdicts.
ROLL_APPLY_FULL = 'apply-full'        # no roll yet, table has a row: apply it
ROLL_APPLY_DELTA = 'apply-delta'      # rolled, table moved: apply the difference
ROLL_SKIP_CURRENT = 'skip-current'    # rolled by the table's value already
ROLL_SKIP_NO_ROW = 'skip-no-row'      # no roll, and the table has none to give
ROLL_DEFER_LOCKED = 'defer-locked'    # locked offsets table not rebuilt for the roll

#: Same tolerance stage_release's roll gate uses on ROLLARC vs the table.
ROLL_TOL_ARCSEC = 1e-6


def roll_apply_plan(header, target_arcsec, locked_unready=False, fn=''):
    """What ``ensure_roll_correction`` should do with one frame.

    ``header`` is the frame's SCI header; ``target_arcsec`` the table value
    (``resolve_roll_arcsec``) or None.  Returns ``(verdict, delta_arcsec,
    reason)``.  Raises :class:`RollStateError` for a crashed apply, a marker
    without a readable value, or a rolled frame whose row is gone.  Kept
    apart from the file I/O so the policy is testable on a bare header.
    """
    label = fn or 'this frame'
    if header.get(PENDING):
        raise RollStateError(
            f"{label}: pending roll-correction marker without completion marker -- a "
            f"previous apply crashed mid-write. Re-create this frame from its _cal; "
            f"refusing to guess the GWCS state.")
    if header.get(MARKER):
        try:
            applied = float(header['ROLLARC'])
        except (KeyError, TypeError, ValueError):
            applied = float('nan')
        if not np.isfinite(applied):
            raise RollStateError(
                f"{label}: carries {MARKER} but no readable ROLLARC "
                f"({header.get('ROLLARC')!r}); the roll it holds is unknown.")
        if target_arcsec is None:
            raise RollStateError(
                f"{label}: carries a {applied:+.3f}\" roll but roll_corrections.csv has "
                f"no row for it. Restore the row, or restore the frame "
                f"(image_roll_wcs --restore / regenerate from an unrolled _cal) if "
                f"the roll is meant to go.")
        delta = float(target_arcsec) - applied
        if abs(delta) <= ROLL_TOL_ARCSEC:
            return ROLL_SKIP_CURRENT, 0.0, f'already carries {applied:+.3f}"'
        return ROLL_APPLY_DELTA, delta, (
            f'carries {applied:+.3f}" but the table says {float(target_arcsec):+.3f}"')
    if target_arcsec is None or float(target_arcsec) == 0.0:
        return ROLL_SKIP_NO_ROW, 0.0, 'no roll configured'
    if locked_unready:
        return ROLL_DEFER_LOCKED, 0.0, (
            'the field\'s locked offsets table was solved on unrotated frames; set '
            'roll_ready in its alignment_config entry once the table is rebuilt '
            'from rolled frames')
    return ROLL_APPLY_FULL, float(target_arcsec), 'first roll'


def _delta_pivot(hdr, delta_arcsec, fn='', why=''):
    """The first roll's pivot, as a sky position in the frame's CURRENT WCS.

    Rotations about one pivot add exactly, so a delta about it lands the frame
    where a single rotation by the new total would have.  The pivot is
    recorded in the frame as it stood when rolled; an alignment shift applied
    since (``RAOFFSET`` changed) carries the pivot with it, since
    S_t R_P = R_(P+t) S_t.  ``ROLLPVOR``/``ROLLPVOD`` hold the shift at roll
    time.  Frames rolled before those keys existed (``image_roll_wcs``) are
    taken as rolled at their current shift; the cost of being wrong is a
    rigid translation of at most delta[rad] * |RAOFFSET, DEOFFSET|, which this
    prints, and which the m2 re-tie that any roll change requires absorbs.
    """
    try:
        ra_p, dec_p = float(hdr['ROLLPVRA']), float(hdr['ROLLPVDE'])
    except (KeyError, TypeError, ValueError) as ex:
        raise RollStateError(
            f"{fn}: {why}, but the first roll's pivot (ROLLPVRA/ROLLPVDE) is "
            f"unreadable ({ex}); a delta about any other point would add a "
            f"translation. Regenerate the frame.") from ex
    raoff, deoff = float(hdr.get('RAOFFSET', 0.0)), float(hdr.get('DEOFFSET', 0.0))
    if 'ROLLPVOR' in hdr and 'ROLLPVOD' in hdr:
        ra_p += (raoff - float(hdr['ROLLPVOR'])) / 3600.0
        dec_p += (deoff - float(hdr['ROLLPVOD'])) / 3600.0
    elif raoff or deoff:
        bound_mas = abs(np.deg2rad(delta_arcsec / 3600.0)) * np.hypot(raoff, deoff) * 1e3
        print(f"roll correction: {fn} predates ROLLPVOR/ROLLPVOD; taking its pivot as "
              f"recorded at the current shift ({raoff:+.3f}, {deoff:+.3f})\". If it was "
              f"taken unshifted the delta adds a rigid translation <= {bound_mas:.3f} mas.")
    return ra_p, dec_p


_DEFERRED_REPORTED = set()


def ensure_roll_correction(fn, verbose=True):
    """Make ``fn`` carry the roll ``roll_corrections.csv`` gives it.

    Idempotent and convergent: a second call is a no-op, and a call after the
    table changed applies only the difference.  Returns the verdict
    (``ROLL_*``).  See the module docstring for the cases.
    """
    hdr0 = fits.getheader(fn, ext=0)
    hdr = fits.getheader(fn, ext=('SCI', 1))
    program, observation = hdr0.get('PROGRAM'), hdr0.get('OBSERVTN')
    target = resolve_roll_arcsec(program, observation, hdr0.get('VISIT', '*'))
    locked_unready = False
    if not hdr.get(MARKER) and target not in (None, 0.0) and program and observation:
        from jwst_gc_pipeline.reduction.alignment_config import roll_blocked_by_locked_table
        prog = str(int(str(program).strip().lstrip('jw')))
        locked_unready = roll_blocked_by_locked_table(prog, f"{int(observation):03d}")
    verdict, delta, why = roll_apply_plan(hdr, target, locked_unready, fn)

    if verdict == ROLL_SKIP_NO_ROW:
        if verbose:
            print(f"roll correction: none for {fn} (program {program} obs {observation} "
                  f"has no row)")
        return verdict
    if verdict == ROLL_SKIP_CURRENT:
        if verbose:
            print(f"roll correction: {fn} {why}; nothing to do")
        return verdict
    if verdict == ROLL_DEFER_LOCKED:
        key = (program, observation)
        if key not in _DEFERRED_REPORTED:
            _DEFERRED_REPORTED.add(key)
            print(f"WARNING: roll correction DEFERRED for program {program} obs "
                  f"{observation} ({target:+.3f}\" in roll_corrections.csv): {why}. "
                  f"Frames of this observation stay unrolled.", flush=True)
        return verdict

    if verdict == ROLL_APPLY_FULL:
        mode = 'post-shift-nrcall' if 'RAOFFSET' in hdr else 'pre-shift-nrcall'
        pivot, dra, ddec, worst = _write_roll(fn, delta, delta, mode=mode)
        if verbose:
            print(f"roll correction applied to {fn}: {delta:+.3f}\" about "
                  f"({pivot[0]:.6f}, {pivot[1]:.6f}) [{mode}]; verified to {worst:.3f} mas")
        return verdict

    # ROLL_APPLY_DELTA: rotate about the pivot the first roll used.
    pivot = _delta_pivot(hdr, delta, fn, why)
    applied = float(hdr['ROLLARC'])
    print(f"STALE ROLL -- RE-CORRECTING {fn}: {why}; applying {delta:+.3f}\" about "
          f"the recorded pivot ({pivot[0]:.6f}, {pivot[1]:.6f}).", flush=True)
    _, _, _, worst = _write_roll(fn, delta, applied + delta, pivot=pivot,
                                 previous_arcsec=applied)
    if verbose:
        print(f"roll correction delta verified to {worst:.3f} mas; {fn} now carries "
              f"{applied + delta:+.3f}\"")
    return verdict
