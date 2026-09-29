"""Field-rotation (telescope roll) correction for NIRCam frames (opt-in, default OFF).

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

Enable with ``ROLL_CORRECTION=1``.  Applied in ``fix_alignment`` after the
DVA and placement corrections and BEFORE the reference shift.  Idempotent
via the ``ROLLCORR`` marker; a ``ROLLPEND`` flag makes a crash between the
GWCS write and the marker write fail loud instead of double-rotating.

WARNING: enabling this on an already-processed field invalidates the
consensus / VIRAC2locked offsets tables solved against the unrotated frames
(per-exposure shifts would stack on a rotation they partly absorbed).
Rebuild the offsets tables for the field after enabling, then re-drizzle
and re-catalog.
"""
import copy
import csv
import os

import numpy as np
import astropy.units as u
from astropy.io import fits

from jwst_gc_pipeline.reduction.fits_wcs_sync import sync_header_to_gwcs

__all__ = ['resolve_roll_arcsec', 'rotation_about_pivot', 'roll_correction_needed',
           'apply_roll_correction', 'nircam_pivot_sky']

MARKER = 'ROLLCORR'
PENDING = 'ROLLPEND'
TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'roll_corrections.csv')

# NRCALL_FULL reference point (SIAF PRD, arcsec).  Used only as the rotation
# pivot: an arcsec-level error in it moves the pivot, i.e. adds a common rigid
# shift of (pivot error) x (delta_roll in rad) ~ 1e-4 mas, which the reference
# tie absorbs.
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
    prog = str(int(str(program).lstrip('jw') or 0))
    obs = _norm_obs(observation)
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

    with fits.open(fn, mode='update') as hdul:
        hdul['SCI'].header[PENDING] = (True, 'roll correction apply in progress')

    from jwst.datamodels import ImageModel
    fa = ImageModel(fn)
    wcsobj = fa.meta.wcs
    ww, (dra, ddec), pivot = roll_adjusted_wcs(wcsobj, delta_roll_arcsec)
    worst = verify_rotation(wcsobj, ww, fa.data.shape, pivot, delta_roll_arcsec)
    fa.meta.oldwcs = copy.copy(wcsobj)
    fa.meta.wcs = ww
    fa.save(fn, overwrite=True)

    with fits.open(fn) as hdul:
        h = hdul['SCI'].header
        _sip_max, _sip_med = sync_header_to_gwcs(h, ww, fa.data.shape, label=os.path.basename(fn))
        h['SIPGWMAX'] = (_sip_max, '[mas] max FITS/SIP vs GWCS disagreement')
        h[MARKER] = (True, 'field roll correction applied (data-qa#346)')
        h['ROLLARC'] = (float(delta_roll_arcsec), '[arcsec] roll applied, +N->E')
        h['ROLLPVRA'] = (pivot[0], '[deg] roll pivot RA (NRCALL_FULL ref)')
        h['ROLLPVDE'] = (pivot[1], '[deg] roll pivot Dec (NRCALL_FULL ref)')
        h['ROLLSHRA'] = (dra, '[deg] ref-point RA coord shift of the roll')
        h['ROLLSHDE'] = (ddec, '[deg] ref-point Dec shift of the roll')
        h['ROLLVMAS'] = (worst, '[mas] max deviation from pure rotation')
        h[PENDING] = (False, 'roll correction apply completed')
        hdul.writeto(fn, overwrite=True)
    if verbose:
        print(f"roll correction applied to {fn}: {delta_roll_arcsec:+.2f}\" about "
              f"({pivot[0]:.6f}, {pivot[1]:.6f}); ref-point shift "
              f"({dra * 3.6e6:+.2f}, {ddec * 3.6e6:+.2f}) mas coord; verified to {worst:.3f} mas")
        print("NOTE: offsets tables solved against the unrotated frames are invalidated -- "
              "rebuild them for this field before use.")
    return float(delta_roll_arcsec)
