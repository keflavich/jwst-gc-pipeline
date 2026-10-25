"""Header-only roll correction of IMAGE WCSes (data-qa#346), after alignment.

WHAT THIS DOES
--------------
Rotates the WCS -- and only the WCS -- of already-aligned NIRCam and MIRI
imager products by
the per-visit roll in ``roll_corrections.csv``, about the same per-visit pivot
the catalog correction uses (``astrometry.catalog_roll_correction``: the visit
coverage centroid).  Pixels, fluxes and fits are untouched; nothing is
re-fit or re-drizzled.  After this step the images agree with the rotated
catalogs in ``<field>/catalogs_rollcorr/``, and a future re-catalog made from
the rotated frames is born rotated.

Products
~~~~~~~~
* **frames** (``*_destreak.fits``, ``*_destreak_o<obs>_crf.fits`` and the
  other aligned frame types that carry ``RAOFFSET``): the GWCS is rotated with
  ``roll_correction.roll_adjusted_wcs`` about the visit pivot, verified
  against ``rotation_about_pivot`` to <0.5 mas on a pixel grid, and the FITS
  SIP is re-synced with ``fits_wcs_sync.sync_header_to_gwcs`` (rule #2).
* **``*_cal.fits``** (the source a regeneration re-derives ``_destreak``
  from): rotated about the pivot moved back by the frame's baked
  ``RAOFFSET``/``DEOFFSET`` (``P - t``).  ``fix_alignment`` re-applies the
  table shift ``t`` after the rotation on regeneration, and
  ``S_t o R_(P-t) = R_P o S_t``, so a regenerated frame reproduces the
  rotated aligned frame instead of losing the roll.
* **i2d mosaics** (``jw<prog>-o<obs>_t*_nircam_*_i2d.fits``,
  ``jw<prog>-o<obs>_t*_miri_<filter>_i2d.fits``): CRVAL and PC in
  the SCI header, the ``FITSImagingWCSTransform`` in the ASDF GWCS and
  ``meta.wcsinfo`` are updated together.  A single-roll mosaic is corrected
  EXACTLY (a rotation of the sphere maps a TAN projection onto a TAN
  projection).  A multi-visit mosaic (brick 1182 o004, cloudc 2221 o002,
  cloudef 2092 o002) gets the best single similarity to the per-pixel
  catalog model; the residual is measured and stored (``ROLLIRES``) and the
  write is refused above ``--i2d-tol-mas``.

MIRI imager
~~~~~~~~~~~
MIRI frames (``*_mirimage_cal.fits``, ``*_mirimage_align.fits``,
``*_mirimage_o<obs>_crf.fits``) of a NIRCam visit are rotated about that
visit's NIRCam pivot.  The roll is an attitude error of the whole
observatory, so one rotation about one point corrects every instrument in
the visit; rotating MIRI about its own center would instead add a
translation of roll x separation (~40 mas at ~7' for an 18" roll).  The
MIRI parallels of 10678 are the case this exists for.  Rotating about the
NIRCam pivot leaves a translation at MIRI (roll x separation); a MIRI bulk
offset (issue #956) absorbs it only if that offset is fit AFTER the roll,
so fit MIRI bulk offsets on rolled frames, never before.

Provenance, idempotency, reversibility
--------------------------------------
Every product gets ``ROLLCORR=T`` with ``ROLLMODE='posthoc-visit-pivot'``,
``ROLLARC``, ``ROLLPVRA``/``ROLLPVDE``, the table sha and the date.  A
product that already carries ``ROLLCORR`` is skipped (this module,
``roll_correction.apply_roll_correction`` and the default-on
``roll_correction.ensure_roll_correction`` in ``fix_alignment`` all honor the
marker, so a frame is never rotated twice; ``ensure_roll_correction`` applies
only the difference when the table value has changed since).  An observation
whose locked offsets table predates the roll
(``alignment_config.roll_blocked_by_locked_table``) is reported
``locked-table`` and left unrolled, as ``fix_alignment`` leaves it.  The
original WCS cards are kept in a header-only ``ROLLWBAK`` extension and the
original GWCS under ``meta.roll_backup_wcs``; :func:`restore_product`
reverts exactly.  Writes go to a temporary file in the same directory and
``os.replace`` it over the original, so a crash leaves the original intact.
Symlinks are refused.

Interaction with the rest of the pipeline
-----------------------------------------
* ``fix_alignment`` skips a frame with ``RAOFFSET``; ``RAOFFSET`` is not
  changed, so the table-disagreement guard is unaffected.
* The offsets table stays valid: rotation about the visit coverage centroid
  preserves the translation at that centroid (the point the consensus was
  tied at) to first order.
* Catalogs record ``ROLLCORR`` at fit time (``crowdsource_catalogs_long``
  provenance stamps), which is what the catalog correction reads to refuse a
  born-rotated catalog.
"""
import copy
import glob
import os
import re
import subprocess
import time

import numpy as np
from astropy.io import fits

from jwst_gc_pipeline.reduction.alignment_config import roll_blocked_by_locked_table
from jwst_gc_pipeline.reduction.roll_correction import (
    MARKER, PENDING, rotation_about_pivot, roll_adjusted_wcs, verify_rotation,
    _tangent, _untangent)

__all__ = ['ImageRollError', 'rotate_frame', 'rotate_i2d', 'restore_product',
           'frame_pivot', 'fit_tan_similarity', 'in_flight_observations',
           'classify_image', 'MODE']

MODE = 'posthoc-visit-pivot'
BACKUP_EXT = 'ROLLWBAK'
RAD2AS = 206264.80624709636

_WCS_KEY_RE = re.compile(
    r'^(WCSAXES|CTYPE\d|CUNIT\d|CRVAL\d|CRPIX\d|CDELT\d|CD\d_\d|PC\d_\d|LONPOLE|LATPOLE|'
    r'RADESYS|EQUINOX|MJDREF|A_\w+|B_\w+|AP_\w+|BP_\w+|S_REGION|SIPGWMAX)$')
_ROLL_KEYS = (MARKER, PENDING, 'ROLLMODE', 'ROLLARC', 'ROLLPVRA', 'ROLLPVDE', 'ROLLSHRA',
              'ROLLSHDE', 'ROLLVMAS', 'ROLLTAB', 'ROLLDATE', 'ROLLVIS', 'ROLLIRES', 'ROLLIMED',
              # roll_correction.ensure_roll_correction delta re-corrections
              'ROLLPVOR', 'ROLLPVOD', 'ROLLPREV', 'ROLLDVMA', 'ROLLDDAT', 'ROLLNDLT')


class ImageRollError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# classification
# ----------------------------------------------------------------------------

_FRAME_RE = re.compile(
    r'^jw(\d{5})(\d{3})(\d{3})_\d{5}_\d{5}_(nrc[a-z0-9]+|mirimage)_(.+)\.fits$')


def classify_image(basename):
    """'cal' | 'frame' | 'i2d-primary' | 'i2d-derived' | 'skip' for a basename."""
    b = basename.lower()
    if 'badastrom' in b or b.endswith(('.tmp.fits', '.rolltmp.fits')):
        return 'skip'
    m = _FRAME_RE.match(b)
    if m:
        det, rest = m.group(4), m.group(5)
        if rest == 'cal':
            return 'cal'
        if rest == 'i2d':
            return 'skip'                 # MAST stage-2 per-exposure i2d: unaligned
        # MIRI's crf carries no destreak/align stem (*_mirimage_o<obs>_crf).
        # A stemless NIRCam crf is a stale pre-destreak product (brick 2221,
        # cloudc 2022/23): left alone.
        crf = r'(destreak|align)_o\d{3}_crf' if det != 'mirimage' \
            else r'((destreak|align)_)?o\d{3}_crf'
        if rest in ('destreak', 'align') or re.fullmatch(crf, rest):
            return 'frame'
        return 'skip'                     # satstar/wingcal/model/residual per-frame products
    if b.startswith('jw') and b.endswith('_i2d.fits') and '_nircam_' in b:
        if re.search(r'-(merged|nrca|nrcb|nrcalong|nrcblong)(_data)?_i2d\.fits$', b):
            return 'i2d-primary'
        return 'i2d-derived'
    if re.fullmatch(r'jw\d{5}-o\d{3}_t\d+_miri_f\d+w_i2d\.fits', b):
        return 'i2d-primary'
    return 'skip'


def _prog_obs_from_name(basename):
    b = basename.lower()
    m = _FRAME_RE.match(b)
    if m:
        return m.group(1).lstrip('0') or '0', m.group(2), m.group(3)
    m = re.match(r'^jw(\d{5})-o(\d{3})_', b)
    if m:
        return m.group(1).lstrip('0') or '0', m.group(2), None
    return None


# ----------------------------------------------------------------------------
# in-flight guard
# ----------------------------------------------------------------------------

def _own_job_ids():
    """SLURM ids naming the job this process runs in (empty off-SLURM)."""
    return {os.environ[k] for k in ('SLURM_JOB_ID', 'SLURM_ARRAY_JOB_ID')
            if os.environ.get(k)}


def in_flight_observations(user=None):
    """(busy, names): observations referenced by queued/running SLURM jobs.

    ``busy`` holds ``(program, obs)`` for names like ``brick2221-o001-cat``
    or ``gc-treasury10678-o063`` and ``(None, obs)`` for names with an obs
    token and no program (``1pass_extract_gc-treasury_F212N_o066``).  A
    product whose observation matches either form is never written.  squeue
    failure raises: the guard must not silently pass.

    The job running this process is left out.  Its name carries the field
    (``quintuplet-rollwcs-apply-pilot``), so counting it marked every product
    of that field busy and the apply wrote nothing.
    """
    user = user or os.environ.get('USER')
    out = subprocess.run(['squeue', '-h', '-u', user, '-o', '%i %j'], check=True,
                         capture_output=True, text=True, timeout=120).stdout
    own = _own_job_ids()
    names = []
    for line in out.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) != 2:
            continue
        jobid, name = parts
        if jobid.split('_')[0] in own:
            continue
        names.append(name.strip())
    busy = set()
    for n in names:
        for m in re.finditer(r'(?:(\d{4,5})-)?o(\d{3})(?![0-9])', n):
            prog = m.group(1).lstrip('0') if m.group(1) else None
            busy.add((prog, m.group(2)))
    return busy, names


_STAGE_TOKEN_RE = re.compile(r'(?:^|[-_])(?:m\d+|reduce|regen|cat|catalog|align|fanout|finalize)'
                             r'(?:$|[-_])', re.I)


def _tokenless_10678_stage(names):
    """True if a 10678 pipeline-stage job carries no observation token.

    Such a job (``gc-treasury10678-regen``) may touch any tile, so every
    10678 observation counts as busy.  Names without a stage token (the
    hourly HiPS/cron jobs) do not, or the guard would never clear.
    """
    for n in names:
        if '10678' not in n or re.search(r'o\d{3}(?![0-9])', n):
            continue
        if _STAGE_TOKEN_RE.search(n):
            return True
    return False


def is_busy(program, obs, field_name, busy, names):
    """True if a job touches this observation or names this field."""
    prog = str(program).lstrip('0')
    if (prog, obs) in busy or (None, obs) in busy:
        return True
    if prog == '10678':
        return _tokenless_10678_stage(names)
    f = field_name.replace('_', '').replace('-', '').lower()
    return any(f in n.replace('_', '').replace('-', '').lower() for n in names)


# ----------------------------------------------------------------------------
# pivots
# ----------------------------------------------------------------------------

def frame_pivot(visit_pivot, sci_header, lineage_offset=None):
    """Pivot to rotate THIS product about.

    Aligned products (``RAOFFSET`` present) use the visit pivot.  An
    unaligned ``_cal`` uses the visit pivot moved back by the aligned
    lineage's (RAOFFSET, DEOFFSET) [arcsec, coordinate convention], so a
    regeneration (rotate-then-shift) reproduces the rotated aligned frame
    (shift-then-rotate).  ``lineage_offset`` = (raoff, deoff) of that lineage.
    """
    ra_p, dec_p = visit_pivot
    if 'RAOFFSET' in sci_header or lineage_offset is None:
        return float(ra_p), float(dec_p)
    raoff, deoff = lineage_offset
    return float(ra_p - raoff / 3600.0), float(dec_p - deoff / 3600.0)


def _lineage_offset(cal_path):
    """(RAOFFSET, DEOFFSET) of the aligned product derived from this _cal."""
    for suffix in ('_destreak.fits', '_align.fits'):
        p = cal_path[:-len('_cal.fits')] + suffix
        if os.path.exists(p):
            h = fits.getheader(p, ('SCI', 1))
            if 'RAOFFSET' in h:
                return float(h['RAOFFSET']), float(h['DEOFFSET'])
    return None


# ----------------------------------------------------------------------------
# write helpers
# ----------------------------------------------------------------------------

def _wcs_backup_hdu(sci_header, fn):
    bak = fits.Header()
    # the backup belongs to THIS file: destreak copies every HDU of a _cal
    # into the _destreak it writes, and restoring the cal's WCS onto a
    # shifted _destreak would drop its alignment
    bak['ROLLBKFN'] = _card(os.path.basename(fn)[:68], 'file this WCS backup was taken from')
    for k, v in sci_header.items():
        if _WCS_KEY_RE.match(k):
            bak[k] = v
    return fits.ImageHDU(header=bak, name=BACKUP_EXT)


def _rotate_s_region(s_region, pivot, roll_arcsec, fn=None):
    if not s_region:
        return s_region
    toks = s_region.split()
    nums = []
    for t in toks:
        try:
            nums.append(float(t))
        except ValueError:
            continue
    if len(nums) < 6 or len(nums) % 2:
        return s_region
    ra, dec = np.array(nums[0::2]), np.array(nums[1::2])
    if fn is not None:
        r2, d2 = fn(ra, dec)
    else:
        r2, d2 = rotation_about_pivot(ra, dec, pivot[0], pivot[1], roll_arcsec)
    return 'POLYGON ICRS  ' + ' '.join(f'{a:.9f} {b:.9f}' for a, b in zip(r2, d2))


def _card(v, c):
    """(value, comment), dropping a comment that would not fit one 80-char card."""
    vlen = max(20, len(str(v)) + 2) if isinstance(v, str) else 20
    return (v, c) if 10 + vlen + 3 + len(c) <= 80 else (v, '')


def _stamp(h, roll, pivot, table_sha, visit_key, extra=()):
    h[MARKER] = (True, 'field roll correction applied (data-qa#346)')
    h['ROLLMODE'] = (MODE, 'rotated after alignment about visit pivot')
    h['ROLLARC'] = (float(roll), '[arcsec] roll applied, +N->E')
    h['ROLLPVRA'] = (float(pivot[0]), '[deg] roll pivot RA')
    h['ROLLPVDE'] = (float(pivot[1]), '[deg] roll pivot Dec')
    # the shift this product carried when the pivot was taken, so a later
    # roll_correction.ensure_roll_correction delta can follow any shift since
    h['ROLLPVOR'] = (float(h.get('RAOFFSET', 0.0)), '[arcsec] RAOFFSET when pivot taken')
    h['ROLLPVOD'] = (float(h.get('DEOFFSET', 0.0)), '[arcsec] DEOFFSET when pivot taken')
    h['ROLLVIS'] = _card(str(visit_key)[:68], 'program-obs-visit of the roll')
    h['ROLLTAB'] = _card(f'sha1:{table_sha}'[:68], 'roll_corrections.csv')
    h['ROLLDATE'] = (time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'applied UTC')
    for k, v, c in extra:
        h[k] = _card(v, c)
    if PENDING in h:
        del h[PENDING]


def _replace_atomically(tmp, fn):
    st = os.stat(fn)
    os.chmod(tmp, st.st_mode & 0o7777)
    os.replace(tmp, fn)


RELEASE_ROOTS = ('/orange/adamginsburg/jwst/releases',)
_RELEASE_TARGETS = {}


def release_targets(roots=RELEASE_ROOTS):
    """Realpaths of every symlink under the release trees (built once per roots).

    A release links live pipeline frames; rotating such a target in place
    would change a published product under its readers.
    """
    roots = tuple(roots)
    if roots not in _RELEASE_TARGETS:
        out = set()
        for root in roots:
            for d, dirs, files in os.walk(root):
                for n in dirs + files:
                    q = os.path.join(d, n)
                    if os.path.islink(q):
                        out.add(os.path.realpath(q))
        _RELEASE_TARGETS[roots] = out
    return _RELEASE_TARGETS[roots]


def _check_writable(fn, release_roots=RELEASE_ROOTS):
    from jwst_gc_pipeline.astrometry.catalog_roll_correction import BLUE_JWST
    if os.path.islink(fn):
        raise ImageRollError(f"{fn}: symlink; refusing (rotate the target instead)")
    if not os.path.isfile(fn):
        raise ImageRollError(f"{fn}: not a regular file")
    if os.path.abspath(fn).startswith(BLUE_JWST):
        raise ImageRollError(f"{fn}: under {BLUE_JWST}; address fields via /orange (#937)")
    if os.path.realpath(fn) in release_targets(release_roots):
        raise ImageRollError(f"{fn}: target of a release symlink; refusing")


# ----------------------------------------------------------------------------
# frames
# ----------------------------------------------------------------------------

def rotate_frame(fn, roll_arcsec, visit_pivot, *, visit_key='', table_sha='',
                 dry_run=False, tol_mas=0.5):
    """Rotate one detector frame's GWCS + SIP header about the visit pivot.

    Returns a dict(status=..., ...).  ``status`` is 'already' when the
    product carries ROLLCORR, 'dry-run' or 'rotated'.
    """
    from jwst.datamodels import ImageModel
    from jwst_gc_pipeline.reduction.fits_wcs_sync import sync_header_to_gwcs
    _check_writable(fn)
    sci = fits.getheader(fn, ('SCI', 1))
    if sci.get(MARKER):
        return dict(status='already', file=fn, mode=sci.get('ROLLMODE'))
    if sci.get(PENDING):
        raise ImageRollError(f"{fn}: carries {PENDING}; a previous apply crashed. "
                             f"Refusing to guess the GWCS state.")
    lineage = None
    if 'RAOFFSET' not in sci and fn.endswith('_cal.fits'):
        lineage = _lineage_offset(fn)
    pivot = frame_pivot(visit_pivot, sci, lineage)
    t0 = time.time()
    fa = ImageModel(fn)
    w0 = fa.meta.wcs
    w1, (dra, ddec), piv = roll_adjusted_wcs(w0, roll_arcsec, pivot=pivot)
    worst = verify_rotation(w0, w1, fa.data.shape, piv, roll_arcsec, tol_mas=tol_mas)
    if dry_run:
        fa.close()
        return dict(status='dry-run', file=fn, verify_mas=worst, pivot=piv,
                    lineage_offset=lineage, seconds=round(time.time() - t0, 2))
    fname_card = fits.getheader(fn, 0).get('FILENAME')
    shape = fa.data.shape
    fa.meta.wcs = w1
    fa.meta.roll_backup_wcs = copy.deepcopy(w0)
    tmp1 = fn + '.rolltmp.fits'
    tmp2 = fn + '.rolltmp2.fits'
    try:
        fa.save(tmp1, overwrite=True)
        fa.close()
        with fits.open(tmp1) as hl:
            h = hl['SCI'].header
            sip_max, _sip_med = sync_header_to_gwcs(h, w1, shape,
                                                    label=os.path.basename(fn))
            h['SIPGWMAX'] = (sip_max, '[mas] max FITS/SIP vs GWCS disagreement')
            if 'S_REGION' in h:
                h['S_REGION'] = _rotate_s_region(h['S_REGION'], piv, roll_arcsec)
            _stamp(h, roll_arcsec, piv, table_sha, visit_key,
                   extra=(('ROLLSHRA', dra, '[deg] ref-point RA coord shift of the roll'),
                          ('ROLLSHDE', ddec, '[deg] ref-point Dec shift of the roll'),
                          ('ROLLVMAS', worst, '[mas] max deviation from pure rotation')))
            if fname_card is not None:
                hl[0].header['FILENAME'] = fname_card
            if BACKUP_EXT in [x.name for x in hl]:
                raise ImageRollError(f"{fn}: already has a {BACKUP_EXT} extension")
            hl.append(_wcs_backup_hdu(sci, fn))
            hl.writeto(tmp2, overwrite=True)
        _replace_atomically(tmp2, fn)
    finally:
        for p in (tmp1, tmp2):
            if os.path.exists(p):
                os.remove(p)
    return dict(status='rotated', file=fn, verify_mas=worst, sip_max_mas=sip_max, pivot=piv,
                lineage_offset=lineage, seconds=round(time.time() - t0, 2))


# ----------------------------------------------------------------------------
# i2d mosaics
# ----------------------------------------------------------------------------

def _pc_from_header(h):
    if 'CD1_1' in h:
        cd = np.array([[h['CD1_1'], h.get('CD1_2', 0.0)], [h.get('CD2_1', 0.0), h['CD2_2']]])
        return None, cd
    pc = np.array([[h.get('PC1_1', 1.0), h.get('PC1_2', 0.0)],
                   [h.get('PC2_1', 0.0), h.get('PC2_2', 1.0)]])
    return pc, None


def _rot2(phi):
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[c, -s], [s, c]])


def fit_tan_similarity(header, target_fn, shape, npts=33, mask=None):
    """New (CRVAL, PC-or-CD) for a plain TAN header whose sky positions move
    by ``target_fn(ra, dec) -> (ra', dec')``.

    The target displacement is fit, in the tangent plane at CRVAL, by a
    rotation + translation; CRVAL moves by the translation and the linear
    matrix rotates.  The sign of the matrix rotation is not assumed: both
    are evaluated and the one that reproduces the target is kept.  Returns
    dict(crval, pc|cd, max_mas, median_mas, roll_arcsec).
    """
    from astropy.wcs import WCS
    ny, nx = shape
    ys = np.linspace(0, ny - 1, npts)
    xs = np.linspace(0, nx - 1, npts)
    xx, yy = np.meshgrid(xs, ys)
    xx, yy = xx.ravel(), yy.ravel()
    if mask is not None:
        keep = mask(xx, yy)
        xx, yy = xx[keep], yy[keep]
    w0 = WCS(header, relax=True)
    ra0, dec0 = w0.pixel_to_world_values(xx, yy)
    ra1, dec1 = target_fn(ra0, dec0)
    c0 = (float(header['CRVAL1']), float(header['CRVAL2']))
    x0, y0 = _tangent(ra0, dec0, *c0)
    x1, y1 = _tangent(ra1, dec1, *c0)
    dx, dy = x1 - x0, y1 - y0
    n = x0.size
    A = np.zeros((2 * n, 3))
    A[:n, 0], A[n:, 0] = y0, -x0
    A[:n, 1] = 1.0
    A[n:, 2] = 1.0
    sol, *_ = np.linalg.lstsq(A, np.concatenate([dx, dy]), rcond=None)
    roll, tx, ty = sol
    cr_ra, cr_dec = _untangent(np.array([tx]), np.array([ty]), *c0)
    # exact for a pure rotation: use the true image of CRVAL, not the linearized one
    best = None
    pc, cd = _pc_from_header(header)
    for sign in (+1, -1):
        h2 = header.copy()
        h2['CRVAL1'], h2['CRVAL2'] = float(cr_ra[0]), float(cr_dec[0])
        R = _rot2(sign * roll)
        if cd is not None:
            new = R @ cd
            h2['CD1_1'], h2['CD1_2'], h2['CD2_1'], h2['CD2_2'] = new.ravel()
        else:
            new = R @ pc
            h2['PC1_1'], h2['PC1_2'], h2['PC2_1'], h2['PC2_2'] = new.ravel()
        w2 = WCS(h2, relax=True)
        ra2, dec2 = w2.pixel_to_world_values(xx, yy)
        dev = np.hypot(((ra2 - ra1 + 180) % 360 - 180) * np.cos(np.radians(dec1)),
                       dec2 - dec1) * 3.6e6
        cand = dict(crval=(float(cr_ra[0]), float(cr_dec[0])), matrix=new,
                    kind='cd' if cd is not None else 'pc', max_mas=float(np.max(dev)),
                    median_mas=float(np.median(dev)), roll_arcsec=float(roll * RAD2AS),
                    sign=sign)
        if best is None or cand['max_mas'] < best['max_mas']:
            best = cand
    return best


def _apply_matrix(h, fit):
    h['CRVAL1'], h['CRVAL2'] = fit['crval']
    if fit['kind'] == 'cd':
        h['CD1_1'], h['CD1_2'], h['CD2_1'], h['CD2_2'] = fit['matrix'].ravel()
    else:
        h['PC1_1'], h['PC1_2'], h['PC2_1'], h['PC2_2'] = fit['matrix'].ravel()


def _i2d_gwcs_with(w0, crval, pc):
    """Copy of a one-step i2d GWCS with a NEW FITSImagingWCSTransform.

    The transform's ``forward`` chain is built from the parameter values at
    construction; assigning ``crval``/``pc`` afterwards changes the
    parameters but not the evaluated mapping, so the transform is rebuilt.
    """
    from gwcs.fitswcs import FITSImagingWCSTransform
    t = w0.forward_transform
    nt = FITSImagingWCSTransform(copy.deepcopy(t.projection),
                                 crpix=[float(v) for v in t.crpix.value],
                                 crval=[float(v) for v in crval],
                                 cdelt=[float(v) for v in t.cdelt.value],
                                 pc=np.asarray(pc, float))
    w1 = copy.deepcopy(w0)
    names = w1.available_frames
    if len(names) != 2:
        raise ImageRollError(f"i2d GWCS has frames {names}; expected detector -> world")
    bbox = None
    try:
        bbox = w0.bounding_box
    except NotImplementedError:
        bbox = None
    w1.set_transform(names[0], names[1], nt)
    if bbox is not None:
        w1.bounding_box = tuple(tuple(b) for b in bbox.bounding_box(order='F')) \
            if hasattr(bbox, 'bounding_box') else bbox
    return w1


def rotate_i2d(fn, model, *, visit_keys=(), table_sha='', dry_run=False, tol_mas=0.05,
               approx_tol_mas=3.0, allow_approx=False):
    """Header/GWCS-only rotation of an i2d mosaic.

    ``model`` is a ``catalog_roll_correction.PointingModel`` for the
    observation (all its visits).  Single-visit models must close to
    ``tol_mas``; multi-visit models are approximated by one similarity and
    written only with ``allow_approx`` and a residual under
    ``approx_tol_mas``.
    """
    from jwst.datamodels import ImageModel
    _check_writable(fn)
    sci = fits.getheader(fn, ('SCI', 1))
    if sci.get(MARKER):
        return dict(status='already', file=fn)
    if 'A_ORDER' in sci or sci.get('CTYPE1', '').endswith('-SIP'):
        raise ImageRollError(f"{fn}: i2d with SIP distortion; expected a plain TAN grid")
    shape = (int(sci['NAXIS2']), int(sci['NAXIS1']))
    t0 = time.time()
    single = len(model.visits) == 1
    fit = fit_tan_similarity(sci, model.apply, shape)
    if single and fit['max_mas'] > tol_mas:
        raise ImageRollError(f"{fn}: single-roll header update misses the rotation by "
                             f"{fit['max_mas']:.4f} mas (> {tol_mas})")
    if not single:
        if fit['max_mas'] > approx_tol_mas:
            raise ImageRollError(f"{fn}: multi-visit residual {fit['max_mas']:.2f} mas "
                                 f"> {approx_tol_mas} mas")
        if not allow_approx and not dry_run:
            raise ImageRollError(f"{fn}: multi-visit mosaic ({len(model.visits)} visits); "
                                 f"best single similarity leaves {fit['max_mas']:.2f} mas "
                                 f"(median {fit['median_mas']:.2f}); pass allow_approx")
    res = dict(file=fn, single=single, max_mas=fit['max_mas'], median_mas=fit['median_mas'],
               roll_fit_arcsec=fit['roll_arcsec'])
    if dry_run:
        res.update(status='dry-run', seconds=round(time.time() - t0, 2))
        return res

    m = ImageModel(fn)
    w0 = m.meta.wcs
    t = w0.forward_transform
    if type(t).__name__ != 'FITSImagingWCSTransform':
        raise ImageRollError(f"{fn}: GWCS forward transform is {type(t).__name__}; "
                             f"expected FITSImagingWCSTransform")
    pc_h, cd_h = _pc_from_header(sci)
    if cd_h is not None:
        raise ImageRollError(f"{fn}: CD-matrix i2d; GWCS parameter update not implemented")
    # gwcs pc/crval must equal the header's before the update
    if not (np.allclose(t.crval.value, [sci['CRVAL1'], sci['CRVAL2']], rtol=0, atol=1e-10)
            and np.allclose(t.pc.value, pc_h, rtol=0, atol=1e-12)):
        raise ImageRollError(f"{fn}: GWCS and FITS header disagree before the update")
    w1 = _i2d_gwcs_with(w0, fit['crval'], fit['matrix'])
    # verify the GWCS against the target model on a grid
    ny, nx = shape
    yy, xx = np.mgrid[0:ny:max(ny // 16, 1), 0:nx:max(nx // 16, 1)]
    ra0, dec0 = w0(xx.ravel().astype(float), yy.ravel().astype(float))
    ra1, dec1 = w1(xx.ravel().astype(float), yy.ravel().astype(float))
    rt, dt = model.apply(ra0, dec0)
    dev = np.hypot(((ra1 - rt + 180) % 360 - 180) * np.cos(np.radians(dt)), dec1 - dt) * 3.6e6
    gmax = float(np.nanmax(dev))
    lim = tol_mas if single else approx_tol_mas
    if not gmax <= lim:
        raise ImageRollError(f"{fn}: updated GWCS deviates from the roll model by {gmax:.3f} mas")
    m.meta.roll_backup_wcs = copy.deepcopy(w0)
    m.meta.wcs = w1
    wi = m.meta.wcsinfo
    wi.crval1, wi.crval2 = fit['crval']
    wi.pc1_1, wi.pc1_2, wi.pc2_1, wi.pc2_2 = [float(v) for v in fit['matrix'].ravel()]
    if wi.s_region:
        wi.s_region = _rotate_s_region(wi.s_region, None, None, fn=model.apply)
    fname_card = fits.getheader(fn, 0).get('FILENAME')
    tmp1, tmp2 = fn + '.rolltmp.fits', fn + '.rolltmp2.fits'
    try:
        m.save(tmp1, overwrite=True)
        m.close()
        with fits.open(tmp1) as hl:
            h = hl['SCI'].header
            _apply_matrix(h, fit)
            if 'S_REGION' in h and wi.s_region:
                h['S_REGION'] = wi.s_region
            piv = (model.visits[0].pivot_ra, model.visits[0].pivot_dec)
            _stamp(h, model.visits[0].roll_arcsec if single else fit['roll_arcsec'], piv,
                   table_sha, ','.join(visit_keys) or model.visits[0].key,
                   extra=(('ROLLIRES', fit['max_mas'], '[mas] max header-vs-roll-model resid'),
                          ('ROLLIMED', fit['median_mas'], '[mas] median resid'),
                          ('ROLLVMAS', gmax, '[mas] GWCS vs roll model, max')))
            if fname_card is not None:
                hl[0].header['FILENAME'] = fname_card
            if BACKUP_EXT in [x.name for x in hl]:
                raise ImageRollError(f"{fn}: already has a {BACKUP_EXT} extension")
            hl.append(_wcs_backup_hdu(sci, fn))
            hl.writeto(tmp2, overwrite=True)
        _replace_atomically(tmp2, fn)
    finally:
        for p in (tmp1, tmp2):
            if os.path.exists(p):
                os.remove(p)
    res.update(status='rotated', gwcs_max_mas=gmax, seconds=round(time.time() - t0, 2))
    return res


# ----------------------------------------------------------------------------
# rollback
# ----------------------------------------------------------------------------

def restore_product(fn):
    """Undo :func:`rotate_frame` / :func:`rotate_i2d` exactly."""
    from jwst.datamodels import ImageModel
    _check_writable(fn)
    sci = fits.getheader(fn, ('SCI', 1))
    if not sci.get(MARKER) or sci.get('ROLLMODE') != MODE:
        return dict(status='not-rotated', file=fn)
    with fits.open(fn) as hl:
        names = [x.name for x in hl]
        if BACKUP_EXT not in names:
            raise ImageRollError(f"{fn}: rotated but no {BACKUP_EXT} backup; cannot restore")
        bak = hl[BACKUP_EXT].header.copy()
    if bak.get('ROLLBKFN') != os.path.basename(fn)[:68]:
        raise ImageRollError(f"{fn}: {BACKUP_EXT} was taken from {bak.get('ROLLBKFN')!r} "
                             f"(inherited, e.g. _cal -> _destreak); restore the source "
                             f"and regenerate this product instead")
    m = ImageModel(fn)
    back = getattr(m.meta, 'roll_backup_wcs', None)
    if back is None:
        raise ImageRollError(f"{fn}: rotated but no meta.roll_backup_wcs; cannot restore")
    m.meta.wcs = back
    try:
        del m.meta.roll_backup_wcs
    except AttributeError:
        m.meta.roll_backup_wcs = None
    for k in ('crval1', 'crval2', 'pc1_1', 'pc1_2', 'pc2_1', 'pc2_2', 's_region'):
        card = k.upper()
        if card in bak and getattr(m.meta.wcsinfo, k, None) is not None:
            setattr(m.meta.wcsinfo, k, bak[card])
    fname_card = fits.getheader(fn, 0).get('FILENAME')
    tmp1, tmp2 = fn + '.rolltmp.fits', fn + '.rolltmp2.fits'
    try:
        m.save(tmp1, overwrite=True)
        m.close()
        with fits.open(tmp1) as hl:
            h = hl['SCI'].header
            for k in list(h.keys()):
                if _WCS_KEY_RE.match(k) and k not in bak:
                    del h[k]
            for k, v in bak.items():
                if _WCS_KEY_RE.match(k):
                    h[k] = v
            for k in _ROLL_KEYS:
                if k in h:
                    del h[k]
            if fname_card is not None:
                hl[0].header['FILENAME'] = fname_card
            keep = fits.HDUList([x for x in hl if x.name != BACKUP_EXT])
            keep.writeto(tmp2, overwrite=True)
        _replace_atomically(tmp2, fn)
    finally:
        for p in (tmp1, tmp2):
            if os.path.exists(p):
                os.remove(p)
    return dict(status='restored', file=fn)


# ----------------------------------------------------------------------------
# enumeration + CLI
# ----------------------------------------------------------------------------

def enumerate_images(field_name, include_derived=False, filters=None, obs=None):
    """{class: [paths]} of a field's NIRCam + MIRI image products (read-only listing).

    ``obs`` (iterable of 3-digit ids) narrows the globs, which matters on
    gc-treasury, whose pipeline directories hold >10^5 files.
    """
    from jwst_gc_pipeline.astrometry.catalog_roll_correction import field_root
    base = field_root(field_name).rstrip('/')
    out = {}
    pats = ['jw*.fits'] if not obs else \
        [p for o in sorted(obs) for p in (f'jw?????{o}???_*.fits', f'jw?????-o{o}_*.fits')]
    for d in sorted(glob.glob(os.path.join(base, 'F*', 'pipeline'))):
        filt = d.split(os.sep)[-2]
        if filters and filt.upper() not in filters:
            continue
        for p in sorted(q for pat in pats for q in glob.glob(os.path.join(d, pat))):
            c = classify_image(os.path.basename(p))
            if c == 'skip' or (c == 'i2d-derived' and not include_derived):
                continue
            out.setdefault(c, []).append(p)
    return out


def main(argv=None):
    import argparse
    import json
    from jwst_gc_pipeline.astrometry import catalog_roll_correction as crc
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--field', required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--restore', action='store_true')
    ap.add_argument('--obs', help='comma list of observations to include')
    ap.add_argument('--filters', help='comma list of filters, e.g. F212N,F480M')
    ap.add_argument('--kinds', default='frame,cal,i2d-primary',
                    help='comma list of frame,cal,i2d-primary,i2d-derived')
    ap.add_argument('--file', action='append', help='only these files (e.g. COPIES)')
    ap.add_argument('--allow-approx-i2d', action='store_true',
                    help='write multi-visit i2d with the best single similarity')
    ap.add_argument('--i2d-approx-tol-mas', type=float, default=3.0)
    ap.add_argument('--footprint-cache')
    ap.add_argument('--manifest')
    ap.add_argument('--verify-sample', type=int, default=0,
                    help='dry-run: build+verify the rotated GWCS on N files per kind')
    ap.add_argument('--no-queue-check', action='store_true',
                    help='skip the squeue in-flight guard (COPIES only; refused under the field roots)')
    args = ap.parse_args(argv)

    kinds = set(args.kinds.split(','))
    obs_set = set(args.obs.split(',')) if args.obs else None
    filters = set(f.upper() for f in args.filters.split(',')) if args.filters else None
    writing = args.apply or args.restore
    field_roots = (crc.ORANGE_JWST, crc.BLUE_JWST)
    if writing and args.no_queue_check and (not args.file or any(
            q.startswith(field_roots) for p in args.file
            for q in (os.path.abspath(p), os.path.realpath(p)))):
        ap.error('--no-queue-check is only for explicit --file copies outside the field roots')
    base = crc.field_root(args.field).rstrip('/')
    busy_obs, names = (set(), [])
    if writing and not args.no_queue_check:
        busy_obs, names = in_flight_observations()
    if args.file:
        listing = {}
        for p in args.file:
            listing.setdefault(classify_image(os.path.basename(p)), []).append(p)
    else:
        listing = enumerate_images(args.field, include_derived='i2d-derived' in kinds,
                                   filters=filters, obs=obs_set)
    only = {o for o in obs_set} if obs_set else None
    models = crc.build_models_for_field(args.field, footprint_cache=args.footprint_cache,
                                        only_obs=only)
    tsha = crc.table_sha()
    manifest = dict(field=args.field, mode='apply' if args.apply else
                    ('restore' if args.restore else 'dry-run'), files=[], counts={})
    verified = {}
    for kind, paths in sorted(listing.items()):
        if kind not in kinds:
            continue
        for p in paths:
            po = _prog_obs_from_name(os.path.basename(p))
            rec = dict(path=p, kind=kind, size_gb=round(os.path.getsize(p) / 1e9, 3))
            if po is None:
                rec['status'] = 'unattributed'
            else:
                prog, obs, visit = po
                if obs_set and obs not in obs_set:
                    continue
                key = (prog, obs)
                if key not in models:
                    rec['status'] = 'no-roll-row'
                elif not args.restore and roll_blocked_by_locked_table(prog, obs):
                    rec['status'] = 'locked-table'
                elif writing and is_busy(prog, obs, args.field, busy_obs, names):
                    rec['status'] = 'in-flight'
                else:
                    vrs = models[key]
                    try:
                        if args.restore:
                            rec.update(restore_product(p))
                        elif kind in ('frame', 'cal'):
                            vr = next((v for v in vrs if v.visit == visit), None)
                            if vr is None:
                                rec['status'] = 'no-visit-row'
                            else:
                                do_verify = args.apply or verified.get(kind, 0) < args.verify_sample
                                if do_verify:
                                    verified[kind] = verified.get(kind, 0) + 1
                                    rec.update(rotate_frame(p, vr.roll_arcsec,
                                                            (vr.pivot_ra, vr.pivot_dec),
                                                            visit_key=vr.key, table_sha=tsha,
                                                            dry_run=not args.apply))
                                else:
                                    rec['status'] = 'planned'
                        else:
                            model = crc._pivot_models(vrs, None, None)
                            do_verify = args.apply or verified.get(kind, 0) < args.verify_sample
                            if do_verify:
                                verified[kind] = verified.get(kind, 0) + 1
                                rec.update(rotate_i2d(p, model, visit_keys=[v.key for v in vrs],
                                                      table_sha=tsha, dry_run=not args.apply,
                                                      approx_tol_mas=args.i2d_approx_tol_mas,
                                                      allow_approx=args.allow_approx_i2d))
                            else:
                                rec['status'] = 'planned' + ('-approx' if len(vrs) > 1 else '')
                    except (ImageRollError, RuntimeError, OSError, KeyError, ValueError) as ex:
                        rec['status'] = 'refused'
                        rec['reason'] = str(ex)[:300]
            st = rec.get('status', '?')
            c = manifest['counts'].setdefault(f'{kind}:{st}', [0, 0.0])
            c[0] += 1
            c[1] += rec['size_gb']
            manifest['files'].append(rec)
            if st in ('rotated', 'restored', 'refused') or (st == 'dry-run'):
                print(f"  {st:9s} {os.path.basename(p)} "
                      + ' '.join(f"{k}={rec[k]:.3f}" for k in ('verify_mas', 'max_mas',
                                                                'gwcs_max_mas', 'seconds')
                                 if isinstance(rec.get(k), float))
                      + (f"  {rec.get('reason', '')}" if st == 'refused' else ''))
    print(f"\n=== {args.field} ({base})  mode={manifest['mode']}")
    for k, (n, gb) in sorted(manifest['counts'].items()):
        print(f"  {k:32s} {n:6d} files {gb:9.1f} GB")
    if args.manifest:
        with open(args.manifest, 'w') as fh:
            json.dump(manifest, fh, indent=1, default=str)
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
