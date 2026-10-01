"""Catalog-level field-roll correction (JWST-GC/data-qa#346), applied to COPIES.

WHAT THIS IS
------------
``reduction/roll_correction.py`` (PR #1004) corrects the per-visit attitude roll
error measured in data-qa#346 at the IMAGE level: it rotates each frame's GWCS
before the reference shift and then requires the offsets tables to be rebuilt
and every field re-drizzled and re-cataloged.  This module applies the SAME
per-(program, observation, visit) rotation to finished CATALOGS instead, so the
science catalogs can carry the correction without a reprocessing cycle.

It is valid only under conditions this module checks or records:

* The data-qa#346 rotation was measured on m3 per-exposure catalogs built from
  frames that ALREADY carried the offsets-table translation.  So the table value
  is the rotation LEFT OVER after the translation tie.  Rotating a catalog from
  the same generation is therefore not a double correction.  A catalog fit on
  frames that were rolled at the image level is refused: its own header carries
  ``ROLLCORR`` (per-frame catalogs), or it was written after the observation's
  frames were rotated (``frame_roll_state``: sampled crf ``ROLLCORR`` +
  ``ROLLDATE`` vs the catalog mtime).  Catalogs older than the image step are
  corrected normally.
* The m2 bulk tie is a translation fitted over the matched stars of each visit.
  A rotation about the centroid of those stars leaves that tie unchanged to
  first order, so the pivot here is the COVERAGE CENTROID of the visit's frames
  (mean of the per-frame ``S_REGION`` centroids; footprint geometry only, no
  astrometric use of the SIP header).  A pivot offset ``d`` changes the bulk by
  ``d * roll``: 0.1 mas per arcsec of pivot error at a 20" roll.
* Catalog rows carry no visit provenance.  Where two visits of one observation
  with different rolls both cover a row (brick 1182 o004, cloudc 2221 o002),
  the displacement applied is the equal-weight mean of the two visits'
  rotations.  The merged position was an average of the two visits, so this is
  the matching correction to within the unknown per-visit frame weights.
* MIRI is excluded by default.  A MIRI parallel shares the attitude but was tied
  to its reference independently, ~7' from the NIRCam pivot; rotating it about
  the NIRCam pivot would add a ~40 mas translation its own tie already absorbed.
* Pixel columns (x/y) still refer to the UNCORRECTED image WCS.  The output
  header says so (``ROLLPIX``).  Rotated catalogs must never be fed back into
  the pipeline as seeds for a fit on the uncorrected frames: a seed would land
  up to r*roll (~15 mas, 0.5 SW px) off its star.

Outputs go to ``<field>/catalogs_rollcorr/<tag>/`` with the input basename; the
input is never modified.  Every output carries provenance keywords (``ROLLCCAT``
et al.), and a file that already carries them is refused as input.

Usage::

    python -m jwst_gc_pipeline.astrometry.catalog_roll_correction --dry-run
    python -m jwst_gc_pipeline.astrometry.catalog_roll_correction --field brick --apply
    python -m jwst_gc_pipeline.astrometry.catalog_roll_correction --field brick \\
        --file /scratch/copy.fits --out-dir /scratch/out --apply --verify-reference

See JWST-GC/data-qa#346 and keflavich/jwst-gc-pipeline#1004.
"""
import argparse
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field as dc_field

import numpy as np
from astropy.io import fits

from jwst_gc_pipeline.reduction.roll_correction import (
    TABLE as ROLL_TABLE, rotation_about_pivot, _tangent, _untangent)

__all__ = ['VisitRoll', 'PointingModel', 'classify_catalog', 'find_position_pairs',
           'build_models_for_field', 'correct_catalog', 'fit_rotation',
           'enumerate_field', 'main']

DEFAULT_TAG = 'v1_dataqa346'
# rows farther than this beyond a pointing's footprint reach are left unrotated
REACH_MARGIN_ARCSEC = 120.0
OUT_SUBDIR = 'catalogs_rollcorr'
MARKER = 'ROLLCCAT'
MIRI_FILTERS = {'f560w', 'f770w', 'f1000w', 'f1065c', 'f1130w', 'f1140c', 'f1280w',
                'f1500w', 'f1550c', 'f1800w', 'f2100w', 'f2300c', 'f2550w'}
_FILT_RE = re.compile(r'(?<![a-z0-9])(f\d{3,4}[wmnc]2?)(?![a-z0-9])', re.I)
_OBS_RE = re.compile(r'(?<![a-z0-9])o(\d{3}(?:-\d{3})*)(?![0-9])')

#: Products that are pipeline INPUTS in the uncorrected image frame, reference
#: catalogs, or not catalogs at all.  Never rotated.
EXCLUDE_TOKENS = ('_i2dseed', 'consensus', 'crossband_seed', 'satstar', 'refcat',
                  'gaia', 'virac', 'twomass', 'wingcal', 'reference_astrometric',
                  'seed_union', 'starsthatshouldbeexcluded', 'fovp101', 'apcorr',
                  'nirspec_selection', 'accretor_candidates', 'forbrandt')
#: Display products derived from a catalog; regenerate them from the rotated one.
DERIVED_TOKENS = ('_carta',)

_PERBAND_RE = re.compile(
    r'^(?P<filt>f\d{3,4}[wmnc]2?)_(?P<mod>nrc[ab]\w*?|merged|mirimage|nrca|nrcb)'
    r'(?:_o(?P<obs>\d{3}(?:-\d{3})*))?_.*(?:dao_basic|daophot|iterative|crowdsource|'
    r'dao)[^/]*\.(?:fits|ecsv)$', re.I)
_CROSSBAND_RE = re.compile(
    r'^(?:basic|iterative|union)_.*(?:photometry_tables|dao_basic).*\.(?:fits|ecsv)$|'
    r'.*crossband_m\d.*\.(?:fits|ecsv)$|^photometry_tables.*\.(?:fits|ecsv)$', re.I)


# ----------------------------------------------------------------------------
# Pointing model: per-visit rotation about the visit's coverage centroid
# ----------------------------------------------------------------------------

def _unit(ra, dec):
    ra, dec = np.radians(ra), np.radians(dec)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def _mean_sky(ras, decs):
    v = _unit(np.asarray(ras, float), np.asarray(decs, float)).reshape(3, -1).sum(axis=1)
    v /= np.linalg.norm(v)
    return float(np.degrees(np.arctan2(v[1], v[0])) % 360.0), float(np.degrees(np.arcsin(v[2])))


@dataclass
class VisitRoll:
    """One visit's roll (arcsec, +N->E), pivot (deg), and footprint polygons."""
    program: str
    observation: str
    visit: str
    roll_arcsec: float
    pivot_ra: float = np.nan
    pivot_dec: float = np.nan
    pivot_source: str = 'none'
    footprints: list = dc_field(default_factory=list)   # list of (N,2) [ra, dec] deg
    # image-level roll state of this observation's frames (frame_roll_state);
    # read fresh on every run, never cached
    frame_roll: dict = dc_field(default_factory=dict)

    @property
    def key(self):
        return f"{self.program}-{self.observation}-{self.visit}"


class PointingModel:
    """Rotation model for the rows of ONE observation (one or more visits).

    ``apply(ra, dec)`` returns rotated positions.  With a single visit, every row
    is rotated about that visit's pivot.  With several, each row takes the
    equal-weight mean displacement of the visits whose footprint covers it; a
    row covered by none takes the nearest visit's rotation.
    """

    def __init__(self, visits):
        if not visits:
            raise ValueError("PointingModel needs at least one visit")
        self.visits = list(visits)
        for v in self.visits:
            if not (np.isfinite(v.pivot_ra) and np.isfinite(v.pivot_dec)):
                raise ValueError(f"visit {v.key} has no pivot")

    @property
    def key(self):
        v0 = self.visits[0]
        return f"{v0.program}-{v0.observation}"

    @property
    def rolls(self):
        return [v.roll_arcsec for v in self.visits]

    def coverage(self, ra, dec, ra0, dec0):
        """(n_visits, n_rows) bool: which visit footprints contain each row."""
        from matplotlib.path import Path
        x, y = _tangent(ra, dec, ra0, dec0)
        pts = np.column_stack([x, y])
        cov = np.zeros((len(self.visits), len(ra)), dtype=bool)
        for i, v in enumerate(self.visits):
            for poly in v.footprints:
                px, py = _tangent(poly[:, 0], poly[:, 1], ra0, dec0)
                cov[i] |= Path(np.column_stack([px, py])).contains_points(pts)
        return cov

    def within_reach(self, ra, dec, margin_arcsec=REACH_MARGIN_ARCSEC):
        """Bool mask: rows within ``margin_arcsec`` of some visit's footprint reach.

        A visit's reach is the largest pivot-to-vertex distance of its frame
        footprints.  A row beyond every visit's reach + margin was not measured
        in this pointing (e.g. a satstar row imported from another observation)
        and must not be rotated with this pointing's roll.  Visits without
        footprints place no limit.
        """
        ra = np.asarray(ra, float)
        dec = np.asarray(dec, float)
        inside = np.zeros(len(ra), dtype=bool)
        for v in self.visits:
            if not v.footprints:
                return np.ones(len(ra), dtype=bool)
            reach = max(np.max(np.hypot(*_tangent(p[:, 0], p[:, 1], v.pivot_ra, v.pivot_dec)))
                        for p in v.footprints) * 206264.806
            r = np.hypot(*_tangent(ra, dec, v.pivot_ra, v.pivot_dec)) * 206264.806
            inside |= r <= reach + margin_arcsec
        return inside

    def apply(self, ra, dec):
        ra = np.asarray(ra, float)
        dec = np.asarray(dec, float)
        good = np.isfinite(ra) & np.isfinite(dec)
        ra_out, dec_out = ra.copy(), dec.copy()
        if not good.any():
            return ra_out, dec_out
        if len(self.visits) == 1:
            v = self.visits[0]
            ra_out[good], dec_out[good] = rotation_about_pivot(
                ra[good], dec[good], v.pivot_ra, v.pivot_dec, v.roll_arcsec)
            return ra_out, dec_out
        ra0, dec0 = _mean_sky([v.pivot_ra for v in self.visits],
                              [v.pivot_dec for v in self.visits])
        rg, dg = ra[good], dec[good]
        x0, y0 = _tangent(rg, dg, ra0, dec0)
        cov = self.coverage(rg, dg, ra0, dec0)
        none = ~cov.any(axis=0)
        if none.any():
            # nearest pivot for rows outside every footprint
            dist = np.array([np.hypot(*_tangent(rg[none], dg[none], v.pivot_ra, v.pivot_dec))
                             for v in self.visits])
            cov[np.argmin(dist, axis=0), np.flatnonzero(none)] = True
        dx = np.zeros_like(x0)
        dy = np.zeros_like(y0)
        for i, v in enumerate(self.visits):
            sel = cov[i]
            if not sel.any():
                continue
            r2, d2 = rotation_about_pivot(rg[sel], dg[sel], v.pivot_ra, v.pivot_dec,
                                          v.roll_arcsec)
            x2, y2 = _tangent(r2, d2, ra0, dec0)
            dx[sel] += x2 - x0[sel]
            dy[sel] += y2 - y0[sel]
        n = cov.sum(axis=0)
        ra_out[good], dec_out[good] = _untangent(x0 + dx / n, y0 + dy / n, ra0, dec0)
        return ra_out, dec_out

    def describe(self):
        return [dict(visit=v.key, roll_arcsec=v.roll_arcsec, pivot_ra=v.pivot_ra,
                     pivot_dec=v.pivot_dec, pivot_source=v.pivot_source,
                     n_footprints=len(v.footprints)) for v in self.visits]


# ----------------------------------------------------------------------------
# Roll table + footprints
# ----------------------------------------------------------------------------

def _norm3(v):
    s = str(v).strip()
    return s if s == '*' else f"{int(s):03d}"


def read_roll_table(table=ROLL_TABLE):
    """Rows of ``roll_corrections.csv`` as dicts (program, observation, visit, roll)."""
    import csv
    with open(table) as fh:
        rows = list(csv.DictReader(line for line in fh if not line.lstrip().startswith('#')))
    out = []
    for r in rows:
        out.append(dict(program=str(int(r['program'])), observation=_norm3(r['observation']),
                        visit=_norm3(r['visit']), roll_arcsec=float(r['delta_roll_arcsec']),
                        roll_err_arcsec=float(r['delta_roll_err_arcsec']),
                        reference=r['reference'], source=r['source']))
    return out


def table_sha(table=ROLL_TABLE):
    with open(table, 'rb') as fh:
        return hashlib.sha1(fh.read()).hexdigest()[:12]


def _sregion_polygon(s):
    nums = [float(t) for t in str(s).replace(',', ' ').split()
            if re.match(r'^-?\d+(\.\d*)?([eE][-+]?\d+)?$', t)]
    if len(nums) < 6 or len(nums) % 2:
        return None
    return np.array(nums, float).reshape(-1, 2)


def visit_footprints(basepath, program, observation, max_frames_per_visit=400):
    """{visit: [polygons]} from the S_REGION of this observation's NIRCam crf/cal
    frames under ``<basepath>/<FILTER>/pipeline``.  Footprint geometry only
    (arcsec-level), used for the pivot and for per-row visit coverage."""
    pat = os.path.join(basepath, '*', 'pipeline',
                       f"jw{int(program):05d}{observation}[0-9][0-9][0-9]_*_nrc*_crf.fits")
    files = sorted(glob.glob(pat))
    if not files:
        files = sorted(glob.glob(pat.replace('_crf.fits', '_cal.fits')))
    by_visit = {}
    rx = re.compile(rf"jw{int(program):05d}{observation}(\d{{3}})_(\d{{5}})_(\d{{5}})_(nrc\w+?)_")
    seen = set()
    for f in files:
        m = rx.search(os.path.basename(f))
        if not m:
            continue
        visit, grp, exp, det = m.groups()
        # one footprint per (visit, exposure, detector): filters repeat the pointing
        k = (visit, grp, exp, det)
        if k in seen or len(by_visit.get(visit, [])) >= max_frames_per_visit:
            continue
        try:
            hdr = fits.getheader(f, ext=('SCI', 1))
        except (OSError, KeyError):
            continue
        s = hdr.get('S_REGION')
        if s is None:
            try:
                s = fits.getval(f, 'S_REGION', ext=0)
            except (OSError, KeyError):
                continue
        poly = _sregion_polygon(s)
        if poly is None:
            continue
        seen.add(k)
        by_visit.setdefault(visit, []).append(poly)
    return by_visit


def frame_roll_state(basepath, program, observation, sample=8):
    """Image-level roll state of one observation's aligned frames.

    Reads the SCI header of up to ``sample`` evenly spaced crf frames (the
    image step rotates whole observations at once, so a sample represents
    the observation).  Returns dict(n_checked, n_rolled, first_epoch,
    modes): ``first_epoch`` is the earliest time [unix s] a sampled frame
    was rotated -- its ``ROLLDATE`` or, for a frame rotated in-pipeline by
    ``roll_correction.apply_roll_correction`` (no ``ROLLDATE``), its mtime.
    A catalog written after ``first_epoch`` was fit on rotated frames.
    """
    import calendar
    pat = os.path.join(basepath, '*', 'pipeline',
                       f"jw{int(program):05d}{observation}[0-9][0-9][0-9]_*_nrc*_crf.fits")
    files = sorted(glob.glob(pat))
    if len(files) > sample:
        files = [files[i] for i in np.linspace(0, len(files) - 1, sample).astype(int)]
    n_rolled = 0
    epochs = []
    modes = set()
    for f in files:
        try:
            hdr = fits.getheader(f, ext=('SCI', 1))
        except (OSError, KeyError):
            continue
        if not hdr.get('ROLLCORR'):
            continue
        n_rolled += 1
        modes.add(str(hdr.get('ROLLMODE', 'pre-shift (roll_correction.py)')))
        d = hdr.get('ROLLDATE')
        if d:
            epochs.append(calendar.timegm(time.strptime(d, '%Y-%m-%dT%H:%M:%SZ')))
        else:
            epochs.append(os.path.getmtime(f))
    return dict(n_checked=len(files), n_rolled=n_rolled,
                first_epoch=min(epochs) if epochs else None, modes=sorted(modes))


def build_models_for_field(field_name, table=ROLL_TABLE, footprint_cache=None,
                           basepath=None, only_obs=None):
    """{(program, obs): [VisitRoll]} for every NIRCam roll-table row of a field.

    ``only_obs`` (a set of 3-digit observation ids) restricts the footprint
    build to those observations -- for --file runs on a many-tile field.
    """
    from jwst_gc_pipeline import fields
    if basepath is None:
        basepath = fields.fields_basepath(field_name)
    rows = read_roll_table(table)
    grouped = {}
    for r in rows:
        try:
            target = fields.target_for_obsid(r['program'], r['observation'], 'nircam')
        except (KeyError, fields.FieldRegistryError):
            continue
        if target != field_name:
            continue
        if only_obs is not None and r['observation'] not in only_obs:
            continue
        grouped.setdefault((r['program'], r['observation']), []).append(r)
    cache = {}
    if footprint_cache and os.path.exists(footprint_cache):
        with open(footprint_cache) as fh:
            cache = json.load(fh)
    models = {}
    dirty = False
    for (prog, obs), rs in grouped.items():
        ck = f"{prog}-{obs}"
        if ck in cache:
            fp = {v: [np.asarray(p) for p in polys] for v, polys in cache[ck].items()}
        else:
            fp = visit_footprints(basepath, prog, obs)
            cache[ck] = {v: [p.tolist() for p in polys] for v, polys in fp.items()}
            dirty = True
        explicit = {r['visit']: r for r in rs if r['visit'] != '*'}
        wild = [r for r in rs if r['visit'] == '*']
        visits_on_disk = sorted(fp) or sorted(explicit) or ['001']
        vrs = []
        for vis in visits_on_disk:
            r = explicit.get(vis) or (wild[0] if wild else None)
            if r is None:
                continue
            polys = fp.get(vis, [])
            if polys:
                cen = [_mean_sky(p[:, 0], p[:, 1]) for p in polys]
                pra, pde = _mean_sky([c[0] for c in cen], [c[1] for c in cen])
                src = f'coverage centroid of {len(polys)} frame footprints'
            else:
                pra = pde = np.nan
                src = 'none (no frames on disk; falls back to catalog centroid)'
            vrs.append(VisitRoll(prog, obs, vis, r['roll_arcsec'], pra, pde, src, polys))
        if vrs:
            state = frame_roll_state(basepath, prog, obs)
            for v in vrs:
                v.frame_roll = state
            models[(prog, obs)] = vrs     # PointingModel built per catalog (pivot fallback)
    if footprint_cache and dirty:
        # atomic, and only when new footprints were measured: concurrent
        # per-tile jobs share one cache and must never see a half-written file
        os.makedirs(os.path.dirname(os.path.abspath(footprint_cache)), exist_ok=True)
        tmp = f"{footprint_cache}.{os.getpid()}.tmp"
        with open(tmp, 'w') as fh:
            json.dump(cache, fh)
        os.replace(tmp, footprint_cache)
    return models


# ----------------------------------------------------------------------------
# Catalog classification and column discovery
# ----------------------------------------------------------------------------

_MSTAGE_RE = re.compile(r'_m[1-9](?![0-9])')


def classify_catalog(basename, include_legacy=False):
    """'perband' | 'crossband' | 'legacy' | 'excluded' | 'derived' | 'unclassified'.

    Only CURRENT m-series products (an ``_m<N>`` stage token, N = 1..9) are
    corrected by default.  The roll was measured on the current image
    generation; pre-m-series products (crowdsource, iterative, *_LOCKED, dated
    hand-made tables) come from older reductions whose translation ties may
    differ, so they are ``legacy`` unless ``include_legacy``.
    """
    b = basename.lower()
    if not b.endswith(('.fits', '.ecsv')):
        return 'excluded'
    if any(t in b for t in EXCLUDE_TOKENS):
        return 'excluded'
    if any(t in b for t in DERIVED_TOKENS):
        return 'derived'
    kind = ('perband' if _PERBAND_RE.match(b) else
            'crossband' if _CROSSBAND_RE.match(b) else 'unclassified')
    if kind != 'unclassified' and not include_legacy and not _MSTAGE_RE.search(b):
        return 'legacy'
    return kind


def find_position_pairs(colnames):
    """[(ra_col, dec_col, band_or_None)] sky-position column pairs.

    Recognises serialized SkyCoord (``<p>.ra``/``<p>.dec``), ``ra``/``dec``,
    ``RA``/``DEC``, ``<band>_ra``/``<band>_dec`` and ``ra_<band>``/``dec_<band>``.
    Offsets (dra/ddec), scatters (std_ra) and errors are not positions and are
    left alone: a rigid rotation changes them by O(roll) = 1e-4 relative.
    """
    names = list(colnames)
    lower = {c.lower(): c for c in names}
    pairs = []
    for c in names:
        cl = c.lower()
        dec = None
        band = None
        if cl.endswith('.ra'):
            dec = lower.get(cl[:-3] + '.dec')
            m = _FILT_RE.search(cl[:-3])
            band = m.group(1) if m else ('ref' if 'ref' in cl[:-3] else None)
        elif cl in ('ra', 'raj2000', 'ra_icrs'):
            dec = lower.get({'ra': 'dec', 'raj2000': 'dej2000', 'ra_icrs': 'de_icrs'}[cl])
        elif re.match(r'^f\d{3,4}[wmnc]2?_ra$', cl):
            dec = lower.get(cl[:-3] + '_dec')
            band = cl[:-3]
        elif re.match(r'^ra_f\d{3,4}[wmnc]2?$', cl):
            dec = lower.get('dec_' + cl[3:])
            band = cl[3:]
        if dec is not None:
            pairs.append((c, dec, band))
    return pairs


def _catalog_bands(basename, colnames):
    b = basename.lower()
    m = _PERBAND_RE.match(b)
    own = m.group('filt') if m else None
    obs = m.group('obs') if m and m.group('obs') else None
    if obs is None:
        mo = _OBS_RE.search(b)
        obs = mo.group(1) if mo else None
    bands = {mm.group(1).lower() for c in colnames for mm in [_FILT_RE.search(c.lower())] if mm}
    return own, obs, bands


def _program_obs_for_band(field_name, band, obs_token, models):
    """(program, obs) key of ``models`` that took ``band``, or None.

    ``obs_token`` (from the filename) narrows it; without one the band must
    belong to exactly one registered proposal of the field."""
    from jwst_gc_pipeline import fields
    f = fields.BY_NAME[field_name]
    cands = []
    for (prog, obs) in models:
        o = f.observation(prog)
        if o is None:
            continue
        if band is not None and band.lower() not in {x.lower() for x in o.filters}:
            continue
        if obs_token is not None and obs not in obs_token.split('-'):
            continue
        cands.append((prog, obs))
    if len(cands) == 1:
        return cands[0]
    if len(cands) > 1 and band is not None and obs_token is None:
        # same band in several observations of one proposal (e.g. 10678 tiles):
        # only resolvable with an obs token
        return None
    return None


# ----------------------------------------------------------------------------
# Rotation fit (verification)
# ----------------------------------------------------------------------------

def fit_rotation(ra_a, dec_a, ra_b, dec_b, pivot, nclip=3, nsig=3.0):
    """Least-squares (roll, tx, ty) moving same-star pairs a -> b.

    Small-angle model in the tangent plane about ``pivot`` (x east, y north):
    ``b - a = roll * (y, -x) + t`` (the sense of ``rotation_about_pivot``: a
    positive roll rotates North toward East).  Returns roll in arcsec and the
    translation in mas, with formal errors, after ``nclip`` sigma-clip passes.
    """
    xa, ya = _tangent(ra_a, dec_a, *pivot)
    xb, yb = _tangent(ra_b, dec_b, *pivot)
    dx, dy = xb - xa, yb - ya
    keep = np.isfinite(dx) & np.isfinite(dy)
    sol = None
    for _ in range(nclip + 1):
        n = int(keep.sum())
        if n < 5:
            return None
        A = np.zeros((2 * n, 3))
        A[:n, 0], A[n:, 0] = ya[keep], -xa[keep]
        A[:n, 1] = 1.0
        A[n:, 2] = 1.0
        rhs = np.concatenate([dx[keep], dy[keep]])
        sol, _res, _rk, _sv = np.linalg.lstsq(A, rhs, rcond=None)
        model_x = sol[0] * ya + sol[1]
        model_y = -sol[0] * xa + sol[2]
        r = np.hypot(dx - model_x, dy - model_y)
        s = 1.4826 * np.nanmedian(np.abs(r[keep]))
        new = keep & (r < nsig * max(s, 1e-12))
        if new.sum() == keep.sum():
            break
        keep = new
    n = int(keep.sum())
    resid = np.concatenate([(dx - (sol[0] * ya + sol[1]))[keep],
                            (dy - (-sol[0] * xa + sol[2]))[keep]])
    sig2 = np.sum(resid ** 2) / max(2 * n - 3, 1)
    cov = sig2 * np.linalg.inv(A.T @ A)
    rad2as = 206264.806
    return dict(roll_arcsec=float(sol[0] * rad2as), roll_err_arcsec=float(np.sqrt(cov[0, 0]) * rad2as),
                tx_mas=float(sol[1] * rad2as * 1e3), ty_mas=float(sol[2] * rad2as * 1e3),
                t_err_mas=float(np.sqrt(cov[1, 1]) * rad2as * 1e3),
                rms_mas=float(np.sqrt(sig2) * rad2as * 1e3), n=n)


def reference_tie(ra, dec, ref_path, pivot, flux=None, max_rows=20000):
    """Same-star (roll, bulk) of catalog positions vs the SPARSE Gaia subset.

    Sanctioned sequence (CLAUDE.md): density-immune ``measure_offset`` detects
    the tie; ``local_residual_map`` (one giant cell) pairs same stars only after
    a verified small tie; the rotation is fitted on those pairs.  Diagnostic
    only -- Gaia is the frame, not the reference catalog, and never blocks.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    from jwst_gc_pipeline.photometry.astrometry_offsets import (
        measure_offset, local_residual_map, GlobalTieNotVerifiedError)
    ref = Table.read(ref_path)
    if 'source' in ref.colnames:
        src = np.array([s.decode() if isinstance(s, bytes) else str(s) for s in ref['source']])
        ref = ref[src == 'GaiaDR3']
    rcols = [c for c in ('ra', 'RA', 'RAJ2000') if c in ref.colnames]
    dcols = [c for c in ('dec', 'DEC', 'DEJ2000') if c in ref.colnames]
    if 'skycoord' in ref.colnames:
        b = ref['skycoord']
    elif rcols and dcols:
        b = SkyCoord(ref[rcols[0]], ref[dcols[0]], unit='deg')
    else:
        return dict(status='no position columns in reference')
    good = np.isfinite(ra) & np.isfinite(dec)
    idx = np.flatnonzero(good)
    if flux is not None and len(idx) > max_rows:
        f = np.asarray(flux, float)[idx]
        idx = idx[np.argsort(np.where(np.isfinite(f), -f, np.inf))[:max_rows]]
    elif len(idx) > max_rows:
        idx = idx[:: int(np.ceil(len(idx) / max_rows))]
    a = SkyCoord(ra[idx] * u.deg, dec[idx] * u.deg)
    tie = measure_offset(a, b, confirm_windows=True)
    if tie is None or not tie.get('ok'):
        return dict(status='no verified tie', n_ref=len(b))
    try:
        lrm = local_residual_map(a, b, tie, cell_arcsec=3600.0, return_pairs=True,
                                 min_stars=5, context='rollcorr verify')
    except GlobalTieNotVerifiedError as ex:
        return dict(status=f'tie not verified: {ex}', n_ref=len(b))
    p = lrm['pairs']
    if len(p['ia']) < 10:
        return dict(status='too few same-star pairs', n_pairs=int(len(p['ia'])))
    fit = fit_rotation(a.ra.deg[p['ia']], a.dec.deg[p['ia']],
                       b.ra.deg[p['ib']], b.dec.deg[p['ib']], pivot)
    if fit is None:
        return dict(status='fit failed')
    fit['status'] = 'ok'
    fit['hist_tie_mas'] = [float(tie['dra']), float(tie['ddec'])]
    return fit


# ----------------------------------------------------------------------------
# Apply to one catalog
# ----------------------------------------------------------------------------

class RollCatalogError(RuntimeError):
    pass


def code_version():
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        out = subprocess.run(['git', '-C', here, 'describe', '--always', '--dirty'],
                             capture_output=True, text=True, check=True, timeout=10)
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        from importlib.metadata import version, PackageNotFoundError
        try:
            return version('jwst_gc_pipeline')
        except PackageNotFoundError:
            return 'unknown'


def _born_rotated(path, headers, keys, models):
    """Reason string if ``path`` was fit on image-level roll-corrected frames.

    Two signals: the catalog's own header carries ``ROLLCORR`` (stamped at
    fit time from the frame), or the observation's frames were rotated
    BEFORE the catalog was written (``frame_roll_state``).  A catalog older
    than every rotation is a pre-rotation catalog and is corrected normally.
    """
    for h in headers:
        if h.get('ROLLCORR'):
            return ('catalog was fit on image-level roll-corrected frames '
                    '(ROLLCORR in its own header)')
    mtime = os.path.getmtime(path)
    for key in keys:
        for v in models.get(key, []):
            st = v.frame_roll or {}
            if st.get('n_rolled') and st.get('first_epoch') is not None \
                    and mtime >= st['first_epoch']:
                return (f"catalog written after the frames of {key[0]}-{key[1]} were "
                        f"image-level roll-corrected ({st['n_rolled']}/{st['n_checked']} "
                        f"sampled frames, {','.join(st['modes'])})")
    return None


def _pivot_models(vrs, ra_all, dec_all):
    """PointingModel with any missing pivot replaced by the catalog centroid."""
    fixed = []
    for v in vrs:
        if not np.isfinite(v.pivot_ra):
            g = np.isfinite(ra_all) & np.isfinite(dec_all)
            pra, pde = _mean_sky(ra_all[g][::max(1, g.sum() // 200000)],
                                 dec_all[g][::max(1, g.sum() // 200000)])
            v = VisitRoll(v.program, v.observation, v.visit, v.roll_arcsec, pra, pde,
                          'catalog row centroid (no frames on disk)', v.footprints,
                          v.frame_roll)
        fixed.append(v)
    return PointingModel(fixed)


def plan_catalog(path, field_name, models, include_miri=False, include_legacy=False):
    """Decide how ``path`` would be corrected, without reading its data.

    Returns dict(status, pairs=[(ra, dec, key or 'per-row-ref')], reason)."""
    base = os.path.basename(path)
    kind = classify_catalog(base, include_legacy=include_legacy)
    if kind in ('excluded', 'derived', 'unclassified', 'legacy'):
        return dict(status=kind, kind=kind)
    if base.lower().endswith('.fits'):
        with fits.open(path, memmap=True) as hl:
            hdus = [h for h in hl if isinstance(h, fits.BinTableHDU)]
            if not hdus:
                return dict(status='excluded', kind=kind, reason='no table HDU')
            colnames = hdus[0].columns.names
            headers = [hl[0].header] + [h.header for h in hdus]
    else:
        colnames = _ecsv_colnames(path)
        headers = [_ecsv_meta_keys(path)]
    if any(MARKER in h for h in headers):
        return dict(status='refused', kind=kind, reason=f'input already carries {MARKER}')
    own, obs, bands = _catalog_bands(base, colnames)
    if not include_miri and ((own in MIRI_FILTERS) or 'mirimage' in base.lower()):
        return dict(status='skipped-miri', kind=kind)
    pairs = find_position_pairs(colnames)
    if not pairs:
        return dict(status='no-position-columns', kind=kind)
    assigned = []
    for ra_c, dec_c, band in pairs:
        if band == 'ref' or (band is None and kind == 'crossband'):
            refname = [c for c in colnames if c.lower().endswith('_filtername') and 'ref' in c.lower()]
            if refname and band == 'ref':
                assigned.append((ra_c, dec_c, ('per-row', refname[0])))
                continue
            # generic ra/dec in a cross-band table: resolvable only if every band
            # maps to one observation
            keys = {_program_obs_for_band(field_name, b, obs, models) for b in bands
                    if b not in MIRI_FILTERS}
            if len(keys) == 1 and None not in keys:
                assigned.append((ra_c, dec_c, keys.pop()))
                continue
            return dict(status='needs-rebuild', kind=kind,
                        reason=f'{ra_c}/{dec_c} mixes observations {sorted(map(str, keys))}; '
                               f'rebuild from rotated per-band inputs')
        b = band or own
        if b is not None and b in MIRI_FILTERS and not include_miri:
            continue
        key = _program_obs_for_band(field_name, b, obs, models)
        if key is None:
            return dict(status='unattributed', kind=kind,
                        reason=f'cannot attribute {ra_c} (band {b}, obs {obs}) to one roll-table observation')
        assigned.append((ra_c, dec_c, key))
    if not assigned:
        return dict(status='skipped-miri', kind=kind)
    keys = {k for _r, _d, k in assigned if k[0] != 'per-row'}
    if any(k[0] == 'per-row' for _r, _d, k in assigned):
        keys |= set(models)
    born = _born_rotated(path, headers, keys, models)
    if born:
        return dict(status='refused', kind=kind, reason=f'image-level roll-corrected: {born}')
    return dict(status='apply', kind=kind, pairs=assigned)


def _ecsv_meta_keys(path):
    """Top-level meta keys of an ECSV file, read from its YAML header only."""
    keys = {}
    in_meta = False
    with open(path) as fh:
        for line in fh:
            if not line.startswith('#'):
                break
            body = line[2:] if line.startswith('# ') else line[1:]
            if body.startswith('meta:'):
                in_meta = True
                continue
            if in_meta:
                if body and not body.startswith((' ', '-', '!')):
                    in_meta = False
                    continue
                m = re.match(r"\s*(?:-\s*)?(?:!!omap\s*)?-?\s*\{?\s*([A-Za-z0-9_]+)\s*:", body)
                if m:
                    keys[m.group(1)] = True
    return keys


def _ecsv_colnames(path):
    with open(path) as fh:
        for line in fh:
            if not line.startswith('#'):
                return line.strip().split()
    return []


def correct_catalog(path, out_path, field_name, models, include_miri=False,
                    verify_reference=None, table=ROLL_TABLE, overwrite_output=False,
                    include_legacy=False):
    """Rotate every sky-position column of ``path`` into ``out_path``.

    Raises ``RollCatalogError`` on any refusal.  Returns a result dict with the
    per-column max displacement, the geometric check, and (optionally) the
    before/after same-star tie vs sparse Gaia.
    """
    if os.path.abspath(path) == os.path.abspath(out_path):
        raise RollCatalogError(f"refusing to overwrite the input in place: {path}")
    if os.path.exists(out_path) and not overwrite_output:
        raise RollCatalogError(f"output exists: {out_path} (pass overwrite_output)")
    plan = plan_catalog(path, field_name, models, include_miri=include_miri,
                        include_legacy=include_legacy)
    if plan['status'] != 'apply':
        raise RollCatalogError(f"{os.path.basename(path)}: {plan['status']} "
                               f"{plan.get('reason', '')}".strip())
    t0 = time.time()
    is_fits = path.lower().endswith('.fits')
    if is_fits:
        hl = fits.open(path, memmap=False)
        ti = next(i for i, h in enumerate(hl) if isinstance(h, fits.BinTableHDU))
        data = hl[ti].data
        get = lambda c: np.array(data[c], dtype=float, copy=True)
        colnames = hl[ti].columns.names

        def put(c, v):
            data[c][:] = v
        units = {c: (hl[ti].header.get(f'TUNIT{i + 1}') or '').strip().lower()
                 for i, c in enumerate(colnames)}
    else:
        from astropy.table import Table
        from astropy.table.serialize import represent_mixins_as_columns
        # flatten SkyCoord/Quantity mixins to their serialized columns
        # (skycoord_ref.ra ...); the __serialized_columns__ meta that this
        # records makes the written file read back as the same mixins.
        tab = represent_mixins_as_columns(Table.read(path, format='ascii.ecsv'))
        get = lambda c: np.array(tab[c], dtype=float, copy=True)
        colnames = tab.colnames

        def put(c, v):
            tab[c][:] = v
        units = {c: str(getattr(tab[c], 'unit', '') or '').lower() for c in colnames}

    results = []
    model_cache = {}
    first_ra = get(plan['pairs'][0][0])
    first_dec = get(plan['pairs'][0][1])
    for ra_c, dec_c, key in plan['pairs']:
        for c in (ra_c, dec_c):
            if units.get(c) not in ('', 'deg', 'degree', 'degrees'):
                raise RollCatalogError(f"{c}: unit {units.get(c)!r} is not degrees")
        ra, dec = get(ra_c), get(dec_c)
        far_all = np.zeros(len(ra), dtype=bool)
        if isinstance(key, tuple) and key[0] == 'per-row':
            fn = np.char.lower(np.char.strip(np.asarray(
                data[key[1]] if is_fits else tab[key[1]]).astype(str)))
            new_ra, new_dec = ra.copy(), dec.copy()
            for band in np.unique(fn):
                k = _program_obs_for_band(field_name, band, _catalog_bands(
                    os.path.basename(path), colnames)[1], models)
                if k is None:
                    raise RollCatalogError(f"{ra_c}: reference band {band!r} not attributable")
                if k not in model_cache:
                    model_cache[k] = _pivot_models(models[k], first_ra, first_dec)
                sel = fn == band
                new_ra[sel], new_dec[sel] = model_cache[k].apply(ra[sel], dec[sel])
                far = sel.copy()
                far[sel] = ~model_cache[k].within_reach(ra[sel], dec[sel])
                far_all |= far
            mkeys = sorted({str(k) for k in model_cache})
        else:
            if key not in model_cache:
                model_cache[key] = _pivot_models(models[key], first_ra, first_dec)
            new_ra, new_dec = model_cache[key].apply(ra, dec)
            far_all = ~model_cache[key].within_reach(ra, dec)
            mkeys = [str(key)]
        # rows beyond the pointing's footprint reach keep their input position
        new_ra[far_all], new_dec[far_all] = ra[far_all], dec[far_all]
        n_far = int((far_all & np.isfinite(ra) & np.isfinite(dec)).sum())
        disp = np.hypot(((new_ra - ra + 180) % 360 - 180) * np.cos(np.radians(dec)),
                        new_dec - dec) * 3.6e6
        # geometric check: displacement <= |roll| * r(pivot) for every visit used
        fin = np.isfinite(disp)
        bound = np.zeros_like(disp)
        for m in model_cache.values():
            for v in m.visits:
                r = np.hypot(*_tangent(ra, dec, v.pivot_ra, v.pivot_dec)) * 206264.806
                bound = np.maximum(bound, np.abs(v.roll_arcsec) / 206264.806 * r * 1e3)
        excess = float(np.nanmax(disp[fin] - bound[fin])) if fin.any() else 0.0
        if excess > 0.05:
            raise RollCatalogError(f"{ra_c}: displacement exceeds |roll|*r by {excess:.3f} mas")
        roundtrip = None
        if len(model_cache) == 1 and len(next(iter(model_cache.values())).visits) == 1 and fin.any():
            v = next(iter(model_cache.values())).visits[0]
            rot = fin & ~far_all
            rb, db = rotation_about_pivot(new_ra[rot], new_dec[rot], v.pivot_ra, v.pivot_dec,
                                          -v.roll_arcsec)
            roundtrip = float(np.max(np.hypot(((rb - ra[rot] + 180) % 360 - 180)
                                              * np.cos(np.radians(dec[rot])),
                                              db - dec[rot])) * 3.6e6) if rot.any() else 0.0
            if roundtrip > 0.01:
                raise RollCatalogError(f"{ra_c}: inverse rotation misses input by {roundtrip:.4f} mas")
        put(ra_c, new_ra)
        put(dec_c, new_dec)
        results.append(dict(ra=ra_c, dec=dec_c, models=mkeys,
                            max_disp_mas=float(np.nanmax(disp)) if fin.any() else 0.0,
                            median_disp_mas=float(np.nanmedian(disp)) if fin.any() else 0.0,
                            roundtrip_mas=roundtrip, n=int(fin.sum()), n_beyond_reach=n_far))

    # separations between bands now rotated differently must be recomputed
    sep_fixed = []
    if len(model_cache) > 1:
        for c in colnames:
            m = re.match(r'^sep_(f\d{3,4}[wmnc]2?)$', c.lower())
            if not m:
                continue
            b = m.group(1)
            ra_b = [p for p in plan['pairs'] if _FILT_RE.search(p[0].lower())
                    and _FILT_RE.search(p[0].lower()).group(1) == b]
            ra_r = [p for p in plan['pairs'] if 'ref' in p[0].lower()]
            if not ra_b or not ra_r:
                continue
            from astropy.coordinates import SkyCoord
            s1 = SkyCoord(get(ra_b[0][0]), get(ra_b[0][1]), unit='deg')
            s2 = SkyCoord(get(ra_r[0][0]), get(ra_r[0][1]), unit='deg')
            sep = s1.separation(s2)
            unit = units.get(c) or 'deg'
            put(c, sep.to_value(unit if unit in ('deg', 'arcsec', 'mas') else 'deg'))
            sep_fixed.append(c)

    prov = dict(ROLLCCAT=(True, 'catalog roll correction applied (data-qa#346)'),
                ROLLCVER=(code_version(), 'catalog_roll_correction code version'),
                ROLLCTAB=(f"roll_corrections.csv sha1:{table_sha(table)}", 'roll table'),
                ROLLCSRC=('JWST-GC/data-qa#346', 'roll measurement'),
                ROLLCIN=(os.path.basename(path)[-68:], 'input catalog basename'),
                ROLLCPIV=('visit coverage centroid', 'rotation pivot'),
                ROLLPIX=('pixel cols refer to UNCORRECTED WCS', 'x/y not rotated'),
                ROLLCDAT=(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'applied UTC'))
    vis_cards = []
    for i, m in enumerate(model_cache.values()):
        for j, v in enumerate(m.visits):
            n = len(vis_cards)
            if n >= 99:
                break
            vis_cards += [(f'ROLV{n:02d}ID', v.key, 'program-obs-visit'),
                          (f'ROLV{n:02d}AS', float(v.roll_arcsec), '[arcsec] roll applied +N->E'),
                          (f'ROLV{n:02d}RA', float(v.pivot_ra), '[deg] pivot RA'),
                          (f'ROLV{n:02d}DE', float(v.pivot_dec), '[deg] pivot Dec')]
    history = [f"rollcorr col {r['ra']}/{r['dec']} models {','.join(r['models'])} "
               f"max {r['max_disp_mas']:.2f} mas"
               + (f" ({r['n_beyond_reach']} rows beyond footprint reach left unrotated)"
                  if r['n_beyond_reach'] else '') for r in results]
    history += [f"rollcorr recomputed {c}" for c in sep_fixed]

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp = out_path + '.tmp'
    if is_fits:
        for h in (hl[0].header, hl[ti].header):
            for k, (v, cmt) in prov.items():
                h[k] = (v, cmt)
            for k, v, cmt in vis_cards:
                h[k] = (v, cmt)
        for line in history:
            hl[ti].header.add_history(line[:70])
        hl.writeto(tmp, overwrite=True)
        hl.close()
    else:
        for k, (v, cmt) in prov.items():
            tab.meta[k] = v
        for k, v, cmt in vis_cards:
            tab.meta[k] = v
        tab.meta['ROLLHIST'] = history
        tab.write(tmp, format='ascii.ecsv', overwrite=True)
    os.replace(tmp, out_path)

    out = dict(input=path, output=out_path, columns=results, sep_recomputed=sep_fixed,
               models={str(k): m.describe() for k, m in model_cache.items()},
               seconds=round(time.time() - t0, 2),
               size_mb=round(os.path.getsize(path) / 1e6, 1))
    if verify_reference is True:
        k0 = next(iter(model_cache))
        verify_reference = _ref_for(field_name, {k0: None})
        out['verify_skipped'] = None if verify_reference else f'no reference catalog for {k0}'
    if verify_reference:
        ra_c, dec_c, key = plan['pairs'][0]
        m = next(iter(model_cache.values()))
        pivot = (m.visits[0].pivot_ra, m.visits[0].pivot_dec)
        flux_c = next((c for c in colnames if c.lower() in ('flux', 'flux_fit_avg')
                       or c.lower().startswith('flux_f')), None)
        flux = get(flux_c) if flux_c else None
        with fits.open(out_path, memmap=True) if is_fits else _nullctx() as hl2:
            if is_fits:
                d2 = hl2[ti].data
                ra2, dec2 = np.asarray(d2[ra_c], float), np.asarray(d2[dec_c], float)
            else:
                from astropy.table import Table
                from astropy.table.serialize import represent_mixins_as_columns
                t2 = represent_mixins_as_columns(Table.read(out_path, format='ascii.ecsv'))
                ra2, dec2 = np.asarray(t2[ra_c], float), np.asarray(t2[dec_c], float)
        before = reference_tie(first_ra, first_dec, verify_reference, pivot, flux=flux)
        after = reference_tie(ra2, dec2, verify_reference, pivot, flux=flux)
        out['verify'] = dict(reference=verify_reference, column=ra_c, before=before, after=after)
    return out


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ----------------------------------------------------------------------------
# Field enumeration + CLI
# ----------------------------------------------------------------------------

def enumerate_field(field_name, include_perframe=False, include_releases=True,
                    release_root='/orange/adamginsburg/jwst/releases'):
    """{class: [paths]} for a field's catalogs (read-only listing)."""
    from jwst_gc_pipeline import fields
    base = fields.fields_basepath(field_name).rstrip('/')
    out = {'catalogs': sorted(p for p in glob.glob(os.path.join(base, 'catalogs', '*'))
                              if os.path.isfile(p))}
    if include_perframe:
        out['perframe'] = sorted(glob.glob(os.path.join(base, 'F*', '*_m[0-9]*_daophot_basic.fits')))
    if include_releases:
        out['release'] = sorted(glob.glob(os.path.join(release_root, 'v*', '**', field_name,
                                                       'catalogs', '*.fits'), recursive=True))
    return out


def fields_in_table(table=ROLL_TABLE):
    from jwst_gc_pipeline import fields
    out = []
    for r in read_roll_table(table):
        try:
            t = fields.target_for_obsid(r['program'], r['observation'], 'nircam')
        except (KeyError, fields.FieldRegistryError):
            continue
        if t not in out:
            out.append(t)
    return out


def _du(paths):
    return sum(os.path.getsize(p) for p in paths if os.path.exists(p))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--field', action='append', help='field(s); default: every field in the roll table')
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--apply', action='store_true')
    ap.add_argument('--file', action='append', help='apply to these files only (e.g. a COPY)')
    ap.add_argument('--out-dir', help=f'output dir (default <field>/{OUT_SUBDIR}/<tag>)')
    ap.add_argument('--tag', default=DEFAULT_TAG)
    ap.add_argument('--include-miri', action='store_true')
    ap.add_argument('--include-perframe', action='store_true')
    ap.add_argument('--include-legacy', action='store_true',
                    help='also correct pre-m-series products (older image generations)')
    ap.add_argument('--verify-reference', action='store_true',
                    help='before/after same-star tie vs the sparse Gaia subset (diagnostic)')
    ap.add_argument('--footprint-cache', help='JSON cache of visit footprints')
    ap.add_argument('--manifest', help='write a JSON manifest here')
    ap.add_argument('--overwrite-output', action='store_true')
    ap.add_argument('--table', default=ROLL_TABLE)
    ap.add_argument('--stages', help='comma list of m-stages to include, e.g. 7,8 '
                                     '(matched as _m<N> in the basename)')
    ap.add_argument('--obs', help='comma list of 3-digit observation ids; only catalogs whose '
                                  'basename carries one of these _o tokens (tiled programs)')
    ap.add_argument('--formats', default='fits,ecsv',
                    help='comma list of file extensions to include (default fits,ecsv)')
    args = ap.parse_args(argv)
    stage_re = (re.compile(r'_m(?:' + '|'.join(args.stages.split(',')) + r')(?![0-9])')
                if args.stages else None)
    obs_set = set(args.obs.split(',')) if args.obs else None
    exts = tuple('.' + e.strip().lower() for e in args.formats.split(','))

    if os.environ.get('ROLL_CORRECTION_ARCSEC') not in (None, ''):
        ap.error('ROLL_CORRECTION_ARCSEC is set; the catalog correction reads the per-visit '
                 'table only. Unset it.')
    from jwst_gc_pipeline import fields
    field_list = args.field or fields_in_table(args.table)
    manifest = dict(tag=args.tag, table_sha=table_sha(args.table), code=code_version(),
                    fields={})
    for fname in field_list:
        base = fields.fields_basepath(fname).rstrip('/')
        fcache = args.footprint_cache or None
        only_obs = obs_set
        if args.file and obs_set is None:
            toks = [_OBS_RE.search(os.path.basename(p).lower()) for p in args.file]
            if all(toks):
                only_obs = {o for t in toks for o in t.group(1).split('-')}
        models = build_models_for_field(fname, table=args.table, footprint_cache=fcache,
                                        only_obs=only_obs)
        out_dir = args.out_dir or os.path.join(base, OUT_SUBDIR, args.tag)
        listing = ({'catalogs': args.file} if args.file else
                   enumerate_field(fname, include_perframe=args.include_perframe))

        def _keep(p):
            b = os.path.basename(p).lower()
            if not b.endswith(exts):
                return False
            if stage_re is not None and not stage_re.search(b):
                return False
            if obs_set is not None:
                mo = _OBS_RE.search(b)
                if mo is None or not set(mo.group(1).split('-')) & obs_set:
                    return False
            return True
        if stage_re is not None or obs_set is not None or args.formats != 'fits,ecsv':
            listing = {k: [p for p in v if _keep(p)] for k, v in listing.items()}
        frec = dict(models={f"{k[0]}-{k[1]}": [dict(visit=v.visit, roll=v.roll_arcsec,
                                                    pivot=[v.pivot_ra, v.pivot_dec],
                                                    pivot_source=v.pivot_source,
                                                    n_footprints=len(v.footprints))
                                               for v in vrs] for k, vrs in models.items()},
                    classes={}, files=[])
        print(f"\n=== {fname}  ({base})  -> {out_dir}")
        for k, vrs in models.items():
            for v in vrs:
                print(f"  roll {v.key:>16s} {v.roll_arcsec:+7.2f}\"  pivot "
                      f"({v.pivot_ra:.5f}, {v.pivot_dec:.5f})  [{v.pivot_source}]")
        for cls, paths in listing.items():
            counts = {}
            for p in paths:
                if cls == 'release':
                    st = dict(status='release-restage')
                else:
                    try:
                        st = plan_catalog(p, fname, models, include_miri=args.include_miri,
                                          include_legacy=args.include_legacy)
                    except (OSError, ValueError, KeyError) as ex:
                        st = dict(status='unreadable', reason=str(ex)[:120])
                key = st['status']
                counts.setdefault(key, [0, 0])
                counts[key][0] += 1
                counts[key][1] += os.path.getsize(p) if os.path.exists(p) else 0
                rec = dict(path=p, status=key, reason=st.get('reason'))
                if args.apply and key == 'apply':
                    outp = os.path.join(out_dir, os.path.basename(p))
                    try:
                        res = correct_catalog(
                            p, outp, fname, models, include_miri=args.include_miri,
                            verify_reference=bool(args.verify_reference),
                            table=args.table, overwrite_output=args.overwrite_output,
                            include_legacy=args.include_legacy)
                        rec.update(res)
                        mx = max(c['max_disp_mas'] for c in res['columns'])
                        print(f"  applied {os.path.basename(p)}  max {mx:.2f} mas  {res['seconds']} s")
                        if 'verify' in res:
                            b, a = res['verify']['before'], res['verify']['after']
                            print(f"    Gaia same-star roll before {b.get('roll_arcsec', np.nan):+.2f}\" "
                                  f"after {a.get('roll_arcsec', np.nan):+.2f}\"  bulk "
                                  f"({b.get('tx_mas', np.nan):+.2f},{b.get('ty_mas', np.nan):+.2f}) -> "
                                  f"({a.get('tx_mas', np.nan):+.2f},{a.get('ty_mas', np.nan):+.2f}) mas  "
                                  f"[{b.get('status')}/{a.get('status')}]")
                    except RollCatalogError as ex:
                        rec['status'] = 'refused'
                        rec['reason'] = str(ex)
                        print(f"  REFUSED {ex}")
                frec['files'].append(rec)
            frec['classes'][cls] = {k: dict(n=v[0], gb=round(v[1] / 1e9, 3)) for k, v in counts.items()}
            for k, v in sorted(counts.items()):
                print(f"  [{cls:8s}] {k:22s} {v[0]:6d} files  {v[1] / 1e9:8.2f} GB")
        manifest['fields'][fname] = frec
    if args.manifest:
        with open(args.manifest, 'w') as fh:
            json.dump(manifest, fh, indent=1, default=str)
        print(f"\nmanifest -> {args.manifest}")
    return 0


def _ref_for(field_name, models):
    """Sparse-Gaia-capable reference catalog for the field's first observation."""
    from jwst_gc_pipeline import fields
    for (prog, obs) in models:
        try:
            p = fields.reference_catalog_path(prog, obs)
        except (KeyError, fields.FieldRegistryError, FileNotFoundError):
            continue
        if p and os.path.exists(p):
            return p
    return None


if __name__ == '__main__':
    sys.exit(main())
