"""Pipeline step 0: settle the BULK astrometric offset before anything downstream.

The bulk offset is the field/visit-level tie to the absolute reference frame --
the term that is arcseconds wide when a guide-star acquisition went wrong.  It is
categorically different from the per-exposure jitter the re-tie loop chases:

* it is **large**, so getting it wrong corrupts every product built on top of it
  (drizzled mosaics, catalogs, cross-band merges) rather than just blurring stars;
* it is **known once**.  For every field currently in the pipeline it has already
  been measured, and re-deriving it on each run is both wasted work and a chance
  to regress a value that is already right.

So step 0 has two modes, and the distinction is the point of this module:

``VERIFY`` (a bulk offset is already recorded)
    Confirm the recorded value still describes the data, and **change nothing**.
    A disagreement RAISES -- it means either the frames moved (a new reduction
    generation) or the recorded value is wrong, and both need a human.

``MEASURE`` (no bulk offset recorded -- new data)
    Measure it, record it, and hand it to the alignment config.

Why it needs m1
---------------

Measuring a bulk offset needs source positions, which means cataloging has to
have run.  That is the one real ordering constraint: **m1 per-frame cataloging
must run before drizzling**.  A rigid field shift moves every source identically,
so a catalog built before alignment measures the bulk offset perfectly well --
but it has to exist.

The catalogs on this archive usually carry an APPLIED correction already: brick
F200W visit 001 frames carry an applied (-17.597, +13.453) arcsec and sickle F187N
an applied (-0.102, -0.034) arcsec, in both cases exactly the recorded bulk.  A
measurement on such catalogs is therefore the RESIDUAL the frames still owe, near
zero -- so what VERIFY compares against is ``recorded - applied``, and
:func:`applied_bulk_mas` is what tells the two states apart.  Comparing a
residual against the full recorded bulk fails by the whole size of the offset and
reports a correctly-tied field as broken.

Measurement method
------------------

This module deliberately contains NO offset-measurement code of its own.  It
calls :func:`~jwst_gc_pipeline.photometry.visit_consensus.measure_reference_tie`,
which is the sanctioned path: an offset-histogram peak with a swept window to
DETECT the tie (density-immune, and correct no matter how large the shift), then
a same-star matched-pair refinement for the precise value once the tie is
verified small.  Nearest-neighbour medians against a dense reference are banned
outright (CLAUDE.md ASTROMETRY RULE #1) and nothing here reintroduces one.

Cost control
------------

A full re-measure is not free, so it is gated on a **generation hash** covering
the frames' WCS-generation stamps, the reference catalog, and the recorded value
itself.  When the hash is unchanged since the last verification the check is a
cheap comparison against the cached result; when anything moves, it re-measures.
That way the expensive path runs exactly when the answer could have changed.
"""

import hashlib
import json
import os
import re
import warnings
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np

__all__ = [
    'BulkOffsetVerificationError', 'BulkOffsetResult',
    'BULK_VERIFY_TOL_MAS', 'generation_hash', 'measure_bulk_offset',
    'verify_recorded_bulk', 'step0_bulk_offset',
    'load_step0_record', 'save_step0_record', 'step0_record_path',
    'bulk_tie_state', 'recorded_bulk_mas', 'recorded_bulk_over_visits',
    'BULK_RECORDED', 'BULK_IN_TABLE', 'BULK_NONE',
    'BULK_OK', 'BULK_TABLE_ABSENT', 'BULK_NO_ROW', 'BULK_NOT_CONFIGURED',
    'BULK_VISITS_MIXED', 'BULK_VISITS_DISAGREE',
    'reference_frame_matches_refcat',
    'exposure_key', 'duplicate_exposure_catalogs', 'refcat_for_frame',
    'default_catalog_glob', 'default_frame_glob', 'REFCAT_PATTERNS',
    'applied_bulk_mas', 'verify_tolerance_mas',
]

#: A recorded bulk offset must still describe the data to within this, on-sky.
#: Set well above the per-exposure jitter the re-tie loop handles (tens of mas)
#: and well below anything that would corrupt a mosaic.  Override with
#: ``BULK_VERIFY_TOL_MAS`` in the environment.
BULK_VERIFY_TOL_MAS = 100.0

#: Escape hatch for a deliberate, justified override.  Not for making a red gate
#: green -- a failed bulk verification means a product is about to be built on a
#: wrong frame.
ALLOW_FAIL_ENV = 'ALLOW_BULK_VERIFY_FAIL'


class BulkOffsetVerificationError(RuntimeError):
    """A recorded bulk offset no longer describes the data."""


@dataclass
class BulkOffsetResult:
    """Outcome of a step-0 check."""

    #: ``'verify'`` or ``'measure'``
    mode: str
    #: measured on-sky offset (mas), or None when the check was served from cache
    measured_dra_mas: Optional[float] = None
    measured_ddec_mas: Optional[float] = None
    #: the value on record (mas, on-sky), when there was one
    recorded_dra_mas: Optional[float] = None
    recorded_ddec_mas: Optional[float] = None
    #: separation between measured and recorded (mas)
    sep_mas: Optional[float] = None
    #: ``'same-star'`` / ``'histogram'`` -- which estimator produced the value
    bulk_source: str = ''
    #: True when the measurement passed all of measure_reference_tie's checks
    apply_ok: bool = False
    #: True when the expensive re-measure was skipped because nothing changed
    from_cache: bool = False
    #: generation hash this result belongs to
    hash: str = ''
    #: human-readable summary
    detail: str = ''

    @property
    def passed(self) -> bool:
        # apply_ok gates BOTH modes.  A measurement the sanctioned estimator
        # refused cannot confirm a recorded tie any more than it can establish a
        # new one -- and on the verify path the refusal usually means the
        # per-tile map is dirty, i.e. a rigid sub-field shift the bulk number
        # alone does not reveal.
        if not self.apply_ok:
            return False
        if self.mode == 'measure':
            return True
        return (self.sep_mas is not None and np.isfinite(self.sep_mas)
                and self.sep_mas <= _tol())


def verify_tolerance_mas() -> float:
    """The tolerance a recorded bulk tie must still describe the data to (mas)."""
    return float(os.environ.get('BULK_VERIFY_TOL_MAS', BULK_VERIFY_TOL_MAS))


#: Internal alias kept for the many call sites inside this module.
_tol = verify_tolerance_mas


# ---------------------------------------------------------------------------
# generation hash -- what makes a re-measure necessary
# ---------------------------------------------------------------------------

def generation_hash(frame_paths, reference_id, recorded=None, extra=''):
    """Hash the things that, if they change, invalidate a bulk verification.

    Covers each frame's WCS-generation stamp (CAL_VER / CRDS context / DVA
    correction state), the identity of the reference catalog, the recorded
    value being checked, AND the tolerance it was checked at (a pass recorded
    under a widened BULK_VERIFY_TOL_MAS must not satisfy a stricter later run).
    Frame *contents* are deliberately not hashed -- a
    re-reduction that leaves the generation identical produces the same
    astrometric solution, and hashing pixels would make this uselessly slow.
    """
    from astropy.io import fits
    from jwst_gc_pipeline.astrometry_utils import generation_stamp

    h = hashlib.sha256()
    h.update(str(reference_id).encode())
    h.update(str(extra).encode())
    # The TOLERANCE is part of what makes a verification meaningful: a pass
    # recorded under a widened BULK_VERIFY_TOL_MAS must not satisfy a later run
    # at the normal tolerance.  Without this a single wide run caches a large
    # disagreement as a pass and re-serves it forever.
    h.update(f"tol={_tol():.6f}".encode())
    if recorded is not None:
        h.update(f"{float(recorded[0]):.6f},{float(recorded[1]):.6f}".encode())
    for path in sorted(frame_paths):
        h.update(os.path.basename(path).encode())
        try:
            with fits.open(path) as fh:
                hdr = dict(fh[0].header)
                hdr.update({k: v for k, v in fh[1].header.items()
                            if k in ('DVACORR',)})
            stamp = generation_stamp(hdr)
            h.update(json.dumps(stamp, sort_keys=True).encode())
        except (OSError, KeyError, IndexError) as ex:
            # An unreadable frame must not silently produce a STABLE hash --
            # that would cache a verification that never really ran.  Fold the
            # error in so the hash changes and the next run re-measures.
            h.update(f"UNREADABLE:{ex}".encode())
    return h.hexdigest()[:32]


# ---------------------------------------------------------------------------
# on-disk record
# ---------------------------------------------------------------------------

def step0_record_path(basepath, proposal_id, field, filtername, visit=None):
    """Where a step-0 outcome is recorded.

    ``visit`` is part of the filename when the run was scoped to one visit: a
    field whose visits carry different bulk ties produces a different result per
    visit, and a shared filename means each run evicts the other's record.
    """
    stem = f'step0_bulk_{proposal_id}_o{field}_{filtername}'
    if visit:
        stem += f'_v{visit}'
    return os.path.join(basepath, 'offsets', f'{stem}.json')


def load_step0_record(path):
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return json.load(fh)


def save_step0_record(path, record):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(record, fh, indent=2, sort_keys=True)


# ---------------------------------------------------------------------------
# measurement (delegated -- see module docstring)
# ---------------------------------------------------------------------------

def measure_bulk_offset(catalog_coords, ref_coords_all, ref_coords_sparse,
                        filtername=None, catalog_mag=None, ref_mag=None,
                        dense=True, context=''):
    """Measure the bulk offset with the sanctioned estimator.

    Thin wrapper over
    :func:`~jwst_gc_pipeline.photometry.visit_consensus.measure_reference_tie`
    so there is exactly one implementation of "how do we measure an offset" in
    the codebase.  Returns its result dict unchanged.

    ``dense`` MUST be forwarded from ``load_reference_catalog(...)['dense']``: a
    Gaia-only reference (``dense=False``) cannot gate on the per-tile map (it is
    noise against a sparse catalog), so ``measure_reference_tie`` falls back to
    the same-star check.  Leaving it True on a Gaia-only field reproduces the
    stranded-bulk bug on the reducer side (apply_ok=False for a real, coherent
    tie).
    """
    from jwst_gc_pipeline.photometry.visit_consensus import measure_reference_tie
    return measure_reference_tie(
        catalog_coords, ref_coords_all, ref_coords_sparse,
        filtername=filtername, consensus_mag=catalog_mag, ref_mag=ref_mag,
        dense=dense, context=context or 'step0 bulk')


def verify_recorded_bulk(recorded_mas, measured_mas, tol_mas=None, context=''):
    """Compare a recorded bulk offset against a fresh measurement.

    Returns the separation in mas.  Raises :class:`BulkOffsetVerificationError`
    when they disagree by more than ``tol_mas`` -- unless ``ALLOW_BULK_VERIFY_FAIL=1``
    is set, in which case it warns and returns.
    """
    if tol_mas is None:
        tol_mas = _tol()
    dra = float(measured_mas[0]) - float(recorded_mas[0])
    ddec = float(measured_mas[1]) - float(recorded_mas[1])
    if not np.isfinite([dra, ddec]).all():
        sep = float('nan')
    else:
        sep = float(np.hypot(dra, ddec))

    ok = np.isfinite(sep) and sep <= tol_mas
    if ok:
        return sep

    msg = (f"BULK OFFSET VERIFICATION FAILED{(' for ' + context) if context else ''}: "
           f"recorded ({recorded_mas[0]:+.1f},{recorded_mas[1]:+.1f}) mas but the "
           f"data now measure ({measured_mas[0]:+.1f},{measured_mas[1]:+.1f}) mas "
           f"-- separation {sep:.1f} mas > {tol_mas:.1f} mas tolerance. Either the "
           f"frames moved (a new reduction generation invalidates the recorded "
           f"tie) or the recorded value is wrong. Do NOT build products on this: "
           f"re-derive the bulk offset and update alignment_config.py. "
           f"({ALLOW_FAIL_ENV}=1 to override deliberately.)")
    if os.environ.get(ALLOW_FAIL_ENV) == '1':
        warnings.warn(msg)
        return sep
    raise BulkOffsetVerificationError(msg)


# ---------------------------------------------------------------------------
# the step
# ---------------------------------------------------------------------------

def step0_bulk_offset(catalog_coords, ref_coords_all, ref_coords_sparse,
                      frame_paths, basepath, proposal_id, field, filtername,
                      recorded_mas=None, reference_id='VIRAC2',
                      catalog_mag=None, ref_mag=None, force=False, visit=None,
                      dense=True):
    """Run step 0 for one (field, filter).

    ``recorded_mas`` is the bulk offset already on record (on-sky mas), or None
    for new data.  With a record present this VERIFIES and returns without
    proposing any change; without one it MEASURES and writes a record for the
    alignment config to adopt.
    """
    context = f"{proposal_id}/o{field}/{filtername}"
    if visit:
        context += f"/v{visit}"
    rec_path = step0_record_path(basepath, proposal_id, field, filtername,
                                 visit=visit)
    ghash = generation_hash(frame_paths, reference_id, recorded=recorded_mas)
    cached = load_step0_record(rec_path)

    cached_ok = False
    if cached is not None and cached.get('hash') == ghash and cached.get('passed'):
        # Re-evaluate rather than trusting the stored boolean: `passed` was
        # computed under whatever tolerance was in force when it was written.
        _csep = cached.get('sep_mas')
        cached_ok = bool(cached.get('apply_ok')) and (
            cached.get('mode') == 'measure'
            or (_csep is not None and np.isfinite(_csep)
                and float(_csep) <= _tol()))
    if not force and cached_ok:
        print(f"[step0] {context}: generation unchanged since "
              f"{cached.get('verified_at', 'a previous run')} -- reusing the "
              f"verified bulk tie (hash {ghash[:12]}).")
        return BulkOffsetResult(
            mode=cached.get('mode', 'verify'),
            measured_dra_mas=cached.get('measured_dra_mas'),
            measured_ddec_mas=cached.get('measured_ddec_mas'),
            recorded_dra_mas=None if recorded_mas is None else float(recorded_mas[0]),
            recorded_ddec_mas=None if recorded_mas is None else float(recorded_mas[1]),
            sep_mas=cached.get('sep_mas'),
            bulk_source=cached.get('bulk_source', ''),
            apply_ok=bool(cached.get('apply_ok')),
            from_cache=True, hash=ghash,
            detail='served from the step0 cache; nothing that affects the tie changed')

    tie = measure_bulk_offset(catalog_coords, ref_coords_all, ref_coords_sparse,
                              filtername=filtername, catalog_mag=catalog_mag,
                              ref_mag=ref_mag, dense=dense, context=context)
    measured = (float(tie.get('dra_mas', float('nan'))),
                float(tie.get('ddec_mas', float('nan'))))
    apply_ok = bool(tie.get('apply_ok'))
    bulk_source = str(tie.get('bulk_source', ''))

    # Which check gates apply_ok depends on the reference: a DENSE (VIRAC2) ref
    # gates on the per-tile map; a Gaia-only ref (dense=False) gates on the
    # same-star refinement (the per-tile map is noise there).
    _gate_hint = ("the per-tile map and the sweep result" if dense else
                  "the same-star refinement (was the tie small enough to refine, "
                  "or did it sweep?) and the sweep result")
    if recorded_mas is None:
        mode = 'measure'
        sep = None
        if not apply_ok:
            raise BulkOffsetVerificationError(
                f"BULK OFFSET MEASUREMENT REJECTED for {context}: measure_reference_tie "
                f"did not sign off (apply_ok=False) on "
                f"({measured[0]:+.1f},{measured[1]:+.1f}) mas. An unverified bulk tie "
                f"must not be recorded -- a spurious or window-limited peak here "
                f"propagates into every downstream product. Check {_gate_hint} "
                f"before recording anything.")
        print(f"[step0] {context}: MEASURED bulk offset "
              f"({measured[0]:+.1f},{measured[1]:+.1f}) mas via {bulk_source}. "
              f"Record it in alignment_config.py to apply it.")
        detail = 'no bulk offset on record; measured and recorded'
    else:
        mode = 'verify'
        if not apply_ok:
            _msg = (
                f"BULK OFFSET VERIFICATION INCONCLUSIVE for {context}: "
                f"measure_reference_tie did not sign off (apply_ok=False) on its "
                f"measurement ({measured[0]:+.1f},{measured[1]:+.1f}) mas, so it "
                f"cannot confirm the recorded tie "
                f"({recorded_mas[0]:+.1f},{recorded_mas[1]:+.1f}) mas either. "
                f"On a dense reference apply_ok is False when the per-tile map is "
                f"not clean -- exactly the 'bulk reads ~0 while half the mosaic is "
                f"shifted' case this step exists to catch, so treating a close "
                f"separation as a pass would defeat the check. Inspect {_gate_hint}. "
                f"({ALLOW_FAIL_ENV}=1 to override deliberately.)")
            if os.environ.get(ALLOW_FAIL_ENV) == '1':
                warnings.warn(_msg)
            else:
                raise BulkOffsetVerificationError(_msg)
        sep = verify_recorded_bulk(recorded_mas, measured, context=context)
        print(f"[step0] {context}: VERIFIED recorded bulk offset "
              f"({recorded_mas[0]:+.1f},{recorded_mas[1]:+.1f}) mas against a fresh "
              f"measurement ({measured[0]:+.1f},{measured[1]:+.1f}) mas via "
              f"{bulk_source} -- separation {sep:.1f} mas. No change applied.")
        detail = 'recorded bulk offset verified; unchanged'

    result = BulkOffsetResult(
        mode=mode, measured_dra_mas=measured[0], measured_ddec_mas=measured[1],
        recorded_dra_mas=None if recorded_mas is None else float(recorded_mas[0]),
        recorded_ddec_mas=None if recorded_mas is None else float(recorded_mas[1]),
        sep_mas=sep, bulk_source=bulk_source, apply_ok=apply_ok,
        from_cache=False, hash=ghash, detail=detail)

    record = asdict(result)
    record['passed'] = result.passed
    record['context'] = context
    record['reference_id'] = reference_id
    record['n_frames'] = len(list(frame_paths))
    record['verified_at'] = _utcnow()
    save_step0_record(rec_path, record)
    return result


def _utcnow():
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


# ---------------------------------------------------------------------------
# where a field's bulk tie actually lives
# ---------------------------------------------------------------------------

#: The three states a field can be in, which must NOT collapse into two.
BULK_RECORDED = 'recorded'     # a constant in alignment_config -- verifiable here
BULK_IN_TABLE = 'in_table'     # carried by a locked/consensus offsets table
BULK_NONE = 'none'             # no config entry at all

#: Outcomes of looking up a field's recorded bulk tie.  These are DISTINCT on
#: purpose: only ``BULK_NOT_CONFIGURED`` means "genuinely new data", and only it
#: may lead to MEASURE.  A missing table file and a table with no matching row
#: are both "we cannot tell", which is an error -- returning a bare ``None`` for
#: all of them re-creates the undeterminable-vs-negative conflation this module
#: exists to remove.
BULK_OK = 'ok'
BULK_TABLE_ABSENT = 'table_absent'
BULK_NO_ROW = 'no_row'
BULK_NOT_CONFIGURED = 'not_configured'

#: Outcomes that only arise once more than one visit is in play.  A field whose
#: frames span several visits has no single "the" bulk tie, and summarising it by
#: whichever visit happens to sort first is a silent choice, not a measurement.
BULK_VISITS_MIXED = 'visits_mixed'          # some visits tied, others not
BULK_VISITS_DISAGREE = 'visits_disagree'    # all tied, but to different values


def bulk_tie_state(proposal_id, field):
    """Return one of ``BULK_RECORDED`` / ``BULK_IN_TABLE`` / ``BULK_NONE``.

    Most fields the pipeline spends its time on (brick, cloudc, sgrc, W51) carry
    their bulk tie inside an offsets table rather than as a constant.  Collapsing
    "tied, but by a table" into "nothing on record" would send those fields into
    MEASURE mode and print a recommendation to record a tie they already have --
    the same undeterminable-vs-negative conflation this module exists to avoid.
    """
    from jwst_gc_pipeline.reduction.alignment_config import (
        RECORDED_BULK, TABLE_CONSENSUS, TABLE_LOCKED, resolve,
    )
    cfg = resolve(proposal_id, field)
    if cfg is None:
        return BULK_NONE
    if cfg.source == RECORDED_BULK:
        return BULK_RECORDED
    if cfg.source in (TABLE_LOCKED, TABLE_CONSENSUS):
        return BULK_IN_TABLE
    return BULK_NONE


def recorded_bulk_mas(basepath, proposal_id, field, filtername, visit,
                      dec_deg, frame_name=None):
    """The bulk tie on record for this (visit, filter), as ``(value, status)``.

    ``value`` is ON-SKY mas, or ``None`` when there is nothing to compare
    against.  ``status`` says WHY, and the distinction is the point:

    ``BULK_OK``               a tie was found; verify against it
    ``BULK_TABLE_ABSENT``     the declared table file is not on disk
    ``BULK_NO_ROW``           the table exists but has no row for this key
    ``BULK_NOT_CONFIGURED``   no config entry -- the ONLY "genuinely new data"

    Collapsing the middle two into "nothing on record" is what would send a
    tied field into MEASURE and print a recommendation to record a tie it
    already has.  A zero-valued row is a real measurement and returns
    ``BULK_OK`` with ``(0.0, 0.0)`` -- it is not the same as no row.
    """
    from jwst_gc_pipeline.reduction import alignment_config as ac

    cfg = ac.resolve(proposal_id, field)
    if cfg is None:
        return None, BULK_NOT_CONFIGURED
    cosd = np.cos(np.radians(float(dec_deg)))
    filt = str(filtername).upper()

    if cfg.source == ac.RECORDED_BULK:
        # visit_key_for handles the per-proposal convention (2092 keys on the
        # 3-character visit suffix, not the full token).
        key = ac.visit_key_for(cfg, frame_name) if frame_name else visit
        dra, ddec, found = ac.lookup_recorded_bulk(cfg, key, filt)
        if not found:
            return None, BULK_NO_ROW
        return (dra * cosd * 1000.0, ddec * 1000.0), BULK_OK

    if cfg.source == ac.TABLE_LOCKED:
        path = (f'{basepath}/offsets/'
                f'Offsets_JWST_Brick{proposal_id}_VIRAC2locked.csv')
        if not os.path.exists(path):
            return None, BULK_TABLE_ABSENT
        from astropy.table import Table
        tbl = Table.read(path)
        if 'Visit' not in tbl.colnames:
            return None, BULK_NO_ROW
        sel = (tbl['Visit'] == visit) & (tbl['Filter'] == filt)
        if not sel.any():
            return None, BULK_NO_ROW
        # row presence decides, NOT the value -- a genuine (0, 0) tie is a
        # measurement, not an absence.
        dra = float(np.median(np.asarray(tbl['dra (arcsec)'][sel], dtype=float)))
        ddec = float(np.median(np.asarray(tbl['ddec (arcsec)'][sel], dtype=float)))
        return (dra * cosd * 1000.0, ddec * 1000.0), BULK_OK

    if cfg.source == ac.TABLE_CONSENSUS:
        path = (f'{basepath}/offsets/'
                f'Offsets_JWST_Brick{proposal_id}_consensus.csv')
        if not os.path.exists(path):
            return None, BULK_TABLE_ABSENT
        from astropy.table import Table
        from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
            BULK_EXPOSURE, BULK_MODULE,
        )
        tbl = Table.read(path)
        sel = ((tbl['Visit'] == visit) & (tbl['Filter'] == filt)
               & (tbl['Exposure'] == BULK_EXPOSURE)
               & (tbl['Module'] == BULK_MODULE))
        if int(sel.sum()) != 1:
            return None, BULK_NO_ROW
        row = tbl[sel]
        return (float(row['dra (arcsec)'][0]) * cosd * 1000.0,
                float(row['ddec (arcsec)'][0]) * 1000.0), BULK_OK
    return None, BULK_NOT_CONFIGURED


def recorded_bulk_over_visits(basepath, proposal_id, field, filtername,
                              frame_names, dec_deg):
    """Resolve the recorded bulk tie across EVERY visit present in the frames.

    Returns ``(value, status, per_visit)`` where ``per_visit`` maps each visit
    token to its own ``(value, status)``.

    Step 0 measures ONE bulk offset from all the supplied catalogs stacked
    together, so it can only verify a field that has one bulk tie.  Taking
    ``sorted(frames)[0]``'s visit and calling that "the" recorded value hides two
    real situations:

    * some visits are tied and others are not (cloudef 2092 obs002 is a two-visit
      mosaic where only visit 002 carries a recorded shift);
    * every visit is tied, but to genuinely different values (brick 1182 obs004
      visit 001 sits ~17" away from visit 002).

    In both cases a single stacked measurement cannot confirm the record, so this
    reports ``BULK_VISITS_MIXED`` / ``BULK_VISITS_DISAGREE`` rather than picking
    one and appearing to succeed.  Scope the run to one visit to proceed.
    """
    per_visit = {}
    for fn in frame_names:
        base = os.path.basename(fn)
        visit = base.split('_')[0]
        if visit in per_visit:
            continue
        per_visit[visit] = recorded_bulk_mas(
            basepath, proposal_id, field, filtername, visit, dec_deg,
            frame_name=base)

    if not per_visit:
        return None, BULK_NOT_CONFIGURED, per_visit

    statuses = {st for _, st in per_visit.values()}
    if len(per_visit) == 1 or statuses == {BULK_NOT_CONFIGURED}:
        (value, status), = list(per_visit.values())[:1]
        return value, status, per_visit

    if statuses != {BULK_OK}:
        # A mix of "tied" and "cannot tell" across visits.  If NOTHING resolved,
        # report the single shared reason instead of inventing a mixed state.
        if BULK_OK not in statuses and len(statuses) == 1:
            return None, statuses.pop(), per_visit
        return None, BULK_VISITS_MIXED, per_visit

    values = [value for value, _ in per_visit.values()]
    first = values[0]
    spread = max(float(np.hypot(v[0] - first[0], v[1] - first[1])) for v in values)
    if spread > _tol():
        return None, BULK_VISITS_DISAGREE, per_visit
    return first, BULK_OK, per_visit


# ---------------------------------------------------------------------------
# one catalog per exposure
# ---------------------------------------------------------------------------

#: ``f187n_nrcb1_visit001_vgroup03102_exp00001`` -- everything after this is the
#: catalog's STAGE tag (``_m1``, ``_group_m1``, ``_resbgsub_group_m5``, ...).
_EXPOSURE_RE = re.compile(
    r'^(?P<filt>[^_]+)_(?P<det>[^_]+)_visit(?P<visit>\d+)_'
    r'vgroup(?P<vgroup>[^_]+)_exp(?P<exp>\d{5})_(?P<stage>.+)$')


def default_catalog_glob(basepath, filtername, mtag='_m1'):
    """The per-frame catalog glob step 0 reads by default.

    Same shape as ``build_virac2_offsets._gather``: the filter token comes FIRST
    and the catalogs sit directly under ``<basepath>/<FILT>/``.  The exposure
    number is pinned to FIVE DIGITS rather than left as ``exp*``, because a greedy
    ``exp*`` also matches the ``_group_`` variant -- a second catalog of the SAME
    exposure -- and a field carrying both stages then feeds every star in twice.
    Lives here rather than in the script so a revert fails a test.
    """
    filt = str(filtername).upper()
    return (f'{basepath}/{filt}/{filt.lower()}_*_visit*_vgroup*_exp?????'
            f'{mtag}_daophot_basic.fits')


def applied_bulk_mas(frame_paths, dec_deg):
    """The offset ALREADY written into these frames' WCS, as on-sky mas.

    Returns ``(dra_mas, ddec_mas, n_with, n_total)``; ``(0.0, 0.0)`` when no frame
    carries one.  ``unified_alignment`` records the applied total in
    ``RAOFFSET``/``DEOFFSET`` (coordinate arcsec), so this is what makes the
    difference between "the catalogs are raw" and "the catalogs are already tied".

    Step 0's docstring says it reads catalogs built on raw-WCS frames, but on this
    archive that is often not the state on disk: brick F200W visit 001 carries an
    applied (-17.597, +13.453) arcsec and sickle F187N an applied (-0.102,
    -0.034) arcsec -- in both cases exactly the recorded bulk.  A fresh
    measurement on such catalogs is the RESIDUAL, near zero, so comparing it with
    the recorded bulk fails by the whole size of the bulk offset and reports a
    correctly-tied field as broken.  The caller has to subtract this to know what
    the measurement should be.

    The keyword is the per-exposure TOTAL (bulk + jitter), so the median across
    frames carries a jitter-sized uncertainty of tens of mas.
    """
    from astropy.io import fits

    dras, ddecs = [], []
    n_total = 0
    for path in frame_paths:
        n_total += 1
        try:
            with fits.open(path) as fh:
                dra = fh[1].header.get('RAOFFSET', fh[0].header.get('RAOFFSET'))
                ddec = fh[1].header.get('DEOFFSET', fh[0].header.get('DEOFFSET'))
        except (OSError, IndexError, KeyError):
            continue
        if dra is None or ddec is None:
            continue
        dras.append(float(dra))
        ddecs.append(float(ddec))
    if not dras:
        return 0.0, 0.0, 0, n_total
    cosd = np.cos(np.radians(float(dec_deg)))
    return (float(np.median(dras)) * cosd * 1000.0,
            float(np.median(ddecs)) * 1000.0, len(dras), n_total)


def default_frame_glob(basepath, filtername, proposal_id, field):
    """The frames whose WCS generation step 0 hashes, for ONE observation.

    Scoped to ``jw<proposal><obs>*`` rather than every ``*_destreak.fits`` in the
    filter directory.  A filter directory holds every observation of the
    proposal: cloudef F480M carries obs002 (two visits) alongside obs005, and an
    unscoped glob folded obs005's frames into obs002's generation hash and
    reported a third visit that had nothing to do with the requested field.
    """
    filt = str(filtername).upper()
    stem = f"jw{str(proposal_id).zfill(5)}{str(field).zfill(3)}"
    return f'{basepath}/{filt}/pipeline/{stem}*_destreak.fits'


def exposure_key(catalog_path):
    """``(filter, detector, visit, vgroup, exposure)`` for a per-frame catalog.

    Returns ``None`` when the name does not follow the pipeline's per-exposure
    convention, so a hand-supplied ``--catalog-glob`` with some other naming
    scheme is left alone rather than rejected.
    """
    m = _EXPOSURE_RE.match(os.path.basename(str(catalog_path)))
    if m is None:
        return None
    return (m.group('filt'), m.group('det'), m.group('visit'),
            m.group('vgroup'), m.group('exp'))


def duplicate_exposure_catalogs(catalog_paths):
    """Exposures represented more than once in ``catalog_paths``.

    Returns ``{exposure_key: [paths]}`` for every exposure with more than one
    catalog.  The pipeline writes several STAGES per exposure (``_m1`` alongside
    ``_group_m1``, ``_resbgsub_group_m5``, ...), and a glob loose enough to match
    two of them feeds the same stars in twice under two different measurements.
    That is not a harmless duplication: on sickle F187N the mixed set moved the
    measured tie from (-11.2, -106.8) to (+73.2, -70.6) mas -- ~60 mas against a
    100 mas tolerance -- so it has to be refused rather than averaged.
    """
    from collections import defaultdict
    seen = defaultdict(list)
    for path in catalog_paths:
        key = exposure_key(path)
        if key is not None:
            seen[key].append(path)
    return {k: v for k, v in seen.items() if len(v) > 1}


def reference_frame_matches_refcat(reference_frame, refcat_path):
    """Does the loaded reference catalog belong to the frame the field is tied to?

    A recorded tie is only comparable with a measurement made against the SAME
    absolute frame.  sickle records a GNS tie while the default refcat search
    finds a ``gaia_virac2_refcat*.fits``; measuring one against the other
    produces a real, large separation whose message ("the recorded value is
    wrong") points an operator at the wrong thing entirely.

    Returns ``(ok, detail)``.  Unknown/unrecognised refcat names return ``True``
    -- this must not become a new way to block a correct run.
    """
    from jwst_gc_pipeline.reduction.alignment_config import GAIA, GNS, VIRAC2
    name = os.path.basename(str(refcat_path)).lower()
    if 'gns' in name:
        found = GNS
    elif 'virac' in name:
        found = VIRAC2
    elif 'gaia' in name:
        found = GAIA
    else:
        return True, f"unrecognised refcat {name!r}; not checking the frame"
    if str(reference_frame) == found:
        return True, f"{found} refcat matches the configured frame"
    return False, (
        f"FRAME MISMATCH: this field's tie is recorded in the {reference_frame} "
        f"frame, but the reference catalog loaded is {found} ({name}). A recorded "
        f"tie can only be verified against a measurement in the SAME frame -- "
        f"comparing across frames produces a real separation that is not an "
        f"astrometry error. Pass --refcat pointing at a {reference_frame} catalog, "
        f"or re-record this field's bulk tie against {found}.")


#: Reference-catalog patterns searched under ``<basepath>/catalogs/``.
#: ``gaia_virac2_refcat*`` is the Galactic-Centre seed catalog; ``gaia_refcat*``
#: is what the fields outside the VVV/VIRAC2 footprint (W51, M4, M92) have.
REFCAT_PATTERNS = ('gaia_virac2_refcat*.fits', 'gaia_refcat*.fits')


def refcat_for_frame(basepath, reference_frame):
    """Pick the reference catalog that belongs to ``reference_frame``.

    Returns ``(path, candidates)``; ``path`` is ``None`` when nothing was found.
    Searching only ``gaia_virac2_refcat*.fits`` misses every field outside the
    VIRAC2 footprint -- W51, M4 and M92 carry ``gaia_refcat.fits`` and nothing
    else -- so those fields could not start without an explicit ``--refcat``.
    Since :func:`reference_frame_matches_refcat` now blocks a cross-frame
    comparison, choosing the catalog that matches the declared frame is what
    keeps the default path usable rather than merely safe.
    """
    import glob as _glob
    candidates = []
    for pattern in REFCAT_PATTERNS:
        candidates.extend(sorted(_glob.glob(os.path.join(
            str(basepath), 'catalogs', pattern))))
    # de-duplicate while preserving order (the patterns overlap)
    seen, ordered = set(), []
    for path in candidates:
        if path not in seen:
            seen.add(path)
            ordered.append(path)
    matching = [p for p in ordered
                if reference_frame_matches_refcat(reference_frame, p)[0]]
    if matching:
        return matching[-1], ordered
    return (ordered[-1] if ordered else None), ordered
