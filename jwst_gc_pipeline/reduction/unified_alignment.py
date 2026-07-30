"""One code path for resolving an exposure's astrometric shift, with the applied
shift split into its BULK and JITTER components.

The NIRCam ``fix_alignment`` used to decide where an exposure's shift came from
with a per-proposal ``if/elif`` chain whose ``else`` arm returned ``(0, 0)``.
This module replaces that chain: the *policy* (which frame, which source) lives
in :mod:`jwst_gc_pipeline.reduction.alignment_config`, and the *mechanism* (read
the table, narrow to this exposure, check the WCS generation) lives here, once,
for every NIRCam field.  MIRI and NIRISS keep their own dispatch -- see the
scope note in ``alignment_config``.

On the generation check: its strong layer compares per-row ``base_*`` stamps
against the frame's.  NOTHING WRITES THOSE COLUMNS YET, so in practice every
field currently falls through to the mtime fallback, which this module itself
calls WEAK.  The strong layer is wired and tested but dormant until a tie builder
stamps the rows.

Bulk / jitter
-------------

Every shift is reported as two components whose sum is exactly the total that
the old code applied::

    total = bulk + jitter

* **bulk** -- the visit-level tie to the absolute reference frame.  Arcsecond
  scale when a guide-star acquisition went wrong; measured once and stable.
* **jitter** -- the per-exposure residual around the visit consensus, tens of
  mas, re-measured on every re-tie iteration.

How the split is obtained depends on the source:

``RECORDED_BULK``
    All bulk by construction; jitter is exactly zero.
``TABLE_CONSENSUS``
    The table already separates them -- a per-visit BULK sentinel row plus
    sparse per-exposure JITTER rows.  The sentinel is read directly and the
    jitter is taken as ``total - bulk``, so the total stays bit-identical to
    what :func:`~jwst_gc_pipeline.photometry.astrometry_checkpoint.lookup_consensus_offset`
    returns.
``TABLE_LOCKED``
    The curated table stores only the total, so the split is DERIVED: bulk is
    the median over all rows sharing this ``(Visit, Filter)``, and jitter is the
    remainder.  For a per-visit table (one row per visit/filter) that puts
    everything in bulk and leaves jitter at zero, which is correct.  The split
    is a reporting convenience here -- ``total`` is what gets applied, and it is
    unchanged.

Keeping the components separate in the FITS header is what lets the staleness
guard stay sharp once bulk and jitter are re-solved on different cadences: a
guard that compares a stored *total* against a freshly computed *component*
degrades into noise, and that guard is what caught the brick-1182 v001 frames
carrying ``+1.9"`` while the table said ``-17.5"``.
"""

import os
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from astropy import units as u
from astropy.table import Table

from jwst_gc_pipeline.reduction import alignment_config as _cfgmod
from jwst_gc_pipeline.reduction.alignment_config import (
    RECORDED_BULK, TABLE_CONSENSUS, TABLE_LOCKED,
)

__all__ = [
    'AlignmentShift', 'resolve_shift',
    'write_alignment_header', 'check_alignment_stale',
    'BULK_RA_KEY', 'BULK_DEC_KEY', 'JITTER_RA_KEY', 'JITTER_DEC_KEY',
    'TOTAL_RA_KEY', 'TOTAL_DEC_KEY',
]

# FITS keywords.  The TOTAL keys keep their historical names so every existing
# reader (and the idempotency check) keeps working on frames written either way.
TOTAL_RA_KEY = 'RAOFFSET'
TOTAL_DEC_KEY = 'DEOFFSET'
BULK_RA_KEY = 'RAOFFBLK'
BULK_DEC_KEY = 'DEOFFBLK'
JITTER_RA_KEY = 'RAOFFJIT'
JITTER_DEC_KEY = 'DEOFFJIT'
SOURCE_KEY = 'ALIGNSRC'
FRAME_KEY = 'ALIGNREF'

# Offsets tables already collapse-checked in this process (warn once per file).
_VALIDATED_OFFSETS_TABLES = set()


@dataclass(frozen=True)
class AlignmentShift:
    """The shift to apply to one exposure, split into components (arcsec, in the
    Δα-COORDINATE convention that ``adjust_wcs(delta_ra=...)`` consumes)."""

    bulk_ra: float = 0.0
    bulk_dec: float = 0.0
    jitter_ra: float = 0.0
    jitter_dec: float = 0.0
    #: False when the field has no configured alignment at all.
    configured: bool = True
    #: False when the configured table does not exist yet.  Distinct from a
    #: table that exists and reports zero -- conflating "undeterminable" with
    #: "measured zero" is the failure mode this module exists to remove, and it
    #: recurs one level down without this flag.
    table_present: bool = True
    #: ``alignment_config`` source constant, or ``''`` when unconfigured.
    source: str = ''
    #: Absolute reference frame this tie is against.
    reference_frame: str = ''
    #: Offsets table actually consumed, for header provenance.
    prov_table: Optional[str] = None
    #: Checkpoint stage that last corrected the row, when the table records it.
    prov_stage: str = ''
    #: This frame's WCS-generation stamp, when it was read.
    frame_generation: Optional[dict] = None

    @property
    def total_ra(self) -> float:
        return self.bulk_ra + self.jitter_ra

    @property
    def total_dec(self) -> float:
        return self.bulk_dec + self.jitter_dec

    @property
    def ra_quantity(self):
        return self.total_ra * u.arcsec

    @property
    def dec_quantity(self):
        return self.total_dec * u.arcsec

    def __str__(self):
        return (f"total=({self.total_ra:+.4f},{self.total_dec:+.4f})\" "
                f"[bulk=({self.bulk_ra:+.4f},{self.bulk_dec:+.4f})\" "
                f"jitter=({self.jitter_ra:+.4f},{self.jitter_dec:+.4f})\"] "
                f"src={self.source or 'NONE'} ref={self.reference_frame or 'NONE'}")


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def resolve_shift(fn, proposal_id, field, filtername, module, basepath,
                  refname=None, use_average=True):
    """Return the :class:`AlignmentShift` for one exposure.

    ``refname`` is the legacy per-proposal reference name (``refnames[...]`` in
    the reduction script), needed only for the non-locked fallback table paths.
    It is passed in rather than imported so this module does not depend on the
    reduction script.
    """
    cfg = _cfgmod.resolve(proposal_id, field)
    if cfg is None:
        print(f"NO CONFIGURED ALIGNMENT for proposal={proposal_id} field={field} "
              f"({os.path.basename(fn)}): leaving this frame at the raw assign_wcs "
              f"frame (0,0). Any astrometry checkpoint correction written for this "
              f"field will NOT be applied -- add an entry to "
              f"jwst_gc_pipeline/reduction/alignment_config.py.", flush=True)
        return AlignmentShift(configured=False, table_present=False)

    if cfg.source == RECORDED_BULK:
        return _shift_from_recorded_bulk(fn, cfg, basepath, proposal_id,
                                         filtername, module)
    if cfg.source == TABLE_CONSENSUS:
        return _shift_from_consensus(fn, cfg, basepath, proposal_id, filtername,
                                     module)
    if cfg.source == TABLE_LOCKED:
        return _shift_from_locked(fn, cfg, basepath, proposal_id, filtername,
                                  refname=refname, use_average=use_average)
    raise ValueError(f"unknown alignment source {cfg.source!r} for proposal "
                     f"{proposal_id} field {field}")


def _shift_from_recorded_bulk(fn, cfg, basepath, proposal_id, filtername, module):
    visit = _cfgmod.visit_key_for(cfg, fn)
    dra, ddec, found = _cfgmod.lookup_recorded_bulk(cfg, visit, filtername)
    if not found and cfg.warn_on_missing:
        print(f"WARNING: no recorded bulk tie for ({visit}, "
              f"{str(filtername).upper()}); leaving {os.path.basename(fn)} at raw "
              f"frame (0,0)")

    # A recorded (hand-measured) bulk does not stop this field from running the
    # m2 re-tie loop.  The recorded constant is the STARTING bulk; the
    # checkpoint's sentinel is the residual consensus->reference tie measured on
    # top of it, so the two SUM into the field's bulk, and the per-exposure rows
    # remain the jitter.  Inert until the checkpoint has written a table.
    jit_ra = jit_dec = 0.0
    prov_table = 'alignment_config.py'
    if cfg.consensus_jitter:
        sent_ra, sent_dec, jit_ra, jit_dec, tbl_name = _consensus_correction(
            fn, basepath, proposal_id, filtername, module)
        dra += sent_ra
        ddec += sent_dec
        if tbl_name:
            prov_table = f'alignment_config.py + {tbl_name}'

    return AlignmentShift(bulk_ra=dra, bulk_dec=ddec,
                          jitter_ra=jit_ra, jitter_dec=jit_dec,
                          source=RECORDED_BULK,
                          reference_frame=cfg.reference_frame,
                          prov_table=prov_table)


def _consensus_correction(fn, basepath, proposal_id, filtername, module):
    """The checkpoint's correction for this exposure: BULK sentinel + JITTER row.

    Returns ``(sentinel_ra, sentinel_dec, jitter_ra, jitter_dec, table_basename)``;
    zeros and ``''`` when no table exists yet.

    The sentinel is INCLUDED, not skipped.  The checkpoint records corrections as
    RESIDUALS measured on frames that already carry whatever tie was applied last
    (``seed_offsets_table_from_consensus``: "the correction is the RESIDUAL after
    the previous tie").  So on a field whose bulk is a recorded constant, the
    sentinel is the *remaining* consensus->reference offset measured on top of
    that constant -- not a second copy of it.  Dropping it would leave the field
    with ``reference_frame=VIRAC2`` in the config and no path to VIRAC2 in the
    applied shift, and would make the checkpoint re-measure and re-add the same
    residual on every iteration: the exact non-convergence this whole change
    exists to remove, one level up.  It would also make this reader and
    ``lookup_consensus_offset`` disagree about the same frame in the same table.
    """
    tblfn = (f'{basepath}/offsets/'
             f'Offsets_JWST_Brick{proposal_id}_consensus.csv')
    if not os.path.exists(tblfn):
        return 0.0, 0.0, 0.0, 0.0, ''
    sent_ra, sent_dec, total_ra, total_dec, _ = _read_consensus(
        tblfn, fn, filtername)
    return (sent_ra, sent_dec, total_ra - sent_ra, total_dec - sent_dec,
            os.path.basename(tblfn))


def _read_consensus(tblfn, fn, filtername):
    """Split a consensus table row set into its BULK and TOTAL parts.

    Returns ``(bulk_ra, bulk_dec, total_ra, total_dec, prov_stage)``.  TOTAL
    comes from the shared ``lookup_consensus_offset`` so this stays bit-identical
    to what the checkpoint itself computes; BULK is the per-visit sentinel row,
    read directly.  Callers derive jitter as ``total - bulk``.
    """
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        BULK_EXPOSURE, BULK_MODULE, lookup_consensus_offset,
    )

    tbl = Table.read(tblfn)
    base = os.path.basename(fn)
    visit = base.split('_')[0]
    exposure = int(base.split('_')[-3])
    thismodule = base.split('_')[-2]
    # exposure numbers restart per visit group, so the group is part of the
    # identity -- narrowing the jitter lookup by it (#183)
    vgroup = base.split('_')[1]

    try:
        total_ra, total_dec = lookup_consensus_offset(
            tbl, visit, exposure, thismodule, filtername, vgroup=vgroup)
    except ValueError as ex:
        raise ValueError(f"{ex} [table={tblfn}, frame={base}]") from ex

    bulk_ra = bulk_dec = 0.0
    sel = ((tbl['Visit'] == visit) & (tbl['Filter'] == filtername)
           & (tbl['Exposure'] == BULK_EXPOSURE) & (tbl['Module'] == BULK_MODULE))
    nb = int(sel.sum())
    if nb > 1:
        raise ValueError(f"consensus BULK match={nb} for visit={visit} "
                         f"filt={filtername} in {tblfn}; expected <=1 row")
    if nb == 1:
        row = tbl[sel]
        bulk_ra = float(row['dra (arcsec)'][0])
        bulk_dec = float(row['ddec (arcsec)'][0])

    prov_stage = ''
    if 'prov_stage' in tbl.colnames and nb == 1:
        prov_stage = str(tbl[sel]['prov_stage'][0])
    return bulk_ra, bulk_dec, total_ra, total_dec, prov_stage


def _shift_from_consensus(fn, cfg, basepath, proposal_id, filtername, module):
    """Consensus table: BULK sentinel row + sparse per-exposure JITTER rows."""
    tblfn = (f'{basepath}/offsets/'
             f'Offsets_JWST_Brick{proposal_id}_consensus.csv')
    if not os.path.exists(tblfn):
        print(f"[consensus] no table {tblfn} yet; leaving "
              f"{os.path.basename(fn)} at frame (0,0)")
        return AlignmentShift(source=TABLE_CONSENSUS,
                              reference_frame=cfg.reference_frame,
                              prov_table=os.path.basename(tblfn),
                              table_present=False, prov_stage='NO_TABLE')

    bulk_ra, bulk_dec, total_ra, total_dec, prov_stage = _read_consensus(
        tblfn, fn, filtername)

    return AlignmentShift(
        bulk_ra=bulk_ra, bulk_dec=bulk_dec,
        jitter_ra=total_ra - bulk_ra, jitter_dec=total_dec - bulk_dec,
        source=TABLE_CONSENSUS, reference_frame=cfg.reference_frame,
        prov_table=os.path.basename(tblfn), prov_stage=prov_stage)


def _shift_from_locked(fn, cfg, basepath, proposal_id, filtername,
                       refname=None, use_average=True):
    """Curated VIRAC2-locked table (per-visit or per-exposure)."""
    base = os.path.basename(fn)
    exposure = int(base.split('_')[-3])
    thismodule = base.split('_')[-2]
    visit = base.split('_')[0]
    # jw<prop><obs><visit>_<VGROUP>_<exposure>_<detector>_...: the exposure
    # number restarts per visit group, so the group is part of the identity.
    vgroup = base.split('_')[1]

    locked_tbl = (f'{basepath}/offsets/'
                  f'Offsets_JWST_Brick{proposal_id}_VIRAC2locked.csv')

    if os.path.exists(locked_tbl):
        offsets_tbl = Table.read(locked_tbl)
        frame_gen = _check_generation(fn, offsets_tbl, locked_tbl)
        _validate_once(offsets_tbl, locked_tbl)

        match = ((offsets_tbl['Visit'] == visit)
                 & (offsets_tbl['Filter'] == filtername))
        # Support BOTH conventions: per-VISIT tables (1 row/visit, no usable
        # Exposure) and per-EXPOSURE tables (N rows/visit).  Narrow by Exposure
        # only when >1 row matches.
        if match.sum() > 1 and 'Exposure' in offsets_tbl.colnames:
            match = match & (offsets_tbl['Exposure'] == exposure)
        # Per-MODULE narrowing (default OFF: filters lock NRCA==NRCB together).
        # Documented exception: F410M, whose filter-specific distortion leaves
        # NRCALONG ~40 mas inconsistent with NRCBLONG.
        if match.sum() > 1 and 'Module' in offsets_tbl.colnames:
            match = match & ((offsets_tbl['Module'] == thismodule)
                             | (offsets_tbl['Module'] == thismodule.strip('1234')))
        # Per-VGROUP narrowing (#183).  A visit can dither across several visit
        # groups (physically disjoint sky tiles) and the exposure number RESTARTS
        # in each, so (visit, exposure) alone is ambiguous -- cloudc has 2 groups
        # in every filter, gc2211 has 6.
        #
        # UNCONDITIONAL, unlike the Exposure/Module narrowing above: a lone
        # surviving row is exactly the dangerous case.  If the table carries a row
        # for the OTHER group's exposure N and none for this one, `match.sum() == 1`
        # and a `> 1` guard would silently apply a DIFFERENT pointing's shift.
        # Narrow always and let the != 1 check raise.  An EMPTY Vgroup cell
        # predates the column (or was preserved by the builder's field-safe merge)
        # and still applies -- see vgroup_row_matches.
        if 'Vgroup' in offsets_tbl.colnames:
            from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
                vgroup_row_matches)
            match = match & np.array([vgroup_row_matches(g, vgroup)
                                      for g in offsets_tbl['Vgroup']])
        if match.sum() != 1:
            raise ValueError(f"module-locked offset match={match.sum()} for {fn} "
                             f"(visit={visit}, exposure={exposure}, "
                             f"vgroup={vgroup}, filter={filtername}); expected "
                             f"exactly 1 row in {locked_tbl}")
        row = offsets_tbl[match]
        _assert_generation_row(fn, row, frame_gen, offsets_tbl)

        total_ra = float(row['dra (arcsec)'][0])
        total_dec = float(row['ddec (arcsec)'][0])
        bulk_ra, bulk_dec = _derive_locked_bulk(offsets_tbl, visit, filtername)
        prov_stage = (str(row['prov_stage'][0])
                      if 'prov_stage' in offsets_tbl.colnames else '')
        print(f"MODULE-LOCKED per-visit offset for {fn}: "
              f"({total_ra} arcsec, {total_dec} arcsec)")
        return AlignmentShift(
            bulk_ra=bulk_ra, bulk_dec=bulk_dec,
            jitter_ra=total_ra - bulk_ra, jitter_dec=total_dec - bulk_dec,
            source=TABLE_LOCKED, reference_frame=cfg.reference_frame,
            prov_table=locked_tbl, prov_stage=prov_stage,
            frame_generation=frame_gen)

    # -- fallbacks: the pre-locked average / per-exposure tables --
    if refname is None:
        raise ValueError(
            f"no locked table {locked_tbl} and no refname supplied for "
            f"proposal {proposal_id}; cannot resolve a fallback offsets table")
    if 'bug' in refname.lower():
        raise ValueError("This is a disallowed reference file")

    if use_average:
        tblfn = (f'{basepath}/offsets/'
                 f'Offsets_JWST_Brick{proposal_id}_{refname}_average.csv')
        print(f"Using average offset table {tblfn}")
        offsets_tbl = Table.read(tblfn)
        match = (((offsets_tbl['Module'] == thismodule)
                  | (offsets_tbl['Module'] == thismodule.strip('1234')))
                 & (offsets_tbl['Filter'] == filtername))
        if 'Visit' in offsets_tbl.colnames:
            match &= (offsets_tbl['Visit'] == visit)
    else:
        tblfn = (f'{basepath}/offsets/'
                 f'Offsets_JWST_Brick{proposal_id}_{refname}.csv')
        print(f"Using offset table {tblfn}")
        offsets_tbl = Table.read(tblfn)
        match = ((offsets_tbl['Visit'] == visit)
                 & (offsets_tbl['Exposure'] == exposure)
                 & ((offsets_tbl['Module'] == thismodule)
                    | (offsets_tbl['Module'] == thismodule.strip('1234')))
                 & (offsets_tbl['Filter'] == filtername))

    if match.sum() != 1:
        raise ValueError(f"too many or too few matches for {fn} "
                         f"(match.sum() = {match.sum()}).  exposure={exposure}, "
                         f"thismodule={thismodule}, filtername={filtername}")
    row = offsets_tbl[match]
    total_ra = float(row['dra (arcsec)'][0])
    total_dec = float(row['ddec (arcsec)'][0])
    bulk_ra, bulk_dec = _derive_locked_bulk(offsets_tbl, visit, filtername)
    prov_stage = (str(row['prov_stage'][0])
                  if 'prov_stage' in offsets_tbl.colnames else '')
    return AlignmentShift(
        bulk_ra=bulk_ra, bulk_dec=bulk_dec,
        jitter_ra=total_ra - bulk_ra, jitter_dec=total_dec - bulk_dec,
        source=TABLE_LOCKED, reference_frame=cfg.reference_frame,
        prov_table=tblfn, prov_stage=prov_stage)


def _derive_locked_bulk(offsets_tbl, visit, filtername):
    """Bulk component of a locked table: the median over every row sharing this
    ``(Visit, Filter)``.

    A per-visit table has exactly one such row, so bulk == total and jitter is
    zero.  A per-exposure table spreads its rows around the visit-level tie, and
    the median is that tie.  Keyed on (Visit, Filter) rather than Visit alone
    because filters legitimately differ here -- F410M carries its own per-module
    rows for a filter-specific distortion term that is not a pointing error.
    """
    if 'Visit' not in offsets_tbl.colnames:
        return 0.0, 0.0
    sel = (offsets_tbl['Visit'] == visit) & (offsets_tbl['Filter'] == filtername)
    if not sel.any():
        return 0.0, 0.0
    return (float(np.median(np.asarray(offsets_tbl['dra (arcsec)'][sel], dtype=float))),
            float(np.median(np.asarray(offsets_tbl['ddec (arcsec)'][sel], dtype=float))))


def _validate_once(offsets_tbl, locked_tbl):
    """One-time collapse check per table per process.

    The ad-hoc VIRAC2locked curation once overwrote brick-1182 visit-001's
    offset with visit-002's (both ~+1.9" for a visit truly ~20" off), so warn
    when distinct visits share a value.
    """
    if locked_tbl in _VALIDATED_OFFSETS_TABLES:
        return
    _VALIDATED_OFFSETS_TABLES.add(locked_tbl)
    from jwst_gc_pipeline.reduction.validate_offsets_table import (
        assert_offsets_table_sane,
    )
    assert_offsets_table_sane(offsets_tbl, context=os.path.basename(locked_tbl))


def _check_generation(fn, offsets_tbl, locked_tbl):
    """Read this frame's WCS-generation stamp and run the weak mtime fallback.

    A correction is only valid on the WCS GENERATION it was solved against.  The
    strong check (per-row ``base_*`` stamps) runs in :func:`_assert_generation_row`
    once the row is known; this does the frame-side read plus the mtime fallback
    used when the table carries no stamps.
    """
    frame_gen = None
    try:
        from jwst_gc_pipeline.astrometry_utils import generation_stamp
        from astropy.io import fits
        with fits.open(fn) as gfh:
            hdr0 = dict(gfh[0].header)
            hdr0.update({k: v for k, v in gfh[1].header.items()
                         if k in ('DVACORR',)})
            frame_gen = generation_stamp(hdr0)
    except (OSError, KeyError, IndexError) as gex:
        print(f"[genlock] could not read generation keys from {fn}: {gex}")

    has_stamps = all(f'base_{col}' in offsets_tbl.colnames
                     for col, _ in _GENERATION_COLUMNS)
    if not has_stamps:
        try:
            t_tbl = os.path.getmtime(locked_tbl)
            t_crf = os.path.getmtime(fn)
        except OSError:
            t_tbl = t_crf = None
        if t_tbl is not None and t_tbl < t_crf - 1.0:
            gmsg = (f"[genlock] offsets table {os.path.basename(locked_tbl)} has no "
                    f"base_* generation stamps and predates crf "
                    f"{os.path.basename(fn)}; the tie may be a reduction "
                    f"generation behind (mtime is a WEAK proxy -- rebuild the "
                    f"table with the stamping builders for a real check).")
            if os.environ.get('GENLOCK_STRICT'):
                raise RuntimeError(gmsg)
            print("WARNING: " + gmsg, flush=True)
    return frame_gen


#: Generation stamp columns, as ``(table column suffix, generation_stamp key)``.
#: These names DIVERGE: the tie builders write ``base_calver`` while
#: ``generation_stamp`` lowercases ``CAL_VER`` to ``cal_ver``.  The check used to
#: index the stamp with the COLUMN spelling, so the moment a table carried the
#: stamps the strongest generation layer would have died on ``KeyError: 'calver'``
#: instead of comparing anything.  It never fired only because nothing populates
#: the columns yet.
_GENERATION_COLUMNS = (('calver', 'cal_ver'),
                       ('crds_ctx', 'crds_ctx'),
                       ('dvacorr', 'dvacorr'))


def _assert_generation_row(fn, row, frame_gen, offsets_tbl):
    """Hard-fail when the matched row was solved on a different WCS generation."""
    has_stamps = all(f'base_{col}' in offsets_tbl.colnames
                     for col, _ in _GENERATION_COLUMNS)
    if not (has_stamps and frame_gen is not None):
        return
    mismatch = {col: (str(row[f'base_{col}'][0]), frame_gen[key])
                for col, key in _GENERATION_COLUMNS
                if str(row[f'base_{col}'][0]) not in ('', 'nan')
                and str(row[f'base_{col}'][0]) != frame_gen[key]}
    if not mismatch:
        return
    gmsg = (f"[genlock] GENERATION MISMATCH for {fn}: the tie row was solved on "
            f"{mismatch} (base vs frame). Applying it would stack a stale "
            f"correction on a moved frame. Rebuild the VIRAC2locked table on THIS "
            f"generation (GENLOCK_ALLOW_MISMATCH=1 to override).")
    if os.environ.get('GENLOCK_ALLOW_MISMATCH') == '1':
        print("WARNING (override): " + gmsg, flush=True)
    else:
        raise RuntimeError(gmsg)


# ---------------------------------------------------------------------------
# Header write / staleness guard
# ---------------------------------------------------------------------------

def write_alignment_header(header, shift: AlignmentShift):
    """Record the applied shift and its components in ``header``.

    The TOTAL keywords keep their historical names and meaning, so nothing that
    reads ``RAOFFSET`` needs to change; the component keywords are additive.
    """
    header[TOTAL_RA_KEY] = (shift.total_ra, 'arcsec, total applied dRA (bulk+jitter)')
    header[TOTAL_DEC_KEY] = (shift.total_dec, 'arcsec, total applied dDec (bulk+jitter)')
    header[BULK_RA_KEY] = (shift.bulk_ra, 'arcsec, bulk (visit->reference) dRA')
    header[BULK_DEC_KEY] = (shift.bulk_dec, 'arcsec, bulk (visit->reference) dDec')
    header[JITTER_RA_KEY] = (shift.jitter_ra, 'arcsec, per-exposure jitter dRA')
    header[JITTER_DEC_KEY] = (shift.jitter_dec, 'arcsec, per-exposure jitter dDec')
    # Always written, including 'NONE'.  Recording "this frame is tied to
    # nothing" as the ABSENCE of a card is the same record-an-absence pattern
    # this module retires elsewhere; a greppable positive is free.
    header[SOURCE_KEY] = (shift.source or 'NONE', 'alignment shift source')
    header[FRAME_KEY] = (shift.reference_frame or 'NONE', 'absolute reference frame')
    if not shift.table_present:
        header['ALIGNTBL'] = ('ABSENT', 'configured offsets table did not exist')
    return header


def check_alignment_stale(header, shift: AlignmentShift, fn, tol_arcsec=None):
    """Compare a frame's baked-in offsets against what would be applied now.

    Returns a description of the disagreement, or ``None`` when the frame is
    current.  When the frame carries the component keywords the comparison is
    made PER COMPONENT -- a bulk that has been re-measured and a jitter that has
    been re-solved are then distinguishable, and a bulk change can no longer be
    masked by an opposite jitter change summing back to the same total.  Frames
    written before the split fall back to comparing totals.
    """
    if tol_arcsec is None:
        tol_arcsec = float(os.environ.get('RAOFFSET_DISAGREE_TOL_ARCSEC', 0.05))

    if TOTAL_RA_KEY not in header:
        return None

    has_components = BULK_RA_KEY in header and JITTER_RA_KEY in header
    if has_components:
        # .get(..., nan) for the SAME reason the total branch uses it: a frame
        # with only some component cards must be reported STALE (non-finite ->
        # not within tolerance), not abort fix_alignment with a KeyError that
        # says nothing about astrometry.
        pairs = (
            ('bulk RA', float(header.get(BULK_RA_KEY, np.nan)), shift.bulk_ra),
            ('bulk Dec', float(header.get(BULK_DEC_KEY, np.nan)), shift.bulk_dec),
            ('jitter RA', float(header.get(JITTER_RA_KEY, np.nan)), shift.jitter_ra),
            ('jitter Dec', float(header.get(JITTER_DEC_KEY, np.nan)), shift.jitter_dec),
        )
    else:
        pairs = (
            ('total RA', float(header[TOTAL_RA_KEY]), shift.total_ra),
            ('total Dec', float(header.get(TOTAL_DEC_KEY, np.nan)), shift.total_dec),
        )

    bad = [(name, baked, now) for name, baked, now in pairs
           if not np.isfinite(baked) or abs(baked - now) > tol_arcsec]
    if not bad:
        return None

    detail = '; '.join(f"{name}: baked {baked:+.4f}\" vs now {now:+.4f}\" "
                       f"(diff {abs(baked - now):.4f}\")"
                       for name, baked, now in bad)
    scope = 'per-component' if has_components else 'total-only (pre-split frame)'
    return (f"STALE ASTROMETRY: {fn} disagrees with the current alignment "
            f"[{scope}, tol {tol_arcsec}\"] -- {detail}. This frame was built from "
            f"an OLD table and the skip-if-present guard is hiding it. Regenerate "
            f"the working copy from _cal (destreak overwrite) so the offsets reset "
            f"and the current table is applied, OR set FORCE_REALIGN_ON_DISAGREE=1 "
            f"to re-apply now.")


def warn_or_raise_if_stale(header, shift: AlignmentShift, fn):
    """Apply the staleness policy: warn, or hard-stop under
    ``FORCE_REALIGN_ON_DISAGREE=1``."""
    msg = check_alignment_stale(header, shift, fn)
    if msg is None:
        return
    if os.environ.get('FORCE_REALIGN_ON_DISAGREE') == '1':
        raise RuntimeError(
            msg + " [FORCE_REALIGN_ON_DISAGREE=1: refusing to silently keep a "
            "stale frame; regenerate it from _cal.]")
    warnings.warn(msg)
