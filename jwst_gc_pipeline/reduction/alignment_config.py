"""Per-field astrometric alignment configuration for the NIRCam reduction --
the single source of truth for HOW each (proposal, observation) is tied to an
absolute reference frame.

SCOPE: NIRCam only.  ``PipelineMIRI.fix_alignment`` and
``PipelineRerunNIRISS.fix_alignment`` still carry their own dispatch and their
own inline policy constants (MIRI keeps a ``_PER_VISIT_SHIFT`` map and a w51
rule; neither writes the component keywords nor runs the staleness guard).
Folding those in is follow-up work -- until then, do not read this file as
repo-wide.

This replaces the per-proposal ``if/elif`` chain that used to live inside
``PipelineRerunNIRCAM-LONG.fix_alignment``.  That chain had grown one branch per
field, each with its own table convention, its own reference frame, and in
several cases hardcoded constants -- and, critically, an ``else`` arm that
returned ``(0, 0)``.  Any proposal without an explicit branch was therefore
silently left at the raw ``assign_wcs`` frame with NO alignment at all, while
the m2 astrometry checkpoint happily measured residuals and wrote corrections
into an offsets table that nothing read (arches/2045, quintuplet/2045,
sgrb2/5365, cloudef/2092 obs 005 all sat in this state; a re-tie loop on such a
field re-measures the identical residual forever and never converges).

The configuration here makes that failure impossible to reach silently: a field
either has an entry, or ``resolve()`` returns ``None`` and the caller is
required to say so out loud.

Two orthogonal pieces of information per field
----------------------------------------------

**Reference frame** -- WHICH absolute frame this field is tied to.  Galactic
Centre fields use VIRAC2 (Gaia is the *frame* but far too sparse to be the
reference *catalog* in the GC -- see CLAUDE.md); halo/disk clusters outside the
VVV footprint use Gaia directly.  The frame is therefore configured per field.

**Shift source** -- WHERE the numbers come from:

``TABLE_LOCKED``
    A curated per-visit (or per-exposure) ``Offsets_JWST_Brick<prop>_VIRAC2locked.csv``.
``TABLE_CONSENSUS``
    ``Offsets_JWST_Brick<prop>_consensus.csv``, seeded and upserted by the m2
    astrometry checkpoint.  Carries BOTH row kinds: a per-visit BULK sentinel
    row and sparse per-exposure JITTER rows.
``RECORDED_BULK``
    A bulk offset already measured and recorded here as a constant.  These are
    pure bulk -- no per-exposure jitter term -- and are the entries that pipeline
    step 0 verifies, leaving the recorded constant in place
    (``jwst_gc_pipeline.reduction.bulk_offset_step0``).

Bulk vs jitter
--------------

Every applied shift decomposes into

    total = bulk + jitter

* **bulk** -- the field/visit-level tie to the absolute reference frame.  Large
  (arcseconds when a guide-star acquisition went wrong), known once, and stable.
* **jitter** -- the small per-exposure residual around the visit consensus,
  tens of mas, re-measured every re-tie iteration.

``RECORDED_BULK`` fields are all bulk.  Consensus tables carry the split
explicitly.  Locked tables carry only the total, so the split is *derived* (see
``unified_alignment``) -- in every case the total is preserved exactly.
"""

import os
from dataclasses import dataclass, field as _dc_field
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    'ANY', 'VIRAC2', 'GAIA', 'GNS',
    'TABLE_LOCKED', 'TABLE_CONSENSUS', 'RECORDED_BULK',
    'BulkEntry', 'FieldAlignment', 'resolve', 'lookup_recorded_bulk',
    'ALIGNMENT_CONFIG', 'offsets_channel', 'offsets_table_path',
    'CHANNEL_LOCKED', 'CHANNEL_CONSENSUS', 'CHANNEL_NONE',
]

#: Wildcard for a ``recorded_bulk`` key component (matches any visit / filter).
ANY = '*'

# ---------------------------------------------------------------------------
# Reference frames
# ---------------------------------------------------------------------------
VIRAC2 = 'VIRAC2'
GAIA = 'Gaia'
GNS = 'GNS'

# ---------------------------------------------------------------------------
# Shift sources
# ---------------------------------------------------------------------------
TABLE_LOCKED = 'locked'
TABLE_CONSENSUS = 'consensus'
RECORDED_BULK = 'recorded_bulk'


@dataclass(frozen=True)
class BulkEntry:
    """One recorded bulk offset.

    ``onsky_mas=False`` (default): ``dra``/``ddec`` are arcsec in the
    Δα-COORDINATE convention that ``adjust_wcs(delta_ra=...)`` consumes, i.e.
    exactly what gets applied.

    ``onsky_mas=True``: ``dra``/``ddec`` are ON-SKY milliarcsec, as measured.
    The RA term is divided by ``cos(dec_ref_deg)`` to convert to the coordinate
    convention; the Dec term is 1:1.  Offsets are stored this way when the
    recorded value came straight off a sky measurement, so the number in this
    file is the number in the audit note.
    """

    dra: float
    ddec: float
    onsky_mas: bool = False


@dataclass(frozen=True)
class FieldAlignment:
    """How one (proposal, observation) is aligned."""

    proposal: str
    #: Observation numbers this applies to; ``None`` = every observation of the
    #: proposal.
    fields: Optional[Tuple[str, ...]]
    #: Absolute reference frame (``VIRAC2`` / ``GAIA`` / ``GNS``).
    reference_frame: str
    #: One of ``TABLE_LOCKED`` / ``TABLE_CONSENSUS`` / ``RECORDED_BULK``.
    source: str
    #: ``RECORDED_BULK`` only: ``{(visit, filter): BulkEntry}``, either key
    #: component may be :data:`ANY`.
    recorded_bulk: Dict[Tuple[str, str], BulkEntry] = _dc_field(default_factory=dict)
    #: ``RECORDED_BULK`` only: how to derive the visit key from the filename.
    #: ``'full'`` -> ``jw01979002001``; ``'suffix3'`` -> ``002``.
    visit_key: str = 'full'
    #: ``RECORDED_BULK`` only: reference declination for the on-sky -> coordinate
    #: RA conversion of ``onsky_mas`` entries.
    dec_ref_deg: Optional[float] = None
    #: ``RECORDED_BULK`` only: print a warning when an exposure has no entry
    #: (it is then left at the raw frame).
    warn_on_missing: bool = False
    #: Read per-exposure JITTER rows from the consensus table even when the BULK
    #: comes from somewhere else (a recorded constant).  This is what lets a
    #: field with a hand-measured bulk still run the m2 re-tie loop: bulk stays
    #: fixed and only the small per-exposure term is re-solved each iteration.
    #: Implied for ``TABLE_CONSENSUS`` (its table carries both row kinds).
    consensus_jitter: bool = False
    #: Band whose visit consensus defines this field's internal frame.  The
    #: consensus catalog is dense and already tied to the reference, so it is a
    #: valid frame for every band of the field; the choice is about which band
    #: gives the best-measured consensus.  Usually F212N / F210M / F200W -- bright,
    #: uncrowded enough to centroid well, and present in most GC programs.
    reference_filter: Optional[str] = None
    #: Free-text provenance -- why these numbers, measured when/how.
    notes: str = ''

    def matches(self, proposal_id, field) -> bool:
        if str(proposal_id) != self.proposal:
            return False
        return self.fields is None or str(field) in self.fields


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------
# NOTE ON GC FIELDS AND GNS: sickle (3958) is a Galactic Centre field whose
# recorded bulk is currently in the GNS frame, while ``refnames`` already calls
# it VIRAC2 -- a live inconsistency inherited from the old dispatch.  Policy is
# that GC fields tie to VIRAC2, so this entry is slated for re-measurement
# against VIRAC2 by pipeline step 0.  Until that measurement exists the recorded
# GNS numbers are kept EXACTLY as they were applied, so this refactor changes no
# pixel; ``reference_frame`` records the truth (GNS) rather than the aspiration.

ALIGNMENT_CONFIG = (

    # -- Galactic Centre: curated VIRAC2-locked per-visit/per-exposure tables --
    FieldAlignment(
        proposal='1182', fields=('004',),
        reference_frame=VIRAC2, source=TABLE_LOCKED,
        notes='brick. Module-locked per-visit tie; see build_virac2_locked_perexp.py.',
    ),
    FieldAlignment(
        proposal='2221', fields=('001', '002'),
        reference_frame=VIRAC2, source=TABLE_LOCKED,
        notes=('brick (001) + cloudc (002). 002 routed here 2026-06-22, replacing '
               'the deprecated F405N-crowdsource frame (~90 mas off Gaia).'),
    ),

    # -- Galactic Centre: m2-checkpoint consensus tables --
    FieldAlignment(
        proposal='4147', fields=None,
        reference_frame=VIRAC2, source=TABLE_LOCKED,
        reference_filter='F212N',
        notes=('sgrc. Per-exposure re-tie; tweakreg is skipped, so without this '
               'the exposures scatter ~2-8 mas around the visit consensus. '
               'DECLARED LOCKED, not consensus: sgrc has a build_virac2_offsets '
               'REGION entry, so its authored table is '
               'Offsets_JWST_Brick4147_VIRAC2locked.csv (96 rows, 8 filters, '
               'builder-shaped) -- and that is the only table on disk. The old '
               'TABLE_CONSENSUS declaration pointed the reducer at a '
               '_consensus.csv the checkpoint never wrote, which is why sgrc '
               'frames came out of a full reduce at RAOFFSET=0.0.'),
    ),
    FieldAlignment(
        proposal='6151', fields=None,
        reference_frame=GAIA, source=TABLE_CONSENSUS,
        reference_filter='F200W',
        notes=('w51. Same class as sgrc; outside the VVV/VIRAC2 footprint, so the '
               'bulk sentinel ties the consensus to gaia_refcat.fits.'),
    ),
    FieldAlignment(
        proposal='2045', fields=('001',),
        reference_frame=VIRAC2, source=TABLE_CONSENSUS,
        reference_filter='F212N',
        notes=('arches. Had NO alignment source at all: 2045 was absent from the '
               'old dispatch, so every frame fell to the else and got (0,0) while '
               'the m2 checkpoint wrote an 86-row consensus table nothing read. '
               'The re-tie loop returned 86 corrections on three consecutive '
               'iterations because the corrections never reached the frames.'),
    ),
    FieldAlignment(
        proposal='2045', fields=('003',),
        reference_frame=VIRAC2, source=TABLE_LOCKED,
        reference_filter='F212N',
        notes=('quintuplet. Same 2045 dispatch gap as arches, but a DIFFERENT '
               'table: quintuplet has a build_virac2_offsets REGION entry and a '
               '24-row builder-shaped VIRAC2locked table on disk (visit '
               'jw02045003001, F212N+F323N), whereas arches has neither and only '
               'a checkpoint-written consensus table. Same proposal, two sources '
               '-- which is why these are separate per-observation entries.'),
    ),
    FieldAlignment(
        proposal='5365', fields=None,
        reference_frame=VIRAC2, source=TABLE_LOCKED,
        reference_filter='F212N',
        notes=('sgrb2. Absent from the old dispatch -> unaligned, while its '
               'builder-written VIRAC2locked table holds 264 real rows across 11 '
               'filters (median |offset| ~126 mas) that nothing read. 14 filter '
               'directories, so the reference band matters most here.'),
    ),
    FieldAlignment(
        proposal='2211', fields=None,
        reference_frame=VIRAC2, source=TABLE_LOCKED,
        reference_filter='F200W',
        notes=('gc2211. Absent from the old dispatch -> unaligned, so its '
               'Offsets_JWST_Brick2211_VIRAC2locked.csv (per-exposure, m2-written, '
               'arcsecond-scale ties) was read by nothing. Five observations '
               '(023/028/046/049/050) that all reduce to visit001 and reuse vgroup '
               'ids, so the table separates them by Visit -- one proposal-wide entry '
               'covers all five. Rebuild is pending --per-module + Vgroup.'),
    ),
    FieldAlignment(
        proposal='2092', fields=('005',),
        reference_frame=VIRAC2, source=TABLE_LOCKED,
        reference_filter='F210M',
        notes=('cloudef obs005. Only obs002 had a branch in the old dispatch, so '
               'obs005 fell through to the else -- even though the shared 2092 '
               'VIRAC2locked table already carries 32 jw02092005001 rows across '
               'all four filters. No F212N in this program, so the reference band '
               'is F210M.'),
    ),

    # -- Recorded bulk offsets (pure bulk, no jitter term) --
    FieldAlignment(
        proposal='2092', fields=('002',),
        reference_frame=VIRAC2, source=RECORDED_BULK,
        visit_key='suffix3', consensus_jitter=True, reference_filter='F210M',
        recorded_bulk={
            ('002', ANY): BulkEntry(0.098, -0.171),
        },
        notes=('Cloud E/F obs002, a 2-visit mosaic. visit002 - visit001 = '
               '(dRA -98, dDec +171) mas measured on 277 matched stars in the '
               'F480M nrcblong overlap (2026-06-10); a PURE translation '
               '(gradients <=0.3 mas/arcsec). Brings visit 002 onto visit 001; '
               'the absolute zero point comes from the subsequent tie.'),
    ),
    FieldAlignment(
        proposal='3958', fields=('007',),
        reference_frame=GNS, source=RECORDED_BULK,
        visit_key='full', dec_ref_deg=-28.805,
        consensus_jitter=True, reference_filter='F210M',
        recorded_bulk={
            (ANY, 'F187N'): BulkEntry(-89.7, -34.2, onsky_mas=True),
            (ANY, 'F210M'): BulkEntry(-88.5, -34.5, onsky_mas=True),
            (ANY, 'F335M'): BulkEntry(-89.5, -33.2, onsky_mas=True),
            (ANY, 'F470N'): BulkEntry(-91.4, -33.9, onsky_mas=True),
            (ANY, 'F480M'): BulkEntry(-90.6, -33.1, onsky_mas=True),
            (ANY, ANY): BulkEntry(-90.0, -34.0, onsky_mas=True),
        },
        notes=('sickle NIRCam. Audit 2026-06-20: had NO per-exposure alignment '
               '(fell through to the else), leaving catalogs ~90 mas off the '
               'GNS-tied mosaics. Single field translation, constant across '
               'filters/exposures to <3 mas. SLATED for VIRAC2 re-measurement '
               '(GC policy) -- see module docstring.'),
    ),
    FieldAlignment(
        proposal='1979', fields=None,
        reference_frame=GAIA, source=RECORDED_BULK,
        visit_key='full', dec_ref_deg=-26.427, warn_on_missing=True,
        recorded_bulk={
            ('jw01979002001', 'F150W2'): BulkEntry(104.7, -180.3, onsky_mas=True),
            ('jw01979002001', 'F322W2'): BulkEntry(-442.9, -87.9, onsky_mas=True),
            ('jw01979003001', 'F150W2'): BulkEntry(-2189.0, 370.7, onsky_mas=True),
            ('jw01979003001', 'F322W2'): BulkEntry(-1914.7, 546.9, onsky_mas=True),
        },
        notes=('M4 (o002 + o003="M-4-shift"): halo cluster outside VIRAC2/VVV. '
               'Audit 2026-07-11: fell through to the else and sat ~2" off Gaia. '
               'Bulk tie = measure_offset histogram of the untied destreak crf vs '
               'gaia_refcat, per (visit, filter) -- M4 differs SW(F150W2) vs '
               'LW(F322W2) by ~300-500 mas so it is keyed per filter.'),
    ),
    FieldAlignment(
        proposal='1334', fields=None,
        reference_frame=GAIA, source=RECORDED_BULK,
        visit_key='full', dec_ref_deg=43.139, warn_on_missing=True,
        recorded_bulk={
            ('jw01334001001', 'F090W'): BulkEntry(-1832.1, -708.2, onsky_mas=True),
            ('jw01334001001', 'F150W'): BulkEntry(-1853.5, -710.6, onsky_mas=True),
            ('jw01334001001', 'F277W'): BulkEntry(-1852.1, -711.7, onsky_mas=True),
            ('jw01334001001', 'F444W'): BulkEntry(-1852.7, -710.7, onsky_mas=True),
        },
        notes=('M92 (o001): halo cluster outside VIRAC2/VVV. A PURE per-visit '
               'shift -- all 4 filters agree to <20 mas -- but kept keyed per '
               'filter for symmetry with M4.'),
    ),
)


def resolve(proposal_id, field) -> Optional[FieldAlignment]:
    """Return the :class:`FieldAlignment` for this (proposal, observation).

    Returns ``None`` when the field has no configured alignment.  Callers MUST
    treat ``None`` as "this field is not tied to anything" and say so loudly --
    that state is the bug this module exists to make visible, not a default.

    Field-specific entries win over proposal-wide ones, so a proposal can have a
    general rule plus an exception.
    """
    exact = [c for c in ALIGNMENT_CONFIG
             if c.matches(proposal_id, field) and c.fields is not None]
    if exact:
        return exact[0]
    wide = [c for c in ALIGNMENT_CONFIG
            if c.matches(proposal_id, field) and c.fields is None]
    return wide[0] if wide else None


def visit_key_for(cfg: FieldAlignment, fn) -> str:
    """Derive this frame's visit key from its filename, per ``cfg.visit_key``."""
    stem = os.path.basename(fn).split('_')[0]
    if cfg.visit_key == 'suffix3':
        return stem[-3:]
    if cfg.visit_key == 'full':
        return stem
    raise ValueError(f"unknown visit_key {cfg.visit_key!r} for proposal "
                     f"{cfg.proposal}; expected 'full' or 'suffix3'")


def lookup_recorded_bulk(cfg: FieldAlignment, visit, filtername):
    """Return ``(dra_arcsec, ddec_arcsec, found)`` in the Δα-COORDINATE
    convention for a ``RECORDED_BULK`` field.

    Keys are tried most-specific first: exact (visit, filter), then filter-only,
    then visit-only, then the catch-all.  ``found`` is False when nothing
    matched, in which case the shift is ``(0, 0)`` and the frame stays at the
    raw frame.
    """
    if cfg.source != RECORDED_BULK:
        raise ValueError(f"lookup_recorded_bulk called for source={cfg.source!r}")
    filt = str(filtername).upper()
    for key in ((visit, filt), (ANY, filt), (visit, ANY), (ANY, ANY)):
        entry = cfg.recorded_bulk.get(key)
        if entry is None:
            continue
        if entry.onsky_mas:
            if cfg.dec_ref_deg is None:
                raise ValueError(
                    f"proposal {cfg.proposal} has an on-sky-mas bulk entry but no "
                    f"dec_ref_deg; cannot convert the RA term to the coordinate "
                    f"convention")
            cosd = np.cos(np.radians(cfg.dec_ref_deg))
            return entry.dra / 1000.0 / cosd, entry.ddec / 1000.0, True
        return entry.dra, entry.ddec, True
    return 0.0, 0.0, False


#: Offsets-table channel names.  ``'none'`` means the field takes no
#: table-driven correction at all, so writing one would produce a table nothing
#: reads.
CHANNEL_LOCKED = 'locked'
CHANNEL_CONSENSUS = 'consensus'
CHANNEL_NONE = 'none'


def offsets_channel(proposal_id, field):
    """Which offsets table m2 REWRITES for this field: ``'locked'`` /
    ``'consensus'`` / ``'none'``.

    This is the single answer to "which file changes when the checkpoint
    corrects this field", and every consumer must ask it rather than guessing
    from what happens to exist on disk.  Guessing has now failed twice: first a
    hardcoded ``_consensus.csv`` (missed every locked field), then a
    locked-before-consensus preference order (missed cloudef obs002, whose bulk
    is a recorded constant but whose per-exposure jitter is written to
    ``_consensus.csv`` -- while a stale ``_VIRAC2locked.csv`` sat beside it and
    won the preference).  Both produced the same silent symptom: the re-tie loop
    watched a file nobody wrote, saw no change, and stopped.
    """
    cfg = resolve(proposal_id, field)
    if cfg is None:
        return CHANNEL_NONE
    if cfg.source == TABLE_CONSENSUS:
        return CHANNEL_CONSENSUS
    if cfg.source == TABLE_LOCKED:
        return CHANNEL_LOCKED
    if cfg.source == RECORDED_BULK:
        # bulk is a constant; only the per-exposure term is table-driven
        return CHANNEL_CONSENSUS if cfg.consensus_jitter else CHANNEL_NONE
    return CHANNEL_NONE


def offsets_table_path(basepath, proposal_id, field):
    """Absolute path of the offsets table m2 rewrites, or ``''`` for ``'none'``.

    Existence is deliberately NOT checked: on the first re-tie iteration the
    table does not exist yet, and "absent now, written by the checkpoint" is
    exactly the transition callers need to observe.
    """
    ch = offsets_channel(proposal_id, field)
    if ch == CHANNEL_LOCKED:
        return f'{basepath}/offsets/Offsets_JWST_Brick{proposal_id}_VIRAC2locked.csv'
    if ch == CHANNEL_CONSENSUS:
        return f'{basepath}/offsets/Offsets_JWST_Brick{proposal_id}_consensus.csv'
    return ''
