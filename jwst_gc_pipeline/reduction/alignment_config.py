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
# RESOLVED 2026-08-04 -- GC FIELDS AND GNS: sickle (3958) was the last field whose
# recorded bulk sat in the GNS frame while ``refnames`` already called it VIRAC2.
# Policy is that GC fields tie to VIRAC2; that tie is now BUILT (a per-exposure
# ``Offsets_JWST_Brick3958_VIRAC2locked.csv``) rather than re-measured by step 0,
# because step 0 refuses to record a fresh tie for a field that already has one.
# The 3958 entry below is now TABLE_LOCKED on VIRAC2 and no longer aspirational.

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
        reference_filter='F210M',
        notes=('w51. Same class as sgrc; outside the VVV/VIRAC2 footprint, so the '
               'bulk sentinel ties the consensus to gaia_refcat.fits. '
               'reference_filter was F200W, which 6151 does not observe at all '
               '(f140m f162m f182m f187n f210m f335m f360m f405n f410m f480m + '
               'MIRI) -- a band whose consensus cannot define the frame because '
               'it does not exist.  F210M is both present and what '
               'consensus_catalog.reference_filter ranks first for this list; '
               'test_reference_filter_agrees_with_alignment_config now keeps the '
               'two from drifting apart again.'),
    ),
    FieldAlignment(
        proposal='1905', fields=None,
        reference_frame=GAIA, source=TABLE_CONSENSUS,
        reference_filter='F212N',
        notes=('wd1 (Westerlund 1). Same class as w51 above: well outside the '
               'VVV/VIRAC2 footprint, so the reference is a pure '
               'gaia_refcat.fits (13074 sources over the footprint, median NN '
               '3.41") and the bulk sentinel ties the consensus to it.\n\n'
               'ABSENT until now, which is what #479 measured: with no entry '
               'every frame stayed at the raw assign_wcs frame and any offsets '
               'table written for the field was read by nothing.  Measured '
               '2026-08-24 on the per-filter m7 catalogs (NOT the merged '
               "catalog -- its skycoord_* columns carry other stars' positions "
               'while PR #300 is open, which manufactured a spurious 2.883" '
               'peak that survived the issue-158 window sweep):\n\n'
               '    all 11 filters, A-B against gaia_refcat.fits\n'
               '      east half   27.6 - 33.2 mas   contrast up to 383\n'
               '      west half   49.3 - 62.0 mas   contrast up to 285\n\n'
               'So the ~40 mas bulk this issue reports is real and consistent '
               'across every band.  A single bulk entry does NOT make the field '
               'clean -- there is a ~20 mas east-west gradient underneath it, '
               'so expect a residual of roughly +/-10 mas after this lands.  It '
               'is registered anyway because ~10 mas beats ~40 mas and because '
               'an unregistered field silently discards every correction the '
               'checkpoint measures; the gradient is tracked separately.\n\n'
               'reference_filter F212N: present in 1905 (f115w f150w f164n '
               'f187n f200w f212n f277w f323n f405n f444w f466n) and what '
               'consensus_catalog.reference_filter ranks first for that list, '
               'so test_reference_filter_agrees_with_alignment_config holds.\n\n'
               'fields=None covers both nircam obsids (001 and 003).  '
               'fields.yaml names a reference_catalog for 001 only; 003 falls '
               'back to the same gaia_refcat.fits in the field tree.'),
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
        proposal='10678', fields=None,
        reference_frame=VIRAC2, source=TABLE_CONSENSUS,
        reference_filter='F212N',
        notes=('gc-treasury (GC Treasury, 139 planned observations over '
               '~1668 exposure-level MAST rows, none executed yet; #413).  '
               'Registered BEFORE any delivery: a field absent here reduces '
               '"successfully" at the raw assign_wcs frame while the m2 '
               'checkpoint -- the second merge iteration, where every '
               'exposure is re-measured against its visit consensus -- writes '
               'corrections into offsets/Offsets_JWST_Brick10678_consensus.csv, '
               'which nothing would read (the 1939/sgra failure class, '
               '~14.8" off).  Proposal-wide (fields=None) because all '
               '139 observations are one field; fields.yaml claims them with '
               'a wildcard for the same reason.  Gaia defines the absolute '
               'frame and VIRAC2 is the reference catalog, per the GC rule.  '
               'Every visit observes F212N+F480M (+MIRI F770W in parallel); '
               'F212N is what consensus_catalog.reference_filter ranks first '
               'for that list.  The table of per-exposure consensus '
               'coordinates, offsets/Offsets_JWST_Brick10678_consensus.csv, '
               'does not exist yet: the m2 checkpoint creates it and updates '
               'it in place on the first reduce.'),
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
        proposal='1939', fields=('001',),
        reference_frame=VIRAC2, source=TABLE_LOCKED,
        reference_filter='F212N',
        notes=('sgra. Same dispatch gap as arches/sgrb2/gc2211, and the largest '
               'consequence of it measured so far: 1939 was absent from the '
               'registry, so every exposure stayed at the raw assign_wcs frame '
               'while Offsets_JWST_Brick1939_VIRAC2locked.csv (36 rows, 3 '
               'filters, m2-written 2026-07-28) went unread.  The table asks for '
               'dra ~10.25" / ddec ~11.85" -- on-sky (8.96", 11.85"), |off| '
               '~14.8" -- and every mosaic on disk is off by exactly that: '
               'F115W/F212N/F405N, nrca/nrcb/merged all read 14.83" +/- 0.01" in '
               'the same direction, agreeing between dense VIRAC2 and sparse '
               'Gaia to a few mas (measured 2026-08-06, offset-histogram with '
               'window sweep; peak at w=30" so window_edge_fraction ~0.49, not a '
               'footprint ridge).  sgra observes F115W+F212N+F405N; F212N is '
               'what consensus_catalog.reference_filter ranks first for that '
               'list.  Frames must be regenerated from _cal -- fix_alignment is '
               'idempotent on a baked RAOFFSET, so re-applying on top of the '
               'stale shift would not correct them.'),
    ),
    FieldAlignment(
        proposal='2211', fields=None,
        reference_frame=VIRAC2, source=TABLE_LOCKED,
        reference_filter='F200W',
        notes=('gc2211. OBSERVATION 023 IS JUNK: all four of its exposures have '
               'tracking errors and are listed in exposure_exclusions.py (#484), '
               'so no 023 row here ever reaches a frame. 028/046/049/050 are '
               'unaffected. '
               'Absent from the old dispatch -> unaligned, so its '
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
    # sickle MIRI (3958 observations 001 and 002, jointly registered as the
    # single field '001-002' -- the two pointings sit 58" apart and overlap, so
    # they are cataloged together).  The NIRCam entry below deliberately covers
    # only obs 007 and says so; that left MIRI with NO table-driven correction
    # channel, and the m2 checkpoint refuses to write a correction it cannot
    # route:
    #
    #   astrom checkpoint [m2] F770W/mirimage: measured 6 real correction(s) for
    #   proposal 3958 observation 001-002, but alignment_config declares NO
    #   table-driven correction channel for this field
    #
    # That refusal is correct -- without an entry the numbers would land in a
    # table fix_alignment never reads and the next re-tie would re-measure the
    # identical residual (the arches/sgrb2 failure).
    #
    # TABLE_CONSENSUS, not TABLE_LOCKED: the authored table
    # Offsets_JWST_Brick3958_VIRAC2locked.csv is 120 rows over the five NIRCam
    # bands (F187N/F210M/F335M/F470N/F480M, 24 exposures each) and carries no
    # MIRI rows at all, so there is nothing for MIRI to lock to.  The m2
    # visit-consensus re-tie bootstraps its own table with provenance.
    #
    # Frame is VIRAC2, matching the NIRCam entry -- the two instruments observe
    # the same sky and a MIRI tie to a different frame would put the field's own
    # bands in disagreement.
    #
    # Anchor is F770W, and it CANNOT be the NIRCam entry's F210M.
    # `promote_reference_filter` resolves the anchor's consensus under the
    # OBSERVATION's token, and obs 001/002 are MIRI-only:
    #
    #     _o007       f210m_o007_consensus.fits        exists
    #     _o001-002   f770w_o001-002_consensus.fits    exists
    #                 f210m_o001-002_consensus.fits    cannot exist -- F210M is
    #                                                  not observed in 001/002
    #
    # An F210M anchor here would raise FileNotFoundError on a checkpoint that can
    # never run.  The anchor is per OBSERVATION, not per field: the two tokens
    # have disjoint band sets.
    #
    # obs 003 is NOT here on purpose.  Those frames live in the sickle tree but
    # 3958/003 is registered to BRICK in fields.yaml (obsids miri: ['003']), and
    # its pointing is 394" from 001/002 -- different sky, a different field's
    # deliverable.
    FieldAlignment(
        proposal='3958', fields=('001-002', '001', '002'),
        reference_frame=VIRAC2, source=TABLE_CONSENSUS,
        reference_filter='F770W',
        notes=('sickle MIRI (F770W/F1130W/F1500W, obs 001+002 jointly '
               'registered as 001-002; 10 crf per band). Registered 2026-09-01 '
               'after the m2 checkpoint refused to route 6 measured corrections '
               'for want of a channel. Consensus rather than locked because the '
               'VIRAC2locked table is NIRCam-only (120 rows, 5 bands, no MIRI). '
               'The single-observation spellings are included so a per-obs run '
               'resolves the same way as the joint one. obs 003 belongs to '
               'brick, not sickle.'),
    ),
    FieldAlignment(
        proposal='3958', fields=('007',),
        reference_frame=VIRAC2, source=TABLE_LOCKED,
        reference_filter='F210M',
        notes=('sickle NIRCam (observation 007; the 3958 MIRI data are obs 001 '
               'and are NOT covered by this entry). Carried out the VIRAC2 '
               're-measurement this module\'s docstring slated it for: GC policy '
               'is that GC fields tie to VIRAC2, and this field was the last one '
               'still recorded in the GNS frame while ``refnames`` already called '
               'it VIRAC2.\n'
               '\n'
               'Same class as sgrc/quintuplet/sgrb2: sickle now has a '
               'build_virac2_offsets REGION entry, so its authored table is '
               'Offsets_JWST_Brick3958_VIRAC2locked.csv (120 rows, 5 filters, '
               '24 exposures each, builder-shaped, per-exposure). LOCKED rather '
               'than RECORDED_BULK because a per-exposure table exists: the '
               'route to VIRAC2 for an already-tied field is to BUILD the VIRAC2 '
               'table, not to blank the recorded bulk and re-measure -- step 0 '
               'refuses the latter ("the field is tied; this (visit, band) is '
               'not"), which is what an empty ``recorded_bulk`` attempt hit.\n'
               '\n'
               'The former GNS bulk was per-filter and near-constant -- F187N '
               '(-89.7,-34.2), F210M (-88.5,-34.5), F335M (-89.5,-33.2), F470N '
               '(-91.4,-33.9), F480M (-90.6,-33.1), default (-90.0,-34.0) mas, '
               'agreeing across filters to <3 mas. Those numbers are deliberately '
               'NOT carried over: they are GNS-frame values and re-using them '
               'against VIRAC2 would bake in the frame difference as if it were '
               'an astrometry correction. The measured GNS->VIRAC2 frame shift is '
               '(+71.74,-70.09) mas (std 1.5/1.45, range <4.8 mas across all five '
               'filters) -- a coherent frame offset, not a per-filter defect.\n'
               '\n'
               'Unlike the previous GNS entry, this DOES change pixels -- sickle '
               'needs re-reduction and re-cataloging.'),
    ),
    FieldAlignment(
        proposal='1979', fields=None,
        reference_frame=GAIA, source=RECORDED_BULK,
        # The recorded bulk is a hand-measured CONSTANT, so it has no
        # exposure axis and `offsets_channel` returned 'none' -- with which
        # the m2 checkpoint REFUSES to write the per-exposure corrections it
        # measures, because they would land in a table fix_alignment never
        # reads.  That refusal blocked this field's m12 finalize outright.
        # `consensus_jitter` routes ONLY the small per-exposure term to the
        # consensus table and leaves the recorded bulk untouched, which is
        # exactly what a hand-measured-bulk field needs.  cloudef 2092/002 is
        # the precedent already in this file.
        consensus_jitter=True,
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
    # ngc6334 -- the Cat's Paw, imaged by TWO proposals over the SAME sky: 6778
    # (3 visits; F090W/F187N/F200W/F277W/F335M/F470N, 450 crf) and 7213 (2 visits;
    # F115W/F162M/F182M/F200W/F356W/F405N/F444W/F470N, 800 crf).  Both were absent
    # from this registry, so every exposure of both stayed at the raw assign_wcs
    # frame -- the same dispatch gap that left sgra/1939 ~14.8" out of place.  The
    # 2026-07-10 audit measured the consequence here as per-filter offsets off the
    # channel anchors: F115W 63, F162M 61, F182M 66 mas (F405N 67 mas).
    #
    # Neither proposal has an offsets table, so there is nothing to LOCK to and
    # the bulk cannot be a recorded constant: TABLE_CONSENSUS is the
    # self-bootstrapping mode, and the m2 checkpoint writes the table (with
    # provenance) as it measures.  reference_frame is VIRAC2 -- the field's
    # refcat is VIRAC2-dominated (22236 VIRAC2 + 1403 GaiaDR3 of 23639 rows at
    # epoch 2026.30), so VVV disk coverage reaches this longitude and Gaia is far
    # too sparse to be the catalog here, exactly as in the GC fields.
    #
    # reference_filter comes from `consensus_catalog.reference_filter`, which
    # ranks a field's bands by closeness to VIRAC2 in wavelength and in which
    # stars they leave unsaturated -- NOT from picking the band the two proposals
    # happen to share.  It answers F187N for 6778 and F182M for 7213, and
    # `test_the_formula_reproduces_the_hand_set_reference_filters` requires the
    # config to agree with it, so that the m2 consensus catalog and the reducer
    # anchor to the same band.
    FieldAlignment(
        proposal='6778', fields=('001',),
        reference_frame=VIRAC2, source=TABLE_CONSENSUS,
        reference_filter='F187N',
        notes=('ngc6334 (Cat\'s Paw), 3 visits, 450 crf over 6 bands. Registered '
               '2026-09-01 after the field was found absent from ALIGNMENT_CONFIG '
               'entirely -- unregistered means the raw assign_wcs frame, and the '
               '2026-07-10 audit had already flagged 61-67 mas per-filter offsets '
               'off the channel anchors. No offsets table exists for 6778, so the '
               'bulk is bootstrapped by the m2 visit-consensus re-tie rather than '
               'locked to a table. Shares sky and the F200W/F470N bands with '
               '7213.'),
    ),
    FieldAlignment(
        proposal='7213', fields=('001',),
        reference_frame=VIRAC2, source=TABLE_CONSENSUS,
        reference_filter='F182M',
        notes=('ngc6334, the SECOND proposal on the same sky as 6778: 2 visits, '
               '800 crf over 8 bands. Registered 2026-09-01 for the same reason '
               'and in the same mode. Kept as its own entry because '
               'reference_frame is per-PROPOSAL (it names the offsets table), so '
               'the two programs cannot share one row even though they image one '
               'field -- the same split brick carries for 1182 and 2221.'),
    ),
    FieldAlignment(
        proposal='9438', fields=('005', '006'),
        reference_frame=GAIA, source=TABLE_CONSENSUS,
        reference_filter='F210M',
        notes=('9438 (Schlafly) o005 = G007.470+00.050 (l=+7.46, b=+0.06) and '
               'o006 = crowded_l3 (l=+3.00, b=+0.00): the two pointings of this '
               'seven-field program that fall inside the VVV bulge footprint, so '
               'they tie to VIRAC2 like the GC fields.  TABLE_CONSENSUS rather '
               'than TABLE_LOCKED because 9438 is NEW -- there is no '
               'build_virac2_offsets region entry and no locked table on disk, '
               'and seed_offsets_table_from_consensus builds its own from the m2 '
               'consensus.  Consensus also keys by (visit, filter, exposure, '
               'module, vgroup), so a per-exposure residual is expressible; '
               'RECORDED_BULK has no exposure axis and is what leaves m92/m4/'
               'ngc6397 unable to correct at all (#589).'),
    ),
    FieldAlignment(
        proposal='9438', fields=('001', '002', '003', '004', '007'),
        reference_frame=GAIA, source=TABLE_CONSENSUS,
        reference_filter='F210M',
        notes=('9438 o001/002/003/004/007 = G028.320, G033.007, G040.954, '
               'G054.093 and crowded_l20, at l = 28, 33, 41, 54 and 20 deg.  All '
               'lie OUTSIDE the VVV footprint (checked per pointing from its own '
               'galactic coordinates), so Gaia is both frame and catalog here, '
               'the same regime as w51/wd1/wd2.  Expect the w51 caveats to '
               'apply: against a Gaia-ONLY refcat the per-tile map is noise and '
               'measure_reference_tie falls back to the same-star refinement '
               '(#411, #263), and the m7 cross-filter local map may not populate '
               'at all on a mosaic-scale field (#565).'),
    ),
    FieldAlignment(
        proposal='1334', fields=None,
        reference_frame=GAIA, source=RECORDED_BULK,
        # The recorded bulk is a hand-measured CONSTANT, so it has no
        # exposure axis and `offsets_channel` returned 'none' -- with which
        # the m2 checkpoint REFUSES to write the per-exposure corrections it
        # measures, because they would land in a table fix_alignment never
        # reads.  That refusal blocked this field's m12 finalize outright.
        # `consensus_jitter` routes ONLY the small per-exposure term to the
        # consensus table and leaves the recorded bulk untouched, which is
        # exactly what a hand-measured-bulk field needs.  cloudef 2092/002 is
        # the precedent already in this file.
        consensus_jitter=True,
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
