"""Every registered field can route a measured NIRCam correction somewhere.

NIRCam, because that is the instrument the tables reach: ``fix_alignment``
resolves a NIRCam shift through ``unified_alignment.resolve_shift``, which reads
exactly the table ``offsets_table_path`` names, while ``PipelineMIRI`` and
``PipelineRerunNIRISS`` open no offsets table at all
(``TABLE_DRIVEN_INSTRUMENTS``).  Asked WITH one of those instruments, a
registered field answers ``'none'`` however it is declared -- that is the
scoping, not a hole in this invariant, and the bottom of this file pins it.

The m2 checkpoint REFUSES to write corrections for a field whose
``offsets_channel`` is ``'none'``:

    astrom checkpoint [m2] F090W/nrca: measured 9 real correction(s) for proposal
    1334 observation 001, but alignment_config declares NO table-driven correction
    channel for this field -- so anything written here would land in a table
    fix_alignment never reads, and the next re-tie would re-measure the identical
    residual (the arches/sgrb2 failure).

That refusal is correct, and it stops the m12 finalize, so the field cannot be
cataloged at all (``ASTROM_CHECKPOINT_WARN_ONLY=1`` demotes it, like every other
blocking error in the checkpoint -- which is a deliberate override, not a
default).  m92 (1334) and m4/ngc6397 (1979) were both in that state --
``RECORDED_BULK`` with ``consensus_jitter`` unset.

``consensus_jitter`` routes ONLY the per-exposure term to the consensus table and
leaves the hand-measured bulk alone, which is what these halo-cluster fields
(outside VIRAC2/VVV, bulk measured by hand) need.
"""
import pytest

from jwst_gc_pipeline.reduction import alignment_config as AC


def _first_field(entry):
    return (entry.fields or ("001",))[0]


@pytest.mark.parametrize(
    "entry", AC.ALIGNMENT_CONFIG,
    ids=[f"{e.proposal}-{'|'.join(e.fields) if e.fields else 'all'}"
         for e in AC.ALIGNMENT_CONFIG])
def test_every_entry_can_route_a_nircam_correction(entry):
    """The invariant: a registered field with nowhere to write is unusable.

    A field is registered precisely so the checkpoint can correct it; an entry
    whose channel is ``'none'`` blocks its own cataloging at the m2 gate.

    Asked for NIRCam explicitly -- the instrument whose reducer reads these
    tables.  The instrument-blind call answers the same thing (omitting the
    argument keeps the historical answer), and both are asserted so a future
    change to that default cannot quietly move this invariant.
    """
    field = _first_field(entry)
    for instrument in (None, 'nircam'):
        channel = AC.offsets_channel(entry.proposal, field,
                                     instrument=instrument)
        assert channel != AC.CHANNEL_NONE, (
            f"proposal {entry.proposal} ({entry.source}) has no correction "
            f"channel for instrument={instrument!r}, so the m2 checkpoint will "
            f"refuse to write measured corrections and the field cannot be "
            f"cataloged"
        )


@pytest.mark.parametrize("proposal,field", [
    ("1334", "001"),   # m92
    ("1979", "002"),   # m4
    ("1979", "003"),   # m4 shifted pointing
])
def test_halo_cluster_fields_route_jitter_to_consensus(proposal, field):
    cfg = AC.resolve(proposal, field)
    assert cfg.source == AC.RECORDED_BULK, "the hand-measured bulk must stay recorded"
    assert cfg.consensus_jitter is True
    assert AC.offsets_channel(proposal, field) == AC.CHANNEL_CONSENSUS


def test_cloudef_precedent_unchanged():
    """2092/002 is the pattern these follow and must keep working."""
    cfg = AC.resolve("2092", "002")
    assert cfg.source == AC.RECORDED_BULK
    assert cfg.consensus_jitter is True
    assert AC.offsets_channel("2092", "002") == AC.CHANNEL_CONSENSUS


@pytest.mark.parametrize("proposal,field,visit,filt,expect", [
    ("1334", "001", "jw01334001001", "F090W", (-1832.1, -708.2)),
    ("1334", "001", "jw01334001001", "F444W", (-1852.7, -710.7)),
    ("1979", "002", "jw01979002001", "F150W2", (104.7, -180.3)),
    ("1979", "003", "jw01979003001", "F322W2", (-1914.7, 546.9)),
])
def test_recorded_bulk_values_are_untouched(proposal, field, visit, filt, expect):
    """Routing the jitter must not disturb the hand-measured bulk.

    These numbers came from an offset-histogram tie against gaia_refcat; a change
    here would silently re-point the field.
    """
    entry = AC.resolve(proposal, field).recorded_bulk[(visit, filt)]
    assert (entry.dra, entry.ddec) == expect


def test_bulk_source_is_still_recorded_not_consensus():
    """`consensus_jitter` must not silently promote these to TABLE_CONSENSUS.

    If it did, the m2 re-tie would re-solve the BULK too, discarding the
    hand-measured tie these fields depend on (they sit outside VIRAC2/VVV).
    """
    for proposal, field in (("1334", "001"), ("1979", "002")):
        assert AC.resolve(proposal, field).source == AC.RECORDED_BULK


# --------------------------------------------------------------------------
# The other half of the invariant: the instruments the tables do NOT reach.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("proposal,field", [
    ("3958", "001-002"),   # sickle MIRI, registered for its FRAME and anchor
    ("3958", "007"),       # sickle NIRCam, TABLE_LOCKED
    ("10678", "088"),      # gc-treasury, proposal-wide TABLE_CONSENSUS
])
@pytest.mark.parametrize("instrument", ["miri", "niriss"])
def test_a_registered_field_still_routes_nothing_to_miri_or_niriss(
        proposal, field, instrument):
    """Registration is about the frame and the anchor, not about a channel.

    Neither reducer opens an offsets table, so a correction written on their
    behalf reaches no frame and the next re-tie re-measures the identical
    residual.  The checkpoint refuses instead, naming the reducer -- the thing
    an operator can act on -- rather than sending them to add an entry that
    would change nothing.
    """
    assert AC.resolve(proposal, field) is not None
    assert AC.offsets_channel(proposal, field,
                              instrument=instrument) == AC.CHANNEL_NONE


def test_the_miri_only_entry_still_supplies_a_frame_and_an_anchor():
    """3958/001-002 is the entry that makes the distinction visible: it is
    registered, it is MIRI-only, and what it gives MIRI is the frame and the
    F770W anchor rather than a table."""
    cfg = AC.resolve("3958", "001-002")
    assert cfg.reference_frame == AC.VIRAC2
    assert cfg.reference_filter == "F770W"
    assert AC.offsets_channel("3958", "001-002") == AC.CHANNEL_CONSENSUS
    assert AC.offsets_channel("3958", "001-002",
                              instrument="miri") == AC.CHANNEL_NONE
