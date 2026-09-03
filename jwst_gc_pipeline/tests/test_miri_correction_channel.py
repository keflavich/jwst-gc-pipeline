"""A MIRI correction has nowhere to go, and the channel must say so.

``alignment_config`` is read by the NIRCam reducer alone -- its own scope note
says so, and ``PipelineMIRI.fix_alignment`` / ``PipelineRerunNIRISS.fix_alignment``
mention no offsets table anywhere.  ``offsets_channel`` took no instrument, so a
proposal-wide entry answered ``'consensus'`` for MIRI exactly as it did for
NIRCam, and the m2 checkpoint seeded ``mirimage`` rows into
``Offsets_JWST_Brick<prop>_consensus.csv``.  Nothing reads them: the frames stay
where they were, the next re-tie measures the identical residual, and the run
reports a correction it never applied.

10678 (gc-treasury) is the field this was found on -- its entry is proposal-wide
``TABLE_CONSENSUS`` and every one of its 139 visits carries a MIRI F770W
parallel -- but the defect is a property of the instrument, not of the entry.
"""
import inspect
from pathlib import Path

import pytest

from jwst_gc_pipeline.reduction import alignment_config as AC

REDUCTION = Path(AC.__file__).resolve().parent


# ---------------------------------------------------------------------------
# The premise: neither reducer opens an offsets table.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("script", ["PipelineMIRI.py", "PipelineRerunNIRISS.py"])
@pytest.mark.parametrize("token", ["offsets", "alignment_config", "resolve_shift"])
def test_the_reducer_never_reads_an_offsets_table(script, token):
    """If this fails, that instrument grew a table reader and the channel it
    is given here should be revisited rather than left at 'none'."""
    src = (REDUCTION / script).read_text()
    assert token not in src, (
        f"{script} now mentions {token!r}; "
        f"alignment_config.TABLE_DRIVEN_INSTRUMENTS may need to include it")


# ---------------------------------------------------------------------------
# The channel, per instrument.
# ---------------------------------------------------------------------------

TREASURY_FIELDS = ("088", "001", "139")


@pytest.mark.parametrize("field", TREASURY_FIELDS)
def test_miri_gets_no_channel_for_10678(field):
    assert AC.offsets_channel("10678", field, instrument="miri") == AC.CHANNEL_NONE


@pytest.mark.parametrize("field", TREASURY_FIELDS)
def test_nircam_keeps_the_consensus_channel_for_10678(field):
    """The NIRCam half of the same proposal-wide entry is untouched."""
    assert AC.offsets_channel("10678", field) == AC.CHANNEL_CONSENSUS
    assert AC.offsets_channel("10678", field,
                              instrument="nircam") == AC.CHANNEL_CONSENSUS


def test_niriss_gets_no_channel_either():
    assert AC.offsets_channel("10678", "088", instrument="niriss") == AC.CHANNEL_NONE


def test_the_instrument_scope_is_not_10678_specific():
    """sickle's MIRI entry (3958/001-002) resolves and keeps its frame and
    anchor -- what it does not get is a table its reducer cannot read."""
    cfg = AC.resolve("3958", "001-002")
    assert cfg is not None and cfg.reference_filter == "F770W"
    assert AC.offsets_channel("3958", "001-002") == AC.CHANNEL_CONSENSUS
    assert AC.offsets_channel("3958", "001-002",
                              instrument="miri") == AC.CHANNEL_NONE
    # a LOCKED NIRCam field is unaffected either way
    assert AC.offsets_channel("3958", "007") == AC.CHANNEL_LOCKED


def test_instrument_case_and_spacing_do_not_change_the_answer():
    assert AC.offsets_channel("10678", "088", instrument=" MIRI ") == AC.CHANNEL_NONE
    assert AC.offsets_channel("10678", "088",
                              instrument="NIRCam") == AC.CHANNEL_CONSENSUS


def test_helper_default_is_the_historical_answer():
    assert AC.instrument_has_table_channel(None) is True
    assert AC.instrument_has_table_channel("nircam") is True
    assert AC.instrument_has_table_channel("miri") is False


# ---------------------------------------------------------------------------
# The table path follows the channel.
# ---------------------------------------------------------------------------

def test_no_table_path_is_offered_to_miri(tmp_path):
    bp = str(tmp_path)
    assert AC.offsets_table_path(bp, "10678", "088").endswith(
        "offsets/Offsets_JWST_Brick10678_consensus.csv")
    assert AC.offsets_table_path(bp, "10678", "088", instrument="miri") == ""


# ---------------------------------------------------------------------------
# The checkpoint side: what m2 asks, and what it does with the answer.
# ---------------------------------------------------------------------------

def test_module_token_maps_to_the_instrument():
    from jwst_gc_pipeline.photometry.cataloging import _instrument_for_module
    assert _instrument_for_module("mirimage") == "miri"
    assert _instrument_for_module("nis") == "niriss"
    for module in ("nrca", "nrcalong", "nrcb3", "merged"):
        assert _instrument_for_module(module) == "nircam"


def test_checkpoint_channel_is_none_for_a_mirimage_merge():
    from jwst_gc_pipeline.photometry.cataloging import (
        _astrom_offsets_channel, _instrument_for_module)
    assert _astrom_offsets_channel(
        "10678", "088", instrument=_instrument_for_module("mirimage")) == "none"
    assert _astrom_offsets_channel(
        "10678", "088", instrument=_instrument_for_module("nrcalong")) == "consensus"
    # unchanged for every caller that does not name an instrument
    assert _astrom_offsets_channel("10678", "088") == "consensus"


def test_an_existing_consensus_table_is_still_not_miris(tmp_path):
    """The lookup is decided by the channel, not by what is on disk: once
    NIRCam has seeded the proposal's consensus table, a MIRI merge must still
    be told there is no table for it."""
    from jwst_gc_pipeline.photometry.cataloging import _astrom_find_offsets_table
    offsets = tmp_path / "offsets"
    offsets.mkdir()
    table = offsets / "Offsets_JWST_Brick10678_consensus.csv"
    table.write_text("Visit,Filter,dra (arcsec),ddec (arcsec)\n")
    bp = str(tmp_path)
    assert _astrom_find_offsets_table(bp, "10678", "088") == str(table)
    assert _astrom_find_offsets_table(bp, "10678", "088",
                                      instrument="nircam") == str(table)
    assert _astrom_find_offsets_table(bp, "10678", "088",
                                      instrument="miri") is None


def test_the_checkpoint_scopes_its_channel_question_to_the_merge_s_module():
    """The wiring, pinned at the call site: without it the channel lookup is
    instrument-blind again and the MIRI rows get written after all."""
    import re

    from jwst_gc_pipeline.photometry import cataloging as _cat
    src = inspect.getsource(_cat._run_astrometry_stage_checkpoint)
    assert "_instrument = _instrument_for_module(module)" in src
    assert re.search(r"_channel = _astrom_offsets_channel\([^)]*instrument=",
                     src, re.S), (
        "the m2 channel lookup no longer names the merge's instrument")
    assert re.search(r"_astrom_find_offsets_table\([^)]*instrument=", src, re.S)


def test_the_refusal_names_the_reducer_not_a_missing_entry():
    """A MIRI operator sent to add an alignment_config entry would add one and
    hit the identical wall, because the entry is not what is missing."""
    from jwst_gc_pipeline.photometry.cataloging import (
        _astrom_no_channel_error, _instrument_for_module)

    err = _astrom_no_channel_error(
        "m2", "mirimage", "F770W", "10678", "088",
        _instrument_for_module("mirimage"), 6)
    msg = str(err)
    assert isinstance(err, RuntimeError)
    assert "PipelineMIRI.fix_alignment" in msg
    assert "reads no offsets table" in msg
    assert "Nothing was written." in msg
    assert "Add an entry to" not in msg


def test_an_unconfigured_nircam_field_still_gets_the_add_an_entry_message():
    """The field-shaped reason is unchanged: that operator CAN fix it there."""
    from jwst_gc_pipeline.photometry.cataloging import (
        _astrom_no_channel_error, _instrument_for_module)

    msg = str(_astrom_no_channel_error(
        "m2", "nrcalong", "F480M", "9999", "001",
        _instrument_for_module("nrcalong"), 3))
    assert "alignment_config declares NO table-driven correction channel" in msg
    assert "Add an entry to" in msg
    assert "PipelineMIRI" not in msg


def test_the_niriss_refusal_names_the_niriss_reducer():
    from jwst_gc_pipeline.photometry.cataloging import (
        _astrom_no_channel_error, _instrument_for_module)

    msg = str(_astrom_no_channel_error(
        "m2", "nis", "F200W", "10678", "001",
        _instrument_for_module("nis"), 2))
    assert "PipelineRerunNIRISS.fix_alignment" in msg
