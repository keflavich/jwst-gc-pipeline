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

def test_the_module_token_names_the_instrument_when_it_can():
    from jwst_gc_pipeline.photometry.cataloging import _instrument_for_merge
    assert _instrument_for_merge("mirimage", "F770W") == "miri"
    assert _instrument_for_merge("nis", "F200W") == "niriss"
    for module in ("nrca", "nrcalong", "nrcb3", "merged"):
        assert _instrument_for_merge(module, "F200W") == "nircam"


def test_a_nircam_shaped_module_token_does_not_make_a_miri_merge_nircam():
    """The token is the first signal, not the only one.

    ``submit_cataloging.sbatch`` defaults ``MODULES`` to ``nrcb`` and a
    cross-module merge is spelled ``merged``, so a MIRI merge can arrive with a
    NIRCam-shaped token.  Deciding on the token alone reads ``nircam`` there and
    takes the silent write path -- fail-open in the direction this scoping
    exists to close.
    """
    from jwst_gc_pipeline.photometry.cataloging import _instrument_for_merge
    for module in ("nrcb", "merged", "nrcalong", ""):
        assert _instrument_for_merge(module, "F770W") == "miri", module


def test_the_instrument_override_is_how_a_niriss_merge_says_so(monkeypatch):
    """NIRISS filter names ARE NIRCam's, so ``--instrument`` /
    ``GC_INSTRUMENT_OVERRIDE`` is its only signal beyond the ``nis`` token --
    and it is part of the standard NIRISS cataloging invocation
    (GETTING_STARTED.md, NOTES_niriss_sgrc_firstlight.md)."""
    from jwst_gc_pipeline.photometry.cataloging import _instrument_for_merge
    monkeypatch.setenv("GC_INSTRUMENT_OVERRIDE", "niriss")
    for module in ("nrcb", "merged", "nis"):
        assert _instrument_for_merge(module, "F200W") == "niriss", module


@pytest.mark.parametrize("module", ["mirimage", "nrca", "nrcb",
                                    "nrcalong", "merged"])
@pytest.mark.parametrize("filt", ["F770W", "F200W", "F480M"])
def test_it_agrees_with_the_spelling_cataloging_already_uses(module, filt):
    """``_sat_is_miri`` (cataloging.py) and ``_miri_field`` both ask
    ``module == 'mirimage' or _instrument_from_filter(filt) == 'MIRI'``.  A
    third spelling of one question is a third answer waiting to disagree.

    ``nis`` is left out because that spelling cannot express it: it has no
    NIRISS branch at all, and a ``nis`` merge of a MIRI filter is not a
    combination any observation can produce.
    """
    from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as _L
    from jwst_gc_pipeline.photometry.cataloging import _instrument_for_merge

    existing = (module == "mirimage"
                or _L._instrument_from_filter(filt) == "MIRI")
    assert (_instrument_for_merge(module, filt) == "miri") is existing


def test_checkpoint_channel_is_none_for_a_mirimage_merge():
    from jwst_gc_pipeline.photometry.cataloging import (
        _astrom_offsets_channel, _instrument_for_merge)
    assert _astrom_offsets_channel(
        "10678", "088",
        instrument=_instrument_for_merge("mirimage", "F770W")) == "none"
    assert _astrom_offsets_channel(
        "10678", "088",
        instrument=_instrument_for_merge("nrcalong", "F480M")) == "consensus"
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
    assert "_instrument = _instrument_for_merge(module, filt)" in src
    assert re.search(r"_channel = _astrom_offsets_channel\([^)]*instrument=",
                     src, re.S), (
        "the m2 channel lookup no longer names the merge's instrument")
    assert re.search(r"_astrom_find_offsets_table\([^)]*instrument=", src, re.S)


def test_the_refusal_names_the_reducer_not_a_missing_entry():
    """A MIRI operator sent to add an alignment_config entry would add one and
    hit the identical wall, because the entry is not what is missing."""
    from jwst_gc_pipeline.photometry.cataloging import (
        _astrom_no_channel_error, _instrument_for_merge)

    err = _astrom_no_channel_error(
        "m2", "mirimage", "F770W", "10678", "088",
        _instrument_for_merge("mirimage", "F770W"), 6)
    msg = str(err)
    assert isinstance(err, RuntimeError)
    assert "PipelineMIRI.fix_alignment" in msg
    assert "reads no offsets table" in msg
    assert "Nothing was written." in msg
    assert "Add an entry to" not in msg


def test_an_unconfigured_nircam_field_still_gets_the_add_an_entry_message():
    """The field-shaped reason is unchanged: that operator CAN fix it there."""
    from jwst_gc_pipeline.photometry.cataloging import (
        _astrom_no_channel_error, _instrument_for_merge)

    msg = str(_astrom_no_channel_error(
        "m2", "nrcalong", "F480M", "9999", "001",
        _instrument_for_merge("nrcalong", "F480M"), 3))
    assert "alignment_config declares NO table-driven correction channel" in msg
    assert "Add an entry to" in msg
    assert "PipelineMIRI" not in msg


def test_the_niriss_refusal_names_the_niriss_reducer():
    from jwst_gc_pipeline.photometry.cataloging import (
        _astrom_no_channel_error, _instrument_for_merge)

    msg = str(_astrom_no_channel_error(
        "m2", "nis", "F200W", "10678", "001",
        _instrument_for_merge("nis", "F200W"), 2))
    assert "PipelineRerunNIRISS.fix_alignment" in msg


# ---------------------------------------------------------------------------
# End to end, through the shipping checkpoint: what the refusal DOES.
#
# The source-shape checks above pin the wiring; these drive
# `_run_astrometry_stage_checkpoint` itself with only `run_visit_checkpoint`
# stubbed, which is where the two operational regressions lived -- the refusal
# ignored ASTROM_CHECKPOINT_WARN_ONLY, and it returned before the
# `_im0_badastrom` quarantine.
# ---------------------------------------------------------------------------

def _drive_no_channel(monkeypatch, tmp_path, *, module, filt, proposal, field,
                      target, warn_only=False, apply_=False, i2d_names=()):
    """Run the real m2 checkpoint on a merge whose channel is ``'none'``.

    One above-floor correction is handed in (100 mas clears every per-field
    floor, sgrc's 8.0 included), and any ``i2d_names`` are created in the
    ``<FILTER>/pipeline`` directory ``find_i2d_for_filter`` globs.
    """
    import types

    from astropy.table import Table

    from jwst_gc_pipeline.photometry import astrometry_checkpoint
    from jwst_gc_pipeline.photometry import cataloging as _cat

    d = tmp_path / filt.upper()
    d.mkdir(parents=True, exist_ok=True)
    Table({'x': [1.0]}).write(
        str(d / f"{filt.lower()}_{module}_visit001_vgroup00101_exp00001_m2_"
                f"daophot_basic.fits"), overwrite=True)
    pipe = d / 'pipeline'
    pipe.mkdir(exist_ok=True)
    for name in i2d_names:
        (pipe / name).write_text('stand-in for a mosaic; only renamed')

    monkeypatch.setattr(
        astrometry_checkpoint, 'run_visit_checkpoint',
        lambda *a, **kw: dict(
            passed=True, failures=[], unverified_blocking=[],
            record_path='/x/rec.json',
            corrections=[dict(visit=f'jw0{proposal}001001', exposure=1,
                              module=module, filtername=filt,
                              dra_onsky_mas=100.0, ddec_onsky_mas=0.0,
                              dec_deg=-28.7)]))
    monkeypatch.setenv('ASTROM_CHECKPOINT_APPLY', '1' if apply_ else '')
    monkeypatch.setenv('ASTROM_CHECKPOINT_WARN_ONLY', '1' if warn_only else '')

    options = types.SimpleNamespace(field=field, proposal_id=proposal,
                                    target=target, modules=module)
    return _cat._run_astrometry_stage_checkpoint(
        'm2', module, filt, str(tmp_path), str(tmp_path), proposal, options,
        {'refcat': {'all': None, 'sparse': None}}, context='test')


NIRISS_SGRC = dict(module='nis', filt='F200W', proposal='4147', field='012',
                   target='sgrc')
MIRI_TREASURY = dict(module='mirimage', filt='F770W', proposal='10678',
                     field='088', target='gc-treasury')
TREASURY_I2D = 'jw10678-o088_t001_miri_clear-f770w-mirimage_data_i2d.fits'
NEIGHBOUR_I2D = 'jw10678-o089_t001_miri_clear-f770w-mirimage_data_i2d.fits'


@pytest.mark.parametrize("case", [NIRISS_SGRC, MIRI_TREASURY],
                         ids=["niriss-sgrc", "miri-treasury"])
def test_the_refusal_still_stops_a_default_run(tmp_path, monkeypatch, case):
    """Nothing here weakens the stop itself: an ordinary run still refuses."""
    with pytest.raises(RuntimeError) as ex:
        _drive_no_channel(monkeypatch, tmp_path, **case)
    assert "NO table-driven correction channel" in str(ex.value)
    assert "Nothing was written." in str(ex.value)


@pytest.mark.parametrize("case", [NIRISS_SGRC, MIRI_TREASURY],
                         ids=["niriss-sgrc", "miri-treasury"])
def test_warn_only_demotes_the_refusal(tmp_path, monkeypatch, capsys, case):
    """``ASTROM_CHECKPOINT_WARN_ONLY=1`` demotes THIS raise too.

    It is the switch every other blocking error in
    ``_run_astrometry_stage_checkpoint`` honours, and NIRISS/sgrc has a written,
    justified procedure that depends on it (NOTES_niriss_sgrc_firstlight.md:
    first light runs under WARN_ONLY with 2-10 mas per-exposure jitter, above
    sgrc's 8.0 mas floor).  A refusal an operator cannot demote stops that run
    instead of warning it -- and the same switch governs the 10678 MIRI leg.
    """
    assert _drive_no_channel(monkeypatch, tmp_path, warn_only=True,
                             **case) is None
    out = capsys.readouterr().out
    assert "WARNING (ASTROM_CHECKPOINT_WARN_ONLY=1)" in out
    assert "NO table-driven correction channel" in out


def test_apply_quarantines_the_mosaics_the_refusal_measured(tmp_path,
                                                            monkeypatch):
    """The measurement stands whatever the channel is, so the stale tag does.

    ``_im0_badastrom`` is what keeps a measurably misaligned mosaic out of
    ``stage_release`` (the sickle F210M precedent, 2026-08-05: a band shipped
    short because that tag was the only thing that could stop it).  Writing the
    correction is what has nowhere to go; renaming the mosaic does not.
    """
    with pytest.raises(RuntimeError):
        _drive_no_channel(monkeypatch, tmp_path, apply_=True,
                          i2d_names=(TREASURY_I2D, NEIGHBOUR_I2D),
                          **MIRI_TREASURY)
    pipe = tmp_path / 'F770W' / 'pipeline'
    assert not (pipe / TREASURY_I2D).exists()
    tagged = pipe / TREASURY_I2D.replace('_i2d.fits', '_i2d_im0_badastrom.fits')
    assert tagged.exists()
    assert (tagged.parent / (tagged.name + '.why.json')).exists()
    # and the tag stays scoped to THIS observation: 10678 puts all 139 tiles in
    # one tree, so an unscoped rename quarantines 138 innocent neighbours.
    assert (pipe / NEIGHBOUR_I2D).exists()


def test_the_quarantine_survives_warn_only(tmp_path, monkeypatch):
    """Demoting the stop does not un-measure the misalignment."""
    _drive_no_channel(monkeypatch, tmp_path, apply_=True, warn_only=True,
                      i2d_names=(TREASURY_I2D,), **MIRI_TREASURY)
    pipe = tmp_path / 'F770W' / 'pipeline'
    assert not (pipe / TREASURY_I2D).exists()
    assert (pipe / TREASURY_I2D.replace('_i2d.fits',
                                        '_i2d_im0_badastrom.fits')).exists()


def test_a_measure_only_run_renames_nothing(tmp_path, monkeypatch):
    """Without ``ASTROM_CHECKPOINT_APPLY=1`` this function changes no file --
    the same gate every other rename in it sits behind."""
    with pytest.raises(RuntimeError):
        _drive_no_channel(monkeypatch, tmp_path, i2d_names=(TREASURY_I2D,),
                          **MIRI_TREASURY)
    assert (tmp_path / 'F770W' / 'pipeline' / TREASURY_I2D).exists()


def test_no_offsets_table_is_written_on_any_of_those_paths(tmp_path,
                                                           monkeypatch):
    """The point of the refusal: the numbers reach no table, ever."""
    for kwargs in (dict(), dict(warn_only=True), dict(apply_=True),
                   dict(apply_=True, warn_only=True)):
        d = tmp_path / str(len(list(tmp_path.iterdir())))
        d.mkdir()
        if kwargs.get('warn_only'):
            _drive_no_channel(monkeypatch, d, i2d_names=(TREASURY_I2D,),
                              **MIRI_TREASURY, **kwargs)
        else:
            with pytest.raises(RuntimeError):
                _drive_no_channel(monkeypatch, d, i2d_names=(TREASURY_I2D,),
                                  **MIRI_TREASURY, **kwargs)
        assert not (d / 'offsets').exists(), kwargs
