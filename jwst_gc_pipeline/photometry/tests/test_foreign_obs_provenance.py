"""A per-frame catalog's OBSERVATION is read from its provenance, not its name.

The frozen-stage checkpoint globs ``{filt}_*visit*_vgroup*_exp*``.  The ``*``
after the filter swallows the per-observation token, so on a directory holding
two observations the glob is obs-blind.  The name-based filter meant to narrow
it compares this run's token against each file's, and the per-frame writer
emits a token for proposals 2211/7213/6778 only -- so on cloudef (2092,
observations 002 and 005 under one basepath) the comparison was ``'' != ''``
and all 24 F360M catalogs were kept, 8 of them the OTHER observation's frames.
That is the aliasing input behind issue #298.

Every per-frame catalog records the crf it was measured on in
``meta['FILENAME']``, and that path carries the observation.  Reading it splits
the directory correctly without renaming any of the ~110k catalogs on disk.
"""
import os

import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry import cataloging
from jwst_gc_pipeline.photometry.cataloging import (
    _catalog_source_frame, _drop_foreign_obs_duplicates, _resolved_obsid)


def _catalog(dirpath, name, source_frame):
    p = str(dirpath) + "/" + name
    t = Table({"x": [1.0]})
    if source_frame is not None:
        t.meta["FILENAME"] = source_frame
    t.write(p)
    return p


# --------------------------------------------------------------- provenance

def test_untokened_legacy_catalogs_are_split_by_provenance(tmp_path, monkeypatch):
    """The cloudef case: 16 catalogs in one directory, 8 of them observation
    005's frames, NONE of them carrying a token in its name."""
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 2)
    fns = ([_catalog(tmp_path, f"f360m_nrcblong_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                     f"/x/jw02092002001_02101_{i:05d}_nrcblong_destreak_o002_crf.fits")
            for i in range(1, 9)]
           + [_catalog(tmp_path, f"f360m_nrcb_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                       f"/x/jw02092005001_02101_{i:05d}_nrcblong_destreak_o005_crf.fits")
              for i in range(1, 9)])
    kept = _drop_foreign_obs_duplicates(fns, "", "f360m", "m2", "merged",
                                        "cloudef", target_obs="002")
    assert len(kept) == 8
    assert all("_o002_" in _catalog_source_frame(f) for f in kept)


def test_the_untokened_writer_is_what_makes_the_name_check_a_no_op(tmp_path, monkeypatch):
    """Pins the premise.  This run's per-frame token on cloudef is '', so the
    name comparison keeps everything and the provenance read is the only thing
    doing the work.  If the token is ever made universal this test should be
    rewritten, not deleted: the split has to keep working either way."""
    from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import obs_token
    assert obs_token("2092", "002") == ""
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 2)
    fns = [_catalog(tmp_path, f"f360m_nrcb_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                    f"/x/jw02092005001_02101_{i:05d}_nrcblong_destreak_o005_crf.fits")
           for i in range(1, 4)]
    assert _drop_foreign_obs_duplicates(fns, "", "f360m", "m2", "merged",
                                        "cloudef", target_obs=None) == fns
    assert _drop_foreign_obs_duplicates(fns, "", "f360m", "m2", "merged",
                                        "cloudef", target_obs="002") == []


def test_a_catalog_with_unreadable_provenance_is_KEPT(tmp_path, monkeypatch, capsys):
    """Fail-safe.  Dropping real exposures builds a consensus from half the
    detectors and PASSES, which is worse than the duplicate it avoids."""
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 2)
    good = _catalog(tmp_path, "f360m_nrcblong_visit001_vgroup02101_exp00001_m2_daophot_basic.fits",
                    "/x/jw02092002001_02101_00001_nrcblong_destreak_o002_crf.fits")
    blind = _catalog(tmp_path, "f360m_nrcblong_visit001_vgroup02101_exp00002_m2_daophot_basic.fits",
                     None)
    kept = _drop_foreign_obs_duplicates([good, blind], "", "f360m",
                                        "m2", "merged", "cloudef",
                                        target_obs="002")
    assert blind in kept
    assert "KEEPING" in capsys.readouterr().out


def test_a_zero_length_catalog_reads_as_unreadable_not_as_a_crash(tmp_path):
    """What a killed job leaves, and what the repo's end-to-end test simulates
    with ``touch()``.  Measured rather than guessed: a zero-length ``.fits``
    raises ``OSError`` out of ``fits.getheader`` and never reaches
    ``Table.read``; a zero-length ``.ecsv``, which does, raises
    ``InconsistentTableError`` -- a ``ValueError`` subclass.  Neither may
    propagate."""
    empty = tmp_path / "f360m_nrcb_visit001_vgroup02101_exp00001_m2_daophot_basic.fits"
    empty.touch()
    assert _catalog_source_frame(str(empty)) is None
    empty_ecsv = tmp_path / "f360m_nrcb_visit001_vgroup02101_exp00002_m2_daophot_basic.ecsv"
    empty_ecsv.touch()
    assert _catalog_source_frame(str(empty_ecsv)) is None


def test_provenance_is_read_from_the_header_not_the_rows(tmp_path, monkeypatch):
    """This runs once per candidate catalog before every frozen-stage
    checkpoint.  ``Table.read`` pulls the whole binary table to reach one
    keyword: 0.60 s/file against 0.071 s for ``fits.getheader(ext=1)``,
    measured on brick F212N -- ~4 core-hours per archive pass per stage."""
    fn = _catalog(tmp_path, "f360m_nrcb_visit001_vgroup02101_exp00001_m2_daophot_basic.fits",
                  "/x/jw02092002001_02101_00001_nrcblong_destreak_o002_crf.fits")

    def _no(*a, **k):
        raise AssertionError("read the whole table to get one header keyword")

    monkeypatch.setattr(Table, "read", staticmethod(_no))
    assert _catalog_source_frame(fn).endswith("_o002_crf.fits")


def test_single_observation_fields_are_unaffected(tmp_path, monkeypatch):
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 1)
    fns = [_catalog(tmp_path, f"f212n_nrcb1_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                    f"/x/jw02221001001_02101_{i:05d}_nrcb1_destreak_o001_crf.fits")
           for i in range(1, 5)]
    kept = _drop_foreign_obs_duplicates(fns, "", "f212n", "m2", "merged",
                                        "brick", target_obs="001")
    assert len(kept) == 4


# ------------------------------------------------------------ obsid resolution

class _Opt:
    def __init__(self, **kw):
        self.target = None
        self.proposal_id = None
        self.field = None
        self.modules = "merged"
        for k, v in kw.items():
            setattr(self, k, v)


def test_an_explicit_field_wins():
    assert _resolved_obsid(_Opt(target="cloudef", proposal_id="2092",
                                field="005")) == "005"


def test_a_multi_observation_target_refuses_to_guess():
    """``fields.default_field_token`` answers ``obsids[0]`` -- on cloudef that
    is the same value whichever observation is running.  Using it here would
    make one of the two runs read its own frames as foreign, drop them, and
    PASS on the other observation's exposures.  None means "keep everything",
    which is what happened before this filter existed."""
    got = _resolved_obsid(_Opt(target="cloudef", proposal_id="2092", field=None))
    assert got is None, got


def test_a_single_observation_target_resolves():
    """Nothing to guess: the registry lists one obsid."""
    assert _resolved_obsid(_Opt(target="brick", proposal_id="2221",
                                field=None)) == "001"


def test_an_unregistered_target_resolves_to_none():
    assert _resolved_obsid(_Opt(target="not-a-field", proposal_id="9999")) is None
    assert _resolved_obsid(_Opt()) is None


# ------------------------------------------------------------ production wiring

def test_the_checkpoint_passes_its_own_observation_down(tmp_path, monkeypatch):
    """BEHAVIOURAL, not a source grep.  There is exactly one production call of
    ``_drop_foreign_obs_duplicates``, and every unit test above supplies
    ``target_obs`` itself -- so deleting the line that wires it leaves them all
    passing.  This one goes through ``_run_astrometry_stage_checkpoint``."""
    cut_bp = tmp_path / "cutouts" / "merged"
    (cut_bp / "F360M").mkdir(parents=True)
    for i in (1, 2):
        _catalog(cut_bp / "F360M",
                 f"f360m_nrcblong_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                 f"/x/jw02092002001_02101_{i:05d}_nrcblong_destreak_o002_crf.fits")

    seen = {}

    def _spy(fns, obs_token, filt, merge_label, module, target, target_obs=None):
        seen["target_obs"] = target_obs
        seen["n_in"] = len(fns)
        return []          # empty -> m2 prints "cannot run" and returns

    monkeypatch.setattr(cataloging, "_drop_foreign_obs_duplicates", _spy)
    monkeypatch.delenv("ASTROM_CHECKPOINT", raising=False)
    opts = _Opt(target="cloudef", proposal_id="2092", field="005",
                each_exposure=True, cutout_region="")
    cataloging._run_astrometry_stage_checkpoint(
        "m2", "merged", "F360M", str(cut_bp), str(tmp_path), "2092",
        opts, {}, context="test")
    assert seen.get("target_obs") == "005", seen
    assert seen.get("n_in") == 2, seen


# ------------------------------------------- the fail-safe must not fail CLOSED

def test_a_source_that_names_no_observation_is_KEPT_not_dropped(tmp_path, monkeypatch, capsys):
    """`want not in basename(src)` treated "this path does not spell an
    observation at all" identically to "it spells a DIFFERENT one".

    That is fail-CLOSED inside the one branch whose stated design is that an
    unidentifiable catalog is KEPT, and it empties the input: a source that is
    not a crf (`..._cal.fits`) drops every catalog, and at m1/m12/m2
    `if not fns:` only prints and returns, so the checkpoint silently ceases to
    exist rather than raising.
    """
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 2)
    fns = [_catalog(tmp_path, f"f360m_nrcb_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                    f"/x/jw02092002001_02101_{i:05d}_nrcblong_cal.fits")
           for i in range(1, 5)]
    kept = _drop_foreign_obs_duplicates(fns, "", "f360m", "m2", "merged",
                                        "cloudef", target_obs="002")
    assert kept == fns, kept
    assert "KEEPING" in capsys.readouterr().out


def test_a_source_naming_an_unrelated_observation_is_still_dropped(tmp_path, monkeypatch):
    """The relaxation must not become a pardon: a source that DOES name an
    observation, and names one that is not ours, is foreign whether or not that
    observation is a registered sibling."""
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 2)
    fns = [_catalog(tmp_path, f"f360m_nrcb_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                    f"/x/jw02092999001_02101_{i:05d}_nrcblong_destreak_o999_crf.fits")
           for i in range(1, 5)]
    assert _drop_foreign_obs_duplicates(fns, "", "f360m", "m2", "merged",
                                        "cloudef", target_obs="002") == []


def test_a_joint_spelling_in_the_source_is_not_read_as_foreign(tmp_path, monkeypatch):
    """`_SRC_OBS_RE` accepts `_o002-998_`, so a product named that way -- none
    are today -- must not be read as foreign to the run it belongs to.  The
    joint spelling stays in `want` alongside its decomposed parts."""
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 2)
    ours = [_catalog(tmp_path, f"f770w_mirimage_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                     f"/x/jw05365002001_02101_{i:05d}_mirimage_destreak_o002-998_crf.fits")
            for i in range(1, 3)]
    theirs = [_catalog(tmp_path, f"f770w_mirimage_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits".replace("exp0000", "exp0001"),
                       f"/x/jw05365007001_02101_{i:05d}_mirimage_destreak_o007_crf.fits")
              for i in range(1, 3)]
    kept = _drop_foreign_obs_duplicates(ours + theirs, "", "f770w", "m2",
                                        "merged", "sgrb2",
                                        target_obs="002-998")
    assert kept == ours, kept


# ------------------------------------------------------- JOINT observations

def test_a_joint_obsid_keeps_BOTH_of_its_observations(tmp_path, monkeypatch, capsys):
    """A joint obsid is a SET, not a string.

    sgrb2's MIRI is registered `002-998` and sickle's `001-002`, and
    `_resolved_obsid` hands the joint token straight through -- but no crf is
    ever named `_o002-998_`; the real names are `_o002_` and `_o998_`.  Tested
    as a single substring, `want` matched nothing and EVERY file was dropped:
    sgrb2 F770W 60 -> 0, sickle F770W 60 -> 0, silent at m2 and
    AstrometryRegressionError at m5, on release-path fields.
    """
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 2)

    def _c(obs, exp):
        return _catalog(tmp_path,
                        f"f770w_mirimage_visit001_vgroup02101_exp{exp:05d}_m2_daophot_basic.fits"
                        .replace("exp0", f"exp{obs[-1]}"),
                        f"/x/jw05365{obs}001_02101_{exp:05d}_mirimage_destreak_o{obs}_crf.fits")

    ours = [_c("002", i) for i in range(1, 4)] + [_c("998", i) for i in range(1, 4)]
    theirs = [_c("007", i) for i in range(1, 4)]
    kept = _drop_foreign_obs_duplicates(ours + theirs, "", "f770w", "m2",
                                        "merged", "sgrb2",
                                        target_obs="002-998")
    assert sorted(kept) == sorted(ours), kept
    # ... and the log names what was demanded, not just what was found
    out = capsys.readouterr().out
    assert "_o002_" in out and "_o998_" in out, out


def test_dropping_every_catalog_says_so_loudly(tmp_path, monkeypatch, capsys):
    """The loudest failure was the quietest line: "excluded 60 ...
    (['<untokened>'])" with no mention of what `want` was, so an operator had
    no way to see that the demanded token can never appear in a crf name."""
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 2)
    fns = [_catalog(tmp_path, f"f770w_mirimage_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                    f"/x/jw05365007001_02101_{i:05d}_mirimage_destreak_o007_crf.fits")
           for i in range(1, 5)]
    assert _drop_foreign_obs_duplicates(fns, "", "f770w", "m2", "merged",
                                        "sgrb2", target_obs="002") == []
    out = capsys.readouterr().out
    assert "4 of 4" in out, out
    assert "EVERY catalog" in out, out
    assert "_o002_" in out, out


# ------------------------------------------------- --field is not trusted blind

def test_a_field_that_is_not_an_obsid_of_this_target_is_refused(capsys):
    """`--field` was taken verbatim and `submit_cataloging.sbatch:63` is
    `FIELD=${FIELD:-012}`, so a typo or a stale default produced a `want` that
    matches nothing -- and the new consequence of that is a silently emptied
    m2 gate or a fatal m5.  An obsid this target does not have cannot be this
    run's."""
    got = _resolved_obsid(_Opt(target="cloudef", proposal_id="2092",
                               field="012"))
    assert got is None, got
    assert "not an obsid of" in capsys.readouterr().out


def test_a_real_obsid_and_a_real_joint_obsid_both_pass_validation():
    assert _resolved_obsid(_Opt(target="cloudef", proposal_id="2092",
                                field="005")) == "005"
    assert _resolved_obsid(_Opt(target="sgrb2", proposal_id="5365",
                                field="002-998",
                                modules="mirimage")) == "002-998"


def test_an_unregistered_target_still_takes_its_field_verbatim():
    """Nothing to validate against; refusing would be a regression for any
    field not yet in the registry."""
    assert _resolved_obsid(_Opt(target="not-a-field", proposal_id="9999",
                                field="007")) == "007"


def test_the_checkpoint_runs_the_REAL_filter_on_mixed_provenance(tmp_path, monkeypatch):
    """The wiring test above stubs `_drop_foreign_obs_duplicates` with a spy,
    so it pins the wiring and never runs the real filter.  A test of THAT
    shape is what surfaced the joint-obsid blocker; this one runs the filter
    end to end through `_run_astrometry_stage_checkpoint` and asserts on what
    reaches the consensus.
    """
    cut_bp = tmp_path / "cutouts" / "merged"
    (cut_bp / "F770W").mkdir(parents=True)
    for obs in ("002", "998", "007"):
        for i in (1, 2):
            _catalog(cut_bp / "F770W",
                     f"f770w_mirimage_visit001_vgroup{obs}01_exp{i:05d}_m2_daophot_basic.fits",
                     f"/x/jw05365{obs}001_{obs}01_{i:05d}_mirimage_destreak_o{obs}_crf.fits")

    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 2)
    seen = {}
    real = cataloging._drop_foreign_obs_duplicates

    def _watch(*a, **k):
        out = real(*a, **k)
        seen["kept"] = list(out)
        return out

    monkeypatch.setattr(cataloging, "_drop_foreign_obs_duplicates", _watch)
    monkeypatch.delenv("ASTROM_CHECKPOINT", raising=False)
    opts = _Opt(target="sgrb2", proposal_id="5365", field="002-998",
                modules="mirimage", each_exposure=True, cutout_region="")
    try:
        cataloging._run_astrometry_stage_checkpoint(
            "m2", "merged", "F770W", str(cut_bp), str(tmp_path), "5365",
            opts, {}, context="test")
    except KeyError:
        # These are one-column stand-ins, so the consensus builder trips on a
        # missing photometry column further down.  That is past the point
        # under test: the filter has already run on the real inputs and `seen`
        # holds its verdict.  KeyError specifically -- anything else should
        # surface.
        pass
    kept = [os.path.basename(f) for f in seen.get("kept", [])]
    assert len(kept) == 4, kept                     # both halves of the joint
    assert not any("_o007_" in _catalog_source_frame(f)
                   for f in seen["kept"]), kept


def test_a_registered_sibling_with_no_frames_on_disk_empties_the_input_LOUDLY(
        tmp_path, monkeypatch, capsys):
    """The shape that survives on wd1 and wd2, and it is not hypothetical.

    wd1's registry lists obsids ['001','003'] and every one of its 96 F200W
    crf is `_o001_`; wd2 lists ['003','005'] and all 64 are `_o005_`.  So a
    `--field 003` wd1 run passes obsid validation -- 003 IS a registered obsid
    -- and then drops 96 of 96.  There is no way to tell from the registry
    alone whether that means "wrong --field" or "that observation has not been
    reduced yet", so the filter cannot silently decide.  What it must do is
    say so: at m1/m12/m2 an empty input only prints and returns, and a frozen
    gate with no inputs is a silently disabled gate, not a pass.
    """
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 2)
    fns = [_catalog(tmp_path, f"f200w_nrcb1_visit001_vgroup02101_exp{i:05d}_m2_daophot_basic.fits",
                    f"/x/jw01905001001_02101_{i:05d}_nrcb1_destreak_o001_crf.fits")
           for i in range(1, 5)]
    assert _drop_foreign_obs_duplicates(fns, "", "f200w", "m2", "merged",
                                        "wd1", target_obs="003") == []
    out = capsys.readouterr().out
    assert "4 of 4" in out and "EVERY catalog" in out, out
    # ... and the same input with the observation that IS on disk is untouched
    assert _drop_foreign_obs_duplicates(fns, "", "f200w", "m2", "merged",
                                        "wd1", target_obs="001") == fns


def test_a_chunked_and_unchunked_copy_of_one_exposure_are_one_identity(
        tmp_path, monkeypatch):
    """`_identity` strips `_chunk\\d+of\\d+`.  The single-observation branch
    has a comment explaining why; the shared branch's copy had none and
    removing it left the suite green.

    It only bites on the UNREADABLE-provenance path -- a catalog whose
    provenance can be read is settled before `_identity` is consulted -- which
    is why nothing reached it.  There, a tokened copy of the same exposure
    means the untokened one is redundant whatever it is; and the checkpoint
    collapses `_chunk\\d+of\\d+` itself further down, so
    `..._m2_chunk00of02_...` and `..._m2_...` land on one `exposure_key`.
    Comparing raw basenames keeps both and reaches DuplicateExposureError by a
    different route.
    """
    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 2)
    tokened = _catalog(
        tmp_path,
        "f360m_nrcblong_o002_visit001_vgroup02101_exp00001_m2_daophot_basic.fits",
        "/x/jw02092002001_02101_00001_nrcblong_destreak_o002_crf.fits")
    # same exposure, pre-token name, chunked, and NO readable provenance
    blind_chunk = _catalog(
        tmp_path,
        "f360m_nrcblong_visit001_vgroup02101_exp00001_m2_chunk00of02_daophot_basic.fits",
        None)
    kept = _drop_foreign_obs_duplicates([tokened, blind_chunk], "_o002",
                                        "f360m", "m2", "merged", "cloudef",
                                        target_obs="002")
    assert kept == [tokened], kept
