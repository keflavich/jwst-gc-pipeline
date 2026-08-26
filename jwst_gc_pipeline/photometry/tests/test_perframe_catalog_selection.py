"""Which per-frame catalogs the astrometry stage checkpoint ingests.

Both of these are filename/identity CONVENTIONS, and both drifted silently
once already:

  * a trailing ``_exp*`` swallowed the grouped-fit variant, so every exposure
    was ingested twice and the checkpoint "measured" a misalignment that was
    pure bookkeeping (arches F212N: 96 plain + 48 grouped);
  * pinning ``_exp`` to five digits to fix that then matched NOTHING at the
    resbgsub stages (``_exp00001_resbgsub_m5_...``), silently turning every
    frozen-stage gate into a no-op.

So the accept-list is tested at each stage against all four real name shapes.
"""
import os
import types

from jwst_gc_pipeline import fields

import pytest
from astropy.io.registry import IORegistryError

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    AstrometryRegressionError,
)
from jwst_gc_pipeline.photometry.cataloging import (
    _drop_foreign_obs_duplicates, _drop_module_level_duplicates,
    _perframe_catalog_re, _run_astrometry_stage_checkpoint,
)


def _name(det="nrcb1", exp=1, seg="", label="m2", chunk=None, filt="f212n",
          tok=""):
    parts = [filt, det]
    if tok:
        parts.append(tok.lstrip("_"))
    parts += ["visit001", "vgroup02101", f"exp{exp:05d}"]
    if seg:
        parts.append(seg)
    parts.append(label)
    if chunk:
        parts.append(chunk)
    return "_".join(parts) + "_daophot_basic.fits"


@pytest.mark.parametrize("label", ["m2", "m5", "m7"])
def test_accepts_plain_resbgsub_and_chunk(label):
    acc = _perframe_catalog_re(label)
    assert acc.search(_name(label=label))
    assert acc.search(_name(seg="resbgsub", label=label))
    assert acc.search(_name(label=label, chunk="chunk1of4"))
    assert acc.search(_name(seg="resbgsub", label=label, chunk="chunk2of4"))


@pytest.mark.parametrize("label", ["m2", "m5", "m7"])
def test_rejects_grouped_fit_variants(label):
    """The grouped fit is a SECOND measurement of the same exposure."""
    acc = _perframe_catalog_re(label)
    assert not acc.search(_name(seg="group", label=label))
    assert not acc.search(_name(seg="resbgsub_group", label=label))


def test_rejects_other_stage_labels():
    acc = _perframe_catalog_re("m2")
    assert not acc.search(_name(label="m5"))
    assert not acc.search(_name(seg="resbgsub", label="m5"))


def test_rejects_non_five_digit_exposure():
    acc = _perframe_catalog_re("m2")
    assert not acc.search("f212n_nrcb1_visit001_vgroup02101_exp1_m2_daophot_basic.fits")


def test_drops_bare_module_when_per_detector_present():
    fns = [_name(det="nrcb", filt="f162m"),
           *[_name(det=f"nrcb{i}", filt="f162m") for i in (1, 2, 3, 4)]]
    kept = _drop_module_level_duplicates(fns, "f162m", "m2", "nrcb")
    assert all("_nrcb_visit" not in os.path.basename(f) for f in kept)
    assert len(kept) == 4


def test_keeps_bare_module_when_it_is_the_only_one():
    """A genuinely module-level field must not be emptied."""
    fns = [_name(det="nrcb", filt="f162m")]
    assert _drop_module_level_duplicates(fns, "f162m", "m2", "nrcb") == fns


def test_keeps_long_detectors():
    """nrcalong/nrcblong ARE the detector for LW -- never module-level dupes."""
    fns = [_name(det="nrcalong", filt="f405n"), _name(det="nrcblong", filt="f405n")]
    assert _drop_module_level_duplicates(fns, "f405n", "m2", "nrcb") == fns


def test_long_detector_supersedes_its_bare_module():
    """Issue #298: for an LW filter `nrcalong` IS the detector, so a bare
    `nrca` catalog can only be the SAME detector written by a run invoked
    `--modules nrca`.  Keeping both ingests one physical frame twice under two
    module tokens, and the m2 checkpoint then writes offsets rows under both --
    which `unified_alignment._read_consensus` resolves to two rows for one
    frame and refuses to reduce.

    This test previously asserted the opposite.  The reasoning it encoded --
    "a bare nrca must not be dropped on the strength of an LW catalog with no
    SW per-detector catalog behind it" -- cannot arise: this function sees ONE
    filter's catalogs and a filter is imaged by one channel, so a SW filter's
    glob never contains `nrcalong`.  What that reasoning genuinely protects is
    covered by test_keeps_bare_module_when_it_is_the_only_one.
    """
    fns = [_name(det="nrca", filt="f405n"), _name(det="nrcalong", filt="f405n")]
    kept = _drop_module_level_duplicates(fns, "f405n", "m2", "nrca")
    assert all("_nrca_visit" not in os.path.basename(f) for f in kept)
    assert len(kept) == 1


def test_long_detector_supersedes_only_its_own_module():
    """`nrcblong` must not drop a bare `nrca`."""
    fns = [_name(det="nrca", filt="f405n"), _name(det="nrcblong", filt="f405n")]
    assert _drop_module_level_duplicates(fns, "f405n", "m2", "nrca") == fns


def test_cloudef_f360m_shape_is_resolved():
    """The exact input set that produced issue #298: 8 bare-module catalogs
    beside 16 `long` ones.

    IMPORTANT, and the reason this test is not the whole story: on the real
    cloudef tree those 8 are observation **005's own frames**, so for an o005
    run this rule discards o005's data and keeps o002's.  It is only correct
    because `_drop_foreign_obs_duplicates` should have removed the foreign set
    FIRST -- and on cloudef it does not, because proposal 2092 gets an empty
    obs token and the filter compares `'' != ''`.

    PR #313 closes that by reading the observation from each catalog's own
    `meta['FILENAME']`, and this branch is now stacked on it -- see
    `test_the_foreign_drop_runs_first_and_saves_o005s_own_frames` below, which
    exercises the two rules in production order.  On its own this rule is right
    about the SPELLING and blind to the OBSERVATION.
    """
    fns = ([_name(det="nrcb", filt="f360m", exp=i) for i in range(1, 9)]
           + [_name(det="nrcblong", filt="f360m", exp=i) for i in range(1, 9)]
           + [_name(det="nrcalong", filt="f360m", exp=i) for i in range(1, 9)])
    kept = _drop_module_level_duplicates(fns, "f360m", "m2", "merged")
    assert len(kept) == 16
    assert not any("_nrcb_visit" in os.path.basename(f) for f in kept)


def test_the_module_rule_is_blind_to_the_observation():
    """States the limit explicitly rather than leaving it implied by the test
    above: this rule cannot tell whose frames it is dropping.  Ordering is what
    makes it safe -- the foreign-observation filter runs first."""
    own = [_name(det="nrcb", filt="f360m", exp=i) for i in range(1, 9)]
    other = [_name(det="nrcblong", filt="f360m", exp=i) for i in range(1, 9)]
    kept = _drop_module_level_duplicates(own + other, "f360m", "m2", "merged")
    # the bare set goes whether or not it is this run's own observation
    assert len(kept) == 8
    assert all("_nrcblong_" in os.path.basename(f) for f in kept)


def test_bare_module_dropped_only_for_its_own_module():
    """A bare `nrca` is not superseded by nrcb1..4."""
    fns = [_name(det="nrca", filt="f212n"),
           *[_name(det=f"nrcb{i}", filt="f212n") for i in (1, 2, 3, 4)]]
    kept = _drop_module_level_duplicates(fns, "f212n", "m2", "nrcb")
    assert any("_nrca_visit" in os.path.basename(f) for f in kept)


# ---------------------------------------------------------------------------
# catalogs belonging to another observation / proposal in the same directory
# ---------------------------------------------------------------------------
#
# The glob is `{filt}_*visit*_vgroup*_exp*`, so the `*` swallows the detector
# AND the per-observation token.  wd1's obs 001 and 003 share a directory,
# VISIT=001 and the same (vgroup, exp) tuples; ngc6334's 6778 and 7213 share a
# directory, a filter list and obsid 001; and every field carries pre-token
# copies of its own frames.  `exposure_key` carries no obs/proposal, so every
# one of those neighbours collides -- issue #259, measured on gc2211 F200W
# (592 files -> 192 keys, all duplicated 2-5x).
#
# gc2211 was the original example and is no longer one: its five observations
# were split into a tree each on 2026-08-21, which removes the collision by
# construction.  wd1 (1905, nircam 001+003) still has the property, so the
# behaviour is exercised against a field that really does share a tree.

def _mixed_shared_tree():
    """One exposure of one detector, as written by two wd1 runs plus the
    pre-token name."""
    return [_name(filt="f200w", tok=t) for t in ("_o001", "_o003", "")]


def test_tokened_run_keeps_only_its_own_observation():
    kept = _drop_foreign_obs_duplicates(_mixed_shared_tree(), "_o001",
                                        "f200w", "m2", "nrcb", "wd1")
    assert kept == [_name(filt="f200w", tok="_o001")]


def test_untokened_run_drops_the_tokened_siblings():
    """Otherwise a run that writes pre-token names still ingests every
    observation's tokened catalogs beside it."""
    kept = _drop_foreign_obs_duplicates(_mixed_shared_tree(), "",
                                        "f200w", "m2", "nrcb", "wd1")
    assert kept == [_name(filt="f200w", tok="")]


def test_proposal_token_form_separates_the_two_ngc6334_proposals():
    """ngc6334's 6778 and 7213 share a directory, a filter list AND obsid 001,
    so the disambiguator is the PROPOSAL (`_j6778`), not `_o001`."""
    fns = [_name(filt="f200w", tok=t) for t in ("_j6778", "_j7213", "")]
    assert _drop_foreign_obs_duplicates(fns, "_j7213", "f200w", "m2", "nrcb",
                                        "ngc6334") \
        == [_name(filt="f200w", tok="_j7213")]
    assert _drop_foreign_obs_duplicates(fns, "_j6778", "f200w", "m2", "nrcb",
                                        "ngc6334") \
        == [_name(filt="f200w", tok="_j6778")]


def test_nothing_is_dropped_when_the_directory_holds_one_observation():
    fns = [_name(filt="f200w", tok="_o023", exp=e) for e in (1, 2, 3)]
    assert _drop_foreign_obs_duplicates(fns, "_o023", "f200w", "m2", "nrcb",
                                        "wd1") == fns


def test_the_drop_is_reported(capsys):
    """It narrows the checkpoint's input, so it may not happen silently."""
    _drop_foreign_obs_duplicates(_mixed_shared_tree(), "_o001", "f200w", "m2",
                                 "nrcb", "wd1")
    out = capsys.readouterr().out
    assert "excluded 2 of 3 foreign-observation per-frame catalog(s)" in out
    assert "o003" in out and "<untokened>" in out
    assert "_o001" in out


def test_checkpoint_passes_its_own_token_to_the_filter(tmp_path, capsys):
    """End-to-end wiring, gc2211_o023: the per-frame writer and the consensus namer
    return the SAME token here, so the filter keeps only o023."""
    (tmp_path / "F200W").mkdir(parents=True)
    (tmp_path / "catalogs").mkdir(parents=True)
    for tok in ("_o046", "_o050"):
        (tmp_path / "F200W" / _name(filt="f200w", tok=tok)).touch()
    _run_astrometry_stage_checkpoint(
        "m2", "nrcb", "f200w", str(tmp_path), str(tmp_path), "2211",
        types.SimpleNamespace(cutout_region="", proposal_id="2211",
                              field="023", target="gc2211_o023"), {}, context="test")
    out = capsys.readouterr().out
    assert "excluded 2 of 2 foreign-observation per-frame catalog(s)" in out
    # nothing of this observation is left, so the checkpoint cannot run -- it
    # must say so rather than measure the neighbours.
    assert "NO per-frame catalogs matched" in out


# ---------------------------------------------------------------------------
# a frozen stage with no per-frame inputs
# ---------------------------------------------------------------------------

def _layout(tmp_path, stage, *, perframe, merged, filt="f200w", module="nis"):
    (tmp_path / filt.upper()).mkdir(parents=True, exist_ok=True)
    (tmp_path / "catalogs").mkdir(parents=True, exist_ok=True)
    if perframe:
        (tmp_path / filt.upper() /
         _name(det="nrcb1", seg="resbgsub", label=stage, filt=filt)).touch()
    if merged:
        (tmp_path / "catalogs" /
         f"{filt}_{module}_indivexp_merged_resbgsub_{stage}_dao_basic.fits").touch()


def _run(tmp_path, stage, module="nis", filt="f200w"):
    return _run_astrometry_stage_checkpoint(
        stage, module, filt, str(tmp_path), str(tmp_path), "4147",
        types.SimpleNamespace(cutout_region=""), {}, context="test")


def test_frozen_stage_missing_inputs_raises_when_the_merge_ran(tmp_path):
    """The merge produced output but nothing matched -> the gate's inputs are
    broken.  That is the case worth failing on."""
    _layout(tmp_path, "m5", perframe=False, merged=True)
    with pytest.raises(AstrometryRegressionError, match="(?i)silently disabled"):
        _run(tmp_path, "m5")


def test_frozen_stage_without_the_stage_at_all_is_skipped(tmp_path):
    """sgrc/niriss F200W legitimately stops at resbgsub_m6 and has no m7
    products.  Hard-stopping it would be a false failure."""
    _layout(tmp_path, "m7", perframe=False, merged=False)
    _run(tmp_path, "m7")        # must not raise


def test_frozen_stage_missing_inputs_has_a_narrow_escape(tmp_path, monkeypatch):
    """...and the escape is scoped to this check, not every checkpoint."""
    monkeypatch.setenv("ASTROM_ALLOW_MISSING_PERFRAME", "1")
    _layout(tmp_path, "m5", perframe=False, merged=True)
    _run(tmp_path, "m5")        # must not raise


def test_m2_missing_inputs_does_not_raise(tmp_path):
    """m2 is the CORRECTING stage, not a frozen one -- unchanged behaviour."""
    _layout(tmp_path, "m2", perframe=False, merged=True)
    _run(tmp_path, "m2")        # must not raise


# --------------------------------------------------------------------------
# A filter imaged by ONE observation: the pre-token names are this run's own
# exposures, not a foreign observation's (issue #259 review).
# --------------------------------------------------------------------------
def _pf(filt, det, tok, exp, stage="m2"):
    t = f"_{tok}" if tok else ""
    return (f"/x/{filt.lower()}_{det}{t}_visit001_vgroup02201"
            f"_exp{exp:05d}_{stage}_daophot_basic.fits")


def test_single_observation_filter_keeps_its_untokened_catalogs():
    """ngc6334 F090W is declared by 6778 and NOT by 7213, so every F090W
    catalog on disk is 6778's whatever its name.  Its nrca detectors exist ONLY
    under the pre-token name: dropping them would build the consensus from nrcb
    alone and PASS, which is worse than the duplicate it avoids."""
    fns = [_pf("F090W", "nrcb1", "j6778", 1), _pf("F090W", "nrcb1", None, 1),
           _pf("F090W", "nrca1", None, 1), _pf("F090W", "nrca2", None, 2)]
    kept = _drop_foreign_obs_duplicates(
        fns, "j6778", "F090W", "m2", "all", "ngc6334")
    dets = sorted(os.path.basename(f).split("_")[1] for f in kept)
    assert dets == ["nrca1", "nrca2", "nrcb1"], kept
    # the tokened copy wins where both spellings name the same exposure
    assert _pf("F090W", "nrcb1", "j6778", 1) in kept
    assert _pf("F090W", "nrcb1", None, 1) not in kept


def test_shared_filter_still_drops_untokened():
    """wd1 images F200W in obs 001 AND 003 from one tree, so an untokened
    basename cannot say which one wrote it and must go."""
    fns = [_pf("F200W", "nrcb1", "o001", 1), _pf("F200W", "nrcb1", "o003", 1),
           _pf("F200W", "nrcb1", None, 1)]
    kept = _drop_foreign_obs_duplicates(
        fns, "o001", "F200W", "m2", "all", "wd1")
    assert kept == [_pf("F200W", "nrcb1", "o001", 1)], kept


def test_shared_filter_of_the_same_field_is_decided_per_filter():
    """ngc6334 F200W IS shared (6778 and 7213 both image it -- the collision the
    `_j` token was introduced for), while F090W is not.  The rule is per
    filter, not per field."""
    assert fields.filter_observation_count("ngc6334", "F090W") == 1
    assert fields.filter_observation_count("ngc6334", "F200W") == 2
    # gc2211 is five FIELDS now, one obsid each; wd1 still shares a tree.
    assert fields.filter_observation_count("wd1", "F200W") == 2
    assert fields.filter_observation_count("gc2211_o023", "F200W") == 1


def test_shared_filter_without_a_written_token_keeps_everything():
    """The filter can only match a token the per-frame WRITER emits.

    ``crowdsource_catalogs_long.obs_token`` returns '' for every proposal except
    2211/7213/6778, so on wd1/wd2/cloudef/sickle no catalog on disk carries a
    token at all.  Filtering those directories on the CONSENSUS token (which
    falls back to ``_o{obsid}`` unconditionally) asks for a spelling that was
    never written and empties the checkpoint's input -- silent at m2, and
    ``AstrometryRegressionError`` at m3+.  Measured on the real tree: wd1 F200W
    96 -> 0, wd2 32 -> 0, cloudef F162M 72 -> 0, sickle F187N 192 -> 0.
    """
    fns = [_pf("F200W", f"nrc{m}{i}", None, 1)
           for m in "ab" for i in (1, 2, 3, 4)]
    assert fields.filter_observation_count("wd1", "F200W") > 1, "premise: shared"
    kept = _drop_foreign_obs_duplicates(fns, "", "F200W", "m2", "all", "wd1")
    assert kept == fns, kept


def test_untokened_chunked_and_unchunked_copies_are_deduped():
    """`_chunk\\d+of\\d+` is collapsed further down the checkpoint, so the two
    spellings land on ONE ``exposure_key``.  Comparing raw basenames keeps both
    and reaches ``DuplicateExposureError`` by a different route."""
    plain = _pf("F212N", "nrcb1", None, 1)
    chunked = plain.replace("_m2_", "_m2_chunk00of02_")
    kept = _drop_foreign_obs_duplicates(
        [plain, chunked], "", "F212N", "m2", "all", "brick")
    assert len(kept) == 1, kept


def test_two_tokens_on_a_single_observation_filter_are_deduped():
    """A not-shared filter carrying two different tokens still collides on
    ``exposure_key``; only untokened files used to be considered for dropping."""
    kept = _drop_foreign_obs_duplicates(
        [_pf("F090W", "nrcb1", "j6778", 1), _pf("F090W", "nrcb1", "j7213", 1)],
        "j6778", "F090W", "m2", "all", "ngc6334")
    assert len(kept) == 1, kept


def test_miri_filter_counts_miri_observations_not_nircam():
    """``Observation.filters`` is the shared NIRCAM_MIRI list, so counting a
    MIRI band against the NIRCam obsids is wrong in both directions."""
    assert fields.filter_observation_count("sgrb2", "F770W") == 3   # miri 001/002/998
    assert fields.filter_observation_count("sgrb2", "F212N") == 1   # nircam 001
    assert fields.filter_observation_count("cloudef", "F770W") == 3  # miri 004/006/008
    # cloudef's NIRCam side is ONE observation since obs 005 became
    # `cloudef_controlfield`; the MIRI count above is untouched by that split,
    # which is the asymmetry this test exists to hold on to.
    assert fields.filter_observation_count("cloudef", "F162M") == 1  # nircam 002
    assert fields.filter_observation_count(
        "cloudef_controlfield", "F162M") == 1                        # nircam 005


def test_checkpoint_filters_on_the_written_token_not_the_consensus_name(
        tmp_path, capsys):
    """End-to-end wiring, wd1: the two tokens DIFFER.

    ``consensus_obs_token('1905', '001')`` is ``_o001`` while the per-frame
    writer emits '', and no wd1 catalog on disk carries a token.  Filtering on
    the consensus name matches nothing and the checkpoint reports no inputs;
    filtering on the written token keeps every frame.
    """
    from jwst_gc_pipeline.photometry.consensus_catalog import consensus_obs_token
    from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import obs_token
    assert consensus_obs_token("1905", "001") == "_o001"
    assert obs_token("1905", "001") == ""
    assert fields.filter_observation_count("wd1", "F200W") > 1, "premise: shared"

    (tmp_path / "F200W").mkdir(parents=True)
    (tmp_path / "catalogs").mkdir(parents=True)
    for det in ("nrcb1", "nrcb2"):
        (tmp_path / "F200W" / _name(filt="f200w", det=det, tok="")).touch()
    # The files are empty touch()es, so the checkpoint gets past selection and
    # then fails READING them -- which is the point: it had inputs to read.
    with pytest.raises(IORegistryError):
        _run_astrometry_stage_checkpoint(
            "m2", "nrcb", "f200w", str(tmp_path), str(tmp_path), "1905",
            types.SimpleNamespace(cutout_region="", proposal_id="1905",
                                  field="001", target="wd1"), {}, context="test")
    out = capsys.readouterr().out
    assert "NO per-frame catalogs matched" not in out, out
    assert "excluded" not in out, out


def test_the_foreign_drop_runs_first_and_saves_o005s_own_frames(tmp_path, monkeypatch):
    """The two rules in PRODUCTION ORDER, on the cloudef shape.

    `_drop_module_level_duplicates` alone discards whichever observation
    happens to be spelled bare, and on cloudef F360M that is observation 005's
    own 8 frames -- so an o005 run derived its corrections entirely from o002's
    frames and wrote them under `Visit=jw02092005001`, 11.4 mas away in the
    median.  What makes the module rule safe is that
    `_drop_foreign_obs_duplicates` removes the other observation FIRST
    (`cataloging.py:3880` before `:3913`), and that filter can only do so
    because it reads each catalog's own `meta['FILENAME']` rather than its
    name -- the per-frame token is '' for proposal 2092 either way.
    """
    from astropy.table import Table
    from jwst_gc_pipeline.photometry.cataloging import (
        _drop_foreign_obs_duplicates, _drop_module_level_duplicates)

    monkeypatch.setattr("jwst_gc_pipeline.fields.filter_observation_count",
                        lambda *a, **k: 2)

    def _cat(det, exp, obs):
        fn = str(tmp_path / (f"f360m_{det}_visit001_vgroup02101_exp{exp:05d}"
                             f"_m2_daophot_basic.fits"))
        t = Table({"x": [1.0]})
        t.meta["FILENAME"] = (f"/x/jw02092{obs}001_02101_{exp:05d}_"
                              f"{det}_destreak_o{obs}_crf.fits")
        t.write(fn)
        return fn

    # o005's frames carry the BARE spelling; o002's carry `long`
    fns = ([_cat("nrcb", i, "005") for i in range(1, 9)]
           + [_cat("nrcblong", i, "002") for i in range(1, 9)])

    # o005 run: the foreign drop keeps its own 8, and the module rule then has
    # nothing to supersede them with
    own = _drop_foreign_obs_duplicates(fns, "", "f360m", "m2", "merged",
                                       "cloudef", target_obs="005")
    kept = _drop_module_level_duplicates(own, "f360m", "m2", "merged")
    assert len(kept) == 8, kept
    assert all("_nrcb_visit" in os.path.basename(f) for f in kept)

    # and the o002 run keeps o002's
    own2 = _drop_foreign_obs_duplicates(fns, "", "f360m", "m2", "merged",
                                        "cloudef", target_obs="002")
    kept2 = _drop_module_level_duplicates(own2, "f360m", "m2", "merged")
    assert len(kept2) == 8, kept2
    assert all("_nrcblong_" in os.path.basename(f) for f in kept2)

    # WITHOUT the foreign drop, the module rule silently hands o005's run
    # o002's frames -- the failure this ordering prevents
    blind = _drop_module_level_duplicates(fns, "f360m", "m2", "merged")
    assert all("_nrcblong_" in os.path.basename(f) for f in blind)


# ---------------------------------------------------------------------------
# a stale tokened copy must not shadow a re-catalogued untokened one (#391)
# ---------------------------------------------------------------------------

def _write(tmp_path, name, mtime):
    p = tmp_path / name
    p.write_bytes(b"x")
    os.utime(p, (mtime, mtime))
    return str(p)


def test_newer_untokened_copy_wins_over_a_stale_tokened_one(tmp_path):
    """cloudef_controlfield F360M: eight `_o005_` nrcblong catalogs recovered
    from a June quarantine sat beside the SAME exposures re-catalogued in
    August, and the June copies won because their names carried a token.  The
    August ones are ~10% longer, and F360M's consensus paired a June nrcblong
    with a same-day nrcalong."""
    old = _write(tmp_path, _name(filt="f360m", det="nrcblong", tok="_o005"),
                 1_700_000_000)
    new = _write(tmp_path, _name(filt="f360m", det="nrcblong"),
                 1_700_086_400)      # one day later
    kept = _drop_foreign_obs_duplicates([old, new], "", "f360m", "m2",
                                        "nrcblong", "cloudef_controlfield")
    assert kept == [new], "the copy written last must be the one kept"


def test_newer_tokened_copy_still_wins(tmp_path):
    """The rule is recency, not spelling: a tokened copy written after the
    untokened one is still the one kept."""
    new = _write(tmp_path, _name(filt="f360m", det="nrcblong", tok="_o005"),
                 1_700_086_400)
    old = _write(tmp_path, _name(filt="f360m", det="nrcblong"),
                 1_700_000_000)
    kept = _drop_foreign_obs_duplicates([old, new], "", "f360m", "m2",
                                        "nrcblong", "cloudef_controlfield")
    assert kept == [new]


def test_equal_mtimes_keep_the_previous_tokened_preference(tmp_path):
    """When the file system cannot separate the two, nothing is picked
    arbitrarily: the pre-#391 behaviour stands."""
    tok = _write(tmp_path, _name(filt="f360m", det="nrcblong", tok="_o005"),
                 1_700_000_000)
    unt = _write(tmp_path, _name(filt="f360m", det="nrcblong"),
                 1_700_000_000)
    kept = _drop_foreign_obs_duplicates([tok, unt], "", "f360m", "m2",
                                        "nrcblong", "cloudef_controlfield")
    assert kept == [tok]


def test_exactly_one_copy_survives_whichever_wins(tmp_path):
    """Whatever the preference, the exposure must be counted once -- keeping
    both is the DuplicateExposureError of #259."""
    for older_is_tokened in (True, False):
        d = tmp_path / f"case{int(older_is_tokened)}"
        d.mkdir()
        t = _write(d, _name(filt="f360m", det="nrcblong", tok="_o005"),
                   1_700_000_000 if older_is_tokened else 1_700_086_400)
        u = _write(d, _name(filt="f360m", det="nrcblong"),
                   1_700_086_400 if older_is_tokened else 1_700_000_000)
        kept = _drop_foreign_obs_duplicates([t, u], "", "f360m", "m2",
                                            "nrcblong", "cloudef_controlfield")
        assert len(kept) == 1
