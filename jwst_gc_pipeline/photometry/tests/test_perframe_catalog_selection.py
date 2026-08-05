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

from jwst_gc_pipeline import fields
import types

import pytest

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


def test_long_detector_does_not_supersede_its_bare_module():
    """`nrcalong` is not a numbered SW detector, so a bare `nrca` catalog has
    nothing superseding it and must be kept."""
    fns = [_name(det="nrca", filt="f405n"), _name(det="nrcalong", filt="f405n")]
    assert _drop_module_level_duplicates(fns, "f405n", "m2", "nrca") == fns


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
# AND the per-observation token.  gc2211's five observations share a directory,
# VISIT=001 and the same (vgroup, exp) tuples; ngc6334's 6778 and 7213 share a
# directory, a filter list and obsid 001; and every field carries pre-token
# copies of its own frames.  `exposure_key` carries no obs/proposal, so every
# one of those neighbours collides -- issue #259 (gc2211 F200W: 592 files ->
# 192 keys, all duplicated 2-5x).

def _mixed_gc2211():
    """One exposure of one detector, as written by three gc2211 runs plus the
    pre-token name."""
    return [_name(filt="f200w", tok=t) for t in ("_o023", "_o046", "_o050", "")]


def test_tokened_run_keeps_only_its_own_observation():
    kept = _drop_foreign_obs_duplicates(_mixed_gc2211(), "_o023",
                                        "f200w", "m2", "nrcb", "gc2211")
    assert kept == [_name(filt="f200w", tok="_o023")]


def test_untokened_run_drops_the_tokened_siblings():
    """Otherwise a run that writes pre-token names still ingests every
    observation's tokened catalogs beside it."""
    kept = _drop_foreign_obs_duplicates(_mixed_gc2211(), "",
                                        "f200w", "m2", "nrcb", "gc2211")
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
                                        "gc2211") == fns


def test_the_drop_is_reported(capsys):
    """It narrows the checkpoint's input, so it may not happen silently."""
    _drop_foreign_obs_duplicates(_mixed_gc2211(), "_o023", "f200w", "m2",
                                 "nrcb", "gc2211")
    out = capsys.readouterr().out
    assert "excluded 3 duplicate per-frame catalog(s)" in out
    assert "o046" in out and "o050" in out and "<untokened>" in out
    assert "_o023" in out


def test_checkpoint_passes_its_own_token_to_the_filter(tmp_path, capsys):
    """End-to-end wiring: the token the checkpoint names its consensus catalog
    with is the token the foreign-observation filter uses."""
    (tmp_path / "F200W").mkdir(parents=True)
    (tmp_path / "catalogs").mkdir(parents=True)
    for tok in ("_o046", "_o050"):
        (tmp_path / "F200W" / _name(filt="f200w", tok=tok)).touch()
    _run_astrometry_stage_checkpoint(
        "m2", "nrcb", "f200w", str(tmp_path), str(tmp_path), "2211",
        types.SimpleNamespace(cutout_region="", proposal_id="2211",
                              field="023", target="gc2211"), {}, context="test")
    out = capsys.readouterr().out
    assert "excluded 2 duplicate per-frame catalog(s)" in out
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
    """gc2211 images F200W in five observations that all use VISIT=001, so an
    untokened basename cannot say which one wrote it and must go."""
    fns = [_pf("F200W", "nrcb1", "o023", 1), _pf("F200W", "nrcb1", "o046", 1),
           _pf("F200W", "nrcb1", None, 1)]
    kept = _drop_foreign_obs_duplicates(
        fns, "o023", "F200W", "m2", "all", "gc2211")
    assert kept == [_pf("F200W", "nrcb1", "o023", 1)], kept


def test_shared_filter_of_the_same_field_is_decided_per_filter():
    """ngc6334 F200W IS shared (6778 and 7213 both image it -- the collision the
    `_j` token was introduced for), while F090W is not.  The rule is per
    filter, not per field."""
    assert fields.filter_observation_count("ngc6334", "F090W") == 1
    assert fields.filter_observation_count("ngc6334", "F200W") == 2
    assert fields.filter_observation_count("gc2211", "F200W") == 5
