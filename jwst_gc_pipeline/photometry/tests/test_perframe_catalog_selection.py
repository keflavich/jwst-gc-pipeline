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

import pytest

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    AstrometryRegressionError,
)
from jwst_gc_pipeline.photometry.cataloging import (
    _drop_module_level_duplicates, _perframe_catalog_re,
    _run_astrometry_stage_checkpoint,
)


def _name(det="nrcb1", exp=1, seg="", label="m2", chunk=None, filt="f212n"):
    parts = [filt, det, "visit001", "vgroup02101", f"exp{exp:05d}"]
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
