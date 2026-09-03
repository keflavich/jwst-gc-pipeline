"""One observation's m2 correction must not stale-tag another's mosaics.

`find_i2d_for_filter` globs on the FILTER alone
(`photometry/astrometry_checkpoint.py`), and every observation of a proposal
writes its stage-3 mosaics into the SAME `<FILTER>/pipeline` directory.  So the
lookup returns every observation's mosaics, and `mark_i2d_stale` -- the caller
at `photometry/cataloging.py`, in the m2 correction branch -- renames all of
them to `*_im0_badastrom.fits`.  A neighbour's good mosaic is then quarantined
by a correction it had nothing to do with, and the release gate refuses it.

That is live on disk today (sickle/F770W/pipeline holds o001, o002 and o003
side by side) and gets much worse for 10678, the GC Treasury: all 139
observations share one tree, so a correction on tile 088 would stale-tag 138
innocent tiles.
"""
import os

import pytest

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    find_i2d_for_filter, mark_i2d_stale, observation_ids)


FILTERNAME = "F410M"


def _mosaic_names(obs):
    """The stage-3 products one observation writes for FILTERNAME."""
    stem = f"jw10678-o{obs}_t001_nircam_clear-{FILTERNAME.lower()}-merged"
    return [f"{stem}_i2d.fits",
            f"{stem}_data_i2d.fits",
            f"{stem}_m2_daophot_basic_mergedcat_residual_i2d.fits"]


@pytest.fixture
def two_observation_tree(tmp_path):
    """A single field tree holding tile 088's and tile 089's mosaics."""
    pipeline = tmp_path / FILTERNAME / "pipeline"
    pipeline.mkdir(parents=True)
    for obs in ("088", "089"):
        for name in _mosaic_names(obs):
            (pipeline / name).write_bytes(b"")
    return tmp_path


def test_unscoped_lookup_still_returns_every_observation(two_observation_tree):
    """The default is unchanged -- that is what a one-observation tree needs."""
    found = {os.path.basename(p)
             for p in find_i2d_for_filter(str(two_observation_tree), FILTERNAME)}
    assert found == set(_mosaic_names("088")) | set(_mosaic_names("089"))


def test_scoped_lookup_excludes_the_other_observation(two_observation_tree):
    """THE DEFECT.  Fails on the unscoped glob: it returns 089's mosaics too."""
    found = {os.path.basename(p)
             for p in find_i2d_for_filter(str(two_observation_tree), FILTERNAME,
                                          observation="088")}
    assert found == set(_mosaic_names("088"))
    assert not any("-o089_" in name for name in found)


def test_stale_tag_leaves_the_neighbour_tile_alone(tmp_path):
    """End to end: tile 088 corrects, tile 089's mosaics stay on disk.

    This is the consequence the release gate sees -- a `*_im0_badastrom.fits`
    rename is the quarantine it refuses on.
    """
    pipeline = tmp_path / FILTERNAME / "pipeline"
    pipeline.mkdir(parents=True)
    for obs in ("088", "089"):
        for name in _mosaic_names(obs):
            (pipeline / name).write_bytes(b"")

    renames = mark_i2d_stale(
        find_i2d_for_filter(str(tmp_path), FILTERNAME, observation="088"),
        reason="m2 checkpoint corrected the offsets table",
        record_dir=str(tmp_path / "astrometry_checkpoints"))

    assert len(renames) == len(_mosaic_names("088"))
    for name in _mosaic_names("089"):
        assert (pipeline / name).exists(), f"{name} was quarantined by tile 088"
    for name in _mosaic_names("088"):
        assert not (pipeline / name).exists()


@pytest.fixture
def joint_association_tree(tmp_path):
    """sickle/F770W as it sits on disk: a joint o001-002 mosaic beside o003."""
    filt = "F770W"
    pipeline = tmp_path / filt / "pipeline"
    pipeline.mkdir(parents=True)
    joint = f"jw03958-o001-002_t001_miri_clear-{filt.lower()}-mirimage_data_i2d.fits"
    solo = f"jw03958-o003_t001_miri_clear-{filt.lower()}-mirimage_data_i2d.fits"
    for name in (joint, solo):
        (pipeline / name).write_bytes(b"")
    return tmp_path, filt, joint, solo


@pytest.mark.parametrize("obs", ["001", "002"])
def test_joint_association_mosaic_belongs_to_each_observation(
        joint_association_tree, obs):
    """`jw03958-o001-002_...` (sickle F770W) is BOTH observations' product."""
    root, filt, joint, solo = joint_association_tree
    found = {os.path.basename(p) for p in
             find_i2d_for_filter(str(root), filt, observation=obs)}
    assert found == {joint}


def test_a_joint_observation_token_claims_both_its_parts(joint_association_tree):
    """`_resolved_obsid` returns `fields.Obs.joint_obsids` VERBATIM.

    sickle MIRI resolves to '001-002' and sgrb2 MIRI to '002-998'.  Treating
    that as one opaque id matches no mosaic, so the scoped lookup would return
    nothing and stale-tag nothing -- fail-open, the mirror image of the defect
    this PR fixes.
    """
    root, filt, joint, solo = joint_association_tree
    assert observation_ids("001-002") == frozenset({"001", "002"})
    found = {os.path.basename(p) for p in
             find_i2d_for_filter(str(root), filt, observation="001-002")}
    assert found == {joint}


def test_other_observation_still_excluded_in_a_joint_tree(joint_association_tree):
    root, filt, joint, solo = joint_association_tree
    found = {os.path.basename(p) for p in
             find_i2d_for_filter(str(root), filt, observation="003")}
    assert found == {solo}


def test_untokened_match_is_kept_rather_than_silently_dropped(tmp_path):
    """A file we cannot attribute stays in the set: dropping it fails OPEN."""
    pipeline = tmp_path / FILTERNAME / "pipeline"
    pipeline.mkdir(parents=True)
    odd = f"handmade-{FILTERNAME.lower()}-mosaic_i2d.fits"
    (pipeline / odd).write_bytes(b"")
    for name in _mosaic_names("088"):
        (pipeline / name).write_bytes(b"")

    found = {os.path.basename(p) for p in
             find_i2d_for_filter(str(tmp_path), FILTERNAME, observation="089")}
    assert found == {odd}


def test_extra_globs_are_not_scoped(tmp_path):
    """A literal pattern the caller named by hand is honoured verbatim."""
    pipeline = tmp_path / FILTERNAME / "pipeline"
    pipeline.mkdir(parents=True)
    for obs in ("088", "089"):
        for name in _mosaic_names(obs):
            (pipeline / name).write_bytes(b"")

    extra = str(pipeline / f"*-o089_*-{FILTERNAME.lower()}-*_i2d.fits")
    found = {os.path.basename(p) for p in
             find_i2d_for_filter(str(tmp_path), FILTERNAME,
                                 extra_globs=(extra,), observation="088")}
    assert found == set(_mosaic_names("088")) | set(_mosaic_names("089"))


@pytest.mark.parametrize("spelling", ["088", "o088", "_o088"])
def test_token_spellings_agree(two_observation_tree, spelling):
    """--field gives '088', apply_m2_checkpoint_corrections --obs-token '_o088'."""
    found = {os.path.basename(p)
             for p in find_i2d_for_filter(str(two_observation_tree), FILTERNAME,
                                          observation=spelling)}
    assert found == set(_mosaic_names("088"))


@pytest.mark.parametrize("value", [None, "", "*", "_o*", "o*"])
def test_unknown_observation_means_unscoped(two_observation_tree, value):
    """The wildcard obsid is a registry token, not an observation number.

    10678 is registered with '*'; resolving that to a literal would match no
    mosaic and stale-tag NOTHING, which is the fail-open half of this defect.
    """
    assert observation_ids(value) is None
    found = find_i2d_for_filter(str(two_observation_tree), FILTERNAME,
                                observation=value)
    assert len(found) == 6


# --------------------------------------------------------------------------
# the parameter is only worth anything if the callers pass it
# --------------------------------------------------------------------------

CALL_SITES = (
    "jwst_gc_pipeline/photometry/cataloging.py",
    "scripts/reduction/run_astrometry_checkpoint.py",
    "scripts/reduction/apply_m2_checkpoint_corrections.py",
)


@pytest.mark.parametrize("relpath", CALL_SITES)
def test_every_stale_tagging_call_site_scopes_by_observation(relpath):
    """A default-None parameter no caller passes fixes nothing.

    All three of these feed `mark_i2d_stale`, so an unscoped call renames
    another observation's mosaics.  AST rather than grep, so a reformat or a
    line break does not silently retire the check.
    """
    import ast

    root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    path = os.path.join(root, relpath)
    if not os.path.exists(path):        # installed package, no scripts/ tree
        pytest.skip(f"{relpath} not present in this checkout")
    tree = ast.parse(open(path).read(), filename=path)
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "find_i2d_for_filter"]
    assert calls, f"no find_i2d_for_filter call found in {relpath}"
    for call in calls:
        kwargs = {kw.arg for kw in call.keywords}
        assert "observation" in kwargs, (
            f"{relpath}:{call.lineno} calls find_i2d_for_filter without "
            f"observation= -- it will stale-tag every other observation's "
            f"mosaics in the shared <FILTER>/pipeline directory")
