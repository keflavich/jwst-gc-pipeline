"""One observation's m2 correction must not stale-tag another's mosaics.

`find_i2d_for_filter` globs on the FILTER alone
(`photometry/astrometry_checkpoint.py`), and every observation of a proposal
writes its stage-3 mosaics into the SAME `<FILTER>/pipeline` directory.  So the
lookup returns every observation's mosaics, and `mark_i2d_stale` -- the caller
at `photometry/cataloging.py`, in the m2 correction branch -- renames all of
them to `*_im0_badastrom.fits`.  A neighbour's good mosaic is then quarantined
by a correction it had nothing to do with, and the release gate refuses it.

That is live on disk today (sickle/F1130W/pipeline returns a joint o001-002
mosaic beside o001's and o003's) and gets much worse for 10678, the GC
Treasury: all 139 observations share one tree, so a correction on tile 088
would stale-tag 138 innocent tiles.

The second half of this file covers the INVERSE failure the scope introduces:
a scope spelling the products do not use matches nothing, `mark_i2d_stale([])`
renames nothing, and the run continues with a corrected offsets table and
un-quarantined stale mosaics.  That is the quieter of the two failures, so it
raises.
"""
import os

import pytest

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    ObservationScopeError, ObservationScopeMatchedNothingError,
    find_i2d_for_filter, mark_i2d_stale, observation_ids, observation_scope)


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
# a scope the parser cannot read matches nothing, which tags nothing
# --------------------------------------------------------------------------
#
# Converting loud over-tagging into silent zero-tagging is the worse of the
# two failures: an over-tag is visible (the release gate refuses a field that
# was quarantined), a zero-tag is not (the gate passes a field whose mosaics
# are stale behind a corrected offsets table).  Every spelling the parser
# could not read degraded that way, measured on the pre-floor branch against
# the real trees.


@pytest.fixture
def shared_tree(tmp_path):
    """ngc6334: proposals 6778 and 7213 in ONE tree, both calling it obs 001.

    `naming.SHARED_TREE_PROPOSALS`.  36 + 36 mosaics sit in
    /orange/adamginsburg/jwst/ngc6334/F200W/pipeline today, and the m2 records
    beside them are tagged `_j6778` / `_j7213` -- so `_j{proposal}` is a
    REQUIRED value of `apply_m2_checkpoint_corrections --obs-token` there
    (`load_corrections` refuses to run in that directory without one).
    """
    filt = "F200W"
    pipeline = tmp_path / filt / "pipeline"
    pipeline.mkdir(parents=True)
    names = {}
    for prop in ("06778", "07213"):
        stem = f"jw{prop}-o001_t001_nircam_clear-{filt.lower()}-merged"
        names[prop] = [f"{stem}_i2d.fits", f"{stem}_data_i2d.fits"]
        for name in names[prop]:
            (pipeline / name).write_bytes(b"")
    return tmp_path, filt, names


@pytest.mark.parametrize("proposal,other", [("6778", "07213"),
                                            ("7213", "06778")])
def test_shared_tree_proposal_token_scopes_by_proposal(shared_tree, proposal,
                                                       other):
    """THE B1 DEFECT.  `_j6778` read as an obsid matched zero mosaics.

    Measured on the pre-floor branch: `observation_ids('_j6778')` ->
    frozenset({'j6778'}), which no basename can carry, so summing
    find_i2d_for_filter over ngc6334's 11 populated filters went 438 -> 0 and
    `mark_i2d_stale([])` renamed nothing and wrote no ledger line.
    """
    root, filt, names = shared_tree
    found = {os.path.basename(p) for p in
             find_i2d_for_filter(str(root), filt,
                                 observation=f"_j{proposal}")}
    assert found == set(names[f"0{proposal}"])
    assert not any(f"jw{other}" in name for name in found)


def test_shared_tree_proposal_token_names_no_observation(shared_tree):
    """It is a PROPOSAL scope: the obsid half stays unconstrained."""
    scope = observation_scope("_j6778")
    assert scope.obsids is None
    assert scope.proposals == frozenset({"06778"})
    assert not scope.unscoped


def test_obsid_scope_alone_cannot_separate_two_proposals(shared_tree):
    """Documented residual: both proposals call it observation 001.

    `_resolved_obsid` yields the obsid only, so the cataloging call site is
    still proposal-blind on a shared tree; separating it needs the proposal in
    the token, which is what `_j{proposal}` supplies.
    """
    root, filt, names = shared_tree
    found = {os.path.basename(p) for p in
             find_i2d_for_filter(str(root), filt, observation="001")}
    assert found == set(names["06778"]) | set(names["07213"])


def test_both_halves_of_a_module_token_are_read(shared_tree):
    """`naming.merged_catalog_module_token` concatenates `_j{prop}_o{obs}`."""
    root, filt, names = shared_tree
    found = {os.path.basename(p) for p in
             find_i2d_for_filter(str(root), filt, observation="_j7213_o001")}
    assert found == set(names["07213"])


@pytest.mark.parametrize("spelling", ["88", "088", "o88", "_o088"])
def test_unpadded_obsid_normalises_like_the_filenames(two_observation_tree,
                                                      spelling):
    """`--obsid`/`--field` are free-form; the mosaics are always three-digit.

    Measured on the pre-floor branch: `--obsid 3` took m4 F322W2 from 52
    mosaics to 0, while `--obsid 003` gave 16.  `naming
    .observation_field_token` is the single source for the padding rule and is
    now delegated to rather than re-derived (issue #316).
    """
    found = {os.path.basename(p)
             for p in find_i2d_for_filter(str(two_observation_tree), FILTERNAME,
                                          observation=spelling)}
    assert found == set(_mosaic_names("088"))


def test_a_scope_that_matches_nothing_raises(two_observation_tree):
    """THE B2 FLOOR.  A wrong-but-shaped obsid tags zero mosaics silently.

    `_resolved_obsid` validates `--field` on SHAPE alone for a wildcard-obsid
    proposal (10678), so a three-digit typo passes validation and then matches
    no mosaic.  Measured on the pre-floor branch: brick F200W 60 -> 0 with
    observation='012'.
    """
    with pytest.raises(ObservationScopeMatchedNothingError) as exc:
        find_i2d_for_filter(str(two_observation_tree), FILTERNAME,
                            observation="012")
    msg = str(exc.value)
    assert "088" in msg and "089" in msg, "the message must name what IS there"


def test_a_scope_that_matches_nothing_raises_for_a_wrong_proposal(shared_tree):
    root, filt, names = shared_tree
    with pytest.raises(ObservationScopeMatchedNothingError):
        find_i2d_for_filter(str(root), filt, observation="_j9999")


def test_the_floor_does_not_fire_on_an_empty_directory(tmp_path):
    """Nothing to tag is not the same as a scope that cannot match."""
    (tmp_path / FILTERNAME / "pipeline").mkdir(parents=True)
    assert find_i2d_for_filter(str(tmp_path), FILTERNAME,
                               observation="088") == []


def test_an_already_quarantined_observation_returns_empty_quietly(tmp_path):
    """A second correction pass has nothing left to rename.

    The scope's own mosaics are on disk as `*_i2d_im0_badastrom.fits`, which
    the `*_i2d.fits` globs cannot see; the observation is real and present, so
    this is a no-op rather than a mis-spelled scope.
    """
    pipeline = tmp_path / FILTERNAME / "pipeline"
    pipeline.mkdir(parents=True)
    for name in _mosaic_names("089"):
        (pipeline / name).write_bytes(b"")
    for name in _mosaic_names("088"):
        (pipeline / (name[:-len(".fits")] + "_im0_badastrom.fits")
         ).write_bytes(b"")

    assert find_i2d_for_filter(str(tmp_path), FILTERNAME,
                               observation="088") == []


def test_an_observation_present_only_as_crf_returns_empty_quietly(tmp_path):
    """A proposal that does not image this filter is not a typo.

    ngc6334 F162M holds 3754 jw07213 products and no jw06778 product at all.
    """
    pipeline = tmp_path / FILTERNAME / "pipeline"
    pipeline.mkdir(parents=True)
    for name in _mosaic_names("089"):
        (pipeline / name).write_bytes(b"")
    (pipeline / f"jw10678-o088_t001_nircam_{FILTERNAME.lower()}_crf.fits"
     ).write_bytes(b"")

    assert find_i2d_for_filter(str(tmp_path), FILTERNAME,
                               observation="088") == []


def test_an_unreadable_token_raises_rather_than_scoping_to_nothing():
    """A token that names neither an observation nor a proposal stops here."""
    with pytest.raises(ObservationScopeError):
        observation_scope("nrcalong")


def test_extra_globs_survive_a_scope_that_matches_no_mosaic(tmp_path):
    """The floor is about the FILTER globs; extra_globs are literal.

    They are unscoped by contract, so they cannot be zeroed by the scope and
    the floor has nothing to protect there.
    """
    pipeline = tmp_path / FILTERNAME / "pipeline"
    pipeline.mkdir(parents=True)
    for name in _mosaic_names("089"):
        (pipeline / name).write_bytes(b"")
    (pipeline / f"jw10678-o088_t001_nircam_{FILTERNAME.lower()}_crf.fits"
     ).write_bytes(b"")
    extra = str(pipeline / f"*-o089_*-{FILTERNAME.lower()}-*_i2d.fits")

    found = {os.path.basename(p) for p in
             find_i2d_for_filter(str(tmp_path), FILTERNAME,
                                 extra_globs=(extra,), observation="088")}
    assert found == set(_mosaic_names("089"))


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
