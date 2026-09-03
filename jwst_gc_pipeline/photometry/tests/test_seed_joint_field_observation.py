"""A joint-field run seeds rows keyed to each frame's real observation.

`seed_offsets_table_from_consensus` built its visit token by interpolating the
run's `--field`.  For a JOINT run that is several observations at once, so
sickle MIRI's `--field 001-002` produced `jw03958001-002001` -- a token no frame
can equal.  `assert_visit_token` refuses it, which is right: every
`lookup_consensus_offset` would return (0, 0) and the re-tie loop would
re-measure the same residual forever while reporting it had written corrections.

The observation is recoverable though.  Each correction's Vgroup is
`{obs:03d}{vgroup:05d}`, so the frames' real observation is carried per
correction -- which is what that guard's own message asks for.

Measured on sickle: corrections carry 00102101 and 00203101; the frames are
jw03958001001_02101 and jw03958002001_03101.
"""
import pytest

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    _bulk_vgroups, _observation_for_correction, assert_visit_token,
    OffsetsTableUpdateError)
from jwst_gc_pipeline.mast_names import jw_prefix


@pytest.mark.parametrize("vgroup,expected", [
    ("00102101", "001"),   # sickle MIRI obs 001
    ("00203101", "002"),   # sickle MIRI obs 002
])
def test_a_joint_field_takes_the_observation_from_the_correction(vgroup, expected):
    assert _observation_for_correction("001-002", {"vgroup": vgroup}) == expected


@pytest.mark.parametrize("field", ["001", "002", "007"])
def test_a_single_observation_run_is_unchanged(field):
    """The overwhelmingly common path must not change behaviour."""
    assert _observation_for_correction(field, {"vgroup": "00203101"}) == field


def test_a_joint_field_without_a_usable_vgroup_still_refuses():
    """Do not guess.  Falling back to the field keeps the guard's refusal.

    A silent wrong observation would be worse than the malformed token: the row
    would key to a real frame that is not this one.
    """
    for corr in ({}, {"vgroup": ""}, {"vgroup": "abc"}, {"vgroup": None}):
        assert _observation_for_correction("001-002", corr) == "001-002"


def test_the_resulting_tokens_are_accepted_and_the_joint_one_is_not():
    """End to end: the fix must produce tokens assert_visit_token accepts."""
    for vgroup, obs in (("00102101", "001"), ("00203101", "002")):
        tok = f"{jw_prefix('3958')}{_observation_for_correction('001-002', {'vgroup': vgroup})}001"
        assert tok == f"jw03958{obs}001"
        assert assert_visit_token(tok, "test") == tok

    with pytest.raises(OffsetsTableUpdateError):
        assert_visit_token("jw03958001-002001", "test")


def test_both_observations_of_one_joint_run_get_distinct_tokens():
    """The point of the fix: one run, two correctly-separated observations."""
    toks = {
        f"{jw_prefix('3958')}{_observation_for_correction('001-002', {'vgroup': vg})}001"
        for vg in ("00102101", "00203101")
    }
    assert toks == {"jw03958001001", "jw03958002001"}


# ---------------------------------------------------------------------------
# The BULK consensus->reference row (the half #640 left open).
#
# #640 gave the PER-EXPOSURE corrections their Vgroup, but the per-visit bulk
# tie carries exposure=None AND module=None and carried no Vgroup either, so
# `_observation_for_correction` still fell back to the field and the joint run
# still died on `jw03958001-002001` -- the same error, one code path over.
# ---------------------------------------------------------------------------

def _exposures(*vgroups):
    """Consensus exposure records, keyed as build_visit_consensus keys them."""
    return [dict(key=("1", i + 1, "mirimage", "F770W", vg))
            for i, vg in enumerate(vgroups)]


def test_a_joint_consensus_yields_one_vgroup_per_observation():
    """sickle MIRI: obs 001 and 002 in one consensus -> one bulk row each."""
    assert _bulk_vgroups(_exposures("00102101", "00102101",
                                    "00203101", "00203101")) == \
        ["00102101", "00203101"]


def test_a_single_observation_consensus_yields_exactly_one_bulk_row():
    """The common path keeps emitting the single bulk row it always did."""
    assert _bulk_vgroups(_exposures("00102101", "00102101")) == ["00102101"]


def test_bulk_vgroups_does_not_invent_an_observation():
    """No parseable Vgroup -> None, so the caller reproduces the old refusal
    rather than guessing an observation the frames may not have."""
    assert _bulk_vgroups([dict(key=("1", 1, "mirimage", "F770W"))]) == [None]
    assert _bulk_vgroups(_exposures("", "abc")) == [None]
    assert _bulk_vgroups([]) == [None]


def test_the_bulk_rows_of_a_joint_run_resolve_to_accepted_tokens():
    """End to end for the bulk path: every emitted row keys to a real frame."""
    exposures = _exposures("00102101", "00203101")
    toks = set()
    for vg in _bulk_vgroups(exposures):
        corr = dict(visit="1", exposure=None, module=None,
                    filtername="F770W", vgroup=vg)
        obs = _observation_for_correction("001-002", corr)
        tok = f"{jw_prefix('3958')}{obs}001"
        assert assert_visit_token(tok, "test") == tok
        toks.add(tok)
    assert toks == {"jw03958001001", "jw03958002001"}


def test_the_bulk_row_of_a_joint_run_used_to_be_refused():
    """Pin the defect: a bulk row with no Vgroup is exactly what raised."""
    corr = dict(visit="1", exposure=None, module=None, filtername="F770W")
    obs = _observation_for_correction("001-002", corr)
    assert obs == "001-002"
    with pytest.raises(OffsetsTableUpdateError):
        assert_visit_token(f"{jw_prefix('3958')}{obs}001", "test")
