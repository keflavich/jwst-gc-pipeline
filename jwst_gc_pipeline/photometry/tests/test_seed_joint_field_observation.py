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
    _observation_for_correction, assert_visit_token, OffsetsTableUpdateError)
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
