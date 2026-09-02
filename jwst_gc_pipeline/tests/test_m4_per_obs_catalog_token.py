"""m4's two observations must not write the same catalog filename.

JWST 1979 images M4 twice -- observation 002 and observation 003 ("M-4-shift").
Their pointings are 174" apart, wider than a NIRCam module, so they are separate
sky; but both live in the single `m4/` tree.  Without a per-observation token
both write

    f150w2_merged_indivexp_merged_m2_dao_basic.fits

and whichever finishes last silently replaces the other's catalog.  Observed on
2026-09-01 with both chains in flight: o003's m2 checkpoint reported
"excluded 91 of 96 foreign-observation per-frame catalog(s)".

Unlike brick -- two PROPOSALS over one sky -- this is one proposal over two
pointings, which is what `PER_OBS_MERGED_FIELDS` keys on `(proposal, obs)` for.
"""
import pytest

from jwst_gc_pipeline.photometry import naming


M4_OBSERVATIONS = ("002", "003")


@pytest.mark.parametrize("field", M4_OBSERVATIONS)
def test_m4_observations_are_tokened(field):
    assert naming.merged_catalog_obs_token("1979", field) == f"_o{field}"


def test_the_two_m4_observations_get_different_tokens():
    """The whole point: one filename per observation, not one for both."""
    a, b = (naming.merged_catalog_obs_token("1979", f) for f in M4_OBSERVATIONS)
    assert a != b, "both m4 observations resolve to the same catalog name"


@pytest.mark.parametrize("proposal,field", [
    ("2221", "002"),   # cloudc -- one observation in its own tree
    ("1905", "001"),   # wd1
    ("6151", "001"),   # w51
    ("5365", "001"),   # sgrb2
])
def test_single_observation_fields_stay_untokened(proposal, field):
    """Tokening a field that does not need it renames every catalog it has.

    Only a tree holding more than one observation of the same instrument needs
    this; everything else must be left alone.
    """
    assert naming.merged_catalog_obs_token(proposal, field) == ""


def test_brick_pair_is_unchanged():
    """brick's two PROPOSALS keep the tokens #597 gave them."""
    assert naming.merged_catalog_obs_token("1182", "004") == "_o004"
    assert naming.merged_catalog_obs_token("2221", "001") == "_o001"


def test_every_entry_is_a_proposal_obs_pair():
    """A bare proposal here would token every observation of it, including the
    single-observation fields that must stay untokened."""
    for entry in naming.PER_OBS_MERGED_FIELDS:
        assert isinstance(entry, tuple) and len(entry) == 2, entry
        proposal, obs = entry
        assert proposal.isdigit(), proposal
        assert obs.isdigit() and len(obs) == 3, obs
