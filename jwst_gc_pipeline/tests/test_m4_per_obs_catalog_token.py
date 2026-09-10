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


# ---------------------------------------------------------------------------
# The negative case that carries the risk.
#
# ngc6397 is the SAME proposal -- 1979 observation 001 -- in its own tree, with
# no collision and a chain that was running while this landed.  Widening the m4
# entry from (proposal, obs) to the proposal would token ngc6397 too, and every
# merged catalog it already has would go unreachable in the module slot: the
# dead chain #597 produced for brick (#620/#625).  The sibling per-frame test
# file pins this; the merged one did not, and its negative list (2221/002,
# 1905/001, 6151/001, 5365/001) is all OTHER proposals, so none of them moves
# when 1979 is widened.
# ---------------------------------------------------------------------------

def test_ngc6397_is_untouched():
    assert naming.merged_catalog_obs_token("1979", "001") == ""
    assert naming.merged_catalog_module_token("1979", "001") == ""
    assert naming.merge_field_for_proposal("1979", "001") is None
    assert "1979" not in naming.PER_OBS_MERGED_PROPOSALS
    assert "1979" not in naming.SHARED_TREE_PROPOSALS


def test_only_the_two_m4_observations_of_1979_are_registered():
    """An exact pin on 1979's rows.

    ``test_every_entry_is_a_proposal_obs_pair`` accepts any well-formed pair, so
    it stays green if ('1979', '001') is added -- which is the one edit that
    breaks ngc6397.  The whole tuple is deliberately NOT pinned: other fields
    legitimately join it (brick already has two rows).
    """
    m4_rows = tuple(obs for prop, obs in naming.PER_OBS_MERGED_FIELDS
                    if prop == "1979")
    assert m4_rows == ("002", "003"), m4_rows


def test_the_token_is_zero_padded():
    """``--field 2`` must not write ``_o2`` while every reader globs ``_o002``.

    ``naming.OBS_TOKEN_PATTERN`` is ``o\\d{3}``; an unpadded token is skipped by
    each reader rather than raising, so the run reports "no catalogs" and moves
    on.
    """
    assert naming.merged_catalog_obs_token("1979", "2") == "_o002"
    assert naming.merged_catalog_obs_token("1979", "03") == "_o003"
    # and the unpadded spelling of the observation that must NOT be tokened
    # stays untokened rather than falling through to a token
    assert naming.merged_catalog_obs_token("1979", "1") == ""


def test_a_shared_tree_with_ONE_obsid_needs_the_j_token_not_this_mechanism():
    """omegacen is NOT fixable the way m4 is, and this pins why.

    omegacen's tree holds proposals 8322 and 12587, each with observation
    ``001``.  ``PER_OBS_MERGED_FIELDS`` disambiguates by OBSERVATION, so
    registering both would give both ``_o001`` -- the names still collide, and
    the field would read as fixed while nothing changed.  Two proposals sharing
    one tree under one obsid need ``SHARED_TREE_PROPOSALS``' ``_j{proposal}``,
    which is what ngc6334 (6778/7213) uses.
    """
    # what registering BOTH omegacen proposals in PER_OBS_MERGED_FIELDS would
    # produce: one token, built from the observation, for two proposals.
    omegacen = {("8322", "001"), ("12587", "001")}
    assert len({f"_o{naming.observation_field_token(obs)}"
                for _, obs in omegacen}) == 1, (
        "an observation token cannot separate two proposals that share an obsid")
    # what the shared-tree mechanism produces: different tokens
    assert (naming.merged_catalog_module_token("6778", "001")
            != naming.merged_catalog_module_token("7213", "001"))
    # omegacen is registered under neither today, so its two proposals are
    # still indistinguishable in a merged name (issue tracked separately)
    assert (naming.merged_catalog_module_token("8322", "001")
            == naming.merged_catalog_module_token("12587", "001") == "")
