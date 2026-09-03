"""sickle's MIRI observations are registered; obs 003 stays brick's.

sickle's only 3958 entry covered NIRCam obs 007, so its MIRI run had no entry at
all and stopped with:

    astrom checkpoint [m2] F770W/mirimage: measured 6 real correction(s) for
    proposal 3958 observation 001-002, but alignment_config declares NO
    table-driven correction channel for this field

What the MIRI entry supplies is the FRAME (VIRAC2, shared with the NIRCam obs)
and the F770W anchor.  It does NOT supply a write channel:
``offsets_channel(..., instrument='miri')`` is ``CHANNEL_NONE`` however the entry
is declared, because ``PipelineMIRI.fix_alignment`` opens no offsets table
(``alignment_config.TABLE_DRIVEN_INSTRUMENTS``).  An above-floor MIRI correction
therefore still refuses -- naming the reducer rather than a missing entry --
while ``ASTROM_CHECKPOINT_APPLY=1`` stale-tags the mosaics it measured and
``ASTROM_CHECKPOINT_WARN_ONLY=1`` demotes the stop.
"""
import pytest
import yaml
from pathlib import Path

from jwst_gc_pipeline.reduction import alignment_config as AC

MIRI_FIELDS = ("001-002", "001", "002")
FIELDS_YAML = Path(AC.__file__).resolve().parents[1] / "fields.yaml"


@pytest.mark.parametrize("field", MIRI_FIELDS)
def test_miri_field_resolves(field):
    assert AC.resolve("3958", field) is not None, (
        f"3958/{field} has no alignment entry, so the m2 checkpoint cannot route "
        f"a measured correction and the MIRI run stops"
    )


@pytest.mark.parametrize("field", MIRI_FIELDS)
def test_the_declared_source_is_consensus(field):
    """The authored VIRAC2locked table is NIRCam-only, so nothing can be LOCKED."""
    assert AC.resolve("3958", field).source == AC.TABLE_CONSENSUS
    assert AC.offsets_channel("3958", field) == "consensus"


@pytest.mark.parametrize("field", MIRI_FIELDS)
def test_but_miri_itself_still_gets_no_write_channel(field):
    """Which is the point of the entry: frame and anchor, not a table.

    ``PipelineMIRI.fix_alignment`` opens no offsets table, so a correction
    written into one on MIRI's behalf reaches no frame and the next re-tie
    re-measures the identical residual.
    """
    assert AC.offsets_channel("3958", field,
                              instrument="miri") == AC.CHANNEL_NONE
    assert AC.offsets_table_path("/bp", "3958", field, instrument="miri") == ""


@pytest.mark.parametrize("field", MIRI_FIELDS)
def test_miri_shares_the_nircam_frame(field):
    """Both instruments observe the same sky; differing frames would split it."""
    assert AC.resolve("3958", field).reference_frame == AC.VIRAC2
    assert AC.resolve("3958", "007").reference_frame == AC.VIRAC2


@pytest.mark.parametrize("field", MIRI_FIELDS)
def test_miri_anchor_is_f770w_and_differs_from_nircam(field):
    """The anchor is per OBSERVATION; the two tokens have disjoint bands.

    `promote_reference_filter` resolves the anchor's consensus under the
    observation's token, and F210M is not observed in 001/002 -- so
    `f210m_o001-002_consensus.fits` can never exist and an F210M anchor here
    would raise on a checkpoint that cannot run.
    """
    assert AC.resolve("3958", field).reference_filter == "F770W"
    assert AC.resolve("3958", "007").reference_filter == "F210M"


def test_nircam_entry_is_untouched():
    """007 keeps its locked, per-exposure VIRAC2 table and its own anchor."""
    cfg = AC.resolve("3958", "007")
    assert cfg.source == AC.TABLE_LOCKED
    assert cfg.reference_filter == "F210M"


def test_obs_003_is_not_claimed_by_sickle():
    """3958/003 is registered to BRICK and sits 394" away -- different sky.

    Its frames live in the sickle tree, so a too-broad entry here would quietly
    pull another field's deliverable into sickle's solution.
    """
    assert AC.resolve("3958", "003") is None

    doc = yaml.safe_load(FIELDS_YAML.read_text())
    brick = doc["fields"]["brick"]["observations"]["3958"]["obsids"]
    sickle = doc["fields"]["sickle"]["observations"]["3958"]["obsids"]
    assert "003" in brick.get("miri", []), "brick no longer claims 3958/003"
    assert "003" not in sickle.get("miri", []), "sickle now claims 3958/003 too"


def test_sickle_yaml_still_registers_the_joint_miri_field():
    """The joint 001-002 spelling is what the cataloging run passes as --field."""
    doc = yaml.safe_load(FIELDS_YAML.read_text())
    obs = doc["fields"]["sickle"]["observations"]["3958"]
    assert "001-002" in (obs.get("joint_obsids") or {}).get("miri", [])


# --------------------------------------------------------------------------
# The anchor is per OBSERVATION, not per field.
# --------------------------------------------------------------------------

def test_the_two_sickle_tokens_have_disjoint_bands():
    """Which is why one anchor cannot serve both instruments.

    ``promote_reference_filter`` resolves the anchor's consensus under the
    observation's token, so an anchor must be a band that observation took.
    """
    from jwst_gc_pipeline.fields import filters_for_observation

    nircam = set(filters_for_observation("sickle", "3958", "007"))
    miri = set(filters_for_observation("sickle", "3958", "001-002"))
    assert nircam and miri
    assert not (nircam & miri), "the two tokens share a band after all"
    assert "F210M" in nircam and "F210M" not in miri
    assert "F770W" in miri and "F770W" not in nircam


@pytest.mark.parametrize("obs,expected", [
    ("007", "F210M"),        # NIRCam
    ("001-002", "F770W"),    # MIRI, the joint field
    ("001", "F770W"),
])
def test_the_formula_agrees_once_scoped_to_the_observation(obs, expected):
    """Unscoped, the ranking sees all eight bands and answers F210M for both."""
    from jwst_gc_pipeline.fields import filters_for_observation
    from jwst_gc_pipeline.photometry.consensus_catalog import reference_filter

    assert reference_filter(
        filters_for_observation("sickle", "3958", obs)).upper() == expected


def test_scoping_declines_rather_than_guesses_when_ambiguous():
    """sgrb2 registers obs 001 under BOTH nircam and miri.

    Returning one instrument's bands there would be a guess, so the helper
    returns nothing and its callers fall back to the union they used before.
    """
    from jwst_gc_pipeline.fields import filters_for_observation
    assert filters_for_observation("sgrb2", "5365", "001") == []


def test_scoping_picks_the_field_that_registers_the_observation():
    """3958 appears under sickle AND brick; 003 is brick's MIRI, not sickle's."""
    from jwst_gc_pipeline.fields import filters_for_observation
    assert filters_for_observation(None, "3958", "001-002") == ["F1130W", "F1500W", "F770W"]
    assert filters_for_observation(None, "3958", "007")
    assert AC.resolve("3958", "003") is None
