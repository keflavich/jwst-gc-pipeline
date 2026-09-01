"""sickle's MIRI observations have a correction channel; obs 003 stays brick's.

The m2 checkpoint refuses to write a correction for a field with no
table-driven channel, because the numbers would land in a table
``fix_alignment`` never reads and the next re-tie would re-measure the identical
residual (the arches/sgrb2 failure).  sickle's only 3958 entry covered NIRCam
obs 007, so its MIRI run stopped with:

    astrom checkpoint [m2] F770W/mirimage: measured 6 real correction(s) for
    proposal 3958 observation 001-002, but alignment_config declares NO
    table-driven correction channel for this field
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
def test_miri_channel_is_consensus(field):
    """The authored VIRAC2locked table is NIRCam-only, so nothing can be LOCKED."""
    assert AC.resolve("3958", field).source == AC.TABLE_CONSENSUS
    assert AC.offsets_channel("3958", field) == "consensus"


@pytest.mark.parametrize("field", MIRI_FIELDS)
def test_miri_shares_the_nircam_frame(field):
    """Both instruments observe the same sky; differing frames would split it."""
    assert AC.resolve("3958", field).reference_frame == AC.VIRAC2
    assert AC.resolve("3958", "007").reference_frame == AC.VIRAC2


@pytest.mark.parametrize("field", MIRI_FIELDS)
def test_miri_anchor_is_f770w(field):
    """The only MIRI band present in every observation."""
    assert AC.resolve("3958", field).reference_filter == "F770W"


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
