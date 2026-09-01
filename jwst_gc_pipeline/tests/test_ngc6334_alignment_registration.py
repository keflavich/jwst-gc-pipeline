"""ngc6334's two proposals are registered, so their exposures leave the raw frame.

A proposal absent from ``ALIGNMENT_CONFIG`` is not a soft default: ``resolve()``
returns ``None`` and every exposure stays on the raw ``assign_wcs`` frame.  That
is what left sgra/1939 ~14.8" out of place, and ngc6334 was in the same state --
with the 2026-07-10 audit already measuring 61-67 mas per-filter offsets off its
channel anchors.

ngc6334 is imaged by TWO proposals over the same sky (6778 and 7213), and
``reference_frame`` is per-PROPOSAL because it names the offsets table.  So they
need two entries, not one, exactly as brick needs separate 1182 and 2221 rows.
"""
import pytest
import yaml
from pathlib import Path

from jwst_gc_pipeline.reduction import alignment_config as AC

NGC6334_PROPOSALS = ("6778", "7213")
FIELDS_YAML = Path(AC.__file__).resolve().parents[1] / "fields.yaml"


@pytest.mark.parametrize("proposal", NGC6334_PROPOSALS)
def test_proposal_resolves(proposal):
    """Unregistered would mean the raw assign_wcs frame for every exposure."""
    cfg = AC.resolve(proposal, "001")
    assert cfg is not None, (
        f"proposal {proposal} (ngc6334) is absent from ALIGNMENT_CONFIG, so every "
        f"exposure stays at the raw assign_wcs frame"
    )


@pytest.mark.parametrize("proposal", NGC6334_PROPOSALS)
def test_reference_frame_is_virac2(proposal):
    """The field's refcat is VIRAC2-dominated; Gaia alone is far too sparse."""
    assert AC.resolve(proposal, "001").reference_frame == AC.VIRAC2


@pytest.mark.parametrize("proposal", NGC6334_PROPOSALS)
def test_source_is_consensus_because_no_table_exists(proposal):
    """There is no offsets table for either proposal, so nothing can be LOCKED.

    ``TABLE_CONSENSUS`` is the self-bootstrapping mode: the m2 checkpoint writes
    the table as it measures.  ``TABLE_LOCKED`` here would point at a file that
    does not exist, and ``RECORDED_BULK`` would need a hand-measured constant
    nobody has measured.
    """
    assert AC.resolve(proposal, "001").source == AC.TABLE_CONSENSUS


@pytest.mark.parametrize("proposal", NGC6334_PROPOSALS)
def test_anchor_band_is_shared_by_both_proposals(proposal):
    """F200W is the anchor for both, so one band defines the field's frame.

    It is the only wide band the two programs have in common (the other shared
    band is the F470N narrowband).
    """
    assert AC.resolve(proposal, "001").reference_filter == "F200W"


def test_the_two_proposals_get_separate_offsets_tables():
    """The reason they cannot share one entry: the table path is per-proposal."""
    paths = {p: AC.offsets_table_path("/tmp/ngc6334", p, "001")
             for p in NGC6334_PROPOSALS}
    assert len(set(paths.values())) == 2, (
        f"both proposals resolved to the same offsets table {paths} -- one "
        f"program's solution would overwrite the other's"
    )
    for p, path in paths.items():
        assert p in str(path)


@pytest.mark.parametrize("proposal", NGC6334_PROPOSALS)
def test_channel_is_resolvable(proposal):
    """A source with no exposure axis raises at dispatch; consensus must not."""
    assert AC.offsets_channel(proposal, "001") == "consensus"


@pytest.mark.parametrize("proposal", NGC6334_PROPOSALS)
def test_fields_yaml_declares_the_same_frame(proposal):
    """fields.yaml and ALIGNMENT_CONFIG must not disagree about the frame.

    sgra and arches both carry `reference_frame` beside `reference_catalog`;
    ngc6334 carried only the catalog, which is what made the gap easy to miss.
    """
    doc = yaml.safe_load(FIELDS_YAML.read_text())
    obs = doc["fields"]["ngc6334"]["observations"][proposal]
    assert obs.get("reference_frame") == "VIRAC2"
    assert obs.get("reference_catalog"), "no reference catalog declared"


def test_unregistered_proposal_still_returns_none():
    """The guard only works while a genuine miss is still a miss."""
    assert AC.resolve("9999", "001") is None
