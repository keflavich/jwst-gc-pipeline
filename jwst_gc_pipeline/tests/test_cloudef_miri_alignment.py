"""cloudef's MIRI observations had no alignment_config entry (2026-09-11).

The MIRI half of 2092 (obs 004/006/008, F770W + F2100W) had never been
downloaded, so nothing was registered for it.  The day it was first reduced, m2
measured six real corrections on F770W and refused them:

    RuntimeError: astrom checkpoint [m2] F770W/mirimage: measured 6 real
    correction(s) for proposal 2092 observation 006, but alignment_config
    declares NO table-driven correction channel for this field

That is the FIELD-shaped refusal -- an operator can fix it by adding an entry --
and is distinct from the instrument-shaped one #832 removed for MIRI.
"""
import pytest
from jwst_gc_pipeline.reduction import alignment_config as AC

MIRI_OBS = ("004", "006", "008")


@pytest.mark.parametrize("field", MIRI_OBS)
def test_cloudef_miri_observations_resolve(field):
    cfg = AC.resolve("2092", field)
    assert cfg is not None, (
        f"2092/{field} has no alignment entry, so m2 cannot route a measured "
        f"correction and the MIRI run stops")


@pytest.mark.parametrize("field", MIRI_OBS)
def test_cloudef_miri_gets_a_consensus_write_channel(field):
    assert AC.offsets_channel("2092", field, instrument="miri") == AC.CHANNEL_CONSENSUS
    assert AC.offsets_table_path("/bp", "2092", field, instrument="miri").endswith(
        "offsets/Offsets_JWST_Brick2092_consensus.csv")


@pytest.mark.parametrize("field", MIRI_OBS)
def test_cloudef_miri_shares_the_nircam_frame(field):
    """Both instruments image the same sky; differing frames would split it."""
    assert AC.resolve("2092", field).reference_frame == AC.VIRAC2
    assert AC.resolve("2092", "002").reference_frame == AC.VIRAC2
    assert AC.resolve("2092", "005").reference_frame == AC.VIRAC2


@pytest.mark.parametrize("field", MIRI_OBS)
def test_cloudef_miri_anchor_is_f770w(field):
    """Only F770W and F2100W were observed; F770W is the closer to VIRAC2's Ks
    and at 21 um the field is dust-emission dominated."""
    assert AC.resolve("2092", field).reference_filter == "F770W"


def test_the_nircam_halves_are_untouched():
    """obs002 keeps its recorded bulk, obs005 its locked table."""
    assert AC.resolve("2092", "002").source == AC.RECORDED_BULK
    assert AC.resolve("2092", "005").source == AC.TABLE_LOCKED
    assert AC.offsets_channel("2092", "005") == AC.CHANNEL_LOCKED


def test_consensus_not_locked_because_the_locked_table_is_nircam_only():
    """The shared 2092 VIRAC2locked table carries no MIRI rows, so there is
    nothing for these observations to lock to."""
    assert AC.resolve("2092", "004").source == AC.TABLE_CONSENSUS
