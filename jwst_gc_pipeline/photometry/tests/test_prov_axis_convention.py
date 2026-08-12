"""Right ascension has two quantities, and a provenance column must say which.

Right ascension is a coordinate like longitude: away from the equator, one
degree of it covers less sky.  So "the source moved 100 mas east" (an ON-SKY
separation) and "its right-ascension number changed by 100 mas" (a COORDINATE
offset) are different, related by cos(declination) -- about 14% at Galactic
Centre declinations.

The per-exposure offsets table stores the coordinate version in its ``dra``
columns.  The provenance record of what a checkpoint added stores the on-sky
version.  Neither name said so, and the validator that checks "the recorded
change matches the change actually made" could therefore only be enforced on
declination: on right ascension the exact factor was unknown, so it allowed
anything inside a 14% window -- wide enough for a 14% corruption to pass.
"""
import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    PROV_DEC_DEG_KEY, PROV_ONSKY_DEC_KEY, PROV_ONSKY_RA_KEY,
    OffsetsTableUpdateError, migrate_prov_column_names, prov_onsky_columns,
    update_offsets_table)

GC_DEC = -28.7          # a Galactic Centre declination; cos -> ~0.877


def _table(path, dec=GC_DEC, legacy=False, n=2):
    """A minimal offsets table carrying both column conventions."""
    t = Table()
    t["Filter"] = ["F212N"] * n
    t["Module"] = ["nrcb1"] * n
    t["Visit"] = ["jw01182004001"] * n
    t["Exposure"] = list(range(1, n + 1))
    for col in ("dra", "ddec", "dra (arcsec)", "ddec (arcsec)"):
        t[col] = np.zeros(n)
    if legacy:
        t["prov_dra_added_mas"] = np.zeros(n)
        t["prov_ddec_added_mas"] = np.zeros(n)
    t.write(path, overwrite=True)
    return path


def _correction(dra_onsky=100.0, ddec_onsky=100.0, dec=GC_DEC, exposure=1):
    return dict(visit="jw01182004001", exposure=exposure, module="nrcb1",
                filtername="F212N", dra_onsky_mas=dra_onsky,
                ddec_onsky_mas=ddec_onsky, dec_deg=dec)


# ---------------------------------------------------------------------------
# The names say which quantity they hold
# ---------------------------------------------------------------------------

def test_the_provenance_column_names_state_the_convention():
    """`prov_dra_added_mas` said neither on-sky nor coordinate.  Three
    mis-diagnoses of one issue turned on not knowing which it was."""
    assert PROV_ONSKY_RA_KEY == "prov_dra_onsky_mas"
    assert PROV_ONSKY_DEC_KEY == "prov_ddec_onsky_mas"


def test_a_legacy_table_is_renamed_without_its_values_changing(tmp_path):
    """The values were always on-sky milliarcseconds; only the name was
    silent.  A rename must not be allowed to look like a correction."""
    t = Table()
    t["prov_dra_added_mas"] = [12.5, -3.0]
    t["prov_ddec_added_mas"] = [7.5, 1.0]
    renamed = migrate_prov_column_names(t)
    assert renamed == {"prov_dra_added_mas": PROV_ONSKY_RA_KEY,
                       "prov_ddec_added_mas": PROV_ONSKY_DEC_KEY}
    assert list(t[PROV_ONSKY_RA_KEY]) == [12.5, -3.0]
    assert list(t[PROV_ONSKY_DEC_KEY]) == [7.5, 1.0]


def test_a_table_carrying_both_spellings_is_left_alone(tmp_path):
    """Merging two columns that both claim to be the record is a curation
    decision, not a rename, so this refuses to guess."""
    t = Table()
    t["prov_dra_added_mas"] = [1.0]
    t[PROV_ONSKY_RA_KEY] = [2.0]
    assert migrate_prov_column_names(t) == {}
    assert list(t["prov_dra_added_mas"]) == [1.0]
    assert list(t[PROV_ONSKY_RA_KEY]) == [2.0]


def test_either_spelling_can_be_read(tmp_path):
    legacy = Table({"prov_dra_added_mas": [0.0], "prov_ddec_added_mas": [0.0]})
    current = Table({PROV_ONSKY_RA_KEY: [0.0], PROV_ONSKY_DEC_KEY: [0.0]})
    assert prov_onsky_columns(legacy) == ("prov_dra_added_mas",
                                          "prov_ddec_added_mas")
    assert prov_onsky_columns(current) == (PROV_ONSKY_RA_KEY,
                                           PROV_ONSKY_DEC_KEY)


# ---------------------------------------------------------------------------
# The declination is recorded, so the two can be reconciled exactly
# ---------------------------------------------------------------------------

def test_the_declination_the_conversion_used_is_recorded(tmp_path):
    """Without it, the coordinate offset a provenance entry implies is only
    bounded to a 14% window."""
    path = _table(str(tmp_path / "off.csv"))
    out = update_offsets_table(path, [_correction()], "m2")
    assert out[PROV_DEC_DEG_KEY][0] == pytest.approx(GC_DEC)


def test_the_recorded_declination_reconciles_the_two_conventions(tmp_path):
    """This is the whole point: on-sky / cos(dec) must equal the coordinate
    change actually written, exactly rather than within 14%."""
    path = _table(str(tmp_path / "off.csv"))
    out = update_offsets_table(path, [_correction(dra_onsky=100.0)], "m2")
    row = out[0]
    on_sky_mas = float(row[PROV_ONSKY_RA_KEY])
    coordinate_arcsec = float(row["dra (arcsec)"])
    cosd = np.cos(np.radians(float(row[PROV_DEC_DEG_KEY])))
    assert coordinate_arcsec == pytest.approx(on_sky_mas / 1000.0 / cosd,
                                              abs=1e-9)
    # and the two really do differ -- otherwise this proves nothing
    assert abs(coordinate_arcsec - on_sky_mas / 1000.0) > 0.01


def test_declination_needs_no_conversion_and_gets_none(tmp_path):
    """Only right ascension has the two-quantity problem."""
    path = _table(str(tmp_path / "off.csv"))
    out = update_offsets_table(path, [_correction(ddec_onsky=100.0)], "m2")
    assert float(out["ddec (arcsec)"][0]) == pytest.approx(0.1, abs=1e-12)


def test_a_row_never_corrected_records_no_declination(tmp_path):
    """NaN rather than 0.0: filling with zero would claim every untouched row
    was corrected on the celestial equator."""
    path = _table(str(tmp_path / "off.csv"), n=2)
    out = update_offsets_table(path, [_correction(exposure=1)], "m2")
    assert np.isfinite(out[PROV_DEC_DEG_KEY][0])
    assert not np.isfinite(out[PROV_DEC_DEG_KEY][1])


def test_a_legacy_table_is_migrated_on_the_next_write(tmp_path):
    path = _table(str(tmp_path / "off.csv"), legacy=True)
    out = update_offsets_table(path, [_correction()], "m2")
    assert PROV_ONSKY_RA_KEY in out.colnames
    assert "prov_dra_added_mas" not in out.colnames


# ---------------------------------------------------------------------------
# What the recorded declination now catches
# ---------------------------------------------------------------------------

def test_a_right_ascension_provenance_off_by_the_cosine_is_now_caught(tmp_path):
    """The corruption class this exists for: the provenance says one thing and
    the column moved by the other convention's amount.  Before the declination
    was recorded this sat inside the allowed 14% window and passed."""
    path = _table(str(tmp_path / "off.csv"))
    # 400 mas on-sky -> 456 mas of coordinate right ascension at this
    # declination.  (Kept under the 0.5" per-exposure correction ceiling.)
    out = update_offsets_table(path,
                               [_correction(dra_onsky=400.0, ddec_onsky=0.0)], "m2")
    assert float(out["dra (arcsec)"][0]) == pytest.approx(0.456, abs=0.002)

    # Hand-edit the other column of the pair so the GAP between them equals
    # the on-sky number (0.400) instead of the coordinate number (0.456) --
    # exactly the mistake of treating an on-sky separation as a coordinate
    # offset.  That gap sits at the very edge of the old +/-14% window, which
    # is why the loose check accepted it.
    out["dra"][0] = 0.456 - 0.400
    out.write(path, overwrite=True)
    # The consistency check runs on the rows a correction TOUCHES (scoped so a
    # field with one stale filter recovers filter by filter), so the next
    # correction has to land on the same row.
    with pytest.raises(OffsetsTableUpdateError, match="provenance explains"):
        update_offsets_table(
        path, [_correction(dra_onsky=1.0, ddec_onsky=0.0, exposure=1)], "m2")


def test_the_same_edit_passes_when_the_declination_was_not_recorded(tmp_path):
    """The other half of the previous test, and the reason this column exists:
    with no declination on the row, the check can only bound the coordinate
    offset to a 14% window, and an on-sky value mistaken for a coordinate one
    lands inside it."""
    path = _table(str(tmp_path / "off.csv"))
    out = update_offsets_table(path,
                               [_correction(dra_onsky=400.0, ddec_onsky=0.0)], "m2")
    out["dra"][0] = 0.456 - 0.400
    out[PROV_DEC_DEG_KEY][0] = np.nan          # as a pre-existing row would be
    out.write(path, overwrite=True)
    update_offsets_table(
        path, [_correction(dra_onsky=1.0, ddec_onsky=0.0, exposure=1)], "m2")


# ---------------------------------------------------------------------------
# What review found: an empty declination is ABSENT, not zero
# ---------------------------------------------------------------------------

def test_an_empty_declination_cell_is_read_as_absent_not_as_the_equator(tmp_path):
    """`np.asarray(col, float)` turns a masked cell into 0.0, which would be
    taken as declination zero -- cos = 1 -- and the row checked EXACTLY against
    a factor that never applied.  That inverts the check in both directions: it
    refuses a correct correction, and accepts the corruption this exists for.

    Forcing "unknown means zero degrees" must therefore fail a test, and this
    is that test.
    """
    path = _table(str(tmp_path / "off.csv"))
    out = update_offsets_table(path,
                               [_correction(dra_onsky=400.0, ddec_onsky=0.0)], "m2")
    out["dra"][0] = 0.456 - 0.456           # a CORRECT row: the exact gap
    out[PROV_DEC_DEG_KEY] = np.ma.array([np.nan] * len(out), mask=[True] * len(out))
    out.write(path, overwrite=True)
    # must NOT raise: the declination is unknown, so the loose bound applies
    update_offsets_table(
        path, [_correction(dra_onsky=1.0, ddec_onsky=0.0, exposure=1)], "m2")


def test_an_accumulated_value_survives_a_table_carrying_both_spellings():
    """Once both column names exist, astropy fills the missing side as a MASKED
    cell rather than leaving the key absent -- so `.get(new, .get(legacy))`
    returns the mask, the fallback never fires, and the accumulated history is
    replaced by zero.  Check the VALUE, not the key."""
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        _accumulated_prov)
    from astropy.table import Table as _T
    t = _T([dict(prov_dra_added_mas=-6.25),
            dict(prov_dra_onsky_mas=-1.25)])
    assert _accumulated_prov(t[0], PROV_ONSKY_RA_KEY,
                             "prov_dra_added_mas") == pytest.approx(-6.25)
    assert _accumulated_prov(t[1], PROV_ONSKY_RA_KEY,
                             "prov_dra_added_mas") == pytest.approx(-1.25)


def test_a_row_with_neither_spelling_accumulates_from_zero():
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        _accumulated_prov)
    assert _accumulated_prov({}, PROV_ONSKY_RA_KEY, "prov_dra_added_mas") == 0.0
