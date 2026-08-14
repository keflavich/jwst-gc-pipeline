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


# ---------------------------------------------------------------------------
# Round 2: the fixes above had no tests, so mutating them back changed nothing
# ---------------------------------------------------------------------------

def _consensus_table(path, legacy_values=(-6.2532, 0.0, 1.0)):
    """A per-exposure corrections table in the older provenance spelling."""
    t = Table()
    t["Filter"] = ["F360M"] * 3
    t["Module"] = ["nrcb1", "nrcb2", "nrcb3"]
    t["Visit"] = ["jw02092002001"] * 3
    t["Exposure"] = [1, 2, 3]
    t["Vgroup"] = ["2101"] * 3
    t["dra (arcsec)"] = [0.0] * 3
    t["ddec (arcsec)"] = [0.0] * 3
    t["prov_dra_added_mas"] = list(legacy_values)
    t["prov_ddec_added_mas"] = [0.0] * 3
    for c in ("prov_stage", "prov_date", "prov_source"):
        t[c] = ["x"] * 3
    t.write(path, overwrite=True)
    return path


def test_the_upsert_path_never_leaves_two_columns_claiming_to_be_the_record(tmp_path):
    """Renaming only the rows a correction TOUCHES is not enough: the untouched
    rows keep the old key, and rebuilding the table from row dictionaries then
    re-creates BOTH columns, each masked where the other holds the value.

    That state is worse than the original problem -- it is permanent, because
    the migration then sees both spellings present and declines to act, every
    time, silently.  So the whole table has to be renamed when it is read.
    """
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        seed_offsets_table_from_consensus)
    path = _consensus_table(str(tmp_path / "Offsets_JWST_Brick2092_consensus.csv"))
    corr = [dict(visit="jw02092002001", exposure=2, module="nrcb2",
                 filtername="F360M", dra_onsky_mas=10.0, ddec_onsky_mas=0.0,
                 dec_deg=-28.49, vgroup="2101")]
    seed_offsets_table_from_consensus(str(tmp_path), "2092", "002", corr,
                                      stage="m2", out_path=path)
    out = Table.read(path)
    assert "prov_dra_added_mas" not in out.colnames, (
        "the legacy column survived, so the table now carries two columns each "
        "claiming to be the provenance record")
    assert PROV_ONSKY_RA_KEY in out.colnames


def test_the_upsert_path_keeps_every_row_s_accumulated_value(tmp_path):
    """The values are what the rename is for.  An untouched row must arrive in
    the new column with its history intact, not stranded in a dropped one."""
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        seed_offsets_table_from_consensus)
    path = _consensus_table(str(tmp_path / "Offsets_JWST_Brick2092_consensus.csv"))
    corr = [dict(visit="jw02092002001", exposure=2, module="nrcb2",
                 filtername="F360M", dra_onsky_mas=10.0, ddec_onsky_mas=0.0,
                 dec_deg=-28.49, vgroup="2101")]
    seed_offsets_table_from_consensus(str(tmp_path), "2092", "002", corr,
                                      stage="m2", out_path=path)
    out = Table.read(path)
    by_exposure = {int(r["Exposure"]): r for r in out}
    assert float(by_exposure[1][PROV_ONSKY_RA_KEY]) == pytest.approx(-6.2532)
    assert float(by_exposure[2][PROV_ONSKY_RA_KEY]) == pytest.approx(10.0)
    assert float(by_exposure[3][PROV_ONSKY_RA_KEY]) == pytest.approx(1.0)


def test_an_all_blank_declination_column_does_not_truncate_the_value(tmp_path):
    """A column of empty cells round-trips from CSV as an integer column, and
    writing -28.7 into it silently stores -28.  Seven tenths of a degree is
    enough to make the exact check reject a correct row on the next write."""
    path = _table(str(tmp_path / "off.csv"))
    t = Table.read(path)
    t[PROV_DEC_DEG_KEY] = np.ma.array([0] * len(t), mask=[True] * len(t))
    t.write(path, overwrite=True)
    assert Table.read(path)[PROV_DEC_DEG_KEY].dtype.kind != "f"   # the trap
    out = update_offsets_table(path, [_correction(dec=-28.7)], "m2")
    assert float(out[PROV_DEC_DEG_KEY][0]) == pytest.approx(-28.7)


# ---------------------------------------------------------------------------
# The validator's half of the change had no coverage at all
# ---------------------------------------------------------------------------

def _diverged(prov_dra_mas, gap_arcsec, dec=None):
    """One row whose two column pairs disagree by `gap`, with `prov` recorded."""
    t = Table()
    t["Filter"] = ["F212N"]; t["Module"] = ["nrcb1"]
    t["Visit"] = ["jw01182004001"]; t["Exposure"] = [1]
    t["dra (arcsec)"] = [gap_arcsec]; t["ddec (arcsec)"] = [0.0]
    t["dra"] = [0.0]; t["ddec"] = [0.0]
    t[PROV_ONSKY_RA_KEY] = [prov_dra_mas]; t[PROV_ONSKY_DEC_KEY] = [0.0]
    if dec is not None:
        t[PROV_DEC_DEG_KEY] = [dec]
    return t


def test_the_validator_uses_the_recorded_declination(tmp_path):
    """`flag_diverged_column_pairs` is the check issue #343 names.  With the
    declination recorded it must reject a gap equal to the ON-SKY number, which
    the loose bound accepts."""
    from jwst_gc_pipeline.reduction.validate_offsets_table import (
        flag_diverged_column_pairs)
    # 400 mas on-sky is 456 mas of coordinate right ascension at this declination
    assert flag_diverged_column_pairs(_diverged(400.0, 0.400, dec=GC_DEC))
    assert not flag_diverged_column_pairs(_diverged(400.0, 0.456024, dec=GC_DEC))


def test_the_validator_falls_back_to_the_loose_bound_without_a_declination():
    """A row written before the column existed keeps the old behaviour."""
    from jwst_gc_pipeline.reduction.validate_offsets_table import (
        flag_diverged_column_pairs)
    assert not flag_diverged_column_pairs(_diverged(400.0, 0.400))


def test_the_validator_reads_an_absent_declination_as_absent():
    """Masked or non-finite means "not recorded", never "the equator"."""
    from jwst_gc_pipeline.reduction.validate_offsets_table import (
        flag_diverged_column_pairs)
    t = _diverged(400.0, 0.400, dec=np.nan)
    assert not flag_diverged_column_pairs(t)
    t2 = _diverged(400.0, 0.400, dec=0.0)
    t2[PROV_DEC_DEG_KEY] = np.ma.array([0.0], mask=[True])
    assert not flag_diverged_column_pairs(t2)


def test_the_validator_converts_degrees_not_radians():
    """`cos(-28.7)` without the degree conversion is 0.885 rather than 0.877 --
    close enough to look right and wrong enough to move the bound."""
    from jwst_gc_pipeline.reduction.validate_offsets_table import (
        flag_diverged_column_pairs)
    exact_deg = 400.0 / 1000.0 / np.cos(np.radians(GC_DEC))
    exact_rad = 400.0 / 1000.0 / np.cos(GC_DEC)
    assert abs(exact_deg - exact_rad) > 1e-3
    assert not flag_diverged_column_pairs(_diverged(400.0, exact_deg, dec=GC_DEC))
