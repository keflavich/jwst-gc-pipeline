"""The consensus table's magnitude ceiling, issue #395.

`seed_offsets_table_from_consensus` refuses a resulting row whose offset is too
large. Three things that test used to get wrong:

1. the per-visit BULK sentinel row -- the row that repairs a wrong-guide-star
   visit, and the reason `MAX_BULK_CORRECTION_ARCSEC` (60") exists -- was held
   to the per-exposure `MAX_CORRECTION_ARCSEC` (0.5"), so on every field whose
   corrections go to a `consensus` table the repair path was closed;
2. it compared `dra (arcsec)`, a right-ascension COORDINATE offset, against an
   ON-SKY ceiling, so the effective limit was cos(dec) smaller and
   field-dependent -- 0.438" at arches;
3. the ceiling was a literal, so the documented `ASTROM_MAX_CORRECTION_ARCSEC`
   override did not reach it.
"""
import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    BULK_EXPOSURE, BULK_MODULE, MAX_CORRECTION_ARCSEC, OffsetsTableUpdateError,
    seed_offsets_table_from_consensus)

DEC = -28.85            # arches / GC declination: cos(dec) = 0.8759
COSD = np.cos(np.radians(DEC))


def _corr(exposure, module, dra_mas, ddec_mas, filt="F212N", visit="1"):
    return dict(visit=visit, exposure=exposure, module=module, filtername=filt,
                dra_onsky_mas=dra_mas, ddec_onsky_mas=ddec_mas, dec_deg=DEC,
                source="m2 visit-consensus")


def _bulk(dra_mas, ddec_mas, filt="F212N", visit="1"):
    """A per-visit consensus->reference tie: exposure and module are both None,
    which is how `seed_offsets_table_from_consensus` recognises it."""
    return _corr(None, None, dra_mas, ddec_mas, filt=filt, visit=visit)


def test_a_large_bulk_tie_can_be_written(tmp_path):
    """The brick-1182 / cloudc-F410M / sgra-1939 class: a whole visit pointed
    seconds of arc away, on a field whose corrections go to a consensus table.

    4.06" is cloudc F410M's measured visit-2 error. Under the 0.5" literal this
    raised, and the message blamed "the upstream per-exposure measurement".
    """
    p = seed_offsets_table_from_consensus(
        str(tmp_path), "2045", "001", [_bulk(4060.0, -1200.0)], stage="m2")
    t = Table.read(p)
    assert len(t) == 1
    assert int(t["Exposure"][0]) == BULK_EXPOSURE
    assert str(t["Module"][0]) == BULK_MODULE
    # the stored value is the COORDINATE offset: on-sky / cos(dec)
    assert float(t["dra (arcsec)"][0]) == pytest.approx(4.060 / COSD, rel=1e-9)


def test_a_bulk_tie_over_the_bulk_ceiling_is_still_refused(tmp_path):
    """60" is the ceiling, not the absence of one."""
    with pytest.raises(OffsetsTableUpdateError) as exc:
        seed_offsets_table_from_consensus(
            str(tmp_path), "2045", "001", [_bulk(80_000.0, 0.0)], stage="m2")
    assert "ASTROM_MAX_BULK_CORRECTION_ARCSEC" in str(exc.value)


def test_a_per_exposure_row_keeps_the_tight_ceiling(tmp_path):
    """The bulk exemption must not widen the per-exposure rows: a consensus
    jitter fix is mas-scale, and 2" of it means the measurement is wrong."""
    with pytest.raises(OffsetsTableUpdateError):
        seed_offsets_table_from_consensus(
            str(tmp_path), "2045", "001", [_corr(2, "nrca1", 2000.0, 0.0)],
            stage="m2")


def test_the_ceiling_is_compared_on_sky_not_in_coordinate(tmp_path):
    """A row whose ON-SKY offset is inside the ceiling and whose COORDINATE
    offset is outside it.

    At dec = -28.85, cos(dec) = 0.8759, so an on-sky 0.48" is written as a
    coordinate 0.548" -- refused by a test that reads the coordinate column and
    compares it against the on-sky 0.5". The row records its own declination,
    so the exact factor is available.
    """
    onsky_mas = 480.0
    coordinate = (onsky_mas / 1000.0) / COSD
    assert coordinate > MAX_CORRECTION_ARCSEC     # the old test refused this
    assert onsky_mas / 1000.0 < MAX_CORRECTION_ARCSEC
    p = seed_offsets_table_from_consensus(
        str(tmp_path), "2045", "001", [_corr(2, "nrca1", onsky_mas, 0.0)],
        stage="m2")
    t = Table.read(p)
    assert float(t["dra (arcsec)"][0]) == pytest.approx(coordinate, rel=1e-9)


def test_the_documented_override_reaches_this_ceiling(tmp_path, monkeypatch):
    """`ASTROM_MAX_CORRECTION_ARCSEC` is documented as raising the per-exposure
    ceiling. Against a literal it did nothing."""
    corr = [_corr(2, "nrca1", 2000.0, 0.0)]
    with pytest.raises(OffsetsTableUpdateError):
        seed_offsets_table_from_consensus(
            str(tmp_path / "before"), "2045", "001", corr, stage="m2")
    monkeypatch.setenv("ASTROM_MAX_CORRECTION_ARCSEC", "5.0")
    p = seed_offsets_table_from_consensus(
        str(tmp_path / "after"), "2045", "001", corr, stage="m2")
    assert len(Table.read(p)) == 1


def test_a_row_with_no_recorded_declination_is_unchanged(tmp_path):
    """cos(dec) = 1 is the conservative end of [COS_DEC_MIN, 1]: an unlabelled
    row is held to exactly the ceiling it was held to before, so the conversion
    can only ever refuse LESS, never more."""
    corr = _corr(2, "nrca1", 400.0, 0.0)
    corr["dec_deg"] = 0.0            # cos(dec) = 1: coordinate == on-sky
    p = seed_offsets_table_from_consensus(
        str(tmp_path), "2045", "001", [corr], stage="m2")
    assert float(Table.read(p)["dra (arcsec)"][0]) == pytest.approx(0.400)
    corr_big = _corr(2, "nrca1", 600.0, 0.0)
    corr_big["dec_deg"] = 0.0
    with pytest.raises(OffsetsTableUpdateError):
        seed_offsets_table_from_consensus(
            str(tmp_path / "big"), "2045", "001", [corr_big], stage="m2")
