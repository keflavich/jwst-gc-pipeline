"""Per-module signs survive the consensus seeder (issue #823).

wd1's m2 records measured a long-wave inter-module differential: nrcalong
+6.7..+7.2 mas and nrcblong -6.2..-6.5 mas in Dec (F277W, F323N, F405N, F466N).
The consensus channel has no per-module row type, so the differential travels as
per-exposure rows, one per module.  These tests pin that the seeder keeps the
two signs apart, that `lookup_consensus_offset` hands each module its own value,
and that seeding further filters into an existing table leaves the rows already
there untouched.
"""
import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    lookup_consensus_offset, seed_offsets_table_from_consensus)

DEC = -61.8             # wd1; any declination serves, the values are read back
COSD = np.cos(np.radians(DEC))
VISIT = "jw01905001001"


def _corr(exposure, module, ddec_mas, filt, dra_mas=3.5):
    return dict(visit="1", exposure=exposure, module=module, filtername=filt,
                dra_onsky_mas=dra_mas, ddec_onsky_mas=ddec_mas, dec_deg=DEC,
                vgroup="02101", source="m2 visit-consensus")


def _lw_pair(filt):
    """nrcalong +6.5 / nrcblong -6.5 mas in Dec at two exposures."""
    return [_corr(e, m, s * 6.5, filt)
            for e in (1, 2)
            for m, s in (("nrcalong", 1.0), ("nrcblong", -1.0))]


def test_lw_module_signs_stay_apart_per_exposure(tmp_path):
    p = seed_offsets_table_from_consensus(
        str(tmp_path), "1905", "001", _lw_pair("F405N"), stage="m2")
    t = Table.read(p)
    assert len(t) == 4
    for row in t:
        want = 6.5e-3 if str(row["Module"]) == "nrcalong" else -6.5e-3
        assert float(row["ddec (arcsec)"]) == pytest.approx(want, rel=1e-9)
        assert float(row["dra (arcsec)"]) == pytest.approx(3.5e-3 / COSD,
                                                           rel=1e-9)


def test_lookup_returns_each_module_its_own_term(tmp_path):
    p = seed_offsets_table_from_consensus(
        str(tmp_path), "1905", "001", _lw_pair("F405N"), stage="m2")
    t = Table.read(p)
    for exposure in (1, 2):
        dra_a, ddec_a = lookup_consensus_offset(
            t, VISIT, exposure, "nrcalong", "F405N", vgroup="02101")
        dra_b, ddec_b = lookup_consensus_offset(
            t, VISIT, exposure, "nrcblong", "F405N", vgroup="02101")
        assert ddec_a == pytest.approx(+6.5e-3, rel=1e-9)
        assert ddec_b == pytest.approx(-6.5e-3, rel=1e-9)
        assert dra_a == pytest.approx(dra_b, rel=1e-9)


def test_seeding_a_new_filter_leaves_existing_rows_untouched(tmp_path):
    seed_offsets_table_from_consensus(
        str(tmp_path), "1905", "001", _lw_pair("F444W"), stage="m2")
    p = tmp_path / "offsets" / "Offsets_JWST_Brick1905_consensus.csv"
    before = {(str(r["Filter"]), int(r["Exposure"]), str(r["Module"])):
              (float(r["dra (arcsec)"]), float(r["ddec (arcsec)"]))
              for r in Table.read(str(p))}
    assert len(before) == 4

    seed_offsets_table_from_consensus(
        str(tmp_path), "1905", "001", _lw_pair("F405N"), stage="m2")
    t = Table.read(str(p))
    assert len(t) == 8
    after = {(str(r["Filter"]), int(r["Exposure"]), str(r["Module"])):
             (float(r["dra (arcsec)"]), float(r["ddec (arcsec)"]))
             for r in t}
    for key, val in before.items():
        assert after[key] == pytest.approx(val, rel=1e-12)
    assert {k[0] for k in after} == {"F444W", "F405N"}


def test_a_second_cycle_adds_the_residual_to_the_same_row(tmp_path):
    """The seeder is cumulative: a re-measured residual adds to the row, so a
    replayed stale record would double-apply (the #823 'do not replay' rule)."""
    seed_offsets_table_from_consensus(
        str(tmp_path), "1905", "001", [_corr(1, "nrcalong", 6.5, "F405N")],
        stage="m2")
    p = seed_offsets_table_from_consensus(
        str(tmp_path), "1905", "001", [_corr(1, "nrcalong", 2.0, "F405N")],
        stage="m2")
    t = Table.read(p)
    assert len(t) == 1
    assert float(t["ddec (arcsec)"][0]) == pytest.approx(8.5e-3, rel=1e-9)
