"""Per-module signs survive the consensus seeder (issue #823).

wd1's m2 records measured a long-wave inter-module differential: nrcalong
+6.7..+7.2 mas and nrcblong -6.2..-6.5 mas in Dec (F277W, F323N, F405N, F466N);
the fixtures here use round +-6.5 mas values rather than the measurements.
The consensus channel has no per-module row type, so the differential travels as
per-exposure rows, one per module.  These tests pin that the seeder keeps the
two signs apart, that `lookup_consensus_offset` hands each module its own value,
and that seeding further filters into an existing table leaves the rows already
there untouched.  The fixtures give the two modules different RA terms and carry
two filters in one seeding call, so a swap of the modules' terms or a collapsed
filter key changes a value that is read back.  The lookup's exact-module-first
choice and its Vgroup narrowing are pinned on the same LW pair.
"""
import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    lookup_consensus_offset, seed_offsets_table_from_consensus)

DEC = -61.8             # wd1; a mid-latitude value, cos(dec) is far from 0
COSD = np.cos(np.radians(DEC))
VISIT = "jw01905001001"


def _corr(exposure, module, ddec_mas, filt, dra_mas=3.5):
    return dict(visit="1", exposure=exposure, module=module, filtername=filt,
                dra_onsky_mas=dra_mas, ddec_onsky_mas=ddec_mas, dec_deg=DEC,
                vgroup="02101", source="m2 visit-consensus")


#: RA term per module (mas), different so a swap of the two modules is visible
RA_MAS = {"nrcalong": 3.5, "nrcblong": -2.5}


def _lw_pair(filt, vgroup="02101"):
    """nrcalong +6.5 / nrcblong -6.5 mas in Dec at two exposures, with a
    different RA term on each module."""
    out = []
    for e in (1, 2):
        for m, s in (("nrcalong", 1.0), ("nrcblong", -1.0)):
            c = _corr(e, m, s * 6.5, filt, dra_mas=RA_MAS[m])
            c["vgroup"] = vgroup
            out.append(c)
    return out


def test_lw_module_signs_stay_apart_per_exposure(tmp_path):
    # two filters in ONE seeding call: a collapsed filter key would merge them
    p = seed_offsets_table_from_consensus(
        str(tmp_path), "1905", "001",
        _lw_pair("F405N") + _lw_pair("F466N"), stage="m2")
    t = Table.read(p)
    assert len(t) == 8
    assert {str(f) for f in t["Filter"]} == {"F405N", "F466N"}
    for row in t:
        mod = str(row["Module"])
        want = 6.5e-3 if mod == "nrcalong" else -6.5e-3
        assert float(row["ddec (arcsec)"]) == pytest.approx(want, rel=1e-9)
        assert float(row["dra (arcsec)"]) == pytest.approx(
            RA_MAS[mod] * 1e-3 / COSD, rel=1e-9)


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
        assert dra_a == pytest.approx(RA_MAS["nrcalong"] * 1e-3 / COSD,
                                      rel=1e-9)
        assert dra_b == pytest.approx(RA_MAS["nrcblong"] * 1e-3 / COSD,
                                      rel=1e-9)


def test_lookup_prefers_the_exact_lw_module_over_the_family_row(tmp_path):
    """cloudef's shape: the pooled short-wave `nrcb` row sits beside `nrcblong`
    for one exposure.  The LW frame must get its own row, not match both."""
    t = Table.read(seed_offsets_table_from_consensus(
        str(tmp_path), "1905", "001", _lw_pair("F405N"), stage="m2"))
    # the seeder refuses to WRITE both spellings for one frame (issue #298), so
    # the table cloudef already carries is built by hand
    pooled = {c: t[c][0] for c in t.colnames}
    pooled.update(Module="nrcb", **{"dra (arcsec)": 9.0e-3,
                                    "ddec (arcsec)": 20.0e-3})
    t.add_row([pooled[c] for c in t.colnames])
    assert len(t) == 5 and str(t["Exposure"][4]) == str(t["Exposure"][0])
    dra, ddec = lookup_consensus_offset(
        t, VISIT, 1, "nrcblong", "F405N", vgroup="02101")
    assert ddec == pytest.approx(-6.5e-3, rel=1e-9)
    assert dra == pytest.approx(RA_MAS["nrcblong"] * 1e-3 / COSD, rel=1e-9)
    # the pooled row still serves the SW frame that names it
    dra, ddec = lookup_consensus_offset(
        t, VISIT, 1, "nrcb", "F405N", vgroup="02101")
    assert ddec == pytest.approx(20.0e-3, rel=1e-9)


def test_lookup_narrows_by_vgroup_when_exposure_numbers_repeat(tmp_path):
    """Exposure numbers restart per visit group; two groups carrying the same
    (visit, filter, exposure, module) must each get their own term."""
    other = _lw_pair("F405N", vgroup="03101")
    for c in other:
        c["ddec_onsky_mas"] += 1.0
    t = Table.read(seed_offsets_table_from_consensus(
        str(tmp_path), "1905", "001", _lw_pair("F405N") + other, stage="m2"))
    assert len(t) == 8
    for vg, extra in (("02101", 0.0), ("03101", 1.0e-3)):
        _, ddec_a = lookup_consensus_offset(
            t, VISIT, 1, "nrcalong", "F405N", vgroup=vg)
        _, ddec_b = lookup_consensus_offset(
            t, VISIT, 1, "nrcblong", "F405N", vgroup=vg)
        assert ddec_a == pytest.approx(6.5e-3 + extra, rel=1e-9)
        assert ddec_b == pytest.approx(-6.5e-3 + extra, rel=1e-9)


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
    # the RA term accumulates too (3.5 mas in each cycle)
    assert float(t["dra (arcsec)"][0]) == pytest.approx(2 * 3.5e-3 / COSD,
                                                        rel=1e-9)
