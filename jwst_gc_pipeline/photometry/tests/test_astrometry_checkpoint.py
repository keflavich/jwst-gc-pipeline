"""Tests for the stage astrometry checkpoints (astrometry_checkpoint.py) and
the local residual map (astrometry_offsets.local_residual_map)."""
import json
import os

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    AstrometryRegressionError, CrossFilterAstrometryError,
    OffsetsTableUpdateError, mark_i2d_stale, provenance_header_cards,
    run_crossfilter_checkpoint, run_visit_checkpoint, update_offsets_table,
)
from jwst_gc_pipeline.photometry.astrometry_offsets import (
    GlobalTieNotVerifiedError, local_residual_map, measure_offset,
)
from .test_visit_consensus import (
    RA0, DEC0, COSD, _exposure_table, _field, _reference_sets, _visit_tables)

DEC_TEST = DEC0


# ---------------------------------------------------------------------------
# measure_offset error bars
# ---------------------------------------------------------------------------

def test_measure_offset_reports_error_bars():
    ra, dec = _field()
    a = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    rng = np.random.default_rng(3)
    b = SkyCoord(ra=(ra + (5.0 + rng.normal(0, 1.0, len(ra))) / 3.6e6 / COSD) * u.deg,
                 dec=dec * u.deg, frame="icrs")
    res = measure_offset(a, b)
    assert res["ok"]
    assert np.isfinite(res["dra_err"]) and res["dra_err"] > 0
    assert res["n_peak"] >= 30
    # ~1 mas scatter over ~400 stars -> sub-0.5 mas standard error
    assert res["dra_err"] < 1.0
    assert res["dra"] == pytest.approx(5.0, abs=3 * max(res["dra_err"], 0.3))


# ---------------------------------------------------------------------------
# local residual map
# ---------------------------------------------------------------------------

def _dense_pair(n=20000, extent=60.0, patch=None, noise_mas=1.0, seed=11):
    """Two catalogs of the same stars; ``patch``: (ra_lo, ra_hi, dec_lo,
    dec_hi, dra_mas) region (arcsec offsets within the field) shifted in b."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, extent, n)   # arcsec within field
    y = rng.uniform(0, extent, n)
    ra = RA0 + x / 3600.0 / COSD
    dec = DEC0 + y / 3600.0
    dra = np.zeros(n)
    if patch:
        lo_x, hi_x, lo_y, hi_y, dra_mas = patch
        inside = (x >= lo_x) & (x < hi_x) & (y >= lo_y) & (y < hi_y)
        dra[inside] = dra_mas
    a = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    b = SkyCoord(ra=(ra + (dra + rng.normal(0, noise_mas, n)) / 3.6e6 / COSD) * u.deg,
                 dec=(dec + rng.normal(0, noise_mas, n) / 3.6e6) * u.deg,
                 frame="icrs")
    return a, b


def test_local_residual_map_clean_field():
    a, b = _dense_pair()
    glob_res = measure_offset(a, b)
    assert glob_res["ok"]
    lrm = local_residual_map(a, b, glob_res, cell_arcsec=10.0, min_stars=20,
                             context="clean")
    assert lrm["n_measured"] > 10
    assert lrm["n_flagged"] == 0
    assert lrm["clean"]


def test_local_residual_map_flags_offset_patch():
    # a 20 mas patch in a 15"x15" corner -- the class of localized
    # misregistration a bulk offset (~0 here) cannot see
    a, b = _dense_pair(patch=(0.0, 15.0, 0.0, 15.0, 20.0))
    glob_res = measure_offset(a, b)
    assert glob_res["ok"]
    lrm = local_residual_map(a, b, glob_res, cell_arcsec=5.0, min_stars=20,
                             tol_mas=15.0, context="patch")
    assert lrm["n_flagged"] >= 4
    flagged = [c for c in lrm["cells"] if c["flagged"]]
    for c in flagged:
        # flagged cells must be inside (or straddling) the patch and carry a
        # significant positive-dra residual (edge cells mix populations, so
        # only require the amplitude to be well above noise, not exactly 20)
        assert (c["ra0"] - RA0) * COSD * 3600.0 < 20.0
        assert (c["dec0"] - DEC0) * 3600.0 < 20.0
        assert c["dra_mas"] > 8.0
        assert c["significant"]
    # cells fully inside the patch must recover the injected amplitude
    interior = [c for c in flagged
                if (c["ra0"] - RA0) * COSD * 3600.0 < 12.0
                and (c["dec0"] - DEC0) * 3600.0 < 12.0]
    assert interior
    for c in interior:
        assert c["dra_mas"] == pytest.approx(20.0, abs=6.0)


def test_local_residual_map_single_star_cell_not_flagged():
    # a huge offset carried by too few stars is NOT a measurement
    a, b = _dense_pair(n=300, patch=(0.0, 3.0, 0.0, 3.0, 50.0))
    glob_res = measure_offset(a, b)
    lrm = local_residual_map(a, b, glob_res, cell_arcsec=3.0, min_stars=10,
                             context="sparse-cell")
    # patch cells have <10 stars at this density -> not measurable -> not flagged
    assert lrm["n_flagged"] == 0


def test_local_residual_map_requires_verified_tie():
    a, b = _dense_pair(n=2000)
    with pytest.raises(GlobalTieNotVerifiedError):
        local_residual_map(a, b, None, context="no-tie")
    with pytest.raises(GlobalTieNotVerifiedError):
        local_residual_map(a, b, dict(ok=False, off=0.0, dra=0, ddec=0,
                                      swept=False), context="bad-tie")
    with pytest.raises(GlobalTieNotVerifiedError):
        # verified but SWEPT (gross shift) -> refuse
        local_residual_map(a, b, dict(ok=True, off=20000.0, dra=20000.0,
                                      ddec=0.0, swept=True), context="swept")
    with pytest.raises(GlobalTieNotVerifiedError):
        # offset not << match radius -> ambiguous pairing -> refuse
        local_residual_map(a, b, dict(ok=True, off=200.0, dra=200.0, ddec=0.0,
                                      swept=False), context="big-offset")


# ---------------------------------------------------------------------------
# offsets-table update
# ---------------------------------------------------------------------------

def _offsets_csv(tmp_path, per_exposure=True):
    rows = []
    for visit, base in (("jw01182004001", -17.5), ("jw01182004002", 1.9)):
        if per_exposure:
            for exp in (1, 2):
                rows.append(dict(Filter="F212N", Module="nrcb1", Visit=visit,
                                 Exposure=exp, dra=base, ddec=0.5))
        else:
            rows.append(dict(Filter="F212N", Module="nrcb", Visit=visit,
                             dra=base, ddec=0.5))
    path = str(tmp_path / "Offsets_JWST_Brick1182_TEST.csv")
    Table(rows).write(path, overwrite=True)
    return path


def test_update_offsets_table_applies_correction_with_provenance(tmp_path):
    path = _offsets_csv(tmp_path)
    corr = [dict(visit="jw01182004001", exposure=1, module="nrcb1",
                 filtername="F212N", dra_onsky_mas=100.0, ddec_onsky_mas=-50.0,
                 dec_deg=DEC_TEST, source="test m2 visit-consensus")]
    out = update_offsets_table(path, corr, "m2")
    row = out[(np.array([str(v) for v in out["Visit"]]) == "jw01182004001")
              & (out["Exposure"] == 1)][0]
    assert row["dra"] == pytest.approx(-17.5 + 0.1 / COSD, abs=1e-6)
    assert row["ddec"] == pytest.approx(0.5 - 0.05, abs=1e-9)
    assert row["prov_stage"] == "m2"
    assert row["prov_dra_added_mas"] == pytest.approx(100.0)
    # untouched row keeps its value and carries no provenance
    other = out[(np.array([str(v) for v in out["Visit"]]) == "jw01182004002")][0]
    assert other["dra"] == pytest.approx(1.9)
    assert str(other["prov_stage"]) == ""
    # backup of the original was kept
    backups = [f for f in os.listdir(tmp_path) if ".pre_m2_" in f]
    assert len(backups) == 1


def test_update_offsets_table_refuses_unmatched_correction(tmp_path):
    path = _offsets_csv(tmp_path)
    with pytest.raises(OffsetsTableUpdateError):
        update_offsets_table(path, [dict(
            visit="jw01182004009", exposure=1, module="nrcb1",
            filtername="F212N", dra_onsky_mas=10.0, ddec_onsky_mas=0.0,
            dec_deg=DEC_TEST)], "m2")


def test_update_offsets_table_refuses_perexposure_on_pervisit_table(tmp_path):
    # a per-VISIT (module-locked) table cannot express a single-exposure fix
    path = _offsets_csv(tmp_path, per_exposure=False)
    with pytest.raises(OffsetsTableUpdateError):
        update_offsets_table(path, [dict(
            visit="jw01182004001", exposure=2, module="nrcb1",
            filtername="F212N", dra_onsky_mas=10.0, ddec_onsky_mas=0.0,
            dec_deg=DEC_TEST)], "m2")


def test_update_offsets_table_refuses_multi_vgroup_on_vgroupless_table(tmp_path):
    # the consensus key is vgroup-aware, so two same-numbered exposures in
    # different visit groups emit SEPARATE corrections -- but no offsets table
    # has a Vgroup column, so both would match one row and be summed
    path = _offsets_csv(tmp_path)
    before = open(path).read()
    corr = [dict(visit="jw01182004001", exposure=1, module="nrcb1",
                 filtername="F212N", vgroup=vg, dra_onsky_mas=0.0,
                 ddec_onsky_mas=100.0, dec_deg=DEC_TEST)
            for vg in ("06201", "12201")]
    with pytest.raises(OffsetsTableUpdateError, match="(?i)more than one visit group"):
        update_offsets_table(path, corr, "m2")
    assert open(path).read() == before


def test_update_offsets_table_single_vgroup_ok(tmp_path):
    # one vgroup per (visit, filter, exposure, module) maps 1:1 -> allowed
    path = _offsets_csv(tmp_path)
    out = update_offsets_table(path, [dict(
        visit="jw01182004001", exposure=1, module="nrcb1", filtername="F212N",
        vgroup="06201", dra_onsky_mas=0.0, ddec_onsky_mas=100.0,
        dec_deg=DEC_TEST)], "m2")
    row = out[(np.array([str(v) for v in out["Visit"]]) == "jw01182004001")
              & (out["Exposure"] == 1)][0]
    assert row["ddec"] == pytest.approx(0.5 + 0.1, abs=1e-9)


def test_update_offsets_table_refuses_visit_collapse(tmp_path):
    # a correction that lands two visits on the SAME value is the brick-1182
    # collapse signature -- must refuse to write.  ~17" wide, but it is a BULK
    # tie (exposure=None, module=None) so the magnitude gate lets it through and
    # the COLLAPSE guard is what fires.
    path = _offsets_csv(tmp_path)
    with pytest.raises(OffsetsTableUpdateError, match="(?i)collaps"):
        update_offsets_table(path, [dict(
            visit="jw01182004001", exposure=None, module=None,
            filtername="F212N",
            dra_onsky_mas=(1.9 - (-17.5)) * 1000.0 * COSD,
            ddec_onsky_mas=0.0, dec_deg=DEC_TEST)], "m2")


def test_update_offsets_table_refuses_oversized_correction(tmp_path):
    # cloudef 2026-07-28: a +102" ddec correction was applied and then
    # compounded across re-tie iterations.  A tie correction is mas-scale.
    path = _offsets_csv(tmp_path)
    with pytest.raises(OffsetsTableUpdateError, match="exceed"):
        update_offsets_table(path, [dict(
            visit="jw01182004001", exposure=1, module="nrcb1",
            filtername="F212N", dra_onsky_mas=24003.4,
            ddec_onsky_mas=102339.0, dec_deg=DEC_TEST)], "m2")


def test_update_offsets_table_oversized_correction_leaves_table_untouched(tmp_path):
    # fail BEFORE writing: no mutation, and no backup file left behind
    path = _offsets_csv(tmp_path)
    before = open(path).read()
    with pytest.raises(OffsetsTableUpdateError):
        update_offsets_table(path, [dict(
            visit="jw01182004001", exposure=1, module="nrcb1",
            filtername="F212N", dra_onsky_mas=0.0, ddec_onsky_mas=600.0,
            dec_deg=DEC_TEST)], "m2")
    assert open(path).read() == before
    assert [f for f in os.listdir(tmp_path) if ".pre_m2_" in f] == []


def test_update_offsets_table_allows_correction_just_under_ceiling(tmp_path):
    # 0.4" < 0.5" ceiling -> applies normally (the gate must not be so tight
    # that a legitimate large-but-plausible tie is blocked)
    path = _offsets_csv(tmp_path)
    out = update_offsets_table(path, [dict(
        visit="jw01182004001", exposure=1, module="nrcb1",
        filtername="F212N", dra_onsky_mas=0.0, ddec_onsky_mas=400.0,
        dec_deg=DEC_TEST)], "m2")
    row = out[(np.array([str(v) for v in out["Visit"]]) == "jw01182004001")
              & (out["Exposure"] == 1)][0]
    assert row["ddec"] == pytest.approx(0.5 + 0.4, abs=1e-9)


def test_update_offsets_table_ceiling_override(tmp_path, monkeypatch):
    # a deliberate gross re-authoring at PER-EXPOSURE granularity stays
    # available, but only via an explicit env override
    monkeypatch.setenv("ASTROM_MAX_CORRECTION_ARCSEC", "30")
    path = _offsets_csv(tmp_path)
    out = update_offsets_table(path, [dict(
        visit="jw01182004001", exposure=1, module="nrcb1",
        filtername="F212N", dra_onsky_mas=0.0, ddec_onsky_mas=20000.0,
        dec_deg=DEC_TEST)], "m2")
    row = out[(np.array([str(v) for v in out["Visit"]]) == "jw01182004001")
              & (out["Exposure"] == 1)][0]
    assert row["ddec"] == pytest.approx(0.5 + 20.0, abs=1e-9)


@pytest.mark.parametrize("ddec_arcsec", [4.0, 13.0, 17.0])
def test_update_offsets_table_allows_gross_guide_star_bulk_tie(tmp_path, ddec_arcsec):
    # Early-Cycle visits that acquired the WRONG GUIDE STAR are really offset by
    # arcseconds (~4", ~13", ~17" cases all exist).  Correcting that is the job,
    # so a per-VISIT BULK tie (exposure=None, module=None) must NOT be blocked by
    # the mas-scale per-exposure ceiling.
    path = _offsets_csv(tmp_path)
    out = update_offsets_table(path, [dict(
        visit="jw01182004001", exposure=None, module=None, filtername="F212N",
        dra_onsky_mas=0.0, ddec_onsky_mas=ddec_arcsec * 1000.0,
        dec_deg=DEC_TEST)], "m2")
    rows = out[np.array([str(v) for v in out["Visit"]]) == "jw01182004001"]
    # applied to EVERY exposure of the visit
    for row in rows:
        assert row["ddec"] == pytest.approx(0.5 + ddec_arcsec, abs=1e-9)


def test_update_offsets_table_refuses_absurd_bulk_tie(tmp_path):
    # ...but a bulk tie beyond the measure_offset sweep ceiling (60") cannot have
    # come from a real swept peak
    path = _offsets_csv(tmp_path)
    with pytest.raises(OffsetsTableUpdateError, match="exceed"):
        update_offsets_table(path, [dict(
            visit="jw01182004001", exposure=None, module=None,
            filtername="F212N", dra_onsky_mas=0.0, ddec_onsky_mas=102339.0,
            dec_deg=DEC_TEST)], "m2")


@pytest.mark.parametrize("dra,ddec", [
    (float("nan"), float("nan")), (float("nan"), 0.0), (0.0, float("inf"))])
def test_update_offsets_table_refuses_nonfinite_correction(tmp_path, dra, ddec):
    # abs(nan) > limit is False, so a NaN would sail through a naive ceiling and
    # poison the row -- and assert_offsets_table_sane cannot catch it either
    # (its collapse comparisons against NaN are all False)
    path = _offsets_csv(tmp_path)
    before = open(path).read()
    with pytest.raises(OffsetsTableUpdateError, match="(?i)non-finite|exceed"):
        update_offsets_table(path, [dict(
            visit="jw01182004001", exposure=1, module="nrcb1",
            filtername="F212N", dra_onsky_mas=dra, ddec_onsky_mas=ddec,
            dec_deg=DEC_TEST)], "m2")
    assert open(path).read() == before


def _module_less_csv(tmp_path):
    """sgrc/cloudef shape: per-exposure rows, NO Module column."""
    rows = [dict(Visit="jw04147012001", Exposure=e, Filter="F115W",
                 **{"dra (arcsec)": 0.0, "ddec (arcsec)": 0.0}) for e in (1, 2)]
    path = str(tmp_path / "Offsets_JWST_Brick4147_VIRAC2locked.csv")
    Table(rows).write(path, overwrite=True)
    return path


def test_update_offsets_table_refuses_multimodule_on_moduleless_table(tmp_path):
    # 8 detectors x +0.4" each is legal per-correction but sums to +3.2" on one
    # row -- the cloudef mechanism, invisible to the magnitude ceiling
    path = _module_less_csv(tmp_path)
    before = open(path).read()
    corr = [dict(visit="jw04147012001", exposure=1, module=m,
                 filtername="F115W", dra_onsky_mas=0.0, ddec_onsky_mas=400.0,
                 dec_deg=DEC_TEST)
            for m in ("nrca1", "nrca2", "nrca3", "nrca4",
                      "nrcb1", "nrcb2", "nrcb3", "nrcb4")]
    with pytest.raises(OffsetsTableUpdateError, match="(?i)more than one module"):
        update_offsets_table(path, corr, "m2")
    assert open(path).read() == before


def test_update_offsets_table_single_module_on_moduleless_table_ok(tmp_path):
    # one module per (visit, filter, exposure) maps 1:1 -> still allowed
    path = _module_less_csv(tmp_path)
    out = update_offsets_table(path, [dict(
        visit="jw04147012001", exposure=1, module="nrcb1", filtername="F115W",
        dra_onsky_mas=0.0, ddec_onsky_mas=400.0, dec_deg=DEC_TEST)], "m2")
    assert out[out["Exposure"] == 1][0]["ddec (arcsec)"] == pytest.approx(0.4)


def test_update_offsets_table_refuses_cumulative_drift(tmp_path, monkeypatch):
    # Legal-sized corrections applied repeatedly must not creep without bound:
    # each +0.4" is under the 0.5" per-correction ceiling, so only the
    # cumulative prov_* bound can catch the runaway.  Lower that bound rather
    # than looping 150x to reach the 60" default.
    monkeypatch.setenv("ASTROM_MAX_BULK_CORRECTION_ARCSEC", "2")
    path = _module_less_csv(tmp_path)
    corr = [dict(visit="jw04147012001", exposure=1, module="nrcb1",
                 filtername="F115W", dra_onsky_mas=0.0, ddec_onsky_mas=400.0,
                 dec_deg=DEC_TEST)]
    applied = 0
    for _ in range(20):             # 0.4" each -> trips once past 2"
        try:
            update_offsets_table(path, corr, "m2", backup=False)
            applied += 1
        except OffsetsTableUpdateError as ex:
            assert "accumulated" in str(ex)
            break
    else:
        pytest.fail("cumulative drift was never bounded")
    # tripped only AFTER the bound was genuinely exceeded, not before
    assert applied == 5, f"expected 5 applies (5 x 0.4\" = 2.0\"), got {applied}"


def test_update_offsets_table_rejects_bad_env_limit(tmp_path, monkeypatch):
    # a 0/negative/garbage ceiling must be a clear config error, not a gate that
    # silently refuses every correction
    path = _offsets_csv(tmp_path)
    corr = [dict(visit="jw01182004001", exposure=1, module="nrcb1",
                 filtername="F212N", dra_onsky_mas=1.0, ddec_onsky_mas=0.0,
                 dec_deg=DEC_TEST)]
    for bad in ("0", "-1", "abc"):
        monkeypatch.setenv("ASTROM_MAX_CORRECTION_ARCSEC", bad)
        with pytest.raises(OffsetsTableUpdateError, match="(?i)positive|not a number"):
            update_offsets_table(path, corr, "m2")
    # whitespace means UNSET, not an error
    monkeypatch.setenv("ASTROM_MAX_CORRECTION_ARCSEC", "   ")
    update_offsets_table(path, corr, "m2")


def test_update_offsets_table_accepts_generator(tmp_path):
    # the checks iterate corrections before the apply loop; a generator would be
    # consumed by them and the update would silently write an unchanged table
    path = _offsets_csv(tmp_path)
    corr = (dict(visit="jw01182004001", exposure=1, module="nrcb1",
                 filtername="F212N", dra_onsky_mas=0.0, ddec_onsky_mas=100.0,
                 dec_deg=DEC_TEST) for _ in range(1))
    out = update_offsets_table(path, corr, "m2")
    row = out[(np.array([str(v) for v in out["Visit"]]) == "jw01182004001")
              & (out["Exposure"] == 1)][0]
    assert row["ddec"] == pytest.approx(0.5 + 0.1, abs=1e-9)


def test_update_offsets_table_perexposure_ceiling_unaffected_by_bulk_limit(tmp_path):
    # the loose BULK limit must not leak onto per-exposure corrections: cloudef's
    # +102" runaway was written to a per-EXPOSURE row, and stays blocked
    path = _offsets_csv(tmp_path)
    with pytest.raises(OffsetsTableUpdateError, match="exceed"):
        update_offsets_table(path, [dict(
            visit="jw01182004001", exposure=1, module="nrcb1",
            filtername="F212N", dra_onsky_mas=0.0, ddec_onsky_mas=4000.0,
            dec_deg=DEC_TEST)], "m2")


# ---------------------------------------------------------------------------
# stale-tagging
# ---------------------------------------------------------------------------

def test_mark_i2d_stale_renames_and_documents(tmp_path):
    p1 = tmp_path / "jw02221-o001_t001_nircam_clear-f212n-merged_i2d.fits"
    p1.write_bytes(b"fake")
    renames = mark_i2d_stale([str(p1)], reason="test", record_dir=str(tmp_path))
    assert len(renames) == 1
    old, new = renames[0]
    assert not os.path.exists(old)
    assert new.endswith("_i2d_im0_badastrom.fits")
    assert os.path.exists(new)
    why = json.load(open(new + ".why.json"))
    assert why["reason"] == "test"
    # ledger entry written
    ledger = (tmp_path / "stale_i2d_renames.json").read_text()
    assert "im0_badastrom" in ledger
    # idempotent-ish: tagging again finds nothing (old gone, new already tagged)
    assert mark_i2d_stale([str(p1), new], reason="again") == []


# ---------------------------------------------------------------------------
# stage checkpoints
# ---------------------------------------------------------------------------

def test_m2_checkpoint_proposes_corrections_and_passes(tmp_path):
    tables = _visit_tables(misaligned={2: (8.0, 0.0)})
    record = run_visit_checkpoint(tables, "m2", filtername="F212N",
                                  record_dir=str(tmp_path), context="test")
    assert record["correcting"]
    assert len(record["corrections"]) == 1
    corr = record["corrections"][0]
    assert corr["exposure"] == 2
    assert corr["dra_onsky_mas"] == pytest.approx(-8.0, abs=1.5)
    # record written
    assert any(f.startswith("checkpoint_m2_") for f in os.listdir(tmp_path))


def test_late_stage_shift_raises_regression(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_LATE_STAGE_ASTROM_SHIFT", raising=False)
    tables = _visit_tables(misaligned={2: (8.0, 0.0)})
    with pytest.raises(AstrometryRegressionError):
        run_visit_checkpoint(tables, "m4", filtername="F212N",
                             record_dir=str(tmp_path), context="test")


def test_late_stage_shift_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_LATE_STAGE_ASTROM_SHIFT", "1")
    tables = _visit_tables(misaligned={2: (8.0, 0.0)})
    record = run_visit_checkpoint(tables, "m4", filtername="F212N",
                                  record_dir=str(tmp_path), context="test")
    assert not record["passed"]
    assert not record["correcting"]
    assert record["corrections"] == []


def test_late_stage_stable_passes(tmp_path):
    tables = _visit_tables()
    record = run_visit_checkpoint(tables, "m5", filtername="F212N",
                                  record_dir=str(tmp_path), context="test")
    assert record["passed"]
    assert record["corrections"] == []


def test_unbuildable_consensus_is_unverified_not_fatal(tmp_path):
    # 2 exposures with almost no stars: cannot verify != measured shift
    tables = _visit_tables(n_exp=2)
    tables = [t[:5] for t in tables]
    for t, src in zip(tables, _visit_tables(n_exp=2)):
        t.meta.update(src.meta)
    record = run_visit_checkpoint(tables, "m4", filtername="F212N",
                                  record_dir=str(tmp_path), context="test")
    assert record["passed"]           # no MEASURED shift
    assert not record["all_verified"]  # but explicitly not verified
    assert record["unverified"]


# ---------------------------------------------------------------------------
# frozen-stage consensus->reference DELTA gate (regression = MOVEMENT since the
# m2 freeze, not a nonzero absolute residual).  Reproduces the bug this branch
# fixes: brick V12 F182M m2 10.09 mas PASS -> m3 10.31 mas false REGRESSION.
# ---------------------------------------------------------------------------

# These target the frozen-stage DELTA control flow (baseline read -> delta vs
# the m2 freeze -> raise/STABLE), not the consensus/reference numerics (already
# covered by test_visit_consensus and the dense reference-tie tests).  The heavy
# build_visit_consensus + measure_reference_tie are monkeypatched to controlled
# values so the branch runs in milliseconds instead of minutes.
import jwst_gc_pipeline.photometry.astrometry_checkpoint as _ac


def _tiny_visit_table():
    """One minimal per-frame catalog that groups to (visit '001', F212N); its
    content is irrelevant -- build_visit_consensus is monkeypatched."""
    ra, dec = _field(n=5)
    return _exposure_table(ra, dec, exposure=1)


def _patch_consensus_and_tie(monkeypatch, dra_now, ddec_now, apply_ok=True):
    coords = SkyCoord(ra=[RA0, RA0] * u.deg, dec=[DEC0, DEC0] * u.deg, frame="icrs")

    def _fake_consensus(tables, context="", **kw):
        return dict(coords=coords, mag=None, exposures=[],
                    anchor_key=("001", 1, "nrcb1", "F212N"),
                    scatter_mas=np.array([1.0]), consensus_ok=True, skipped=[])

    def _fake_tie(cons_coords, ref_all, ref_sparse, **kw):
        return dict(off_mas=float(np.hypot(dra_now, ddec_now)), apply_ok=apply_ok,
                    dra_mas=float(dra_now), ddec_mas=float(ddec_now),
                    cross_reference={"agree": True, "sep_mas": 0.0},
                    cross_reference_gross_ok=True, per_tile={"clean": True},
                    swept=False,
                    vs_full={"dra": float(dra_now), "ddec": float(ddec_now)})

    monkeypatch.setattr(_ac, "build_visit_consensus", _fake_consensus)
    monkeypatch.setattr(_ac, "measure_reference_tie", _fake_tie)


def _write_m2_baseline(record_dir, dra, ddec, visit="001", filt="F212N",
                       vs_full=None):
    """Write an m2 record.  ``(dra, ddec)`` is the REPORTED bulk
    (``reference_tie.dra_mas/ddec_mas``) -- the same-star tie the frozen-stage
    delta gate must compare against.  ``vs_full`` (default = same as the bulk)
    is the histogram check A; set it DIFFERENT to model the post-same-star record
    where histogram != same-star (the false-regression bug)."""
    vf = vs_full if vs_full is not None else (dra, ddec)
    rec = dict(visits=[dict(visit=visit, reference_tie=dict(
        dra_mas=dra, ddec_mas=ddec,
        vs_full=dict(dra=vf[0], ddec=vf[1])))])
    with open(os.path.join(record_dir, f"checkpoint_m2_{filt}_latest.json"), "w") as fh:
        json.dump(rec, fh)


_DUMMY_REFCAT = dict(all=None, sparse=None, mag=None)  # measure_reference_tie is patched


def test_frozen_stage_stable_tie_no_regression(tmp_path, monkeypatch):
    """m2 froze a 10 mas reference tie; m3 re-measures the SAME (10, 0) tie ->
    delta 0 <= tol -> STABLE, no raise, passed.  (The exact brick F182M case:
    m2 ~10 mas PASS must NOT become an m3 false regression.)"""
    _write_m2_baseline(str(tmp_path), 10.0, 0.0)
    _patch_consensus_and_tie(monkeypatch, dra_now=10.0, ddec_now=0.0)
    rec = run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=_DUMMY_REFCAT,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert rec["passed"]
    assert rec["failures"] == []


def test_frozen_stage_moved_tie_raises(tmp_path, monkeypatch):
    """m2 froze the tie at (10, 0); the solution then MOVED to (20, 0) ->
    delta 10 > tol -> AstrometryRegressionError (the real regression)."""
    _write_m2_baseline(str(tmp_path), 10.0, 0.0)
    _patch_consensus_and_tie(monkeypatch, dra_now=20.0, ddec_now=0.0)
    with pytest.raises(AstrometryRegressionError):
        run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=_DUMMY_REFCAT,
                             filtername="F212N", record_dir=str(tmp_path),
                             context="test")


def test_frozen_stage_no_m2_baseline_raises(tmp_path, monkeypatch):
    """A frozen stage with an apply_ok tie but NO m2 baseline record (fail
    closed): cannot prove the solution didn't move -> raise."""
    _patch_consensus_and_tie(monkeypatch, dra_now=10.0, ddec_now=0.0)
    # record_dir has no checkpoint_m2_F212N_latest.json
    with pytest.raises(AstrometryRegressionError):
        run_visit_checkpoint([_tiny_visit_table()], "m4", refcat=_DUMMY_REFCAT,
                             filtername="F212N", record_dir=str(tmp_path),
                             context="test")


def test_frozen_baseline_reads_samestar_not_histogram(tmp_path, monkeypatch):
    """The baseline must compare the SAME estimator the m3 tie uses (same-star
    reported bulk), NOT the histogram vs_full.  m2 recorded reported bulk
    (+1,-6) [same-star] with vs_full (+6.7,-7.5) [histogram]; m3 re-measures the
    same (+1,-6) same-star tie.  Reading vs_full would compute a spurious ~5.9
    mas 'movement' (the histogram-vs-same-star method difference) and RAISE --
    the exact brick F182M m3 false regression (2026-07-19).  With the fix, the
    stable tie is STABLE."""
    _write_m2_baseline(str(tmp_path), 1.0, -6.0, vs_full=(6.7, -7.5))
    _patch_consensus_and_tie(monkeypatch, dra_now=1.0, ddec_now=-6.0)
    rec = run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=_DUMMY_REFCAT,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert rec["passed"]
    assert rec["failures"] == []


def test_frozen_baseline_legacy_vs_full_fallback(tmp_path, monkeypatch):
    """A legacy m2 record with only vs_full (no reported-bulk field) still reads
    a baseline (backward compatible)."""
    rec_path = os.path.join(str(tmp_path), "checkpoint_m2_F212N_latest.json")
    with open(rec_path, "w") as fh:
        json.dump(dict(visits=[dict(visit="001",
                  reference_tie=dict(vs_full=dict(dra=10.0, ddec=0.0)))]), fh)
    _patch_consensus_and_tie(monkeypatch, dra_now=20.0, ddec_now=0.0)  # moved 10
    with pytest.raises(AstrometryRegressionError):
        run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=_DUMMY_REFCAT,
                             filtername="F212N", record_dir=str(tmp_path),
                             context="test")


def test_frozen_stage_m2_refused_its_own_tie_is_unverified(tmp_path, monkeypatch):
    """m2 MEASURED a reference tie and REFUSED it (apply_ok False -- untrustworthy).
    A refused tie is a rejected measurement, not a freeze point, so m3's clean tie
    cannot have 'moved' away from it -> UNVERIFIED, no raise.

    w51 F140M (2026-08-02): m2 rejected a 7827 mas swept-histogram peak
    (per_tile clean=False, swept=True) and recorded it in `unverified`; m3 then
    measured a clean 32 mas SAME-STAR tie and raised 'MOVED 7794.98 mas since the
    m2 freeze' -- blocking the field because the measurement got BETTER."""
    rec_path = os.path.join(str(tmp_path), "checkpoint_m2_F212N_latest.json")
    with open(rec_path, "w") as fh:
        json.dump(dict(visits=[dict(visit="001", reference_tie=dict(
            apply_ok=False, dra_mas=-7694.2, ddec_mas=1436.1,
            off_mas=7827.1, swept=True))]), fh)
    _patch_consensus_and_tie(monkeypatch, dra_now=-30.8, ddec_now=9.8)
    rec = run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=_DUMMY_REFCAT,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert rec["passed"]
    assert rec["failures"] == []
    # still held by the release gate's all_verified check
    assert not rec["all_verified"]
    assert any("m2 MEASURED but REFUSED" in u for u in rec["unverified"])


def test_frozen_stage_stable_against_refused_m2_tie_keeps_its_pass(
        tmp_path, monkeypatch):
    """A REFUSED m2 tie is still a MEASUREMENT, and a later stage that lands on
    top of it has proved the solution did not move.

    sgra F212N on disk: m2 refused a 48.49 mas tie (-48.247,-4.890) because its
    independent checks disagreed; m3 measures (-47.836,-4.926), a delta of
    0.41 mas, and records all_verified: true.  Discarding the baseline just
    because m2 declined to APPLY it would convert that verified pass into
    UNVERIFIED and hand sgra a release-gate block it does not have today."""
    rec_path = os.path.join(str(tmp_path), "checkpoint_m2_F212N_latest.json")
    with open(rec_path, "w") as fh:
        json.dump(dict(visits=[dict(visit="001", reference_tie=dict(
            apply_ok=False, dra_mas=-48.247, ddec_mas=-4.890,
            off_mas=48.49, swept=False))]), fh)
    _patch_consensus_and_tie(monkeypatch, dra_now=-47.836, ddec_now=-4.926)
    rec = run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=_DUMMY_REFCAT,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert rec["passed"]
    assert rec["failures"] == []
    # the whole point: the stability result is KEPT, not discarded
    assert rec["all_verified"], rec["unverified"]


def test_frozen_stage_moved_from_refused_m2_tie_is_unverified_not_failure(
        tmp_path, monkeypatch):
    """Moving away from a refused tie is still not a frozen-solution regression
    -- there was no frozen solution -- but it is not a pass either."""
    rec_path = os.path.join(str(tmp_path), "checkpoint_m2_F212N_latest.json")
    with open(rec_path, "w") as fh:
        json.dump(dict(visits=[dict(visit="001", reference_tie=dict(
            apply_ok=False, dra_mas=-7694.20, ddec_mas=1436.12,
            off_mas=7827.1, swept=True))]), fh)
    _patch_consensus_and_tie(monkeypatch, dra_now=-30.83, ddec_now=9.80)
    rec = run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=_DUMMY_REFCAT,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert rec["passed"]                      # w51 F140M still unblocks
    assert rec["failures"] == []
    assert not rec["all_verified"]
    assert any("m2 MEASURED but REFUSED" in u for u in rec["unverified"])


def test_frozen_stage_m2_applied_tie_still_gates(tmp_path, monkeypatch):
    """The exemption is only for a REFUSED tie.  An m2 tie that WAS applied
    remains a real frozen baseline, and moving away from it still raises."""
    rec_path = os.path.join(str(tmp_path), "checkpoint_m2_F212N_latest.json")
    with open(rec_path, "w") as fh:
        json.dump(dict(visits=[dict(visit="001", reference_tie=dict(
            apply_ok=True, dra_mas=10.0, ddec_mas=0.0))]), fh)
    _patch_consensus_and_tie(monkeypatch, dra_now=40.0, ddec_now=0.0)
    with pytest.raises(AstrometryRegressionError):
        run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=_DUMMY_REFCAT,
                             filtername="F212N", record_dir=str(tmp_path),
                             context="test")


# ---------------------------------------------------------------------------
# frozen-stage PER-EXPOSURE DELTA gate (regression = a single exposure MOVED
# since the m2 freeze, NOT a nonzero absolute vs-consensus offset).  Reproduces
# brick F115W m3 (2026-07-20): 14 exposures 2.0-3.0 mas off consensus falsely
# raised because the frozen gate re-checked ABSOLUTE magnitude -- that magnitude
# is intrinsic per-exposure centroid scatter (the SAME 2-3 mas at m2), so the
# bluest/sparsest filter could never pass a frozen stage.
# ---------------------------------------------------------------------------

def _exp(key, dra, ddec, misaligned):
    return dict(key=key, n_reliable=50, raoffset_meta=0.0, deoffset_meta=0.0,
                component=0, internal_tie=True, unverified=False,
                misaligned=misaligned,
                vs_consensus=dict(dra=dra, ddec=ddec,
                                  off=float(np.hypot(dra, ddec)),
                                  dra_err=0.05, ddec_err=0.05, swept=False,
                                  npairs=200, contrast=50.0, ok=True,
                                  window_arcsec=3.0, n_peak=100))


def _patch_consensus_exposures(monkeypatch, exposures):
    coords = SkyCoord(ra=[RA0, RA0] * u.deg, dec=[DEC0, DEC0] * u.deg, frame="icrs")

    def _fake_consensus(tables, context="", **kw):
        return dict(coords=coords, mag=None, exposures=exposures,
                    anchor_key=("001", 1, "nrcb1", "F212N"),
                    scatter_mas=np.array([1.0]), consensus_ok=True, skipped=[])

    monkeypatch.setattr(_ac, "build_visit_consensus", _fake_consensus)


def _write_m2_exposure_baseline(record_dir, exp_offsets, visit="001",
                                filt="F212N"):
    """exp_offsets: list of (key_tuple, dra, ddec) recorded per-exposure at m2."""
    exps = [dict(key=list(k), dra=dra, ddec=ddec) for k, dra, ddec in exp_offsets]
    rec = dict(visits=[dict(visit=visit, exposures=exps)])
    with open(os.path.join(record_dir, f"checkpoint_m2_{filt}_latest.json"),
              "w") as fh:
        json.dump(rec, fh)


_K = ("001", 8, "nrca2", "F212N")


def test_frozen_perexposure_intrinsic_scatter_no_regression(tmp_path, monkeypatch):
    """m2 recorded the exposure 2.5 mas off consensus (intrinsic scatter,
    misaligned); m3 re-measures ~the same (2.6, 0.1) -> delta 0.14 <= tol ->
    STABLE, no raise.  The exact brick F115W m3 case: absolute >2 mas that never
    moved must not be a frozen-stage regression."""
    _write_m2_exposure_baseline(str(tmp_path), [(_K, 2.5, 0.0)])
    _patch_consensus_exposures(monkeypatch, [_exp(_K, 2.6, 0.1, misaligned=True)])
    rec = run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=None,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert rec["passed"]
    assert rec["failures"] == []


def test_frozen_perexposure_moved_raises(tmp_path, monkeypatch):
    """m2 froze the exposure at (0, 0) (aligned); it then MOVED to (5, 0) ->
    delta 5 > tol -> AstrometryRegressionError (a real per-exposure regression)."""
    _write_m2_exposure_baseline(str(tmp_path), [(_K, 0.0, 0.0)])
    _patch_consensus_exposures(monkeypatch, [_exp(_K, 5.0, 0.0, misaligned=True)])
    with pytest.raises(AstrometryRegressionError):
        run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=None,
                             filtername="F212N", record_dir=str(tmp_path),
                             context="test")


def test_frozen_perexposure_no_m2_baseline_raises(tmp_path, monkeypatch):
    """A misaligned exposure at a frozen stage with NO m2 per-exposure baseline
    (new/renamed frame): cannot prove it didn't move -> fail closed -> raise."""
    # record dir has no checkpoint_m2_F212N_latest.json
    _patch_consensus_exposures(monkeypatch, [_exp(_K, 3.0, 0.0, misaligned=True)])
    with pytest.raises(AstrometryRegressionError):
        run_visit_checkpoint([_tiny_visit_table()], "m4", refcat=None,
                             filtername="F212N", record_dir=str(tmp_path),
                             context="test")


def _write_m2_skipped(record_dir, skipped_keys, exp_offsets=(), visit="001",
                      filt="F212N"):
    """m2 record whose consensus deliberately EXCLUDED ``skipped_keys``."""
    exps = [dict(key=list(k), dra=dra, ddec=ddec) for k, dra, ddec in exp_offsets]
    rec = dict(visits=[dict(visit=visit, exposures=exps,
                            consensus=dict(skipped=[list(k)
                                                    for k in skipped_keys]))])
    with open(os.path.join(record_dir, f"checkpoint_m2_{filt}_latest.json"),
              "w") as fh:
        json.dump(rec, fh)


def test_frozen_perexposure_m2_skipped_is_unverified_not_regression(
        tmp_path, monkeypatch):
    """m2 SKIPPED the exposure from its consensus (too few reliable stars), so it
    has no frozen baseline by construction.  m3 measures it 16 mas off: that is
    its FIRST measurement, not a movement -> UNVERIFIED, no raise.

    arches F212N (2026-08-02): a snowball storm cut exposure 4's source count ~31%
    on all eight detectors, m2 skipped all eight and said so, and m3 then killed
    the m4-m8 chain over the defect m2 had already handled."""
    _write_m2_skipped(str(tmp_path), [_K])
    _patch_consensus_exposures(monkeypatch, [_exp(_K, -4.9, 15.3, misaligned=True)])
    rec = run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=None,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert rec["passed"]
    assert rec["failures"] == []
    # still held by the release gate's all_verified check
    assert not rec["all_verified"]
    assert any("m2 SKIPPED this exposure" in u for u in rec["unverified"])


def test_frozen_perexposure_unrelated_skip_still_raises(tmp_path, monkeypatch):
    """The skip list excuses only the exposures IN it.  A different exposure with
    no baseline is still an unexplained frozen-stage frame -> raise."""
    other = ("001", 9, "nrcb3", "F212N")
    _write_m2_skipped(str(tmp_path), [other])
    _patch_consensus_exposures(monkeypatch, [_exp(_K, 3.0, 0.0, misaligned=True)])
    with pytest.raises(AstrometryRegressionError):
        run_visit_checkpoint([_tiny_visit_table()], "m3", refcat=None,
                             filtername="F212N", record_dir=str(tmp_path),
                             context="test")


def test_m2_perexposure_scatter_still_corrects_not_raises(tmp_path, monkeypatch):
    """At m2 (correcting) the same 2.6 mas misaligned exposure is a CORRECTION,
    never a raise -- the frozen delta gate only governs m3+."""
    _patch_consensus_exposures(monkeypatch, [_exp(_K, 2.6, 0.1, misaligned=True)])
    rec = run_visit_checkpoint([_tiny_visit_table()], "m2", refcat=None,
                               filtername="F212N", record_dir=str(tmp_path),
                               context="test")
    assert rec["failures"] == []
    assert len(rec["corrections"]) == 1


# ---------------------------------------------------------------------------
# cross-filter checkpoint
# ---------------------------------------------------------------------------

def _crossfilter_catalogs(n=6000, extent=60.0, second_offset_mas=0.0,
                          patch=None, seed=5):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, extent, n)
    y = rng.uniform(0, extent, n)
    ra = RA0 + x / 3600.0 / COSD
    dec = DEC0 + y / 3600.0
    dra = np.full(n, second_offset_mas)
    if patch:
        lo_x, hi_x, lo_y, hi_y, dra_mas = patch
        inside = (x >= lo_x) & (x < hi_x) & (y >= lo_y) & (y < hi_y)
        dra[inside] += dra_mas

    def _tbl(ra_deg, dec_deg, noise_mas=0.5, rng=rng):
        t = Table()
        nn = len(ra_deg)
        t["skycoord"] = SkyCoord(
            ra=(ra_deg + rng.normal(0, noise_mas, nn) / 3.6e6 / COSD) * u.deg,
            dec=(dec_deg + rng.normal(0, noise_mas, nn) / 3.6e6) * u.deg,
            frame="icrs")
        t["flux_fit"] = rng.uniform(1e3, 1e5, nn)
        t["flux_err"] = t["flux_fit"] / 100.0
        t["qfit"] = rng.uniform(0.01, 0.05, nn)
        return t

    return {
        "F212N": _tbl(ra, dec),
        "F405N": _tbl(ra + dra / 3.6e6 / COSD, dec),
    }


def test_crossfilter_agreement_passes(tmp_path):
    cats = _crossfilter_catalogs()
    record = run_crossfilter_checkpoint(cats, record_dir=str(tmp_path),
                                        cell_min_stars=15, context="test")
    assert record["passed"]
    assert record["anchor_filter"] == "F212N"


def test_crossfilter_bulk_offset_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_CROSSFILTER_ASTROM_FAIL", raising=False)
    cats = _crossfilter_catalogs(second_offset_mas=12.0)
    with pytest.raises(CrossFilterAstrometryError):
        run_crossfilter_checkpoint(cats, record_dir=str(tmp_path),
                                   cell_min_stars=15, context="test")


def test_crossfilter_local_patch_fails(tmp_path, monkeypatch):
    monkeypatch.delenv("ALLOW_CROSSFILTER_ASTROM_FAIL", raising=False)
    # bulk agrees (~0) but a 15"x15" corner is 25 mas off in one filter --
    # exactly the overlap-region corruption the release keeps hitting
    cats = _crossfilter_catalogs(patch=(0.0, 15.0, 0.0, 15.0, 25.0))
    with pytest.raises(CrossFilterAstrometryError) as exc:
        run_crossfilter_checkpoint(cats, record_dir=str(tmp_path),
                                   cell_min_stars=15, cell_arcsec=5.0,
                                   context="test")
    assert "local" in str(exc.value)


def test_crossfilter_single_filter_skips(tmp_path):
    cats = {"F212N": _crossfilter_catalogs()["F212N"]}
    record = run_crossfilter_checkpoint(cats, record_dir=str(tmp_path))
    assert record["passed"]


# ---------------------------------------------------------------------------
# provenance cards
# ---------------------------------------------------------------------------

def test_provenance_header_cards_shape():
    cards = provenance_header_cards("m2", 12.5, -3.0, "visit-consensus",
                                    "VIRAC2+GaiaDR3", "/x/Offsets_test.csv")
    keys = [k for k, v, c in cards]
    assert keys == ["APROVST", "APROVMT", "APROVDR", "APROVDD", "APROVRF",
                    "APROVTB", "APROVDT"]
    d = {k: v for k, v, c in cards}
    assert d["APROVDR"] == 12.5
    assert d["APROVTB"] == "Offsets_test.csv"
    assert all(len(k) <= 8 for k in keys)


# ---------------------------------------------------------------------------
# Vgroup: exposure numbers restart per visit group
# ---------------------------------------------------------------------------

def _vgroup_csv(tmp_path):
    """Two visit groups of ONE visit, each with exposure 1 -- cloudc's shape."""
    rows = [dict(Filter="F212N", Module="nrcb1", Visit="jw02221002001",
                 Vgroup=vg, Exposure=1, dra=0.0, ddec=0.0)
            for vg in ("06201", "12201")]
    path = str(tmp_path / "Offsets_JWST_Brick2221_TEST.csv")
    Table(rows).write(path, overwrite=True)
    return path


def test_update_offsets_table_narrows_on_vgroup(tmp_path):
    """With a Vgroup column the two groups are separate rows, so each
    correction lands on its own instead of both summing onto one."""
    path = _vgroup_csv(tmp_path)
    corr = [dict(visit="jw02221002001", exposure=1, module="nrcb1",
                 filtername="F212N", vgroup=vg, dra_onsky_mas=0.0,
                 ddec_onsky_mas=mas, dec_deg=DEC_TEST)
            for vg, mas in (("06201", 100.0), ("12201", -50.0))]
    out = update_offsets_table(path, corr, "m2")
    # NB a CSV round-trip makes astropy infer int64 for a digit column, so the
    # zero-padded "06201" comes back as 6201 -- which is exactly why matching
    # goes through same_vgroup rather than str comparison.
    by = {int(str(r["Vgroup"])): r for r in out}
    assert by[6201]["ddec"] == pytest.approx(0.1, abs=1e-9)
    assert by[12201]["ddec"] == pytest.approx(-0.05, abs=1e-9)


def test_same_vgroup_survives_csv_int_coercion():
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import same_vgroup
    assert same_vgroup(6201, "06201")        # the CSV round-trip case
    assert same_vgroup("06201", "6201")
    assert same_vgroup("06201", "06201")
    assert not same_vgroup("06201", "12201")
    assert not same_vgroup(6201, 12201)


def test_vgroup_table_accepts_what_a_vgroupless_one_refuses(tmp_path):
    """The refusal added in #169 is lifted once the table can express it."""
    corr = [dict(visit="jw02221002001", exposure=1, module="nrcb1",
                 filtername="F212N", vgroup=vg, dra_onsky_mas=0.0,
                 ddec_onsky_mas=10.0, dec_deg=DEC_TEST)
            for vg in ("06201", "12201")]
    # Vgroup-less: still refused
    old = _offsets_csv(tmp_path)
    with pytest.raises(OffsetsTableUpdateError, match="(?i)more than one visit group"):
        update_offsets_table(old, corr, "m2")
    # Vgroup-carrying: applied
    new = _vgroup_csv(tmp_path)
    out = update_offsets_table(new, corr, "m2")
    assert all(r["ddec"] == pytest.approx(0.01, abs=1e-9) for r in out)


def test_lookup_consensus_offset_disambiguates_vgroups(tmp_path):
    """Without narrowing, two groups sharing exposure 1 raise 'match=2'."""
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        lookup_consensus_offset)
    t = Table([dict(Filter="F212N", Module="nrcb1", Visit="jw02221002001",
                    Vgroup=vg, Exposure=1,
                    **{"dra (arcsec)": d, "ddec (arcsec)": 0.0})
               for vg, d in (("06201", 0.01), ("12201", 0.02))])
    assert lookup_consensus_offset(t, "jw02221002001", 1, "nrcb1", "F212N",
                                   vgroup="06201")[0] == pytest.approx(0.01)
    assert lookup_consensus_offset(t, "jw02221002001", 1, "nrcb1", "F212N",
                                   vgroup="12201")[0] == pytest.approx(0.02)
    with pytest.raises(ValueError, match="match=2"):
        lookup_consensus_offset(t, "jw02221002001", 1, "nrcb1", "F212N")


def test_vgroup_key_normalises_csv_roundtrip_forms():
    """A CSV round-trip mangles this column two ways; both must key the same as
    the value that was written."""
    import numpy.ma as ma
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import vgroup_key
    assert vgroup_key("06201") == vgroup_key(6201) == vgroup_key("6201")
    # the BULK rows' empty cell comes back MASKED, str() == '--'
    assert vgroup_key(ma.masked) == ""
    assert vgroup_key("--") == vgroup_key("") == vgroup_key(None) == ""
    assert vgroup_key("06201") != vgroup_key("12201")


def test_update_offsets_table_refuses_a_vgroupless_correction_that_spans_groups(tmp_path):
    """A per-exposure correction with no visit group, on a table that HAS one:
    the shift would be added to every group's row (the accumulation
    _assert_vgroup_granularity refuses in the mirror case)."""
    path = _vgroup_csv(tmp_path)
    corr = [dict(visit="jw02221002001", exposure=1, module="nrcb1",
                 filtername="F212N", dra_onsky_mas=0.0, ddec_onsky_mas=10.0,
                 dec_deg=DEC_TEST)]
    with pytest.raises(OffsetsTableUpdateError, match="(?i)carries NO visit group"):
        update_offsets_table(path, corr, "m2")


def test_update_offsets_table_ignores_a_stringified_missing_vgroup(tmp_path):
    """exposure_key stringifies a missing VGROUP meta to the literal "None".
    That must read as "unknown", not as a token to narrow on -- narrowing on it
    would match no row and hard-fail the whole re-tie."""
    path = _offsets_csv(tmp_path)          # no Vgroup column
    corr = [dict(visit="jw01182004001", exposure=1, module="nrcb1",
                 filtername="F212N", vgroup="None", dra_onsky_mas=0.0,
                 ddec_onsky_mas=10.0, dec_deg=DEC_TEST)]
    out = update_offsets_table(path, corr, "m2")
    row = out[(np.array([str(v) for v in out["Visit"]]) == "jw01182004001")
              & (out["Exposure"] == 1)][0]
    assert row["ddec"] == pytest.approx(0.5 + 0.01, abs=1e-9)


def test_vgroup_row_matches_treats_empty_as_unknown():
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import vgroup_row_matches
    import numpy.ma as ma
    assert vgroup_row_matches("", "06201")
    assert vgroup_row_matches(ma.masked, "06201")
    assert vgroup_row_matches("06201", "6201")
    assert not vgroup_row_matches("12201", "06201")


def test_vgroup_key_keeps_a_non_digit_token_whole():
    """MIRI/parallel groups carry a trailing letter; truncating to the digit
    prefix would silently return a DIFFERENT group."""
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        same_vgroup, vgroup_key)
    assert vgroup_key("0210b") == "0210b"
    assert not same_vgroup("0210b", "0210")


def test_build_virac2_offsets_parses_whole_vgroup_token():
    from jwst_gc_pipeline.reduction.build_virac2_offsets import parse_vgroup
    assert parse_vgroup("f212n_nrcb1_visit001_vgroup06201_exp00001_m3_daophot_basic.fits") == "6201"
    # the zero-padded and bare spellings of one group key identically
    assert (parse_vgroup("f182m_nrcb3_visit001_vgroup07101_exp00003_m3_daophot_basic.fits")
            == parse_vgroup("f182m_nrcb3_visit001_vgroup7101_exp00003_m3_daophot_basic.fits"))
    # a non-digit token is kept whole, not truncated to its digit prefix
    assert parse_vgroup("f2550w_mirimage_visit001_vgroup0020210b_exp00001_m3_daophot_basic.fits") == "0020210b"


# ---------------------------------------------------------------------------
# module-FAMILY rows vs per-DETECTOR corrections (the sgrc/cloudc divergence)
# ---------------------------------------------------------------------------

def _module_family_csv(tmp_path):
    """sgrc/cloudc/cloudef shape AFTER --per-module rebuild: the table HAS a
    Module column, but its values are module FAMILIES -- SW filters carry
    nrca/nrcb, LW filters nrcalong/nrcblong.  The m2 consensus meanwhile emits
    one correction per DETECTOR (nrca1..nrca4)."""
    rows = []
    for filt, mods in (("F115W", ("nrca", "nrcb")),
                       ("F405N", ("nrcalong", "nrcblong"))):
        for mod in mods:
            for e in (1, 2):
                rows.append(dict(Visit="jw04147012001", Exposure=e,
                                 Filter=filt, Module=mod,
                                 **{"dra (arcsec)": 0.0, "ddec (arcsec)": 0.0}))
    path = str(tmp_path / "Offsets_JWST_Brick4147_VIRAC2locked.csv")
    Table(rows).write(path, overwrite=True)
    return path


def _detector_corrections(dets, filt="F115W", ddec=400.0, exposure=1):
    return [dict(visit="jw04147012001", exposure=exposure, module=m,
                 filtername=filt, dra_onsky_mas=0.0, ddec_onsky_mas=ddec,
                 dec_deg=DEC_TEST) for m in dets]


def test_refuses_detector_corrections_on_module_family_table(tmp_path):
    """4 detectors of one module all match its single family row and are SUMMED.

    The Module column EXISTS, so _assert_module_granularity returns early -- this
    is the case that guard could not see, and it is what made sgrc's table run
    away (185.7 -> 525.7 -> 1678.5 mas over three re-tie iterations).
    """
    path = _module_family_csv(tmp_path)
    before = open(path).read()
    corr = _detector_corrections(("nrca1", "nrca2", "nrca3", "nrca4"))
    with pytest.raises(OffsetsTableUpdateError,
                       match="(?i)more than one correction"):
        update_offsets_table(path, corr, "m2")
    assert open(path).read() == before


def test_pooling_collapses_detectors_to_the_family_row(tmp_path):
    """Pooled corrections apply 1:1, and the pooled value is the MEDIAN."""
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        pool_corrections_to_table_granularity)
    path = _module_family_csv(tmp_path)
    # residuals that largely cancel: median 3, sum 12 -- the whole point
    corr = []
    for m, d in (("nrca1", 1.0), ("nrca2", 5.0), ("nrca3", 3.0), ("nrca4", 3.0)):
        corr.extend(_detector_corrections((m,), ddec=d))
    pooled = pool_corrections_to_table_granularity(corr, path)
    assert len(pooled) == 1
    assert pooled[0]["ddec_onsky_mas"] == pytest.approx(3.0)
    assert pooled[0]["module"] == "nrca"
    assert "median of 4" in pooled[0]["source"]
    out = update_offsets_table(path, pooled, "m2")
    hit = out[(out["Module"] == "nrca") & (out["Exposure"] == 1)]
    assert len(hit) == 1
    assert hit[0]["ddec (arcsec)"] == pytest.approx(0.003)
    # the other module's row is untouched
    other = out[(out["Module"] == "nrcb") & (out["Exposure"] == 1)]
    assert other[0]["ddec (arcsec)"] == pytest.approx(0.0)


def test_pooled_sum_would_have_been_four_times_the_median(tmp_path):
    """Pin the actual divergence factor: summing 4 detectors over-corrects ~4x."""
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        pool_corrections_to_table_granularity)
    path = _module_family_csv(tmp_path)
    corr = _detector_corrections(("nrca1", "nrca2", "nrca3", "nrca4"), ddec=10.0)
    summed = sum(c["ddec_onsky_mas"] for c in corr)
    pooled = pool_corrections_to_table_granularity(corr, path)
    assert summed == pytest.approx(40.0)
    assert pooled[0]["ddec_onsky_mas"] == pytest.approx(10.0)


def test_lw_correction_does_not_leak_onto_the_sw_row(tmp_path):
    """_module_variants maps 'nrcalong' -> {'nrcalong', 'nrca'} (READ semantics).

    In the WRITE direction that adds an LW correction to the SW nrca row as
    well.  The apply path must resolve the module against the values the table
    carries FOR THIS FILTER, so an F405N correction can only touch LW rows.
    """
    path = _module_family_csv(tmp_path)
    out = update_offsets_table(path, _detector_corrections(
        ("nrcalong",), filt="F405N", ddec=400.0), "m2")
    lw = out[(out["Module"] == "nrcalong") & (out["Exposure"] == 1)]
    sw = out[(out["Module"] == "nrca") & (out["Exposure"] == 1)]
    assert lw[0]["ddec (arcsec)"] == pytest.approx(0.4)
    assert sw[0]["ddec (arcsec)"] == pytest.approx(0.0)


def test_bare_module_token_on_an_lw_filter_hits_the_long_row(tmp_path):
    """cloudef's checkpoint emits module='nrcb' (the CLI token) for LW filters.

    The family is unambiguous once the filter is known -- F405N rows are all
    LW -- so it must land on nrcblong, never on the SW nrcb row.
    """
    path = _module_family_csv(tmp_path)
    out = update_offsets_table(path, _detector_corrections(
        ("nrcb",), filt="F405N", ddec=400.0), "m2")
    lw = out[(out["Module"] == "nrcblong") & (out["Exposure"] == 1)]
    sw = out[(out["Module"] == "nrcb") & (out["Exposure"] == 1)]
    assert lw[0]["ddec (arcsec)"] == pytest.approx(0.4)
    assert sw[0]["ddec (arcsec)"] == pytest.approx(0.0)


def test_pooling_is_a_noop_at_matching_granularity(tmp_path):
    """A table whose rows ARE detectors keeps every correction distinct."""
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        pool_corrections_to_table_granularity)
    rows = [dict(Visit="jw04147012001", Exposure=1, Filter="F115W", Module=m,
                 **{"dra (arcsec)": 0.0, "ddec (arcsec)": 0.0})
            for m in ("nrca1", "nrca2", "nrca3", "nrca4")]
    path = str(tmp_path / "Offsets_JWST_Brick4147_VIRAC2locked.csv")
    Table(rows).write(path, overwrite=True)
    corr = _detector_corrections(("nrca1", "nrca2", "nrca3", "nrca4"), ddec=7.0)
    pooled = pool_corrections_to_table_granularity(corr, path)
    assert len(pooled) == 4
    assert [c["module"] for c in pooled] == ["nrca1", "nrca2", "nrca3", "nrca4"]
    out = update_offsets_table(path, pooled, "m2")
    assert all(r["ddec (arcsec)"] == pytest.approx(0.007) for r in out)


def test_different_exposures_are_never_pooled_together(tmp_path):
    """Pooling groups by the MATCHED ROW SET, so exposures stay independent."""
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        pool_corrections_to_table_granularity)
    path = _module_family_csv(tmp_path)
    corr = (_detector_corrections(("nrca1", "nrca2"), ddec=10.0, exposure=1)
            + _detector_corrections(("nrca1", "nrca2"), ddec=30.0, exposure=2))
    pooled = pool_corrections_to_table_granularity(corr, path)
    assert len(pooled) == 2
    by_exp = {c["exposure"]: c["ddec_onsky_mas"] for c in pooled}
    assert by_exp[1] == pytest.approx(10.0)
    assert by_exp[2] == pytest.approx(30.0)


def test_bulk_correction_composes_with_per_exposure_ones(tmp_path):
    """A whole-visit bulk tie touches every row and must NOT trip the guard.

    ``exposure=None, module=None`` is the consensus->reference shift; it is
    broad by design and composes with each exposure's jitter (that is exactly
    the BULK-row + jitter-row sum lookup_consensus_offset performs).  Treating
    it as a collision would refuse every real checkpoint: sgrc F212N/F360M/
    F480M and cloudc F182M all carry one.
    """
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        pool_corrections_to_table_granularity)
    path = _module_family_csv(tmp_path)
    bulk = dict(visit="jw04147012001", exposure=None, module=None,
                filtername="F115W", dra_onsky_mas=0.0, ddec_onsky_mas=100.0,
                dec_deg=DEC_TEST, source="m2 consensus->reference")
    corr = [bulk] + _detector_corrections(("nrca1", "nrca2"), ddec=10.0)
    pooled = pool_corrections_to_table_granularity(corr, path)
    assert len(pooled) == 2                    # bulk passed through + 1 pooled
    assert any(c["module"] is None for c in pooled)
    out = update_offsets_table(path, pooled, "m2")   # must not raise
    # exposure 1 nrca got bulk + pooled jitter; exposure 2 nrca got bulk only
    e1 = out[(out["Module"] == "nrca") & (out["Exposure"] == 1)][0]
    e2 = out[(out["Module"] == "nrca") & (out["Exposure"] == 2)][0]
    assert e1["ddec (arcsec)"] == pytest.approx(0.110)
    assert e2["ddec (arcsec)"] == pytest.approx(0.100)


# --- pooling refusals: what pooling must NOT silently absorb -----------------

def _pool(corr, path, **kw):
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        pool_corrections_to_table_granularity)
    return pool_corrections_to_table_granularity(corr, path, **kw)


def test_pooling_refuses_to_average_across_module_families(tmp_path):
    """A Module-LESS table lands every module on one row (sgrb2's shape).

    Pooling there would convert _assert_module_granularity's actionable
    "rebuild the table --per-module" refusal into a silent A/B-averaged shift.
    The justification for pooling is that detectors sit at fixed SIAF positions
    WITHIN one module; it does not extend to medianing module A against B.
    """
    path = _module_less_csv(tmp_path)
    before = open(path).read()
    corr = _detector_corrections(("nrca1", "nrca2", "nrcb1", "nrcb2"))
    with pytest.raises(OffsetsTableUpdateError,
                       match="(?i)module families|per-module"):
        _pool(corr, path)
    with pytest.raises(OffsetsTableUpdateError):
        update_offsets_table(path, corr, "m2", pool=True)
    assert open(path).read() == before


def test_pooling_refuses_repeated_module_in_one_group(tmp_path):
    """Two corrections for ONE module on one row are not its detectors.

    They are two physically distinct things the table cannot tell apart --
    typically two visit groups against a Vgroup-less table (sgrb2's records
    carry no vgroup at all).  Pooling must not absorb what the vgroup guard
    exists to stop.
    """
    path = _module_family_csv(tmp_path)
    corr = _detector_corrections(("nrca1",), ddec=10.0) + \
        _detector_corrections(("nrca1",), ddec=90.0)
    with pytest.raises(OffsetsTableUpdateError,
                       match="(?i)more than one correction|Vgroup"):
        _pool(corr, path)


def test_magnitude_ceiling_sees_members_not_the_median(tmp_path):
    """A blown-up detector must stop the run, not be averaged out of existence.

    median <= max, so pooling cannot inflate past the ceiling -- the risk runs
    the other way.  Un-pooled this raises; pooled it used to become 2.5 mas.
    """
    path = _module_family_csv(tmp_path)
    corr = []
    for m, d in (("nrca1", 2.0), ("nrca2", 2.0), ("nrca3", 3.0),
                 ("nrca4", 30000.0)):
        corr.extend(_detector_corrections((m,), ddec=d))
    with pytest.raises(OffsetsTableUpdateError, match="(?i)magnitude limit"):
        _pool(corr, path)


def test_pooling_refuses_a_bimodal_group(tmp_path, monkeypatch):
    """A group whose members disagree wildly has no meaningful middle."""
    monkeypatch.setenv("ASTROM_MAX_POOL_SPREAD_MAS", "20")
    path = _module_family_csv(tmp_path)
    corr = []
    for m, d in (("nrca1", 1.0), ("nrca2", 1.0), ("nrca3", 100.0),
                 ("nrca4", 100.0)):
        corr.extend(_detector_corrections((m,), ddec=d))
    with pytest.raises(OffsetsTableUpdateError, match="(?i)peak-to-peak"):
        _pool(corr, path)


def test_pooled_entry_carries_its_dispersion(tmp_path):
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        pool_corrections_to_table_granularity)
    path = _module_family_csv(tmp_path)
    corr = []
    for m, d in (("nrca1", 1.0), ("nrca2", 5.0), ("nrca3", 3.0), ("nrca4", 3.0)):
        corr.extend(_detector_corrections((m,), ddec=d))
    pooled = pool_corrections_to_table_granularity(corr, path)[0]
    assert pooled["pooled_n"] == 4
    assert pooled["pooled_spread_mas"] == pytest.approx(4.0)
    assert pooled["pooled_stat"] == "median"
    assert "ptp 4.00mas" in pooled["source"]


def test_unknown_pool_stat_raises_rather_than_becoming_the_mean(tmp_path):
    path = _module_family_csv(tmp_path)
    corr = _detector_corrections(("nrca1", "nrca2"), ddec=10.0)
    with pytest.raises(ValueError, match="(?i)pool stat"):
        _pool(corr, path, stat="medain")


def test_partial_row_set_overlap_is_refused(tmp_path):
    """Two corrections whose matched row-sets overlap but differ cannot pool."""
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        pool_corrections_to_table_granularity)
    rows = [dict(Visit="jw04147012001", Exposure=1, Filter="F115W", Module=m,
                 **{"dra (arcsec)": 0.0, "ddec (arcsec)": 0.0})
            for m in ("nrca", "nrca1")]
    path = str(tmp_path / "Offsets_JWST_Brick4147_VIRAC2locked.csv")
    Table(rows).write(path, overwrite=True)
    # 'nrca1' matches its exact row only; a bare 'nrca' matches the family row
    # only -- but 'nrca2' (absent) falls back to BOTH by family.
    corr = _detector_corrections(("nrca1",)) + _detector_corrections(("nrca2",))
    with pytest.raises(OffsetsTableUpdateError,
                       match="(?i)row-sets|partial overlap"):
        pool_corrections_to_table_granularity(corr, path)


def test_mixed_channel_rows_for_one_filter_are_refused(tmp_path):
    """The single-channel-per-filter invariant is enforced, not just assumed."""
    rows = [dict(Visit="jw04147012001", Exposure=1, Filter="F115W", Module=m,
                 **{"dra (arcsec)": 0.0, "ddec (arcsec)": 0.0})
            for m in ("nrca", "nrcalong")]
    path = str(tmp_path / "Offsets_JWST_Brick4147_VIRAC2locked.csv")
    Table(rows).write(path, overwrite=True)
    with pytest.raises(OffsetsTableUpdateError, match="(?i)mix LW and SW"):
        update_offsets_table(path, _detector_corrections(("nrca1",)), "m2")


def test_update_offsets_table_pool_flag_applies_the_collapse(tmp_path):
    """pool=True performs the collapse the guard's message names."""
    path = _module_family_csv(tmp_path)
    corr = _detector_corrections(("nrca1", "nrca2", "nrca3", "nrca4"), ddec=10.0)
    with pytest.raises(OffsetsTableUpdateError):
        update_offsets_table(path, corr, "m2")            # strict by default
    out = update_offsets_table(path, corr, "m2", pool=True)
    hit = out[(out["Module"] == "nrca") & (out["Exposure"] == 1)]
    assert hit[0]["ddec (arcsec)"] == pytest.approx(0.010)   # median, not 4x
