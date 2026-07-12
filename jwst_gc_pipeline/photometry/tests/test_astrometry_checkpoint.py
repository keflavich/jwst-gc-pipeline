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
from .test_visit_consensus import RA0, DEC0, COSD, _field, _visit_tables

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


def test_update_offsets_table_refuses_visit_collapse(tmp_path):
    # a correction that lands two visits on the SAME value is the brick-1182
    # collapse signature -- must refuse to write
    path = _offsets_csv(tmp_path)
    with pytest.raises(OffsetsTableUpdateError):
        update_offsets_table(path, [dict(
            visit="jw01182004001", exposure=None, module=None,
            filtername="F212N",
            dra_onsky_mas=(1.9 - (-17.5)) * 1000.0 * COSD,
            ddec_onsky_mas=0.0, dec_deg=DEC_TEST)], "m2")


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
