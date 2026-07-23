"""Tests for the per-detector affine (v1: translation) tie (detector_tie.py).

Synthetic multi-detector visits only -- no data dependencies.  Behaviors under
test (the ones that could silently corrupt astrometry if wrong):
  * injected per-detector shifts are measured and recovered (differentially --
    the consensus zero point is internal);
  * the refusal floor (n_pairs / significance / apply floor) leaves a detector
    UNCORRECTED rather than guessing;
  * offsets-table round trip: detector rows (Exposure=-1, Module=<detector>)
    seed + lookup, summed with bulk and jitter rows, exact-detector REPLACES
    module-family (never sums);
  * update_offsets_table guards: no Module column -> refuse; family+detector
    rows -> the correction lands on the exact row only;
  * run_visit_checkpoint integration: default OFF; when ON, the per-exposure
    jitter corrections are residuals AFTER the detector tie (no double
    correction).
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry.detector_tie import (
    DETECTOR_TIE_MIN_PAIRS, detector_tie_corrections,
    group_frames_by_detector, measure_detector_tie,
    measure_visit_detector_ties, per_detector_tie_enabled,
)
from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    BULK_EXPOSURE, BULK_MODULE, OffsetsTableUpdateError,
    lookup_consensus_offset, run_visit_checkpoint,
    seed_offsets_table_from_consensus, update_offsets_table,
)
from jwst_gc_pipeline.photometry.visit_consensus import build_visit_consensus

RA0, DEC0 = 266.5, -28.7
COSD = np.cos(np.radians(DEC0))
ENV = "ASTROM_M2_PER_DETECTOR_TIE"


def _field(n=400, extent_arcsec=90.0, seed=42):
    rng = np.random.default_rng(seed)
    ra = RA0 + rng.uniform(0, extent_arcsec, n) / 3600.0 / COSD
    dec = DEC0 + rng.uniform(0, extent_arcsec, n) / 3600.0
    return ra, dec


def _exposure_table(ra, dec, visit="001", exposure=1, module="nrcb1",
                    filtername="F212N", dra_mas=0.0, ddec_mas=0.0,
                    noise_mas=1.0, seed=0):
    rng = np.random.default_rng(1000 + seed)
    n = len(ra)
    ra_obs = ra + (dra_mas + rng.normal(0, noise_mas, n)) / 3.6e6 / COSD
    dec_obs = dec + (ddec_mas + rng.normal(0, noise_mas, n)) / 3.6e6
    tbl = Table()
    tbl["skycoord"] = SkyCoord(ra=ra_obs * u.deg, dec=dec_obs * u.deg,
                               frame="icrs")
    tbl["flux_fit"] = rng.uniform(1e3, 1e5, n)
    tbl["flux_err"] = tbl["flux_fit"] / 100.0
    tbl["qfit"] = rng.uniform(0.01, 0.05, n)
    tbl.meta.update(VISIT=visit, EXPOSURE=f"{exposure:05d}", MODULE=module,
                    FILTER=filtername, RAOFFSET=0.0, DEOFFSET=0.0)
    return tbl


def _detector_visit(det_shifts, n_exp=4, n=400, noise_mas=1.0):
    """Synthetic visit: each detector observes the SAME star field (so every
    detector is tied through the consensus), shifted by its injected
    placement error.  det_shifts: detector -> (dra_mas, ddec_mas)."""
    ra, dec = _field(n=n)
    tables = []
    seed = 0
    for det, (dra, ddec) in det_shifts.items():
        for e in range(1, n_exp + 1):
            seed += 1
            tables.append(_exposure_table(
                ra, dec, exposure=e, module=det, dra_mas=dra, ddec_mas=ddec,
                noise_mas=noise_mas, seed=seed))
    return tables


SHIFTS = {"nrca3": (+3.0, -5.0), "nrcb4": (-2.5, +2.5), "nrcb1": (0.0, 0.0)}


def _measure_ties(refcat=None, det_shifts=SHIFTS, **kwargs):
    tables = _detector_visit(det_shifts)
    cons = build_visit_consensus(tables, context="dettie-test")
    from jwst_gc_pipeline.photometry.visit_consensus import (
        catalog_coords, exposure_key, select_reliable_stars)
    frames = group_frames_by_detector(
        (exposure_key(t), catalog_coords(t)[select_reliable_stars(t)], None)
        for t in tables)
    assert set(frames) == set(det_shifts)
    return measure_visit_detector_ties(frames, cons["coords"], refcat=refcat,
                                       context="dettie-test", **kwargs)


def test_injected_detector_shifts_recovered():
    ties = _measure_ties()
    # the consensus zero point is internal -> assert the DIFFERENTIALS.
    # measured tie = (consensus - detector) = -shift + common zero point.
    for da, db in (("nrca3", "nrcb4"), ("nrca3", "nrcb1"), ("nrcb4", "nrcb1")):
        ddra = ties[da]["dra_mas"] - ties[db]["dra_mas"]
        dddec = ties[da]["ddec_mas"] - ties[db]["ddec_mas"]
        exp_dra = -(SHIFTS[da][0] - SHIFTS[db][0])
        exp_ddec = -(SHIFTS[da][1] - SHIFTS[db][1])
        assert ddra == pytest.approx(exp_dra, abs=0.7), (da, db)
        assert dddec == pytest.approx(exp_ddec, abs=0.7), (da, db)
    # the shifted detectors are significant, well-measured, applying
    for det in ("nrca3", "nrcb4"):
        assert ties[det]["apply"], ties[det]
        assert ties[det]["n_pairs"] >= DETECTOR_TIE_MIN_PAIRS
        assert ties[det]["sem_mas"] < 1.0


def test_reference_gross_crosscheck_passes_on_consistent_refcat():
    """A VIRAC2-like refcat offset by a pure visit bulk must not veto the
    internal ties (the cross-check compares detector DIFFERENTIALS)."""
    ra, dec = _field()
    ref = SkyCoord(ra=(ra + 10.0 / 3.6e6 / COSD) * u.deg,
                   dec=(dec - 4.0 / 3.6e6) * u.deg, frame="icrs")
    ties = _measure_ties(refcat=dict(all=ref, sparse=None))
    for det in ("nrca3", "nrcb4"):
        assert ties[det]["apply"], ties[det]
        assert ties[det]["vs_reference"] is not None
        assert ties[det]["vs_reference"]["split_mas"] < 5.0


def test_floor_refusal_too_few_stars():
    """A detector below the combined-pair floor is REFUSED (left
    uncorrected), never guessed."""
    tables = _detector_visit(SHIFTS)
    cons = build_visit_consensus(tables, context="dettie-floor")
    # one frame with 40 stars drawn FROM the consensus field (so a coherent
    # peak exists) -> n_pairs ~40 < 50 floor
    ra, dec = _field()
    tiny = SkyCoord(ra=ra[:40] * u.deg, dec=dec[:40] * u.deg, frame="icrs")
    rec = measure_detector_tie([(tiny, None)], cons["coords"], context="tiny")
    assert not rec["measurable"]
    assert rec["refuse_reason"]
    ties = measure_visit_detector_ties({"nrca1": [(tiny, None)]},
                                       cons["coords"])
    assert not ties["nrca1"]["apply"]


def test_subfloor_tie_not_applied():
    """A real but sub-floor (<1 mas) placement must not be corrected."""
    ties = _measure_ties(det_shifts={"nrca1": (0.3, -0.2),
                                     "nrcb1": (0.0, 0.0),
                                     "nrcb2": (0.0, 0.0)})
    assert not ties["nrca1"]["apply"]
    assert "apply floor" in str(ties["nrca1"]["refuse_reason"]) \
        or "not significant" in str(ties["nrca1"]["refuse_reason"])


def test_detector_tie_corrections_schema():
    """Only APPLYING detectors are emitted, with the per-detector row scope
    (exposure=None + module=<detector>) and provenance in ``source``."""
    ties = {
        "nrca3": dict(apply=True, dra_mas=-3.0, ddec_mas=5.0, n_pairs=1200,
                      sem_mas=0.11),
        "nrcb1": dict(apply=False, dra_mas=0.2, ddec_mas=0.1, n_pairs=1500,
                      sem_mas=0.10, refuse_reason="|tie| below floor"),
    }
    corrs = detector_tie_corrections(ties, "001", "F212N", DEC0, stage="m2")
    assert len(corrs) == 1
    c = corrs[0]
    assert c["exposure"] is None
    assert c["module"] == "nrca3"
    assert c["dra_onsky_mas"] == pytest.approx(-3.0)
    assert c["ddec_onsky_mas"] == pytest.approx(5.0)
    assert "per-detector tie" in c["source"]


def test_group_frames_by_detector():
    a = SkyCoord(ra=[266.5] * u.deg, dec=[-28.7] * u.deg, frame="icrs")
    b = SkyCoord(ra=[266.6, 266.7] * u.deg, dec=[-28.6, -28.5] * u.deg,
                 frame="icrs")
    groups = group_frames_by_detector([
        (("001", 1, "NRCA3", "F212N"), a, None),
        (("001", 2, "nrca3", "F212N"), b, {"dra": 1.0}),
        (("001", 1, "nrcb1", "F212N"), a, None),
    ])
    assert set(groups) == {"nrca3", "nrcb1"}
    assert len(groups["nrca3"]) == 2
    assert groups["nrca3"][1][1] == {"dra": 1.0}
    assert len(groups["nrcb1"]) == 1


# ---------------------------------------------------------------------------
# offsets-table round trip
# ---------------------------------------------------------------------------

def _seed_table(tmp_path, corrections):
    return seed_offsets_table_from_consensus(
        str(tmp_path), "4147", "003", corrections, stage="m2")


def test_detector_row_seed_and_lookup_roundtrip(tmp_path):
    corrections = [
        # per-visit bulk (consensus -> VIRAC2)
        dict(visit="001", exposure=None, module=None, filtername="F212N",
             dra_onsky_mas=10.0, ddec_onsky_mas=-4.0, dec_deg=DEC0,
             source="m2 consensus->reference"),
        # per-DETECTOR tie
        dict(visit="001", exposure=None, module="nrca3", filtername="F212N",
             dra_onsky_mas=-3.0, ddec_onsky_mas=5.0, dec_deg=DEC0,
             source="m2 per-detector tie"),
        # per-exposure jitter (residual after the detector row)
        dict(visit="001", exposure=5, module="nrca3", filtername="F212N",
             dra_onsky_mas=2.0, ddec_onsky_mas=0.0, dec_deg=DEC0,
             source="m2 visit-consensus"),
    ]
    path = _seed_table(tmp_path, corrections)
    tbl = Table.read(path)
    det_rows = tbl[(tbl["Exposure"] == BULK_EXPOSURE)
                   & (tbl["Module"] != BULK_MODULE)]
    assert len(det_rows) == 1
    assert str(det_rows["Module"][0]) == "nrca3"

    visit_tok = "jw04147003001"
    # frame of the tied detector, flagged exposure: jitter + detector + bulk
    dra, ddec = lookup_consensus_offset(tbl, visit_tok, 5, "nrca3", "F212N")
    assert dra * 1000 * COSD == pytest.approx(10.0 - 3.0 + 2.0, abs=1e-6)
    assert ddec * 1000 == pytest.approx(-4.0 + 5.0 + 0.0, abs=1e-6)
    # unflagged exposure of the tied detector: detector + bulk only
    dra, ddec = lookup_consensus_offset(tbl, visit_tok, 6, "nrca3", "F212N")
    assert dra * 1000 * COSD == pytest.approx(10.0 - 3.0, abs=1e-6)
    assert ddec * 1000 == pytest.approx(-4.0 + 5.0, abs=1e-6)
    # other detector: bulk only
    dra, ddec = lookup_consensus_offset(tbl, visit_tok, 5, "nrcb1", "F212N")
    assert dra * 1000 * COSD == pytest.approx(10.0, abs=1e-6)
    assert ddec * 1000 == pytest.approx(-4.0, abs=1e-6)


def test_exact_detector_row_replaces_family_row():
    """A frame matching BOTH a detector row and a module-family row must use
    the exact-detector row ALONE (REPLACES, never sums); a frame with only
    the family row uses the family value."""
    tbl = Table(rows=[
        {"Visit": "jw04147003001", "Filter": "F212N",
         "Exposure": BULK_EXPOSURE, "Module": "nrca",
         "dra (arcsec)": 0.010, "ddec (arcsec)": 0.0},
        {"Visit": "jw04147003001", "Filter": "F212N",
         "Exposure": BULK_EXPOSURE, "Module": "nrca1",
         "dra (arcsec)": -0.003, "ddec (arcsec)": 0.002},
    ])
    dra, ddec = lookup_consensus_offset(tbl, "jw04147003001", 3, "nrca1", "F212N")
    assert dra == pytest.approx(-0.003, abs=1e-9)   # exact row only, not summed
    assert ddec == pytest.approx(0.002, abs=1e-9)
    dra, ddec = lookup_consensus_offset(tbl, "jw04147003001", 3, "nrca2", "F212N")
    assert dra == pytest.approx(0.010, abs=1e-9)    # family row
    assert ddec == pytest.approx(0.0, abs=1e-9)


def test_seed_refuses_gross_detector_row(tmp_path):
    with pytest.raises(OffsetsTableUpdateError):
        _seed_table(tmp_path, [
            dict(visit="001", exposure=None, module="nrca3", filtername="F212N",
                 dra_onsky_mas=600.0, ddec_onsky_mas=0.0, dec_deg=DEC0),
        ])  # 600 mas: trips the mas-scale sanity guard (>0.5")


def test_update_offsets_table_refuses_module_without_column(tmp_path):
    path = str(tmp_path / "Offsets_locked.csv")
    Table(rows=[{"Visit": "jw02221001001", "Filter": "F212N",
                 "dra (arcsec)": 0.1, "ddec (arcsec)": 0.0}]).write(path)
    with pytest.raises(OffsetsTableUpdateError, match="no.*Module column"):
        update_offsets_table(path, [dict(
            visit="001", exposure=None, module="nrca3", filtername="F212N",
            dra_onsky_mas=-5.0, ddec_onsky_mas=2.0, dec_deg=DEC0)], "m2")


def test_update_offsets_table_exact_module_preference(tmp_path):
    """With family + exact-detector rows present, a detector-scoped correction
    lands ONLY on the exact row (REPLACES semantics at authoring time)."""
    path = str(tmp_path / "Offsets_locked.csv")
    Table(rows=[
        {"Visit": "jw02221001001", "Filter": "F212N", "Module": "nrca",
         "dra (arcsec)": 0.1, "ddec (arcsec)": 0.0},
        {"Visit": "jw02221001001", "Filter": "F212N", "Module": "nrca3",
         "dra (arcsec)": 0.1, "ddec (arcsec)": 0.0},
    ]).write(path)
    out = update_offsets_table(path, [dict(
        visit="001", exposure=None, module="nrca3", filtername="F212N",
        dra_onsky_mas=-5.0, ddec_onsky_mas=2.0, dec_deg=DEC0)], "m2")
    fam = out[np.asarray(out["Module"]) == "nrca"]
    det = out[np.asarray(out["Module"]) == "nrca3"]
    assert float(fam["dra (arcsec)"][0]) == pytest.approx(0.1, abs=1e-9)
    assert float(det["dra (arcsec)"][0]) == pytest.approx(
        0.1 - 0.005 / COSD, abs=1e-9)
    assert float(det["prov_dra_added_mas"][0]) == pytest.approx(-5.0)


# ---------------------------------------------------------------------------
# checkpoint integration
# ---------------------------------------------------------------------------

def test_checkpoint_default_off(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert not per_detector_tie_enabled()
    tables = _detector_visit({"nrca3": (6.0, 0.0), "nrcb1": (0.0, 0.0),
                              "nrcb2": (0.0, 0.0)})
    record = run_visit_checkpoint(tables, "m2", refcat=None,
                                  filtername="F212N", context="dettie-off")
    assert record["visits"][0]["detector_ties"] is None
    # without the tie, every nrca3 frame is individually flagged
    det_corr = [c for c in record["corrections"]
                if c.get("exposure") is None and c.get("module") is not None]
    assert not det_corr
    jitter = [c for c in record["corrections"] if c.get("exposure") is not None]
    assert len(jitter) == 4
    assert all(c["module"] == "nrca3" for c in jitter)


def test_checkpoint_detector_tie_no_double_correction(monkeypatch):
    """Flag ON: ONE detector row absorbs the 6 mas placement; the per-exposure
    jitter corrections become residuals (~0 -> none emitted).  The applied sum
    per frame (jitter + detector) equals the frame's measured offset once."""
    monkeypatch.setenv(ENV, "1")
    tables = _detector_visit({"nrca3": (6.0, 0.0), "nrcb1": (0.0, 0.0),
                              "nrcb2": (0.0, 0.0)})
    record = run_visit_checkpoint(tables, "m2", refcat=None,
                                  filtername="F212N", context="dettie-on")
    det_corr = [c for c in record["corrections"]
                if c.get("exposure") is None and c.get("module") is not None]
    assert len(det_corr) == 1
    assert det_corr[0]["module"] == "nrca3"
    assert det_corr[0]["dra_onsky_mas"] == pytest.approx(-6.0, abs=1.0)
    assert det_corr[0]["ddec_onsky_mas"] == pytest.approx(0.0, abs=1.0)
    # jitter corrections are residuals AFTER the detector row: the injected
    # shift is purely a detector term, so no frame should still exceed 2 mas
    jitter = [c for c in record["corrections"] if c.get("exposure") is not None]
    assert not jitter, jitter
    ties = record["visits"][0]["detector_ties"]
    assert ties["nrca3"]["apply"]
    assert not ties["nrcb1"]["apply"]
    assert not ties["nrcb2"]["apply"]


def test_checkpoint_frozen_stage_never_measures_ties(monkeypatch, tmp_path):
    """m3+ is FROZEN: even with the flag on, no detector ties are measured and
    no corrections are emitted (the movement gate vs the m2 baseline governs,
    unchanged)."""
    monkeypatch.setenv(ENV, "1")
    monkeypatch.setenv("ALLOW_LATE_STAGE_ASTROM_SHIFT", "1")  # isolate the gate
    tables = _detector_visit({"nrca3": (6.0, 0.0), "nrcb1": (0.0, 0.0),
                              "nrcb2": (0.0, 0.0)})
    record = run_visit_checkpoint(tables, "m3", refcat=None,
                                  filtername="F212N", context="dettie-m3",
                                  record_dir=str(tmp_path))
    assert record["visits"][0]["detector_ties"] is None
    assert not record["corrections"]
