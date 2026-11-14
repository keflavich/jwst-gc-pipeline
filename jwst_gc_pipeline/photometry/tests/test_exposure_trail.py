"""Tests for the trailed-exposure reading (exposure_trail.py, issue #733).

Synthetic frames only: Gaussian stars rendered into a small FITS file with a
rotated TAN WCS, optionally smeared along one pixel direction.  The grading
tests use hand-made shape records so that they pin the decision rules without
depending on the moment estimator.
"""
import math

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry import astrometry_checkpoint as ac
from jwst_gc_pipeline.photometry import exposure_trail as et
from jwst_gc_pipeline.photometry.astrometry_checkpoint import run_visit_checkpoint

from jwst_gc_pipeline.frame_wcs import MissingGwcsWarning

from .test_visit_consensus import _visit_tables

# The synthetic frames are plain FITS images with no GWCS.
pytestmark = pytest.mark.filterwarnings(
    f"ignore::{MissingGwcsWarning.__module__}.{MissingGwcsWarning.__name__}")

SIZE = 360
PSF_SIGMA = 1.6


def _write_frame(path, positions, trail_px=0.0, trail_theta=0.0, sigma=PSF_SIGMA,
                 roll_deg=0.0, seed=0):
    """Render Gaussian stars (smeared along ``trail_theta`` by ``trail_px``)."""
    rng = np.random.default_rng(seed)
    img = rng.normal(0.0, 0.02, (SIZE, SIZE)) + 1.0
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(float)
    n_sub = 25
    offsets = (np.linspace(-0.5, 0.5, n_sub) * trail_px) if trail_px else np.zeros(1)
    for x0, y0 in positions:
        for off in offsets:
            xs = x0 + off * math.cos(trail_theta)
            ys = y0 + off * math.sin(trail_theta)
            img += (100.0 / len(offsets)) * np.exp(
                -((xx - xs) ** 2 + (yy - ys) ** 2) / (2 * sigma ** 2))
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [266.5, -28.7]
    wcs.wcs.crpix = [SIZE / 2, SIZE / 2]
    scale = 0.031 / 3600.0
    c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    wcs.wcs.cd = scale * np.array([[-c, s], [s, c]])
    hdr = wcs.to_header()
    hdul = fits.HDUList([fits.PrimaryHDU(),
                         fits.ImageHDU(img.astype("f4"), header=hdr, name="SCI")])
    hdul.writeto(path, overwrite=True)
    return wcs


def _star_grid(seed=1):
    rng = np.random.default_rng(seed)
    gx, gy = np.meshgrid(np.arange(30, SIZE - 30, 50), np.arange(30, SIZE - 30, 50))
    pos = np.column_stack([gx.ravel(), gy.ravel()]).astype(float)
    return pos + rng.uniform(-3, 3, pos.shape)


def _catalog(path, positions, exposure=1, module="nrca1", vgroup="00001"):
    n = len(positions)
    tbl = Table()
    tbl["x_fit"], tbl["y_fit"] = positions[:, 0], positions[:, 1]
    tbl["flux_fit"] = np.linspace(2e4, 1e3, n)
    tbl["flux_err"] = tbl["flux_fit"] / 100.0
    tbl["qfit"] = 0.02
    tbl.meta.update(FILENAME=str(path), VISIT="001", EXPOSURE=f"{exposure:05d}",
                    MODULE=module, FILTER="F200W", VGROUP=vgroup)
    return tbl


def _shape(sigma=1.6, ratio=0.95, pa=30.0, n=200):
    return dict(status="ok", sigma=sigma, axis_ratio=ratio, pa_sky_deg=pa,
                pa_coherence=0.9, n_stars=n)


def _record(exposure, module, shape, visit="001", vgroup="00001", stem=None):
    return dict(key=(visit, exposure, module, "F200W", vgroup), shape=shape,
                stem=stem or f"jw02211049001_02201_{exposure:05d}")


TRAILED = dict(sigma=2.6, ratio=0.6)
DETECTORS = ("nrca1", "nrca2", "nrca3", "nrca4")


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------

def test_adaptive_moments_recovers_a_round_star():
    yy, xx = np.mgrid[0:29, 0:29].astype(float)
    star = np.exp(-((xx - 14.3) ** 2 + (yy - 13.8) ** 2) / (2 * 2.0 ** 2))
    cx, cy, mxx, myy, mxy = et.adaptive_moments(star, 14.0, 14.0)
    assert cx == pytest.approx(14.3, abs=0.02)
    assert cy == pytest.approx(13.8, abs=0.02)
    assert math.sqrt(mxx) == pytest.approx(2.0, rel=0.03)
    assert math.sqrt(myy) == pytest.approx(2.0, rel=0.03)
    assert abs(mxy) < 0.05


def test_adaptive_moments_does_not_settle_on_empty_stamp():
    assert et.adaptive_moments(np.zeros((29, 29)), 14.0, 14.0) is None


def test_trail_is_measured_where_the_stamp_is_wider_than_the_psf(tmp_path):
    """A 6 px smear along one axis raises sigma and lowers the axis ratio, and
    the major axis is reported in the SKY frame (the frame is rolled)."""
    pos = _star_grid()
    roll = 40.0
    theta_pix = math.radians(20.0)
    clean, trailed = tmp_path / "clean.fits", tmp_path / "trailed.fits"
    _write_frame(clean, pos, roll_deg=roll)
    wcs = _write_frame(trailed, pos, trail_px=6.0, trail_theta=theta_pix,
                       roll_deg=roll, seed=1)
    c = et.measure_exposure_shape(_catalog(clean, pos), min_stars=20)
    t = et.measure_exposure_shape(_catalog(trailed, pos), min_stars=20)
    assert c["status"] == t["status"] == "ok"
    assert c["sigma"] == pytest.approx(PSF_SIGMA, rel=0.05)
    assert c["axis_ratio"] > 0.95
    assert t["sigma"] > c["sigma"] + 0.5
    assert t["axis_ratio"] < 0.8
    assert t["pa_coherence"] > 0.9
    expected = et._sky_position_angle_deg(wcs, SIZE / 2, SIZE / 2, theta_pix)
    diff = abs((t["pa_sky_deg"] - expected + 90.0) % 180.0 - 90.0)
    assert diff < 5.0


def test_missing_frame_is_a_reason_not_an_exception(tmp_path):
    pos = _star_grid()
    tbl = _catalog(tmp_path / "absent.fits", pos)
    res = et.measure_exposure_shape(tbl)
    assert res["status"] == "cannot_measure"
    assert "not on disk" in res["reason"]
    tbl.meta.pop("FILENAME")
    assert et.measure_exposure_shape(tbl)["status"] == "cannot_measure"


def test_too_few_isolated_stars_cannot_be_measured(tmp_path):
    pos = _star_grid()[:5]
    _write_frame(tmp_path / "f.fits", pos)
    res = et.measure_exposure_shape(_catalog(tmp_path / "f.fits", pos))
    assert res["status"] == "cannot_measure"
    assert res["n_stars"] < et.TRAIL_MIN_STARS


def test_a_blended_pair_is_not_isolated(tmp_path):
    pos = np.array([[100.0, 100.0], [108.0, 100.0]])
    _write_frame(tmp_path / "f.fits", pos)
    cand, _, _ = et._select_stars(_catalog(tmp_path / "f.fits", pos), 14, 400)
    assert len(cand) == 0


# ---------------------------------------------------------------------------
# grading
# ---------------------------------------------------------------------------

def _visit(trailed=None, n_exp=4, detectors=DETECTORS, pas=None, only=None):
    """Records for ``n_exp`` exposures; ``trailed`` is an exposure number whose
    detectors (all, or those in ``only``) are smeared at ``pas[module]``."""
    recs = []
    for e in range(1, n_exp + 1):
        for i, mod in enumerate(detectors):
            shape = _shape(sigma=1.6 + 0.02 * i)
            if e == trailed and (only is None or mod in only):
                shape = _shape(sigma=TRAILED["sigma"], ratio=TRAILED["ratio"],
                               pa=(pas or {}).get(mod, 35.0))
            recs.append(_record(e, mod, shape))
    return recs


def _verdicts(recs, **kw):
    return {e["exposure"]: e for e in et.grade_visit_trails(recs, reviewed=kw.get("reviewed", {}))}


def test_a_clean_visit_flags_nothing():
    v = _verdicts(_visit())
    assert {e["verdict"] for e in v.values()} == {"ok"}


def test_one_trailed_exposure_among_three_clean_is_the_only_flag():
    v = _verdicts(_visit(trailed=4))
    assert v[4]["verdict"] == "flagged"
    assert v[4]["n_over_threshold"] == 4
    assert [v[e]["verdict"] for e in (1, 2, 3)] == ["ok"] * 3
    assert v[4]["pa_coherence"] > 0.99


def test_a_smear_on_half_the_detectors_with_a_common_angle_is_flagged():
    v = _verdicts(_visit(trailed=2, only=("nrca1", "nrca2")))
    assert v[2]["verdict"] == "flagged"


def test_a_single_detector_smear_is_not_flagged():
    v = _verdicts(_visit(trailed=3, only=("nrca2",)))
    assert v[3]["verdict"] == "ok"
    assert v[3]["n_over_threshold"] == 1


def test_smears_at_unrelated_angles_are_a_detector_problem_not_a_drift():
    pas = {"nrca1": 0.0, "nrca2": 60.0, "nrca3": 120.0, "nrca4": 30.0}
    v = _verdicts(_visit(trailed=3, pas=pas))
    assert v[3]["verdict"] == "ok"
    assert v[3]["n_over_threshold"] == 4
    assert "disagree" in v[3]["reason"]


def test_a_defocus_without_elongation_is_not_a_trail():
    recs = _visit()
    for r in recs:
        if r["key"][1] == 2:
            r["shape"] = _shape(sigma=2.6, ratio=0.95)
    assert _verdicts(recs)[2]["verdict"] == "ok"


def test_two_exposures_cannot_be_graded():
    v = _verdicts(_visit(trailed=2, n_exp=2))
    assert {e["verdict"] for e in v.values()} == {"cannot_grade"}


def test_a_visit_with_every_exposure_trailed_is_not_seen_by_siblings():
    """The all-trailed visit is the blind spot of a sibling comparison; this
    pins that it is not mis-reported as a verdict about one exposure."""
    recs = _visit()
    for r in recs:
        r["shape"] = _shape(sigma=TRAILED["sigma"], ratio=TRAILED["ratio"])
    assert {e["verdict"] for e in _verdicts(recs).values()} == {"ok"}


def test_unmeasured_detectors_do_not_count_as_clean_or_trailed():
    recs = _visit(trailed=4)
    for r in recs:
        if r["key"][2] in ("nrca3", "nrca4") and r["key"][1] == 4:
            r["shape"] = dict(status="cannot_measure", reason="too few stars")
    v = _verdicts(recs)
    assert v[4]["n_graded"] == 2
    assert v[4]["verdict"] == "flagged"


def test_a_reviewed_stem_is_recorded_not_flagged():
    stem = "jw02211049001_02201_00004"
    v = _verdicts(_visit(trailed=4), reviewed={stem: "round in the pixels, small defect"})
    assert v[4]["verdict"] == "reviewed"
    assert "round in the pixels" in v[4]["reason"]


def test_visit_groups_are_graded_separately():
    recs = _visit(trailed=4) + [
        _record(e, m, _shape(), vgroup="00002") for e in (1, 2, 3, 4) for m in DETECTORS]
    out = et.grade_visit_trails(recs, reviewed={})
    flagged = [(e["vgroup"], e["exposure"]) for e in out if e["verdict"] == "flagged"]
    assert flagged == [("00001", 4)]


def test_reviewed_registry_entries_name_what_was_seen():
    for stem, reason in et.TRAIL_REVIEWED.items():
        assert et.exposure_stem(stem + "_nrca1_cal.fits") == stem
        assert len(reason) > 20


# ---------------------------------------------------------------------------
# end to end on synthetic pixels
# ---------------------------------------------------------------------------

def test_assess_visit_trails_flags_the_trailed_frames_it_reads(tmp_path):
    pos = _star_grid()
    tables = []
    for e in (1, 2, 3, 4):
        for k, mod in enumerate(("nrca1", "nrca2")):
            path = tmp_path / f"jw02211049001_02201_{e:05d}_{mod}_crf.fits"
            trailed = e == 4
            _write_frame(path, pos, trail_px=6.0 if trailed else 0.0,
                         trail_theta=math.radians(20.0), roll_deg=40.0,
                         seed=10 * e + k)
            tables.append(_catalog(path, pos, exposure=e, module=mod))
    block = et.assess_visit_trails(
        tables, measure=lambda t: et.measure_exposure_shape(t, min_stars=20),
        reviewed={})
    verdicts = {x["exposure"]: x["verdict"] for x in block["exposures"]}
    assert verdicts == {1: "ok", 2: "ok", 3: "ok", 4: "flagged"}
    assert block["mode"] == "record_only"
    assert block["thresholds"]["provisional"] is True
    assert block["flagged"] == [["jw02211049001_02201_00004"]]


# ---------------------------------------------------------------------------
# checkpoint integration
# ---------------------------------------------------------------------------

def test_m2_checkpoint_carries_the_trail_block_and_gates_nothing(tmp_path):
    tables = _visit_tables(misaligned={2: (8.0, 0.0)})
    record = run_visit_checkpoint(tables, "m2", filtername="F212N",
                                  record_dir=str(tmp_path), context="test")
    trail = record["visits"][0]["trail"]
    assert trail["mode"] == "record_only"
    # the synthetic catalogs name no frame: nothing could be measured, and that
    # is said rather than passed over
    assert trail["n_measured"] == 0
    assert {e["verdict"] for e in trail["exposures"]} == {"cannot_grade"}
    assert record["passed"]
    assert len(record["corrections"]) == 1
    assert not record["unverified"]


def test_a_flagged_trail_is_recorded_and_does_not_change_the_gate(tmp_path, monkeypatch, capsys):
    base = run_visit_checkpoint(_visit_tables(), "m2", filtername="F212N",
                                record_dir=str(tmp_path / "a"), context="test")
    flagged = dict(mode="record_only", thresholds={}, n_measured=4, n_frames=4,
                   flagged=[["jw0_x"]],
                   exposures=[dict(exposure=4, stems=["jw0_x"], verdict="flagged",
                                   reason="synthetic")])
    monkeypatch.setattr(ac, "assess_visit_trails", lambda tables: flagged)
    rec = run_visit_checkpoint(_visit_tables(), "m2", filtername="F212N",
                               record_dir=str(tmp_path / "b"), context="test")
    assert rec["visits"][0]["trail"]["flagged"] == [["jw0_x"]]
    assert "TRAILED EXPOSURE" in capsys.readouterr().out
    for key in ("passed", "all_verified", "failures", "unverified",
                "unverified_blocking", "corrections"):
        assert rec[key] == base[key]


def test_a_frozen_stage_does_not_open_any_pixels(tmp_path, monkeypatch):
    def _no_pixels(tables):
        raise AssertionError("a frozen stage must not re-measure the pixels")
    monkeypatch.setattr(ac, "assess_visit_trails", _no_pixels)
    rec = run_visit_checkpoint(_visit_tables(), "m5", filtername="F212N",
                               record_dir=str(tmp_path), context="test")
    assert rec["visits"][0]["trail"] is None
