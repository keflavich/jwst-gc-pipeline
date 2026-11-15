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

from jwst_gc_pipeline.frame_wcs import MissingGwcsWarning, gwcs_from_file

from .test_visit_consensus import _visit_tables

# The synthetic frames are plain FITS images with no GWCS.
pytestmark = pytest.mark.filterwarnings(
    f"ignore::{MissingGwcsWarning.__module__}.{MissingGwcsWarning.__name__}")

SIZE = 360
PSF_SIGMA = 1.6


def _write_frame(path, positions, trail_px=0.0, trail_theta=0.0, sigma=PSF_SIGMA,
                 roll_deg=0.0, seed=0, flip=False, pixel_aspect=1.0):
    """Render Gaussian stars (smeared along ``trail_theta`` by ``trail_px``).

    ``flip`` reverses the parity of the WCS (east to the right) and
    ``pixel_aspect`` makes the y pixels that much longer than the x pixels.
    """
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
    sgn = 1.0 if flip else -1.0
    wcs.wcs.cd = scale * np.array([[sgn * c, s * pixel_aspect],
                                   [s, -sgn * c * pixel_aspect]])
    hdr = wcs.to_header()
    hdul = fits.HDUList([fits.PrimaryHDU(),
                         fits.ImageHDU(img.astype("f4"), header=hdr, name="SCI")])
    hdul.writeto(path, overwrite=True)
    return wcs


def _expected_sky_pa(wcs, theta):
    """East-of-north angle (mod 180 deg) of a pixel direction, from the CD
    matrix alone: xi is east and eta north in the tangent plane."""
    xi, eta = wcs.wcs.cd @ np.array([math.cos(theta), math.sin(theta)])
    return math.degrees(math.atan2(xi, eta)) % 180.0


def _angle_diff_deg(a, b):
    return abs((a - b + 90.0) % 180.0 - 90.0)


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
    # the expected angle comes from the CD matrix, not from the function under test
    assert _angle_diff_deg(t["pa_sky_deg"], _expected_sky_pa(wcs, theta_pix)) < 5.0


@pytest.mark.parametrize("flip", [False, True])
@pytest.mark.parametrize("pixel_aspect", [1.0, 1.1])
def test_sky_position_angle_follows_the_cd_matrix(tmp_path, flip, pixel_aspect):
    """Both WCS parities and non-square pixels: the major axis is reported
    east of north as the CD matrix maps it."""
    pos = _star_grid()
    theta_pix = math.radians(30.0)
    frame = tmp_path / "trailed.fits"
    wcs = _write_frame(frame, pos, trail_px=6.0, trail_theta=theta_pix, roll_deg=40.0,
                       flip=flip, pixel_aspect=pixel_aspect)
    res = et.measure_exposure_shape(_catalog(frame, pos), min_stars=20)
    assert res["status"] == "ok"
    expected = _expected_sky_pa(wcs, theta_pix)
    assert _angle_diff_deg(res["pa_sky_deg"], expected) < 5.0


def test_trail_is_measured_at_the_real_f212n_psf_width(tmp_path):
    """The brick F212N frames have a PSF sigma near 1.0 px, not the 1.6 px of
    the other tests: the estimator must still recover it and see a trail."""
    pos = _star_grid()
    clean, trailed = tmp_path / "clean.fits", tmp_path / "trailed.fits"
    _write_frame(clean, pos, sigma=1.0, roll_deg=40.0)
    _write_frame(trailed, pos, sigma=1.0, trail_px=4.0, trail_theta=math.radians(20.0),
                 roll_deg=40.0, seed=1)
    c = et.measure_exposure_shape(_catalog(clean, pos), min_stars=20)
    t = et.measure_exposure_shape(_catalog(trailed, pos), min_stars=20)
    assert c["status"] == t["status"] == "ok"
    assert c["sigma"] == pytest.approx(1.0, rel=0.1)
    assert c["axis_ratio"] > 0.95
    # the grader's excess is in quadrature
    assert math.sqrt(t["sigma"] ** 2 - c["sigma"] ** 2) > et.TRAIL_EXCESS_SIGMA_PX
    assert c["axis_ratio"] - t["axis_ratio"] > et.TRAIL_AXIS_RATIO_DROP


def test_missing_frame_is_a_reason_not_an_exception(tmp_path):
    pos = _star_grid()
    tbl = _catalog(tmp_path / "absent.fits", pos)
    res = et.measure_exposure_shape(tbl)
    assert res["status"] == "cannot_measure"
    assert "not on disk" in res["reason"]
    tbl.meta.pop("FILENAME")
    assert et.measure_exposure_shape(tbl)["status"] == "cannot_measure"


def test_an_unreadable_gwcs_is_a_reason_not_an_exception(tmp_path, monkeypatch):
    """The reading is advisory: a GwcsReadError (a RuntimeError) from the frame
    must not stop the checkpoint that carries it."""
    pos = _star_grid()
    frame = tmp_path / "frame.fits"
    _write_frame(frame, pos)

    def _boom(*args, **kwargs):
        raise et.GwcsReadError("synthetic: ASDF extension unreadable")

    monkeypatch.setattr(et, "frame_wcs", _boom)
    res = et.measure_exposure_shape(_catalog(frame, pos))
    assert res["status"] == "cannot_measure"
    assert "GwcsReadError" in res["reason"]


def test_a_non_finite_sky_angle_is_not_measured(tmp_path, monkeypatch):
    """NaN < coherence minimum is False, so a NaN angle would read as a
    coherent drift in the grading; it must not leave the measurement."""
    pos = _star_grid()
    frame = tmp_path / "frame.fits"
    _write_frame(frame, pos, trail_px=6.0, trail_theta=0.3)
    monkeypatch.setattr(et, "_sky_position_angle_deg", lambda *a, **k: float("nan"))
    res = et.measure_exposure_shape(_catalog(frame, pos))
    assert res["status"] == "cannot_measure"
    assert "not finite" in res["reason"]


def test_a_yaml_corrupt_asdf_is_a_reason_not_an_exception(tmp_path):
    """A text-corrupted ASDF block fails in the YAML parser (not a ValueError);
    it must reach the reading as a GwcsReadError and become a reason."""
    pos = _star_grid()
    frame = tmp_path / "frame.fits"
    _write_frame(frame, pos)
    # a valid ASDF header over a YAML body that does not parse
    junk = np.frombuffer(
        b"#ASDF 1.0.0\n#ASDF_STANDARD 1.5.0\n%YAML 1.1\n%TAG ! tag:stsci.edu:asdf/\n"
        b"--- !core/asdf-1.1.0\nmeta: {wcs: [unclosed\n...\n", dtype="u1")
    with fits.open(frame, mode="update") as hdul:
        hdul.append(fits.ImageHDU(junk, name="ASDF"))
    with pytest.raises(et.GwcsReadError):
        gwcs_from_file(str(frame), use_cache=False)
    res = et.measure_exposure_shape(_catalog(frame, pos))
    assert res["status"] == "cannot_measure"
    assert "GwcsReadError" in res["reason"]


def test_a_non_2d_sci_is_a_reason_not_an_exception(tmp_path):
    pos = _star_grid()
    frame = tmp_path / "frame.fits"
    _write_frame(frame, pos)
    with fits.open(frame, mode="update") as hdul:
        hdul["SCI"].data = np.zeros((2, SIZE, SIZE), dtype="f4")
    res = et.measure_exposure_shape(_catalog(frame, pos))
    assert res["status"] == "cannot_measure"


def test_the_recorded_frame_spelling_is_resolved(tmp_path, monkeypatch):
    """meta['FILENAME'] can name the /blue or the /orange spelling of a frame
    that exists under the other one; the helper that maps them is used."""
    pos = _star_grid()
    frame = tmp_path / "frame.fits"
    _write_frame(frame, pos)
    tbl = _catalog("/blue/gone/frame.fits", pos)
    assert et.measure_exposure_shape(tbl, min_stars=20)["status"] == "cannot_measure"
    monkeypatch.setattr(et, "_resolve_existing_path", lambda path: str(frame))
    assert et.measure_exposure_shape(tbl, min_stars=20)["status"] == "ok"


def test_stars_with_an_unusable_stamp_do_not_use_up_the_star_quota(tmp_path):
    """The brightest stars sit on NaN pixels; the quota of stars is filled
    from the ones after them, instead of being spent before the stamp check."""
    pos = _star_grid()
    frame = tmp_path / "frame.fits"
    _write_frame(frame, pos)
    n_bad = 16
    with fits.open(frame, mode="update") as hdul:
        for x0, y0 in pos[:n_bad]:       # _catalog() makes row 0 the brightest
            hdul["SCI"].data[int(round(y0)) + 3, int(round(x0)) + 3] = np.nan
    res = et.measure_exposure_shape(_catalog(frame, pos), n_max=15, min_stars=10)
    assert res["status"] == "ok", res
    assert res["n_stars"] == 15


def test_stars_that_fail_the_reliable_cut_are_not_measured(tmp_path):
    pos = _star_grid()
    tbl = _catalog(tmp_path / "f.fits", pos)
    cand, _, _ = et._select_stars(tbl, 14, None)
    assert len(cand) == len(pos)
    tbl["qfit"][:10] = 0.5                 # poorly fit
    tbl["flux_err"][10:20] = tbl["flux_fit"][10:20]    # S/N 1
    cand, _, _ = et._select_stars(tbl, 14, None)
    assert set(cand) == set(range(20, len(pos)))


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


def test_a_nan_sky_angle_in_a_record_does_not_read_as_agreement():
    recs = _visit(trailed=4)
    for r in recs:
        if r["key"][1] == 4 and r["key"][2] == "nrca1":
            r["shape"]["pa_sky_deg"] = float("nan")
    v = _verdicts(recs)
    assert v[4]["verdict"] == "ok"
    assert "disagree" in v[4]["reason"]


def test_position_angles_either_side_of_180_agree():
    pas = {"nrca1": 179.0, "nrca2": 1.0, "nrca3": 179.0, "nrca4": 1.0}
    v = _verdicts(_visit(trailed=3, pas=pas))
    assert v[3]["verdict"] == "flagged"
    assert v[3]["pa_coherence"] > 0.99


def _n_detector_visit(n_det, n_trailed_det, n_exp=4):
    detectors = tuple(f"nrca{i}" for i in range(1, n_det + 1))
    return _visit(trailed=4, n_exp=n_exp, detectors=detectors,
                  only=detectors[:n_trailed_det])


@pytest.mark.parametrize("n_det,n_over,verdict", [
    (2, 2, "flagged"), (2, 1, "ok"),      # the floor of two detectors
    (3, 2, "flagged"), (3, 1, "ok"),
    (5, 3, "flagged"), (5, 2, "ok"),      # ceil(5 / 2) = 3, not floor
    (6, 3, "flagged"), (6, 2, "ok"),
    (8, 4, "flagged"), (8, 3, "ok"),
])
def test_how_many_detectors_it_takes_to_flag(n_det, n_over, verdict):
    assert _verdicts(_n_detector_visit(n_det, n_over))[4]["verdict"] == verdict


def test_one_graded_detector_cannot_be_graded():
    v = _verdicts(_n_detector_visit(1, 1))
    assert v[4]["verdict"] == "cannot_grade"


def test_the_excess_is_in_quadrature_not_a_difference():
    """1.6 -> 1.7 px is a 0.1 px difference but a 0.57 px quadrature excess."""
    recs = _visit()
    for r in recs:
        if r["key"][1] == 4:
            r["shape"] = _shape(sigma=1.7, ratio=0.80)
    assert _verdicts(recs)[4]["verdict"] == "flagged"
    for r in recs:
        if r["key"][1] == 4:
            r["shape"] = _shape(sigma=1.65, ratio=0.80)      # 0.4 px in quadrature
    assert _verdicts(recs)[4]["verdict"] == "ok"


@pytest.mark.parametrize("n_exp,n_trailed,verdict", [
    (4, 2, "flagged"),
    (4, 3, "ok"),      # the sibling median is itself trailed: documented blind spot
    (8, 4, "flagged"),
    (8, 5, "ok"),
])
def test_the_sibling_comparison_goes_blind_when_most_exposures_trail(
        n_exp, n_trailed, verdict):
    recs = _visit(n_exp=n_exp)
    for r in recs:
        if r["key"][1] <= n_trailed:
            r["shape"] = _shape(sigma=TRAILED["sigma"], ratio=TRAILED["ratio"])
    v = _verdicts(recs)
    assert {v[e]["verdict"] for e in range(1, n_trailed + 1)} == {verdict}


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


def test_an_unmeasured_visit_is_logged(tmp_path, capsys):
    run_visit_checkpoint(_visit_tables(), "m2", filtername="F212N",
                         record_dir=str(tmp_path), context="test")
    out = capsys.readouterr().out
    assert "trail reading (advisory)" in out
    assert "0 of " in out and "frames measured" in out
