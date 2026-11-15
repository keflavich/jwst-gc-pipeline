"""Meta-tests for test_residual_model_policy.py -- they must run ALWAYS.

pytest ORs skip conditions, so a function-level ``skipif(False)`` cannot cancel
a module-level ``pytestmark``.  The previous attempt put these beside the
policy tests with ``skipif(False)`` and they skipped with the rest -- on the
only tree where the divergence they guard actually crashes.  They live here,
outside that mark, instead.
"""
import glob
import json

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from . import test_residual_model_policy as policy
from .test_residual_model_policy import (
    _REQUIRED, f480m, over_rendered, over_subtracted, cross_run_reason,
    MODEL_PEAK_CEILING, RESID_CORE_FLOOR, SAME_RUN_HOURS,
    nearest_neighbour_arcsec, star_halfwidth_px, annulus_sky,
    integrated_ratio, star_record, record_diff, write_record,
    integrated_median_failure, integrated_gross_failure, report_record,
    record_update_requested, MAX_DROPPED_FRACTION, INTEGRATED_MIN_STARS,
    BOX_HALFWIDTH_CAP_PX, RECORD_UPDATE_ENV)


def test_each_required_glob_matches_at_most_one_file():
    """The cross-run hazard in issue #266: a glob matching both ``resbgsub_m5``
    and a ``group`` variant lets `_latest` take numerator and denominator from
    different runs."""
    for what, pattern in _REQUIRED.items():
        assert len(glob.glob(pattern)) <= 1, (what, glob.glob(pattern))


def test_skip_predicate_covers_every_product_the_fixture_opens(monkeypatch):
    """The predicate and the fixture must glob the SAME patterns.

    They diverged for a month: the predicate checked
    `*mergedcat_residual_i2d.fits` while the fixture needed the `_m7_` variant,
    so a field with m2..m5 products but no m7 passed the predicate and then
    opened None -- three ERRORS on main since 2026-07-05 (issue #266).

    Checked by RECORDING what the fixture actually globs, not by inspecting its
    source: a source check only proves the label strings appear, and passes for
    a fixture that builds its paths inline or opens a fourth product.
    """
    seen = []
    real_latest = policy._latest

    def _record(pattern):
        seen.append(pattern)
        return real_latest(pattern)

    # patch the POLICY module's namespace -- the fixture resolves `_latest`
    # there, not here
    monkeypatch.setattr(policy, "_latest", _record)
    # ... and stub everything the fixture does BETWEEN the globs, so that a
    # tree with no products still reaches all three of them.  Unstubbed, on CI
    # -- the environment this whole change is about -- `_read_points` raised
    # before a single glob was recorded and the test went RED, and stubbing
    # only that one got as far as the first `_img(None)` skip and then failed
    # on `missing`.  Trading a silent skip for a red CI is not a fix.  What
    # this test checks is WHICH PATTERNS the fixture asks for; opening the
    # files is not part of that.
    monkeypatch.setattr(policy, "_read_points", lambda path: None)
    monkeypatch.setattr(policy, "_img", lambda path, what="": (None, None))
    monkeypatch.setattr(policy, "_peaks", lambda arr, w, stars, box=3: None)
    monkeypatch.setattr(policy, "_troughs", lambda arr, w, stars, box=3: None)
    monkeypatch.setattr(policy, "_integrated_products",
                        lambda *a, **k: {})
    try:
        f480m.__wrapped__()
    except (pytest.skip.Exception, OSError, ValueError):
        # NAMED, not `BaseException`.  Widening to BaseException did catch the
        # `Skipped` that `except Exception` let escape -- `pytest.skip` raises
        # a BaseException subclass, which is what turned this guard into a skip
        # in the scenario it exists to catch -- but it also swallows
        # KeyboardInterrupt and SystemExit.  `Skipped` has a public handle, so
        # the specific form covers all the documented exits (the skip from
        # `_img(None)`, an unreadable FITS, a malformed region file) while a
        # NameError or TypeError from a future fixture edit still surfaces --
        # which is the breakage this test is for.  KeyError is deliberately
        # NOT here: the fixture indexes `_REQUIRED`, so a KeyError means the
        # predicate and the fixture have gone out of step -- the exact
        # condition under test.  Swallowing it let that mutant live.
        pass
    if not seen:
        pytest.skip("the fixture bailed before its first glob, so there is "
                    "nothing to compare against the skip predicate")
    missing = [p for p in _REQUIRED.values() if p not in seen]
    assert not missing, (
        "the fixture did not open every _REQUIRED product, so the predicate "
        "checks something the fixture does not need -- or stopped early:\n  "
        + "\n  ".join(missing))
    unknown = [p for p in seen if p not in set(_REQUIRED.values())]
    assert not unknown, (
        "the fixture opens product(s) the skip predicate does not check, so a "
        "tree missing them errors instead of skipping:\n  "
        + "\n  ".join(unknown))


#: The three products on disk 2026-08-21 -- one generation, and the ordering a
#: generation has: the data mosaic is drizzled first, the m5 QA products follow.
_SAME_RUN = ("2026-08-21T08:05:17.713",   # data
             "2026-08-21T12:50:35.615",   # residual
             "2026-08-21T12:51:01.406")   # model

#: The 2026-07-05 state that made this file red for five weeks with no commit
#: behind it: a partial re-run rewrote the data mosaic and left the June QA
#: products in place (issue #266 item 2).
_CROSS_RUN = ("2026-07-05T19:11:39",
              "2026-06-27T18:40:04",
              "2026-06-27T18:40:04")


def test_cross_run_reason_accepts_one_generation():
    assert cross_run_reason(*_SAME_RUN) is None


def test_cross_run_reason_catches_a_data_mosaic_newer_than_its_qa_products():
    reason = cross_run_reason(*_CROSS_RUN)
    assert reason is not None
    # the skip has to NAME both sides, or it is another unreadable line
    assert "2026-07-05T19:11:39" in reason and "2026-06-27T18:40:04" in reason


def test_cross_run_reason_catches_a_residual_and_model_from_two_runs():
    reason = cross_run_reason("2026-08-21T08:05:17",
                              "2026-08-21T12:50:35",
                              "2026-08-19T12:51:01")
    assert reason is not None and "not from one cataloging run" in reason


def test_cross_run_reason_tolerates_the_within_run_gap():
    """The residual and model are written seconds apart; the check must not fire
    on that, or it becomes a permanent skip."""
    later = f"2026-08-21T{8 + int(SAME_RUN_HOURS):02d}:00:00"
    assert cross_run_reason("2026-08-21T07:00:00",
                            "2026-08-21T08:00:00", later) is None


def test_cross_run_reason_reports_an_unreadable_date_rather_than_passing():
    reason = cross_run_reason(None, _SAME_RUN[1], "not-a-date")
    assert reason is not None
    assert "data i2d" in reason and "model i2d" in reason
    assert "residual i2d" not in reason


def test_fixture_consults_the_generation_check(monkeypatch):
    """Wiring pin.  The predicate above is a pure function; nothing in it forces
    the fixture to call it, and an uncalled guard is the shape this file's
    history is made of.  Stub it to complain and assert the fixture skips."""
    monkeypatch.setattr(policy, "cross_run_reason",
                        lambda *a, **k: "STUB cross-run complaint")
    monkeypatch.setattr(policy, "_read_points", lambda path: None)
    monkeypatch.setattr(policy, "_img", lambda path, what="": (None, None))
    monkeypatch.setattr(policy, "_peaks", lambda arr, w, stars, box=3: None)
    monkeypatch.setattr(policy, "_troughs", lambda arr, w, stars, box=3: None)
    monkeypatch.setattr(policy, "_integrated_products",
                        lambda *a, **k: {})
    with pytest.raises(pytest.skip.Exception, match="STUB cross-run complaint"):
        f480m.__wrapped__()


class _FakeWCS:
    """Identity world->pixel, so the box arithmetic can be tested without a FITS."""

    def __init__(self, xy):
        self._xy = np.asarray(xy, float)

    def world_to_pixel(self, stars):
        return self._xy[:, 0], self._xy[:, 1]


def test_troughs_reads_the_core_minimum_and_peaks_the_maximum():
    """`_peaks` alone cannot see a crater: both a hole and a clean subtraction
    have a small positive maximum, which is why the over-subtraction assertions
    need their own reduction over the same box (issue #266 item 4)."""
    arr = np.zeros((21, 21))
    arr[10, 10] = -500.0          # crater
    arr[10, 11] = +3.0            # the max `_peaks` would report
    w = _FakeWCS([[10, 10]])
    assert policy._peaks(arr, w, None, box=3)[0] == pytest.approx(3.0)
    assert policy._troughs(arr, w, None, box=3)[0] == pytest.approx(-500.0)


def test_troughs_is_nan_off_image():
    arr = np.zeros((21, 21))
    w = _FakeWCS([[-5, -5]])
    assert np.isnan(policy._troughs(arr, w, None)[0])


def test_over_rendered_flags_only_above_the_ceiling():
    d = np.array([100.0, 100.0, 100.0, 100.0])
    m = np.array([146.0,             # the worst star measured 2026-08-21: kept
                  MODEL_PEAK_CEILING * 100.0 - 1,
                  MODEL_PEAK_CEILING * 100.0 + 1,
                  1000.0])
    assert list(over_rendered(d, m)) == [2, 3]


def test_over_subtracted_flags_only_below_the_floor():
    d = np.array([100.0, 100.0, 100.0, 100.0])
    rmin = np.array([-47.0,          # the worst star measured 2026-08-21: kept
                     RESID_CORE_FLOOR * 100.0 + 1,
                     RESID_CORE_FLOOR * 100.0 - 1,
                     -900.0])
    assert list(over_subtracted(d, rmin)) == [2, 3]


def test_two_sided_predicates_ignore_unusable_stars():
    """A star off the mosaic, or with a non-positive data peak, cannot be graded
    either way -- the same `ok` mask the one-sided assertions already use."""
    d = np.array([np.nan, 0.0, -5.0, 100.0])
    m = np.array([1e6, 1e6, 1e6, 1e6])
    rmin = np.array([-1e6, -1e6, -1e6, -1e6])
    assert list(over_rendered(d, m)) == [3]
    assert list(over_subtracted(d, rmin)) == [3]


# --- integrated-flux helpers (issue #695): synthetic arrays, no products -------

PIX_ARCSEC = 0.063


def _tan_wcs(n=161, scale=PIX_ARCSEC):
    """A north-up TAN grid centred on (RA, Dec) = (266, -29), ``n`` px square."""
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [266.0, -29.0]
    w.wcs.crpix = [(n + 1) / 2.0, (n + 1) / 2.0]
    w.wcs.cdelt = [-scale / 3600.0, scale / 3600.0]
    return w


def _sky_at(w, x, y):
    return w.pixel_to_world(np.asarray(x, float), np.asarray(y, float))


def _gauss(n, x0, y0, sigma, flux):
    yy, xx = np.mgrid[:n, :n]
    g = np.exp(-0.5 * ((xx - x0) ** 2 + (yy - y0) ** 2) / sigma ** 2)
    return flux * g / g.sum()


def _scene(sky=5.0, data_sigma=2.0, model_sigma=2.0, model_scale=1.0, n=161):
    """One star at the centre of a flat sky; the model has the same flux."""
    w = _tan_wcs(n)
    c = (n - 1) / 2.0
    star = _sky_at(w, [c], [c])
    data = sky + _gauss(n, c, c, data_sigma, 1000.0)
    model = model_scale * _gauss(n, c, c, model_sigma, 1000.0)
    return w, star, data, model


def test_integrated_ratio_is_one_for_a_matching_model():
    w, star, data, model = _scene()
    half = np.array([12])
    sky = annulus_sky(data, w, star, half)
    assert sky[0] == pytest.approx(5.0)
    assert integrated_ratio(model, data, w, star, half, sky)[0] == \
        pytest.approx(1.0, abs=0.01)


def test_integrated_ratio_scales_with_the_model():
    w, star, data, model = _scene(model_scale=2.0)
    half = np.array([12])
    sky = annulus_sky(data, w, star, half)
    assert integrated_ratio(model, data, w, star, half, sky)[0] == \
        pytest.approx(2.0, abs=0.02)


def test_sharper_model_inflates_the_peak_ratio_but_not_the_integral():
    """The bias the issue describes: same flux, narrower model."""
    w, star, data, model = _scene(data_sigma=2.5, model_sigma=1.5)
    half = np.array([12])
    sky = annulus_sky(data, w, star, half)
    integrated = integrated_ratio(model, data, w, star, half, sky)[0]
    peak = policy._peaks(model, w, star)[0] / policy._peaks(data - 5.0, w, star)[0]
    assert integrated == pytest.approx(1.0, abs=0.02)
    assert peak > 1.5


def test_sky_annulus_masks_a_neighbours_box():
    """A bright neighbour inside the annulus must not move the sky.

    The median is robust to a small contaminated patch, so the neighbour here
    fills a large part of the ring and the sky has noise for it to push on.
    """
    w = _tan_wcs()
    c = 80
    stars = _sky_at(w, [c, c + 37], [c, c])
    rng = np.random.default_rng(695)
    data = 5.0 + rng.normal(0.0, 1.0, (161, 161))
    clean = annulus_sky(data, w, stars, np.array([12, 1]))[0]
    data[c - 20:c + 21, c + 37 - 20:c + 37 + 21] = 1e4
    masked = annulus_sky(data, w, stars, np.array([12, 20]))
    assert masked[0] == pytest.approx(clean, abs=0.1)
    # with the neighbour's box too small to cover it, the contamination shows
    leaky = annulus_sky(data, w, stars, np.array([12, 1]))
    assert leaky[0] > clean + 0.1


def test_sky_annulus_is_nan_when_it_leaves_the_image():
    w = _tan_wcs(121)
    stars = _sky_at(w, [10], [60])
    data = np.full((121, 121), 5.0)
    assert np.isnan(annulus_sky(data, w, stars, np.array([12]))[0])


def test_integrated_ratio_is_nan_for_negative_flux():
    """Data below the sky sums to a negative flux; ``flux != 0`` would turn
    that into a negative "ratio"."""
    w = _tan_wcs()
    sky = 5.0
    data = np.full((161, 161), sky - 5.0)
    model = np.ones((161, 161))
    stars = _sky_at(w, [80], [80])
    out = integrated_ratio(model, data, w, stars, np.array([5]), np.array([sky]))
    assert np.isnan(out[0])


def test_integrated_ratio_is_nan_off_image_and_for_empty_or_negative_flux():
    w = _tan_wcs()
    data = np.full((161, 161), 5.0)
    model = np.ones((161, 161))
    stars = _sky_at(w, [3, 80, 80], [80, 80, 80])
    half = np.array([5, 5, 5])
    sky = np.array([5.0, 5.0, 5.0])
    off = integrated_ratio(model, data, w, stars, half, sky)
    assert np.isnan(off[0])                       # box off the image
    assert np.isnan(off[1])                       # sum(data - sky) == 0
    allnan = np.full((161, 161), np.nan)
    assert np.isnan(integrated_ratio(model, allnan, w, stars, half, sky)[1])
    with pytest.raises(ValueError, match="one pixel grid"):
        integrated_ratio(model[:-1], data, w, stars, half, sky)


def test_star_halfwidth_clips_at_both_ends():
    nn = np.array([0.1, 0.62, 3.0, 9.18, 50.0, np.inf])
    assert list(star_halfwidth_px(nn, PIX_ARCSEC)) == [3, 4, 12, 12, 12, 12]


def test_nearest_neighbour_arcsec_excludes_the_star_itself():
    stars = SkyCoord([10.0, 10.0 + 1.0 / 3600, 10.0 + 10.0 / 3600] * u.deg,
                     [0.0, 0.0, 0.0] * u.deg)
    assert nearest_neighbour_arcsec(stars) == pytest.approx([1.0, 1.0, 9.0], abs=1e-3)


def _record_for(ratio_a=0.9):
    w = _tan_wcs()
    stars = _sky_at(w, [60, 100], [80, 80])
    return star_record(stars, np.array([2.5, 2.5]), np.array([12, 12]),
                       np.array([ratio_a, np.nan]), np.array([0.05, 0.1]),
                       {"data": "2026-08-21T08:05:17"}, "abc1234")


def test_record_diff_is_empty_for_equal_input_and_ignores_the_git_sha(tmp_path):
    path = tmp_path / "rec.json"
    write_record(_record_for(), str(path))
    assert record_diff(_record_for(), str(path)) == ""
    other = _record_for()
    other["git_sha"] = "ffffff0"
    assert record_diff(other, str(path)) == ""


def test_record_diff_names_the_star_that_moved(tmp_path):
    path = tmp_path / "rec.json"
    write_record(_record_for(0.9), str(path))
    diff = record_diff(_record_for(1.2), str(path))
    assert "-   \"model_ratio\": 0.9" in diff and "+   \"model_ratio\": 1.2" in diff


def test_record_diff_reports_a_missing_or_corrupt_file_without_raising(tmp_path):
    assert "no committed record" in record_diff(_record_for(), str(tmp_path / "x.json"))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert "unreadable" in record_diff(_record_for(), str(bad))


def test_record_is_valid_json_with_null_for_nan(tmp_path):
    path = tmp_path / "rec.json"
    write_record(_record_for(), str(path))
    rec = json.loads(path.read_text())
    assert rec["stars"][1]["model_ratio"] is None
    assert rec["stars"][0]["halfwidth_px"] == 12


# --- box edges, dropped pixels, halfwidth hardening ----------------------------

def test_integrated_ratio_box_edges_are_inclusive_on_both_sides():
    """A box that fits exactly is measured; one pixel further out is not."""
    n, h = 161, 5
    w = _tan_wcs(n)
    data = np.full((n, n), 6.0)
    model = np.ones((n, n))
    # box [x - h, x + h] fits for h <= x <= n - 1 - h
    xs = [h - 1, h, n - 1 - h, n - h]
    stars = _sky_at(w, xs, [80] * 4)
    out = integrated_ratio(model, data, w, stars, np.full(4, h), np.full(4, 5.0))
    assert np.isnan(out[0]) and np.isnan(out[3])
    assert out[1] == pytest.approx(1.0) and out[2] == pytest.approx(1.0)
    # the same on the y axis
    stars = _sky_at(w, [80] * 4, xs)
    out = integrated_ratio(model, data, w, stars, np.full(4, h), np.full(4, 5.0))
    assert list(np.isnan(out)) == [True, False, False, True]


def test_integrated_ratio_drops_nan_pixels_from_both_sums_and_counts_them():
    """A NaN data pixel takes the model's flux there out of the numerator, so
    the ratio is the finite-pixel ratio; the count says how many were lost."""
    n, h = 161, 5
    w = _tan_wcs(n)
    data = np.full((n, n), 6.0)
    model = np.ones((n, n))
    model[80, 80] = 50.0
    stars = _sky_at(w, [80], [80])
    half, sky = np.array([h]), np.array([5.0])
    full, nd = integrated_ratio(model, data, w, stars, half, sky,
                                return_dropped=True)
    assert nd[0] == 0 and full[0] == pytest.approx((121 + 49) / 121)
    data[80, 80] = np.nan
    ratio, nd = integrated_ratio(model, data, w, stars, half, sky,
                                 return_dropped=True)
    assert nd[0] == 1
    assert ratio[0] == pytest.approx(120 / 120)       # the 50 left with the pixel
    # a NaN in the MODEL drops the data pixel as well
    data[80, 80], model[80, 80] = 6.0, np.nan
    ratio, nd = integrated_ratio(model, data, w, stars, half, sky,
                                 return_dropped=True)
    assert nd[0] == 1 and ratio[0] == pytest.approx(1.0)


def test_integrated_ratio_is_nan_above_the_dropped_fraction():
    n, h = 161, 5
    w = _tan_wcs(n)
    model = np.ones((n, n))
    stars = _sky_at(w, [80], [80])
    half, sky = np.array([h]), np.array([5.0])
    box = (2 * h + 1) ** 2
    limit = int(MAX_DROPPED_FRACTION * box)            # 24 of 121

    def with_dropped(k):
        flat = np.full(box, 6.0)
        flat[:k] = np.nan
        d = np.full((n, n), 6.0)
        d[75:86, 75:86] = flat.reshape(11, 11)         # the star's whole box
        return d

    ratio, nd = integrated_ratio(model, with_dropped(limit), w, stars, half,
                                 sky, return_dropped=True)
    assert nd[0] == limit and np.isfinite(ratio[0])    # at the limit: measured
    ratio, nd = integrated_ratio(model, with_dropped(limit + 1), w, stars,
                                 half, sky, return_dropped=True)
    assert nd[0] == limit + 1 and np.isnan(ratio[0])   # one more: not a ratio


def test_dropped_count_is_nan_where_the_box_is_off_the_image():
    w = _tan_wcs()
    stars = _sky_at(w, [2], [80])
    data = np.full((161, 161), 6.0)
    ratio, nd = integrated_ratio(np.ones_like(data), data, w, stars,
                                 np.array([5]), np.array([5.0]),
                                 return_dropped=True)
    assert np.isnan(ratio[0]) and np.isnan(nd[0])


def test_star_halfwidth_maps_a_nan_separation_to_the_floor():
    out = star_halfwidth_px(np.array([np.nan, 0.62, np.inf]), PIX_ARCSEC)
    assert list(out) == [3, 4, 12]
    assert out.dtype.kind == "i"


def test_sky_annulus_starts_at_the_default_inner_radius():
    """A halo out to 37 px must not be read as sky: the annulus has to start
    at SKY_INNER_PX (30), not inside the halo."""
    w = _tan_wcs()
    c = 80
    stars = _sky_at(w, [c], [c])
    data = np.full((161, 161), 5.0)
    data[c - 37:c + 38, c - 37:c + 38] = 50.0
    assert annulus_sky(data, w, stars, np.array([12]))[0] == pytest.approx(5.0)


def test_record_carries_the_dropped_pixel_count():
    w = _tan_wcs()
    stars = _sky_at(w, [60, 100], [80, 80])
    rec = star_record(stars, np.array([2.5, 2.5]), np.array([12, 12]),
                      np.array([0.9, np.nan]), np.array([0.05, 0.1]),
                      {"data": "x"}, "abc1234", np.array([2.0, np.nan]))
    assert [r["n_dropped_px"] for r in rec["stars"]] == [2, None]
    assert all(r["n_dropped_px"] is None for r in _record_for()["stars"])


# --- ensemble policy functions and the record plumbing -------------------------

def _ensemble(median, n=40):
    """``n`` isolated stars (box at the cap) with ``median`` as the median."""
    return np.full(n, median), np.full(n, BOX_HALFWIDTH_CAP_PX)


def test_median_band_has_both_edges():
    for good in (0.7, 0.88, 1.3):
        assert integrated_median_failure(*_ensemble(good)) is None
    assert "outside" in integrated_median_failure(*_ensemble(0.69))
    assert "outside" in integrated_median_failure(*_ensemble(1.31))


def test_median_is_taken_over_the_isolated_stars_only():
    ratio, half = _ensemble(0.88, 30)
    # stars in close pairs (small box) with wild ratios must not move it
    ratio = np.append(ratio, np.full(40, 5.0))
    half = np.append(half, np.full(40, 4))
    assert integrated_median_failure(ratio, half) is None
    # ... nor stars with no ratio
    ratio = np.append(ratio, np.full(40, np.nan))
    half = np.append(half, np.full(40, BOX_HALFWIDTH_CAP_PX))
    assert integrated_median_failure(ratio, half) is None


def test_median_needs_enough_isolated_stars():
    ratio, half = _ensemble(0.88, INTEGRATED_MIN_STARS - 1)
    assert "need" in integrated_median_failure(ratio, half)
    ratio, half = _ensemble(0.88, INTEGRATED_MIN_STARS)
    assert integrated_median_failure(ratio, half) is None


def test_gross_bounds_have_both_edges():
    base = np.full(40, 1.0)
    assert integrated_gross_failure(base) is None
    # 10% of 40 = 4 outside is allowed, 5 is not; above and below
    for bad in (0.29, 2.01):
        ok = base.copy()
        ok[:4] = bad
        assert integrated_gross_failure(ok) is None
        ok[:5] = bad
        assert "outside" in integrated_gross_failure(ok)
    inside = base.copy()
    inside[:10] = [0.3, 2.0] * 5                       # the bounds are inclusive
    assert integrated_gross_failure(inside) is None


def test_gross_test_ignores_nan_and_needs_enough_stars():
    ratio = np.append(np.full(25, 1.0), np.full(20, np.nan))
    assert integrated_gross_failure(ratio) is None
    assert "need" in integrated_gross_failure(np.full(INTEGRATED_MIN_STARS - 1, 1.0))


def test_integrated_products_record_the_residual_fraction_and_dropped_count():
    """``int_resid`` is the RESIDUAL summed over the model's box and sky, not a
    second copy of the model ratio."""
    n, c = 161, 80
    w = _tan_wcs(n)
    star = _sky_at(w, [c], [c])
    data = 5.0 + _gauss(n, c, c, 2.0, 1000.0)
    model = 0.5 * _gauss(n, c, c, 2.0, 1000.0)
    resid = 0.25 * _gauss(n, c, c, 2.0, 1000.0)
    data[c, c] = np.nan
    out = policy._integrated_products(data, resid, model, w, star,
                                      {"data": "x"})
    assert out["int_ratio"][0] == pytest.approx(0.5, abs=0.02)
    assert out["int_resid"][0] == pytest.approx(0.25, abs=0.02)
    assert out["n_dropped"][0] == 1
    row = out["record"]["stars"][0]
    assert row["resid_fraction"] == pytest.approx(0.25, abs=0.02)
    assert row["model_ratio"] == pytest.approx(0.5, abs=0.02)
    assert row["n_dropped_px"] == 1


def test_record_update_needs_exactly_one():
    assert record_update_requested({RECORD_UPDATE_ENV: "1"})
    for value in ("0", "false", "", "yes", "true", "2"):
        assert not record_update_requested({RECORD_UPDATE_ENV: value})
    assert not record_update_requested({})


def test_record_update_reads_the_process_environment_by_default(monkeypatch):
    monkeypatch.setenv(RECORD_UPDATE_ENV, "1")
    assert record_update_requested()
    monkeypatch.setenv(RECORD_UPDATE_ENV, "0")
    assert not record_update_requested()
    monkeypatch.delenv(RECORD_UPDATE_ENV)
    assert not record_update_requested()


def test_report_record_only_writes_when_asked(tmp_path):
    path = tmp_path / "rec.json"
    write_record(_record_for(0.9), str(path))
    before = path.read_text()
    text = report_record(_record_for(1.2), str(path), update=False)
    assert path.read_text() == before
    assert "regenerated" in text and "1.2" in text
    assert "(no differences)" in report_record(_record_for(0.9), str(path))
    text = report_record(_record_for(1.2), str(path), update=True)
    assert text.startswith("wrote ")
    assert path.read_text() != before
    assert json.loads(path.read_text())["stars"][0]["model_ratio"] == 1.2
