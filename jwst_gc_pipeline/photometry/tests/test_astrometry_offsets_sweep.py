"""Window-sweep + two-reference hardening of the sanctioned offset helper.

Regression guard for the brick-1182 v001 trap: a LARGE rigid offset must be
recovered (not read as ~0/incoherent) because a fixed narrow window has no true
pairs inside it.
"""
import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord

from jwst_gc_pipeline.photometry.astrometry_offsets import (
    measure_offset, agree_across_references, KDTreeReference)


def _field(n=800, seed=0):
    rng = np.random.RandomState(seed)
    ra = 266.4 + rng.rand(n) * 0.05
    dec = -28.9 + rng.rand(n) * 0.05
    return ra, dec


def _shift(ra, dec, dra_arcsec, ddec_arcsec):
    cosd = np.cos(np.radians(dec))
    return SkyCoord((ra + dra_arcsec / 3600.0 / cosd) * u.deg,
                    (dec + ddec_arcsec / 3600.0) * u.deg)


def test_small_offset_recovered_at_narrow_window():
    ra, dec = _field()
    a = SkyCoord(ra * u.deg, dec * u.deg)
    b = _shift(ra, dec, 0.5, 0.3)
    r = measure_offset(a, b)
    assert r["ok"]
    assert abs(r["dra"] - 500) < 40 and abs(r["ddec"] - 300) < 40
    assert not r["swept"]           # offset (0.5") < initial window, no widening needed


def test_large_offset_recovered_by_sweep():
    # The brick-1182 v001 case: a ~20" rigid offset. A 3" window sees noise; the
    # sweep must widen and recover the true peak.
    ra, dec = _field(n=1500, seed=3)
    a = SkyCoord(ra * u.deg, dec * u.deg)
    b = _shift(ra, dec, 15.4, -13.5)          # ~20.4" offset
    r = measure_offset(a, b)                    # sweep ON by default
    assert r["ok"], f"large offset not recovered: {r}"
    assert abs(r["dra"] - 15400) < 300 and abs(r["ddec"] + 13500) < 300
    assert r["swept"] and r["window_arcsec"] >= 30.0


def test_no_sweep_misses_large_offset():
    # Prove the sweep is what saves it: with sweep off, a 3" window does NOT find
    # the 20" tie (this is the exact failure the sweep guards).
    ra, dec = _field(n=1500, seed=3)
    a = SkyCoord(ra * u.deg, dec * u.deg)
    b = _shift(ra, dec, 15.4, -13.5)
    r = measure_offset(a, b, sweep=False)
    assert (r is None) or (not r["ok"]) or (abs(r["dra"] - 15400) > 1000)


def test_two_reference_agreement():
    ra, dec = _field(n=1000, seed=7)
    a = SkyCoord(ra * u.deg, dec * u.deg)
    # both references are the same clean tie -> agree
    ref = _shift(ra, dec, -1.66, -0.98)
    sparse = ref[::5]
    out = agree_across_references(a, ref, sparse, label_a="dense", label_b="sparse")
    assert out["agree"], out
    assert out["sep_mas"] < 100


# ---------------------------------------------------------------------------
# issue #1232 -- the window-scaled bin and the significance gate
# ---------------------------------------------------------------------------

def test_window_bin_is_capped_above_15_arcsec():
    from jwst_gc_pipeline.photometry import astrometry_offsets as ao
    got = [ao._window_bin_arcsec(w, 0.02) for w in (3.0, 10.0, 30.0, 60.0, 300.0)]
    assert got == pytest.approx([0.02, 10.0 / 150.0, 0.1, 0.1, 0.1])
    # the window/3000 term binds only beyond 300 arcsec
    assert ao._window_bin_arcsec(600.0, 0.02) == pytest.approx(0.2)
    # a caller-supplied coarser base bin still wins
    assert ao._window_bin_arcsec(60.0, 0.25) == pytest.approx(0.25)


def _dense_pair(shift_arcsec, scatter_arcsec, seed, rho=1.5, size=50.0,
                completeness=0.87):
    """Reference and query lists of one uniform field (stars/arcsec^2 ``rho``,
    ``size`` arcsec square); the query is the reference moved by
    ``-shift`` with per-star scatter and ``completeness``, so ``measure_offset
    (query, reference)`` should read ``+shift``."""
    rng = np.random.default_rng(seed)
    n = int(rho * size * size)
    x = rng.uniform(0, size, n)
    y = rng.uniform(0, size, n)
    keep = rng.random(n) < completeness
    qx = x[keep] + rng.normal(0, scatter_arcsec, keep.sum()) - shift_arcsec[0]
    qy = y[keep] + rng.normal(0, scatter_arcsec, keep.sum()) - shift_arcsec[1]

    def sky(px, py):
        return SkyCoord((266.4 + px / 3600.0 / np.cos(np.radians(-28.9))) * u.deg,
                        (-28.9 + py / 3600.0) * u.deg)
    return sky(qx, qy), KDTreeReference(sky(x, y))


def test_dense_field_20_arcsec_offset_is_ok():
    a, ref = _dense_pair((14.0, 14.0), 0.04, seed=1)
    r = measure_offset(a, ref)
    assert r["ok"], r
    assert np.hypot(r["dra"] - 14000, r["ddec"] - 14000) < 300
    assert r["window_arcsec"] >= 30.0
    assert r["bin_arcsec"] == pytest.approx(0.1)
    assert r["contrast"] > 10.0
    assert r["peak_significance"] >= r["min_peak_significance"]


@pytest.mark.parametrize("scatter", [0.1])
def test_dense_field_20_arcsec_offset_confirms(scatter):
    # 100 mas scatter: the capped bin shrinks the confirmation tolerance to
    # 300 mas, which a broadened peak must still meet
    a, ref = _dense_pair((14.0, 14.0), scatter, seed=2)
    r = measure_offset(a, ref, confirm_windows=True)
    assert r["window_consistent"] is True, r.get("window_confirmation")
    assert r["ok"] and not r["alias_rejected"]


@pytest.mark.parametrize("seed", range(2))
def test_dense_field_without_a_tie_is_not_ok_at_wide_windows(seed):
    # Two independent uniform fields of the same density: every peak is noise.
    # The fine wide-window bin leaves a median occupied bin of several pairs,
    # so the contrast floor alone is not enough here.
    a, _ = _dense_pair((0.0, 0.0), 0.04, seed=seed)
    _, ref = _dense_pair((0.0, 0.0), 0.04, seed=1000 + seed)
    r = measure_offset(a, ref)
    assert r is None or not r["ok"], r


def test_hist_peak_significance_rejects_a_dense_noise_maximum():
    from jwst_gc_pipeline.photometry import astrometry_offsets as ao
    rng = np.random.default_rng(1232)
    m, bw = 60.0, 0.1
    n = 4_000_000
    r = m * np.sqrt(rng.random(n))
    th = rng.uniform(0, 2 * np.pi, n)
    out = ao._hist_peak(r * np.cos(th), r * np.sin(th), m, bw)
    sig = out[9]
    assert len(out) == 10
    assert sig < ao.min_peak_significance(m, bw)
    # the same background plus 400 coincident pairs is far above the floor
    out = ao._hist_peak(np.concatenate([r * np.cos(th), np.full(400, 20.0)]),
                        np.concatenate([r * np.sin(th), np.full(400, -7.0)]),
                        m, bw)
    assert out[9] > ao.min_peak_significance(m, bw)
    assert abs(out[0] - 20000) < 100 and abs(out[1] + 7000) < 100


def _weak_small_tie(seed, rho=1.0, size=150.0, n_query=100, n_true=30):
    """A 100-star query list, 30 of which are in a dense reference at a 36 mas
    offset with 40 mas scatter: the per-tile regime, a weak tie with few
    matched stars in a dense field."""
    rng = np.random.default_rng(seed)
    n = int(rho * size * size)
    x = rng.uniform(0, size, n)
    y = rng.uniform(0, size, n)
    idx = rng.choice(n, n_true, replace=False)
    qx = np.concatenate([x[idx] + 0.03 + rng.normal(0, 0.04, n_true),
                         rng.uniform(0, size, n_query - n_true)])
    qy = np.concatenate([y[idx] - 0.02 + rng.normal(0, 0.04, n_true),
                         rng.uniform(0, size, n_query - n_true)])

    def sky(px, py):
        return SkyCoord((266.0 + px / 3600.0 / np.cos(np.radians(-28.8))) * u.deg,
                        (-28.8 + py / 3600.0) * u.deg)
    return sky(qx, qy), sky(x, y)


@pytest.mark.parametrize("seed", [0, 1, 3])
def test_weak_small_tie_is_not_lost_to_the_wide_window_floor(seed):
    # The 0.1" bin of the 30" window gathers the scattered pairs and out-scores
    # the 10" window on contrast, but its peak sits just under the floor sized
    # for its 2.8M bins.  The narrower window reads the same peak and passes
    # its own gate, so the tie stays ok at the narrower window.
    a, b = _weak_small_tie(seed)
    r = measure_offset(a, b)
    assert r["ok"], r
    assert r["window_arcsec"] <= 15.0
    assert r["min_peak_significance"] is None
    assert np.hypot(r["dra"] + 30.0, r["ddec"] - 20.0) < 60.0


# --- the narrower-window fallback, one decision per test ---------------------
# ``_measure_at_window`` is replaced by a table of per-window readings so each
# branch of the fallback is driven directly instead of through a seeded field.
# The significance floors are ``min_peak_significance(w, 0.1)``: 8.01 at 30",
# 8.28 at 60".

def _reading(window, dra, ddec, contrast, sig, bin_arcsec=None):
    from jwst_gc_pipeline.photometry import astrometry_offsets as ao
    return dict(dra=float(dra), ddec=float(ddec), off=float(np.hypot(dra, ddec)),
                npairs=5000, contrast=float(contrast), window_arcsec=float(window),
                dra_err=5.0, ddec_err=5.0, n_peak=20, peak_margin=2.0,
                peak_significance=float(sig),
                bin_arcsec=(ao._window_bin_arcsec(window, 0.02)
                            if bin_arcsec is None else bin_arcsec))


def _measure_from_table(monkeypatch, readings, windows=(3.0, 10.0, 30.0, 60.0)):
    from jwst_gc_pipeline.photometry import astrometry_offsets as ao
    by_window = {r["window_arcsec"]: r for r in readings}
    monkeypatch.setattr(
        ao, "_measure_at_window",
        lambda a, b, w, bin_arcsec, min_pairs: (
            dict(by_window[w]) if w in by_window else None))
    return ao.measure_offset(None, None, sweep=True,
                             sweep_windows=windows[1:])


def test_fallback_takes_the_narrower_window_that_reads_the_same_peak(monkeypatch):
    r = _measure_from_table(monkeypatch, [
        _reading(3, 100, 50, 3.0, 0.0), _reading(10, 100, 50, 6.0, 0.0),
        _reading(60, 120, 40, 9.0, 5.0)])
    assert r["ok"] and r["window_arcsec"] == 10.0
    assert r["min_peak_significance"] is None


def test_fallback_picks_the_highest_contrast_of_the_narrower_windows(monkeypatch):
    r = _measure_from_table(monkeypatch, [
        _reading(3, 100, 50, 6.0, 0.0), _reading(10, 100, 50, 9.0, 0.0),
        _reading(60, 100, 50, 12.0, 5.0)])
    assert r["ok"] and r["window_arcsec"] == 10.0 and r["contrast"] == 9.0
    r = _measure_from_table(monkeypatch, [
        _reading(3, 100, 50, 9.0, 0.0), _reading(10, 100, 50, 6.0, 0.0),
        _reading(60, 100, 50, 12.0, 5.0)])
    assert r["ok"] and r["window_arcsec"] == 3.0 and r["contrast"] == 9.0


def test_fallback_needs_the_narrower_window_to_pass_its_own_gate(monkeypatch):
    # contrast 4 < 5: the 3" window reads the same peak but is itself refused
    r = _measure_from_table(monkeypatch, [
        _reading(3, 100, 50, 4.0, 0.0), _reading(60, 100, 50, 9.0, 5.0)])
    assert not r["ok"] and r["window_arcsec"] == 60.0


def test_fallback_never_goes_to_a_wider_window(monkeypatch):
    # the 30" window wins on contrast and is refused; the 60" window passes
    # its own floor on the same peak but is not narrower, so it cannot certify it
    r = _measure_from_table(monkeypatch, [
        _reading(30, 100, 50, 20.0, 5.0), _reading(60, 100, 50, 8.0, 9.0)])
    assert not r["ok"] and r["window_arcsec"] == 30.0


def test_fallback_is_not_tried_when_the_winner_is_already_ok(monkeypatch):
    r = _measure_from_table(monkeypatch, [
        _reading(3, 100, 50, 6.0, 0.0), _reading(60, 100, 50, 9.0, 12.0)])
    assert r["ok"] and r["window_arcsec"] == 60.0
    assert r["peak_significance"] == 12.0


def test_fallback_tolerance_is_three_thousand_of_the_coarser_bin(monkeypatch):
    # bin 0.1" -> 300 mas: a peak 200 mas from the refused wide peak is the same
    # peak, one 400 mas away is a different one and stays refused.  (The 100 mas
    # floor alone would reject the first, an unbounded tolerance accept the second.)
    r = _measure_from_table(monkeypatch, [
        _reading(10, 100, 50, 6.0, 0.0), _reading(60, 300, 50, 9.0, 5.0)])
    assert r["ok"] and r["window_arcsec"] == 10.0
    r = _measure_from_table(monkeypatch, [
        _reading(10, 100, 50, 6.0, 0.0), _reading(60, 500, 50, 9.0, 5.0)])
    assert not r["ok"] and r["window_arcsec"] == 60.0


# --- the refusal text names the gate that refused ----------------------------

def test_refusal_text_names_the_significance_floor_when_contrast_clears(monkeypatch):
    # contrast 11.0 clears the 5.0 floor; the 60" peak reads 4.57 sigma against
    # 8.28.  "contrast 11.0 < 5.0" would send the reader to the wrong cause.
    from jwst_gc_pipeline.photometry import astrometry_offsets as ao
    by_window = {60.0: _reading(60, 100, 50, 11.0, 4.57)}
    monkeypatch.setattr(ao, "_measure_at_window",
                        lambda a, b, w, bin_arcsec, min_pairs: (
                            dict(by_window[w]) if w in by_window else None))
    with pytest.raises(ao.NoCoherentTieError) as exc:
        ao.measure_offset(None, None, maxsep=60.0, sweep=False,
                          raise_on_low_contrast=True, context="probe")
    msg = str(exc.value)
    assert "clears 5.0" in msg and "4.57 sigma" in msg and "8.28" in msg, msg
    assert "< 5.0" not in msg, msg


def test_refusal_text_keeps_the_contrast_wording_when_contrast_is_low(monkeypatch):
    from jwst_gc_pipeline.photometry import astrometry_offsets as ao
    by_window = {3.0: _reading(3, 100, 50, 2.0, 0.0)}
    monkeypatch.setattr(ao, "_measure_at_window",
                        lambda a, b, w, bin_arcsec, min_pairs: (
                            dict(by_window[w]) if w in by_window else None))
    with pytest.raises(ao.NoCoherentTieError) as exc:
        ao.measure_offset(None, None, sweep=False, raise_on_low_contrast=True)
    assert "best contrast 2.0 < 5.0" in str(exc.value)
