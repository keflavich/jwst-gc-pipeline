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
