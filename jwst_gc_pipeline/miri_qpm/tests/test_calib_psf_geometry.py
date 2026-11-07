"""Synthetic forward model through the calibration; RL; the static pane geometry."""
import os

import numpy as np
import pytest

from jwst_gc_pipeline.miri_qpm import calib, geometry, psf

REF = calib.refdata()


def _forward(S, k=0.26, ngroups=6, tgroup=2.775):
    """rate whose R1 inverse is the sky S (crop frame)."""
    reg = REF.region
    Y = REF.flat * (REF.a[reg] * S + REF.b[reg])
    R = np.where(reg > 0, Y / k, 0.0)
    return R, np.zeros(reg.shape, np.uint32), np.full(reg.shape, 0.01), k, ngroups, tgroup


def test_forward_model_recovers_sky_and_masks():
    rng = np.random.default_rng(1)
    reg = REF.region
    S = 50 + 300 * rng.random(reg.shape)
    R, dq, err, k, ng, tg = _forward(S)
    good_pane = (reg > 0) & (REF.badpix != 1)
    iy, ix = np.argwhere(good_pane)[[100, 2000, 30000]].T
    dq[iy[0], ix[0]] = 2                                    # SATURATED
    R[iy[1], ix[1]] = 1.01 * calib.FLUENCE_DN / (ng * tg)   # near-saturated ramp
    dq[iy[2], ix[2]] = 1                                    # DO_NOT_USE
    Q, var, var_rn, qdq = calib.calibrate(R, dq, err, None, k, ng, tg, REF)
    assert qdq[iy[0], ix[0]] & calib.DQ_SATURATED and qdq[iy[1], ix[1]] & calib.DQ_FLUENCE
    assert qdq[iy[2], ix[2]] & calib.DQ_RATE_OTHER
    bad = (REF.badpix == 1) & (reg > 0)
    assert np.isnan(Q[bad]).all() and (qdq[bad] & calib.DQ_BADPIX).all()
    # STScI-like: only DO_NOT_USE (and bad pixels) are blank; SATURATED with a finite slope and >= 40 kDN are kept,
    # flagged, and EST_DROP removes them for the estimators (which see the pre-change mask)
    assert np.isnan(Q[reg == 0]).all() and np.isfinite(Q[iy[:2], ix[:2]]).all() and np.isnan(Q[iy[2], ix[2]])
    assert not qdq[iy[0], ix[0]] & calib.DQ_DO_NOT_USE and qdq[iy[2], ix[2]] & calib.DQ_DO_NOT_USE
    ok = np.isfinite(Q)
    assert ok.sum() == good_pane.sum() - 1 and np.isfinite(var[iy[:2], ix[:2]]).all()
    np.testing.assert_allclose(Q[ok & ((qdq & calib.DQ_FLUENCE) == 0)], S[ok & ((qdq & calib.DQ_FLUENCE) == 0)],
                               rtol=2e-6)                    # float32 output
    est = ok & ((qdq & calib.EST_DROP) == 0)
    assert est.sum() == good_pane.sum() - 3
    assert (var[ok] > 0).all() and (var_rn[ok] > 0).all()


def test_rl_conserves_flux_and_ignores_kernel_scale():
    rng = np.random.default_rng(2)
    d = 100 + 20 * rng.random((60, 80))
    d[30, 40] += 5000
    K = REF.kernels[3]
    u1 = psf.richardson_lucy(d, K, 30)
    u2 = psf.richardson_lucy(d, 2.7 * K, 30)
    np.testing.assert_allclose(u1, u2, rtol=1e-10)          # invariant to the kernel scale (never divide by sum K)
    inner = (slice(5, -5), slice(15, -15))
    assert abs(u1[inner].sum() / d[inner].sum() - 1) < 2e-3


def test_rl_is_neutral_on_flat_sky():
    """A flat sky stays flat everywhere, edges included (needs the fill_nan fix and the edge padding)."""
    reg = REF.region
    Q = np.where(reg > 0, 123.0, np.nan).astype(np.float32)
    Q[300, 100] = Q[301:306, 50:56] = np.nan               # an isolated hole and a 30-px NaN cluster
    out = psf.deconvolve(Q, reg, REF.kernels)
    fin = np.isfinite(Q)
    assert np.array_equal(np.isfinite(out), fin)           # NaN in = NaN out
    np.testing.assert_allclose(out[fin], 123.0, rtol=1e-5)


def test_fill_nan_stays_within_valid_range():
    rng = np.random.default_rng(5)
    a = 100 + rng.random((80, 120))
    valid = np.zeros(a.shape, bool)
    valid[20:60, 30:90] = True
    f = psf.fill_nan(a, valid)
    assert f.min() >= a[valid].min() - 1e-9 and f.max() <= a[valid].max() + 1e-9


def test_edge_margin_counts_large_holes_only():
    reg = REF.region
    Q = np.where(reg > 0, 1.0, np.nan)
    yc, xc = np.argwhere(reg == 2).mean(0).astype(int)
    Q[yc, xc] = np.nan                                      # isolated hole: no margin
    Q[yc + 40:yc + 45, xc:xc + 6] = np.nan                  # 30-px cluster: margin
    m = psf.edge_margin(Q, reg, 3)
    assert not m[yc - 2:yc + 3, xc - 2:xc + 3].any()
    assert m[yc + 38, xc + 2] and m[yc + 47, xc + 2] and not m[yc + 49, xc + 2]
    yy, xx = np.nonzero(reg == 2)
    assert m[yy.min(), xx.min() + 50] and m[yy.min() + 2, xx.min() + 50] and not m[yy.min() + 3, xx.min() + 50]
    # round (Euclidean) margin: the diagonal corner (3, 3) of the cluster is outside it, inside the old chessboard one
    c = psf.edge_margin(Q, reg, 3, round_=False)
    assert not m[yc + 47, xc + 8] and c[yc + 47, xc + 8] and m[yc + 46, xc + 7] and (m <= c).all()


def test_variance_scaling():
    reg = REF.region
    v = psf.scale_variance(np.ones(reg.shape), reg)
    assert [float(np.unique(v[reg == g])[0]) for g in (1, 2, 3)] == [1.3, 1.3, 3.1]


# fullframe_preview.region_shift on region_models.npy at these points (computed once from the original file)
EXPECTED = {
    1: ([22.0, 130.25, 241.0], [-4.371628410617447, -3.6226500664888825, -2.8733290744107114],
        [-0.8497902007931153, -0.6079191636745307, -0.36575862556059413]),
    2: ([244.0, 350.0, 462.0], [-3.742010938071253, -3.2237868651404273, -2.7000281137282327],
        [-0.6148269942814539, -0.30100781898790174, 0.020386745401571114]),
    3: ([466.0, 575.5, 682.0], [-3.11477171324513, -2.8662265112254612, -2.6207126259043045],
        [-0.19416345553261877, -0.010825070678541479, 0.16900611375342806]),
}


def test_region_shift_bit_exact():
    x = np.array([9.0, 120.5, 231.0])
    for g, (y, ex, ey) in EXPECTED.items():
        dx, dy = geometry.region_shift(REF.models, g, x, np.array(y))
        assert dx.tolist() == ex and dy.tolist() == ey


@pytest.mark.skipif(not os.environ.get("MIRI_QPM_REGION_MODELS"),
                    reason="set MIRI_QPM_REGION_MODELS to the original region_models.npy")
def test_region_shift_against_original():
    m = np.load(os.environ["MIRI_QPM_REGION_MODELS"], allow_pickle=True).item()
    rng = np.random.default_rng(3)
    x, y = rng.uniform(0, 1032, 500), rng.uniform(0, 1024, 500)
    for g in (1, 2, 3):
        cx, cy = m[g]
        A = np.c_[np.ones(len(x)), (x - 500) / 500, (y - 500) / 500]   # fullframe_preview.region_shift
        dx, dy = geometry.region_shift(REF.models, g, x, y)
        assert np.array_equal(dx, A @ cx) and np.array_equal(dy, A @ cy)


def test_pane_positions_apply_shift():
    X, Y = geometry.pane_positions(REF)
    yy, xx = np.nonzero(REF.region == 1)
    dx, dy = geometry.region_shift(REF.models, 1, xx + REF.x0 + 0.0, yy + REF.y0 + 0.0)
    assert np.array_equal(X[yy, xx], xx + REF.x0 + dx) and np.array_equal(Y[yy, xx], yy + REF.y0 + dy)
    z = REF.region == 0
    assert not np.array_equal(X[z], np.nonzero(z)[1] + REF.x0 + 0.0)    # non-pane pixels: nearest pane's shift


def test_pane_pixmap_continuous_for_drizzle():
    """Drizzling the panes with pane_positions as the pixmap (identity sky) gives no negative output weight, and
    each pane's outermost pixels keep their full footprint (as with a continuous pixmap)."""
    from drizzle.resample import Drizzle
    from scipy import ndimage
    reg = REF.region
    for g in (1, 2, 3):        # panes never touch (incl. diagonally), so "nearest pane" is unique at every edge
        assert not (ndimage.binary_dilation(reg == g, np.ones((3, 3))) & (reg > 0) & (reg != g)).any()
    X, Y = geometry.pane_positions(REF)
    d = Drizzle(kernel="square", out_shape=(reg.shape[0] + 20, reg.shape[1] + 20), disable_ctx=True)
    d.add_image(np.ones(reg.shape, np.float32), exptime=1.0, in_units="cps",
                pixmap=np.dstack([X - REF.x0 + 10, Y - REF.y0 + 10]), weight_map=(reg > 0).astype(np.float32))
    assert d.out_wht.min() > -1e-6
    assert abs(d.out_wht.sum() - (reg > 0).sum()) < 1e-3 * (reg > 0).sum()
