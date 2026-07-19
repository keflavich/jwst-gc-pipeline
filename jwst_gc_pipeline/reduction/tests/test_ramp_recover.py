"""Unit tests for the ramp-SLOPE saturated-star recovery
(saturated_star_finding.ramp_slope_map / ramp_recover_saturated).

The design premise (validated on real frames 2026-07-18): cal(MJy/sr) is
proportional to the ramp-fit SLOPE, not group-0; saturation must be read from
the ramp DATA (the _ramp.fits GROUPDQ broadcasts SATURATED to every group), and
a per-pixel slope + LOCAL cal/slope calibration recovers the rim crowding-immune.
"""
import numpy as np
import pytest

from jwst_gc_pipeline.reduction.saturated_star_finding import (
    ramp_slope_map, ramp_recover_saturated)
try:
    from jwst.datamodels import dqflags
    PSAT = dqflags.pixel['SATURATED']
except (ImportError, KeyError):  # pragma: no cover
    PSAT = 2

CEIL = 50000.0


def _linear_ramp(rate, ng, ceiling=CEIL, pedestal=0.0):
    """A pixel's ramp DN per group: pedestal + rate*t, clipped at the well."""
    t = np.arange(ng, dtype=float)
    return np.minimum(pedestal + rate * t, ceiling)


def test_slope_recovers_true_rate_over_presat_groups():
    """A pixel that saturates partway up the ramp: the slope must equal the
    true pre-saturation rate (not be dragged down by the railed groups)."""
    ng = 7
    rate = 12000.0                         # rails at group ~4
    ramp = np.zeros((ng, 3, 3))
    ramp[:, 1, 1] = _linear_ramp(rate, ng)
    slope, n_good = ramp_slope_map(ramp, ceiling=CEIL)
    # only leading below-ceiling groups counted; slope ~ true rate
    assert n_good[1, 1] >= 2
    assert np.isfinite(slope[1, 1])
    assert abs(slope[1, 1] - rate) / rate < 0.05


def test_group0_railed_is_deep_core():
    """A pixel already at the well in group-0 has no usable groups -> NaN slope
    (deep core, unrecoverable from data)."""
    ng = 7
    ramp = np.full((ng, 3, 3), CEIL)       # railed from group 0
    slope, n_good = ramp_slope_map(ramp, ceiling=CEIL)
    assert n_good[1, 1] == 0
    assert not np.isfinite(slope[1, 1])


def test_saturation_read_from_data_not_broadcast_groupdq():
    """GROUPDQ that flags SATURATED on EVERY group (the real _ramp.fits pattern)
    must NOT zero out a recoverable pixel -- saturation comes from the data."""
    ng = 7
    ramp = np.zeros((ng, 3, 3))
    ramp[:, 1, 1] = _linear_ramp(9000.0, ng)
    gdq = np.full((ng, 3, 3), dqflags.group['SATURATED'], dtype=int)  # broadcast
    slope, n_good = ramp_slope_map(ramp, gdq, ceiling=CEIL)
    assert n_good[1, 1] >= 2 and np.isfinite(slope[1, 1])


def test_recover_fills_saturated_rim_with_slope_times_local_K():
    """End-to-end: a bright saturated star on a field of unsaturated calibrators
    with a known cal = K*slope. The recovered rim (NaN-blanked cal) must come
    back at ~K*slope, and the deep core (railed at g0) stays deep."""
    rng = np.random.default_rng(0)
    ny = nx = 96
    ng = 7
    K = 0.17
    # unsaturated calibrator field: random rates -> cal = K*slope (+small noise)
    rates = rng.uniform(200, 4000, size=(ny, nx))
    ramp = rates[None] * np.arange(ng)[:, None, None]
    ramp = np.minimum(ramp, CEIL)
    cal = K * rates * (1 + 0.02 * rng.standard_normal((ny, nx)))
    dq = np.zeros((ny, nx), int)
    # inject a saturated star at center: high rate -> rails partway
    cy = cx = 48
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - cy, xx - cx)
    star_rate = 60000.0 * np.exp(-(r ** 2) / (2 * 2.5 ** 2))
    ramp[:, r < 12] = np.minimum((star_rate[r < 12][None] *
                                  np.arange(ng)[:, None]), CEIL)
    # cal + DQ: saturated core NaN-blanked & flagged where group0 rails
    railed_g0 = ramp[0] >= CEIL
    sat_region = (r < 6)
    dq[sat_region] |= PSAT
    cal[sat_region] = np.nan
    recovered, rim, deep, Kg = ramp_recover_saturated(cal, dq, ramp,
                                                      slope_min=20.0,
                                                      cal_block=32)
    assert Kg == pytest.approx(K, rel=0.1)
    # rim pixels that had a measurable slope are filled (finite) and positive
    assert rim.sum() > 0
    assert np.isfinite(recovered[rim]).all()
    # a mid-rim saturated pixel with pre-sat groups is recovered near K*slope
    ring = sat_region & ~railed_g0
    if ring.any():
        slope, ngood = ramp_slope_map(ramp, ceiling=CEIL)
        rec_ring = recovered[ring & rim]
        exp_ring = (K * slope)[ring & rim]
        good = np.isfinite(rec_ring) & np.isfinite(exp_ring) & (exp_ring > 0)
        assert good.sum() > 0
        assert np.nanmedian(rec_ring[good] / exp_ring[good]) == pytest.approx(1.0, rel=0.15)


def test_no_ramp_is_noop():
    cal = np.ones((10, 10))
    dq = np.zeros((10, 10), int); dq[5, 5] |= PSAT
    rec, rim, deep, K = ramp_recover_saturated(cal, dq, None)
    assert np.array_equal(rec, cal) and rim.sum() == 0


def test_misflagged_low_slope_sat_pixel_not_made_negative():
    """Bug regression: a DQ-SATURATED pixel whose leading groups are flat/
    decreasing (read noise -- the SATURATED mask is any-group over-inclusive)
    has a small/negative slope.  It must NOT be 'recovered' to a negative flux;
    it goes to deep_core for the fallback instead."""
    rng = np.random.default_rng(1)
    ny = nx = 96
    ng = 7
    K = 0.17
    rates = rng.uniform(200, 4000, size=(ny, nx))
    ramp = np.minimum(rates[None] * np.arange(ng)[:, None, None], CEIL)
    cal = K * rates
    dq = np.zeros((ny, nx), int)
    # a genuinely saturated star (recoverable) ...
    ramp[:, 48, 48] = _linear_ramp(60000.0, ng)
    dq[48, 48] |= PSAT
    # ... and a MISFLAGGED "saturated" pixel with a flat/decreasing leading ramp
    ramp[:, 20, 20] = np.array([100., 50.] + [0.] * (ng - 2))
    dq[20, 20] |= PSAT
    recovered, rim, deep, Kg = ramp_recover_saturated(cal, dq, ramp)
    # no recovered pixel is negative
    assert np.all(recovered[rim] >= 0)
    # the misflagged pixel is NOT in the rim and IS routed to deep_core
    assert not rim[20, 20]
    assert deep[20, 20]


def test_kband_clamp_bounds_inflated_local_K():
    """Crowding clamp: a block whose local cal/slope is inflated far above the
    field-global K (a bright neighbour's cal over a modest slope) must NOT
    over-brighten the recovered rim.  With the clamp the recovered rim is bounded
    at ~Kglobal*k_band*slope; with the clamp disabled (k_band=0) it runs away to
    the full inflated ~6*Kglobal*slope."""
    rng = np.random.default_rng(7)
    ny = nx = 96
    ng = 7
    K = 0.17
    block = 32
    rates = rng.uniform(200, 4000, size=(ny, nx))
    ramp = np.minimum(rates[None] * np.arange(ng)[:, None, None], CEIL)
    cal = K * rates                                  # field-global cal = K*slope
    # inflate ONE 32x32 block's cal/slope ~6x (crowding contamination); it stays
    # a minority of the frame so Kglobal (median over all valid) is still ~K
    cal[0:block, 0:block] = 6.0 * K * rates[0:block, 0:block]
    dq = np.zeros((ny, nx), int)
    # a saturated star inside the inflated block, railing partway (measurable slope)
    cy = cx = 16
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - cy, xx - cx)
    ramp[:, r < 8] = np.minimum(
        (55000.0 * np.exp(-(r[r < 8] ** 2) / 10.0))[None] * np.arange(ng)[:, None], CEIL)
    dq[r < 4] |= PSAT
    cal[r < 4] = np.nan
    slope, _ = ramp_slope_map(ramp, ceiling=CEIL)

    rec_c, rim_c, _, Kg_c = ramp_recover_saturated(cal, dq, ramp, cal_block=block,
                                                   k_band=4.0)
    rec_u, rim_u, _, Kg_u = ramp_recover_saturated(cal, dq, ramp, cal_block=block,
                                                   k_band=0.0)  # clamp disabled
    assert Kg_c == pytest.approx(K, rel=0.2)          # global median ~ K, not 6K
    ring = rim_c & rim_u & np.isfinite(slope) & (slope > 20)
    assert ring.sum() > 0
    # clamp caps recovered rim at ~Kglobal*k_band*slope; unclamped runs to ~6*K*slope
    cap = Kg_c * 4.0 * slope[ring]
    assert np.all(rec_c[ring] <= cap * 1.05)
    # clamp materially reduces the over-bright rim vs disabled
    assert np.nanmedian(rec_c[ring]) < 0.8 * np.nanmedian(rec_u[ring])
    # a normal (uninflated) saturated star elsewhere is untouched by the clamp
    ramp2 = ramp.copy(); cal2 = K * rates.copy(); dq2 = np.zeros((ny, nx), int)
    r2 = np.hypot(yy - 70, xx - 70)
    ramp2[:, r2 < 8] = np.minimum(
        (55000.0 * np.exp(-(r2[r2 < 8] ** 2) / 10.0))[None] * np.arange(ng)[:, None], CEIL)
    dq2[r2 < 4] |= PSAT; cal2[r2 < 4] = np.nan
    a, rim_a, _, _ = ramp_recover_saturated(cal2, dq2, ramp2, cal_block=block, k_band=4.0)
    b, rim_b, _, _ = ramp_recover_saturated(cal2, dq2, ramp2, cal_block=block, k_band=0.0)
    common = rim_a & rim_b
    assert np.allclose(a[common], b[common], rtol=1e-6)


def test_recovery_holds_under_noise_and_pedestal():
    """cal = K*slope must be recovered to a few % even with a reset pedestal +
    read noise on the ramp (not the noise-free cal==K*slope idealisation)."""
    rng = np.random.default_rng(2)
    ny = nx = 96
    ng = 7
    K = 0.17
    rates = rng.uniform(300, 5000, size=(ny, nx))
    ped = rng.normal(0, 30, size=(ny, nx))
    ramp = ped[None] + rates[None] * np.arange(ng)[:, None, None]
    ramp += rng.normal(0, 15, size=ramp.shape)          # read noise per group
    ramp = np.minimum(ramp, CEIL)
    cal = K * rates                                     # cal ~ K * true rate
    dq = np.zeros((ny, nx), int)
    yy, xx = np.mgrid[0:ny, 0:nx]
    r = np.hypot(yy - 48, xx - 48)
    ramp[:, r < 10] = np.minimum(
        (70000.0 * np.exp(-(r[r < 10] ** 2) / 8.0))[None] * np.arange(ng)[:, None], CEIL)
    dq[r < 5] |= PSAT
    cal[r < 5] = np.nan
    slope, ng_ = ramp_slope_map(ramp, ceiling=CEIL)
    recovered, rim, deep, Kg = ramp_recover_saturated(cal, dq, ramp, ceiling=CEIL)
    assert Kg == pytest.approx(K, rel=0.05)
    ring = (r < 5) & rim & np.isfinite(slope)
    if ring.any():
        exp = (K * slope)[ring]
        got = recovered[ring]
        m = np.isfinite(exp) & (exp > 0)
        assert np.nanmedian(got[m] / exp[m]) == pytest.approx(1.0, rel=0.1)


def test_load_ramp_cube_shapes_and_noop(tmp_path):
    """_load_ramp_cube: 4-D and 3-D SCI reduce to (ng,ny,nx); GROUPDQ mismatched
    to SCI is dropped (not a crash); MIRI/other detectors and missing ramp -> None."""
    from astropy.io import fits
    from jwst_gc_pipeline.photometry.cataloging import _load_ramp_cube

    def _write_ramp(path, sci, gdq=None):
        hdul = fits.HDUList([fits.PrimaryHDU(),
                             fits.ImageHDU(sci.astype('float32'), name='SCI')])
        if gdq is not None:
            hdul.append(fits.ImageHDU(gdq.astype('int32'), name='GROUPDQ'))
        hdul.writeto(path, overwrite=True)

    ng, ny, nx = 5, 8, 8
    sci4 = np.ones((1, ng, ny, nx)); gdq4 = np.zeros((1, ng, ny, nx))
    crf = str(tmp_path / 'jw0_02101_00001_nrca1_destreak_o001_crf.fits')
    fits.PrimaryHDU().writeto(crf, overwrite=True)
    ramp = str(tmp_path / 'jw0_02101_00001_nrca1_ramp.fits')
    _write_ramp(ramp, sci4, gdq4)
    out = _load_ramp_cube(crf)
    assert out is not None and out[0].shape == (ng, ny, nx)
    assert out[1] is not None and out[1].shape == (ng, ny, nx)
    # GROUPDQ shape MISMATCH -> dropped, SCI still returned (no crash)
    _write_ramp(ramp, sci4, np.zeros((1, ng + 1, ny, nx)))
    out = _load_ramp_cube(crf)
    assert out is not None and out[1] is None
    # MIRI detector: regex doesn't match -> None
    crf_miri = str(tmp_path / 'jw0_02101_00001_mirimage_destreak_o001_crf.fits')
    fits.PrimaryHDU().writeto(crf_miri, overwrite=True)
    assert _load_ramp_cube(crf_miri) is None
    # missing ramp -> None
    crf_noramp = str(tmp_path / 'jw0_09999_00001_nrcb3_destreak_o001_crf.fits')
    fits.PrimaryHDU().writeto(crf_noramp, overwrite=True)
    assert _load_ramp_cube(crf_noramp) is None
