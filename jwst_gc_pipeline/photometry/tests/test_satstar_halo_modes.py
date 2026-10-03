"""Halo-mode saturated-star flux (#1013): a halo that changes per exposure
must not move the flux, while the standard masked-core fit follows it."""
import numpy as np
import pytest

from jwst_gc_pipeline.photometry.satstar_halo_modes import (
    azimuthal_median, fit_flux_with_halo_modes, halo_mode_flux_ratio, log_hats,
    satstar_halo_knots, satstar_halo_mode_ratios, spike_mask_from_psf)

N, C = 241, 120.3


def _psf(c=C):
    """Gaussian core + r^-3 smooth halo + six r^-2 diffraction spikes."""
    yy, xx = np.indices((N, N))
    dx, dy = xx - c, yy - c
    r = np.hypot(dx, dy)
    th = np.arctan2(dy, dx)
    core = np.exp(-0.5 * (r / 2.0) ** 2)
    halo = 0.05 * (np.maximum(r, 3.0) / 10.0) ** -3
    ang = np.abs(((th[..., None] - np.arange(6) * np.pi / 3 + np.pi) % (2 * np.pi)) - np.pi)
    perp = r[..., None] * np.sin(np.minimum(ang, np.pi / 2))
    spikes = (np.exp(-0.5 * (perp / 1.0) ** 2) * (ang < np.pi / 2)).sum(-1) * 0.3 * (np.maximum(r, 3.0) / 10.0) ** -2
    p = core + halo + spikes
    return p / p.sum(), r


def _star(halo_scale, F=1e7, B=5.0, seed=0, rcore=20.0):
    """A saturated star whose halo over 15-100 px is ``halo_scale`` x the model's."""
    p, r = _psf()
    pbar = azimuthal_median(p, r)
    bump = np.where((r > 15) & (r < 100), np.sin(np.pi * (np.log(r / 15) / np.log(100 / 15))), 0.0)
    truth = F * (p + (halo_scale - 1.0) * bump * pbar) + B
    rng = np.random.default_rng(seed)
    err = np.sqrt(np.clip(truth, 1, None))
    data = truth + rng.normal(size=truth.shape) * err
    mask = r < rcore
    return data, err, mask, p, r


def test_log_hats_partition_of_unity():
    r = np.linspace(1, 300, 2000)
    h = log_hats(r, (15., 35., 80., 150.))
    inside = (r >= 15) & (r <= 150)
    assert np.allclose(h[inside].sum(-1), 1.0)
    assert np.all(h[~inside] == 0)


def test_spike_mask_finds_spikes_not_halo():
    p, r = _psf()
    s = spike_mask_from_psf(p, r)
    ring = (r > 30) & (r < 80)
    # six ~3-px-wide spikes cover a small fraction of the annulus
    assert 0.01 < s[ring].mean() < 0.15


@pytest.mark.parametrize('scale', [0.8, 1.0, 1.2])
def test_halo_change_does_not_move_halo_mode_flux(scale):
    data, err, mask, p, r = _star(scale)
    s = fit_flux_with_halo_modes(data, err, mask, p, C, C, knots=None, rmax=110)
    h = fit_flux_with_halo_modes(data, err, mask, p, C, C, knots=(15., 35., 60., 100.), rmax=110)
    assert h.flux == pytest.approx(1e7, rel=0.01)
    if scale != 1.0:
        # the standard fit follows the halo (the #1013 failure)
        assert abs(s.flux / 1e7 - 1) > 3 * abs(h.flux / 1e7 - 1)
        assert np.sign(s.flux / 1e7 - 1) == np.sign(scale - 1)


def test_no_knots_is_the_standard_fit():
    data, err, mask, p, r = _star(1.0)
    s = fit_flux_with_halo_modes(data, err, mask, p, C, C, knots=None, rmax=110, niter=1)
    good = ~mask & (r <= 110)
    A = np.stack([p[good], np.ones(good.sum())], 1) / err[good][:, None]
    ref = np.linalg.lstsq(A, data[good] / err[good], rcond=None)[0]
    assert s.flux == pytest.approx(ref[0], rel=1e-9)
    assert s.bkg == pytest.approx(ref[1], rel=1e-6)


def test_knots_and_ratio_helper():
    assert satstar_halo_knots(5.0, 40.0) == pytest.approx((15.0, 24.494897, 40.0), rel=1e-5)
    assert satstar_halo_knots(30.0, 35.0) is None
    data, err, mask, p, r = _star(1.2)
    ratio, s, h = halo_mode_flux_ratio(data, err, mask, p, C, C, r_core=20.0, rmax=110.0)
    assert ratio == pytest.approx(h.flux / s.flux)
    assert ratio < 0.99            # the standard fit over-reads a 20%-bright halo
    nan_ratio, s2, h2 = halo_mode_flux_ratio(data, err, mask, p, C, C, r_core=100.0, rmax=110.0)
    assert np.isnan(nan_ratio) and s2 is None and h2 is None


def test_clipping_rejects_field_stars():
    """Bright unmodelled field stars on the fit pixels: the 5-sigma clip must
    keep the halo-mode flux on the truth (with clipping off it is pulled)."""
    data, err, mask, p, r = _star(1.2)
    yy, xx = np.indices(data.shape)
    rng = np.random.default_rng(3)
    for _ in range(25):
        ang, rr = rng.uniform(0, 2 * np.pi), rng.uniform(25, 100)
        x, y = C + rr * np.cos(ang), C + rr * np.sin(ang)
        data += 3e4 * np.exp(-0.5 * ((xx - x) ** 2 + (yy - y) ** 2) / 1.2 ** 2)
    kn = (15., 35., 60., 100.)
    h = fit_flux_with_halo_modes(data, err, mask, p, C, C, knots=kn, rmax=110)
    h1 = fit_flux_with_halo_modes(data, err, mask, p, C, C, knots=kn, rmax=110, niter=1)
    assert h.flux == pytest.approx(1e7, rel=0.01)
    assert abs(h1.flux / 1e7 - 1) > 2 * abs(h.flux / 1e7 - 1)


def test_ratios_from_a_gridded_psf_model():
    """The production entry point: evaluate a photutils GriddedPSFModel at the
    fitted position (argument order flux, x_0, y_0) and return the ratio."""
    from astropy.nddata import NDData
    from photutils.psf import GriddedPSFModel
    pc, _ = _psf(c=(N - 1) / 2)
    grid = GriddedPSFModel(NDData(pc[None], meta={'grid_xypos': [(0.0, 0.0)], 'oversampling': 1}))
    data, err, mask, p, r = _star(1.2)
    got = satstar_halo_mode_ratios(data, err, mask, grid, [(C, C)], r_core=20.0, rmax=110.0)
    want, _, _ = halo_mode_flux_ratio(data, err, mask, grid.evaluate(*np.indices(data.shape)[::-1], 1.0, C, C),
                                      C, C, r_core=20.0, rmax=110.0)
    assert got.shape == (1,) and np.isfinite(got[0])
    assert got[0] == pytest.approx(want, rel=1e-12)
    assert got[0] < 0.99      # a 20%-bright halo: the standard fit over-reads
    # a core too large for the fit radius records NaN, not a number
    assert np.isnan(satstar_halo_mode_ratios(data, err, mask, grid, [(C, C)], r_core=100.0, rmax=110.0)[0])


def test_wide_ratios_on_a_full_frame():
    """The wide-radius entry point: stamp from a full frame in detector
    coordinates, DQ-SATURATED core masked, ratio NaN below the area cut."""
    from astropy.nddata import NDData
    from photutils.psf import GriddedPSFModel
    from jwst_gc_pipeline.photometry.satstar_halo_modes import satstar_halo_mode_ratios_wide
    pc, _ = _psf(c=(N - 1) / 2)
    grid = GriddedPSFModel(NDData(pc[None], meta={'grid_xypos': [(0.0, 0.0)], 'oversampling': 1}))
    data, err, mask, p, r = _star(1.2)
    # embed the stamp in a larger frame at an offset, so a coordinate mix-up fails
    off = (37, 61)
    frame = np.zeros((N + 80, N + 120)); ferr = np.ones_like(frame)
    dq = np.zeros(frame.shape, np.uint32)
    sl = (slice(off[1], off[1] + N), slice(off[0], off[0] + N))
    frame[sl], ferr[sl] = data, err
    dq[sl][r < 17] = 2                  # SATURATED; + 3 px dilation = the r < 20 mask
    x, y = C + off[0], C + off[1]
    area = float((r < 17).sum())
    got = satstar_halo_mode_ratios_wide(frame, ferr, dq, grid, [(x, y)], area, rmax=110.0, area_min=300)
    assert got.shape == (1,) and np.isfinite(got[0])
    assert got[0] < 0.99                # a 20%-bright halo: the standard fit over-reads
    _, s, h = halo_mode_flux_ratio(data, err, r < 20, p, C, C, r_core=np.sqrt(area / np.pi) + 3, rmax=110.0)
    assert h.flux == pytest.approx(1e7, rel=0.02)
    # below the core-area threshold: not measured
    assert np.isnan(satstar_halo_mode_ratios_wide(frame, ferr, dq, grid, [(x, y)], 100.0,
                                                  rmax=110.0, area_min=300)[0])
