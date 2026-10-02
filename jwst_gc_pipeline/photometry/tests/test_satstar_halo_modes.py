"""Halo-mode saturated-star flux (#1013): a halo that changes per exposure
must not move the flux, while the standard masked-core fit follows it."""
import numpy as np
import pytest

from jwst_gc_pipeline.photometry.satstar_halo_modes import (
    azimuthal_median, fit_flux_with_halo_modes, halo_mode_flux_ratio, log_hats,
    satstar_halo_knots, spike_mask_from_psf)

N, C = 241, 120.3


def _psf():
    """Gaussian core + r^-3 smooth halo + six r^-2 diffraction spikes."""
    yy, xx = np.indices((N, N))
    dx, dy = xx - C, yy - C
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
