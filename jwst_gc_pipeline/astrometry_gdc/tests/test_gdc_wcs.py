"""Affine-anchored GDC sky solution tests (synthetic maps + astropy WCS)."""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS

from jwst_gc_pipeline.astrometry_gdc.stdgdc import STDGDC
from jwst_gc_pipeline.astrometry_gdc.gdc_wcs import GDCSkySolution

N = 128  # synthetic detector size


def synthetic_gdc(amp=0.5):
    """STDGDC from arrays: identity + a smooth nonlinear distortion field.

    Map values follow the file convention: 1-based corrected position of the
    raw 1-based pixel (i+1, j+1) stored at [j, i].
    """
    jj, ii = np.mgrid[0:N, 0:N].astype(float)
    x1, y1 = ii + 1.0, jj + 1.0  # raw 1-based
    xgc = x1 + amp * np.sin(2 * np.pi * y1 / N) * np.cos(np.pi * x1 / N)
    ygc = y1 + amp * np.cos(2 * np.pi * x1 / N) * np.sin(np.pi * y1 / N)
    mgc = np.zeros_like(xgc)
    return STDGDC(xgc, ygc, mgc)


def synthetic_wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [N / 2 + 0.5, N / 2 + 0.5]  # 1-based FITS crpix
    w.wcs.crval = [266.55, -28.71]
    scale = 0.031 / 3600.0
    theta = np.deg2rad(20.0)
    w.wcs.cd = np.array([[-scale * np.cos(theta), scale * np.sin(theta)],
                         [scale * np.sin(theta), scale * np.cos(theta)]])
    return w


@pytest.fixture(scope='module')
def solution():
    return GDCSkySolution(synthetic_wcs(), synthetic_gdc(), grid_n=16,
                          shape=(N, N))


def test_affine_anchor_preserves_mean_position(solution):
    """(c) the anchored solution agrees with the original WCS on average
    (<0.1 mas over the anchor grid) while changing the distortion field."""
    w = synthetic_wcs()
    gx, gy = solution.grid_x.ravel(), solution.grid_y.ravel()
    sky_orig = SkyCoord(w.pixel_to_world(gx, gy)).icrs
    sky_gdc = solution.gdc_sky(gx, gy)
    dlon, dlat = sky_orig.spherical_offsets_to(sky_gdc)
    mean_shift = np.hypot(dlon.to_value(u.mas).mean(), dlat.to_value(u.mas).mean())
    assert mean_shift < 0.1
    # ... but the FIELD changed: 0.5 pix * 31 mas/pix distortion amplitude
    per_point = np.hypot(dlon.to_value(u.mas), dlat.to_value(u.mas))
    assert per_point.max() > 3.0
    assert solution.affine_rms_mas > 1.0


def test_identity_gdc_gives_zero_delta():
    """With an identity distortion field the anchored solution must reproduce
    the original WCS everywhere (residual only from numerics, << 0.01 mas)."""
    sol = GDCSkySolution(synthetic_wcs(), synthetic_gdc(amp=0.0), grid_n=12,
                         shape=(N, N))
    assert sol.affine_rms_mas < 0.01
    assert np.max(np.abs(sol.delta_xi_mas)) < 0.01
    assert np.max(np.abs(sol.delta_eta_mas)) < 0.01


def test_delta_map_matches_gdc_field(solution):
    """The diagnostic delta map has zero mean (affine intercept) and tracks
    the injected sinusoidal field's amplitude (~0.5 pix ~ 15.5 mas)."""
    _, _, dxi, deta = solution.delta_map()
    assert abs(dxi.mean()) < 1e-6
    assert abs(deta.mean()) < 1e-6
    amp = np.hypot(dxi, deta).max()
    assert 5.0 < amp < 30.0


def test_indexing_shift_sanity_synthetic():
    """(d) on synthetic maps too: +1 pixel in -> ~+1 pixel out."""
    gdc = synthetic_gdc()
    x = np.linspace(10, N - 12, 25)
    y = np.linspace(10, N - 12, 25)
    xc0, yc0 = gdc.forward(x, y)
    xc1, yc1 = gdc.forward(x + 1.0, y + 1.0)
    np.testing.assert_allclose(xc1 - xc0, 1.0, atol=0.06)
    np.testing.assert_allclose(yc1 - yc0, 1.0, atol=0.06)


def test_gdc_sky_scalar_and_vector(solution):
    s1 = solution.gdc_sky(10.0, 20.0)
    s2 = solution.gdc_sky(np.array([10.0, 30.0]), np.array([20.0, 40.0]))
    assert s2.size == 2
    assert s1.separation(s2[0]).to_value(u.mas) < 1e-6


def test_provenance_fields(solution):
    prov = solution.provenance()
    assert len(prov['gdc_affine_x']) == 3
    assert len(prov['gdc_affine_y']) == 3
    assert prov['gdc_affine_rms_mas'] == solution.affine_rms_mas


def synthetic_gdc_with_hole(amp=0.5):
    """Like the real NRCB4/F212N library file: a border region stored as 0
    (unmeasured), which reads as a huge negative 'correction'."""
    gdc = synthetic_gdc(amp=amp)
    xgc = gdc.xgc.copy()
    ygc = gdc.ygc.copy()
    # last row + upper half of the last column, as in NRCB4/F212N
    xgc[-1, :] = 0.0
    ygc[-1, :] = 0.0
    xgc[N // 2:, -1] = 0.0
    ygc[N // 2:, -1] = 0.0
    return STDGDC(xgc, ygc, np.zeros_like(xgc))


def test_border_hole_masked_from_affine_anchor():
    """Unmeasured (zero-stored) map regions must not poison the affine
    anchor: the solution on a holed map must match the clean-map solution to
    well under a mas (the real NRCB4/F212N hole gave an 11.9 ARCSEC affine
    rms before masking)."""
    with pytest.warns(UserWarning, match='invalid/unmeasured'):
        holed = GDCSkySolution(synthetic_wcs(), synthetic_gdc_with_hole(),
                               grid_n=16, shape=(N, N))
    clean = GDCSkySolution(synthetic_wcs(), synthetic_gdc(), grid_n=16,
                           shape=(N, N))
    assert holed.n_anchor_invalid > 0
    assert holed.affine_rms_mas < clean.affine_rms_mas * 1.5
    # interior stars: consistent with the clean-map solution.  Dropping the
    # border anchor points legitimately perturbs the affine by a small
    # fraction of the (here exaggerated, 15 mas) distortion amplitude, so the
    # bound is a few mas -- vs the ~10 ARCSEC scale of the unmasked poisoning.
    x = np.linspace(5, N - 20, 20)
    y = np.linspace(5, N - 20, 20)
    sep = holed.gdc_sky(x, y).separation(clean.gdc_sky(x, y))
    assert sep.to_value(u.mas).max() < 3.0


def test_star_on_hole_falls_back_to_original_wcs():
    """A star inside the unmeasured region gets the frame's original WCS
    position (zero delta), not garbage."""
    with pytest.warns(UserWarning, match='invalid/unmeasured'):
        sol = GDCSkySolution(synthetic_wcs(), synthetic_gdc_with_hole(),
                             grid_n=16, shape=(N, N))
    w = synthetic_wcs()
    x = np.array([N - 1.0, 20.0])   # first star on the zeroed last column
    y = np.array([N - 10.0, 20.0])
    sky = sol.gdc_sky(x, y)
    assert sol.n_star_fallback == 1
    orig = SkyCoord(w.pixel_to_world(x[0], y[0])).icrs
    assert sky[0].separation(orig).to_value(u.mas) < 1e-6
