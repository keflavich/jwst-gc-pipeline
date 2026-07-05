"""Regression tests for the flat-topped saturated-core satstar model.

STPSF is sharply peaked; a charge-bled saturated core is a flat-topped plateau.
Subtracting ``amp*PSF`` therefore leaves an UNDER-subtraction ring at the core
edge (cloudc F770W: "every star with a saturated core is undersubtracted") or,
when the amplitude is inflated to clear the core, an OVER-subtraction divot.
``flattop_satstar_model`` replaces the model inside the plateau by the data
itself so the core residual -> ~0 while leaving the PSF wings untouched.

These tests exercise the pure helper directly (no fitter run) with synthetic
peaked-PSF-vs-flat-core cutouts.
"""
import numpy as np

from jwst_gc_pipeline.reduction.saturated_star_finding import (
    flattop_satstar_model)


def _grids(n=41):
    yy, xx = np.mgrid[0:n, 0:n]
    r = np.hypot(xx - n // 2, yy - n // 2)
    return r


def _peaked_psf(r, fwhm=3.0):
    sigma = fwhm / 2.3548
    p = np.exp(-0.5 * (r / sigma) ** 2)
    return p / p.max()


def _flat_core_star(r, core_r=4.0, plateau=1000.0, fwhm=3.0):
    """A flat-topped (charge-bled) star: constant ``plateau`` inside ``core_r``,
    peaked-PSF wings outside, matched at the core edge."""
    sigma = fwhm / 2.3548
    wing = plateau * np.exp(-0.5 * ((r - core_r) / sigma) ** 2)
    return np.where(r <= core_r, plateau, wing)


def test_flattop_zeros_the_core_residual():
    """Peaked model under-subtracts a flat core (bright ring); flat-top clears it."""
    r = _grids()
    data = _flat_core_star(r)                       # flat-topped truth (bg-subtracted)
    # A peaked amp*PSF whose PEAK matches the core level under-fits the plateau
    model = data.max() * _peaked_psf(r)
    ring_before = np.nanmax(np.abs(data - model))   # big positive ring
    ft = flattop_satstar_model(model, data, plateau_frac=0.15)
    core = r <= 4.0
    # Core residual is ~0 after flat-top (data subtracted flat there).
    assert np.nanmax(np.abs((data - ft)[core])) < 1e-6
    assert ring_before > 100.0                      # sanity: there WAS a ring


def test_flattop_never_oversubtracts_the_core():
    """An INFLATED peaked model (peak >> data) would gouge a divot; flat-top caps
    the model to the data everywhere it exceeds it -> residual >= ~0 in the core."""
    r = _grids()
    data = _flat_core_star(r)
    model = 5.0 * data.max() * _peaked_psf(r)       # grossly over-predicts core
    ft = flattop_satstar_model(model, data, plateau_frac=0.15)
    resid = data - ft
    core = r <= 6.0
    # No deep negative pit: model never exceeds data where data is finite.
    assert np.nanmin(resid[core]) > -1e-6


def test_flattop_leaves_wings_untouched():
    """Outside the plateau (faint wings) the PSF model is preserved exactly."""
    r = _grids()
    data = _flat_core_star(r)
    model = data.max() * _peaked_psf(r)
    ft = flattop_satstar_model(model, data, plateau_frac=0.15)
    # Far wing: below plateau_frac*peak AND model<=data -> unchanged.
    wing = (r > 12) & (model <= data)
    assert np.allclose(ft[wing], model[wing])


def test_flattop_keeps_psf_in_nan_core():
    """A saturated (NaN) core has no data to subtract; keep the PSF estimate."""
    r = _grids()
    data = _flat_core_star(r)
    data[r <= 2.0] = np.nan                          # deep-saturated centre
    model = data.max() if np.isfinite(data).any() else 1.0
    model = np.nanmax(data[np.isfinite(data)]) * _peaked_psf(r)
    ft = flattop_satstar_model(model, data, plateau_frac=0.15)
    nan_core = r <= 2.0
    # PSF retained (not zeroed) where data is NaN.
    assert np.all(ft[nan_core] == model[nan_core])


def test_flattop_all_nan_returns_input_unchanged():
    """Fully-saturated cutout (nothing to subtract) -> peaked model preserved."""
    r = _grids(11)
    model = _peaked_psf(r) * 1000.0
    data = np.full_like(model, np.nan)
    ft = flattop_satstar_model(model, data)
    assert ft is model                               # identity: no change


def test_flattop_core_mask_covers_undersub_ring():
    """The pipeline peak-cap holds the model AT the coadd-core level, so a
    peaked model UNDER-predicts a rounded shoulder whose level sits just below
    plateau_frac*peak -- an amplitude-only mask misses that RING.  The geometric
    ``core_mask`` (a radius from the fit centre) covers it: the shoulder residual
    collapses to ~0 only when core_mask is supplied."""
    r = _grids()
    # rounded (not perfectly flat) core: bright plateau + a shoulder ring that
    # sits below 0.15*peak but is still strongly under-predicted by a capped PSF.
    data = _flat_core_star(r, core_r=3.0, plateau=1000.0)
    model = 267.0 * _peaked_psf(r)                  # peak-capped low (coadd core)
    shoulder = (r >= 4) & (r <= 7)
    resid_no_mask = data - flattop_satstar_model(model, data, plateau_frac=0.15)
    # without a core_mask the shoulder ring survives (amplitude mask misses it)...
    assert np.nanmax(resid_no_mask[shoulder]) > 50.0
    # ...with a geometric core_mask out to r=8 it is subtracted flat -> ~0.
    cm = r <= 8.0
    resid_mask = data - flattop_satstar_model(model, data, plateau_frac=0.15,
                                              core_mask=cm)
    assert np.nanmax(np.abs(resid_mask[shoulder])) < 1e-6


def test_flattop_output_nonnegative():
    """The flat-top model is physically non-negative (parity with psf clip)."""
    r = _grids()
    data = _flat_core_star(r) - 50.0                 # bg-subtracted -> some neg wings
    model = data.max() * _peaked_psf(r)
    ft = flattop_satstar_model(model, data, plateau_frac=0.15)
    assert np.nanmin(ft) >= 0.0
