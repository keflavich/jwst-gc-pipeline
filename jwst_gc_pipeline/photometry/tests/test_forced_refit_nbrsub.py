"""Neighbour- and background-subtracted overshoot refit (#932).

``_manual_phot_pass`` re-fits each overshooting source as forced photometry
at its seed.  On main that solve runs on the raw frame as a single source in
a 5x5 box with no background term, so neighbour light, spike light and the
local background pedestal all enter the flux.  With env
``DAOPHOT_REFIT_NBRSUB=1`` the solve runs on the frame with every
non-overshooting source's model removed, and the row's own ``local_bkg`` is
subtracted from its stamp.
"""
import types

import numpy as np
import pytest
from astropy.nddata import NDData
from astropy.table import Table
from photutils.psf import GriddedPSFModel

from jwst_gc_pipeline.photometry import cataloging as C
from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as L

FWHM = 2.5
SHAPE = (100, 100)
BRIGHT = (50.0, 50.0)
FAINT = (55.0, 50.0)
F_BRIGHT = 5.0e4
F_FAINT = 3.0e3
PEDESTAL = 30.0


def _gaussian_grid_psf():
    ov = 4
    n = 25 * ov + 1
    c = (n - 1) / 2
    yy, xx = np.mgrid[:n, :n]
    sig = FWHM / 2.3548 * ov
    p = np.exp(-((xx - c) ** 2 + (yy - c) ** 2) / (2 * sig ** 2))
    p /= p.sum() / ov ** 2
    return GriddedPSFModel(NDData(p[None], meta={'grid_xypos': [(0, 0)],
                                                 'oversampling': ov}))


def _gauss(xx, yy, flux, x0, y0):
    sig = FWHM / 2.3548
    return (flux / (2 * np.pi * sig ** 2)
            * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sig ** 2)))


def _scene(seed=3):
    yy, xx = np.mgrid[:SHAPE[0], :SHAPE[1]].astype(float)
    img = (PEDESTAL + _gauss(xx, yy, F_BRIGHT, *BRIGHT)
           + _gauss(xx, yy, F_FAINT, *FAINT))
    return img + np.random.default_rng(seed).normal(0, 1.0, SHAPE)


# ---------------------------------------------------------------------------
# forced_psf_photometry(local_bkg=...)
# ---------------------------------------------------------------------------
def test_forced_solve_absorbs_a_pedestal_without_local_bkg():
    psf = _gaussian_grid_psf()
    yy, xx = np.mgrid[:41, :41].astype(float)
    img = _gauss(xx, yy, 1000.0, 20.0, 20.0) + 25.0
    init = Table({'x_init': [20.0], 'y_init': [20.0]})
    raw = L.forced_psf_photometry(img, psf, init, fit_shape=(5, 5))
    sub = L.forced_psf_photometry(img, psf, init, fit_shape=(5, 5),
                                  local_bkg=25.0)
    # the pedestal enters as 25 * sum(p) / sum(p^2) on the 5x5 stamp
    p = np.asarray(psf.evaluate(xx[18:23, 18:23], yy[18:23, 18:23], 1.0,
                                20.0, 20.0), float)
    lever = p.sum() / (p ** 2).sum()
    assert raw['flux_fit'][0] == pytest.approx(1000.0 + 25.0 * lever, rel=1e-3)
    assert sub['flux_fit'][0] == pytest.approx(1000.0, rel=1e-3)


def test_forced_solve_local_bkg_per_source_and_nonfinite():
    psf = _gaussian_grid_psf()
    yy, xx = np.mgrid[:41, :61].astype(float)
    img = _gauss(xx, yy, 800.0, 15.0, 20.0) + _gauss(xx, yy, 500.0, 45.0, 20.0)
    img[:, 30:] += 40.0
    init = Table({'x_init': [15.0, 45.0], 'y_init': [20.0, 20.0]})
    out = L.forced_psf_photometry(img, psf, init, fit_shape=(5, 5),
                                  local_bkg=np.array([np.nan, 40.0]))
    np.testing.assert_allclose(out['flux_fit'], [800.0, 500.0], rtol=1e-3)


def test_forced_solve_default_unchanged():
    psf = _gaussian_grid_psf()
    yy, xx = np.mgrid[:41, :41].astype(float)
    img = _gauss(xx, yy, 1000.0, 20.0, 20.0) + 5.0
    init = Table({'x_init': [20.0], 'y_init': [20.0]})
    a = L.forced_psf_photometry(img, psf, init, fit_shape=(5, 5))
    b = L.forced_psf_photometry(img, psf, init, fit_shape=(5, 5),
                                local_bkg=None)
    np.testing.assert_array_equal(a['flux_fit'], b['flux_fit'])


# ---------------------------------------------------------------------------
# env switch
# ---------------------------------------------------------------------------
def test_refit_nbrsub_env(monkeypatch):
    monkeypatch.delenv('DAOPHOT_REFIT_NBRSUB', raising=False)
    assert C._daophot_refit_nbrsub() is False
    monkeypatch.setenv('DAOPHOT_REFIT_NBRSUB', '1')
    assert C._daophot_refit_nbrsub() is True
    monkeypatch.setenv('DAOPHOT_REFIT_NBRSUB', ' off ')
    assert C._daophot_refit_nbrsub() is False


@pytest.mark.parametrize('raw', ['abc', '2', 'nan'])
def test_refit_nbrsub_env_rejects_malformed_values(monkeypatch, raw):
    monkeypatch.setenv('DAOPHOT_REFIT_NBRSUB', raw)
    with pytest.raises(ValueError, match='DAOPHOT_REFIT_NBRSUB'):
        C._daophot_refit_nbrsub()


# ---------------------------------------------------------------------------
# _subset_model_image
# ---------------------------------------------------------------------------
class _FakePhot:
    """Minimal stand-in exposing ``results`` and ``make_model_image``."""

    def __init__(self, results):
        self.results = results

    def make_model_image(self, shape, psf_shape=None, **kw):
        yy, xx = np.mgrid[:shape[0], :shape[1]].astype(float)
        out = np.zeros(shape)
        for r in self.results:
            out += _gauss(xx, yy, r['flux_fit'], r['x_fit'], r['y_fit'])
        return out


def test_subset_model_image_renders_only_kept_rows_and_restores():
    res = Table({'x_fit': [20.0, 40.0], 'y_fit': [20.0, 20.0],
                 'flux_fit': [100.0, 300.0]})
    phot = _FakePhot(res)
    m = C._subset_model_image(phot, np.array([False, True]), (41, 61))
    assert phot.results is res and len(phot.results) == 2
    assert m.sum() == pytest.approx(300.0, rel=1e-3)
    assert m[20, 20] < 1e-6 * m[20, 40]
    none = C._subset_model_image(phot, np.array([False, False]), (41, 61))
    assert not none.any()


# ---------------------------------------------------------------------------
# _manual_phot_pass: faint star next to a bright one, on a pedestal
# ---------------------------------------------------------------------------
def _options():
    return types.SimpleNamespace(group=False, satstar_artifact_sigK=3.0,
                                 satstar_artifact_ratio=1.0)


def _run_pass(monkeypatch, nbrsub):
    """Run one pass with the faint star forced into the overshoot refit."""
    if nbrsub:
        monkeypatch.setenv('DAOPHOT_REFIT_NBRSUB', '1')
    else:
        monkeypatch.delenv('DAOPHOT_REFIT_NBRSUB', raising=False)
    real = C._filter_or_flag_model_overshoot

    def flag_faint(phot_obj, modsky, data, **kw):
        over = real(phot_obj, modsky, data, **kw)
        if kw.get('action') == 'flag' and kw.get('label') == 't':
            res = phot_obj.results
            d = np.hypot(np.asarray(res['x_init'], float) - FAINT[0],
                         np.asarray(res['y_init'], float) - FAINT[1])
            over = np.asarray(over, bool).copy()
            over[np.argmin(d)] = True
            res['model_overshoot'] = over
        return over

    monkeypatch.setattr(C, '_filter_or_flag_model_overshoot', flag_faint)
    img = _scene()
    seed = Table({'x_init': [BRIGHT[0], FAINT[0]],
                  'y_init': [BRIGHT[1], FAINT[1]],
                  'flux_init': [F_BRIGHT, F_FAINT]})
    zeros = np.zeros(SHAPE)
    res, _, _ = C._manual_phot_pass(
        data=img, mask=np.zeros(SHAPE, bool), err=np.ones(SHAPE),
        bad=np.zeros(SHAPE, bool), dao_psf_model=_gaussian_grid_psf(),
        init_params=seed, aperture_radius_pix=2 * FWHM, localbkg_inner=6,
        localbkg_outer=10, grouper=None, options=_options(),
        dq=np.zeros(SHAPE, np.uint32), satstar_model_subtracted=zeros,
        label='t', near_sat_dist_pix=1.0)
    d = np.hypot(np.asarray(res['x_fit'], float) - FAINT[0],
                 np.asarray(res['y_fit'], float) - FAINT[1])
    k = int(np.argmin(d))
    if d[k] > 0.5:
        return None, res
    assert bool(res['forced_refit'][k])
    return float(res['flux_fit'][k]), res


def test_refit_on_raw_frame_over_brightens_faint_neighbour(monkeypatch):
    flux, _ = _run_pass(monkeypatch, nbrsub=False)
    # main behaviour: inflated by neighbour light and the pedestal (or so
    # inflated that the final phantom drop removes the row)
    assert flux is None or flux > 1.1 * F_FAINT


def test_nbrsub_refit_recovers_faint_neighbour_flux(monkeypatch):
    flux, res = _run_pass(monkeypatch, nbrsub=True)
    assert flux == pytest.approx(F_FAINT, rel=0.05)
    # the bright star is untouched by the refit
    d = np.hypot(np.asarray(res['x_fit'], float) - BRIGHT[0],
                 np.asarray(res['y_fit'], float) - BRIGHT[1])
    k = int(np.argmin(d))
    assert not bool(res['forced_refit'][k])
    assert float(res['flux_fit'][k]) == pytest.approx(F_BRIGHT, rel=0.02)
