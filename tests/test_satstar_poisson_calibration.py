"""Tests for scripts/analysis/satstar_poisson (issue #1008).

* ``noise_only`` (numpy only): the noise-only simulator and the sigma(chi) report that
  ``loo_crop.py`` with ``NOISEONLY=<seed>`` uses to calibrate the LOO statistic.
* ``patternfit.fit_nuisance(anchor=True)``: the anchor exposure holds its flux fixed and
  fits only the pedestal and the gradient (needs ``finufft``; skipped without it).
"""
import importlib
import os
import sys

import numpy as np
import pytest

SATSTAR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       'scripts', 'analysis', 'satstar_poisson')


@pytest.fixture(scope='module')
def noise_only():
    sys.path.insert(0, SATSTAR)
    try:
        yield importlib.import_module('noise_only')
    finally:
        sys.path.remove(SATSTAR)
        sys.modules.pop('noise_only', None)


def test_draw_noise_only_matches_the_cutout_variance(noise_only):
    rng = np.random.default_rng(1008)
    n = 200000
    model = rng.uniform(1, 100, n)
    w = rng.uniform(0.01, 4.0, n)
    d = np.full(n, -7.0)
    sim = noise_only.draw_noise_only(model, d, w, rng)
    z = (sim-model)*np.sqrt(w)
    assert abs(z.mean()) < 5/np.sqrt(n)
    assert abs(z.std()-1) < 5/np.sqrt(2*n)


def test_draw_noise_only_leaves_zero_weight_pixels_unchanged(noise_only):
    rng = np.random.default_rng(3)
    model = np.arange(10.0); d = np.full(10, np.nan); w = np.ones(10); w[[2, 5]] = 0
    sim = noise_only.draw_noise_only(model, d, w, rng)
    assert np.isnan(sim[[2, 5]]).all()
    assert np.isfinite(np.delete(sim, [2, 5])).all()


def test_draw_noise_only_is_reproducible_per_seed(noise_only):
    model = np.linspace(0, 1, 50); w = np.ones(50)
    a = noise_only.draw_noise_only(model, model, w, np.random.default_rng(5))
    b = noise_only.draw_noise_only(model, model, w, np.random.default_rng(5))
    c = noise_only.draw_noise_only(model, model, w, np.random.default_rng(6))
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_sigma_chi_centred_and_uncentred(noise_only):
    rng = np.random.default_rng(2)
    chi = rng.normal(0.5, 1.0, 100000)
    s = noise_only.sigma_chi(chi)
    assert s['n'] == chi.size
    assert abs(s['sig_centred']-1) < 0.02          # the offset is removed
    assert s['sig'] > s['sig_centred']+0.1         # the uncentred estimator keeps it
    assert abs(s['chi2']-1.25) < 0.02
    empty = noise_only.sigma_chi(np.array([]))
    assert empty['n'] == 0 and np.isnan(empty['sig'])


def test_radial_report_bins_and_mask(noise_only):
    rng = np.random.default_rng(4)
    R = rng.uniform(0, 600, 400000)
    chi = rng.normal(0, 1, R.size)
    ok = np.ones(R.size, bool); ok[R > 450] = False
    rep = noise_only.radial_report(chi, R, ok)
    assert list(rep) == ['30-50', '50-80', '80-120', '120-200', '200-300', '300-500']
    for v in rep.values():
        assert abs(v['sig']-1) < 0.05
    assert rep['300-500']['n'] > 0
    assert max(R[ok]) <= 450
    assert noise_only.radial_report(chi, R, np.zeros(R.size, bool))['30-50']['n'] == 0


@pytest.fixture(scope='module')
def patternfit():
    pytest.importorskip('finufft')
    saved = {k: os.environ.get(k) for k in ('QN', 'QH', 'KMAX')}
    os.environ.update(QN='64', QH='1', KMAX='0.45')
    sys.path.insert(0, SATSTAR)
    try:
        yield importlib.import_module('patternfit')
    finally:
        sys.path.remove(SATSTAR)
        for m in ('patternfit', 'bandpsf', 'exposure', 'halobasis'):
            sys.modules.pop(m, None)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class _Anchor:
    """The attributes fit_nuisance(anchor=True) reads from a QE."""

    def __init__(self, pf, rng, npix=3000):
        self.qx = rng.uniform(-20, 20, npix); self.qy = rng.uniform(-20, 20, npix)
        self.X = rng.uniform(-1, 1, npix); self.Y = rng.uniform(-1, 1, npix)
        self.w = rng.uniform(0.5, 2.0, npix)
        self.a = 1.0; self.c = 0.0; self.g = np.zeros(2); self.beta = np.zeros(3)
        self._pf = pf

    def mult(self):
        return 1.0-self.beta[0]*self._pf.KX**2-self.beta[1]*self._pf.KY**2-self.beta[2]*self._pf.KX*self._pf.KY


def _pattern(pf, rng):
    p = rng.normal(0, 1, (pf.N, pf.N))+5.0
    return pf.from_G(pf.to_F(p)*pf.N**2).reshape(pf.N, pf.N)


def test_anchor_fit_recovers_pedestal_and_gradient_and_keeps_flux(patternfit):
    rng = np.random.default_rng(7)
    Q = _pattern(patternfit, rng)
    e = _Anchor(patternfit, rng)
    v = e.a*patternfit.nufft2(e.qx, e.qy, patternfit.to_F(Q))
    e.d = v+0.7+0.2*e.X-0.3*e.Y
    patternfit.fit_nuisance(e, Q, anchor=True)
    assert e.a == 1.0
    assert e.c == pytest.approx(0.7, abs=1e-6)
    assert e.g == pytest.approx([0.2, -0.3], abs=1e-6)


def test_anchor_fit_does_not_absorb_a_flux_error_into_the_pedestal(patternfit):
    """With the anchor flux off, c and g are the plain least-squares solution for
    d - a*v on [1, X, Y]; a flux column in the design would change them (#1008)."""
    rng = np.random.default_rng(8)
    Q = _pattern(patternfit, rng)
    e = _Anchor(patternfit, rng)
    v = patternfit.nufft2(e.qx, e.qy, patternfit.to_F(Q))
    e.d = 1.05*v+0.7+0.2*e.X-0.3*e.Y+rng.normal(0, 1, v.size)/np.sqrt(e.w)
    e.a = 1.0
    patternfit.fit_nuisance(e, Q, anchor=True, niter=1)
    A = np.stack([np.ones_like(v), e.X, e.Y], 1)*np.sqrt(e.w)[:, None]
    ref = np.linalg.lstsq(A, np.sqrt(e.w)*(e.d-e.a*v), rcond=None)[0]
    assert e.a == 1.0
    assert e.c == pytest.approx(ref[0], abs=1e-5)
    assert e.g == pytest.approx(ref[1:], abs=1e-5)
