"""Gibbs steps: the linear background (constant + Fourier templates) and the
flux-function slope.

* Background: a source-free image with a known constant + 4 Fourier
  amplitudes; the Gibbs draws must have the analytic posterior mean and
  covariance ((T W T^T + P)^-1 for weights W and prior precision P).
* Slope: with a flat likelihood (zero weights) and a flat slope prior on
  ``slope_bounds``, the sampled slope is uniform on the bounds (its
  conditional given the fluxes is a truncated Gamma, and the fluxes follow
  p(f | s)).
Tolerances: 4 batch-means standard errors (20 batches).
"""
import numpy as np
import pytest

from ..pcat_sampler import PCATConfig, fourier_templates, run_pcat
from ._helpers import get_epsf, have_compiler, lib_dir

pytestmark = pytest.mark.skipif(not have_compiler(), reason='no C compiler')


def _batch_se(a, nb=20):
    a = np.asarray(a, float)
    m = a.shape[0] // nb
    return a[:m * nb].reshape((nb, m) + a.shape[1:]).mean(axis=1).std(axis=0, ddof=1) / np.sqrt(nb)


def test_fourier_background_posterior():
    epsf, os_, _ = get_epsf()
    rng = np.random.default_rng(11)
    shape = (40, 44)
    T = np.concatenate([np.ones((1,) + shape), fourier_templates(*shape, 1)], axis=0)
    beta_true = np.array([0.5, 0.8, -0.4, 0.3, 0.6])
    sig = 1.0
    img = np.tensordot(beta_true, T, axes=1) + rng.normal(0, sig, shape)
    w = np.full(shape, 1 / sig ** 2)
    psig = 10.0
    cfg = PCATConfig(trueminf=1e7, nsamp=2000, nloop=1, fourier_order=1, fourier_prior_sig=psig,
                     move_weights=dict(movestar=0., birth_death=0., merge_split=0., bkg=1.),
                     burnin_frac=0.0, seed=4, lib_dir=str(lib_dir()), verbose=0,
                     store_models=False, bkg=[0.0])
    res = run_pcat(img, w, epsf, os_, cfg, nc=11)
    b = res.samples['beta'][:, 0, :]
    Tm = T.reshape(T.shape[0], -1)
    prec = Tm @ (w.ravel()[:, None] * Tm.T) + np.diag([0.0] + [1 / psig ** 2] * 4)
    cov = np.linalg.inv(prec)
    mean = cov @ (Tm @ (w.ravel() * img.ravel()))
    se = _batch_se(b)
    assert np.all(np.abs(b.mean(axis=0) - mean) < 4 * se), (b.mean(axis=0), mean, se)
    assert np.allclose(b.std(axis=0), np.sqrt(np.diag(cov)), rtol=0.12)
    assert res.samples['n'].max() == 0


def test_slope_prior_uniform():
    epsf, os_, _ = get_epsf()
    lo, hi = 1.3, 3.5
    cfg = PCATConfig(trueminf=1.0, flux_slope=2.0, sample_slope=True, slope_bounds=(lo, hi),
                     alpha_parsimony=0.04, nsamp=1000, nloop=40, regsize=14, float_bkg=False,
                     seed=9, burnin_frac=0.05, store_models=False, lib_dir=str(lib_dir()), verbose=0,
                     move_weights=dict(movestar=20., birth_death=40., merge_split=40.))
    shape = (30, 47)
    res = run_pcat(np.zeros(shape), np.zeros(shape), epsf, os_, cfg, nc=11)
    s = res.samples['slope'][res.post_burnin()]
    assert np.all((s >= lo) & (s <= hi))
    assert abs(s.mean() - 0.5 * (lo + hi)) < 4 * _batch_se(s)
    # variance of U(lo, hi): (hi - lo)^2 / 12
    v = (s - s.mean()) ** 2
    assert abs(v.mean() - (hi - lo) ** 2 / 12) < 4 * _batch_se(v)
