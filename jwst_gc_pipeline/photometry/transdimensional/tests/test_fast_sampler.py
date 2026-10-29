"""The compiled engine (``engine='numba'``, :mod:`..fast_sampler`).

* prior recovery with a flat likelihood (zero weights), as in test_prior, for
  the PCAT star move and the ``move_nstar`` variant;
* the incrementally updated model equals a full re-render after many
  proposals, the scratch images are left clear and the region lists match the
  positions;
* the full render equals :class:`PCATSampler`'s model for the same catalogue;
* background templates factor into y and x functions;
* a central crop of the ``nc`` coefficient table equals the table built at the
  smaller size (the adaptive stamps rely on it).
"""
import numpy as np
import pytest

from ..pcat_sampler import BandData, PCATConfig, PCATSampler, fourier_templates, run_pcat
from ..psf_poly import build_cf
from ._helpers import get_epsf, have_compiler, lib_dir

numba = pytest.importorskip('numba')

from ..fast_sampler import FastPCATSampler, separable_factors  # noqa: E402

pytestmark = pytest.mark.skipif(not have_compiler(), reason='no C compiler')


def _batch_se(a, nb=20):
    a = np.asarray(a, float)
    m = a.size // nb
    return a[:m * nb].reshape(nb, m).mean(axis=1).std(ddof=1) / np.sqrt(nb)


@pytest.mark.parametrize('move_nstar', [0, 2])
def test_prior_recovery_exact_numba(move_nstar):
    epsf, os_, _ = get_epsf()
    shape = (30, 47)
    alpha = 0.04
    cfg = PCATConfig(trueminf=1.0, flux_slope=2.0, alpha_parsimony=alpha, nsamp=600, nloop=40,
                     regsize=14, float_bkg=False, seed=5, burnin_frac=0.05, store_models=False,
                     lib_dir=str(lib_dir()), verbose=0, engine='numba', move_nstar=move_nstar,
                     move_weights=dict(movestar=20., birth_death=40., merge_split=40.))
    res = run_pcat(np.zeros(shape), np.zeros(shape), epsf, os_, cfg, nc=11)
    assert res.info['engine'] == 'numba'
    pb = res.post_burnin()
    n = res.samples['n'][pb]
    q = np.exp(-1.5 * alpha)
    assert abs(n.mean() - q / (1 - q)) < 4 * _batch_se(n)
    lnf = np.array([np.nanmean(np.log(r)) for r in res.samples['f'][pb, 0] if np.isfinite(r).any()])
    assert abs(lnf.mean() - 1.0) < 4 * _batch_se(lnf)


def _scene(seed=3, shape=(70, 83), nstar=25, nc=21):
    epsf, os_, _ = get_epsf()
    cf, _ = build_cf(epsf, os_, nc)
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, shape[1] - 1, nstar)
    y = rng.uniform(0, shape[0] - 1, nstar)
    f = 20.0 * np.exp(rng.exponential(1.0, nstar))
    f[:3] = [3000.0, 800.0, 400.0]
    img = rng.normal(0.0, 1.0, shape) + 0.3
    w = np.ones(shape)
    w[5:9, 40:44] = 0.0
    return cf, nc, img, w, dict(x=x, y=y, f=f)


def _sampler(engine, stamp_tol=None, move_nstar=0, nsamp=1):
    cf, nc, img, w, init = _scene()
    cfg = PCATConfig(trueminf=15.0, flux_slope=2.0, nsamp=nsamp, nloop=200, regsize=24, float_bkg=True,
                     fourier_order=1, seed=7, burnin_frac=0.0, store_models=False, lib_dir=str(lib_dir()),
                     verbose=0, init_catalog=init, bkg=0.3, engine=engine, stamp_tol=stamp_tol,
                     stamp_sizes=(7, 11, 15), move_nstar=move_nstar)
    bands = [BandData(image=img, weight=w, cf=cf, nc=nc, basis='quartic15', name='pivot')]
    cls = FastPCATSampler if engine == 'numba' else PCATSampler
    return cls(bands, cfg)


def test_full_render_matches_python():
    fs = _sampler('numba')
    fs.model32 = np.zeros_like(fs.data32)
    fs.resid32 = np.zeros_like(fs.data32)
    fs._rebuild_fast()
    ps = _sampler('python')
    ps._rebuild_model()
    ny, nx = ps.model[0].shape
    np.testing.assert_allclose(fs.model32[:, :nx], ps.model[0], rtol=0, atol=2e-3)
    assert not fs.model32[:, nx:].any()


@pytest.mark.parametrize('stamp_tol,move_nstar', [(None, 0), (0.05, 0), (0.05, 1)])
def test_incremental_model_consistent(stamp_tol, move_nstar):
    s = _sampler('numba', stamp_tol=stamp_tol, move_nstar=move_nstar)
    names = ['movestar', 'birth_death', 'merge_split', 'bkg']
    wv = np.array([60.0, 60.0, 30.0, 5.0])
    cumw = np.cumsum(wv / wv.sum())
    cumw[-1] = 1 + 1e-12
    s.model32 = np.zeros_like(s.data32)
    s.resid32 = np.zeros_like(s.data32)
    peak = 3000.0
    for _ in range(5):
        s._set_offset(s.rng.integers(s.R), s.rng.integers(s.R))
        s._rebuild_fast()
        s.run_loop(200, cumw)
        m_inc, r_inc = s.model32.copy(), s.resid32.copy()
        assert not s.dm.any() and not s.dm2.any()
        s._rebuild_fast()
        np.testing.assert_allclose(m_inc, s.model32, rtol=0, atol=2e-6 * peak)
        np.testing.assert_allclose(r_inc * s.wpos, s.resid32, rtol=0, atol=2e-6 * peak)
        n = int(s.nst[0])
        col = np.floor((s.sx[:n] + 0.5 + s.dpad + s.ox) / s.R).astype(int)
        row = np.floor((s.sy[:n] + 0.5 + s.dpad + s.oy) / s.R).astype(int)
        lab = row * s.nregx + col
        np.testing.assert_array_equal(lab, s.sreg[:n])
        for r in np.unique(lab):
            np.testing.assert_array_equal(np.sort(s.rlist[r, :s.rcnt[r]]), np.flatnonzero(lab == r))
        assert s.rcnt[:s.nlab].sum() == n
    assert s.fstats[names.index('movestar'), 2] > 0
    assert s.fstats[names.index('birth_death'), 1] > 0


def test_separable_factors():
    ny, nx = 23, 31
    T = np.concatenate([np.ones((1, ny, nx)), fourier_templates(ny, nx, 2)]).reshape(17, -1)
    Vy, Ux, tx = separable_factors(T, ny, nx)
    assert Ux.shape[0] == 5
    np.testing.assert_allclose(np.einsum('ti,tj->tij', Vy, Ux[tx]).reshape(T.shape), T, atol=1e-12)
    yy, xx = np.mgrid[:ny, :nx]
    assert separable_factors(np.vstack([T, (xx + yy).ravel()[None] * 1.0]), ny, nx) is None


def test_crop_equals_smaller_table():
    epsf, os_, _ = get_epsf()
    big, _ = build_cf(epsf, os_, 31)
    small, _ = build_cf(epsf, os_, 11)
    k = big.shape[0]
    crop = big.reshape(k, 31, 31)[:, 10:21, 10:21]
    np.testing.assert_allclose(crop, small.reshape(k, 11, 11), rtol=0, atol=1e-7 * np.abs(big).max())
