"""The compiled engine (``engine='numba'``, :mod:`..fast_sampler`).

* prior recovery with a flat likelihood (zero weights), as in test_prior, for
  the PCAT star move and the ``move_nstar`` variant;
* prior recovery with residual-guided births (``birth_resid_frac``) under a
  flat likelihood whose residual still concentrates the birth density;
* the incrementally updated model equals a full re-render after many
  proposals, the scratch images are left clear and the region lists match the
  positions;
* the full render equals :class:`PCATSampler`'s model for the same catalogue;
* background templates factor into y and x functions;
* a central crop of the ``nc`` coefficient table equals the table built at the
  smaller size (the adaptive stamps rely on it);
* the adaptive burn-in (``burnin_auto``) finds the end of a relaxation and
  drops the samples before it from the model mean.
"""
import numpy as np
import pytest

from ..pcat_sampler import BandData, PCATConfig, PCATSampler, fourier_templates, run_pcat
from ..psf_poly import build_cf
from ._helpers import get_epsf, have_compiler, lib_dir

numba = pytest.importorskip('numba')

from ..fast_sampler import FastPCATSampler, equilibration_index, separable_factors  # noqa: E402

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


def test_prior_recovery_resid_births():
    """Weights of 1e-16 make the likelihood flat, while the residual-guided birth
    density (independent of the weight scale) puts most residual births on a
    6 x 8 px block 5 sigma above the rest.  Positions must stay uniform: the
    number of stars on the block is a/A of N, as are N and ln f of the prior."""
    epsf, os_, _ = get_epsf()
    shape = (30, 47)
    alpha = 0.04
    rng = np.random.default_rng(1)
    img = rng.normal(0.0, 1.0, shape)
    img[10:16, 20:28] += 5.0
    # residual births mix N more slowly here than uniform ones: 200 loops per sample and a 20%
    # burn-in from the empty start (40 and 5% left N 10% low over 24 seeds)
    cfg = PCATConfig(trueminf=1.0, flux_slope=2.0, alpha_parsimony=alpha, nsamp=1500, nloop=200,
                     regsize=14, float_bkg=False, bkg=[0.0], seed=5, burnin_frac=0.2, store_models=False,
                     lib_dir=str(lib_dir()), verbose=0, engine='numba', birth_resid_frac=0.7,
                     move_weights=dict(movestar=20., birth_death=40., merge_split=40.))
    res = run_pcat(img, np.full(shape, 1e-16), epsf, os_, cfg, nc=11)
    assert res.info['birth_resid_frac'] == 0.7
    pb = res.post_burnin()
    n = res.samples['n'][pb]
    q = np.exp(-1.5 * alpha)
    assert abs(n.mean() - q / (1 - q)) < 4 * _batch_se(n)
    x, y = res.samples['x'][pb], res.samples['y'][pb]
    on = (x >= 19.5) & (x < 27.5) & (y >= 9.5) & (y < 15.5)
    d = on.sum(axis=1) - 48.0 / (shape[0] * shape[1]) * n
    assert abs(d.mean()) < 4 * _batch_se(d)
    lnf = np.array([np.nanmean(np.log(r)) for r in res.samples['f'][pb, 0] if np.isfinite(r).any()])
    assert abs(lnf.mean() - 1.0) < 4 * _batch_se(lnf)


def test_resid_births_need_integer_pad():
    cf, nc, img, w, init = _scene()
    cfg = PCATConfig(trueminf=15.0, regsize=24, domain_pad=0.5, birth_resid_frac=0.5, engine='numba',
                     lib_dir=str(lib_dir()), verbose=0)
    with pytest.raises(ValueError, match='integer domain_pad'):
        FastPCATSampler([BandData(image=img, weight=w, cf=cf, nc=nc, basis='quartic15', name='pivot')], cfg)


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


def _sampler(engine, stamp_tol=None, move_nstar=0, nsamp=1, birth_resid_frac=0.0):
    cf, nc, img, w, init = _scene()
    cfg = PCATConfig(trueminf=15.0, flux_slope=2.0, nsamp=nsamp, nloop=200, regsize=24, float_bkg=True,
                     fourier_order=1, seed=7, burnin_frac=0.0, store_models=False, lib_dir=str(lib_dir()),
                     verbose=0, init_catalog=init, bkg=0.3, engine=engine, stamp_tol=stamp_tol,
                     stamp_sizes=(7, 11, 15), move_nstar=move_nstar, birth_resid_frac=birth_resid_frac)
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


@pytest.mark.parametrize('stamp_tol,move_nstar,wres', [(None, 0, 0.0), (0.05, 0, 0.0), (0.05, 1, 0.0),
                                                        (0.05, 1, 0.5)])
def test_incremental_model_consistent(stamp_tol, move_nstar, wres):
    s = _sampler('numba', stamp_tol=stamp_tol, move_nstar=move_nstar, birth_resid_frac=wres)
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
        if wres > 0:
            # birth weights and their sums per region and per row and region column, kept up to
            # date by the accepted proposals
            ok = (s.rmap[:, None] >= 0) & (s.cmap[None, :] >= 0)
            w = np.where(ok, np.maximum(r_inc.astype(float) * s.swt32, 0.0) ** 2, 0.0)
            np.testing.assert_allclose(s.wimg, w, rtol=1e-6, atol=0)
            wi = s.wimg.astype(float)
            lab = np.where(ok, s.rmap[:, None] * s.nregx + s.cmap[None, :], -1)
            z = np.bincount(lab[ok], weights=wi[ok], minlength=s.nlab)
            np.testing.assert_allclose(s.zreg[:s.nlab], z, rtol=1e-9, atol=1e-9 * z.max())
            for cg in range(s.nregx):
                rs = (wi * (ok & (s.cmap[None, :] == cg))).sum(axis=1)
                np.testing.assert_allclose(s.rowsum[:, cg], rs, rtol=1e-9, atol=1e-9 * z.max())
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


def test_equilibration_index():
    rng = np.random.default_rng(1)
    t = np.arange(2000)
    # logL relaxes as exp(-t/40) from 3000 below its stationary level (sd 10); N from 600 to 450
    logL = -3000.0 * np.exp(-t / 40.0) + rng.normal(0, 10, t.size)
    n = np.round(450 + 150 * np.exp(-t / 40.0) + rng.normal(0, 5, t.size))
    # logL is within 2 sd at t = 40 ln(3000/20) = 200; N enters its 1-99 per cent range near t = 120
    assert 170 < equilibration_index(logL, n) < 230
    assert equilibration_index(rng.normal(0, 1, 500), np.full(500, 7.0)) < 5


def test_adaptive_burnin():
    cf, nc, img, w, init = _scene()
    init = dict(init, f=init['f'] * 4.0)
    cfg = PCATConfig(trueminf=15.0, flux_slope=2.0, nsamp=400, nloop=10, regsize=24, float_bkg=True,
                     fourier_order=1, seed=11, store_models=True, lib_dir=str(lib_dir()), verbose=0,
                     init_catalog=init, bkg=0.3, engine='numba', move_nstar=1, burnin=5,
                     burnin_auto=True, target_ess=1e9, ess_min_samples=50, ess_check_every=25)
    bands = [BandData(image=img, weight=w, cf=cf, nc=nc, basis='quartic15', name='pivot')]
    res = FastPCATSampler(bands, cfg).run()
    info = res.info
    assert info['stop_reason'] == 'nsamp' and info['nsamp_run'] == 400
    assert info['t_eq'] >= 5
    assert info['nburn'] >= 2 * info['t_eq'] and (info['nburn'] - 5) % 25 == 0
    pb = res.post_burnin()
    assert pb[0] == info['nburn'] and pb.size == 400 - info['nburn']
    # the burn-in removes the start of the relaxation: post-burn-in logL is stationary
    lp = res.samples['logL'][pb]
    assert lp[:lp.size // 2].mean() > lp[lp.size // 2:].mean() - 3 * lp.std()
    assert res.samples['logL'][0] < lp.min()
    assert all(len(h) == 3 for h in info['ess_history'])
