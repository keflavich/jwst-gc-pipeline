"""The bright merge-split variant of the compiled engine (``bright_ms_frac``).

* it is off by default and then leaves the extra statistics rows empty;
* with a flat likelihood (zero weights) a chain whose merge/split proposals are all
  bright-variant proposals samples the prior: N follows P(N) ~ exp(-pen N), ln f the power law,
  the sources above the bright threshold are the fraction (f_br / f_min)^(1 - s) of N with
  ln(f / f_br) ~ Exp(s - 1), positions are uniform, and the number of pairs closer than the
  merge radius equals that of a chain without the bright variant (two-sample test);
* the incrementally updated model equals a full re-render after many bright proposals;
* invalid settings raise.

Sample sizes and tolerances (4 batch-means standard errors, 20 batches; combined for the
two-sample test) are fixed before the runs: 20000 samples of 40 proposals, burn-in 5%.
"""
import numpy as np
import pytest
from scipy import stats

from ..pcat_sampler import BandData, PCATConfig, run_pcat
from ._helpers import get_epsf, have_compiler, lib_dir

numba = pytest.importorskip('numba')

from ..fast_sampler import FastPCATSampler  # noqa: E402
from .test_fast_sampler import _scene  # noqa: E402

pytestmark = pytest.mark.skipif(not have_compiler(), reason='no C compiler')

_FBR = 3.0
_RMIN, _RMAX = 0.1, 4.0
_SHAPE = (30, 47)
_ALPHA = 0.04
_NSAMP = 20000


def _batch_se(a, nb=20):
    a = np.asarray(a, float)
    m = a.size // nb
    return a[:m * nb].reshape(nb, m).mean(axis=1).std(ddof=1) / np.sqrt(nb)


def _prior_run(bfrac, seed=5):
    epsf, os_, _ = get_epsf()
    cfg = PCATConfig(trueminf=1.0, flux_slope=2.0, alpha_parsimony=_ALPHA, nsamp=_NSAMP, nloop=40,
                     regsize=14, float_bkg=False, seed=seed, burnin_frac=0.05, store_models=False,
                     lib_dir=str(lib_dir()), verbose=0, engine='numba',
                     move_weights=dict(movestar=20., birth_death=40., merge_split=60.),
                     bright_ms_frac=bfrac, bright_ms_fbr=_FBR, bright_ms_rmin=_RMIN, bright_ms_rmax=_RMAX)
    return run_pcat(np.zeros(_SHAPE), np.zeros(_SHAPE), epsf, os_, cfg, nc=11)


def _pairs(x, y, rmin, rmax):
    """Number of pairs with rmin <= separation <= rmax in each sample (NaN-padded rows)."""
    out = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        k = np.isfinite(x[i])
        xi, yi = x[i, k], y[i, k]
        d = np.hypot(xi[:, None] - xi[None], yi[:, None] - yi[None])
        out[i] = np.sum(np.triu((d >= rmin) & (d <= rmax), 1))
    return out


def test_off_by_default():
    assert PCATConfig().bright_ms_frac == 0.0
    res = _prior_run(0.0)
    assert 'bright_split' not in res.accept
    assert res.info['bright_ms_frac'] == 0.0


@pytest.fixture(scope='module')
def prior_runs():
    return {'plain': _prior_run(0.0), 'all': _prior_run(1.0), 'half': _prior_run(0.5, seed=6)}


@pytest.mark.parametrize('which', ['all', 'half'])
def test_prior_recovery_bright_ms(prior_runs, which):
    res = prior_runs[which]
    assert res.info['bright_ms_frac'] == (1.0 if which == 'all' else 0.5)
    assert res.accept['bright_split']['nacc'] > 100 and res.accept['bright_merge']['nacc'] > 100
    pb = res.post_burnin()
    n = res.samples['n'][pb]
    q = np.exp(-1.5 * _ALPHA)
    assert abs(n.mean() - q / (1 - q)) < 4 * _batch_se(n)
    f = res.samples['f'][pb, 0]
    lnf = np.array([np.nanmean(np.log(r)) for r in f if np.isfinite(r).any()])
    assert abs(lnf.mean() - 1.0) < 4 * _batch_se(lnf)
    # sources above the bright threshold: a fraction 1 / f_br of N (s = 2), ln(f / f_br) ~ Exp(1)
    nb = np.nansum(f >= _FBR, axis=1)
    d = nb - n / _FBR
    assert abs(d.mean()) < 4 * _batch_se(d)
    lnb = np.array([np.mean(np.log(r[r >= _FBR] / _FBR)) if np.any(r >= _FBR) else np.nan for r in f])
    lnb = lnb[np.isfinite(lnb)]
    assert abs(lnb.mean() - 1.0) < 4 * _batch_se(lnb)
    # positions uniform
    x = res.samples['x'][pb].ravel()
    y = res.samples['y'][pb].ravel()
    x, y = x[np.isfinite(x)][::25], y[np.isfinite(y)][::25]
    assert stats.kstest((x + 0.5) / _SHAPE[1], 'uniform').pvalue > 1e-3
    assert stats.kstest((y + 0.5) / _SHAPE[0], 'uniform').pvalue > 1e-3
    xb = res.samples['x'][pb][f >= _FBR]
    yb = res.samples['y'][pb][f >= _FBR]
    assert stats.kstest((xb[::25] + 0.5) / _SHAPE[1], 'uniform').pvalue > 1e-3
    assert stats.kstest((yb[::25] + 0.5) / _SHAPE[0], 'uniform').pvalue > 1e-3


@pytest.mark.parametrize('which', ['all', 'half'])
def test_pair_counts_match_plain_chain(prior_runs, which):
    """Pairs in the merge-radius range follow the same distribution with and without the
    variant (a wrong separation density would pile up or deplete them)."""
    ref, res = prior_runs['plain'], prior_runs[which]
    pr = _pairs(ref.samples['x'][ref.post_burnin()][::10], ref.samples['y'][ref.post_burnin()][::10], _RMIN, _RMAX)
    ps = _pairs(res.samples['x'][res.post_burnin()][::10], res.samples['y'][res.post_burnin()][::10], _RMIN, _RMAX)
    se = np.hypot(_batch_se(pr), _batch_se(ps))
    assert abs(pr.mean() - ps.mean()) < 4 * se


def test_incremental_model_with_bright_ms():
    """The model kept by accepted bright proposals equals a full re-render."""
    cf, nc, img, w, init = _scene()
    cfg = PCATConfig(trueminf=15.0, flux_slope=2.0, nsamp=1, nloop=200, regsize=24, float_bkg=True,
                     fourier_order=1, seed=7, burnin_frac=0.0, store_models=False, lib_dir=str(lib_dir()),
                     verbose=0, init_catalog=init, bkg=0.3, engine='numba', stamp_tol=0.05,
                     stamp_sizes=(7, 11, 15), move_nstar=1, bright_ms_frac=1.0, bright_ms_fbr=60.0,
                     bright_ms_rmin=0.1, bright_ms_rmax=4.0)
    s = FastPCATSampler([BandData(image=img, weight=w, cf=cf, nc=nc, basis='quartic15', name='pivot')], cfg)
    wv = np.array([20.0, 20.0, 80.0, 5.0])
    cumw = np.cumsum(wv / wv.sum())
    cumw[-1] = 1 + 1e-12
    s.model32 = np.zeros_like(s.data32)
    s.resid32 = np.zeros_like(s.data32)
    for _ in range(8):
        s._set_offset(s.rng.integers(s.R), s.rng.integers(s.R))
        s._rebuild_fast()
        s.run_loop(200, cumw)
        m_inc, r_inc = s.model32.copy(), s.resid32.copy()
        assert not s.dm.any() and not s.dm2.any()
        s._rebuild_fast()
        np.testing.assert_allclose(m_inc, s.model32, rtol=0, atol=2e-6 * 3000.0)
        np.testing.assert_allclose(r_inc * s.wpos, s.resid32, rtol=0, atol=2e-6 * 3000.0)
        n = int(s.nst[0])
        col = np.floor((s.sx[:n] + 0.5 + s.dpad + s.ox) / s.R).astype(int)
        row = np.floor((s.sy[:n] + 0.5 + s.dpad + s.oy) / s.R).astype(int)
        lab = row * s.nregx + col
        np.testing.assert_array_equal(lab, s.sreg[:n])
        assert s.rcnt[:s.nlab].sum() == n
    # rows 9 and 10 of the statistics are the bright split and merge
    assert s.fstats[9, 1] > 0 and s.fstats[10, 1] > 0


@pytest.mark.parametrize('kw', [dict(bright_ms_frac=1.5), dict(bright_ms_frac=0.5, bright_ms_fbr=1.0),
                                dict(bright_ms_frac=0.5, bright_ms_rmin=2.0, bright_ms_rmax=1.0),
                                dict(bright_ms_frac=0.5, count_prior='legacy')])
def test_invalid_settings(kw):
    cf, nc, img, w, init = _scene()
    base = dict(trueminf=15.0, regsize=24, lib_dir=str(lib_dir()), verbose=0, engine='numba', bright_ms_fbr=60.0)
    base.update(kw)
    with pytest.raises(ValueError):
        FastPCATSampler([BandData(image=img, weight=w, cf=cf, nc=nc, basis='quartic15', name='pivot')],
                        PCATConfig(**base))
