"""With a flat likelihood (zero weights) the chain samples the prior.

Checks the transdimensional acceptance ratios (birth/death, split/merge with
the exact count terms): N follows P(N) ~ exp(-pen N), pivot fluxes follow the
power law (ln(f/fmin) ~ Exp(s-1)) and positions are uniform over a
non-square image whose size is not a multiple of the region size.
Tolerances: 4 batch-means standard errors (20 batches).
"""
import numpy as np
import pytest
from scipy import stats

from ..pcat_sampler import PCATConfig, run_pcat
from ._helpers import get_epsf, have_compiler, lib_dir


def _batch_se(a, nb=20):
    a = np.asarray(a, float)
    m = a.size // nb
    return a[:m * nb].reshape(nb, m).mean(axis=1).std(ddof=1) / np.sqrt(nb)


@pytest.mark.skipif(not have_compiler(), reason='no C compiler')
def test_prior_recovery_exact():
    epsf, os_, _ = get_epsf()
    shape = (30, 47)
    alpha = 0.04
    cfg = PCATConfig(trueminf=1.0, flux_slope=2.0, alpha_parsimony=alpha, nsamp=400, nloop=40,
                     regsize=14, float_bkg=False, seed=5, burnin_frac=0.05, store_models=False,
                     lib_dir=str(lib_dir()), verbose=0,
                     move_weights=dict(movestar=20., birth_death=40., merge_split=40.))
    res = run_pcat(np.zeros(shape), np.zeros(shape), epsf, os_, cfg, nc=11)
    pb = res.post_burnin()
    n = res.samples['n'][pb]
    q = np.exp(-1.5 * alpha)
    assert abs(n.mean() - q / (1 - q)) < 4 * _batch_se(n)
    lnf = np.array([np.nanmean(np.log(r)) for r in res.samples['f'][pb, 0] if np.isfinite(r).any()])
    assert abs(lnf.mean() - 1.0) < 4 * _batch_se(lnf)
    x = res.samples['x'][pb].ravel()
    y = res.samples['y'][pb].ravel()
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    # thin to roughly independent draws before the KS test
    x, y = x[::25], y[::25]
    assert stats.kstest((x + 0.5) / shape[1], 'uniform').pvalue > 1e-3
    assert stats.kstest((y + 0.5) / shape[0], 'uniform').pvalue > 1e-3
