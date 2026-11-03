"""Free flux scale of the non-pivot bands (``band_scale_sig``, move ``bscale``; python engine).

* off by default: no ``cmu`` trace, ``bscale`` in the move weights without ``band_scale_sig``
  (and the reverse) raises;
* prior recovery: with a flat likelihood (zero weights) the centre mu of the colour prior has its
  hyperprior N(mu0, tau) as marginal, and the colours stay N(mu, sig) around the current mu, so
  the standardised colours (c - mu) / sig have mean 0 and variance 1, the pivot fluxes follow the
  power law and N follows q^N (tolerance: 4 batch-means standard errors, 20 batches; 5000
  samples of 40 proposals, fixed before the runs);
* data: a second exposure whose fluxes are 4% above the first, tied by a 0.01 mag colour prior,
  gives a posterior mean mu = -2.5 log10(1.04) = -0.0425 mag within 0.015 mag, and a lower chi2
  of that exposure than the same run with the centre fixed at zero.
"""
import numpy as np
import pytest

from ._helpers import get_epsf, have_compiler, lib_dir

pytestmark = pytest.mark.skipif(not have_compiler(), reason='no C compiler')

MU0, SIG, TAU = 0.02, 0.05, 0.1


def _batch_se(a, nb=20):
    a = np.asarray(a, float)
    m = a.size // nb
    return a[:m * nb].reshape(nb, m).mean(axis=1).std(ddof=1) / np.sqrt(nb)


def _flat_two_band(cfg_kw):
    from ..pcat_sampler import BandData, PCATConfig, run_pcat
    from ..psf_poly import build_cf
    epsf, os_, _ = get_epsf()
    shape = (30, 47)
    nc = 11
    cf, _ = build_cf(epsf, os_, nc)
    z = np.zeros(shape)
    extra = [BandData(z, z, cf, nc, affine=np.array([[1, 0, 0.3], [0, 1, -0.2]]), name='b1')]
    kw = dict(trueminf=1.0, flux_slope=2.0, nsamp=5000, nloop=40, regsize=14, float_bkg=False,
              burnin_frac=0.05, store_models=False, lib_dir=str(lib_dir()), verbose=0, seed=3,
              alpha_parsimony=0.15, pen_nparams=3, color_mus=[MU0], color_sigs=[SIG], color_moves=True,
              color_step_fac=3.0, split_color_scale=1.0,
              move_weights=dict(movestar=20., birth_death=20., merge_split=60.))
    kw.update(cfg_kw)
    return run_pcat(z, z, epsf, os_, PCATConfig(**kw), nc=nc, extra_bands=extra)


def test_off_by_default():
    res = _flat_two_band(dict(nsamp=20))
    assert 'cmu' not in res.samples and 'bscale' not in res.accept


@pytest.mark.parametrize('kw', [dict(band_scale_sig=TAU),                      # no 'bscale' weight
                                dict(move_weights=dict(movestar=1., bscale=1.)),   # weight without the sigma
                                dict(band_scale_sig=-1.0,
                                     move_weights=dict(movestar=1., bscale=1.)),
                                dict(band_scale_sig=TAU, band_scale_nstep=0,
                                     move_weights=dict(movestar=1., bscale=1.))])
def test_invalid_settings(kw):
    with pytest.raises(ValueError):
        _flat_two_band(dict(nsamp=2, **kw))


def test_prior_recovery_band_scale():
    res = _flat_two_band(dict(band_scale_sig=TAU, band_scale_steps=(0.01, 0.05, 0.2),
                              move_weights=dict(movestar=20., birth_death=20., merge_split=60., bscale=10.)))
    assert res.accept['bscale']['nacc'] > 1000
    pb = res.post_burnin()
    mu = res.samples['cmu'][pb, 0]
    assert abs(mu.mean() - MU0) < 4 * _batch_se(mu), mu.mean()
    sq = (mu - MU0) ** 2
    assert abs(sq.mean() - TAU ** 2) < 4 * _batch_se(sq), sq.mean()
    n = res.samples['n'][pb]
    q = np.exp(-res.info['pen'])
    assert abs(n.mean() - q / (1 - q)) < 4 * _batch_se(n), n.mean()
    f = res.samples['f'][pb]
    lnf = np.array([np.nanmean(np.log(r)) for r in f[:, 0] if np.isfinite(r).any()])
    assert abs(lnf.mean() - 1.0) < 4 * _batch_se(lnf), lnf.mean()
    z1, z2 = [], []
    for s in range(f.shape[0]):
        ok = np.isfinite(f[s, 0])
        if not ok.any():
            continue
        c = -2.5 * np.log10(f[s, 1, ok] / f[s, 0, ok])
        t = (c - res.samples['cmu'][pb][s, 0]) / SIG
        z1.append(t.mean())
        z2.append(np.mean(t * t))
    z1, z2 = np.array(z1), np.array(z2)
    assert abs(z1.mean()) < 4 * _batch_se(z1), z1.mean()
    assert abs(z2.mean() - 1.0) < 4 * _batch_se(z2), z2.mean()


def test_recovers_band_scale():
    from .. import mocks
    from ..pcat_sampler import BandData, PCATConfig, run_pcat
    from ..psf_poly import build_cf
    epsf, os_, _ = get_epsf()
    rng = np.random.default_rng(23)
    tr = mocks.TruthRenderer(epsf, os_)
    shape = (44, 50)
    x = np.array([12.3, 30.7, 21.1, 38.4, 44.0])
    y = np.array([10.2, 14.9, 30.5, 33.3, 8.5])
    f0 = np.array([2000., 1200., 3000., 1500., 2500.])
    scale = 1.04
    img0, w0 = mocks.make_mock(shape, x, y, f0, tr, 1.0, 0.2, rng)
    img1, w1 = mocks.make_mock(shape, x + 0.3, y - 0.2, f0 * scale, tr, 1.0, 0.2, rng)
    nc = 25
    cf, _ = build_cf(epsf, os_, nc)
    fac = mocks.psf_snr_factor(epsf, os_)
    out = {}
    for k, sig in (('free', 0.1), ('fixed', None)):
        b1 = BandData(image=img1, weight=w1, cf=cf, nc=nc, affine=np.array([[1, 0, 0.3], [0, 1, -0.2]]),
                      name='exp2')
        mw = dict(movestar=60., birth_death=60., merge_split=30., bkg=5.)
        if sig is not None:
            mw['bscale'] = 5.
        cfg = PCATConfig(trueminf=5.0 / fac, nsamp=60, nloop=100, regsize=28, seed=3, lib_dir=str(lib_dir()),
                         verbose=0, color_mus=[0.0], color_sigs=[0.01], color_moves=True, color_step_fac=3.0,
                         split_color_scale=1.0, pen_nparams=3, move_weights=mw, band_scale_sig=sig,
                         init_catalog=dict(x=x + 0.1, y=y - 0.1, f=np.vstack([f0, f0]) * 0.97))
        out[k] = run_pcat(img0, w0, epsf, os_, cfg, nc=nc, extra_bands=[b1])
    res = out['free']
    pb = res.post_burnin()
    mu = res.samples['cmu'][pb, 0]
    assert abs(mu.mean() - (-2.5 * np.log10(scale))) < 0.015, (mu.mean(), mu.std())
    c2 = {k: np.mean(np.asarray(r.samples['chi2'])[r.post_burnin()][:, 1]) for k, r in out.items()}
    assert c2['free'] < c2['fixed'], c2
    assert res.info['leak'] == 0
