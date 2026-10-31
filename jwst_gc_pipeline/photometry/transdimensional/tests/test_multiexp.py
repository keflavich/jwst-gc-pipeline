"""Multi-band / multi-exposure options of the sampler: colour-space moves
(``color_moves``), the colour-scaled split fraction (``split_color_scale``) and
the parsimony parameter count (``pen_nparams``).

* prior recovery: with a flat likelihood (zero weights) and three bands (one
  tight colour prior, one broad), the chain samples the prior in N, pivot flux
  and both colours (tolerance: 4 batch-means standard errors, 20 batches);
* defaults: ``color_moves`` on a single band leaves the chain unchanged;
* a joint fit of two exposures of the same scene (different sub-pixel shifts,
  unit flux ratio through a 0.01 mag colour prior) recovers positions and
  fluxes and gives smaller flux posterior widths than the first exposure alone.
"""
import numpy as np
import pytest

from ._helpers import get_epsf, have_compiler, lib_dir

pytestmark = pytest.mark.skipif(not have_compiler(), reason='no C compiler')


def _batch_se(a, nb=20):
    a = np.asarray(a, float)
    m = a.size // nb
    return a[:m * nb].reshape(nb, m).mean(axis=1).std(ddof=1) / np.sqrt(nb)


def test_pen_nparams_sets_penalty():
    from ..pcat_sampler import BandData, PCATConfig, PCATSampler
    from ..psf_poly import build_cf
    epsf, os_, _ = get_epsf()
    cf, _ = build_cf(epsf, os_, 11)
    z = np.zeros((20, 22))
    bands = [BandData(z, z, cf, 11), BandData(z, z, cf, 11, name='b1')]
    common = dict(alpha_parsimony=0.8, regsize=14, color_mus=[0.0], color_sigs=[0.01],
                  lib_dir=str(lib_dir()), verbose=0)
    assert np.isclose(PCATSampler(bands, PCATConfig(**common)).pen, 2.0 * 0.8)
    assert np.isclose(PCATSampler(bands, PCATConfig(pen_nparams=3, **common)).pen, 1.5 * 0.8)
    with pytest.raises(ValueError):
        PCATSampler(bands, PCATConfig(split_color_scale=-1.0, **common))
    with pytest.raises(ValueError):
        PCATSampler(bands, PCATConfig(**dict(common, color_sigs=[0.0])))


def test_color_moves_single_band_is_default():
    from ..pcat_sampler import PCATConfig, run_pcat
    from .. import mocks
    epsf, os_, _ = get_epsf()
    tr = mocks.TruthRenderer(epsf, os_)
    rng = np.random.default_rng(2)
    img, w = mocks.make_mock((30, 34), np.array([10.2, 21.7]), np.array([12.5, 18.1]),
                             np.array([300., 80.]), tr, 1.0, 0.2, rng)
    common = dict(trueminf=2.0, nsamp=6, nloop=40, regsize=16, seed=4, lib_dir=str(lib_dir()), verbose=0)
    a = run_pcat(img, w, epsf, os_, PCATConfig(**common), nc=15)
    b = run_pcat(img, w, epsf, os_, PCATConfig(color_moves=True, color_step_fac=3.0,
                                               split_color_scale=1.0, **common), nc=15)
    for k in ('n', 'logL', 'x', 'y', 'f'):
        assert np.array_equal(a.samples[k], b.samples[k], equal_nan=True), k


MUS = np.array([0.02, -0.4])
SIGS = np.array([0.01, 0.3])


def _flat_three_band(cfg_kw, init=None):
    from ..pcat_sampler import BandData, PCATConfig, run_pcat
    from ..psf_poly import build_cf
    epsf, os_, _ = get_epsf()
    shape = (30, 47)
    nc = 11
    cf, _ = build_cf(epsf, os_, nc)
    z = np.zeros(shape)
    extra = [BandData(z, z, cf, nc, affine=np.array([[1, 0, 0.3], [0, 1, -0.2]]), name='tight'),
             BandData(z, z, cf, nc, affine=np.array([[1, 0, -0.4], [0, 1, 0.1]]), name='broad')]
    kw = dict(trueminf=1.0, flux_slope=2.0, nsamp=400, nloop=40, regsize=14, float_bkg=False,
              burnin_frac=0.05, store_models=False, lib_dir=str(lib_dir()), verbose=0,
              color_mus=MUS, color_sigs=SIGS, color_moves=True, split_color_scale=1.0,
              init_catalog=init)
    kw.update(cfg_kw)
    cfg = PCATConfig(**kw)
    return run_pcat(z, z, epsf, os_, cfg, nc=nc, extra_bands=extra)


def _check_flux_colour_prior(res):
    pb = res.post_burnin()
    f = res.samples['f'][pb]
    lnf = np.array([np.nanmean(np.log(r)) for r in f[:, 0] if np.isfinite(r).any()])
    assert abs(lnf.mean() - 1.0) < 4 * _batch_se(lnf), lnf.mean()
    for b in (1, 2):
        # per-sample mean of the standardised colour and of its square
        z1, z2 = [], []
        for s in range(f.shape[0]):
            ok = np.isfinite(f[s, 0])
            if not ok.any():
                continue
            c = -2.5 * np.log10(f[s, b, ok] / f[s, 0, ok])
            t = (c - MUS[b - 1]) / SIGS[b - 1]
            z1.append(t.mean())
            z2.append(np.mean(t * t))
        z1, z2 = np.array(z1), np.array(z2)
        assert abs(z1.mean()) < 4 * _batch_se(z1), (b, z1.mean())
        assert abs(z2.mean() - 1.0) < 4 * _batch_se(z2), (b, z2.mean())


def test_color_move_kernel_fixed_n():
    """movestar only (fixed N = 25): pivot flux power law and both colours."""
    rng = np.random.default_rng(1)
    n = 25
    f0 = 1.0 * np.exp(rng.exponential(1.0, n))
    f = np.vstack([f0, f0 * 10 ** (-0.4 * MUS[0]), f0 * 10 ** (-0.4 * MUS[1])])
    init = dict(x=rng.uniform(0, 46, n), y=rng.uniform(0, 29, n), f=f)
    res = _flat_three_band(dict(seed=11, color_step_fac=3.0, move_weights=dict(movestar=1.0)), init)
    assert np.all(res.samples['n'] == n)
    assert 0.2 < res.accept['movestar']['frac'] < 0.95, res.accept['movestar']
    _check_flux_colour_prior(res)


def _init_from_prior(n, seed):
    rng = np.random.default_rng(seed)
    f0 = 1.0 * np.exp(rng.exponential(1.0, n))
    c = rng.normal(MUS[:, None], SIGS[:, None], (2, n))
    f = np.vstack([f0, f0 * 10 ** (-0.4 * c)])
    return dict(x=rng.uniform(0, 46, n), y=rng.uniform(0, 29, n), f=f)


def test_prior_recovery_color_split():
    """split/merge-dominated chain (weights 20:20:60) with the colour-scaled split fraction
    and pen_nparams: N follows q^N, q = exp(-pen), mean q / (1 - q); pivot flux and colours
    follow their priors (a split proposal density that does not match its reverse merge
    biases the flux distribution).

    alpha = 0.15 gives mean N = 3.96 and an N autocorrelation time of 10-16 samples
    (flat-likelihood runs of this configuration, 10000 samples each), so 20 batches of
    250 samples give a valid batch-means error.  At alpha = 0.04 (mean N = 16.2) the
    autocorrelation time is 80-145 samples and a 400-sample chain cannot resolve N."""
    alpha = 0.15
    res = _flat_three_band(dict(seed=7, alpha_parsimony=alpha, color_step_fac=3.0, pen_nparams=3,
                                nsamp=5000,
                                move_weights=dict(movestar=20., birth_death=20., merge_split=60.)),
                           _init_from_prior(4, 3))
    assert np.isclose(res.info['pen'], 1.5 * alpha)
    n = res.samples['n'][res.post_burnin()]
    q = np.exp(-1.5 * alpha)
    assert abs(n.mean() - q / (1 - q)) < 4 * _batch_se(n), (n.mean(), q / (1 - q))
    _check_flux_colour_prior(res)
    assert res.accept['split']['nacc'] > 50 and res.accept['merge']['nacc'] > 50


def test_two_exposure_joint_fit():
    from .. import mocks
    from ..condense import condense
    from ..pcat_sampler import BandData, PCATConfig, run_pcat
    from ..psf_poly import build_cf
    epsf, os_, _ = get_epsf()
    rng = np.random.default_rng(9)
    tr = mocks.TruthRenderer(epsf, os_)
    shape = (44, 50)
    x = np.array([12.3, 30.7, 21.1, 38.4])
    y = np.array([10.2, 14.9, 30.5, 33.3])
    f0 = np.array([400., 120., 600., 60.])
    img0, w0 = mocks.make_mock(shape, x, y, f0, tr, 1.0, 0.2, rng)
    dx, dy = 0.43, -0.27
    img1, w1 = mocks.make_mock(shape, x + dx, y + dy, f0, tr, 1.0, 0.2, rng)
    nc = 25
    cf, _ = build_cf(epsf, os_, nc)
    fac = mocks.psf_snr_factor(epsf, os_)
    b1 = BandData(image=img1, weight=w1, cf=cf, nc=nc, affine=np.array([[1, 0, dx], [0, 1, dy]]),
                  name='exp2')
    common = dict(trueminf=5.0 / fac, nsamp=300, nloop=100, regsize=28, seed=3, lib_dir=str(lib_dir()),
                  verbose=0)
    init = dict(x=x + 0.2, y=y - 0.2, f=f0 * 0.9)
    cfg = PCATConfig(color_mus=[0.0], color_sigs=[0.01], color_moves=True, color_step_fac=3.0,
                     split_color_scale=1.0, pen_nparams=3,
                     init_catalog=dict(init, f=np.vstack([init['f'], init['f']])), **common)
    res = run_pcat(img0, w0, epsf, os_, cfg, nc=nc, extra_bands=[b1])
    res1 = run_pcat(img0, w0, epsf, os_, PCATConfig(init_catalog=init, **common), nc=nc)
    pb = res.post_burnin()
    c = condense(res.samples['x'], res.samples['y'], res.samples['f'], sample_idx=pb, return_samples=True)
    c1 = condense(res1.samples['x'], res1.samples['y'], res1.samples['f'], sample_idx=res1.post_burnin(),
                  return_samples=True)
    sel = c['prob'] > 0.5
    ic, it, d = mocks.match_truth(c['x'][sel], c['y'][sel], x, y, 1.0)
    assert ic.size == 4 and sel.sum() == 4
    assert np.all(d < 0.2)
    fl = c['flux'][:, sel][:, ic]
    assert np.all(np.abs(fl[0] / f0[it] - 1) < 0.1), fl[0] / f0[it]
    # the colour prior ties the exposure-2 flux to the pivot flux
    assert np.all(np.abs(fl[1] / fl[0] - 1) < 0.03), fl[1] / fl[0]
    # joint fit: smaller pivot-flux posterior width than exposure 1 alone (expected ratio 1/sqrt(2))
    sel1 = c1['prob'] > 0.5
    jc, jt, _ = mocks.match_truth(c1['x'][sel1], c1['y'][sel1], x, y, 1.0)
    sd2 = np.nanstd(c['samp_f'][np.flatnonzero(sel)[ic], 0, :], axis=1)
    sd1 = np.nanstd(c1['samp_f'][np.flatnonzero(sel1)[jc], 0, :], axis=1)
    o2 = np.argsort(it)
    o1 = np.argsort(jt)
    assert np.median(sd2[o2] / sd1[o1]) < 0.9, sd2[o2] / sd1[o1]
    assert res.accept['movestar']['frac'] > 0.15, res.accept['movestar']
    assert res.info['leak'] == 0


QUAD = np.array([[2.0e-4, -1.0e-4, 1.5e-4], [-1.0e-4, 2.5e-4, 1.0e-4]])


def test_quadratic_map_geometry():
    """``BandData.quad``: band positions follow M p + t + Q [x^2, xy, y^2] and the
    band-pixel -> pivot inverse used for region labels round-trips."""
    from ..pcat_sampler import BandData, PCATConfig, PCATSampler
    from ..psf_poly import build_cf
    epsf, os_, _ = get_epsf()
    cf, _ = build_cf(epsf, os_, 11)
    z = np.zeros((30, 34))
    A = np.array([[0.998, 0.012, 0.4], [-0.01, 1.001, -0.3]])
    bands = [BandData(z, z, cf, 11), BandData(z, z, cf, 11, affine=A, quad=QUAD, name='b1')]
    common = dict(regsize=14, color_mus=[0.0], color_sigs=[0.01], lib_dir=str(lib_dir()), verbose=0)
    s = PCATSampler(bands, PCATConfig(**common))
    rng = np.random.default_rng(5)
    x, y = rng.uniform(-2, 35, 50), rng.uniform(-2, 31, 50)
    bx, by = s._band_pos(1, x, y)
    q = np.vstack([x * x, x * y, y * y])
    exp = A[:, :2] @ np.vstack([x, y]) + A[:, 2:] + QUAD @ q
    assert np.allclose(bx - s.pads[1], exp[0], atol=1e-9) and np.allclose(by - s.pads[1], exp[1], atol=1e-9)
    sx, sy = s.pix_piv[1]
    P = s.pads[1]
    qy, qx = np.mgrid[0:sx.shape[0], 0:sx.shape[1]]
    rx, ry = s._band_pos(1, sx.ravel() - 0.5 - s.dpad, sy.ravel() - 0.5 - s.dpad)
    assert np.max(np.abs(rx - P - (qx.ravel() - P))) < 1e-6
    assert np.max(np.abs(ry - P - (qy.ravel() - P))) < 1e-6
    with pytest.raises(ValueError):
        PCATSampler([BandData(z, z, cf, 11, quad=QUAD), bands[1]], PCATConfig(**common))


def test_quadratic_map_joint_fit():
    """exposure 2 with a 0.2-0.9 px second-order distortion: the joint fit with
    ``quad`` recovers the positions and the unit flux ratio and fits exposure 2
    better than the same fit with the affine part only."""
    from .. import mocks
    from ..condense import condense
    from ..pcat_sampler import BandData, PCATConfig, run_pcat
    from ..psf_poly import build_cf
    epsf, os_, _ = get_epsf()
    rng = np.random.default_rng(19)
    tr = mocks.TruthRenderer(epsf, os_)
    shape = (44, 50)
    x = np.array([12.3, 30.7, 21.1, 38.4, 44.0])
    y = np.array([10.2, 14.9, 30.5, 33.3, 8.5])
    f0 = np.array([400., 120., 600., 150., 300.])
    img0, w0 = mocks.make_mock(shape, x, y, f0, tr, 1.0, 0.2, rng)
    dq = QUAD @ np.vstack([x * x, x * y, y * y])
    img1, w1 = mocks.make_mock(shape, x + 0.3 + dq[0], y - 0.2 + dq[1], f0, tr, 1.0, 0.2, rng)
    nc = 25
    cf, _ = build_cf(epsf, os_, nc)
    fac = mocks.psf_snr_factor(epsf, os_)
    A = np.array([[1, 0, 0.3], [0, 1, -0.2]])
    common = dict(trueminf=5.0 / fac, nsamp=40, nloop=100, regsize=28, seed=3, lib_dir=str(lib_dir()),
                  verbose=0, color_mus=[0.0], color_sigs=[0.01], color_moves=True, color_step_fac=3.0,
                  split_color_scale=1.0, pen_nparams=3,
                  init_catalog=dict(x=x + 0.1, y=y - 0.1, f=np.vstack([f0, f0]) * 0.9))
    out = {}
    for k, qd in (('quad', QUAD), ('affine', None)):
        b1 = BandData(image=img1, weight=w1, cf=cf, nc=nc, affine=A, quad=qd, name='exp2')
        out[k] = run_pcat(img0, w0, epsf, os_, PCATConfig(**common), nc=nc, extra_bands=[b1])
    res = out['quad']
    pb = res.post_burnin()
    c = condense(res.samples['x'], res.samples['y'], res.samples['f'], sample_idx=pb)
    sel = c['prob'] > 0.5
    ic, it, d = mocks.match_truth(c['x'][sel], c['y'][sel], x, y, 1.0)
    assert ic.size == 5 and sel.sum() == 5
    assert np.all(d < 0.15), d
    fl = c['flux'][:, sel][:, ic]
    assert np.all(np.abs(fl[0] / f0[it] - 1) < 0.1), fl[0] / f0[it]
    assert np.all(np.abs(fl[1] / fl[0] - 1) < 0.03), fl[1] / fl[0]
    chi2 = {k: np.mean(np.asarray(r.samples['chi2'])[r.post_burnin()], axis=0) for k, r in out.items()}
    # chi2 per pixel of exposure 2: 1.01 with quad, 1.15 affine-only (2200 pixels)
    assert chi2["quad"][1] < chi2["affine"][1] - 100, chi2
