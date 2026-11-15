"""Tests for scripts/analysis/satstar_poisson (issue #1008).

* ``noise_only`` (numpy only): the noise-only simulator and the sigma(chi) report that
  ``loo_crop.py`` with ``NOISEONLY=<seed>`` uses to calibrate the LOO statistic.
* ``patternfit.fit_nuisance(anchor=True)``: the anchor exposure holds its flux fixed and
  fits only the pedestal and the gradient (needs ``finufft``; skipped without it).
* ``loo_crop.py`` with ``NOISEONLY=<seed>`` end to end on synthetic cutouts (needs
  ``finufft``; about 3 minutes on a loaded node, Q solves capped at 10 CG iterations).
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


def test_json_safe_writes_strict_json_for_empty_bins(noise_only):
    import json
    rep = noise_only.radial_report(np.array([0.5, -1.0]), np.array([40., 60.]), np.ones(2, bool))
    assert np.isnan(rep['300-500']['sig'])
    text = json.dumps(noise_only.json_safe(dict(bins=rep, seed=3)), allow_nan=False)
    back = json.loads(text)
    assert back['bins']['300-500']['sig'] is None and back['bins']['300-500']['n'] == 0
    assert back['bins']['30-50']['n'] == 1 and back['seed'] == 3


# ---- end-to-end NOISEONLY run of loo_crop.py on synthetic cutouts (needs finufft) ----

NPIX = 64
SHIFTS = [(0.0, 0.0), (0.37, 0.61), (0.74, 0.22), (1.21, 0.88)]     # dither offsets (px)
GAIN = [1.0, 1.02, 0.98, 1.01]
PEDESTAL = [0.0, 6.0, -5.0, 9.0]


def _truth(qx, qy):
    """A pattern far below the band limit (0.45 cycles/px), so the truth is in the model family."""
    return (80.0+120.0*np.exp(-(qx**2+qy**2)/200.0)
            + 20.0*np.cos(2*np.pi*(0.07*qx+0.03*qy)))


def _write_cutouts(prefix, rng):
    """cutout.py-format npz files: identity jacobian, v2v3 = pixel + dither, star at the cutout centre
    on the sky, data = gain*truth(q) + pedestal + Gaussian noise of variance 4 + 0.05|signal|."""
    yy, xx = np.mgrid[:NPIX, :NPIX].astype(float)
    for k, (sx, sy) in enumerate(SHIFTS, 1):
        v2 = xx+sx; v3 = yy+sy
        sig = GAIN[k-1]*_truth(v2-NPIX/2, v3-NPIX/2)
        var = 4.0+0.05*np.abs(sig)
        d = sig+PEDESTAL[k-1]+rng.normal(0, 1, sig.shape)*np.sqrt(var)
        np.savez(f'{prefix}_e{k}.npz', d=d, dq=np.zeros(d.shape, np.uint32), var=var, vf=0*var, vr=0*var,
                 v2=v2, v3=v3, X0=0, Y0=0, xt=NPIX/2-sx, yt=NPIX/2-sy, vt=np.array([NPIX/2, NPIX/2]),
                 J=np.eye(2))


# loo_crop.py through runpy with the Q solves capped (one prox step, 10 CG iterations): the
# test checks what NOISEONLY does to the data and the output files, not convergence.
_DRIVER = """
import runpy, sys
sys.path.insert(0, {satstar!r})
import patternfit
_solve_Q = patternfit.solve_Q
patternfit.solve_Q = lambda *a, **k: _solve_Q(*a, **dict(k, maxiter=10, nprox=1))
sys.argv = ['loo_crop.py'] + sys.argv[1:]
runpy.run_path({script!r}, run_name='__main__')
"""


def _run_loo_crop(workdir, tag, noiseonly=None):
    import subprocess
    env = dict(os.environ, QN='64', QH='1', KMAX='0.45', MINCOV='2', OMP_NUM_THREADS='1')
    env.pop('NOISEONLY', None)
    if noiseonly is not None:
        env['NOISEONLY'] = noiseonly
    code = _DRIVER.format(satstar=SATSTAR, script=os.path.join(SATSTAR, 'loo_crop.py'))
    return subprocess.run([sys.executable, '-c', code, tag, '26', '2', '--prefix', 'tgt'],
                          cwd=workdir, env=env, capture_output=True, text=True, timeout=900)


@pytest.fixture(scope='module')
def loo_runs(tmp_path_factory):
    pytest.importorskip('finufft')
    wd = tmp_path_factory.mktemp('loo')
    _write_cutouts(str(wd/'tgt'), np.random.default_rng(1008))
    runs = {}
    for name, noiseonly in (('plain', None), ('s1', '1'), ('s2', '2')):
        r = _run_loo_crop(str(wd), name, noiseonly)
        assert r.returncode == 0, r.stdout[-2000:]+r.stderr[-2000:]
        runs[name] = dict(npz=dict(np.load(wd/f'{name}_e2.npz')), stdout=r.stdout, wd=wd)
    runs['real_d'] = np.load(wd/'tgt_e2.npz')['d']
    return runs


def test_loo_crop_plain_run_writes_no_simulated_data(loo_runs):
    assert 'd' not in loo_runs['plain']['npz'] and 'noiseonly_seed' not in loo_runs['plain']['npz']
    assert not (loo_runs['plain']['wd']/'plain_e2_chi.json').exists()


def test_loo_crop_noiseonly_stores_simulated_data_and_calibrated_chi(loo_runs):
    import json
    real = loo_runs['real_d']
    for name, seed in (('s1', 1), ('s2', 2)):
        z = loo_runs[name]['npz']
        assert int(z['noiseonly_seed']) == seed
        ok = np.isfinite(z['d']) & np.isfinite(z['pred'])
        assert ok.sum() > 2000
        assert not np.allclose(z['d'][ok], real[ok])           # simulated, not the observed, data
        # the simulation keeps the fitted pedestal/gradient (add()): the mean offset to the
        # observed data stays far below the 5-9 counts of pedestal
        assert abs(np.mean(z['d'][ok]-real[ok])) < 1.5
        # chi on the stored simulated data
        var = 4.0+0.05*np.abs(z['pred'])
        chi = (z['d']-z['pred'])/np.sqrt(var+z['varQ'])
        m = ok & (z['cov'] >= 2)
        c = chi[m]
        assert 0.8 < 1.4826*np.median(np.abs(c-np.median(c))) < 1.2
        rep = json.load(open(loo_runs[name]['wd']/f'{name}_e2_chi.json'))
        assert rep['seed'] == seed and rep['heldout'] == 2
        assert rep['bins']['300-500']['sig'] is None               # crop of 30 px: empty bin, strict JSON
    a, b = loo_runs['s1']['npz'], loo_runs['s2']['npz']
    m = np.isfinite(a['d']) & np.isfinite(b['d'])
    assert not np.allclose(a['d'][m], b['d'][m])                   # the seeds differ


def test_loo_crop_rejects_a_malformed_seed_before_fitting(loo_runs):
    pytest.importorskip('finufft')
    import time
    t0 = time.time()
    r = _run_loo_crop(str(loo_runs['plain']['wd']), 'bad', noiseonly='abc')
    assert r.returncode != 0 and 'not an integer seed' in r.stderr+r.stdout
    assert 'joint it' not in r.stdout and time.time()-t0 < 60
