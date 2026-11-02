"""Condensed catalog (Portillo+2017) on synthetic chains with known answers."""
import numpy as np
import pytest

from ..condense import condense, njit, _mutual_closest_edges, _seeds

needs_numba = pytest.mark.skipif(njit is None, reason='numba not installed')


def _chains(nsamp=200, seed=0):
    """Synthetic chain: A (always present), B (60 %), C (25 %), a close pair
    D1/D2 1.6 px apart (always), plus one unrelated random source per sample."""
    rng = np.random.default_rng(seed)
    truth = dict(A=(10.0, 10.0, 100.0, 1.0), B=(20.0, 14.0, 30.0, 0.60), C=(31.0, 30.0, 8.0, 0.25),
                 D1=(40.0, 10.0, 50.0, 1.0), D2=(41.6, 10.0, 40.0, 1.0))
    sig_pos = 0.05
    nmax = 8
    x = np.full((nsamp, nmax), np.nan)
    y = np.full((nsamp, nmax), np.nan)
    f = np.full((nsamp, 1, nmax), np.nan)
    present = {}
    for k, (tx, ty, tf, p) in truth.items():
        present[k] = rng.uniform(size=nsamp) < p if p < 1 else np.ones(nsamp, bool)
    for s in range(nsamp):
        j = 0
        for k, (tx, ty, tf, p) in truth.items():
            if present[k][s]:
                x[s, j] = tx + rng.normal(0, sig_pos)
                y[s, j] = ty + rng.normal(0, sig_pos)
                f[s, 0, j] = tf * (1 + 0.03 * rng.normal())
                j += 1
        x[s, j], y[s, j] = rng.uniform(0, 60, 2)       # unrelated spurious source
        f[s, 0, j] = 3.0
        order = rng.permutation(nmax)                   # storage order is arbitrary
        x[s], y[s], f[s, 0] = x[s, order], y[s, order], f[s, 0, order]
    return x, y, f, truth, present, sig_pos


def _find(c, tx, ty):
    d = np.hypot(c['x'] - tx, c['y'] - ty)
    k = int(np.argmin(d))
    assert d[k] < 0.2
    return k


def test_prevalence_flux_and_errors():
    x, y, f, truth, present, sig = _chains()
    c = condense(x, y, f)
    for name, (tx, ty, tf, p) in truth.items():
        k = _find(c, tx, ty)
        assert c['prob'][k] == pytest.approx(present[name].mean(), abs=1e-12)
        assert c['flux'][0, k] == pytest.approx(tf, rel=0.02)
        assert c['flux_err'][0, k] == pytest.approx(0.03 * tf, rel=0.25)
        assert c['x_err'][k] == pytest.approx(sig, rel=0.3)
        assert c['y_err'][k] == pytest.approx(sig, rel=0.3)
        assert c['flux_p16'][0, k] < c['flux'][0, k] < c['flux_p84'][0, k]
    # the unrelated random sources rarely line up: nothing else above 5 %
    known = [_find(c, tx, ty) for tx, ty, *_ in truth.values()]
    other = np.setdiff1d(np.arange(c['prob'].size), known)
    assert np.all(c['prob'][other] < 0.05)


def test_thinning_and_sample_subset():
    x, y, f, truth, present, sig = _chains()
    c = condense(x, y, f, sample_idx=np.arange(100, 200), max_samples=50)
    assert c['nsamp_used'] == 50
    k = _find(c, 10.0, 10.0)
    assert c['prob'][k] == 1.0


def test_seed_graph_thinning():
    """Seeding from 40 of 200 samples gives the same sources and the same
    prevalences (step 3 uses all 200 samples)."""
    x, y, f, truth, present, sig = _chains()
    full = condense(x, y, f, seed_max_samples=None)
    thin = condense(x, y, f, seed_max_samples=40)
    assert thin['nsamp_used'] == 200 and thin['nsamp_seed'] == 40
    for name, (tx, ty, tf, p) in truth.items():
        kf, kt = _find(full, tx, ty), _find(thin, tx, ty)
        assert thin['prob'][kt] == full['prob'][kf] == pytest.approx(present[name].mean(), abs=1e-12)
        assert thin['flux'][0, kt] == full['flux'][0, kf]


@pytest.mark.parametrize('engine', ['numpy', pytest.param('numba', marks=needs_numba)])
def test_mutual_closest_edges(engine):
    from ..condense import _mutual_closest_edges_numba
    mce = _mutual_closest_edges if engine == 'numpy' else _mutual_closest_edges_numba
    # sample 0: points a (0,0), b (0.5,0); sample 1: point c (0.3, 0)
    X = np.array([0.0, 0.5, 0.3])
    Y = np.zeros(3)
    sid = np.array([0, 0, 1])
    e = mce(X, Y, sid, 0.75)
    # c's closest in sample 0 is b (0.2) and b's closest in sample 1 is c
    assert e.tolist() == [[1, 2]]


@needs_numba
def test_engines_identical():
    """The compiled grid search returns the numpy / cKDTree catalogue exactly, on the
    synthetic chains and on a dense field with exact distance and flux ties (grid
    positions, quantised fluxes)."""
    from ..condense import _mutual_closest_edges_numba
    x, y, f, *_ = _chains()
    rng = np.random.default_rng(4)
    ns, nmax = 120, 300
    xd = rng.integers(0, 60, (ns, nmax)) * 0.25 + rng.normal(0, 0.02, (ns, 1)) * (rng.uniform(size=(ns, nmax)) < 0.5)
    yd = rng.integers(0, 60, (ns, nmax)) * 0.25
    fd = rng.integers(1, 4, (ns, 1, nmax)).astype(float)
    xd[:, -20:] = np.nan
    for xx, yy, ff in ((x, y, f), (xd, yd, fd)):
        a = condense(xx, yy, ff, engine='numba', return_samples=True, seed_max_samples=50)
        b = condense(xx, yy, ff, engine='numpy', return_samples=True, seed_max_samples=50)
        assert a['x'].size > 0
        for k in ('x', 'y', 'prob', 'degree', 'flux', 'flux_err', 'samp_x', 'samp_y', 'samp_f'):
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    good = np.isfinite(xd)
    sid = np.repeat(np.arange(ns), nmax).reshape(ns, nmax)[good]
    np.testing.assert_array_equal(_mutual_closest_edges_numba(xd[good], yd[good], sid, 0.75),
                                  _mutual_closest_edges(xd[good], yd[good], sid, 0.75))


def test_seed_selection_degree_updates():
    # star graph centre 0 with leaves 1..3, and an edge 4-5
    edges = np.array([[0, 1], [0, 2], [0, 3], [4, 5]])
    seeds, deg = _seeds(6, edges)
    assert seeds[0] == 0 and deg[0] == 3
    assert set(seeds[1:].tolist()) == {4} or set(seeds[1:].tolist()) == {5}
    assert deg[1] == 1


def test_empty_chain():
    x = np.full((5, 1), np.nan)
    c = condense(x, x, x[:, None, :])
    assert c['x'].size == 0 and c['nsamp_used'] == 5
