"""End-to-end recovery on a sparse mock: 10 stars, 60x60 px, S/N 20-200.

The mock is drawn with the ePSF spline over the full ePSF field of view
(:mod:`..mocks`), independent of the sampler's polynomial renderer.  The
sampler starts from an empty catalog.
"""
import numpy as np
import pytest

from .. import mocks
from ..condense import condense
from ..pcat_sampler import PCATConfig, run_pcat
from ._helpers import get_epsf, have_compiler, lib_dir


@pytest.mark.skipif(not have_compiler(), reason='no C compiler')
def test_sparse_mock_recovery():
    epsf, os_, _ = get_epsf()
    rng = np.random.default_rng(7)
    fac = mocks.psf_snr_factor(epsf, os_)
    snr = np.exp(rng.uniform(np.log(20), np.log(200), 10))
    f = snr / fac
    pts = []
    while len(pts) < 10:
        p = rng.uniform(3, 56, 2)
        if all(np.hypot(*(p - q)) > 6 for q in pts):
            pts.append(p)
    pts = np.array(pts)
    tr = mocks.TruthRenderer(epsf, os_)
    img, w = mocks.make_mock((60, 60), pts[:, 0], pts[:, 1], f, tr, 1.0, bkg=0.5, rng=rng)
    cfg = PCATConfig(trueminf=5 / fac, nsamp=40, nloop=250, regsize=28, seed=3,
                     lib_dir=str(lib_dir()), verbose=0)
    res = run_pcat(img, w, epsf, os_, cfg, nc=25)
    pb = res.post_burnin()
    c = condense(res.samples['x'], res.samples['y'], res.samples['f'], sample_idx=pb)
    sel = c['prob'] > 0.5
    ic, it, d = mocks.match_truth(c['x'][sel], c['y'][sel], pts[:, 0], pts[:, 1], 0.5)
    assert ic.size == 10                               # every star recovered
    assert sel.sum() - ic.size <= 1                    # at most one spurious source
    fl = c['flux'][0][sel][ic]
    fe = c['flux_err'][0][sel][ic]
    pull = (fl - f[it]) / fe
    assert np.all(np.abs(pull) < 4)
    assert np.all(np.abs(fl / f[it] - 1) < 0.1)
    assert np.all(d < 0.5)
    # the chain's N settles on the truth
    assert abs(np.median(res.samples['n'][pb]) - 10) <= 1
    assert res.info['leak'] == 0.0
