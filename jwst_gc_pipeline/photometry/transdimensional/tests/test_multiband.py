"""Two-band run with an affine pixel map: the second band's grid is shifted by
(0.37, -0.21) px; four bright stars with known flux ratios; the sampler starts
near the truth.  Checks the band-1 fluxes and the positions."""
import numpy as np
import pytest

from ._helpers import get_epsf, have_compiler, lib_dir

pytestmark = pytest.mark.skipif(not have_compiler(), reason='no C compiler')


def test_two_band_affine():
    from .. import mocks
    from ..condense import condense
    from ..pcat_sampler import BandData, PCATConfig, run_pcat
    from ..psf_poly import build_cf
    epsf, os_, _ = get_epsf()
    rng = np.random.default_rng(5)
    tr = mocks.TruthRenderer(epsf, os_)
    shape = (44, 50)
    x = np.array([12.3, 30.7, 21.1, 38.4])
    y = np.array([10.2, 14.9, 30.5, 33.3])
    f0 = np.array([400., 250., 600., 150.])
    f1 = f0 * np.array([0.5, 0.6, 0.4, 0.5])
    img0, w0 = mocks.make_mock(shape, x, y, f0, tr, 1.0, 0.2, rng)
    dx, dy = 0.37, -0.21
    img1, w1 = mocks.make_mock(shape, x + dx, y + dy, f1, tr, 1.0, 0.1, rng)
    nc = 25
    cf, _ = build_cf(epsf, os_, nc)
    b1 = BandData(image=img1, weight=w1, cf=cf, nc=nc,
                  affine=np.array([[1, 0, dx], [0, 1, dy]]), name='b1')
    fac = mocks.psf_snr_factor(epsf, os_)
    cfg = PCATConfig(trueminf=5.0 / fac, nsamp=40, nloop=100, regsize=28, seed=3,
                     color_mus=[0.75], color_sigs=[0.5], lib_dir=str(lib_dir()), verbose=0,
                     init_catalog=dict(x=x + 0.2, y=y - 0.2, f=np.vstack([f0, f1]) * 0.9))
    res = run_pcat(img0, w0, epsf, os_, cfg, nc=nc, extra_bands=[b1])
    pb = res.post_burnin()
    c = condense(res.samples['x'], res.samples['y'], res.samples['f'], sample_idx=pb)
    sel = c['prob'] > 0.5
    ic, it, d = mocks.match_truth(c['x'][sel], c['y'][sel], x, y, 1.0)
    assert ic.size == 4 and sel.sum() == 4
    assert np.all(d < 0.2)
    r1 = c['flux'][1][sel][ic] / f1[it]
    assert np.all(np.abs(r1 - 1) < 0.06), r1
    assert res.info['leak'] == 0
