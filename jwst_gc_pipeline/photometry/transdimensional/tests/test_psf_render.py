"""Phase convention and accuracy of the polynomial PSF renderer.

The renderer (quartic15 basis fitted to the ePSF nodes, nbin = oversampling)
must reproduce photutils ``GriddedPSFModel`` built from the same ePSF to
1e-3 of the peak inside the stamp, conserve the flux in the stamp to 0.1 %,
and place the source where GriddedPSFModel places it to 0.005 px, at
integer, half-integer, mixed and random 0-based positions.
"""
import numpy as np
import pytest
from scipy.optimize import least_squares

from .. import kernel as K
from ..psf_poly import build_cf, encircled_energy
from ._helpers import EPSF_PATH, get_epsf, have_compiler, lib_dir

pytestmark = pytest.mark.localdata

NC = 31
BASIS = 'quartic15'


@pytest.fixture(scope='module')
def setup():
    epsf, os_, _ = get_epsf(require_real=True)
    from ..psf import to_gridded_psf_model
    gpm = to_gridded_psf_model(str(EPSF_PATH))
    cf, info = build_cf(epsf, os_, NC, basis=BASIS, fit='nodes')
    lib = K.load_kernel('auto', lib_dir=lib_dir()) if have_compiler() else None
    rend = K.PSFRenderer(cf, NC, K.basis_terms(BASIS), lib=lib)
    peak = float(epsf.max())
    return dict(epsf=epsf, os=os_, gpm=gpm, rend=rend, info=info, peak=peak)


def _positions():
    rng = np.random.default_rng(20261028)
    base = [(40.0, 40.0), (40.5, 40.5), (40.0, 40.5), (40.5, 40.0), (40.25, 39.75),
            (39.999, 40.001)]
    rnd = [tuple(v) for v in 39.0 + 2.0 * rng.uniform(size=(20, 2))]
    return base + rnd


def _gpm_eval(gpm, x0, y0, flux, yy, xx):
    m = gpm.copy()
    m.x_0 = x0
    m.y_0 = y0
    m.flux = flux
    return m(xx, yy)


@pytest.mark.parametrize('pos', _positions())
def test_render_matches_gridded_psf_model(setup, pos):
    x, y = pos
    img = np.zeros((81, 81), np.float32)
    setup['rend'].render([x], [y], [1.0], img)
    r = NC // 2
    ix, iy = int(np.ceil(x)), int(np.ceil(y))
    yy, xx = np.mgrid[iy - r:iy + r + 1, ix - r:ix + r + 1]
    stamp = img[iy - r:iy + r + 1, ix - r:ix + r + 1].astype(np.float64)
    # nothing outside the stamp footprint
    assert np.count_nonzero(img) == np.count_nonzero(stamp)
    ref = _gpm_eval(setup['gpm'], x, y, 1.0, yy, xx)
    peak = setup['peak']
    assert np.max(np.abs(stamp - ref)) < 1e-3 * peak
    assert abs(stamp.sum() / ref.sum() - 1) < 1e-3

    # position and flux of a GriddedPSFModel fit to the rendered stamp
    def resid(p):
        return (_gpm_eval(setup['gpm'], p[0], p[1], p[2], yy, xx) - stamp).ravel()
    sol = least_squares(resid, [x + 0.03, y - 0.03, 0.98], x_scale=[0.1, 0.1, 0.1], xtol=1e-12,
                        ftol=1e-14, gtol=1e-14)
    assert abs(sol.x[0] - x) <= 0.005
    assert abs(sol.x[1] - y) <= 0.005
    assert abs(sol.x[2] - 1.0) < 1e-3


def test_stamp_is_not_renormalised(setup):
    """A flux-1 source centred on a pixel deposits EE(nc) (the ePSF node sum)."""
    img = np.zeros((81, 81), np.float32)
    setup['rend'].render([40.0], [40.0], [1.0], img)
    ee = encircled_energy(setup['epsf'], setup['os'], NC)
    assert abs(img.sum() - ee) < 1e-5
    assert setup['info']['ee_nc'] == pytest.approx(ee)
    assert 0.85 < ee < 0.97


def test_wrong_phase_fails(setup):
    """Negative control: evaluating the polynomial at 1 - dx (the other phase
    convention) misplaces a half-integer source by far more than the tolerance."""
    x, y = 40.3, 40.3
    r = NC // 2
    ix = iy = 41
    dd = K.design_rows([1 - (ix - x)], [1 - (iy - y)], setup['rend'].terms)
    stamp = (dd @ setup['rend'].cf).reshape(NC, NC).astype(np.float64)
    yy, xx = np.mgrid[iy - r:iy + r + 1, ix - r:ix + r + 1]
    ref = _gpm_eval(setup['gpm'], x, y, 1.0, yy, xx)
    assert np.max(np.abs(stamp - ref)) > 2e-2 * setup['peak']
