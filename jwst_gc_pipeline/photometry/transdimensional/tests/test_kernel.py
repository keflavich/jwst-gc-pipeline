"""C kernel (OpenBLAS, MKL, plain loops) against the numpy reference paths."""
import importlib.util

import numpy as np
import pytest

from .. import kernel as K
from .._vendor import image_eval as VE
from ..psf_poly import build_cf
from ._helpers import PCATDE_DIR, get_epsf, have_compiler, lib_dir

pytestmark = pytest.mark.skipif(not have_compiler(), reason='no C compiler')


def _lib(blas):
    try:
        return K.load_kernel(blas, lib_dir=lib_dir())
    except (RuntimeError, OSError) as err:
        pytest.skip(f'{blas} kernel unavailable: {err}')


@pytest.fixture(scope='module')
def psf():
    epsf, os_, _ = get_epsf()
    nc = 25
    rad = nc // 2
    c = epsf.shape[0] // 2
    psf0 = epsf[c - os_ * rad:c - os_ * rad + nc * os_, c - os_ * rad:c - os_ * rad + nc * os_]
    cf10 = VE.psf_poly_fit(psf0.astype(np.float32), os_)
    return dict(epsf=epsf, os=os_, nc=nc, psf0=psf0, cf10=cf10)


@pytest.mark.parametrize('blas', ['openblas', 'mkl', 'none'])
@pytest.mark.parametrize('shape', [(64, 64), (47, 71)])   # (ny, nx)
def test_c_matches_numpy_image_model_eval(psf, blas, shape):
    lib = _lib(blas)
    rng = np.random.default_rng(3)
    ny, nx = shape
    n = 60
    x = rng.uniform(1, nx - 2, n).astype(np.float32)
    y = rng.uniform(1, ny - 2, n).astype(np.float32)
    x[:5] = x[5:10]        # sources sharing a pixel exercise the kernel's hash merge
    y[:5] = y[5:10]
    f = rng.uniform(1, 100, n).astype(np.float32)
    imsz = (nx, ny)
    ref = rng.normal(0, 1, (ny, nx)).astype(np.float32)
    w = rng.uniform(0.5, 2, (ny, nx)).astype(np.float32)
    kw = dict(regsize=16, margin=4, offsetx=3, offsety=5, weights=w, ref=ref)
    img_np, d2_np = VE.image_model_eval(x, y, f, 0.5, imsz, psf['nc'], psf['cf10'], **kw)
    img_c, d2_c = VE.image_model_eval(x, y, f, 0.5, imsz, psf['nc'], psf['cf10'],
                                      lib=lib.pcat_model_eval, **kw)
    peak = np.abs(img_np).max()
    assert img_c.shape == img_np.shape == (ny, nx)
    assert np.max(np.abs(img_c - img_np)) < 2e-6 * peak
    np.testing.assert_allclose(d2_c, d2_np, rtol=2e-5)


@pytest.mark.parametrize('blas', ['openblas', 'mkl', 'none'])
def test_renderer_c_matches_numpy_quartic(psf, blas):
    lib = _lib(blas)
    cf, _ = build_cf(psf['epsf'], psf['os'], 25, basis='quartic15', fit='nodes')
    terms = K.basis_terms('quartic15')
    rc = K.PSFRenderer(cf, 25, terms, lib=lib)
    rn = K.PSFRenderer(cf, 25, terms, lib=None)
    rng = np.random.default_rng(4)
    x = rng.uniform(13, 70, 80)
    y = rng.uniform(13, 40, 80)
    x[:4] = x[4:8]
    y[:4] = y[4:8]
    f = rng.uniform(-5, 50, 80)
    a = np.zeros((55, 85), np.float32)
    b = np.zeros((55, 85), np.float32)
    rc.render(x, y, f, a)
    rn.render(x, y, f, b)
    assert np.max(np.abs(a - b)) < 2e-6 * np.abs(b).max()


def test_region_helpers_match_numpy():
    lib = _lib('openblas')
    rng = np.random.default_rng(5)
    npix, nlab = 5000, 7
    dm = rng.normal(0, 1, npix).astype(np.float32)
    dm[rng.uniform(size=npix) < 0.5] = 0
    res = rng.normal(0, 1, npix).astype(np.float32)
    w = rng.uniform(0, 2, npix).astype(np.float32)
    lab = rng.integers(-1, nlab, npix).astype(np.int32)
    out = np.zeros(nlab + 1)
    lib.pcat_dchi2_regions(npix, dm, res, w, lab, nlab, out)
    v = w.astype(float) * (dm.astype(float) ** 2 - 2 * res * dm.astype(float))
    ok = lab >= 0
    np.testing.assert_allclose(out[:nlab], np.bincount(lab[ok], v[ok], minlength=nlab), rtol=1e-10)
    assert out[nlab] == pytest.approx(np.sum(np.abs(dm[~ok]) * w[~ok]))
    acc = (rng.uniform(size=nlab) < 0.5).astype(np.int32)
    r1, m1 = res.copy(), np.zeros(npix, np.float32)
    lib.pcat_apply_accept(npix, dm, lab, acc, r1, m1)
    sel = ok.copy()
    sel[ok] = acc[lab[ok]] > 0
    np.testing.assert_array_equal(m1, np.where(sel, dm, 0))
    np.testing.assert_array_equal(r1, np.where(sel, res - dm, res))


def test_cf_nodes_cubic_matches_psf_poly_fit(psf):
    """build_cf(basis='cubic10', fit='nodes') reproduces PCAT's psf_poly_fit
    (which zero-pads the outermost node row/column; those pixels are skipped).

    psf_poly_fit solves the normal equations in float32 (np.linalg.inv of a
    float32 A^T A); build_cf uses a float64 pseudo-inverse.  The rendered
    stamps differ by < 5e-4 of the peak, the float32 solve's rounding level."""
    nc = psf['nc']
    cf, _ = build_cf(psf['epsf'], psf['os'], nc, basis='cubic10', fit='nodes')
    a = cf.reshape(10, nc, nc)[:, :-1, :-1]
    b = psf['cf10'].reshape(10, nc, nc)[:, :-1, :-1]
    peak = psf['epsf'].max()
    terms = K.basis_terms('cubic10')
    for dx, dy in [(0.0, 0.0), (0.5, 0.5), (0.9, 0.2)]:
        d = K.design_rows([dx], [dy], terms).astype(np.float64)
        sa = np.einsum('k,kab->ab', d[0], a.astype(np.float64))
        sb = np.einsum('k,kab->ab', d[0], b.astype(np.float64))
        assert np.max(np.abs(sa - sb)) < 5e-4 * peak


def _load_clone():
    path = PCATDE_DIR / 'image_eval.py'
    if not path.exists():
        pytest.skip('PCAT-DE clone not available')
    spec = importlib.util.spec_from_file_location('pcatde_image_eval', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vendored_matches_clone(psf):
    """Vendored psf_poly_fit and the numpy image_model_eval equal the clone's
    (square image, no template: the cases the vendor patches leave unchanged)."""
    ce = _load_clone()
    p0 = psf['psf0'].astype(np.float32)
    np.testing.assert_array_equal(VE.psf_poly_fit(p0, psf['os']), ce.psf_poly_fit(p0, psf['os']))
    rng = np.random.default_rng(6)
    x = rng.uniform(1, 50, 20).astype(np.float32)
    y = rng.uniform(1, 50, 20).astype(np.float32)
    f = rng.uniform(1, 10, 20).astype(np.float32)
    a = VE.image_model_eval(x, y, f, 0.0, (52, 52), psf['nc'], psf['cf10'])
    b = ce.image_model_eval(x, y, f, 0.0, (52, 52), psf['nc'], psf['cf10'])
    np.testing.assert_array_equal(a, b)
