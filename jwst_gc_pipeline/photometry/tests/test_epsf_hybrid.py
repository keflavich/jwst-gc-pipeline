"""Hybrid PSF (ePSF core + STPSF wing): conventions the pipeline relies on.

Synthetic grids only (no stpsf data, no network).  The STPSF stand-in is a
narrow Gaussian ePSF plus a faint wide wing on a 2x2 GriddedPSFModel; the core
is a broader, slightly off-centre Gaussian -- the sign of the real STPSF/ePSF
difference (STPSF too sharp, issue #1007).
"""
import sys

import numpy as np
import pytest
from astropy.nddata import NDData
from photutils.psf import GriddedPSFModel

from jwst_gc_pipeline.photometry import epsf_hybrid as eh

OS = 2          # STPSF grid oversampling, as in the pipeline cache (samp2)
FOV = 61        # native px (the cache uses 101; smaller keeps the test fast)
POS = [0.0, 2047.0]


def _gauss(ux, uy, s, x0=0.0, y0=0.0):
    return np.exp(-((ux - x0) ** 2 + (uy - y0) ** 2) / (2 * s * s)) / (2 * np.pi * s * s)


def _stpsf_like(ux, uy, scale=1.0):
    # core sigma 0.9 px + a 2% wing of sigma 8 px; unit total flux per native pixel area
    return scale * (0.98 * _gauss(ux, uy, 0.9) + 0.02 * _gauss(ux, uy, 8.0))


def _stpsf_grid():
    n = FOV * OS
    origin = (n - 1) / 2.0
    u = (np.arange(n) - origin) / OS
    UX, UY = np.meshgrid(u, u)
    data, xy = [], []
    for y in POS:
        for x in POS:
            # a slight position dependence so interpolation is exercised
            data.append(_stpsf_like(UX, UY, 1.0 + 0.01 * (x + y) / 2047))
            xy.append((x, y))
    return GriddedPSFModel(NDData(np.asarray(data), meta={'grid_xypos': xy, 'oversampling': OS,
                                                          'detector': ('NRCB5', 'det')}))


def _core(dx=0.04, dy=-0.03, sigma=1.1, phase_ripple=0.0):
    O, R = 4, 12
    k = (np.arange(2 * O * R + 1) - O * R) / O
    UX, UY = np.meshgrid(k, k)
    P = _gauss(UX, UY, sigma, dx, dy)
    P /= P[np.hypot(UX, UY) <= 10].sum() / O ** 2
    if phase_ripple:
        # the real cores' artifact: the native-pixel sum differs between the
        # O x O sub-pixel phases (by ~1-2%)
        # (+, +, -, -) over the O = 4 phases: visible on a 2x and a 4x grid alike
        sgn = np.where((np.arange(P.shape[0]) % O) < O // 2, 1.0, -1.0)
        P = P * (1 + phase_ripple * 0.5 * (sgn[:, None] + sgn[None, :]))
    return eh.EPSFCore(np.repeat(P[None], 4, 0), O, R, [512.0, 1536.0], [512.0, 1536.0],
                       meta={'DETECTOR': 'NRCB5', 'FILTER': 'F480M', 'PROGRAM': '10678'})


def _radii(grid):
    n = grid.data.shape[1]
    u = (np.arange(n) - grid.origin[0]) / OS
    UX, UY = np.meshgrid(u, u)
    return UX, UY, np.hypot(UX, UY)


def test_wing_is_stpsf_and_inner_flux_is_preserved():
    st = _stpsf_grid()
    hyb = eh.make_hybrid_grid(st, _core())
    UX, UY, r = _radii(hyb)
    for (x, y), H in zip(hyb.grid_xypos, hyb.data):
        D = eh._stpsf_data_at(st, x, y)
        np.testing.assert_allclose(H[r >= eh.R1], D[r >= eh.R1], rtol=0, atol=0)
        assert H[r <= eh.R0].sum() == pytest.approx(D[r <= eh.R0].sum(), rel=1e-10)
        # total normalisation (zero point / aperture corrections) unchanged
        assert H.sum() == pytest.approx(D.sum(), rel=2e-3)


def _phase_flux(g, r=10):
    k = np.arange(-30, 31)
    X, Y = np.meshgrid(k, k)
    ph = np.arange(4) / 4
    f = np.array([[g.evaluate(X + 1000, Y + 1000, 1.0, 1000 + px, 1000 + py)[np.hypot(X - px, Y - py) <= r].sum()
                   for px in ph] for py in ph])
    return np.ptp(f) / f.mean()


@pytest.mark.parametrize('os_', [2, 4])
def test_flux_inside_r0_does_not_depend_on_pixel_phase(os_, monkeypatch):
    """A core whose native-pixel sum varies with sub-pixel phase (PR #1009
    review: 1.3-1.6% at F212N on the samp4 grids) must not hand that to the
    hybrid: per-phase scaling keeps STPSF's phase dependence."""
    monkeypatch.setattr(sys.modules[__name__], 'OS', os_)
    st = _stpsf_grid()
    core = _core(phase_ripple=0.02)
    assert _phase_flux(st) < 2e-3
    hyb = eh.make_hybrid_grid(st, core)
    assert _phase_flux(hyb) < _phase_flux(st) + 1e-3
    # and the global-scale construction really did inherit the ripple
    monkeypatch.setattr(eh, '_phase_scale', lambda D, C, inner, oy, ox: D[inner].sum() / C[inner].sum())
    assert _phase_flux(eh.make_hybrid_grid(st, core)) > 5e-3


def test_core_shape_is_the_epsf_and_centroid_is_stpsf():
    st = _stpsf_grid()
    core = _core()
    hyb = eh.make_hybrid_grid(st, core)
    UX, UY, r = _radii(hyb)
    H = hyb.data[0]
    x, y = hyb.grid_xypos[0]
    cx_h, cy_h = eh._centroid(H, UX, UY, eh.R_CENTROID)
    cx_s, cy_s = eh._centroid(eh._stpsf_data_at(st, x, y), UX, UY, eh.R_CENTROID)
    assert abs(cx_h - cx_s) < 1e-4 and abs(cy_h - cy_s) < 1e-4
    # the core was moved by (about) its own offset from STPSF's centre
    assert hyb.meta['epsf_centroid_shift_max_px'] == pytest.approx(np.hypot(0.04, 0.03), abs=5e-3)
    # inside r0 the hybrid is the (broader) ePSF, not STPSF
    inner = r <= 3
    C = core.sample(x, y, UX + 0.04, UY - 0.03)       # the core, re-centred on STPSF's (0, 0)
    ratio = H[inner] / C[inner]
    assert np.std(ratio) / np.mean(ratio) < 1e-3
    assert H.max() < eh._stpsf_data_at(st, x, y).max()


def test_photutils_evaluation_of_the_hybrid():
    hyb = eh.make_hybrid_grid(_stpsf_grid(), _core())
    yy, xx = np.mgrid[0:61, 0:61]
    img = hyb.evaluate(xx + 970, yy + 970, 1.0, 1000.3, 1000.6)
    # unit flux -> image sums to the in-stamp flux of the STPSF stand-in (~1)
    assert img.sum() == pytest.approx(1.0, abs=0.02)


def test_noop_unless_configured(tmp_path):
    st = _stpsf_grid()
    assert eh.maybe_apply_epsf_core(st, 'NRCB5', 'F480M', environ={}) is st
    env = {eh.EPSF_CORE_DIR_ENV: str(tmp_path)}
    assert eh.maybe_apply_epsf_core(st, 'NRCB5', 'F480M', environ=env) is st   # no file


def test_configured_roundtrip_and_detector_from_meta(tmp_path):
    st = _stpsf_grid()
    core = _core()
    eh.write_epsf_core(tmp_path / eh.epsf_core_filename('NRCB5', 'F480M'), core, 'NRCB5', 'F480M')
    env = {eh.EPSF_CORE_DIR_ENV: str(tmp_path)}
    hyb = eh.maybe_apply_epsf_core(st, None, 'F480M', environ=env)    # detector from grid meta
    assert isinstance(hyb, GriddedPSFModel) and hyb is not st
    assert hyb.meta['epsf_core'].endswith('epsf_core_nrcb5_f480m.fits')
    # list (merged-module) path: each element by its own metadata
    out = eh.maybe_apply_epsf_core([st, st], None, 'F480M', environ=env)
    assert all(isinstance(g, GriddedPSFModel) and g is not st for g in out)
    # wrong filter file name -> plain STPSF
    assert eh.maybe_apply_epsf_core(st, 'NRCB5', 'F410M', environ=env) is st


def test_mislabelled_core_file_raises(tmp_path):
    eh.write_epsf_core(tmp_path / eh.epsf_core_filename('NRCB5', 'F480M'), _core(), 'NRCA5', 'F480M')
    with pytest.raises(ValueError, match='NRCA5'):
        eh.maybe_apply_epsf_core(_stpsf_grid(), 'NRCB5', 'F480M',
                                 environ={eh.EPSF_CORE_DIR_ENV: str(tmp_path)})


def test_filename_token_and_catalog_provenance(tmp_path):
    assert eh.hybrid_psf_token({}) == ''
    env = {eh.EPSF_CORE_DIR_ENV: str(tmp_path)}
    assert eh.hybrid_psf_token(env) == '_hybpsf'
    # residual globs match flags by substring: the token must not read as '_epsf'
    assert '_epsf' not in eh.hybrid_psf_token(env)
    eh._APPLIED.clear()
    assert eh.psf_provenance_meta('F480M', environ={}) == {'PSFMODEL': 'STPSF'}
    fn = tmp_path / eh.epsf_core_filename('NRCB5', 'F480M')
    eh.write_epsf_core(fn, _core(), 'NRCB5', 'F480M')
    hyb = eh.maybe_apply_epsf_core(_stpsf_grid(), 'NRCB5', 'F480M', environ=env, program='10678')
    meta = eh.psf_provenance_meta('F480M', environ=env)
    assert meta['PSFMODEL'] == 'STPSF+EPSFCORE'
    assert meta['EPSFCORE'] == fn.name and meta['EPSFSHA'] == eh._sha256(fn)
    assert meta['EPSFPROG'] == '10678' and (meta['EPSFR0'], meta['EPSFR1']) == (eh.R0, eh.R1)
    assert meta['EPSFSHFT'] == pytest.approx(1e3 * hyb.meta['epsf_centroid_shift_max_px'])
    assert all(len(k) <= 8 for k in meta)
    # a filter with no core file: the token is on (configuration), PSFMODEL says STPSF
    eh.maybe_apply_epsf_core(_stpsf_grid(), 'NRCB5', 'F410M', environ=env)
    assert eh.psf_provenance_meta('F410M', environ=env)['PSFMODEL'] == 'STPSF'
    eh._APPLIED.clear()


def test_other_program_is_refused_unless_allowed(tmp_path):
    eh.write_epsf_core(tmp_path / eh.epsf_core_filename('NRCB5', 'F480M'), _core(), 'NRCB5', 'F480M')
    env = {eh.EPSF_CORE_DIR_ENV: str(tmp_path)}
    with pytest.raises(ValueError, match='program 10678, not 2221'):
        eh.maybe_apply_epsf_core(_stpsf_grid(), 'NRCB5', 'F480M', environ=env, program='2221')
    with pytest.warns(UserWarning, match='using it anyway'):
        hyb = eh.maybe_apply_epsf_core(_stpsf_grid(), 'NRCB5', 'F480M', program='2221',
                                       environ={**env, eh.ALLOW_OTHER_PROGRAM_ENV: '1'})
    assert isinstance(hyb, GriddedPSFModel)
    # leading zeros do not matter
    eh.maybe_apply_epsf_core(_stpsf_grid(), 'NRCB5', 'F480M', environ=env, program='010678')


def test_hybrid_is_cached_per_core_and_grid(tmp_path, monkeypatch):
    eh.write_epsf_core(tmp_path / eh.epsf_core_filename('NRCB5', 'F480M'), _core(), 'NRCB5', 'F480M')
    env = {eh.EPSF_CORE_DIR_ENV: str(tmp_path)}
    eh._HYBRID_CACHE.clear()
    calls = []
    real = eh.make_hybrid_grid
    monkeypatch.setattr(eh, 'make_hybrid_grid', lambda g, c: calls.append(1) or real(g, c))
    a = eh.maybe_apply_epsf_core(_stpsf_grid(), 'NRCB5', 'F480M', environ=env)
    b = eh.maybe_apply_epsf_core(_stpsf_grid(), 'NRCB5', 'F480M', environ=env)   # a fresh, equal grid
    assert a is b and len(calls) == 1
    other = _stpsf_grid()
    other.data[0] *= 1.01
    assert eh.maybe_apply_epsf_core(other, 'NRCB5', 'F480M', environ=env) is not a and len(calls) == 2
    eh._HYBRID_CACHE.clear()


def test_core_shape_validation():
    with pytest.raises(ValueError, match='does not match'):
        eh.EPSFCore(np.zeros((3, 97, 97)), 4, 12, [1.0, 2.0], [1.0, 2.0])
    with pytest.raises(ValueError, match='radius'):
        eh.EPSFCore(np.zeros((1, 81, 81)), 4, 10, [1.0], [1.0])


def test_get_psf_model_wiring(tmp_path, monkeypatch):
    """get_psf_model hands its STPSF grid (and the cache detector) to
    maybe_apply_epsf_core and returns what that gives back -- and with
    PSF_EPSF_CORE_DIR unset the grid is the cached object, untouched."""
    ccl = pytest.importorskip('jwst_gc_pipeline.photometry.crowdsource_catalogs_long')
    st = _stpsf_grid()
    fn = tmp_path / 'nircam_nrcb5_f480m_fovp101_samp2_npsf16.fits'
    fn.write_text('fake')
    monkeypatch.setattr(ccl, 'to_griddedpsfmodel', lambda f: st)
    monkeypatch.delenv(eh.EPSF_CORE_DIR_ENV, raising=False)
    kw = dict(module='nrcb', use_webbpsf=True, use_grid=True, instrument='NIRCam',
              psf_cache_dir=str(tmp_path))
    grid, _ = ccl.get_psf_model('F480M', '10678', '081', **kw)
    assert grid is st

    eh.write_epsf_core(tmp_path / eh.epsf_core_filename('NRCB5', 'F480M'), _core(), 'NRCB5', 'F480M')
    monkeypatch.setenv(eh.EPSF_CORE_DIR_ENV, str(tmp_path))
    grid, _ = ccl.get_psf_model('F480M', '10678', '081', **kw)
    assert grid is not st and grid.meta['epsf_core'].endswith('epsf_core_nrcb5_f480m.fits')
