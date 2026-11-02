"""Tests for the 2026-10-29 adapter changes: tuned step defaults, noise_scale,
fixed wing model, bright-star merge, per-sample N(>f), prevalence calibration
and catalog/image writer options."""
import json

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from .. import mocks, pcat_io
from ..condense import condense, merge_bright, n_above_samples
from ..kernel import PSFRenderer, basis_terms
from ..pcat_sampler import PCATConfig, PCATSampler, BandData, run_pcat
from ..psf_poly import build_cf, _spline
from ..wings import wing_image, select_wing_sources
from ._helpers import get_epsf, have_compiler, lib_dir


# --------------------------------------------------------------------- defaults
def test_tuned_defaults():
    cfg = PCATConfig()
    assert cfg.pos_step_fac == 5.0
    assert cfg.flux_step_fac == 5.0
    assert cfg.move_weights == dict(movestar=60., birth_death=60., merge_split=30., bkg=5.)
    assert cfg.noise_scale == 1.0


def _crowded_mock(seed=11, n=60, shape=(80, 80)):
    epsf, os_, _ = get_epsf()
    fac = mocks.psf_snr_factor(epsf, os_)
    rng = np.random.default_rng(seed)
    f = mocks.power_law_fluxes(n, 5 / fac, 2.0, rng, fmax=600 / fac)
    x = rng.uniform(-0.5, shape[1] - 0.5, n)
    y = rng.uniform(-0.5, shape[0] - 0.5, n)
    tr = mocks.TruthRenderer(epsf, os_)
    img, w = mocks.make_mock(shape, x, y, f, tr, 1.0, bkg=0.0, rng=rng)
    return epsf, os_, fac, img, w, x, y, f


@pytest.mark.skipif(not have_compiler(), reason='no C compiler')
def test_movestar_acceptance_crowded():
    """Regression: with the default step factors the movestar acceptance on a
    crowded mock (P-cutout density, S/N 5-600) stays in 0.2-0.6 (0.40 measured;
    unit steps give 0.82, see runs/pcat/improve/defaults/step_acceptance.json)."""
    epsf, os_, fac, img, w, x, y, f = _crowded_mock()
    cfg = PCATConfig(trueminf=3 / fac, nsamp=15, nloop=200, regsize=20, seed=1,
                     init_catalog=dict(x=x, y=y, f=f), bkg=0.0, store_models=False,
                     lib_dir=str(lib_dir()), verbose=0)
    res = run_pcat(img, w, epsf, os_, cfg, nc=15)
    acc = res.accept['movestar']['frac']
    assert 0.2 <= acc <= 0.6, acc


# ------------------------------------------------------------------ noise scale
def _band(shape=(40, 40), sig=2.0):
    epsf, os_, _ = get_epsf()
    cf, _ = build_cf(epsf, os_, 11)
    rng = np.random.default_rng(1)
    img = rng.normal(0, sig, shape)
    w = np.full(shape, 1 / sig ** 2)
    return BandData(image=img, weight=w, cf=cf, nc=11, name='pivot')


@pytest.mark.parametrize('scale', [1.0, 0.8, 1.25])
def test_noise_scale_weights(scale):
    bd = _band()
    base = PCATSampler([bd], PCATConfig(regsize=20, blas='numpy', verbose=0))
    s = PCATSampler([bd], PCATConfig(regsize=20, blas='numpy', verbose=0, noise_scale=scale))
    sl = s.unpad[0]
    assert np.allclose(s.wts[0][sl], base.wts[0][sl] / scale ** 2, rtol=1e-6)
    assert s.errf[0] == pytest.approx(base.errf[0] * scale, rel=1e-6)
    assert s.loglike()[0] == pytest.approx(base.loglike()[0] / scale ** 2, rel=1e-5)


def test_noise_scale_rejects_nonpositive():
    with pytest.raises(ValueError):
        PCATSampler([_band()], PCATConfig(regsize=20, blas='numpy', verbose=0, noise_scale=0.0))


# ------------------------------------------------------------------ fixed model
def test_fixed_model_is_subtracted_and_restored():
    """The sampler sees image - fixed; returned models include fixed; the
    residual equals image - returned model."""
    epsf, os_, _ = get_epsf()
    rng = np.random.default_rng(3)
    shape = (40, 40)
    fixed = np.outer(np.linspace(0, 3, shape[0]), np.ones(shape[1]))
    img = rng.normal(0, 1, shape) + fixed
    w = np.ones(shape)
    cfg = PCATConfig(trueminf=1e6, nsamp=6, nloop=5, regsize=20, blas='numpy', verbose=0,
                     move_weights=dict(movestar=0., birth_death=0., merge_split=0., bkg=1.),
                     store_models=True)
    res = run_pcat(img, w, epsf, os_, cfg, nc=11, fixed_model=fixed)
    m = res.model_mean[0]
    # no sources: the sampled model is the constant background, close to 0 after removing fixed
    assert abs(np.mean(m - fixed)) < 0.1
    assert np.allclose(res.residual_mean[0], img - m, atol=1e-5)
    assert np.allclose(res.residual_median[0], img - res.model_median[0], atol=1e-5)
    assert res.info['fixed_model']['sum'] == pytest.approx(fixed.sum())
    with pytest.raises(ValueError):
        run_pcat(img, w, epsf, os_, cfg, nc=11, fixed_model=fixed[:-1])


# ------------------------------------------------------------------ wing image
def test_wing_image_complements_stamp():
    """stamp (sampler renderer) + wings = full ePSF spline render, and the wing
    image is zero on the stamp footprint."""
    epsf, os_, _ = get_epsf()
    nc = 15
    shape = (121, 121)
    x, y, f = np.array([60.3]), np.array([59.8]), np.array([1000.0])
    wing = wing_image(x, y, f, epsf, os_, nc, shape)
    cx, cy, r = int(np.ceil(x[0])), int(np.ceil(y[0])), nc // 2
    assert np.all(wing[cy - r:cy + r + 1, cx - r:cx + r + 1] == 0)
    assert wing[cy, cx + r + 1] > 0
    full = mocks.TruthRenderer(epsf, os_).render(x, y, f, shape)
    cf, _ = build_cf(epsf, os_, nc)
    stamp = np.zeros(shape, np.float32)
    PSFRenderer(cf, nc, basis_terms('quartic15'), lib=None).render(x, y, f, stamp)
    tot = stamp + wing
    # the renderer's centring is ceil(x); TruthRenderer uses round(x): compare where both cover
    h = mocks.TruthRenderer(epsf, os_).half
    sl = (slice(cy - h + 1, cy + h), slice(cx - h + 1, cx + h))
    assert np.max(np.abs(tot[sl] - full[sl])) < 2e-3 * full.max()
    # wing flux = f * (ePSF sum - EE(nc)) within the rendered window
    assert wing.sum() == pytest.approx(full.sum() - stamp.sum(), rel=2e-3)


def test_select_wing_sources():
    cond = dict(flux=np.array([[10., 500., 2000., 3000.]]), prob=np.array([1., 1., 0.3, 0.9]))
    assert list(select_wing_sources(cond, 400.)) == [1, 3]
    assert list(select_wing_sources(cond, 400., prob_min=0.2)) == [1, 2, 3]


# ------------------------------------------------------------------ bright merge
def _bright_chain(nsamp=300, seed=0):
    """A bright star held as two anti-correlated components (total 20000) at
    the same position, a 1300-count core component 0.7 px away that trades
    flux with it, an independent 300-count neighbour 3.2 px away, and an
    independent 2000-count star 1.2 px away."""
    rng = np.random.default_rng(seed)
    nmax = 6
    x = np.full((nsamp, nmax), np.nan)
    y = np.full((nsamp, nmax), np.nan)
    f = np.full((nsamp, 1, nmax), np.nan)
    for s in range(nsamp):
        tot = 20000 + rng.normal(0, 60)
        a = rng.uniform(0.4, 0.7)
        core = 1300 + rng.normal(0, 300)
        rows = [(50.0, 50.0, a * (tot - core)), (50.1, 50.0, (1 - a) * (tot - core)),
                (50.7, 50.0, core), (50.0, 53.2, 300 + rng.normal(0, 30)),
                (48.8, 50.0, 2000 + rng.normal(0, 40))]
        for j, (xx, yy, ff) in enumerate(rows):
            x[s, j] = xx + rng.normal(0, 0.02)
            y[s, j] = yy + rng.normal(0, 0.02)
            f[s, 0, j] = ff
    return x, y, f


def test_merge_bright_merges_split_components_only():
    x, y, f = _bright_chain()
    c = condense(x, y, f, radius=0.3, return_samples=True)
    assert c['samp_x'].shape == (c['x'].size, 300)
    assert c['x'].size == 5
    m = merge_bright(c, f_bright=5000., r_merge=1.5, var_frac=0.8)
    assert m['x'].size == 3                      # split + core merged into one
    k = int(np.argmax(m['flux'][0]))
    assert m['flux'][0, k] == pytest.approx(20000, rel=0.01)
    assert m['flux_err'][0, k] < 100             # the sum is stable
    assert m['nmerge'][k] == 2
    assert sorted(m['nmerge']) == [0, 0, 2]
    # the independent neighbours survive with their own fluxes
    fl = np.sort(m['flux'][0])
    assert fl[0] == pytest.approx(300, rel=0.05)
    assert fl[1] == pytest.approx(2000, rel=0.02)
    assert m['prob'][k] == 1.0
    assert len(m['merge_log']) == 2
    # without samples the merge refuses
    c2 = condense(x, y, f, radius=0.3)
    with pytest.raises(ValueError):
        merge_bright(c2, 5000.)


def test_n_above_samples_merged_is_stable():
    x, y, f = _bright_chain()
    c = condense(x, y, f, radius=0.3, return_samples=True)
    m = merge_bright(c, f_bright=5000.)
    raw = n_above_samples(c, [1000., 5000.])
    mer = n_above_samples(m, [1000., 5000.])
    # raw: two split components (always > 1000) + core (~1300, flickers) + 2000 star
    assert raw[1000.].std() > 0.2
    assert np.all(mer[1000.] == 2)
    assert np.all(mer[5000.] == 1)
    reg = n_above_samples(m, [1000.], region=(49.5, 60, 40, 60))
    assert np.all(reg[1000.] == 1)


# ------------------------------------------------------------ prevalence calib.
def test_apply_prob_cal(tmp_path):
    cal = dict(knots=dict(prob=[0.0, 0.2, 0.5, 1.0], p_real=[0.1, 0.4, 0.8, 0.95]))
    p = np.array([0.0, 0.1, 0.35, 0.75, 1.0])
    out = pcat_io.apply_prob_cal(p, cal)
    assert np.allclose(out, [0.1, 0.25, 0.6, 0.875, 0.95])
    assert np.all(np.isnan(pcat_io.apply_prob_cal(p, None)))
    path = tmp_path / 'cal.json'
    path.write_text(json.dumps(cal))
    assert pcat_io.load_prob_cal(path)['path'] == str(path)
    with pytest.raises(ValueError):
        pcat_io.apply_prob_cal(p, dict(knots=dict(prob=[0, 1], p_real=[0.9, 0.1])))
    path.write_text(json.dumps(dict(foo=1)))
    with pytest.raises(ValueError):
        pcat_io.load_prob_cal(path)


def _cond(n=4):
    return dict(x=np.array([0., 50., 100., 170.])[:n], y=np.array([40., 60., 80., 120.])[:n],
                x_err=np.full(n, 0.01), y_err=np.full(n, 0.01), x_med=np.zeros(n), y_med=np.zeros(n),
                flux=np.full((1, n), 100.), flux_err=np.full((1, n), 5.), flux_mean=np.full((1, n), 100.),
                flux_p16=np.full((1, n), 95.), flux_p84=np.full((1, n), 105.),
                prob=np.array([0.1, 0.4, 0.9, 1.0])[:n], nmatch=np.ones(n, int), degree=np.ones(n, int),
                nsamp_used=10, nmerge=np.zeros(n, int))


def _wcs(crpix):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crval = [266.54, -28.71]
    w.wcs.crpix = crpix
    w.wcs.cdelt = [-0.031 / 3600, 0.031 / 3600]
    return w


def test_write_catalog_prob_cal_and_box_frame(tmp_path):
    cal = dict(knots=dict(prob=[0.0, 1.0], p_real=[0.2, 1.0]), path='/x/cal.json')
    # frame shifted by (+100, -700) px from P: x_frame = x_P + 100, y_frame = y_P - 700
    wf, wp = _wcs([1101., 301.]), _wcs([1001., 1001.])
    cond = _cond()
    # cutout origin chosen so the second and third sources land in the P box
    t = pcat_io.write_catalog(tmp_path / 'c.fits', cond, x0=100 + 60, y0=-700 + 740, wcs=wf,
                              box_wcs=wp, prob_cal=cal)
    assert np.allclose(t['x_P'], cond['x'] + 60, atol=1e-5)
    assert np.allclose(t['y_P'], cond['y'] + 740, atol=1e-5)
    assert list(t['in_box']) == list(pcat_io.in_box(cond['x'] + 60, cond['y'] + 740))
    assert list(t['in_box']) == [False, True, True, False]
    assert np.allclose(t['prob'], cond['prob'])
    assert np.allclose(t['prob_cal'], 0.2 + 0.8 * cond['prob'])
    r = Table.read(tmp_path / 'c.fits')
    assert r.meta['PROBCAL'] == 'cal.json'
    assert 'nmerge' in r.colnames
    # without box_wcs the frame pixels are taken as P pixels
    t2 = pcat_io.write_catalog(tmp_path / 'd.fits', cond, x0=60, y0=740)
    assert list(t2['in_box']) == [False, True, True, False]
    assert np.all(np.isnan(t2['prob_cal']))


def test_write_images_mean_primary(tmp_path):
    class R:
        model_mean = [np.ones((4, 4))]
        model_median = [np.full((4, 4), 2.0)]
        residual_mean = [np.full((4, 4), 3.0)]
        residual_median = [np.full((4, 4), 4.0)]
    pcat_io.write_images(tmp_path, R(), primary='mean')
    with fits.open(tmp_path / 'model.fits') as h:
        assert h[0].header['MODEL'] == 'posterior mean'
        assert np.all(h[0].data == 1.0) and np.all(h['MEDIAN'].data == 2.0)
    with fits.open(tmp_path / 'residual.fits') as h:
        assert np.all(h[0].data == 3.0) and np.all(h['MEDIAN'].data == 4.0)
    pcat_io.write_images(tmp_path, R())
    with fits.open(tmp_path / 'model.fits') as h:
        assert h[0].header['MODEL'] == 'posterior median'
        assert np.all(h[0].data == 2.0) and np.all(h['MEAN'].data == 1.0)
    with pytest.raises(ValueError):
        pcat_io.write_images(tmp_path, R(), primary='mode')


# ------------------------------------------------------------------ ring flags
def _ring_psf(oversample=5, size=101, radii=(3.0, 5.0)):
    """Oversampled array with a Gaussian core and two Gaussian rings at ``radii``
    (detector px), centred at the array centre."""
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(xx - (size - 1) / 2.0, yy - (size - 1) / 2.0) / oversample
    img = np.exp(-0.5 * (r / 0.8) ** 2)
    for rr in radii:
        img += 0.02 * np.exp(-0.5 * ((r - rr) / 0.25) ** 2)
    return img / img.sum()


def test_psf_ring_radii_synthetic():
    from ..rings import psf_ring_radii
    rr = psf_ring_radii(_ring_psf(), 5)
    assert rr.size == 2
    assert np.allclose(rr, [3.0, 5.0], atol=0.1)
    rr2 = psf_ring_radii(_ring_psf(radii=(4.0,)), 5, rmin=1.5, rmax=8.0)
    assert np.allclose(rr2, [4.0], atol=0.1)


def test_psf_ring_radii_real_epsf():
    """The F182M e16 ePSF azimuthal mean (0.1 px annuli) has its first ring
    maxima at 3.05 and 3.25 px and further local maxima at 5.4-6.1 and 7.0-7.6 px
    (the radii quoted in rings.py)."""
    from ..rings import psf_ring_radii
    epsf, os_, real = get_epsf(require_real=True)
    rr = psf_ring_radii(epsf, os_)
    assert np.any((rr > 2.95) & (rr < 3.35))
    assert np.any((rr > 5.3) & (rr < 6.2))
    assert rr.min() > 2.9


def test_flag_psf_rings():
    from ..rings import flag_psf_rings
    rr = np.array([3.2, 5.0])
    # parent (1e4) at the origin; candidates: on ring 0 (faint: flagged), on ring 0 but
    # 5 % of the parent (kept), between rings (kept), on ring 1 (flagged), and a second
    # parent below f_bright with a faint source on its ring (kept)
    x = np.array([0.0, 3.2, 0.0, 4.1, -5.0, 40.0, 43.2])
    y = np.array([0.0, 0.0, 3.25, 0.0, 0.0, 0.0, 0.0])
    f = np.array([1e4, 100.0, 500.0, 100.0, 50.0, 900.0, 10.0])
    flag, parent, ring = flag_psf_rings(x, y, f, rr, f_bright=1000.0, halfwidth=0.3, max_ratio=0.03)
    assert list(flag) == [False, True, False, False, True, False, False]
    assert list(parent[flag]) == [0, 0]
    assert list(ring[flag]) == [0, 1]
    assert np.all(parent[~flag] == -1) and np.all(ring[~flag] == -1)
    # empty inputs
    f0, p0, r0 = flag_psf_rings(np.zeros(0), np.zeros(0), np.zeros(0), rr)
    assert f0.size == 0 and p0.size == 0 and r0.size == 0
    f1, _, _ = flag_psf_rings(x, y, f, np.zeros(0))
    assert not f1.any()


def test_write_catalog_ring_columns(tmp_path):
    cond = _cond()
    t0 = pcat_io.write_catalog(tmp_path / 'a.fits', cond)
    assert 'ring_flag' not in t0.colnames and 'ring_parent' not in t0.colnames
    cond = dict(cond, ring_flag=np.array([False, True, False, True]), ring_parent=np.array([-1, 2, -1, 2]))
    pcat_io.write_catalog(tmp_path / 'b.fits', cond)
    r = Table.read(tmp_path / 'b.fits')
    assert list(r['ring_flag']) == [False, True, False, True]
    assert list(r['ring_parent']) == [-1, 2, -1, 2]
    assert len(r) == 4
