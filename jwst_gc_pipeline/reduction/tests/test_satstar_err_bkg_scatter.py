"""Satstar fit errors gain the local background scatter (SATSTAR_ERR_BKG_SCATTER).

The crf ERR holds read and Poisson noise only.  On wd2 F150W the background of
a crowded satstar's fit box scatters ~25x more than ERR, so the background
pixels dominate chi2 and 15-18 mag stars came out 0.4-0.8 mag faint.
``bkg_scatter_fit_error`` adds the robust annulus scatter in quadrature.
"""
import inspect

import numpy as np
import pytest

from jwst_gc_pipeline.reduction import saturated_star_finding as ssf

SHAPE = (81, 81)
X0 = Y0 = 40.0
INNER, OUTER = 15, 30


def _scene(sigma=12.0, err=0.5, seed=1):
    rng = np.random.default_rng(seed)
    cutout = 20.0 + rng.normal(0.0, sigma, SHAPE)
    errs = np.full(SHAPE, err)
    valid = np.ones(SHAPE, dtype=bool)
    return cutout, errs, valid


def test_recovers_gaussian_scatter():
    cutout, err, valid = _scene(sigma=12.0)
    err_eff, sigma = ssf.bkg_scatter_fit_error(err, cutout, valid, X0, Y0,
                                               INNER, OUTER)
    assert sigma == pytest.approx(12.0, rel=0.08)
    np.testing.assert_allclose(err_eff, np.sqrt(err ** 2 + sigma ** 2))
    assert np.all(err_eff >= err)


def test_star_inside_annulus_hole_is_ignored():
    # A bright star at the centre (inside bkg_inner) must not inflate sigma.
    cutout, err, valid = _scene(sigma=2.0)
    yy, xx = np.indices(SHAPE)
    cutout = cutout + 1e5 * np.exp(-((xx - X0) ** 2 + (yy - Y0) ** 2) / 8.0)
    _, sigma = ssf.bkg_scatter_fit_error(err, cutout, valid, X0, Y0,
                                         INNER, OUTER)
    assert sigma == pytest.approx(2.0, rel=0.1)


def test_masked_pixels_are_excluded():
    cutout, err, valid = _scene(sigma=2.0)
    yy, xx = np.indices(SHAPE)
    rr = np.hypot(xx - X0, yy - Y0)
    # Half the annulus is wild, but masked: sigma must ignore it.
    wild = (rr >= INNER) & (rr < OUTER) & (xx > X0)
    cutout[wild] = 1e4 * (-1.0) ** np.arange(wild.sum())
    valid[wild] = False
    cutout[0, 0] = np.nan
    _, sigma = ssf.bkg_scatter_fit_error(err, cutout, valid, X0, Y0,
                                         INNER, OUTER)
    assert sigma == pytest.approx(2.0, rel=0.12)


def test_too_few_pixels_leaves_err_unchanged():
    cutout, err, valid = _scene()
    valid[:] = False
    valid[0, :5] = True
    err_eff, sigma = ssf.bkg_scatter_fit_error(err, cutout, valid, X0, Y0,
                                               INNER, OUTER)
    assert np.isnan(sigma)
    assert err_eff is err


def test_zero_scatter_leaves_err_unchanged():
    cutout = np.full(SHAPE, 7.0)
    err = np.full(SHAPE, 0.5)
    valid = np.ones(SHAPE, dtype=bool)
    err_eff, sigma = ssf.bkg_scatter_fit_error(err, cutout, valid, X0, Y0,
                                               INNER, OUTER)
    assert sigma == 0
    assert err_eff is err


def test_gated_on_env_and_feeds_fit_error():
    # The full get_saturated_stars needs real PSF grids and a full frame, so
    # the wiring is checked on the source (as in test_satstar_float_seed.py).
    src = inspect.getsource(ssf.get_saturated_stars)
    assert "_fit_switches['err_bkg_scatter']" in src
    i_call = src.index("bkg_scatter_fit_error(")
    i_fit = src.index("error=err_cutout_eff", i_call)
    assert i_call < i_fit


@pytest.mark.parametrize('value, on', [(None, False), ('', False), ('0', False),
                                       ('1', True), ('on', True)])
def test_switch_is_read_with_the_other_fit_switches(value, on):
    env = {} if value is None else {'SATSTAR_ERR_BKG_SCATTER': value}
    assert ssf.satstar_fit_switches(env)['err_bkg_scatter'] is on


def test_switch_typo_raises():
    with pytest.raises(ValueError, match='SATSTAR_ERR_BKG_SCATTER'):
        ssf.satstar_fit_switches({'SATSTAR_ERR_BKG_SCATTER': 'yse'})
