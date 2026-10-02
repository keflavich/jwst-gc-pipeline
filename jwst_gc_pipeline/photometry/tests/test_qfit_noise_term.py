"""Vetting: pixel-noise qfit bound on the data-i2d prominence keep.

``star_prom_min > 0`` keeps a source whose prominence clears it whatever its
qfit.  qfit = sum|resid|/flux of a perfect PSF fit is ~c/(S/N) from pixel noise
alone, so ``qfit_snr_k > 0`` makes that keep also need
``qfit <= sqrt(qfit_max**2 + (qfit_snr_k / snr)**2)``: a faint star with a
noisy qfit stays, a bright fit far above the noise floor (a wing or ring fit,
a blend, an emission knot) is refused.  ``qfit_snr_k = 0`` is the keep without
the bound, and the bound touches no other branch.
"""
import ast
import inspect
import os

import numpy as np
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry.cataloging import _filter_extended_emission
from jwst_gc_pipeline.photometry.manual_defaults import MANUAL_DEFAULTS

NY = NX = 200
STAR, KNOT = (50, 50), (150, 150)


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [NX / 2, NY / 2]
    w.wcs.crval = [266.5, -28.7]
    w.wcs.cdelt = [-0.063 / 3600, 0.063 / 3600]
    return w


def _gauss(data, x, y, amp, sig=1.0):
    yy, xx = np.mgrid[0:NY, 0:NX]
    data += amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sig ** 2))


def _image(seed=3):
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, (NY, NX))
    _gauss(data, *STAR, amp=30.0)
    # a faint knot on a bright patch: prominence 4-7, kept by peak_SB
    data[KNOT[1] - 18:KNOT[1] + 18, KNOT[0] - 18:KNOT[0] + 18] += 40.0
    _gauss(data, *KNOT, amp=7.0)
    return data


def _run(qfit, snr, qfit_snr_k, star_prom_min=7.0, local_bkg=-0.05, flags=0,
         xy=STAR, data=None):
    """One source; local_bkg < 0 keeps it out of the peak_SB branch."""
    w = _wcs()
    flux_err = 10.0 if np.isfinite(snr) else 0.0
    flux = 10.0 * snr if np.isfinite(snr) else 80.0
    cat = Table({'skycoord': w.pixel_to_world([xy[0]], [xy[1]]),
                 'qfit': [qfit], 'flags': [flags], 'local_bkg': [local_bkg],
                 'flux': [flux], 'flux_err': [flux_err],
                 'group_size': [1], 'id': [0]})
    with np.errstate(divide='ignore', invalid='ignore'):
        out = _filter_extended_emission(cat, data_i2d_image=_image() if data is None else data,
                                        ww_i2d=w, star_prom_min=star_prom_min,
                                        star_prom_peak_min=4.0, qfit_snr_k=qfit_snr_k,
                                        sky_clean_keep=False, label='test')
    return len(out) == 1, float(cat['prominence'][0])


def test_star_is_prominent_and_off_the_peak_sb_branch():
    kept, prom = _run(0.45, 8.0, 0.0, star_prom_min=0.0)
    assert prom > 10 and not kept


def test_faint_star_with_noisy_qfit_kept():
    # S/N 8: bound sqrt(0.2^2 + (5/8)^2) = 0.656
    for k in (0.0, 5.0):
        assert _run(0.45, 8.0, k)[0]


def test_bright_fit_above_noise_floor_refused():
    # S/N 60: bound sqrt(0.2^2 + (5/60)^2) = 0.217; qfit 0.45 >= 0.4 is also
    # outside the bright-isolated keep
    assert _run(0.45, 60.0, 0.0)[0]
    assert not _run(0.45, 60.0, 5.0)[0]


def test_bound_is_quadrature_sum():
    # S/N 16, k 5: sqrt(0.2^2 + 0.3125^2) = 0.371.  A linear sum (0.5125)
    # would keep 0.38; k/S/N alone (0.3125) would refuse 0.36
    assert _run(0.36, 16.0, 5.0)[0]
    assert not _run(0.38, 16.0, 5.0)[0]
    # k scales the noise term: k 6 -> 0.430
    assert _run(0.42, 16.0, 6.0)[0]


def test_nan_qfit_refused_by_bound():
    assert _run(np.nan, 8.0, 0.0)[0]
    assert not _run(np.nan, 8.0, 5.0)[0]


def test_unmeasured_snr_gets_flat_qfit_max():
    # flux_err 0 -> S/N inf: the S/N floor passes it, the bound is qfit_max
    assert _run(0.45, np.inf, 0.0)[0]
    assert not _run(0.45, np.inf, 5.0)[0]
    assert _run(0.15, np.inf, 5.0)[0]


def test_bound_leaves_other_branches_alone():
    # peak_SB branch (local_bkg > 0, knot prominence >= star_prom_peak_min 4)
    kept, prom = _run(0.9, 60.0, 5.0, local_bkg=0.05, xy=KNOT)
    assert 4 <= prom < 7 and kept
    # keep_flags
    assert _run(0.9, 60.0, 5.0, flags=1)[0]
    # qfit <= qfit_max
    assert _run(0.19, 60.0, 5.0)[0]
    # bright isolated (S/N >= 20, qfit < 0.4)
    assert _run(0.39, 60.0, 5.0)[0]


def _random_field(seed=11, n_star=90):
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, (NY, NX))
    xs = rng.uniform(12, NX - 12, n_star)
    ys = rng.uniform(12, NY - 12, n_star)
    for x, y, a in zip(xs, ys, np.exp(rng.uniform(np.log(3), np.log(120), n_star))):
        _gauss(data, x, y, a)
    w = _wcs()
    qfit = rng.uniform(0.0, 1.0, n_star)
    qfit[rng.random(n_star) < 0.1] = np.nan
    snr = np.exp(rng.uniform(np.log(3), np.log(100), n_star))
    flux_err = np.where(rng.random(n_star) < 0.07, 0.0, 10.0)
    cat = Table({'skycoord': w.pixel_to_world(xs, ys), 'qfit': qfit,
                 'flags': (rng.random(n_star) < 0.05).astype(float),
                 'local_bkg': rng.normal(0.0, 0.05, n_star),
                 'flux': 10.0 * snr, 'flux_err': flux_err,
                 'group_size': rng.integers(1, 3, n_star), 'id': np.arange(n_star)})
    return data, w, cat


def _kept(data, w, cat, **kw):
    t = cat.copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        out = _filter_extended_emission(t, data_i2d_image=data, ww_i2d=w,
                                        star_prom_peak_min=4.0,
                                        sky_clean_keep=False, label='test', **kw)
    keep = np.zeros(len(cat), bool)
    keep[np.asarray(out['id'])] = True
    return keep, t


def test_k0_is_prominence_keep_and_k5_only_removes_its_refusals():
    data, w, cat = _random_field()
    k_default, t = _kept(data, w, cat, star_prom_min=7.0)
    k0, _ = _kept(data, w, cat, star_prom_min=7.0, qfit_snr_k=0.0)
    k5, _ = _kept(data, w, cat, star_prom_min=7.0, qfit_snr_k=5.0)
    no_prom, _ = _kept(data, w, cat, star_prom_min=0.0)
    # the function default is the keep without the bound
    assert np.array_equal(k0, k_default)
    qf = np.asarray(cat['qfit'], float)
    with np.errstate(divide='ignore', invalid='ignore'):
        snr = np.asarray(cat['flux'], float) / np.asarray(cat['flux_err'], float)
    bound = np.full(len(cat), 0.2)
    ok = np.isfinite(snr) & (snr > 0)
    bound[ok] = np.hypot(0.2, 5.0 / snr[ok])
    within = np.isfinite(qf) & (qf <= bound)
    # within the bound k 5 is k 0; outside it the prominence keep is off
    assert np.array_equal(k5, np.where(within, k0, no_prom))
    assert not np.any(k5 & ~k0)
    # the field exercises both sides
    prom = np.asarray(t['prominence'], float)
    refused = k0 & ~k5
    assert refused.sum() >= 5 and np.all(prom[refused] >= 7)
    assert (k5 & ~no_prom).sum() >= 5


def test_pipeline_default_and_wiring():
    assert MANUAL_DEFAULTS['manual_ext_qfit_snr_k'] == 5.0
    # run_manual_pipeline reads qfit_snr_k from its own option
    from jwst_gc_pipeline.photometry import cataloging
    tree = ast.parse(inspect.getsource(cataloging.run_manual_pipeline))
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call)
             and getattr(c.func, 'id', None) == '_filter_extended_emission']
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    mopt = [c for c in ast.walk(kw['qfit_snr_k']) if isinstance(c, ast.Call)
            and getattr(c.func, 'id', None) == 'mopt']
    assert len(mopt) == 1 and mopt[0].args[1].value == 'manual_ext_qfit_snr_k'
    # and the CLI defines that option with the canonical default
    src = open(os.path.join(os.path.dirname(cataloging.__file__),
                            'crowdsource_catalogs_long.py')).read()
    assert ('"--manual-ext-qfit-snr-k", dest="manual_ext_qfit_snr_k",\n'
            "                    type='float', default=MANUAL_DEFAULTS['manual_ext_qfit_snr_k']"
            in src)
