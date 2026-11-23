"""SATSTAR_FITTER_MAXITERS: the iteration cap of the per-star satstar LevMar fit.

Unset it is 100, photutils' own ``PSFPhotometry`` default, so the fit and the
cache key are what they were before the switch.  A fit that reaches the cap
returns the iterate it stopped at (photutils flag bit 8); a larger cap lets a
slowly converging fit finish, and the cache key records the cap so a catalog
fitted under another one is refit.
"""
import numpy as np
import pytest
from astropy.io import fits

import jwst_gc_pipeline.reduction.satstar_deblend as SD
import jwst_gc_pipeline.reduction.saturated_star_finding as SSF
from jwst_gc_pipeline.reduction.saturated_star_finding import (
    SATSTAR_FITTER_MAXITERS_DEFAULT, satstar_fit_switch_signature,
    satstar_fit_switches, satstar_fitter_maxiters)
from jwst_gc_pipeline.reduction.tests.test_satstar_recovered_cap import (
    _two_star_frame)
from jwst_gc_pipeline.reduction.tests.test_satstar_wing_multiplier import (
    _EvalPRF)

ENV = 'SATSTAR_FITTER_MAXITERS'


def test_default_is_photutils_default():
    assert SATSTAR_FITTER_MAXITERS_DEFAULT == 100
    assert satstar_fitter_maxiters({}) == 100
    assert satstar_fitter_maxiters({ENV: ''}) == 100
    assert satstar_fitter_maxiters({ENV: '  '}) == 100
    assert satstar_fit_switches({})['fitter_maxiters'] == 100


@pytest.mark.parametrize('raw, expect', [('1', 1), ('1000', 1000),
                                         (' 250 ', 250)])
def test_a_positive_integer_is_read(raw, expect):
    assert satstar_fitter_maxiters({ENV: raw}) == expect
    assert satstar_fit_switches({ENV: raw})['fitter_maxiters'] == expect


@pytest.mark.parametrize('raw', ['abc', '0', '-5', '1.5', '1e3'])
def test_anything_else_raises_naming_the_variable(raw):
    with pytest.raises(ValueError, match=ENV):
        satstar_fitter_maxiters({ENV: raw})


def _frame(tmp_path):
    path = tmp_path / 'a_cal.fits'
    fits.PrimaryHDU(header=fits.Header({'FILTER': 'F200W', 'PUPIL': 'CLEAR'})
                    ).writeto(path, overwrite=True)
    return str(path)


def test_cache_key_is_unchanged_at_the_default_and_records_another_cap(
        tmp_path):
    path = _frame(tmp_path)
    base = satstar_fit_switch_signature(path, env={})
    assert satstar_fit_switch_signature(path, env={ENV: '100'}) == base
    sig = satstar_fit_switch_signature(path, env={ENV: '1000'})
    assert sig == '_'.join(p for p in (base, 'mi1000') if p)
    assert satstar_fit_switch_signature(path, env={ENV: '50'}) != sig


def _fit_kwargs(monkeypatch, env):
    """Keyword arguments of every ``PSFPhotometry`` the satstar fit builds."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    fh, sat = _two_star_frame()
    n = sat.shape[0]
    seen = []

    class _Recorder(SSF.PSFPhotometry):
        def __init__(self, *args, **kw):
            seen.append(dict(kw))
            super().__init__(*args, **kw)

    def _records(saturated, sources, coms, sizes, zeroframe, data, fwhm,
                 **kw):
        return [{'com': c, 'label': 1, 'forced': False,
                 'sat_area': int(sat.sum())}
                for c in [(100.0, 93.0), (100.0, 107.0)]]

    def _anchor(data, dq, zeroframe, **kw):
        return data, sat.copy(), np.zeros_like(sat), None

    monkeypatch.setattr(SD, 'build_deblended_source_records', _records)
    monkeypatch.setattr(SSF, 'zeroframe_fit_anchor', _anchor)
    monkeypatch.setattr(SSF, 'get_psf',
                        lambda header, **kw: _EvalPRF(fwhm=1.6))
    monkeypatch.setattr(SSF.stpsf, 'NIRCam', lambda *a, **kw: object())
    monkeypatch.setattr(SSF, 'PSFPhotometry', _Recorder)
    SSF.get_saturated_stars(fh, zeroframe=np.ones((n, n)),
                            zeroframe_deblend=True, plot=False)
    return seen


def test_the_fit_gets_photutils_default_when_unset(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    seen = _fit_kwargs(monkeypatch, {})
    assert seen, "the satstar fit built no PSFPhotometry"
    assert {kw['fitter_maxiters'] for kw in seen} == {100}


def test_the_fit_gets_the_requested_cap(monkeypatch):
    seen = _fit_kwargs(monkeypatch, {ENV: '1000'})
    assert seen, "the satstar fit built no PSFPhotometry"
    assert {kw['fitter_maxiters'] for kw in seen} == {1000}
