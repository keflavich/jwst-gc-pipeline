"""``SATSTAR_OPTIN_FILTERS`` scopes the opt-in satstar fit switches by bandpass.

The chain launcher takes one environment per chain.  gc-treasury needs
``SATSTAR_ZF_KEEP_FINITE`` / ``SATSTAR_OBS_PK_FROM_CRF`` /
``SATSTAR_QFIT_LOCAL_GATE`` on F480M and off on F212N, in every phase, so the
scope is applied around the whole per-frame satstar fit
(``load_or_make_satstar_catalog``) and the cache key (``SATFITSW``) follows the
effective switches.
"""
import os

import numpy as np
import pytest
from astropy.io import fits

import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as CL
import jwst_gc_pipeline.reduction.saturated_star_finding as SSF
from jwst_gc_pipeline.reduction.saturated_star_finding import (
    SATSTAR_OPTIN_ENV, satstar_fit_switch_signature, satstar_fit_switches,
    satstar_optin_in_scope, satstar_optin_scope, satstar_optin_scoped)

ON = {'SATSTAR_ZF_KEEP_FINITE': '1', 'SATSTAR_OBS_PK_FROM_CRF': '1',
      'SATSTAR_QFIT_LOCAL_GATE': '1', 'SATSTAR_QFIT_LOCAL_R': '10',
      'SATSTAR_DEBLEND_ORDER': '1'}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in [n for n in os.environ if 'SATSTAR' in n]:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv('SATSTAR_SEED_CORE_DQ', '0')


def _set_optins(monkeypatch, filters=None):
    for k, v in ON.items():
        monkeypatch.setenv(k, v)
    if filters is not None:
        monkeypatch.setenv('SATSTAR_OPTIN_FILTERS', filters)


def _frame(tmp_path, filtername, name='f_crf.fits'):
    fn = tmp_path / name
    fits.HDUList([fits.PrimaryHDU(header=fits.Header(
        {'INSTRUME': 'NIRCAM', 'FILTER': filtername}))]).writeto(fn)
    return str(fn)


def test_unset_applies_to_every_bandpass():
    assert satstar_optin_in_scope('F212N')
    assert satstar_optin_in_scope('F480M')
    assert satstar_optin_in_scope(None)
    assert satstar_optin_in_scope('')


@pytest.mark.parametrize('listed', ['F480M', 'f480m', 'F405N,F480M',
                                    'F405N F480M', ' F480M '])
def test_listed_band_is_in_scope_others_not(monkeypatch, listed):
    monkeypatch.setenv('SATSTAR_OPTIN_FILTERS', listed)
    assert satstar_optin_in_scope('F480M')
    assert satstar_optin_in_scope('f480m')
    assert not satstar_optin_in_scope('F212N')


@pytest.mark.parametrize('bad', ['480M', 'F480', 'F480M;F212N', 'wide'])
def test_malformed_names_raise(monkeypatch, bad):
    monkeypatch.setenv('SATSTAR_OPTIN_FILTERS', bad)
    with pytest.raises(ValueError, match='SATSTAR_OPTIN_FILTERS'):
        satstar_optin_in_scope('F480M')


def test_set_scope_with_no_filter_keyword_raises(monkeypatch):
    monkeypatch.setenv('SATSTAR_OPTIN_FILTERS', 'F480M')
    with pytest.raises(ValueError, match='no FILTER keyword'):
        satstar_optin_in_scope('')


def test_scope_removes_and_restores_only_the_optins(monkeypatch):
    _set_optins(monkeypatch, 'F480M')
    monkeypatch.setenv('SATSTAR_ZF_FLAT', '1')
    with satstar_optin_scope('F212N'):
        for k in SATSTAR_OPTIN_ENV:
            assert k not in os.environ
        assert os.environ['SATSTAR_ZF_FLAT'] == '1'
    assert os.environ['SATSTAR_QFIT_LOCAL_R'] == '10'
    for k in SATSTAR_OPTIN_ENV[:3]:
        assert os.environ[k] == '1'


def test_scope_restores_after_an_exception(monkeypatch):
    _set_optins(monkeypatch, 'F480M')
    with pytest.raises(RuntimeError):
        with satstar_optin_scope('F212N'):
            raise RuntimeError('fit failed')
    for k in SATSTAR_OPTIN_ENV:
        assert k in os.environ


def test_in_scope_band_keeps_the_switches(monkeypatch):
    _set_optins(monkeypatch, 'F480M')
    with satstar_optin_scope('F480M'):
        for k in SATSTAR_OPTIN_ENV:
            assert k in os.environ


def test_effective_switches_follow_the_scope(monkeypatch):
    _set_optins(monkeypatch, 'F480M')
    with satstar_optin_scope('F480M'):
        sw = satstar_fit_switches()
    assert sw['keep_finite'] and sw['obs_pk_from_crf'] and sw['qfit_local_gate']
    with satstar_optin_scope('F212N'):
        sw = satstar_fit_switches()
    assert not (sw['keep_finite'] or sw['obs_pk_from_crf']
                or sw['qfit_local_gate'])


def test_cache_key_follows_the_scope(monkeypatch, tmp_path):
    """The key is computed inside the scope, so an F212N catalog fitted
    without the opt-ins carries the key of a run that never exported them."""
    fn = _frame(tmp_path, 'F212N')
    bare = satstar_fit_switch_signature(fn)
    _set_optins(monkeypatch, 'F480M')
    wrapped = satstar_optin_scoped(
        lambda filename: satstar_fit_switch_signature(filename))
    assert wrapped(fn) == bare
    assert satstar_fit_switch_signature(fn) != bare
    fn480 = _frame(tmp_path, 'F480M', 'g_crf.fits')
    assert wrapped(fn480) == satstar_fit_switch_signature(fn480) != bare


def test_decorator_reads_the_frame_filter_and_restores(monkeypatch, tmp_path):
    _set_optins(monkeypatch, 'F480M')
    seen = {}

    @satstar_optin_scoped
    def fit(filename, **kw):
        seen[filename] = {k: os.environ.get(k) for k in SATSTAR_OPTIN_ENV}
        return filename

    f212 = _frame(tmp_path, 'F212N', 'a_crf.fits')
    f480 = _frame(tmp_path, 'F480M', 'b_crf.fits')
    fit(f212)
    fit(filename=f480)
    assert set(seen[f212].values()) == {None}
    assert set(seen[f480].values()) == {'1', '10'}
    assert all(k in os.environ for k in SATSTAR_OPTIN_ENV)


def test_decorator_does_not_read_the_header_when_unset(monkeypatch):
    _set_optins(monkeypatch)

    @satstar_optin_scoped
    def fit(filename):
        return {k: os.environ.get(k) for k in SATSTAR_OPTIN_ENV}

    assert set(fit('/nonexistent/frame_crf.fits').values()) == {'1', '10'}


def test_decorator_unreadable_frame_with_scope_set_raises(monkeypatch):
    _set_optins(monkeypatch, 'F480M')

    @satstar_optin_scoped
    def fit(filename):
        return 1

    with pytest.raises(ValueError, match='no FILTER keyword'):
        fit('/nonexistent/frame_crf.fits')


def test_production_entry_point_is_scoped():
    """``load_or_make_satstar_catalog`` is the one place the per-frame fit and
    its cache key are computed; it must be wrapped."""
    wrapped = CL.load_or_make_satstar_catalog
    assert hasattr(wrapped, '__wrapped__')
    assert wrapped.__wrapped__.__name__ == 'load_or_make_satstar_catalog'
    assert CL.satstar_optin_scoped is SSF.satstar_optin_scoped


def _pupil_frame(tmp_path, filt, pupil, name):
    fn = tmp_path / name
    fits.HDUList([fits.PrimaryHDU(header=fits.Header(
        {'INSTRUME': 'NIRCAM', 'FILTER': filt, 'PUPIL': pupil}))]).writeto(fn)
    return str(fn)


@pytest.mark.parametrize('listed, kept', [('F405N', True), ('F480M', False),
                                          ('F444W', False)])
def test_pupil_wheel_band_is_matched_by_its_science_bandpass(
        monkeypatch, tmp_path, listed, kept):
    """FILTER=F444W, PUPIL=F405N is an F405N exposure."""
    fn = _pupil_frame(tmp_path, 'F444W', 'F405N', 'p_crf.fits')
    _set_optins(monkeypatch, listed)

    @satstar_optin_scoped
    def fit(filename):
        return {k: os.environ.get(k) for k in SATSTAR_OPTIN_ENV}

    seen = set(fit(fn).values())
    assert seen == ({'1', '10'} if kept else {None})


def test_frame_naming_no_bandpass_raises_with_scope_set(monkeypatch, tmp_path):
    """FILTER=CLEAR, PUPIL=CLEAR names no science bandpass; with a scope set the
    frame cannot be classified, so the fit stops rather than guessing."""
    fn = _pupil_frame(tmp_path, 'CLEAR', 'CLEAR', 'c_crf.fits')
    _set_optins(monkeypatch, 'F480M')

    @satstar_optin_scoped
    def fit(filename):
        return 1

    with pytest.raises(ValueError):
        fit(fn)
