"""The per-exposure satstar cache must key on the env fit switches (#972).

``_satstar_recovery_signature`` (meta ``SATRECOV``) keys the cache on
``options`` only.  The ZEROFRAME R-curve guard, keep-finite, observed peak from
the crf and the local-qfit gate are ENV switches, so before this key a run that
turned them on over an existing tree silently got the old catalogs back.  The
guard is also ON by default now, which the env itself cannot show, so the key
(meta ``SATFITSW``, ``satstar_fit_switch_signature``) resolves the defaults.

It is ``''`` for every configuration that fits as the code did before the
switches, and an unstamped catalog reads as ``''``: a frame the switches cannot
change (no ramp, so no ZEROFRAME anchor) keeps its cache.
"""
import builtins
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

import jwst_gc_pipeline.photometry.crowdsource_catalogs_long as CL
import jwst_gc_pipeline.reduction.saturated_star_finding as SSF
from jwst_gc_pipeline.photometry import satstar_cache as SC
from jwst_gc_pipeline.reduction.saturated_star_finding import (
    satstar_fit_switch_signature)

FRAME = 'jw02221001001_07101_00001_nrca2_destreak_o001_crf.fits'
RAMP = 'jw02221001001_07101_00001_nrca2_ramp.fits'


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in [n for n in os.environ if 'SATSTAR' in n]:
        monkeypatch.delenv(name, raising=False)


def _frame(tmp_path, with_ramp):
    fn = tmp_path / FRAME
    fits.HDUList([fits.PrimaryHDU(header=fits.Header(
                      {'INSTRUME': 'NIRCAM', 'DETECTOR': 'NRCA2',
                       'FILTER': 'F182M'})),
                  fits.ImageHDU(data=np.zeros((8, 8), 'float32'), name='SCI'),
                  fits.ImageHDU(data=np.zeros((8, 8), 'int32'), name='DQ')]
                 ).writeto(fn, overwrite=True)
    if with_ramp:
        fits.PrimaryHDU().writeto(tmp_path / RAMP, overwrite=True)
    return str(fn)


def _write_cache(path, satrecov='off', satfitsw=None):
    t = Table({'flux': [1.0, 2.0]})
    if satrecov is not None:
        t.meta['SATRECOV'] = satrecov
    if satfitsw is not None:
        t.meta['SATFITSW'] = satfitsw
    t.write(path, overwrite=True)


def _counting_fit(monkeypatch):
    """Stub remove_saturated_stars: count calls and stamp what the real one
    stamps, from the signatures it is handed."""
    calls = {'n': 0, 'fit_switch_signature': []}

    def _stub(filename, overwrite=True, file_suffix='', recovery_signature=None,
              fit_switch_signature=None, **kw):
        calls['n'] += 1
        calls['fit_switch_signature'].append(fit_switch_signature)
        _write_cache(filename.replace('.fits',
                                      f'{file_suffix}_satstar_catalog.fits'),
                     satrecov=recovery_signature,
                     satfitsw=fit_switch_signature or None)
    monkeypatch.setattr(CL, 'remove_saturated_stars', _stub)
    return calls


def _load(fn, tmp_path):
    return CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path),
                                           recovery_signature='off')


# --------------------------------------------------------------------------
# the signature
# --------------------------------------------------------------------------

def test_signature_is_empty_without_a_ramp(tmp_path, monkeypatch):
    """No ramp -> no ZEROFRAME anchor -> the anchor switches cannot act."""
    fn = _frame(tmp_path, with_ramp=False)
    assert satstar_fit_switch_signature(fn) == ''
    monkeypatch.setenv('SATSTAR_ZF_KEEP_FINITE', '1')
    monkeypatch.setenv('SATSTAR_OBS_PK_FROM_CRF', '1')
    assert satstar_fit_switch_signature(fn) == ''


def test_signature_with_a_ramp_carries_the_default_guard(tmp_path, monkeypatch):
    fn = _frame(tmp_path, with_ramp=True)
    assert satstar_fit_switch_signature(fn) == 'zfg1.3'
    monkeypatch.setenv('SATSTAR_ZF_RCURVE_GUARD', '0')
    assert satstar_fit_switch_signature(fn) == ''          # = pre-#972 fit
    monkeypatch.delenv('SATSTAR_ZF_RCURVE_GUARD')
    monkeypatch.setenv('SATSTAR_ZF_RCURVE_MAXSTEP', '1.5')
    assert satstar_fit_switch_signature(fn) == 'zfg1.5'
    monkeypatch.delenv('SATSTAR_ZF_RCURVE_MAXSTEP')
    monkeypatch.setenv('SATSTAR_ZF_KEEP_FINITE', '1')
    monkeypatch.setenv('SATSTAR_OBS_PK_FROM_CRF', '1')
    assert satstar_fit_switch_signature(fn) == 'zfg1.3ko'


def test_signature_follows_whether_the_anchor_runs(tmp_path, monkeypatch):
    """SATSTAR_ZEROFRAME_FIT=0 turns the anchor off, unless the deblend loads
    the ZEROFRAME anyway (remove_saturated_stars does the same)."""
    fn = _frame(tmp_path, with_ramp=True)
    monkeypatch.setenv('SATSTAR_ZEROFRAME_FIT', '0')
    assert satstar_fit_switch_signature(fn) == ''
    assert satstar_fit_switch_signature(
        fn, deblend_with_zeroframe=True) == 'zfg1.3'


@pytest.mark.parametrize('value, on', [
    ('0', False), ('false', False), ('False', False), ('FALSE', False),
    ('off', False), ('no', False), (' Off ', False), ('1', True),
    ('true', True), ('on', True), ('YES', True), ('', True), ('  ', True)])
def test_zeroframe_fit_switch_spellings(tmp_path, monkeypatch, value, on):
    """SATSTAR_ZEROFRAME_FIT is parsed like the #972 switches.  Before, only
    0/false/False turned it off, so 'off', 'no' and 'FALSE' left the anchor
    ON and keyed the cache as if it were on (pre-existing, found in review)."""
    fn = _frame(tmp_path, with_ramp=True)
    monkeypatch.setenv('SATSTAR_ZEROFRAME_FIT', value)
    assert SSF._zeroframe_fit_enabled() is on
    assert satstar_fit_switch_signature(fn) == ('zfg1.3' if on else '')


def test_zeroframe_fit_switch_typo_raises(tmp_path, monkeypatch):
    fn = _frame(tmp_path, with_ramp=True)
    monkeypatch.setenv('SATSTAR_ZEROFRAME_FIT', 'of')
    with pytest.raises(ValueError, match='SATSTAR_ZEROFRAME_FIT'):
        satstar_fit_switch_signature(fn)


@pytest.mark.parametrize('value, loads', [
    (None, True), ('1', True), ('off', False), ('no', False), ('FALSE', False)])
def test_remove_saturated_stars_honours_zeroframe_fit_off(tmp_path, monkeypatch,
                                                         value, loads):
    """The writer reads the same switch: with it off, no first read is handed
    to the fitter."""
    fn = _frame(tmp_path, with_ramp=False)
    for attr in ('satstar_wingcal_measurements', 'satstar_rejected',
                 'satstar_model', 'satstar_resid', 'satstar_flagimg'):
        monkeypatch.delattr(builtins, attr, raising=False)
    if value is not None:
        monkeypatch.setenv('SATSTAR_ZEROFRAME_FIT', value)
    seen = {}
    monkeypatch.setattr(SSF, '_find_zeroframe_for',
                        lambda filename: np.ones((8, 8)))

    def _fit(fh, **kw):
        seen.update(kw)
        return Table({'flux_fit': [1.0]})
    monkeypatch.setattr(SSF, 'get_saturated_stars', _fit)
    SSF.remove_saturated_stars(fn, recovery_signature='off')
    assert ('zeroframe' in seen) is loads


def test_signature_keys_the_local_qfit_with_or_without_a_ramp(tmp_path,
                                                              monkeypatch):
    fn = _frame(tmp_path, with_ramp=False)
    monkeypatch.setenv('SATSTAR_QFIT_LOCAL_GATE', '1')
    assert satstar_fit_switch_signature(fn) == 'ql10g1'
    monkeypatch.setenv('SATSTAR_QFIT_LOCAL_MAX', '0.8')
    assert satstar_fit_switch_signature(fn) == 'ql10g0.8'
    monkeypatch.delenv('SATSTAR_QFIT_LOCAL_GATE')
    monkeypatch.setenv('SATSTAR_QFIT_LOCAL_R', '12')
    assert satstar_fit_switch_signature(fn) == 'ql12'      # column only


# --------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------

def test_catalog_built_without_the_guard_is_refit_once(tmp_path, monkeypatch):
    """A pre-#972 catalog (no SATFITSW) of a frame with a ramp was fitted with
    the collapsed curve; the default guard run must refit it, stamp the new
    catalog, and reuse that one afterwards."""
    fn = _frame(tmp_path, with_ramp=True)
    _write_cache(tmp_path / FRAME.replace('.fits', '_satstar_catalog.fits'))
    calls = _counting_fit(monkeypatch)
    out = _load(fn, tmp_path)
    assert calls['n'] == 1
    assert calls['fit_switch_signature'] == ['zfg1.3']
    assert str(out.meta.get('SATFITSW')) == 'zfg1.3'
    _load(fn, tmp_path)
    assert calls['n'] == 1


@pytest.mark.parametrize('name, value', [
    ('SATSTAR_ZF_KEEP_FINITE', '1'),
    ('SATSTAR_OBS_PK_FROM_CRF', '1'),
    ('SATSTAR_QFIT_LOCAL_GATE', '1'),
    ('SATSTAR_QFIT_LOCAL_R', '10'),
    ('SATSTAR_ZF_RCURVE_GUARD', '0'),
    ('SATSTAR_ZF_RCURVE_MAXSTEP', '2'),
])
def test_changing_a_switch_refits_a_cached_catalog(tmp_path, monkeypatch,
                                                  name, value):
    fn = _frame(tmp_path, with_ramp=True)
    _write_cache(tmp_path / FRAME.replace('.fits', '_satstar_catalog.fits'),
                 satfitsw='zfg1.3')          # built by a default run
    calls = _counting_fit(monkeypatch)
    _load(fn, tmp_path)
    assert calls['n'] == 0                   # same switches -> reused
    monkeypatch.setenv(name, value)
    _load(fn, tmp_path)
    assert calls['n'] == 1                   # switch changed -> refit
    monkeypatch.delenv(name)
    _load(fn, tmp_path)
    assert calls['n'] == 2                   # and back again


def test_frame_without_a_ramp_keeps_its_old_catalog(tmp_path, monkeypatch):
    """The guard cannot change a frame the anchor never runs on, so its
    pre-#972 catalog stays valid: no needless refit."""
    fn = _frame(tmp_path, with_ramp=False)
    _write_cache(tmp_path / FRAME.replace('.fits', '_satstar_catalog.fits'))
    calls = _counting_fit(monkeypatch)
    _load(fn, tmp_path)
    assert calls['n'] == 0


def test_local_qfit_gate_refits_even_without_a_ramp(tmp_path, monkeypatch):
    fn = _frame(tmp_path, with_ramp=False)
    _write_cache(tmp_path / FRAME.replace('.fits', '_satstar_catalog.fits'))
    monkeypatch.setenv('SATSTAR_QFIT_LOCAL_GATE', '1')
    calls = _counting_fit(monkeypatch)
    _load(fn, tmp_path)
    assert calls['n'] == 1


def test_extended_catalog_is_checked_the_same_way(tmp_path, monkeypatch):
    fn = _frame(tmp_path, with_ramp=True)
    _write_cache(tmp_path / FRAME.replace('.fits',
                                          '_extended_satstar_catalog.fits'))
    calls = _counting_fit(monkeypatch)
    _load(fn, tmp_path)
    assert calls['n'] == 1


def test_no_recovery_signature_keeps_the_old_always_reuse_path(tmp_path,
                                                              monkeypatch):
    """Callers that pass no recovery_signature (the legacy crowdsource step)
    opted out of cache keying; that stays true for the switches."""
    fn = _frame(tmp_path, with_ramp=True)
    _write_cache(tmp_path / FRAME.replace('.fits', '_satstar_catalog.fits'))
    calls = _counting_fit(monkeypatch)
    CL.load_or_make_satstar_catalog(fn, path_prefix=str(tmp_path))
    assert calls['n'] == 0


def test_cross_phase_content_key_sees_the_default_guard(tmp_path):
    """The cross-phase key hashes SATSTAR* env only when set, so it cannot
    see a changed default; the resolved signature is fed when non-empty."""
    fn = _frame(tmp_path, with_ramp=True)
    base = SC.satstar_content_key(fn, path_prefix=str(tmp_path))
    assert SC.satstar_content_key(fn, path_prefix=str(tmp_path),
                                  fit_switch_signature='') == base
    assert SC.satstar_content_key(fn, path_prefix=str(tmp_path),
                                  fit_switch_signature='zfg1.3') != base


def test_remove_saturated_stars_stamps_the_switch_signature(tmp_path,
                                                            monkeypatch):
    """The real writer, with the fitter stubbed: SATFITSW is stamped when the
    signature is non-empty and left off when it is ''."""
    fn = _frame(tmp_path, with_ramp=False)
    for attr in ('satstar_wingcal_measurements', 'satstar_rejected',
                 'satstar_model', 'satstar_resid', 'satstar_flagimg'):
        monkeypatch.delattr(builtins, attr, raising=False)
    monkeypatch.setattr(SSF, 'get_saturated_stars',
                        lambda fh, **kw: Table({'flux_fit': [1.0]}))
    cat = fn.replace('.fits', '_satstar_catalog.fits')
    SSF.remove_saturated_stars(fn, recovery_signature='off',
                               fit_switch_signature='zfg1.3ko')
    meta = Table.read(cat).meta
    assert meta['SATRECOV'] == 'off' and meta['SATFITSW'] == 'zfg1.3ko'
    SSF.remove_saturated_stars(fn, recovery_signature='off',
                               fit_switch_signature='')
    assert 'SATFITSW' not in Table.read(cat).meta
