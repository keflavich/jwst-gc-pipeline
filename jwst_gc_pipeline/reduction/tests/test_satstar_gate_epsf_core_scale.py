"""``SATSTAR_GATE_EPSF_CORE_SCALE`` follows the implied-peak gate to the ePSF-core PSF.

The implied-peak gate rejects a fitted satstar whose model peak is below half
the severity floor.  The floor was set with STPSF fits; the hybrid grid
(``PSF_EPSF_CORE_DIR``) has a lower peak per unit flux, and the fit takes a lower
flux with it, so the implied peak of a star near the gate falls by 2.4 % (NRCA5)
to 4.5 % (NRCB5) in F480M and the star is handed to the daophot channel (the
o105 F480M leftover at (4452, 858): implied peak 2543.4 with STPSF, 2410.9 with
the core, gate 2500).  The switch is opt-in, default off, and scales only the
model-implied peak.  Synthetic grids and cores only, as in
``test_satstar_hybrid_psf.py``.
"""
import inspect
import os

import numpy as np
import pytest
from astropy.io import fits

from jwst_gc_pipeline.photometry import epsf_hybrid as eh
from jwst_gc_pipeline.photometry.tests.test_epsf_hybrid import _core, _stpsf_grid
from jwst_gc_pipeline.reduction import saturated_star_finding as ssf

FLOOR = 5000.0     # SAT_SEVERITY_FLOOR['f480m']; the gate is 0.5 * FLOOR = 2500
ON = 'SATSTAR_GATE_EPSF_CORE_SCALE'
VALUE = 'SATSTAR_GATE_EPSF_CORE_SCALE_VALUE'


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (ON, VALUE, eh.EPSF_CORE_DIR_ENV, 'SATSTAR_SEED_CORE_DQ'):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(eh, '_APPLIED', {})
    monkeypatch.setattr(eh, '_HYBRID_CACHE', {})


@pytest.fixture
def cores(tmp_path):
    d = tmp_path / 'cores'
    d.mkdir()
    for det in ('NRCA5', 'NRCB5'):
        eh.write_epsf_core(d / eh.epsf_core_filename(det, 'F480M'), _core(), det, 'F480M')
    return str(d)


@pytest.fixture
def hybrid(cores):
    return eh.maybe_apply_epsf_core(_stpsf_grid(), 'NRCB5', 'F480M',
                                    environ={eh.EPSF_CORE_DIR_ENV: cores})


def _frame(tmp_path, filtername, detector, name='f_crf.fits'):
    fn = tmp_path / name
    fits.HDUList([fits.PrimaryHDU(header=fits.Header(
        {'INSTRUME': 'NIRCAM', 'FILTER': filtername, 'DETECTOR': detector}))]).writeto(fn)
    return str(fn)


def _former_inline_gate(implied, obs, floor):
    """The expression ``get_saturated_stars`` carried before the predicate."""
    return bool(np.isfinite(implied) and implied < 0.5 * float(floor)
                and not (np.isfinite(obs) and obs >= 0.5 * float(floor)))


@pytest.mark.parametrize('floor', [2500., 4000., 5000., 8000.])
@pytest.mark.parametrize('obs', [np.nan, 0., 1959.1, 2499.9, 2500., 4000.])
@pytest.mark.parametrize('implied', [np.nan, -1., 0., 1000., 2400., 2499.9, 2500., 2500.1, 6000.])
def test_scale_one_is_the_former_expression(implied, obs, floor):
    want = _former_inline_gate(implied, obs, floor)
    assert ssf.satstar_implied_peak_gate_rejects(implied, obs, floor) is want
    assert ssf.satstar_implied_peak_gate_rejects(implied, obs, floor, 1.0) is want


def test_default_off_keeps_the_gate_and_the_cache_key(hybrid, cores, tmp_path):
    assert ssf._resolve_satstar_gate_core_scale('f480m', 'NRCB5', hybrid) == 1.0
    assert ssf._resolve_satstar_gate_core_scale('f480m', 'NRCA5', None) == 1.0
    assert ssf.satstar_fit_switches()['gate_epsf_core_scale'] is False
    fn = _frame(tmp_path, 'F480M', 'NRCBLONG')
    bare = ssf.satstar_fit_switch_signature(fn, env={})
    assert not [p for p in bare.split('_') if p.startswith('gs')]
    # off, the other variables do nothing
    env = {eh.EPSF_CORE_DIR_ENV: cores, VALUE: '0.9'}
    assert ssf.satstar_fit_switch_signature(fn, env=env) == ssf.satstar_fit_switch_signature(
        fn, env={eh.EPSF_CORE_DIR_ENV: cores})


def test_f5_star_numbers(monkeypatch, hybrid):
    """o105 F480M nrcblong exposure 2: STPSF implied peak 2543.4 (kept), core 2410.9 with observed peak 1959.1."""
    reject = ssf.satstar_implied_peak_gate_rejects
    assert not reject(2543.4, 1959.1, FLOOR)                     # production: accepted
    assert reject(2410.9, 1959.1, FLOOR)                         # run, gate off: rejected
    monkeypatch.setenv(ON, '1')
    scale = ssf._resolve_satstar_gate_core_scale('f480m', 'NRCB5', hybrid)
    assert scale == 0.955
    assert not reject(2410.9, 1959.1, FLOOR, scale)              # run, gate scaled: accepted
    assert reject(2380.0, 1959.1, FLOOR, scale)                  # below 0.955 * 2500 = 2387.5: still rejected
    # the observed-peak second chance is data: it keeps the unscaled half floor
    assert reject(2380.0, 2450.0, FLOOR, scale)
    assert not reject(2380.0, 2500.0, FLOOR, scale)


def test_on_scales_only_a_psf_that_carries_the_core(monkeypatch, hybrid):
    monkeypatch.setenv(ON, '1')
    assert ssf.satstar_fit_switches()['gate_epsf_core_scale'] is True
    assert ssf._resolve_satstar_gate_core_scale('f480m', 'NRCB5', hybrid) == 0.955
    assert ssf._resolve_satstar_gate_core_scale('F480M', 'nrcb5', hybrid) == 0.955
    assert ssf._resolve_satstar_gate_core_scale('f480m', 'NRCA5', hybrid) == 0.976
    # the per-source cutout copy shares the grid meta
    cut = ssf.psf_in_cutout_coords(hybrid, 100, 200)
    assert ssf._psf_carries_epsf_core(cut)
    assert ssf._resolve_satstar_gate_core_scale('f480m', 'NRCB5', cut) == 0.955
    # a plain STPSF grid (no core for the detector, or PSF_EPSF_CORE_DIR unset)
    plain = _stpsf_grid()
    assert not ssf._psf_carries_epsf_core(plain)
    assert ssf._resolve_satstar_gate_core_scale('f480m', 'NRCB5', plain) == 1.0
    # no PSF given: the table entry (the cache signature has no PSF in hand)
    assert ssf._resolve_satstar_gate_core_scale('f480m', 'NRCB5', None) == 0.955


def test_bands_and_detectors_without_an_entry_keep_the_gate(monkeypatch, hybrid):
    monkeypatch.setenv(ON, '1')
    assert set(ssf.SAT_GATE_EPSF_CORE_SCALE) == {('f480m', 'NRCA5'), ('f480m', 'NRCB5')}
    for band, det in (('f212n', 'NRCA1'), ('f212n', 'NRCB4'), ('f480m', 'NRCB1'),
                      ('f480m', None), ('f182m', 'NRCB5')):
        assert ssf._resolve_satstar_gate_core_scale(band, det, hybrid) == 1.0


def test_value_replaces_the_table(monkeypatch, hybrid):
    monkeypatch.setenv(ON, '1')
    monkeypatch.setenv(VALUE, '0.9')
    assert ssf._resolve_satstar_gate_core_scale('f480m', 'NRCB5', hybrid) == 0.9
    assert ssf._resolve_satstar_gate_core_scale('f212n', 'NRCA1', hybrid) == 0.9
    assert ssf._resolve_satstar_gate_core_scale('f480m', None, hybrid) == 0.9
    # a PSF without the core still keeps the unscaled gate
    assert ssf._resolve_satstar_gate_core_scale('f480m', 'NRCB5', _stpsf_grid()) == 1.0
    monkeypatch.setenv(VALUE, '')
    assert ssf._resolve_satstar_gate_core_scale('f480m', 'NRCB5', hybrid) == 0.955


@pytest.mark.parametrize('bad', ['abc', '0', '-0.5', 'nan', 'inf', '1,2'])
def test_unreadable_value_raises(monkeypatch, hybrid, bad):
    monkeypatch.setenv(ON, '1')
    monkeypatch.setenv(VALUE, bad)
    with pytest.raises(ValueError, match=VALUE):
        ssf._resolve_satstar_gate_core_scale('f480m', 'NRCB5', hybrid)


def test_unreadable_switch_raises(monkeypatch, hybrid):
    monkeypatch.setenv(ON, 'maybe')
    with pytest.raises(ValueError, match=ON):
        ssf._resolve_satstar_gate_core_scale('f480m', 'NRCB5', hybrid)
    with pytest.raises(ValueError, match=ON):
        ssf.satstar_fit_switches()


def test_cache_key_follows_the_effective_scale(monkeypatch, cores, tmp_path):
    fb = _frame(tmp_path, 'F480M', 'NRCBLONG', 'b_crf.fits')
    fa = _frame(tmp_path, 'F480M', 'NRCALONG', 'a_crf.fits')
    f2 = _frame(tmp_path, 'F212N', 'NRCB1', 'c_crf.fits')
    off = {eh.EPSF_CORE_DIR_ENV: cores}
    on = {eh.EPSF_CORE_DIR_ENV: cores, ON: '1'}
    on_no_core = {ON: '1'}

    def parts(fn, env):
        return ssf.satstar_fit_switch_signature(fn, env=env).split('_')

    assert 'gs0.955' in parts(fb, on)
    assert 'gs0.976' in parts(fa, on)
    assert parts(fb, on) != parts(fb, off)
    # no scale for the frame -> the key of a run that never exported the switch
    assert parts(f2, on) == parts(f2, off)
    assert parts(fb, on_no_core) == parts(fb, {})
    assert 'gs0.9' in parts(fb, {**on, VALUE: '0.9'})
    assert parts(fb, {**on, VALUE: '1.0'}) == parts(fb, off)


def test_docstring_and_wiring():
    doc = ssf.satstar_fit_switches.__doc__
    assert ON in doc and VALUE in doc
    src = inspect.getsource(ssf.get_saturated_stars)
    assert 'satstar_implied_peak_gate_rejects(' in src
    assert '_resolve_satstar_gate_core_scale(' in src
    assert '_implied < 0.5 * float(_sev_floor)' not in src
