"""Opt-in uncapped satstar catalog flux (NIRCAM_SATSTAR_UNCAPPED_FLUX_FILTERS,
#1186): listed bandpasses report ``flux_fit_precap`` (wing-calibrated) in the
catalog; the model amplitude columns stay capped."""
import numpy as np
from astropy.io import fits
from astropy.table import Table

import jwst_gc_pipeline.reduction.saturated_star_finding as SSF
from jwst_gc_pipeline.reduction.saturated_star_finding import (
    apply_uncapped_flux, satstar_fit_switch_signature,
    satstar_uncapped_flux_filters)

ENV = 'NIRCAM_SATSTAR_UNCAPPED_FLUX_FILTERS'


def _tab():
    """Row 0 capped, row 1 uncapped, row 2 capped but out of field, row 3 NaN
    pre-cap.  flux_fit = flux_fit_raw / wingcal_ratio as apply_wing_selfcal
    leaves it."""
    raw = np.array([100., 200., 300., 400.])
    ratio = np.array([1.25, 1.0, 1.0, 1.1])
    return Table({
        'flux_fit': raw / ratio,
        'flux_err': np.array([10., 20., 30., 40.]) / ratio,
        'flux_fit_raw': raw,
        'wingcal_ratio': ratio,
        'flux_fit_precap': np.array([150., 200., 450., np.nan]),
        'outside_fov_seed': np.array([False, False, True, False]),
    })


def test_list_parsing():
    assert satstar_uncapped_flux_filters({}) == frozenset()
    assert satstar_uncapped_flux_filters({ENV: '  '}) == frozenset()
    got = satstar_uncapped_flux_filters({ENV: 'f323n, F405N  f466n,'})
    assert got == {'F323N', 'F405N', 'F466N'}


def test_default_is_noop_and_schema_unchanged():
    t = _tab()
    ref = t.copy()
    out = apply_uncapped_flux(t, 'F323N', env={})
    assert out.colnames == ref.colnames
    for c in ref.colnames:
        np.testing.assert_array_equal(out[c], ref[c])


def test_unlisted_band_untouched():
    out = apply_uncapped_flux(_tab(), 'F212N', env={ENV: 'F323N'})
    assert 'flux_uncapped' not in out.colnames
    np.testing.assert_allclose(out['flux_fit'], [80., 200., 300., 400 / 1.1])


def test_replaces_only_capped_rows_of_listed_band():
    ref = _tab()
    out = apply_uncapped_flux(_tab(), 'f323n', env={ENV: 'F323N'})
    assert list(out['flux_uncapped']) == [True, False, False, False]
    np.testing.assert_allclose(out['flux_fit'][0], 150. / 1.25)
    # flux_err scales by the factor flux_fit changed (precap / raw = 1.5)
    np.testing.assert_allclose(out['flux_err'][0], ref['flux_err'][0] * 1.5)
    np.testing.assert_allclose(out['flux_err'][0] / out['flux_fit'][0],
                               ref['flux_err'][0] / ref['flux_fit'][0])
    # uncapped, out-of-field and NaN-precap rows are unchanged
    for i in (1, 2, 3):
        assert out['flux_fit'][i] == ref['flux_fit'][i]
        assert out['flux_err'][i] == ref['flux_err'][i]


def test_model_amplitudes_untouched():
    ref = _tab()
    out = apply_uncapped_flux(_tab(), 'F323N', env={ENV: 'F323N'})
    np.testing.assert_array_equal(out['flux_fit_raw'], ref['flux_fit_raw'])
    np.testing.assert_array_equal(out['flux_fit_precap'], ref['flux_fit_precap'])
    np.testing.assert_array_equal(out['wingcal_ratio'], ref['wingcal_ratio'])


def test_without_wingcal_columns():
    t = _tab()
    t.remove_columns(['flux_fit_raw', 'wingcal_ratio'])
    t['flux_fit'] = [100., 200., 300., 400.]
    out = apply_uncapped_flux(t, 'F323N', env={ENV: 'F323N'})
    np.testing.assert_allclose(out['flux_fit'][0], 150.)
    assert list(out['flux_uncapped']) == [True, False, False, False]


def test_none_table():
    assert apply_uncapped_flux(None, 'F323N', env={ENV: 'F323N'}) is None


def _hdr(filt, pupil):
    return fits.Header({'FILTER': filt, 'PUPIL': pupil})


def test_pupil_band_lookup():
    assert SSF._frame_bandpass(_hdr('F444W', 'F405N')) == 'F405N'
    assert SSF._frame_bandpass(_hdr('F150W2', 'F162M')) == 'F162M'
    assert SSF._frame_bandpass(_hdr('F323N', 'CLEAR')) == 'F323N'
    assert SSF._frame_bandpass(_hdr('CLEAR', 'CLEAR')) == ''


def _write(path, filt, pupil):
    fits.PrimaryHDU(header=_hdr(filt, pupil)).writeto(path, overwrite=True)
    return str(path)


def test_signature_only_for_listed_bandpass(tmp_path):
    pupil_frame = _write(tmp_path / 'a_cal.fits', 'F444W', 'F405N')
    other = _write(tmp_path / 'b_cal.fits', 'F444W', 'F466N')
    env = {ENV: 'f405n'}
    base = satstar_fit_switch_signature(pupil_frame, env={})
    sig = satstar_fit_switch_signature(pupil_frame, env=env)
    assert sig == '_'.join(p for p in (base, 'uf') if p)
    assert satstar_fit_switch_signature(other, env=env) == \
        satstar_fit_switch_signature(other, env={})
