"""Opt-in F150W wing multiplier (SATSTAR_WING_MULT_FILTERS, #1036): listed
bandpasses divide the satstar catalog flux by a ratio ramped over r_mask;
the model amplitude columns and ``wingcal_ratio`` stay as they were."""
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from photutils.psf import CircularGaussianPRF

from jwst_gc_pipeline.reduction.tests.test_satstar_recovered_cap import (
    _two_star_frame)

import jwst_gc_pipeline.reduction.satstar_deblend as SD
import jwst_gc_pipeline.reduction.saturated_star_finding as SSF
from jwst_gc_pipeline.photometry.wingcal import (
    WING_MULT_TABLE, wing_multiplier_ratio)
from jwst_gc_pipeline.reduction.saturated_star_finding import (
    apply_wing_multiplier, satstar_fit_switch_signature,
    satstar_wing_mult_filters)

ENV = 'SATSTAR_WING_MULT_FILTERS'
FULL = 10 ** (0.4 * 0.275)


def _tab():
    """r_mask 0.564 (1 px core), 3.5, 4.25, 5, 8, NaN, and an out-of-field
    row at 8 px."""
    rmask = np.array([0.564, 3.5, 4.25, 5.0, 8.0, np.nan, 8.0])
    raw = np.array([100., 200., 300., 400., 500., 600., 700.])
    ratio = np.array([1.0, 1.0, 1.0, 1.05, 1.0, 1.0, 1.0])
    return Table({
        'flux_fit': raw / ratio,
        'flux_err': raw / 10. / ratio,
        'flux_fit_raw': raw,
        'wingcal_ratio': ratio,
        'wingcal_rmask': rmask,
        'outside_fov_seed': np.array([False] * 6 + [True]),
    })


def test_ramp_shape():
    r = np.array([0.0, 0.564, 3.0, 3.5, 4.25, 5.0, 8.0, 40.0, np.nan])
    got = wing_multiplier_ratio(r, 'f150w')
    np.testing.assert_allclose(got[:4], 1.0)
    np.testing.assert_allclose(got[4], 0.5 * (1.0 + FULL))
    np.testing.assert_allclose(got[5:8], FULL)
    assert got[8] == 1.0
    finite = got[:8]
    assert (np.diff(finite) >= 0).all()
    assert FULL == pytest.approx(1.2886, abs=1e-3)


def test_unknown_bandpass_raises():
    with pytest.raises(KeyError):
        wing_multiplier_ratio([5.0], 'F200W')


def test_list_parsing():
    assert satstar_wing_mult_filters({}) == frozenset()
    assert satstar_wing_mult_filters({ENV: '  '}) == frozenset()
    assert satstar_wing_mult_filters({ENV: 'f150w,'}) == {'F150W'}
    with pytest.raises(ValueError, match='F200W'):
        satstar_wing_mult_filters({ENV: 'F150W F200W'})
    assert set(WING_MULT_TABLE) == {'F150W'}


def test_default_is_noop_and_schema_unchanged():
    t = _tab()
    ref = t.copy()
    out = apply_wing_multiplier(t, 'F150W', env={})
    assert out.colnames == ref.colnames
    for c in ref.colnames:
        np.testing.assert_array_equal(out[c], ref[c])


def test_unlisted_band_untouched():
    ref = _tab()
    out = apply_wing_multiplier(_tab(), 'F200W', env={ENV: 'F150W'})
    assert 'wingmult_ratio' not in out.colnames
    np.testing.assert_array_equal(out['flux_fit'], ref['flux_fit'])


def test_listed_band_divides_catalog_flux_only():
    ref = _tab()
    out = apply_wing_multiplier(_tab(), 'f150w', env={ENV: 'F150W'})
    expect = np.array([1.0, 1.0, 0.5 * (1.0 + FULL), FULL, FULL, 1.0, 1.0])
    np.testing.assert_allclose(out['wingmult_ratio'], expect)
    np.testing.assert_allclose(out['flux_fit'], ref['flux_fit'] / expect)
    # flux_err scales with flux_fit, so the S/N of a row is kept.
    np.testing.assert_allclose(out['flux_err'] / out['flux_fit'],
                               ref['flux_err'] / ref['flux_fit'])
    # The audit trail and the model amplitudes are not touched, and the
    # multiplier stays out of wingcal_ratio (the pooled-fallback sentinel).
    for c in ('flux_fit_raw', 'wingcal_ratio', 'wingcal_rmask'):
        np.testing.assert_array_equal(out[c], ref[c])
    # NaN r_mask and the out-of-field row keep their flux.
    assert out['flux_fit'][5] == ref['flux_fit'][5]
    assert out['flux_fit'][6] == ref['flux_fit'][6]


def test_without_rmask_column_records_unit_ratio():
    t = _tab()
    t.remove_column('wingcal_rmask')
    ref = t.copy()
    out = apply_wing_multiplier(t, 'F150W', env={ENV: 'F150W'})
    np.testing.assert_array_equal(out['flux_fit'], ref['flux_fit'])
    np.testing.assert_array_equal(out['wingmult_ratio'], np.ones(len(ref)))


def test_none_table():
    assert apply_wing_multiplier(None, 'F150W', env={ENV: 'F150W'}) is None


def _write(path, filt):
    fits.PrimaryHDU(header=fits.Header({'FILTER': filt, 'PUPIL': 'CLEAR'})
                    ).writeto(path, overwrite=True)
    return str(path)


def test_signature_only_for_listed_bandpass(tmp_path):
    f150 = _write(tmp_path / 'a_cal.fits', 'F150W')
    f200 = _write(tmp_path / 'b_cal.fits', 'F200W')
    env = {ENV: 'F150W'}
    base = satstar_fit_switch_signature(f150, env={})
    sig = satstar_fit_switch_signature(f150, env=env)
    assert sig == '_'.join(p for p in (base, 'wm') if p)
    assert satstar_fit_switch_signature(f200, env=env) == \
        satstar_fit_switch_signature(f200, env={})


class _EvalPRF(CircularGaussianPRF):
    """Gaussian PRF with the ``evaluate(x, y, flux, x0, y0)`` signature of a
    gridded PSF model, which the satstar model rendering calls."""

    def evaluate(self, x, y, flux, x0, y0, fwhm=None):
        return super().evaluate(x, y, flux, x0, y0,
                                self.fwhm.value if fwhm is None else fwhm)


def test_get_saturated_stars_applies_multiplier_after_uncapped_flux(
        monkeypatch):
    """Wiring (#1036): ``get_saturated_stars`` calls ``apply_wing_multiplier``
    once, after ``apply_uncapped_flux``.  The uncapped-flux step rewrites
    ``flux_fit`` from ``flux_fit_precap``, so a multiplier applied before it
    would be overwritten.  The uncapped step is replaced by a rewrite of
    ``flux_fit`` to ``2 * flux_fit_raw`` and the catalog flux must come out
    divided by the multiplier."""
    monkeypatch.setenv(ENV, 'F150W')
    fh, sat = _two_star_frame()
    n = sat.shape[0]

    def _records(saturated, sources, coms, sizes, zeroframe, data, fwhm,
                 **kw):
        return [{'com': c, 'label': 1, 'forced': False,
                 'sat_area': int(sat.sum())}
                for c in [(100.0, 93.0), (100.0, 107.0)]]

    def _anchor(data, dq, zeroframe, **kw):
        return data, sat.copy(), np.zeros_like(sat), None

    calls = []
    _wm = SSF.apply_wing_multiplier

    def _uncapped(tab, filtername, env=None):
        calls.append('uncapped')
        tab['flux_fit'] = 2.0 * np.asarray(tab['flux_fit_raw'], dtype=float)
        return tab

    def _wing_mult(tab, filtername, env=None):
        calls.append('wing_mult')
        return _wm(tab, filtername, env=env)

    monkeypatch.setattr(SD, 'build_deblended_source_records', _records)
    monkeypatch.setattr(SSF, 'zeroframe_fit_anchor', _anchor)
    monkeypatch.setattr(SSF, 'get_psf',
                        lambda header, **kw: _EvalPRF(fwhm=1.6))
    monkeypatch.setattr(SSF.stpsf, 'NIRCam', lambda *a, **kw: object())
    monkeypatch.setattr(SSF, 'apply_uncapped_flux', _uncapped)
    monkeypatch.setattr(SSF, 'apply_wing_multiplier', _wing_mult)
    tab = SSF.get_saturated_stars(fh, zeroframe=np.ones((n, n)),
                                  zeroframe_deblend=True, plot=False)
    assert tab is not None and len(tab)
    assert calls == ['uncapped', 'wing_mult']
    ratio = np.asarray(tab['wingmult_ratio'], dtype=float)
    assert (ratio >= 1.0).all()
    ok = ~np.asarray(tab['outside_fov_seed'], dtype=bool) \
        if 'outside_fov_seed' in tab.colnames else np.ones(len(tab), bool)
    np.testing.assert_allclose(
        np.asarray(tab['flux_fit'], dtype=float)[ok],
        2.0 * np.asarray(tab['flux_fit_raw'], dtype=float)[ok] / ratio[ok])
