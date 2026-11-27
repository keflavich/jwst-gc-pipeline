"""SATSTAR_EDGE_POS_BOUND: a saturated component within SATSTAR_EDGE_TOUCH_PX of
the frame edge gets a wider position bound around its seed.

The 4-px reference-pixel border (DO_NOT_USE, NaN in SCI) cuts the core of a
bright star at the detector edge, so the centre of mass of the visible
saturated pixels sits several pixels inside the star.  The nominal bound,
``max(size_saturated, 1.5 FWHM)``, then pins the fitted position at its edge
(o138 F212N nrca1: ``x_fit`` = ``x_init`` - 5.000) and the flux compensates
3x low.
"""
import numpy as np
import pytest
from astropy.io import fits

import jwst_gc_pipeline.reduction.satstar_deblend as SD
import jwst_gc_pipeline.reduction.saturated_star_finding as SSF
from jwst_gc_pipeline.reduction.saturated_star_finding import (
    SATSTAR_EDGE_MIN_AREA_DEFAULT, SATSTAR_EDGE_TOUCH_PX,
    component_edge_distance, satstar_edge_min_area, satstar_edge_pos_bound,
    satstar_fit_switch_signature, satstar_fit_switches,
    widened_position_bound)
from jwst_gc_pipeline.reduction.tests.test_satstar_recovered_cap import (
    _two_star_frame)
from jwst_gc_pipeline.reduction.tests.test_satstar_wing_multiplier import (
    _EvalPRF)

ENV = 'SATSTAR_EDGE_POS_BOUND'
AREA_ENV = 'SATSTAR_EDGE_MIN_AREA'


def test_default_is_off():
    assert satstar_edge_pos_bound({}) == 0.0
    assert satstar_edge_pos_bound({ENV: ''}) == 0.0
    assert satstar_edge_pos_bound({ENV: ' '}) == 0.0
    assert satstar_edge_pos_bound({ENV: '0'}) == 0.0
    assert satstar_fit_switches({})['edge_pos_bound'] == 0.0


def test_a_number_is_read():
    assert satstar_edge_pos_bound({ENV: '12'}) == 12.0
    assert satstar_edge_pos_bound({ENV: ' 7.5 '}) == 7.5
    assert satstar_fit_switches({ENV: '20'})['edge_pos_bound'] == 20.0


@pytest.mark.parametrize('raw', ['abc', '-1', 'nan', 'inf', '1,5'])
def test_anything_else_raises_naming_the_variable(raw):
    with pytest.raises(ValueError, match=ENV):
        satstar_edge_pos_bound({ENV: raw})


@pytest.mark.parametrize('rows, cols, expect', [
    (slice(529, 532), slice(3, 6), 3),          # left edge, as o138 nrca1 exp5
    (slice(10, 20), slice(30, 40), 10),         # top edge of the array
    (slice(100, 2043), slice(100, 200), 5),     # rows stop 5 px from row 2048
    (slice(100, 200), slice(2000, 2048), 0),    # on the last column
    (slice(0, 5), slice(500, 510), 0),          # on the first row
    (slice(900, 910), slice(900, 910), 900),    # interior
])
def test_component_edge_distance(rows, cols, expect):
    assert component_edge_distance((rows, cols), (2048, 2048)) == expect


def test_a_component_without_a_slice_has_no_distance():
    assert component_edge_distance(None, (2048, 2048)) is None
    assert component_edge_distance((None, None), (2048, 2048)) is None


def test_widened_position_bound():
    near, far = SATSTAR_EDGE_TOUCH_PX, SATSTAR_EDGE_TOUCH_PX + 1
    assert widened_position_bound(5.0, near, 12.0) == 12.0
    assert widened_position_bound(5.0, 0, 12.0) == 12.0
    # never narrower than the nominal bound
    assert widened_position_bound(15.0, near, 12.0) == 15.0
    # switch off, component not at the edge, or distance unknown
    assert widened_position_bound(5.0, near, 0.0) == 5.0
    assert widened_position_bound(5.0, far, 12.0) == 5.0
    assert widened_position_bound(5.0, None, 12.0) == 5.0


def test_widened_position_bound_needs_the_saturated_area():
    near = SATSTAR_EDGE_TOUCH_PX
    assert widened_position_bound(5.0, near, 12.0, 141, 100) == 12.0
    assert widened_position_bound(5.0, near, 12.0, 100, 100) == 12.0
    assert widened_position_bound(5.0, near, 12.0, 99, 100) == 5.0
    # an unknown area does not pass a floor, and passes no floor
    assert widened_position_bound(5.0, near, 12.0, None, 100) == 5.0
    assert widened_position_bound(5.0, near, 12.0, None, 0) == 12.0


def test_min_area_default_and_parsing():
    assert SATSTAR_EDGE_MIN_AREA_DEFAULT == 100
    assert satstar_edge_min_area({}) == 100
    assert satstar_edge_min_area({AREA_ENV: ' '}) == 100
    assert satstar_edge_min_area({AREA_ENV: '0'}) == 0
    assert satstar_edge_min_area({AREA_ENV: ' 50 '}) == 50
    assert satstar_fit_switches({})['edge_min_area'] == 100
    assert satstar_fit_switches({AREA_ENV: '50'})['edge_min_area'] == 50


@pytest.mark.parametrize('raw', ['abc', '-1', '1.5', '1e2'])
def test_min_area_anything_else_raises_naming_the_variable(raw):
    with pytest.raises(ValueError, match=AREA_ENV):
        satstar_edge_min_area({AREA_ENV: raw})


def _frame(tmp_path):
    path = tmp_path / 'a_cal.fits'
    fits.PrimaryHDU(header=fits.Header({'FILTER': 'F200W', 'PUPIL': 'CLEAR'})
                    ).writeto(path, overwrite=True)
    return str(path)


def test_cache_key_is_unchanged_when_off_and_records_the_bound(tmp_path):
    path = _frame(tmp_path)
    base = satstar_fit_switch_signature(path, env={})
    assert satstar_fit_switch_signature(path, env={ENV: '0'}) == base
    sig = satstar_fit_switch_signature(path, env={ENV: '12'})
    assert sig == '_'.join(p for p in (base, 'eb12') if p)
    assert satstar_fit_switch_signature(path, env={ENV: '20'}) != sig
    both = satstar_fit_switch_signature(
        path, env={ENV: '12', 'SATSTAR_FITTER_MAXITERS': '1000'})
    assert both == '_'.join(p for p in (base, 'mi1000', 'eb12') if p)


def test_cache_key_records_a_non_default_area_floor_only_with_the_bound(tmp_path):
    path = _frame(tmp_path)
    base = satstar_fit_switch_signature(path, env={})
    # the default floor stays out of the key; a changed one enters it
    assert satstar_fit_switch_signature(
        path, env={ENV: '12', AREA_ENV: '100'}) == '_'.join(
            p for p in (base, 'eb12') if p)
    assert satstar_fit_switch_signature(
        path, env={ENV: '12', AREA_ENV: '50'}) == '_'.join(
            p for p in (base, 'eb12a50') if p)
    # with the bound off the floor changes nothing, so it leaves the key alone
    assert satstar_fit_switch_signature(path, env={AREA_ENV: '50'}) == base


def _x_bounds(monkeypatch, env, edge_distance):
    """``x_0`` bounds the satstar fit sets on each model it fits, with
    ``component_edge_distance`` stubbed to ``edge_distance``."""
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    fh, sat = _two_star_frame()
    n = sat.shape[0]
    seen = []

    class _Recorder(SSF.PSFPhotometry):
        def __call__(self, *args, **kw):
            seen.append(tuple(self.psf_model.x_0.bounds))
            return super().__call__(*args, **kw)

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
    monkeypatch.setattr(SSF, 'component_edge_distance',
                        lambda comp_slice, shape: edge_distance)
    SSF.get_saturated_stars(fh, zeroframe=np.ones((n, n)),
                            zeroframe_deblend=True, plot=False)
    return seen


def _half_widths(bounds):
    return {round(0.5 * (hi - lo), 6) for lo, hi in bounds}


def test_the_fit_keeps_the_nominal_bound_when_off(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    seen = _x_bounds(monkeypatch, {}, edge_distance=0)
    assert seen, "the satstar fit set no position bounds"
    assert max(_half_widths(seen)) < 30.0


def test_an_edge_clipped_component_gets_the_wider_bound(monkeypatch):
    seen = _x_bounds(monkeypatch, {ENV: '30'}, edge_distance=2)
    assert seen, "the satstar fit set no position bounds"
    assert _half_widths(seen) == {30.0}


def test_an_interior_component_keeps_the_nominal_bound(monkeypatch):
    seen = _x_bounds(monkeypatch, {ENV: '30'}, edge_distance=87)
    assert seen, "the satstar fit set no position bounds"
    assert max(_half_widths(seen)) < 30.0


def test_a_component_below_the_area_floor_keeps_the_nominal_bound(monkeypatch):
    # the two-star component holds 225 saturated pixels
    seen = _x_bounds(monkeypatch, {ENV: '30', AREA_ENV: '1000'},
                     edge_distance=2)
    assert seen, "the satstar fit set no position bounds"
    assert max(_half_widths(seen)) < 30.0
    seen = _x_bounds(monkeypatch, {ENV: '30', AREA_ENV: '200'},
                     edge_distance=2)
    assert _half_widths(seen) == {30.0}
