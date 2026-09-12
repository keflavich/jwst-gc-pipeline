"""Band choice and quality cuts for the release pages' colour diagrams.

The requested diagram is F405N-F466N against F182M-F210M.  Naming those
filters literally would draw it for almost no field -- the GC programmes use
F212N where that says F210M and F480M where it says F466N -- so bands are
chosen by wavelength slot.  These pin the choice each released field actually
gets, because a diagram with the wrong axes is worse than none: it looks right.
"""
import importlib.util
import os

import numpy as np
import pytest
from astropy.table import Table

_MOD = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'scripts', 'release', 'make_ccd_plot.py'))
_spec = importlib.util.spec_from_file_location('make_ccd_plot_t', _MOD)
cc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cc)


def _sorted(bands):
    return sorted(bands, key=lambda b: (cc._WAVE.get(b, 99.0), b))


# --- which diagram each released field gets ---------------------------------

@pytest.mark.parametrize('field,bands,kind,xlabel,ylabel', [
    # The requested diagram, in the two spellings the survey actually uses.
    ('brick/cloudc', ['F182M', 'F187N', 'F212N', 'F405N', 'F410M', 'F466N'],
     'ccd', 'F182M - F212N', 'F405N - F466N'),
    # w51 has the literal filters named in the request.
    ('w51', ['F140M', 'F162M', 'F182M', 'F187N', 'F210M', 'F335M', 'F360M',
             'F405N', 'F410M', 'F480M'],
     'ccd', 'F182M - F210M', 'F405N - F480M'),
    # "roughly H-K vs 400-480" for a field with neither F182M nor F405N.
    ('cloudef_controlfield', ['F162M', 'F210M', 'F360M', 'F480M'],
     'ccd', 'F162M - F210M', 'F360M - F480M'),
    # Three bands, no LW pair: adjacent colours.
    ('sgra', ['F115W', 'F212N', 'F405N'], 'ccd', 'F115W - F212N', 'F212N - F405N'),
    ('sgrb2', ['F300M', 'F360M', 'F410M'], 'ccd', 'F300M - F360M', 'F360M - F410M'),
    ('sgrc', ['F162M', 'F405N', 'F480M'], 'ccd', 'F162M - F405N', 'F405N - F480M'),
    # Two bands: colour-magnitude, and the magnitude is the REDDER band.
    ('gc2211', ['F200W', 'F277W'], 'cmd', 'F200W - F277W', 'F277W'),
    ('quintuplet', ['F212N', 'F323N'], 'cmd', 'F212N - F323N', 'F323N'),
])
def test_each_released_field_gets_the_right_axes(field, bands, kind,
                                                 xlabel, ylabel):
    got_kind, _used, got_x, got_y = cc.choose_axes(_sorted(bands))
    assert (got_kind, got_x, got_y) == (kind, xlabel, ylabel), field


def test_one_band_draws_nothing():
    """Better no diagram than an axis against itself."""
    assert cc.choose_axes(['F212N'])[0] is None
    assert cc.choose_axes([])[0] is None


def test_a_single_lw_band_cannot_satisfy_both_halves_of_a_pair():
    """F444W is in the lw_red list, and an earlier cut of `choose_axes` let a
    field whose only LW band was F444W plot F444W - F444W, which is zero
    everywhere and looks like a real flat locus."""
    kind, used, x, y = cc.choose_axes(_sorted(['F182M', 'F212N', 'F444W']))
    assert x != y
    if kind == 'ccd':
        xb, xr, yb, yr = used
        assert xb != xr and yb != yr


def test_the_bluest_band_goes_on_the_left_of_every_colour():
    """A colour written red-minus-blue is the negative of the convention and
    silently flips the extinction sequence."""
    for bands in (['F182M', 'F212N', 'F405N', 'F466N'],
                  ['F115W', 'F212N', 'F405N'],
                  ['F200W', 'F277W']):
        kind, used, _x, _y = cc.choose_axes(_sorted(bands))
        pairs = [used[0:2], used[2:4]] if kind == 'ccd' else [used]
        for blue, red in pairs:
            assert cc._WAVE[blue] < cc._WAVE[red], (bands, blue, red)


# --- the cuts ----------------------------------------------------------------

def _table(n=100, snr=None, mag_nan=()):
    """A merged-catalog stand-in for two bands."""
    t = Table()
    snr = snr if snr is not None else {}
    for band in ('f182m', 'f212n'):
        t[f'mag_vega_{band}'] = np.full(n, 15.0)
        s = np.full(n, float(snr.get(band, 50.0)))
        t[f'flux_{band}'] = s
        t[f'flux_err_{band}'] = np.ones(n)
    for band, idx in mag_nan:
        col = np.array(t[f'mag_vega_{band}'], dtype=float)
        col[idx] = np.nan
        t[f'mag_vega_{band}'] = col
    return t


def test_a_source_must_clear_the_snr_floor_in_every_band():
    """S/N in one band and noise in the other is not a colour."""
    t = _table(snr={'f182m': 50.0, 'f212n': 3.0})
    _mags, keep = cc.select(t, ('f182m', 'f212n'.upper()))
    assert keep.sum() == 0
    t = _table(snr={'f182m': 50.0, 'f212n': 50.0})
    _mags, keep = cc.select(t, ('F182M', 'F212N'))
    assert keep.sum() == 100


def test_a_source_missing_one_band_is_dropped():
    """A row exists in the merged table if it was seen in ANY band, so
    'matched in N bands' has to be enforced here rather than assumed."""
    t = _table(mag_nan=[('f212n', slice(0, 40))])
    _mags, keep = cc.select(t, ('F182M', 'F212N'))
    assert keep.sum() == 60


def test_zero_flux_error_does_not_pass_as_infinite_signal():
    t = _table()
    t['flux_err_f212n'] = np.zeros(len(t))
    _mags, keep = cc.select(t, ('F182M', 'F212N'))
    assert keep.sum() == 0


def test_the_snr_floor_is_the_documented_one():
    assert cc.SNR_MIN == 10.0


# --- picking which staged table to draw --------------------------------------

def _rank(token):
    import re as _re
    m = _re.fullmatch(r'(resbgsub_)?m(\d+)', token)
    return None if m is None else 10 * int(m.group(2)) + (1 if m.group(1) else 0)


def test_two_merge_iterations_of_one_field_are_not_two_pointings():
    """brick v1.0 stages `_m7` beside `_m8`.  Drawing both wrote one filename
    twice and kept whichever finished last -- a coin toss over which iteration
    the page showed."""
    got = cc.select_tables(
        ['/r/basic_merged_x_resbgsub_m7.fits',
         '/r/basic_merged_x_resbgsub_m8.fits'], _rank)
    assert [os.path.basename(p) for _o, p in got] == \
        ['basic_merged_x_resbgsub_m8.fits']


def test_per_observation_tables_are_each_drawn():
    """brick v1.7 stages o001 (2221 bands) and o004 (1182 bands).  They are
    different sky AND different filters; picking one by file size chose the one
    whose bands are not the requested diagram."""
    got = cc.select_tables(
        ['/r/basic_merged_x_resbgsub_m8_o001.fits',
         '/r/basic_merged_x_resbgsub_m8_o004.fits'], _rank)
    assert [o for o, _p in got] == ['o001', 'o004']


def test_the_untokened_pooled_table_is_dropped_when_per_obs_ones_exist():
    """gc2211 stages a combined table beside its per-pointing ones; drawing it
    too would put several pointings' loci on one axis without saying so."""
    got = cc.select_tables(
        ['/r/basic_merged_x_resbgsub_m7.fits',
         '/r/basic_merged_x_resbgsub_m7_o023.fits',
         '/r/basic_merged_x_resbgsub_m7_o050.fits'], _rank)
    assert [o for o, _p in got] == ['o023', 'o050']


def test_an_untokened_table_alone_is_kept():
    got = cc.select_tables(['/r/basic_merged_x_resbgsub_m8.fits'], _rank)
    assert [o for o, _p in got] == [None]
