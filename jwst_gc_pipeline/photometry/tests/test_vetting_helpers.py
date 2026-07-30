"""The three blocks pulled out of run_manual_pipeline.

Each was nested 8-10 levels deep inside the phase/module/filter loops, so none
of them could be tested.
"""
import types

import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry import cataloging as C


def _vetted(n, ra0=266.5):
    t = Table()
    t['skycoord'] = SkyCoord(ra0 + np.arange(n) * 0.01,
                             -28.7 + np.zeros(n), unit='deg')
    t['flux'] = np.linspace(10.0, 20.0, n)
    t['qfit'] = np.full(n, 0.1)
    return t


def test_carta_export_has_plain_float_coordinates(tmp_path):
    # CARTA cannot read the SkyCoord mixin column.
    out = tmp_path / 'cat_carta.fits'
    C._write_carta_catalog(_vetted(3), str(out))
    got = Table.read(out)
    assert got['ra'].dtype.kind == 'f' and got['ra'].dtype.itemsize == 8
    assert got['dec'].dtype.kind == 'f'
    assert 'flux' in got.colnames and 'qfit' in got.colnames
    assert 'skycoord' not in got.colnames


def test_carta_export_accepts_plain_radec_input(tmp_path):
    t = Table({'ra': [266.5, 266.6], 'dec': [-28.7, -28.8], 'flux': [1.0, 2.0]})
    out = tmp_path / 'c_carta.fits'
    C._write_carta_catalog(t, str(out))
    assert len(Table.read(out)) == 2


def test_combine_gathers_every_obs(tmp_path):
    merged = tmp_path / 'cat.fits'
    _vetted(1).write(merged, overwrite=True)          # placeholder, not read
    for obs, ra0 in (('001', 266.5), ('002', 267.5)):
        _vetted(3, ra0).write(tmp_path / f'cat_o{obs}_vetted.fits',
                              overwrite=True)
    combined = tmp_path / 'cat_vetted.fits'
    C._combine_per_obs_vetted(str(tmp_path / 'cat_o002_vetted.fits'),
                              str(merged), str(combined), this_obs_only=False)
    assert len(Table.read(combined)) == 6
    assert (tmp_path / 'cat_vetted_carta.fits').exists()


def test_combine_keeps_one_obs_when_asked(tmp_path):
    # A joint run, or gc2211: the `_o*` glob would pick up stale siblings.
    merged = tmp_path / 'cat.fits'
    _vetted(1).write(merged, overwrite=True)
    _vetted(3, 266.5).write(tmp_path / 'cat_o001_vetted.fits', overwrite=True)
    mine = tmp_path / 'cat_o002_vetted.fits'
    _vetted(2, 267.5).write(mine, overwrite=True)
    combined = tmp_path / 'cat_vetted.fits'
    C._combine_per_obs_vetted(str(mine), str(merged), str(combined),
                              this_obs_only=True)
    assert len(Table.read(combined)) == 2


def test_combine_survives_an_unreadable_sibling(tmp_path, capsys):
    merged = tmp_path / 'cat.fits'
    _vetted(1).write(merged, overwrite=True)
    _vetted(3, 266.5).write(tmp_path / 'cat_o001_vetted.fits', overwrite=True)
    (tmp_path / 'cat_o002_vetted.fits').write_text('not a fits file')
    combined = tmp_path / 'cat_vetted.fits'
    C._combine_per_obs_vetted(str(tmp_path / 'cat_o001_vetted.fits'),
                              str(merged), str(combined), this_obs_only=False,
                              label='t')
    assert len(Table.read(combined)) == 3
    assert 'cannot read' in capsys.readouterr().out


def test_combine_writes_nothing_when_there_is_nothing(tmp_path):
    combined = tmp_path / 'cat_vetted.fits'
    C._combine_per_obs_vetted(str(tmp_path / 'missing.fits'),
                              str(tmp_path / 'cat.fits'), str(combined),
                              this_obs_only=False)
    assert not combined.exists()


def test_saturated_pixels_fall_back_to_the_model():
    # Neither recovery method enabled -> saturated pixels come from the model.
    data = np.full((4, 4), 100.0)
    dq = np.zeros((4, 4), dtype=int)
    was_sat = np.zeros((4, 4), dtype=bool)
    was_sat[1, 1] = True
    model = np.full((4, 4), 7.0)
    out = C._fill_saturated_pixels('nofile.fits', data, dq, was_sat, model,
                                   data.copy(), types.SimpleNamespace())
    assert out[1, 1] == 7.0
    assert out[0, 0] == 100.0


def test_saturated_pixels_fall_back_when_there_is_no_ramp():
    # Recovery on, but the frame has no sibling _ramp.fits: use the model,
    # do not raise.
    data = np.full((4, 4), 100.0)
    dq = np.zeros((4, 4), dtype=int)
    was_sat = np.zeros((4, 4), dtype=bool)
    was_sat[2, 2] = True
    model = np.full((4, 4), 5.0)
    options = types.SimpleNamespace(satstar_ramp_recover=True,
                                    satstar_zeroframe_recover=True)
    out = C._fill_saturated_pixels('/nonexistent_nrca1_cal.fits', data, dq,
                                   was_sat, model, data.copy(), options)
    assert out[2, 2] == 5.0


def test_ramp_slope_helper_declines_without_a_ramp():
    assert C._fill_from_ramp_slope(
        '/nonexistent_nrca1_cal.fits', np.zeros((4, 4)), np.zeros((4, 4), int),
        np.zeros((4, 4), bool), np.zeros((4, 4)), np.zeros((4, 4)), 3) is None
