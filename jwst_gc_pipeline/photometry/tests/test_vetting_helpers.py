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


# --- the recovery ladder itself -------------------------------------------
# The tests above only reach the model-only branch.  These stub the ramp
# loaders so the masks (rim / deep / done / was_sat & ~done) are exercised:
# invert one of them and these fail.

CAL = 'frame_nrca1_cal.fits'


def _frame():
    """4x4 frame: pixel (1,1) is a saturated rim, (2,2) a deep core."""
    data = np.full((4, 4), 100.0)
    dq = np.zeros((4, 4), dtype=int)
    was_sat = np.zeros((4, 4), dtype=bool)
    was_sat[1, 1] = was_sat[2, 2] = True
    model = np.full((4, 4), 5.0)
    return data, dq, was_sat, model


def _masks():
    rim = np.zeros((4, 4), dtype=bool)
    rim[1, 1] = True
    deep = np.zeros((4, 4), dtype=bool)
    deep[2, 2] = True
    return rim, deep


def _stub_ramp(monkeypatch, *, slope_ratio=1.0, group0=True, core_ratio=1.0):
    from jwst_gc_pipeline.reduction import saturated_star_finding as SSF
    rim, deep = _masks()
    monkeypatch.setattr(C, '_load_ramp_cube',
                        lambda p: (np.zeros((3, 4, 4)), None))
    monkeypatch.setattr(C, '_load_ramp_group0',
                        lambda p: np.zeros((4, 4)) if group0 else None)
    monkeypatch.setattr(SSF, 'ramp_recover_saturated',
                        lambda *a, **k: (np.full((4, 4), 11.0), rim, deep,
                                         slope_ratio))
    monkeypatch.setattr(SSF, 'zeroframe_recover_saturated',
                        lambda *a, **k: (np.full((4, 4), 22.0), deep, deep,
                                         core_ratio))


def test_ramp_slope_fills_the_rim_and_group0_fills_the_deep_core(monkeypatch):
    _stub_ramp(monkeypatch)
    data, dq, was_sat, model = _frame()
    options = types.SimpleNamespace(satstar_ramp_recover=True)
    out = C._fill_saturated_pixels(CAL, data, dq, was_sat, model,
                                   data.copy(), options)
    assert out[1, 1] == 11.0      # rim -> ramp slope
    assert out[2, 2] == 22.0      # deep core -> group 0
    assert out[0, 0] == 100.0     # untouched


def test_deep_core_falls_back_to_the_model_without_group0(monkeypatch):
    _stub_ramp(monkeypatch, group0=False)
    data, dq, was_sat, model = _frame()
    out = C._fill_saturated_pixels(
        CAL, data, dq, was_sat, model, data.copy(),
        types.SimpleNamespace(satstar_ramp_recover=True))
    assert out[1, 1] == 11.0      # rim still recovered
    assert out[2, 2] == 5.0       # deep core -> model


def test_a_non_finite_scale_rejects_the_whole_ramp_recovery(monkeypatch):
    _stub_ramp(monkeypatch, slope_ratio=np.nan)
    data, dq, was_sat, model = _frame()
    out = C._fill_saturated_pixels(
        CAL, data, dq, was_sat, model, data.copy(),
        types.SimpleNamespace(satstar_ramp_recover=True))
    assert out[1, 1] == 5.0 and out[2, 2] == 5.0


def test_a_group0_failure_keeps_the_slope_rim(monkeypatch):
    # Regression: an unreadable group 0 must not discard the rim already
    # recovered from the ramp slope.
    _stub_ramp(monkeypatch)
    monkeypatch.setattr(C, '_load_ramp_group0',
                        lambda p: (_ for _ in ()).throw(OSError('bad ramp')))
    data, dq, was_sat, model = _frame()
    out = C._fill_saturated_pixels(
        CAL, data, dq, was_sat, model, data.copy(),
        types.SimpleNamespace(satstar_ramp_recover=True))
    assert out[1, 1] == 11.0
    assert out[2, 2] == 5.0


def test_ramp_slope_wins_over_zeroframe_when_both_are_enabled(monkeypatch):
    _stub_ramp(monkeypatch)
    data, dq, was_sat, model = _frame()
    out = C._fill_saturated_pixels(
        CAL, data, dq, was_sat, model, data.copy(),
        types.SimpleNamespace(satstar_ramp_recover=True,
                              satstar_zeroframe_recover=True))
    assert out[1, 1] == 11.0      # 11 = ramp, 22 = zeroframe


def test_zeroframe_only_fills_its_rim(monkeypatch):
    _stub_ramp(monkeypatch)
    data, dq, was_sat, model = _frame()
    out = C._fill_saturated_pixels(
        CAL, data, dq, was_sat, model, data.copy(),
        types.SimpleNamespace(satstar_zeroframe_recover=True))
    assert out[2, 2] == 22.0      # the zeroframe stub's rim is `deep`
    assert out[1, 1] == 5.0       # not its rim -> model


def test_the_callers_array_is_never_modified_in_place(monkeypatch):
    _stub_ramp(monkeypatch)
    data, dq, was_sat, model = _frame()
    filled = data.copy()
    C._fill_saturated_pixels(CAL, data, dq, was_sat, model, filled,
                             types.SimpleNamespace(satstar_ramp_recover=True))
    assert np.array_equal(filled, data)


def test_the_dedup_removes_a_repeated_source(tmp_path):
    # The other combine tests use well-separated positions, so the dedup
    # branch never fires.
    merged = tmp_path / 'cat.fits'
    _vetted(1).write(merged, overwrite=True)
    same = _vetted(3, 266.5)
    same.write(tmp_path / 'cat_o001_vetted.fits', overwrite=True)
    same.write(tmp_path / 'cat_o002_vetted.fits', overwrite=True)
    combined = tmp_path / 'cat_vetted.fits'
    C._combine_per_obs_vetted(str(tmp_path / 'cat_o002_vetted.fits'),
                              str(merged), str(combined), this_obs_only=False)
    assert len(Table.read(combined)) == 3     # 6 stacked -> 3 after dedup


def test_a_failed_carta_export_does_not_stop_the_run(tmp_path, monkeypatch,
                                                     capsys):
    merged = tmp_path / 'cat.fits'
    _vetted(1).write(merged, overwrite=True)
    _vetted(3).write(tmp_path / 'cat_o001_vetted.fits', overwrite=True)
    monkeypatch.setattr(C, '_write_carta_catalog',
                        lambda *a: (_ for _ in ()).throw(OSError('disk full')))
    combined = tmp_path / 'cat_vetted.fits'
    C._combine_per_obs_vetted(str(tmp_path / 'cat_o001_vetted.fits'),
                              str(merged), str(combined), this_obs_only=False)
    assert combined.exists()                  # the science catalog survived
    assert 'CARTA catalog export failed' in capsys.readouterr().out
