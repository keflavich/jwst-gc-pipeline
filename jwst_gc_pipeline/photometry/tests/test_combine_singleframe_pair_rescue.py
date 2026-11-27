"""``MERGE_PAIR_RESCUE_RATIO``: a bright star next to a faint row keeps its own merged row.

Loop 1 of ``combine_singleframe`` seeds the base catalog in frame order, and a
row within ``min_offset`` of a base row never starts one.  When the first frame
that covers a star holds only a faint row beside it, the faint row seeds the
base row and the star's bright rows lose the mutual nearest-neighbour test in
every frame that also fits the faint row (the star leaves the merged catalog),
or pass it in a frame that does not and are averaged with the faint rows, which
carry the smaller errors and set the flux (see ``merge_pair_rescue``).

The rescue is off unless the variable is above 0.  Rows are placed east of one
point so the separations below are the offsets themselves.  F212N has a PSF
FWHM of 0.072", so the split-row radius is 0.054" and ``min_offset`` 0.1".
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.photometry import merge_catalogs as MC
from jwst_gc_pipeline.photometry import merge_pair_rescue as MPR

RA0, DEC0 = 266.5, -28.8
COSD = np.cos(np.deg2rad(DEC0))
RATIO, FLOOR = 'MERGE_PAIR_RESCUE_RATIO', 'MERGE_PAIR_RESCUE_FLUX_MIN'


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (RATIO, FLOOR, 'MERGE_DEDUP_OFFSET_ARCSEC', 'MERGE_DEDUP_FWHM_FRAC'):
        monkeypatch.delenv(name, raising=False)


def _sci_file(tmp_path):
    path = tmp_path / 'pair_sci.fits'
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(data=np.zeros((4, 4)), name='SCI')]
                 ).writeto(path, overwrite=True)
    return str(path)


def _frame(tmp_path, f, rows, dao=True):
    """One exposure catalog.  ``rows`` are (east offset in arcsec, flux, flux_err)
    from (RA0, DEC0); a far reference star of flux 500 rides along in every frame."""
    rows = list(rows) + [(20.0, 500.0, 10.0)]
    off = np.array([r[0] for r in rows], dtype=float)
    flux = np.array([r[1] for r in rows], dtype='float32')
    err = np.array([r[2] for r in rows], dtype='float32')
    n = len(rows)
    sc = SkyCoord(ra=(RA0 + off / 3600.0 / COSD) * u.deg,
                  dec=np.full(n, DEC0) * u.deg, frame='icrs')
    t = Table()
    t['id'] = np.arange(1, n + 1)
    if dao:
        t['skycoord_centroid'] = sc
        t['flux_fit'] = flux
        t['flux_err'] = err
        t['qfit'] = np.full(n, 0.1, dtype='float32')
        t['cfit'] = np.zeros(n, dtype='float32')
        t['flux_init'] = flux
        t['flags'] = np.zeros(n, dtype='int32')
        t['forced_refit'] = np.zeros(n, dtype=bool)
    else:
        t['skycoord'] = sc
        t['flux'] = flux
        t['dflux'] = err
        t['qf'] = np.full(n, 1.0, dtype='float32')
        t['fracflux'] = np.full(n, 0.9, dtype='float32')
    t.meta.update(exposure=f + 1, MODULE='nrca', filter='f212n',
                  FILENAME=_sci_file(tmp_path),
                  ra_offset=0.0 * u.arcsec, dec_offset=0.0 * u.arcsec)
    return t


def _combine(tbls):
    return MC.combine_singleframe(tbls, nanaverage=MC.nanaverage_numpy,
                                  verbose=False, filtername='f212n')


def _rows_near_origin(out, radius_as=1.0):
    sep = out['skycoord_avg'].separation(SkyCoord(RA0 * u.deg, DEC0 * u.deg))
    near = out[sep.arcsec < radius_as]
    return near[np.argsort(near['skycoord_avg'].ra.deg)]


def _flux_col(dao):
    return 'flux_fit_avg' if dao else 'flux_avg'


# A star of 15000 at +0.085" with a faint row of 200 at the origin.  Frame 0
# holds only the faint row; frames 1-4 hold both.
PAIR_BRIGHT_X = [0.085, 0.086, 0.084, 0.085]
PAIR_FAINT_X = [0.0, 0.002, -0.002, 0.001, 0.0]


def _pair_frames(tmp_path, dao=True, bright=15000.0, faint=200.0, bright_x=PAIR_BRIGHT_X):
    frames = [_frame(tmp_path, 0, [(PAIR_FAINT_X[0], faint, 5.0)], dao=dao)]
    for k, x in enumerate(bright_x):
        frames.append(_frame(tmp_path, k + 1, [(PAIR_FAINT_X[k + 1], faint, 5.0),
                                               (x, bright, 40.0)], dao=dao))
    return frames


@pytest.mark.parametrize('dao', [True, False])
def test_the_pair_loses_its_bright_star_without_the_rescue(tmp_path, dao):
    """Guard on the fixture: the default merge drops the bright star."""
    rows = _rows_near_origin(_combine(_pair_frames(tmp_path, dao=dao)))
    assert len(rows) == 1
    assert rows['nmatch'][0] == 5
    assert rows[_flux_col(dao)][0] == pytest.approx(200.0, rel=0.02)


@pytest.mark.parametrize('dao', [True, False])
def test_the_rescue_keeps_the_bright_star(tmp_path, monkeypatch, dao):
    monkeypatch.setenv(RATIO, '3')
    out = _combine(_pair_frames(tmp_path, dao=dao))
    rows = _rows_near_origin(out)
    assert len(rows) == 2
    faint, bright = rows
    assert faint['nmatch'] == 5
    assert faint[_flux_col(dao)] == pytest.approx(200.0, rel=0.02)
    assert bright['nmatch'] == 4
    assert bright[_flux_col(dao)] == pytest.approx(15000.0, rel=1e-3)
    expect = SkyCoord((RA0 + np.mean(PAIR_BRIGHT_X) / 3600.0 / COSD) * u.deg, DEC0 * u.deg)
    assert bright['skycoord_avg'].separation(expect).to_value(u.mas) < 1.0
    # one base row started by the rescue; everything else is as before
    assert out.meta['PAIRRESC'] == 1
    assert len(out) == 3


def test_the_rescue_is_logged(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(RATIO, '3')
    monkeypatch.setenv(FLOOR, '100')
    _combine(_pair_frames(tmp_path))
    assert "pair rescue (flux ratio >= 3, flux >= 100): 1 base row(s)" in capsys.readouterr().out


@pytest.mark.parametrize('off_value', [None, '', '  ', '0', '0.0'])
def test_off_is_the_default_merge(tmp_path, monkeypatch, off_value):
    default = _combine(_pair_frames(tmp_path))
    if off_value is not None:
        monkeypatch.setenv(RATIO, off_value)
        monkeypatch.setenv(FLOOR, 'not read while the rescue is off')
    out = _combine(_pair_frames(tmp_path))
    assert 'PAIRRESC' not in out.meta
    assert out.colnames == default.colnames
    for name in default.colnames:
        if name == 'skycoord_avg':
            assert np.array_equal(out[name].ra.deg, default[name].ra.deg)
            assert np.array_equal(out[name].dec.deg, default[name].dec.deg)
        else:
            assert np.array_equal(np.asarray(out[name]), np.asarray(default[name]),
                                  equal_nan=True), name


def test_the_rescue_leaves_every_other_row_alone(tmp_path, monkeypatch):
    """Rows away from the pair come out exactly as in the default merge."""
    default = _combine(_pair_frames(tmp_path))
    monkeypatch.setenv(RATIO, '3')
    out = _combine(_pair_frames(tmp_path))
    far_d = default[default['skycoord_avg'].separation(SkyCoord(RA0 * u.deg, DEC0 * u.deg)).arcsec > 5]
    far_o = out[out['skycoord_avg'].separation(SkyCoord(RA0 * u.deg, DEC0 * u.deg)).arcsec > 5]
    assert len(far_d) == len(far_o) == 1
    assert far_o['nmatch'][0] == far_d['nmatch'][0] == 5
    assert far_o['flux_fit_avg'][0] == far_d['flux_fit_avg'][0]
    # the faint row of the pair is the same row in both merges
    faint_d = _rows_near_origin(default)[0]
    faint_o = _rows_near_origin(out)[0]
    assert faint_o['flux_fit_avg'] == faint_d['flux_fit_avg']
    assert faint_o['nmatch'] == faint_d['nmatch']


def test_a_frame_without_a_faint_row_is_not_averaged_with_the_faint_rows(tmp_path, monkeypatch):
    """The bright row passes the mutual test in a frame with no faint row and
    the 1 / flux_err^2 weights then let the faint row set the merged flux."""
    frames = [_frame(tmp_path, 0, [(0.0, 300.0, 5.0)]),
              _frame(tmp_path, 1, [(0.085, 8000.0, 40.0)])]
    rows = _rows_near_origin(_combine(frames))
    assert len(rows) == 1
    assert rows['nmatch'][0] == 2
    assert rows['flux_fit_avg'][0] < 500.0
    monkeypatch.setenv(RATIO, '3')
    faint, bright = _rows_near_origin(_combine(frames))
    assert (faint['nmatch'], bright['nmatch']) == (1, 1)
    assert faint['flux_fit_avg'] == pytest.approx(300.0)
    assert bright['flux_fit_avg'] == pytest.approx(8000.0)


@pytest.mark.parametrize('bright, bright_x, why', [
    (500.0, PAIR_BRIGHT_X, 'a flux ratio of 2.5 is below 3'),
    (15000.0, [0.04, 0.041, 0.039, 0.04], 'a row inside the split-row radius (0.054") is the same star'),
])
def test_rows_that_do_not_qualify_are_not_rescued(tmp_path, monkeypatch, bright, bright_x, why):
    monkeypatch.setenv(RATIO, '3')
    out = _combine(_pair_frames(tmp_path, bright=bright, bright_x=bright_x))
    assert out.meta['PAIRRESC'] == 0, why
    assert len(_rows_near_origin(out)) == 1


def test_the_flux_floor(tmp_path, monkeypatch):
    frames = _pair_frames(tmp_path, bright=900.0, faint=100.0)
    monkeypatch.setenv(RATIO, '3')
    monkeypatch.setenv(FLOOR, '1000')
    assert _combine(frames).meta['PAIRRESC'] == 0
    monkeypatch.setenv(FLOOR, '800')
    assert _combine(frames).meta['PAIRRESC'] == 1


def test_a_nan_flux_never_starts_a_row(tmp_path, monkeypatch):
    frames = _pair_frames(tmp_path)
    for t in frames[1:]:
        t['flux_fit'][t['flux_fit'] > 1000] = np.nan
    monkeypatch.setenv(RATIO, '3')
    assert _combine(frames).meta['PAIRRESC'] == 0


def test_a_row_without_coordinates_leaves_the_seed_fluxes_aligned(tmp_path, monkeypatch):
    """A row with a NaN coordinate leaves its frame before Loop 1 places anything, so
    the seed flux of each base row still belongs to that base row (a seed flux of
    1e6 on the faint row would hide the bright star)."""
    frames = _pair_frames(tmp_path)
    frames[0] = _frame(tmp_path, 0, [(np.nan, 1.0e6, 40.0), (PAIR_FAINT_X[0], 200.0, 5.0)])
    monkeypatch.setenv(RATIO, '3')
    out = _combine(frames)
    assert out.meta['PAIRRESC'] == 1
    faint, bright = _rows_near_origin(out)
    assert faint['flux_fit_avg'] == pytest.approx(200.0, rel=0.02)
    assert bright['flux_fit_avg'] == pytest.approx(15000.0, rel=1e-3)


@pytest.mark.parametrize('ratio, flux_min, expect', [
    (None, None, None), ('', None, None), ('0', 'junk', None),
    ('3', None, (3.0, 0.0)), ('2.5', '500', (2.5, 500.0)), ('3', ' ', (3.0, 0.0)),
])
def test_settings(monkeypatch, ratio, flux_min, expect):
    if ratio is not None:
        monkeypatch.setenv(RATIO, ratio)
    if flux_min is not None:
        monkeypatch.setenv(FLOOR, flux_min)
    assert MPR.settings() == expect


@pytest.mark.parametrize('name, value', [
    (RATIO, 'abc'), (RATIO, 'nan'), (RATIO, 'inf'), (RATIO, '1'), (RATIO, '0.5'), (RATIO, '-3'),
    (FLOOR, 'abc'), (FLOOR, '-1'), (FLOOR, 'nan'),
])
def test_bad_settings_name_the_variable(monkeypatch, name, value):
    monkeypatch.setenv(RATIO, '3')
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        MPR.settings()


def test_a_bad_setting_stops_the_merge_before_any_matching(tmp_path, monkeypatch):
    monkeypatch.setenv(RATIO, '1')
    with pytest.raises(ValueError, match=RATIO):
        _combine(_pair_frames(tmp_path))


def test_rescue_rows():
    sep = u.Quantity([0.02, 0.06, 0.09, 0.09, 0.09, 0.09, 0.09], u.arcsec)
    flux = np.array([9000.0, 9000.0, 9000.0, 2000.0, 9000.0, np.nan, -5.0])
    seed = np.array([100.0, 100.0, 100.0, 100.0, np.nan, 100.0, 100.0])
    got = MPR.rescue_rows(sep, flux, seed, 0.054 * u.arcsec, 10.0, 500.0)
    # rows: inside the split-row radius; qualifies; qualifies; flux 2000 is 20 x the seed, above the floor of 500;
    # NaN seed flux; NaN flux; negative flux
    assert got.tolist() == [False, True, True, True, False, False, False]
    # a ratio of 30 leaves the 20 x row out
    got = MPR.rescue_rows(sep, flux, seed, 0.054 * u.arcsec, 30.0, 500.0)
    assert got.tolist() == [False, True, True, False, False, False, False]
