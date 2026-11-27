"""``MERGE_PARTIAL_BOX_FRAC_MIN``: a row fitted to part of its pixel box does not outvote a full-box row.

``combine_singleframe`` weights the rows of a star by 1 / ``flux_err``^2 and averages their
positions without weights.  A row that lost most of its 5 x 5 fit box can carry a smaller
``flux_err`` than the full-box row of the same star when its flux is smaller, and with two
rows the MAD clip rejects neither, so the merged flux follows the partial row and the merged
position sits between the two (see ``merge_partial_box``).

The star below is gc-treasury 10678 o130 F212N, mosaic pixel (5065, 68): a full-box row of
flux 14433.84 +- 64.715 and a row of 10 of 25 pixels, flux 1295.42 +- 31.101, 0.0924"
(2.96 pixels) from it.  Rows are placed east of one point so the separations below are the
offsets themselves.  The switch is off unless the variable is above 0.
"""
import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.photometry import merge_catalogs as MC
from jwst_gc_pipeline.photometry import merge_partial_box as MPB

RA0, DEC0 = 266.5, -28.8
COSD = np.cos(np.deg2rad(DEC0))
FRAC, ZMAX = 'MERGE_PARTIAL_BOX_FRAC_MIN', 'MERGE_PARTIAL_BOX_ZMAX'

FULL = (0.0, 14433.84, 64.715, 25)        # east offset ("), flux, flux_err, n_pixels_fit
PART = (0.0924, 1295.42, 31.101, 10)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (FRAC, ZMAX, 'MERGE_PAIR_RESCUE_RATIO', 'MERGE_DEDUP_OFFSET_ARCSEC', 'MERGE_DEDUP_FWHM_FRAC'):
        monkeypatch.delenv(name, raising=False)


def _sci_file(tmp_path):
    path = tmp_path / 'pbox_sci.fits'
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(data=np.zeros((4, 4)), name='SCI')]
                 ).writeto(path, overwrite=True)
    return str(path)


def _frame(tmp_path, f, rows, dao=True, npix_col=True):
    """One exposure catalog.  ``rows`` are (east offset in arcsec, flux, flux_err, n_pixels_fit)
    from (RA0, DEC0); a far reference star of flux 500 fitted over the whole box rides along in
    every frame, so the fit box of the call is 25 pixels."""
    rows = list(rows) + [(20.0, 500.0, 10.0, 25)]
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
        if npix_col:
            t['n_pixels_fit'] = np.array([r[3] for r in rows], dtype='int32')
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


def _star(out):
    """The merged row of the star near the origin (the far reference star is dropped)."""
    sep = out['skycoord_avg'].separation(SkyCoord(RA0 * u.deg, DEC0 * u.deg))
    near = out[sep.arcsec < 1.0]
    assert len(near) == 1
    return near[0]


def _east_arcsec(row):
    return (row['skycoord_avg'].ra.deg - RA0) * COSD * 3600.0


def _pair(tmp_path, partial=PART, full=FULL, **kwargs):
    frames = [_frame(tmp_path, 0, [full], **kwargs), _frame(tmp_path, 1, [partial], **kwargs)]
    if kwargs.get('dao', True):
        frames[1]['flags'][0] = 1       # the partial row of the o130 star has flags 1
    return frames


def _iv_mean(*rows):
    w = np.array([1.0 / r[2]**2 for r in rows])
    return float((np.array([r[1] for r in rows]) * w).sum() / w.sum())


# ---------------------------------------------------------------------------------------------
# the merge
# ---------------------------------------------------------------------------------------------

def test_the_default_merge_follows_the_partial_row(tmp_path):
    """Guard on the fixture: the partial row outweighs the full-box row 4.3 to 1, and the position is the midpoint."""
    star = _star(_combine(_pair(tmp_path)))
    assert (FULL[2] / PART[2])**2 == pytest.approx(4.33, abs=0.01)
    assert star['flux_fit_avg'] == pytest.approx(3760.5, rel=1e-4)
    assert star['flux_fit_avg'] == pytest.approx(_iv_mean(FULL, PART), rel=1e-5)
    assert _east_arcsec(star) == pytest.approx(PART[0] / 2, abs=1e-4)
    assert star['nmatch'] == star['nmatch_good'] == 2


def test_the_switch_leaves_the_partial_row_out(tmp_path, monkeypatch):
    default = _star(_combine(_pair(tmp_path)))
    monkeypatch.setenv(FRAC, '0.5')
    out = _combine(_pair(tmp_path))
    star = _star(out)
    assert star['flux_fit_avg'] == pytest.approx(FULL[1], rel=1e-5)
    assert star['flux_err_prop'] == pytest.approx(FULL[2], rel=1e-5)
    assert _east_arcsec(star) == pytest.approx(0.0, abs=1e-4)
    # the row is still a detection: nmatch_good counts it and the new column says it is not averaged
    assert star['nmatch'] == star['nmatch_good'] == 2
    assert star['n_partial_box_dropped'] == 1
    assert out.meta['PBOXDROP'] == 1
    # the other averaged columns follow the same weights: flags_avg is that of the full-box row
    assert star['flags_avg'] == 0.0
    assert default['flags_avg'] == pytest.approx(0.81, abs=0.01)


def test_the_switch_is_logged(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(FRAC, '0.5')
    _combine(_pair(tmp_path))
    assert ("MERGE_PARTIAL_BOX_FRAC_MIN=0.5: left 1 row(s) of 1 source(s) out of the averages "
            "(fit box 25 pixels)") in capsys.readouterr().out


@pytest.mark.parametrize('off_value', [None, '', '  ', '0', '0.0'])
def test_off_is_the_default_merge(tmp_path, monkeypatch, off_value):
    default = _combine(_pair(tmp_path))
    if off_value is not None:
        monkeypatch.setenv(FRAC, off_value)
        monkeypatch.setenv(ZMAX, 'not read while the switch is off')
    out = _combine(_pair(tmp_path))
    assert 'PBOXDROP' not in out.meta
    assert 'n_partial_box_dropped' not in out.colnames
    assert out.colnames == default.colnames
    for name in default.colnames:
        if name == 'skycoord_avg':
            assert np.array_equal(out[name].ra.deg, default[name].ra.deg)
            assert np.array_equal(out[name].dec.deg, default[name].dec.deg)
        else:
            assert np.array_equal(np.asarray(out[name]), np.asarray(default[name]), equal_nan=True), name


def test_a_star_with_only_partial_rows_is_averaged_as_before(tmp_path, monkeypatch):
    """No row at or above the fraction, so nothing outvotes anything: the default merge."""
    low = (0.0924, 1295.42, 31.101, 10)
    high = (0.0, 14433.84, 64.715, 10)
    default = _star(_combine(_pair(tmp_path, partial=low, full=high)))
    monkeypatch.setenv(FRAC, '0.5')
    out = _combine(_pair(tmp_path, partial=low, full=high))
    star = _star(out)
    assert star['flux_fit_avg'] == default['flux_fit_avg']
    assert _east_arcsec(star) == _east_arcsec(default)
    assert star['n_partial_box_dropped'] == 0


def test_rows_at_the_fraction_are_kept(tmp_path, monkeypatch):
    """12 of 25 pixels is below 0.5, 13 is not."""
    monkeypatch.setenv(FRAC, '0.5')
    thirteen = (PART[0], PART[1], PART[2], 13)
    assert _star(_combine(_pair(tmp_path, partial=thirteen)))['n_partial_box_dropped'] == 0
    twelve = (PART[0], PART[1], PART[2], 12)
    assert _star(_combine(_pair(tmp_path, partial=twelve)))['n_partial_box_dropped'] == 1


def test_a_clipped_row_is_not_a_full_box_partner(tmp_path, monkeypatch):
    """Three partial rows and one full-box row of a very different flux: the sigma clip rejects the full-box
    row, so no kept row is at or above the fraction and the three partial rows stay."""
    rows = [(0.0, 5000.0, 60.0, 25), (0.0, 1000.0, 30.0, 10), (0.0, 1010.0, 30.0, 10), (0.0, 990.0, 30.0, 10)]
    frames = [_frame(tmp_path, k, [r]) for k, r in enumerate(rows)]
    default = _star(_combine(frames))
    assert default['nmatch'] == 4 and default['nmatch_good'] == 3
    monkeypatch.setenv(FRAC, '0.5')
    star = _star(_combine(frames))
    assert star['n_partial_box_dropped'] == 0
    assert star['flux_fit_avg'] == default['flux_fit_avg']
    assert star['flux_fit_avg'] == pytest.approx(1000.0, abs=15.0)


def test_several_partial_rows_and_several_full_rows(tmp_path, monkeypatch):
    """Five rows that the sigma clip keeps (it rejects a row far from the others, so these lie within 1.7 MAD
    sigma): three at 24-25 pixels and two at 10 and 5 whose smaller errors give them most of the weight."""
    full = [(0.0, 14433.84, 64.715, 25), (0.002, 14500.0, 65.0, 24), (-0.002, 14400.0, 64.5, 25)]
    part = [(0.004, 14200.0, 30.0, 10), (-0.004, 14150.0, 25.0, 5)]
    frames = [_frame(tmp_path, k, [r]) for k, r in enumerate(full + part)]
    default = _star(_combine(frames))
    assert default['nmatch_good'] == 5
    assert default['flux_fit_avg'] == pytest.approx(_iv_mean(*full, *part), rel=1e-5)
    monkeypatch.setenv(FRAC, '0.5')
    star = _star(_combine(frames))
    assert star['n_partial_box_dropped'] == 2
    assert star['nmatch_good'] == 5
    assert star['flux_fit_avg'] == pytest.approx(_iv_mean(*full), rel=1e-5)
    assert abs(default['flux_fit_avg'] - star['flux_fit_avg']) > 100.0


# ---------------------------------------------------------------------------------------------
# MERGE_PARTIAL_BOX_ZMAX
# ---------------------------------------------------------------------------------------------

def test_zmax_leaves_out_the_row_that_disagrees(tmp_path, monkeypatch):
    """The partial row is 2e2 sigma from the full-box row: left out."""
    monkeypatch.setenv(FRAC, '0.5')
    monkeypatch.setenv(ZMAX, '3')
    star = _star(_combine(_pair(tmp_path)))
    assert star['n_partial_box_dropped'] == 1
    assert star['flux_fit_avg'] == pytest.approx(FULL[1], rel=1e-5)


def test_zmax_keeps_the_row_that_agrees(tmp_path, monkeypatch):
    """A partial row 1.5 sigma from the full-box row keeps its vote."""
    agree = (0.001, 14433.84 - 1.5 * np.hypot(64.715, 31.101), 31.101, 10)
    monkeypatch.setenv(FRAC, '0.5')
    monkeypatch.setenv(ZMAX, '3')
    star = _star(_combine(_pair(tmp_path, partial=agree)))
    assert star['n_partial_box_dropped'] == 0
    assert star['flux_fit_avg'] == pytest.approx(_iv_mean(FULL, agree), rel=1e-5)
    monkeypatch.setenv(ZMAX, '1')
    assert _star(_combine(_pair(tmp_path, partial=agree)))['n_partial_box_dropped'] == 1


def test_zmax_without_a_reference_row_keeps_every_row(tmp_path, monkeypatch):
    monkeypatch.setenv(FRAC, '0.5')
    monkeypatch.setenv(ZMAX, '3')
    both = (0.0, 14433.84, 64.715, 10)
    star = _star(_combine(_pair(tmp_path, full=both)))
    assert star['n_partial_box_dropped'] == 0


# ---------------------------------------------------------------------------------------------
# frames without the column
# ---------------------------------------------------------------------------------------------

def test_frames_without_n_pixels_fit_are_never_changed(tmp_path, monkeypatch, capsys):
    default = _star(_combine(_pair(tmp_path, npix_col=False)))
    monkeypatch.setenv(FRAC, '0.5')
    out = _combine(_pair(tmp_path, npix_col=False))
    star = _star(out)
    assert star['flux_fit_avg'] == default['flux_fit_avg']
    assert _east_arcsec(star) == _east_arcsec(default)
    assert out.meta['PBOXDROP'] == 0
    assert "2 of 2 frames have no n_pixels_fit column" in capsys.readouterr().out


def test_a_frame_without_the_column_is_neither_dropped_nor_a_reference(tmp_path, monkeypatch):
    """The full-box row is in a frame without the column: the partial row has no reference and stays."""
    frames = _pair(tmp_path)
    frames[0] = _frame(tmp_path, 0, [FULL], npix_col=False)
    default = _star(_combine(frames))
    monkeypatch.setenv(FRAC, '0.5')
    star = _star(_combine(frames))
    assert star['n_partial_box_dropped'] == 0
    assert star['flux_fit_avg'] == default['flux_fit_avg']


def test_dolphot_style_frames_are_not_touched(tmp_path, monkeypatch):
    """The dolphot tables carry no n_pixels_fit; the switch changes nothing and does not fail."""
    frames = _pair(tmp_path, dao=False)
    default = _star(_combine(frames))
    monkeypatch.setenv(FRAC, '0.5')
    star = _star(_combine(frames))
    assert star['flux_avg'] == default['flux_avg']
    assert _east_arcsec(star) == _east_arcsec(default)


# ---------------------------------------------------------------------------------------------
# settings and partial_rows
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize('frac, zmax, expect', [
    (None, None, None), ('', '3', None), ('0', 'junk', None), (' ', None, None),
    ('0.5', None, (0.5, None)), ('0.5', ' ', (0.5, None)), ('0.5', '0', (0.5, None)),
    ('0.6', '3', (0.6, 3.0)), ('1', '2.5', (1.0, 2.5)),
])
def test_settings(monkeypatch, frac, zmax, expect):
    if frac is not None:
        monkeypatch.setenv(FRAC, frac)
    if zmax is not None:
        monkeypatch.setenv(ZMAX, zmax)
    assert MPB.settings() == expect


@pytest.mark.parametrize('name, value', [
    (FRAC, 'abc'), (FRAC, 'nan'), (FRAC, 'inf'), (FRAC, '1.5'), (FRAC, '-0.5'),
    (ZMAX, 'abc'), (ZMAX, '-1'), (ZMAX, 'nan'),
])
def test_bad_settings_name_the_variable(monkeypatch, name, value):
    monkeypatch.setenv(FRAC, '0.5')
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        MPB.settings()


def test_a_bad_setting_stops_the_merge_before_any_matching(tmp_path, monkeypatch):
    monkeypatch.setenv(FRAC, '1.5')
    with pytest.raises(ValueError, match=FRAC):
        _combine(_pair(tmp_path))


def _arrays():
    nan = np.nan
    n_fit = np.array([[25, 10, 25, nan],      # one partial row, two full rows
                      [10, 10, nan, nan],     # only partial rows
                      [24, 12, 13, nan],      # 12 is partial, 24 and 13 are above 0.5 of 25
                      [25, 5, nan, nan],      # partial row without a flux error
                      [25, 5, nan, nan],      # partial row clipped
                      [nan, nan, nan, nan]])
    flux = np.array([[1000., 100., 1010., nan],
                     [100., 120., nan, nan],
                     [1000., 400., 1010., nan],
                     [1000., 100., nan, nan],
                     [1000., 100., nan, nan],
                     [nan, nan, nan, nan]])
    err = np.array([[10., 5., 10., nan],
                    [5., 5., nan, nan],
                    [10., 8., 10., nan],
                    [10., np.nan, nan, nan],
                    [10., 5., nan, nan],
                    [nan, nan, nan, nan]])
    usable = np.isfinite(flux)
    usable[4, 1] = False
    return n_fit, usable, flux, err


def test_partial_rows_without_zmax():
    n_fit, usable, flux, err = _arrays()
    drop, box = MPB.partial_rows(n_fit, usable, flux, err, 0.5)
    assert box == 25.0
    expect = np.zeros(n_fit.shape, dtype=bool)
    expect[0, 1] = True
    expect[2, 1] = True
    assert np.array_equal(drop, expect)


def test_partial_rows_with_zmax():
    n_fit, usable, flux, err = _arrays()
    drop, box = MPB.partial_rows(n_fit, usable, flux, err, 0.5, zmax=3.0)
    # row 0: flux 100 against 1005 +- 7.1 (the two rows at 25 pixels): 105 sigma, left out.  row 2: flux 400 against
    # the same mean (the rows at 24 and 13 pixels): 57 sigma, left out.
    expect = np.zeros(n_fit.shape, dtype=bool)
    expect[0, 1] = True
    expect[2, 1] = True
    assert np.array_equal(drop, expect)
    # a row 1 sigma from the others stays
    flux2 = flux.copy()
    flux2[0, 1] = 1005.0 - np.sqrt(5.0**2 + 10.0**2 / 2)
    drop, _ = MPB.partial_rows(n_fit, usable, flux2, err, 0.5, zmax=3.0)
    assert not drop[0].any()
    drop, _ = MPB.partial_rows(n_fit, usable, flux2, err, 0.5, zmax=0.5)
    assert drop[0, 1]


def test_a_star_never_loses_all_its_rows():
    rng = np.random.default_rng(3)
    n_fit = rng.integers(3, 26, size=(500, 6)).astype(float)
    n_fit[rng.random(n_fit.shape) < 0.3] = np.nan
    n_fit[0, 0] = 25.0
    flux = rng.uniform(100, 1000, size=n_fit.shape)
    err = rng.uniform(5, 50, size=n_fit.shape)
    usable = np.isfinite(n_fit)
    for z in (None, 1.0, 3.0):
        drop, box = MPB.partial_rows(n_fit, usable, flux, err, 0.5, zmax=z)
        assert box == 25.0
        assert not (drop & ~usable).any()
        has_rows = usable.any(axis=1)
        assert not (has_rows & (drop.sum(axis=1) == usable.sum(axis=1))).any()


def test_blocks_do_not_change_the_result(monkeypatch):
    rng = np.random.default_rng(5)
    n_fit = rng.integers(3, 26, size=(300, 5)).astype(float)
    n_fit[0, 0] = 25.0
    flux = rng.uniform(100, 1000, size=n_fit.shape)
    err = rng.uniform(5, 50, size=n_fit.shape)
    usable = rng.random(n_fit.shape) < 0.9
    for z in (None, 2.0):
        whole, _ = MPB.partial_rows(n_fit, usable, flux, err, 0.5, zmax=z)
        with monkeypatch.context() as patch:
            patch.setattr(MPB, '_BLOCK', 7)
            blocked, _ = MPB.partial_rows(n_fit, usable, flux, err, 0.5, zmax=z)
        assert np.array_equal(whole, blocked)


def test_no_pixel_counts_at_all():
    drop, box = MPB.partial_rows(np.full((4, 3), np.nan), np.ones((4, 3), bool), np.ones((4, 3)), np.ones((4, 3)), 0.5)
    assert not drop.any()
    assert np.isnan(box)
