"""Opt-in restore of the daophot row a satstar overwrote and the in-field dedup
drops (``satstar_dao_restore``, ``SATSTAR_DAO_RESTORE_FRAC_MIN``).

``replace_saturated`` writes the satstar's flux and position over the daophot row
it pairs with the satstar.  The in-field dedup (``SATSTAR_INFIELD_DEDUP='cofit'``)
then drops the ``replaced_saturated`` row of a satstar that no exposure fit beside
a better supported neighbour, and a real star that has its own daophot detections
leaves the catalog with the row.  The switch records the overwritten daophot row
at the merge and puts it back at the dedup.

These tests pin:
* the switch is off unless ``FRAC_MIN`` is above 0, and a bad value raises
  naming its variable, in the module and at the start of ``run_manual_pipeline``;
* ``replace_saturated`` writes the ``satstar_dao_*`` columns only when asked, with
  the flux, flux error and position the row held before the overwrite, and NaN on
  every other row;
* a row qualifies on the flux ratio window (lower end included, upper end
  excluded), the separation limit, the surviving-neighbour distance and a usable
  record, and the transient-like rows of o105 F212N (0.002-0.03 of the satstar
  flux, 63-79 mas away) do not;
* a restored row is the daophot row: its flux, error and position, not
  ``replaced_saturated``, its satstar ensemble columns blank, ``is_saturated``
  kept; every other row is untouched;
* the in-field dedup keeps the restorable rows it would drop and drops the rest
  as it did, a second pass changes nothing, the input table is not modified, and
  a table without the record is dropped as before, with a WARNING.
"""
import inspect
import types

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import MaskedColumn, Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry import cataloging as C
from jwst_gc_pipeline.photometry import merge_catalogs as MC
from jwst_gc_pipeline.photometry import satstar_dao_restore as R

ENVS = (R.ENV_FRAC_MIN, R.ENV_FRAC_MAX, R.ENV_SEP_MAX, R.ENV_NEIGHBOUR_MIN)

PIXSCALE = 0.063 / 3600.0      # deg/px, NIRCam LW
SHAPE = (400, 400)             # ~25" on a side
RA0, DEC0 = 266.70, -28.55

CFG = R.Settings(0.75, 1.25, 0.03, 0.1)


@pytest.fixture
def clean_env(monkeypatch):
    for name in ENVS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [200.5, 200.5]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.cdelt = [-PIXSCALE, PIXSCALE]
    return w


def _data_i2d(tmp_path):
    path = tmp_path / 'data_i2d.fits'
    hdu = fits.ImageHDU(np.zeros(SHAPE, dtype='float32'), name='SCI')
    hdu.header.update(_wcs().to_header())
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)
    return str(path)


def _sky(offsets_arcsec):
    """Positions ``(east, north)`` arcsec from the field centre."""
    off = np.atleast_2d(np.asarray(offsets_arcsec, float))
    return SkyCoord(RA0 * u.deg + off[:, 0] * u.arcsec / np.cos(np.deg2rad(DEC0)),
                    DEC0 * u.deg + off[:, 1] * u.arcsec)


def _table(offsets, flux, prior_flux=None, prior_offsets=None, prior_err=None,
           nframes=None, replaced=True):
    """A merged table of satstar rows; with ``prior_flux`` it also carries the
    ``satstar_dao_*`` record (``prior_offsets`` is where the daophot row was)."""
    n = len(offsets)
    table = Table({
        'skycoord': _sky(offsets),
        'flux': np.asarray(flux, float),
        'flux_err': np.full(n, 20.0),
        'satstar_nframes': np.asarray([1] * n if nframes is None else nframes,
                                      dtype=np.int32),
        'satstar_nmeas': np.full(n, 4, dtype=np.int32),
        'satstar_std_ra': np.full(n, 0.002),
        'satstar_std_dec': np.full(n, 0.003),
        'replaced_saturated': np.full(n, replaced, dtype=bool),
        'is_saturated': np.ones(n, dtype=bool),
    })
    if prior_flux is not None:
        sky = _sky(prior_offsets)
        table[R.PRIOR_FLUX] = np.asarray(prior_flux, float)
        table[R.PRIOR_FLUX_ERR] = (np.full(n, 3.0) if prior_err is None
                                   else np.asarray(prior_err, float))
        table[R.PRIOR_RA] = sky.ra.deg
        table[R.PRIOR_DEC] = sky.dec.deg
    return table


# ---- settings ---------------------------------------------------------------

@pytest.mark.parametrize('raw', [None, '', '   ', '0', '0.0'])
def test_off_unless_frac_min_is_above_zero(clean_env, raw):
    if raw is not None:
        clean_env.setenv(R.ENV_FRAC_MIN, raw)
    assert R.settings() is None
    assert not R.enabled()


def test_other_variables_are_not_read_while_off(clean_env):
    for name in ENVS[1:]:
        clean_env.setenv(name, 'junk')
    assert R.settings() is None


def test_defaults_when_on(clean_env):
    clean_env.setenv(R.ENV_FRAC_MIN, '0.75')
    cfg = R.settings()
    assert cfg == R.Settings(0.75, 1.25, 0.03, 0.1)
    assert R.enabled()


def test_values_are_read(clean_env):
    clean_env.setenv(R.ENV_FRAC_MIN, '0.5')
    clean_env.setenv(R.ENV_FRAC_MAX, '2')
    clean_env.setenv(R.ENV_SEP_MAX, '0.01')
    clean_env.setenv(R.ENV_NEIGHBOUR_MIN, '0')
    assert R.settings() == R.Settings(0.5, 2.0, 0.01, 0.0)


@pytest.mark.parametrize('env', ENVS)
@pytest.mark.parametrize('raw', ['half', '-1', 'nan', 'inf'])
def test_bad_value_raises_naming_the_variable(clean_env, env, raw):
    clean_env.setenv(R.ENV_FRAC_MIN, '0.75')
    clean_env.setenv(env, raw)
    with pytest.raises(ValueError, match=rf'^{env}='):
        R.settings()


@pytest.mark.parametrize('frac_max', ['0.75', '0.5'])
def test_frac_max_must_exceed_frac_min(clean_env, frac_max):
    clean_env.setenv(R.ENV_FRAC_MIN, '0.75')
    clean_env.setenv(R.ENV_FRAC_MAX, frac_max)
    with pytest.raises(ValueError, match=rf'^{R.ENV_FRAC_MAX}='):
        R.settings()


class _ReachedFirstStep(Exception):
    pass


@pytest.mark.parametrize('env', ENVS)
def test_bad_value_stops_run_manual_pipeline_at_entry(clean_env, env):
    reached = []

    def _first_step(*args, **kwargs):
        reached.append(True)
        raise _ReachedFirstStep

    clean_env.setattr(C._L, '_cutout_out_basepath', _first_step)
    args = (types.SimpleNamespace(), ['nrcb'], ['F480M'],
            {'10678': {'gc-treasury': 1}}, '10678', 'gc-treasury', '111',
            '/nonexistent-basepath', {}, {})
    clean_env.setenv(R.ENV_FRAC_MIN, '0.75')
    with pytest.raises(_ReachedFirstStep):
        C.run_manual_pipeline(*args)
    reached.clear()
    clean_env.setenv(env, 'x')
    with pytest.raises(ValueError, match=rf'^{env}='):
        C.run_manual_pipeline(*args)
    assert reached == []


def test_merge_records_the_prior_only_through_the_per_frame_call():
    src = ' '.join(inspect.getsource(MC.merge_individual_frames).split())
    assert '_record_dao_prior = satstar_dao_restore.enabled()' in src
    assert 'record_daophot_prior=_record_dao_prior)' in src
    # the check comes before the per-frame work, not at the replace_saturated call
    assert src.index('_record_dao_prior =') < src.index('desat = ')


# ---- flux_columns -----------------------------------------------------------

@pytest.mark.parametrize('names, expected', [
    (['flux', 'dflux', 'flux_err'], ('flux', 'dflux')),
    (['flux', 'flux_err', 'eflux'], ('flux', 'flux_err')),
    (['flux', 'eflux'], ('flux', 'eflux')),
    (['flux'], ('flux', None)),
    (['flux_fit', 'flux_err'], ('flux_fit', 'flux_err')),
    (['flux_fit'], ('flux_fit', None)),
    (['mag'], (None, None)),
])
def test_flux_columns_follow_replace_saturated(names, expected):
    assert R.flux_columns(Table({n: [1.0] for n in names})) == expected


# ---- recording at the merge -------------------------------------------------

def test_record_prior_keeps_the_overwritten_row():
    table = Table({'skycoord': _sky([(0, 0), (1, 0), (2, 0)]),
                   'flux_fit': [10.0, 20.0, 30.0],
                   'flux_err': [1.0, 2.0, 3.0]})
    before = table['skycoord'].copy()
    n = R.record_prior(table, [0, 2], 'flux_fit', 'flux_err')
    assert n == 2
    assert np.allclose(table[R.PRIOR_FLUX], [10.0, np.nan, 30.0], equal_nan=True)
    assert np.allclose(table[R.PRIOR_FLUX_ERR], [1.0, np.nan, 3.0], equal_nan=True)
    assert np.allclose(table[R.PRIOR_RA][[0, 2]], before.ra.deg[[0, 2]])
    assert np.allclose(table[R.PRIOR_DEC][[0, 2]], before.dec.deg[[0, 2]])
    assert np.isnan(table[R.PRIOR_RA][1]) and np.isnan(table[R.PRIOR_DEC][1])


def test_record_prior_resets_a_stale_record_and_handles_no_error_column():
    table = Table({'skycoord': _sky([(0, 0), (1, 0)]), 'flux': [10.0, 20.0]})
    R.record_prior(table, [0, 1], 'flux', None)
    assert np.isnan(table[R.PRIOR_FLUX_ERR]).all()
    R.record_prior(table, [1], 'flux', None)
    assert np.isnan(table[R.PRIOR_FLUX][0]) and table[R.PRIOR_FLUX][1] == 20.0
    R.record_prior(table, [], 'flux', None)
    assert all(np.isnan(table[c]).all() for c in R.PRIOR_COLUMNS)


def test_record_prior_reads_masked_flux_as_nan():
    table = Table({'skycoord': _sky([(0, 0), (1, 0)]),
                   'flux': MaskedColumn([10.0, 20.0], mask=[False, True]),
                   'flux_err': [1.0, 2.0]})
    R.record_prior(table, [0, 1], 'flux', 'flux_err')
    assert table[R.PRIOR_FLUX][0] == 10.0 and np.isnan(table[R.PRIOR_FLUX][1])


def _replace(monkeypatch, tmp_path, flavour, **kwargs):
    """Two satstars (flux 60000; the second 10" away has no daophot row) and a
    daophot row 5 mas from the first, with flux 54000 (0.9 of it); plus a faint
    unrelated row.  Returns ``(catalog after replacement, the row's skycoord
    before)``."""
    ra0, dec0 = 266.5, -28.7
    far = SkyCoord(ra0 * u.deg, dec0 * u.deg).spherical_offsets_by(
        10 * u.arcsec, 0 * u.arcsec)
    satstar = Table({
        'skycoord_fit': SkyCoord([ra0, far.ra.deg] * u.deg, [dec0, far.dec.deg] * u.deg),
        'flux_fit': [60000.0, 70000.0], 'flux_err': [10.0, 11.0],
        'x_fit': [81.0, 81.0], 'y_fit': [81.0, 81.0],
        'x_err': [0.01, 0.01], 'y_err': [0.01, 0.01],
        'xcentroid': [512.0, 700.0], 'ycentroid': [300.0, 300.0],
    })
    monkeypatch.setattr(MC, 'load_satstar_catalog', lambda *a, **k: satstar)

    class _FakeSvo:
        @staticmethod
        def get_filter_list(_):
            t = Table({'filterID': ['JWST/NIRCam.F182M'], 'ZeroPoint': [850.0]})
            t.add_index('filterID')
            return t
    monkeypatch.setattr(MC, 'SvoFps', _FakeSvo)
    (tmp_path / 'reduction').mkdir(exist_ok=True)
    Table({'Filter': ['F182M'], 'PSF FWHM (arcsec)': [0.06]}).write(
        tmp_path / 'reduction' / 'fwhm_table.ecsv', format='ascii.ecsv',
        overwrite=True)
    row = SkyCoord(ra0 * u.deg, dec0 * u.deg).spherical_offsets_by(
        0.005 * u.arcsec, 0 * u.arcsec)
    other = SkyCoord(ra0 * u.deg, dec0 * u.deg).spherical_offsets_by(
        -5 * u.arcsec, 0 * u.arcsec)
    sky = SkyCoord([row.ra.deg, other.ra.deg] * u.deg,
                   [row.dec.deg, other.dec.deg] * u.deg)
    if flavour == 'flux_fit':
        cat = Table({'skycoord': sky, 'flux_fit': [54000.0, 100.0],
                     'flux_err': [4.0, 1.0]})
    else:
        cat = Table({'skycoord': sky, 'flux': [54000.0, 100.0],
                     'dflux': [4.0, 1.0], 'x': [530.0, 10.0], 'y': [300.0, 10.0],
                     'dx': [0.01, 0.01], 'dy': [0.01, 0.01]})
    before = cat['skycoord'].copy()
    MC.replace_saturated(cat, 'f182m', basepath=str(tmp_path) + '/', **kwargs)
    return cat, before


@pytest.mark.parametrize('flavour, fcol, ecol', [('flux_fit', 'flux', 'flux_err'),
                                                 ('flux', 'flux', 'dflux')])
def test_replace_saturated_records_the_overwritten_row(
        monkeypatch, tmp_path, flavour, fcol, ecol):
    # the daophot flavour (``flux_fit``) comes back renamed ``flux``
    cat, before = _replace(monkeypatch, tmp_path, flavour,
                           record_daophot_prior=True)
    assert len(cat) == 3                                  # satstar 2 appended
    assert bool(cat['replaced_saturated'][0])
    assert cat[fcol][0] == 60000.0 and cat[ecol][0] == 10.0   # overwritten
    assert cat[R.PRIOR_FLUX][0] == 54000.0                # what it held
    assert cat[R.PRIOR_FLUX_ERR][0] == 4.0
    assert np.isclose(cat[R.PRIOR_RA][0], before.ra.deg[0], rtol=0, atol=1e-12)
    assert np.isclose(cat[R.PRIOR_DEC][0], before.dec.deg[0], rtol=0, atol=1e-12)
    for name in R.PRIOR_COLUMNS:                          # unmatched + appended
        assert np.isnan(cat[name][1:]).all()


@pytest.mark.parametrize('flavour', ['flux_fit', 'flux'])
def test_replace_saturated_writes_no_record_by_default(monkeypatch, tmp_path, flavour):
    cat, _ = _replace(monkeypatch, tmp_path, flavour)
    assert not [c for c in R.PRIOR_COLUMNS if c in cat.colnames]
    assert bool(cat['replaced_saturated'][0])


# ---- which rows qualify -----------------------------------------------------

def _case(ratio=0.9, sep_mas=5.0, survivor=None, flux=1e4, cfg=CFG):
    """Row 0 is the candidate: a satstar at the origin whose daophot row held
    ``ratio`` x its flux, ``sep_mas`` east of it.  ``survivor`` adds a row that
    stays, at that offset (arcsec)."""
    offsets = [(0.0, 0.0)] + ([survivor] if survivor is not None else [])
    n = len(offsets)
    table = _table(offsets, [flux] * n,
                   prior_flux=[flux * ratio] + [np.nan] * (n - 1),
                   prior_offsets=[(sep_mas / 1000.0, 0.0)] + offsets[1:])
    return bool(R.restorable(table, [0], [1] if survivor is not None else [],
                             'flux', cfg)[0])


@pytest.mark.parametrize('ratio, sep_mas, expected', [
    (0.90, 5.0, True),        # the o105 F212N rows: 0.80-1.06, 0.2-5 mas
    (0.80, 5.0, True),
    (1.06, 21.0, True),
    (0.75, 5.0, True),        # the lower end of the window is included
    (1.25, 5.0, False),       # the upper end is not
    (0.70, 5.0, False),
    (1.30, 5.0, False),
    (0.002, 70.0, False),     # the dqsat transients: 0.002-0.03 of the satstar
    (0.03, 63.0, False),
    (0.90, 29.0, True),
    (0.90, 31.0, False),
    (1.00, 70.0, False),      # in the window but not at the satstar
])
def test_flux_ratio_and_separation_window(ratio, sep_mas, expected):
    assert _case(ratio, sep_mas) is expected


@pytest.mark.parametrize('survivor, neighbour_min, expected', [
    (None, 0.1, True),
    ((0.17, 0.0), 0.1, True),     # the closest o105 F212N neighbour, 0.17"
    ((0.08, 0.0), 0.1, False),    # a surviving satstar row on the same star
    ((0.08, 0.0), 0.0, True),     # the check is off at 0
    ((0.0, -0.08), 0.1, False),
])
def test_a_surviving_satstar_row_near_the_daophot_position_blocks(
        survivor, neighbour_min, expected):
    cfg = R.Settings(0.75, 1.25, 0.03, neighbour_min)
    assert _case(survivor=survivor, cfg=cfg) is expected


@pytest.mark.parametrize('prior_flux', [np.nan, 0.0, -5.0])
def test_unusable_record_does_not_qualify(prior_flux):
    table = _table([(0.0, 0.0)], [1e4], prior_flux=[prior_flux],
                   prior_offsets=[(0.005, 0.0)])
    assert not R.restorable(table, [0], [], 'flux', CFG).any()


@pytest.mark.parametrize('flux', [0.0, -1.0, np.nan])
def test_unusable_satstar_flux_does_not_qualify(flux):
    table = _table([(0.0, 0.0)], [flux], prior_flux=[9e3],
                   prior_offsets=[(0.005, 0.0)])
    assert not R.restorable(table, [0], [], 'flux', CFG).any()


def test_table_without_the_record_restores_nothing():
    table = _table([(0.0, 0.0), (0.4, 0.0)], [1e4, 1e4])
    assert not R.restorable(table, [0, 1], [], 'flux', CFG).any()


@pytest.mark.parametrize('column', ['x', 'dy', 'x_fit', 'mag_ab'])
def test_table_with_columns_the_record_does_not_cover_restores_nothing(column):
    table = _table([(0.0, 0.0)], [1e4], prior_flux=[9e3],
                   prior_offsets=[(0.005, 0.0)])
    assert R.restorable(table, [0], [], 'flux', CFG).all()
    table[column] = [1.0]
    assert not R.restorable(table, [0], [], 'flux', CFG).any()


def test_each_row_is_judged_alone():
    table = _table([(0.0, 0.0), (0.4, 0.0), (0.0, 0.45)], [1e4, 1e4, 1e4],
                   prior_flux=[9e3, 20.0, 9.5e3],
                   prior_offsets=[(0.005, 0.0), (0.47, 0.0), (0.0, 0.452)])
    assert R.restorable(table, [0, 1, 2], [], 'flux', CFG).tolist() == [
        True, False, True]


# ---- restoring a row --------------------------------------------------------

def test_restore_turns_the_row_back_into_the_daophot_row():
    table = _table([(0.0, 0.0), (0.4, 0.0), (0.0, 0.45)], [1e4, 2e4, 3e4],
                   prior_flux=[9e3, 1.9e4, 2.9e4],
                   prior_offsets=[(0.005, 0.0), (0.401, 0.0), (0.0, 0.452)],
                   prior_err=[7.0, 8.0, 9.0], nframes=[2, 3, 4])
    keep = table.copy()
    R.restore(table, [0, 2], 'flux', 'flux_err')
    assert table['flux'].tolist() == [9e3, 2e4, 2.9e4]
    assert table['flux_err'].tolist() == [7.0, 20.0, 9.0]
    sep = table['skycoord'].separation(_sky([(0.005, 0.0), (0.4, 0.0), (0.0, 0.452)]))
    assert (sep.arcsec < 1e-6).all()
    assert table['replaced_saturated'].tolist() == [False, True, False]
    assert table['is_saturated'].all()
    assert table['satstar_nframes'].tolist() == [0, 3, 0]
    assert table['satstar_nmeas'].tolist() == [0, 4, 0]
    assert np.isnan(table['satstar_std_ra'][[0, 2]]).all()
    assert np.isnan(table['satstar_std_dec'][[0, 2]]).all()
    assert table[R.RESTORED_COLUMN].tolist() == [True, False, True]
    # row 1 and the record itself are as before
    for name in keep.colnames:
        if name == 'skycoord':
            assert keep['skycoord'][1].separation(table['skycoord'][1]).arcsec == 0
        else:
            assert np.array_equal(np.asarray(keep[name])[1], np.asarray(table[name])[1],
                                  equal_nan=True), name
    for name in R.PRIOR_COLUMNS:
        assert np.array_equal(keep[name], table[name], equal_nan=True)


def test_restore_of_no_row_still_adds_the_flag_column():
    table = _table([(0.0, 0.0)], [1e4], prior_flux=[9e3],
                   prior_offsets=[(0.005, 0.0)])
    R.restore(table, [], 'flux', 'flux_err')
    assert table[R.RESTORED_COLUMN].tolist() == [False]
    assert bool(table['replaced_saturated'][0])


# ---- select (what the dedup calls) ------------------------------------------

def test_select_picks_from_the_dropped_rows_and_reports(capsys):
    # rows 0-2 are the in-field candidates; row 1 stays, 0 and 2 are dropped
    table = _table([(0.0, 0.0), (0.6, 0.0), (0.0, 0.45)], [1e4, 1e4, 1e4],
                   prior_flux=[9e3, 9e3, 20.0],
                   prior_offsets=[(0.005, 0.0), (0.6, 0.0), (0.0, 0.52)])
    chosen = R.select(table, np.array([0, 1, 2]), np.array([True, False, True]),
                      'flux', CFG)
    assert chosen.tolist() == [0]
    assert 'kept 1 of the 2 dropped row(s) as daophot rows' in capsys.readouterr().out


def test_select_warns_when_the_merge_recorded_nothing(capsys):
    table = _table([(0.0, 0.0), (0.6, 0.0)], [1e4, 1e4])
    chosen = R.select(table, np.array([0, 1]), np.array([True, False]), 'flux', CFG)
    assert len(chosen) == 0
    out = capsys.readouterr().out
    assert 'WARNING' in out and R.PRIOR_FLUX in out and R.ENV_FRAC_MIN in out


def test_select_warns_for_columns_the_record_does_not_cover(capsys):
    table = _table([(0.0, 0.0), (0.6, 0.0)], [1e4, 1e4],
                   prior_flux=[9e3, 9e3], prior_offsets=[(0.005, 0.0), (0.6, 0.0)])
    table['x'] = [1.0, 2.0]
    chosen = R.select(table, np.array([0, 1]), np.array([True, False]), 'flux', CFG)
    assert len(chosen) == 0
    assert 'WARNING' in capsys.readouterr().out


# ---- the in-field dedup -----------------------------------------------------

# one clique, every pair under 1" and never co-fit: B is the best supported
# member and survives; A, C and D would be dropped
PT_B, PT_A, PT_C, PT_D = (0.0, 0.0), (0.4, 0.0), (0.0, 0.45), (0.115, 0.0)
FLUX = [37324.0, 64743.0, 12000.0, 20000.0]
FRAMES = [6, 1, 1, 2]
PRIOR_FLUX = [np.nan, 0.9 * 64743.0, 0.002 * 12000.0, 0.95 * 20000.0]
#: A's daophot row 5 mas off, C's 70 mas off, D's 20 mas off (and 0.095" from B)
PRIOR_OFFSETS = [PT_B, (0.405, 0.0), (0.0, 0.52), (0.095, 0.0)]


def _runs():
    # every per-exposure run fits one member: nothing was ever fit beside B
    return [_sky([p]) for p in (PT_B, PT_A, PT_C, PT_D, PT_B)]


def _dedup(table, tmp_path, **kwargs):
    return C._clean_offfov_dups_and_offfield(
        table, 'F480M', _data_i2d(tmp_path), str(tmp_path),
        cofit_positions=_runs(), **kwargs)


def _clique(with_record=True):
    return _table([PT_B, PT_A, PT_C, PT_D], FLUX, nframes=FRAMES,
                  prior_flux=PRIOR_FLUX if with_record else None,
                  prior_offsets=PRIOR_OFFSETS if with_record else None)


def test_off_drops_the_rows_as_before(clean_env, tmp_path):
    out, ndup, noff = _dedup(_clique(), tmp_path)
    assert (ndup, noff) == (3, 0)
    assert len(out) == 1 and out['flux'][0] == FLUX[0]
    assert R.RESTORED_COLUMN not in out.colnames


def test_on_keeps_the_restorable_row_as_a_daophot_row(clean_env, tmp_path, capsys):
    clean_env.setenv(R.ENV_FRAC_MIN, '0.75')
    table = _clique()
    before = {name: np.array(table[name]) for name in table.colnames
              if name != 'skycoord'}
    out, ndup, noff = _dedup(table, tmp_path)
    assert (ndup, noff) == (2, 0)                 # C (transient) and D (on B)
    assert len(out) == 2
    survivor, restored = out[0], out[1]
    # B stays the satstar row it was
    assert bool(survivor['replaced_saturated']) and survivor['flux'] == FLUX[0]
    assert survivor['satstar_nframes'] == 6 and not survivor[R.RESTORED_COLUMN]
    # A is its daophot row: flux, error, position
    assert restored['flux'] == pytest.approx(0.9 * 64743.0)
    assert restored['flux_err'] == 3.0
    assert not bool(restored['replaced_saturated']) and bool(restored['is_saturated'])
    assert restored[R.RESTORED_COLUMN]
    assert restored['satstar_nframes'] == 0 and restored['satstar_nmeas'] == 0
    assert np.isnan(restored['satstar_std_ra'])
    assert _sky([PRIOR_OFFSETS[1]])[0].separation(restored['skycoord']).arcsec < 1e-6
    assert 'kept 1 of the 3 dropped row(s) as daophot rows' in capsys.readouterr().out
    # the input table is not changed
    for name, values in before.items():
        assert np.array_equal(values, np.array(table[name]), equal_nan=True), name


def test_neighbour_check_off_also_keeps_the_row_on_the_survivor(clean_env, tmp_path):
    clean_env.setenv(R.ENV_FRAC_MIN, '0.75')
    clean_env.setenv(R.ENV_NEIGHBOUR_MIN, '0')
    out, ndup, _ = _dedup(_clique(), tmp_path)
    assert ndup == 1 and len(out) == 3            # only the transient goes


def test_a_second_pass_changes_nothing(clean_env, tmp_path):
    clean_env.setenv(R.ENV_FRAC_MIN, '0.75')
    out, _, _ = _dedup(_clique(), tmp_path)
    again, ndup, noff = _dedup(out, tmp_path)
    assert (ndup, noff) == (0, 0)
    assert len(again) == len(out)
    assert again['flux'].tolist() == out['flux'].tolist()
    assert again[R.RESTORED_COLUMN].tolist() == out[R.RESTORED_COLUMN].tolist()


def test_on_without_the_record_drops_as_before_and_warns(clean_env, tmp_path, capsys):
    clean_env.setenv(R.ENV_FRAC_MIN, '0.75')
    out, ndup, _ = _dedup(_clique(with_record=False), tmp_path)
    assert ndup == 3 and len(out) == 1
    assert out[R.RESTORED_COLUMN].tolist() == [False]
    assert 'WARNING' in capsys.readouterr().out


def test_on_leaves_a_table_with_nothing_to_drop_alone(clean_env, tmp_path):
    clean_env.setenv(R.ENV_FRAC_MIN, '0.75')
    table = _table([PT_B, (5.0, 5.0)], [1e4, 2e4], nframes=[3, 3],
                   prior_flux=[9e3, 1.9e4], prior_offsets=[(0.005, 0.0), (5.0, 5.0)])
    out, ndup, noff = _dedup(table, tmp_path)
    assert (ndup, noff) == (0, 0)
    assert out['flux'].tolist() == [1e4, 2e4]
    assert out['replaced_saturated'].all()
    assert not out[R.RESTORED_COLUMN].any()


def test_cleanup_reads_the_switch_through_satstar_dao_restore():
    src = ' '.join(inspect.getsource(C._clean_offfov_dups_and_offfield).split())
    assert '_dao_cfg = _dao_restore.settings()' in src
    assert ('dao_rows = _dao_restore.select( merged, inf_idx, _inf_drop, fcol, '
            '_dao_cfg) _inf_drop[np.isin(inf_idx, dao_rows)] = False '
            'drop[inf_idx[_inf_drop]] = True') in src
    assert '_dao_restore.settings()' in inspect.getsource(C.run_manual_pipeline)
