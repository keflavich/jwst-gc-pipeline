"""Observation scoping of the satstar channel (issue #925).

gc-treasury (proposal 10678) keeps 139 tiles under one tree, so every tile's
per-exposure satstar catalogs sit in the same ``{FILTER}/pipeline`` directory.
``load_satstar_catalog`` globbed all of them for every tile's merge, and
``replace_saturated`` appends each unmatched satstar as a new row, so a tile's
merged catalog carried the whole program's saturated stars.  Measured on o132
m8: F480M 11.5-12.0 had 3,830 rows outside the tile's exposures against 124 inside, and
36-49% of every tile's rows were other tiles' stars.

These tests pin the fix: a scoped merge reads only its own observation's
catalogs, caches them under a per-observation name, drops any row outside the
union of its exposures, and fails loudly when that footprint cannot be built.
A field whose tree holds one observation takes the unchanged path.
"""
import os

import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from jwst_gc_pipeline.photometry import merge_catalogs as MC

PIXSCALE = 63.0 / 3600.0 / 1000.0   # NIRCam LW, deg/px
SHAPE = (128, 128)                   # (ny, nx); a detector ~8" on a side
RA0, DEC0 = 266.95, -28.42


def _wcs(ra, dec):
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [64.5, 64.5]
    w.wcs.crval = [ra, dec]
    w.wcs.cdelt = [-PIXSCALE, PIXSCALE]
    return w


def _write_frame(path, wcs_now):
    hdu1 = fits.ImageHDU(np.zeros(SHAPE, dtype='float32'), name='SCI')
    hdu1.header.update(wcs_now.to_header(relax=True))
    fits.HDUList([fits.PrimaryHDU(), hdu1]).writeto(path, overwrite=True)


def _write_satstar(frame_path, wcs_now, x, y, flux=5000.0):
    sky = wcs_now.pixel_to_world(np.asarray(x, float), np.asarray(y, float))
    tbl = Table({'xcentroid': np.asarray(x, float),
                 'ycentroid': np.asarray(y, float),
                 'flux_fit': np.full(len(x), flux),
                 'flux_err': np.full(len(x), 10.0),
                 'x_fit': np.asarray(x, float), 'y_fit': np.asarray(y, float),
                 'x_err': np.full(len(x), 0.01), 'y_err': np.full(len(x), 0.01)})
    tbl['skycoord_fit'] = sky
    tbl.write(str(frame_path).replace('.fits', '_m7_satstar_catalog.fits'),
              overwrite=True)
    return sky


def _plain_wcs(frame):
    """Stand-in for the GWCS: the synthetic frames carry a plain TAN header."""
    return WCS(fits.getheader(frame, 'SCI'))


@pytest.fixture
def two_obs_tree(tmp_path, monkeypatch):
    """A gc-treasury-shaped tree: two observations of one proposal, 2' apart
    on the sky, two exposures each, all in one F480M/pipeline directory."""
    monkeypatch.setattr(MC, '_footprint_frame_wcs', _plain_wcs)
    pdir = tmp_path / 'F480M' / 'pipeline'
    pdir.mkdir(parents=True)
    (tmp_path / 'catalogs').mkdir()
    sky = {}
    for obs, dra in (('001', 0.0), ('002', 2.0 / 60.0)):
        stars = []
        for expo in (1, 2):
            w = _wcs(RA0 + dra, DEC0)
            frame = (pdir / f'jw10678{obs}001_02101_0000{expo}_nrcalong'
                     f'_destreak_o{obs}_crf.fits')
            _write_frame(str(frame), w)
            stars.append(_write_satstar(frame, w, [30.0, 90.0], [40.0, 100.0]))
        sky[obs] = stars[0]
    return tmp_path, sky


def _nearest_sep_arcsec(tbl, targets):
    got = SkyCoord(tbl['skycoord_fit'])
    _, sep, _ = SkyCoord(targets).match_to_catalog_sky(got)
    return sep.to_value(u.arcsec)


# --- the scope rule ------------------------------------------------------

@pytest.mark.parametrize('proposal, field, expected', [
    ('10678', '132', '_o132'),   # gc-treasury: 139 tiles in one tree
    ('10678', '5', '_o005'),     # normalised like every other obs token
    ('1979', '3', '_o003'),      # m4: two pointings share one tree
    ('2221', '001', ''),         # brick 6-band half: one obs per filter dir
    ('6151', '001', ''),         # w51: single observation
    ('7213', '001', ''),         # ngc6334: proposal token, not an obs token
    (None, '132', ''),
    ('10678', None, ''),
])
def test_scope_follows_the_per_frame_obs_token(proposal, field, expected):
    assert MC.satstar_obs_scope(proposal, field) == expected


def test_cache_path_is_per_observation_only_when_scoped():
    assert MC.consolidated_satstar_cache_path('/b', 'F480M') == \
        '/b/catalogs/f480m_consolidated_satstar_catalog.fits'
    assert MC.consolidated_satstar_cache_path('/b', 'F480M', '_o132') == \
        '/b/catalogs/f480m_o132_consolidated_satstar_catalog.fits'


def test_file_membership_reads_the_exposure_name():
    name = 'jw10678132001_02101_00001_nrcalong_destreak_o132_crf_m7_satstar_catalog.fits'
    assert MC.satstar_catalog_in_observation(name, '10678', '_o132')
    assert not MC.satstar_catalog_in_observation(name, '10678', '_o133')
    # same observation number, different proposal
    assert not MC.satstar_catalog_in_observation(name, '2211', '_o132')
    # a non-JWST name is accepted only with the pipeline's own token
    assert MC.satstar_catalog_in_observation('cut_o132_crf_m7_satstar_catalog.fits',
                                             '10678', '_o132')
    assert not MC.satstar_catalog_in_observation('cut_crf_m7_satstar_catalog.fits',
                                                 '10678', '_o132')


# --- two observations in one tree ---------------------------------------

def test_scoped_load_does_not_pool_the_other_observation(two_obs_tree):
    base, sky = two_obs_tree
    basepath = str(base) + '/'

    o1 = MC.load_satstar_catalog('f480m', target='gc-treasury', basepath=basepath,
                                 proposal_id='10678', field='001')
    o2 = MC.load_satstar_catalog('f480m', target='gc-treasury', basepath=basepath,
                                 proposal_id='10678', field='002')

    # each observation holds its own two stars, deduplicated across exposures
    assert len(o1) == 2 and len(o2) == 2
    assert np.all(_nearest_sep_arcsec(o1, sky['001']) < 0.01)
    assert np.all(_nearest_sep_arcsec(o2, sky['002']) < 0.01)
    # and nothing within an arcminute of the other observation's stars
    assert np.all(_nearest_sep_arcsec(o1, sky['002']) > 60)
    assert np.all(_nearest_sep_arcsec(o2, sky['001']) > 60)

    # one cache per observation, each stamped with its scope
    for obs in ('001', '002'):
        cache = base / 'catalogs' / f'f480m_o{obs}_consolidated_satstar_catalog.fits'
        assert cache.exists()
        assert Table.read(cache).meta['SATOBSSC'] == f'_o{obs}'
    assert not (base / 'catalogs' / 'f480m_consolidated_satstar_catalog.fits').exists()


def test_unscoped_load_still_pools_which_is_the_defect(two_obs_tree):
    """The pre-fix behaviour, kept for a caller that names no observation: all
    four stars.  This is what every 10678 tile's merge used to receive."""
    base, _ = two_obs_tree
    pooled = MC.load_satstar_catalog('f480m', target='gc-treasury',
                                     basepath=str(base) + '/')
    assert len(pooled) == 4


def test_scoped_cache_is_reused_and_not_served_to_another_scope(two_obs_tree, capsys):
    base, sky = two_obs_tree
    basepath = str(base) + '/'
    MC.load_satstar_catalog('f480m', target='gc-treasury', basepath=basepath,
                            proposal_id='10678', field='001')
    capsys.readouterr()
    again = MC.load_satstar_catalog('f480m', target='gc-treasury', basepath=basepath,
                                    proposal_id='10678', field='001')
    out = capsys.readouterr().out
    assert 'Using consolidated satstar catalog' in out, out
    assert np.all(_nearest_sep_arcsec(again, sky['001']) < 0.01)

    # A cache whose recorded scope disagrees is rebuilt, never served.
    cache = base / 'catalogs' / 'f480m_o001_consolidated_satstar_catalog.fits'
    tbl = Table.read(cache)
    tbl.meta['SATOBSSC'] = '_o002'
    tbl.write(cache, overwrite=True)
    capsys.readouterr()
    MC.load_satstar_catalog('f480m', target='gc-treasury', basepath=basepath,
                            proposal_id='10678', field='001')
    assert 'Building consolidated satstar catalog' in capsys.readouterr().out


def test_replace_saturated_appends_only_this_observations_stars(two_obs_tree,
                                                                  monkeypatch):
    """End to end through the append path: o001's merge must not gain o002's
    saturated stars as new rows."""
    base, sky = two_obs_tree

    class _FakeSvo:
        @staticmethod
        def get_filter_list(_):
            t = Table({'filterID': ['JWST/NIRCam.F480M'], 'ZeroPoint': [250.0]})
            t.add_index('filterID')
            return t
    monkeypatch.setattr(MC, 'SvoFps', _FakeSvo)

    # one faint daophot row in o001, far from every satstar
    faint = SkyCoord([RA0 + 1.0 / 3600.0] * u.deg, [DEC0 - 3.0 / 3600.0] * u.deg)
    cat = Table({'skycoord': faint, 'flux': [10.0], 'dflux': [1.0],
                 'x': [60.0], 'y': [20.0], 'dx': [0.01], 'dy': [0.01]})
    MC.replace_saturated(cat, 'f480m', target='gc-treasury',
                         basepath=str(base) + '/',
                         proposal_id='10678', field='001')

    assert len(cat) == 3          # the daophot row + o001's two satstars
    added = SkyCoord(cat['skycoord'][np.asarray(cat['replaced_saturated'], bool)])
    _, sep1, _ = added.match_to_catalog_sky(sky['001'])
    _, sep2, _ = added.match_to_catalog_sky(sky['002'])
    assert np.all(sep1.to_value(u.arcsec) < 0.01)
    assert np.all(sep2.to_value(u.arcsec) > 60)


@pytest.mark.parametrize('func', ['replace_saturated', 'flag_near_saturated'])
def test_merge_entry_points_forward_the_scope(func, monkeypatch):
    seen = {}

    def _fake(filtername, target='brick', basepath='', proposal_id=None, field=None):
        seen.update(proposal_id=proposal_id, field=field)
        return None
    monkeypatch.setattr(MC, 'load_satstar_catalog', _fake)
    cat = Table({'flux': [1.0]})
    getattr(MC, func)(cat, 'f480m', target='gc-treasury', basepath='/nowhere/',
                      proposal_id='10678', field='132')
    assert seen == {'proposal_id': '10678', 'field': '132'}


# --- footprint guard ----------------------------------------------------

def test_footprint_guard_drops_a_row_outside_the_observation(two_obs_tree, capsys):
    """A row whose position lies outside every exposure of the observation is
    never kept.  Here it comes from a centroid far off the array, which the
    FITS/SIP WCS extrapolates to sky arcminutes away."""
    base, sky = two_obs_tree
    pdir = base / 'F480M' / 'pipeline'
    w = _wcs(RA0, DEC0)
    frame = pdir / 'jw10678001001_02101_00003_nrcalong_destreak_o001_crf.fits'
    _write_frame(str(frame), w)
    # x=-3: centroid 3 px off the array edge (inside the 1" margin) -> kept;
    # x=3000: ~3' away -> dropped.
    _write_satstar(frame, w, [-3.0, 3000.0], [64.0, 64.0])
    offchip_edge = w.pixel_to_world(-3.0, 64.0)
    far = w.pixel_to_world(3000.0, 64.0)

    got = MC.load_satstar_catalog('f480m', target='gc-treasury',
                                  basepath=str(base) + '/',
                                  proposal_id='10678', field='001')
    out = capsys.readouterr().out
    assert 'footprint guard dropped 1 of' in out, out
    gsky = SkyCoord(got['skycoord_fit'])
    assert np.min(gsky.separation(far).to_value(u.arcsec)) > 60
    assert np.min(gsky.separation(offchip_edge).to_value(u.arcsec)) < 0.01
    assert len(got) == 3


def test_footprint_mask_geometry(tmp_path, monkeypatch):
    monkeypatch.setattr(MC, '_footprint_frame_wcs', _plain_wcs)
    w = _wcs(RA0, DEC0)
    frame = tmp_path / 'jw10678001001_02101_00001_nrcalong_o001_crf.fits'
    _write_frame(str(frame), w)
    cat = str(frame).replace('.fits', '_m7_satstar_catalog.fits')
    x = np.array([64.0, 0.0, 127.0, -10.0, -20.0, 400.0, np.nan])
    y = np.array([64.0, 0.0, 127.0, 64.0, 64.0, 64.0, 64.0])
    pts = w.pixel_to_world(np.nan_to_num(x, nan=64.0), y)
    ra = np.asarray(pts.ra.deg)
    ra[np.isnan(x)] = np.nan
    pts = SkyCoord(ra * u.deg, np.asarray(pts.dec.deg) * u.deg)
    inside = MC.satstar_in_observation_footprint(pts, [cat], margin_arcsec=1.0)
    # centre, both corners, 10 px (0.63") off the edge are in; 20 px (1.26")
    # and 336 px off are out; a NaN position is out.
    np.testing.assert_array_equal(inside,
                                  [True, True, True, True, False, False, False])


def test_footprint_guard_fails_loudly_without_a_gwcs(two_obs_tree, monkeypatch):
    """The synthetic frames carry no GWCS; with the real accessor the guard must
    raise rather than fall back to SIP or skip itself."""
    base, _ = two_obs_tree
    monkeypatch.setattr(MC, '_footprint_frame_wcs',
                        lambda f: MC.frame_wcs(f, require_gwcs=True))
    with pytest.raises(MC.SatstarFootprintError):
        MC.load_satstar_catalog('f480m', target='gc-treasury',
                                basepath=str(base) + '/',
                                proposal_id='10678', field='001')


def test_footprint_guard_fails_loudly_without_a_frame(tmp_path):
    cat = str(tmp_path / 'jw10678001001_02101_00001_nrcalong_o001_crf_m7_satstar_catalog.fits')
    with pytest.raises(MC.SatstarFootprintError):
        MC.satstar_in_observation_footprint(
            SkyCoord([RA0] * u.deg, [DEC0] * u.deg), [cat])


# --- single-observation fields are unchanged -----------------------------

def _brick_tree(tmp_path):
    pdir = tmp_path / 'F182M' / 'pipeline'
    pdir.mkdir(parents=True)
    (tmp_path / 'catalogs').mkdir()
    w = _wcs(266.54, -28.71)
    for expo in (1, 2):
        frame = pdir / f'jw02221001001_02101_0000{expo}_nrca1_o001_crf.fits'
        _write_frame(str(frame), w)
        _write_satstar(frame, w, [30.0, 90.0], [40.0, 100.0])
    return tmp_path


def test_single_observation_field_takes_the_unchanged_path(tmp_path, monkeypatch):
    """brick 2221/001 with its field named resolves to no scope: same cache
    name, no scope key, no footprint guard, and the same table as a call that
    names no observation."""
    def _no_guard(*a, **k):
        raise AssertionError('footprint guard must not run on an unscoped field')
    monkeypatch.setattr(MC, 'satstar_in_observation_footprint', _no_guard)

    base = _brick_tree(tmp_path)
    basepath = str(base) + '/'
    cache = base / 'catalogs' / 'f182m_consolidated_satstar_catalog.fits'

    plain = MC.load_satstar_catalog('f182m', target='brick', basepath=basepath)
    plain_bytes = cache.read_bytes()
    os.remove(cache)
    named = MC.load_satstar_catalog('f182m', target='brick', basepath=basepath,
                                    proposal_id='2221', field='001')
    assert cache.exists()
    assert sorted(p.name for p in (base / 'catalogs').iterdir()) == [cache.name]
    assert 'SATOBSSC' not in Table.read(cache).meta
    assert cache.read_bytes() == plain_bytes
    assert plain.colnames == named.colnames
    for col in plain.colnames:
        if col == 'skycoord_fit':
            assert np.all(SkyCoord(plain[col]).separation(
                SkyCoord(named[col])).to_value(u.mas) == 0)
        else:
            np.testing.assert_array_equal(np.asarray(plain[col]),
                                          np.asarray(named[col]))


def test_cross_band_merge_forwards_the_scope(tmp_path, monkeypatch):
    """The cross-band merge (merge_daophot -> merge_catalogs) calls
    replace_saturated a second time; it must name the same observation."""
    from .test_gc_treasury_obs_scoping import (
        _fake_svo, _treasury_registry, _write_m7_vetted)
    (tmp_path / 'catalogs').mkdir()
    (tmp_path / 'reduction').mkdir()
    Table({'Filter': ['F212N', 'F480M'],
           'PSF FWHM (arcsec)': [0.072, 0.162],
           'PSF FWHM (pixel)': [2.3, 2.5]}).write(
        tmp_path / 'reduction' / 'fwhm_table.ecsv')
    _fake_svo(monkeypatch)
    _treasury_registry(monkeypatch)
    monkeypatch.setattr(MC, 'sanity_check_individual_table', lambda tbl: None)
    calls = {}

    def _recorder(tbls, **kwargs):
        calls.update(kwargs)
    monkeypatch.setattr(MC, 'merge_catalogs', _recorder)
    for filt in ('f212n', 'f480m'):
        _write_m7_vetted(tmp_path, filt, '001')
    MC.merge_daophot(module='nrcblong', daophot_type='basic',
                     indivexp=True, resbgsub=True, iteration_label='m7',
                     target='gc-treasury', basepath=str(tmp_path),
                     ref_filter='f480m', filternames_override=['f212n', 'f480m'],
                     field='001', vetted=True)
    assert calls['satstar_field'] == '001'
    assert MC.satstar_obs_scope(calls['satstar_proposal_id'],
                                calls['satstar_field']) == '_o001'


def _write_seed_cache(path, ra):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Table({'skycoord_fit': SkyCoord([ra] * u.deg, [DEC0] * u.deg)}).write(path)


def test_sibling_seed_reads_the_observations_own_cache(tmp_path):
    """The opt-in sibling-exposure seed (``--satstar-sibling-seed``) reads the
    consolidated cache.  A scoped merge writes ``{filt}_oNNN_...``, so the seed
    must read that name; the unscoped (program-wide) file on a shared tree is
    other tiles' stars.  Unscoped callers read the unchanged name."""
    base = str(tmp_path)
    _write_seed_cache(MC.consolidated_satstar_cache_path(base, 'F480M'), RA0)
    _write_seed_cache(MC.consolidated_satstar_cache_path(base, 'F480M', '_o132'),
                      RA0 + 0.01)
    scoped = MC.satstar_sibling_seed_positions('F480M', base, verbose=False,
                                               proposal_id='10678', field='132')
    assert scoped.ra.deg[0] == pytest.approx(RA0 + 0.01)
    unscoped = MC.satstar_sibling_seed_positions('F480M', base, verbose=False)
    assert unscoped.ra.deg[0] == pytest.approx(RA0)
    single = MC.satstar_sibling_seed_positions('F480M', base, verbose=False,
                                               proposal_id='2221', field='001')
    assert single.ra.deg[0] == pytest.approx(RA0)


def test_partner_seed_reader_is_scoped():
    """The partner-band seed in ``_prepare_frame_for_photometry`` resolves its
    cache through the scoped helpers (a source check: the reader sits in the
    middle of the frame-preparation function)."""
    import inspect

    from jwst_gc_pipeline.photometry import cataloging as C
    src = inspect.getsource(C._prepare_frame_for_photometry)
    assert "_partner}_consolidated_satstar_catalog.fits" not in src
    assert 'consolidated_satstar_cache_path(basepath, _partner, _pscope)' in src
    assert 'satstar_sibling_seed_positions(\n            filtername, basepath, proposal_id=proposal_id, field=field)' in src
