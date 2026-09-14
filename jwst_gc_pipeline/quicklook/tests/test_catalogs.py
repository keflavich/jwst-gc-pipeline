"""Which catalogs the builder picks, and that what it plots is Vega."""
import math

import numpy as np
import pytest

from jwst_gc_pipeline.quicklook import catalogs as C

astropy = pytest.importorskip('astropy')
from astropy.coordinates import SkyCoord            # noqa: E402
from astropy.table import Table                     # noqa: E402
import astropy.units as u                           # noqa: E402


def _jicama(dirpath, band, module, obs, stage, vetted=True, n=5, flux=100.0):
    tail = '_vetted' if vetted else ''
    name = (f'{band}_{module}_o{obs}_indivexp_merged_m{stage}_dao_basic{tail}.fits')
    tbl = Table({'flux': np.full(n, flux, dtype=float)})
    tbl['skycoord'] = SkyCoord(np.linspace(266.5, 266.51, n) * u.deg,
                               np.full(n, -28.7) * u.deg)
    tbl.meta['PIXSCALE'] = 0.031222960287669514
    tbl.meta['FILTER'] = band.upper()
    path = dirpath / name
    tbl.write(path, overwrite=True)
    return path


def _mast(dirpath, band, obs, n=5, abmag=18.0):
    tbl = Table({'aper_total_abmag': np.full(n, abmag, dtype=float)})
    tbl['sky_centroid'] = SkyCoord(np.linspace(266.5, 266.51, n) * u.deg,
                                   np.full(n, -28.7) * u.deg)
    path = dirpath / f'jw10678-o{obs}_t001_nircam_clear-{band}_cat.ecsv'
    tbl.write(path, overwrite=True)
    return path


def test_a_pointing_short_a_band_is_left_out_and_named(tmp_path):
    cat = tmp_path / 'catalogs'; cat.mkdir()
    _jicama(cat, 'f212n', 'nrca', '133', 2)
    assert C.find_jicama(cat) == {}
    assert C.incomplete_pointings(cat, tmp_path / 'nope') == {'o133': ['f480m']}


def test_the_pipeline_catalog_wins_over_mast_for_the_same_pointing(tmp_path):
    cat = tmp_path / 'catalogs'; cat.mkdir()
    mast = tmp_path / 'mast'; mast.mkdir()
    for band in C.BANDS:
        _jicama(cat, band, 'merged', '132', 2)
        _mast(mast, band, '132')
    chosen = C.choose_sources(C.find_jicama(cat), C.find_mast(mast))
    assert chosen['o132']['source'] == 'jicama'


def test_mast_carries_a_pointing_the_pipeline_has_not_reached(tmp_path):
    """The fallback is per POINTING, so a tile MAST covers still plots while
    the pipeline is working through the others."""
    cat = tmp_path / 'catalogs'; cat.mkdir()
    mast = tmp_path / 'mast'; mast.mkdir()
    for band in C.BANDS:
        _jicama(cat, band, 'merged', '132', 2)
        _mast(mast, band, '140')
    chosen = C.choose_sources(C.find_jicama(cat), C.find_mast(mast))
    assert chosen['o132']['source'] == 'jicama'
    assert chosen['o140']['source'] == 'mast'


def test_the_latest_merge_stage_wins_within_a_module(tmp_path):
    cat = tmp_path / 'catalogs'; cat.mkdir()
    for band in C.BANDS:
        _jicama(cat, band, 'merged', '132', 2)
        _jicama(cat, band, 'merged', '132', 4)
    assert '_m4_' in C.find_jicama(cat)['o132']['f212n'].name


def test_the_whole_tile_at_an_earlier_stage_beats_one_module_at_a_later_one(tmp_path):
    """The real shape of the data: the modules advance independently, so
    ``nrca`` reaches m3 while ``merged`` is still at m2 (o127 and o129 on
    2026-09-14).  Ranking by stage first plotted half of o127 -- 49k pairs
    instead of 86k -- under the label GC_127."""
    cat = tmp_path / 'catalogs'; cat.mkdir()
    for band in C.BANDS:
        _jicama(cat, band, 'nrca', '127', 3, vetted=True)
        _jicama(cat, band, 'nrcb', '127', 2, vetted=True)
        _jicama(cat, band, 'merged', '127', 2, vetted=True)
    assert '_merged_' in C.find_jicama(cat)['o127']['f212n'].name


def test_coverage_outranks_vetting(tmp_path):
    """A vetted single-module file would label half a tile with the whole
    tile's name; the unvetted two-module file is the tile."""
    cat = tmp_path / 'catalogs'; cat.mkdir()
    for band in C.BANDS:
        _jicama(cat, band, 'nrca', '132', 4, vetted=True)
        _jicama(cat, band, 'merged', '132', 4, vetted=False)
    assert '_merged_' in C.find_jicama(cat)['o132']['f212n'].name


def test_vetting_breaks_the_tie_within_one_module(tmp_path):
    cat = tmp_path / 'catalogs'; cat.mkdir()
    for band in C.BANDS:
        _jicama(cat, band, 'merged', '132', 4, vetted=False)
        _jicama(cat, band, 'merged', '132', 4, vetted=True)
    assert 'vetted' in C.find_jicama(cat)['o132']['f212n'].name


def test_a_two_module_pointing_prefers_the_merged_file(tmp_path):
    cat = tmp_path / 'catalogs'; cat.mkdir()
    for band in C.BANDS:
        _jicama(cat, band, 'nrca', '132', 2)
        _jicama(cat, band, 'nrcb', '132', 2)
        _jicama(cat, band, 'merged', '132', 2)
    assert '_merged_' in C.find_jicama(cat)['o132']['f212n'].name


def test_an_ab_magnitude_is_converted_and_not_passed_through(tmp_path):
    """The catalogs on this page are Vega. An AB magnitude read straight out of
    a MAST table would be 1.83 mag wrong in F212N and 3.44 in F480M, which is
    larger than every feature the diagram is meant to show."""
    mast = tmp_path / 'mast'; mast.mkdir()
    path = _mast(mast, 'f212n', '140', abmag=18.0)
    _, mag = C.load_band(path, 'f212n', allow_network=False)
    zp = C.FALLBACK_VEGA_ZEROPOINT_JY['f212n']
    expect = -2.5 * math.log10(C.AB_ZEROPOINT_JY * 10 ** (-0.4 * 18.0) / zp)
    assert mag[0] == pytest.approx(expect)
    assert mag[0] < 18.0 - 1.5      # Vega is brighter than AB in this band


def test_instrumental_flux_becomes_a_vega_magnitude(tmp_path):
    cat = tmp_path / 'catalogs'; cat.mkdir()
    path = _jicama(cat, 'f480m', 'merged', '132', 2, flux=1000.0)
    _, mag = C.load_band(path, 'f480m', allow_network=False)
    pix_sr = (0.031222960287669514 * math.pi / (180 * 3600)) ** 2
    flux_jy = 1000.0 * pix_sr * 1e6
    expect = -2.5 * math.log10(flux_jy / C.FALLBACK_VEGA_ZEROPOINT_JY['f480m'])
    assert mag[0] == pytest.approx(expect)


def test_the_pixel_area_key_written_by_the_merge_step_also_works():
    """Two writers, two spellings: an area in deg^2 and a side in arcsec."""
    side = 0.031222960287669514
    from_area = C._pixel_area_sr({'pixelscale_deg2': (side / 3600) ** 2 * u.deg ** 2})
    from_side = C._pixel_area_sr({'PIXSCALE': side})
    assert from_area == pytest.approx(from_side, rel=1e-10)


def test_only_pairs_inside_the_match_radius_are_kept():
    blue = SkyCoord([266.500, 266.600] * u.deg, [-28.7, -28.7] * u.deg)
    red = SkyCoord([266.500 + 1e-5, 266.900] * u.deg, [-28.7, -28.7] * u.deg)
    b, r = C.pair_bands(blue, [17.0, 18.0], red, [15.0, 16.0],
                        max_sep_arcsec=0.1)
    assert len(b) == 1 and b[0] == 17.0 and r[0] == 15.0


def test_the_cmd_is_blue_minus_red_against_blue(tmp_path):
    cat = tmp_path / 'catalogs'; cat.mkdir()
    paths = {C.BLUE_BAND: _jicama(cat, C.BLUE_BAND, 'merged', '132', 2, flux=100.0),
             C.RED_BAND: _jicama(cat, C.RED_BAND, 'merged', '132', 2, flux=100.0)}
    colour, mag = C.pointing_cmd(paths, allow_network=False)
    _, blue_mag = C.load_band(paths[C.BLUE_BAND], C.BLUE_BAND, allow_network=False)
    _, red_mag = C.load_band(paths[C.RED_BAND], C.RED_BAND, allow_network=False)
    assert mag[0] == pytest.approx(blue_mag[0])
    assert colour[0] == pytest.approx(blue_mag[0] - red_mag[0])


def test_the_cached_zeropoints_match_svo_to_a_millimag():
    """A stale hardcoded zeropoint is a photometric offset nobody re-derives.
    Skipped when SVO is unreachable -- the point is to catch drift, not to make
    the suite depend on a network service."""
    svo = pytest.importorskip('astroquery.svo_fps')
    try:
        table = svo.SvoFps.get_filter_list('JWST')
    except (OSError, ValueError) as err:
        pytest.skip(f'SVO unreachable: {err}')
    table.add_index('filterID')
    for band, cached in C.FALLBACK_VEGA_ZEROPOINT_JY.items():
        live = float(table.loc[f'JWST/NIRCam.{band.upper()}']['ZeroPoint'])
        drift = abs(-2.5 * math.log10(live / cached))
        assert drift < 0.001, f'{band}: cached zeropoint is {drift:.4f} mag off SVO'


def test_a_seed_source_list_is_not_mistaken_for_a_catalog(tmp_path):
    """``_i2dseed`` shares the naming convention but holds the seed positions a
    merge stage starts from, not its photometry."""
    cat = tmp_path / 'catalogs'; cat.mkdir()
    for band in C.BANDS:
        _jicama(cat, band, 'merged', '132', 2, vetted=False)
        seed = _jicama(cat, band, 'merged', '132', 3, vetted=False)
        seed.rename(seed.with_name(seed.name.replace('_dao_basic',
                                                     '_dao_basic_i2dseed')))
    chosen = C.find_jicama(cat)['o132']['f212n']
    assert 'i2dseed' not in chosen.name and '_m2_' in chosen.name
