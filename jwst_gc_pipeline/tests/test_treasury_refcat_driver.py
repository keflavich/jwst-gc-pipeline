"""The per-tile reference-catalog driver for program 10678.

What this pins, and what breaks without it:

* ``--observations`` range parsing.  ``088-139`` must be the 52 scheduled
  visits INCLUSIVE of both ends and zero-padded; an off-by-one or an unpadded
  ``88`` sends the builder a token no product name matches.
* the observation token in the filename.  ``pick_refcat`` matches
  ``_o(\\d{3})\\.fits$``; a tile written without the token, or with an
  unpadded one, is handed to every observation alike -- the gc2211 o023
  failure, a -9.28" "correction" measured against a neighbour's sky.
* the cone covering BOTH apertures.  Every 10678 visit is NIRCam prime with
  MIRI in parallel 7.79' away, so a cone fitted to the prime alone (the
  per-tile builder's 6' default) contains no MIRI sky at all.
* the registry.  10678 must register no catalog until per-tile files exist,
  so the lookup raises instead of pointing 139 tiles at one file.
"""
import importlib.util
import os

import numpy as np
import pytest
from astropy.table import Table

from jwst_gc_pipeline import fields as F
from jwst_gc_pipeline.reduction.build_gaia_virac2_refcat_byquery import (
    refcat_filename)

SCRIPTS = os.path.join(os.path.dirname(__file__), '..', '..',
                       'scripts', 'reduction')


def _load():
    spec = importlib.util.spec_from_file_location(
        'build_treasury_refcats',
        os.path.join(SCRIPTS, 'build_treasury_refcats.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DRIVER = _load()


# --------------------------------------------------------------------------
# --observations
# --------------------------------------------------------------------------

def test_a_range_is_inclusive_and_zero_padded():
    """``088-139`` is the 52 scheduled visits, both ends included."""
    got = DRIVER.parse_observations('088-139')
    assert got[0] == '088' and got[-1] == '139'
    assert len(got) == 52
    assert got == sorted(got)
    assert all(len(o) == 3 and o.isdigit() for o in got)


def test_a_range_pads_an_unpadded_bound():
    assert DRIVER.parse_observations('1-3') == ['001', '002', '003']
    assert DRIVER.parse_observations('88') == ['088']


def test_commas_and_ranges_mix_and_deduplicate():
    assert DRIVER.parse_observations('001,005,088-090,088') == [
        '001', '005', '088', '089', '090']


def test_all_and_none_mean_every_observation():
    assert DRIVER.parse_observations('all') is None
    assert DRIVER.parse_observations(None) is None


@pytest.mark.parametrize('spec', ['139-088', 'ninety', '12-x', ''])
def test_a_malformed_range_raises(spec):
    with pytest.raises(ValueError):
        DRIVER.parse_observations(spec)


# --------------------------------------------------------------------------
# the per-tile token
# --------------------------------------------------------------------------

def test_every_tile_gets_its_own_tokened_filename():
    """The token is what ``pick_refcat`` selects on; without it one catalog
    is handed to all 139 tiles."""
    names = {obsid: refcat_filename('2026.69', obsid)
             for obsid in ('088', '089', '139')}
    assert names['088'] == 'gaia_virac2_refcat_epoch2026.69_o088.fits'
    assert len(set(names.values())) == 3
    # the same token, unpadded, must name the SAME file
    assert refcat_filename('2026.69', '88') == names['088']


def test_the_build_command_carries_the_tile_token_and_cone():
    tile = DRIVER.Tile(obsid='088', target='GC_88', ra=266.70769,
                       dec=-28.72457, radius_deg=0.1434,
                       instruments=('miri', 'nircam'))
    cmd = DRIVER.build_command(tile, '/base', epoch=2026.69, python='py')
    assert cmd[:3] == [
        'py', '-m',
        'jwst_gc_pipeline.reduction.build_gaia_virac2_refcat_byquery']
    assert cmd[cmd.index('--obs-token') + 1] == '088'
    assert cmd[cmd.index('--base') + 1] == '/base'
    assert float(cmd[cmd.index('--ra') + 1]) == pytest.approx(266.70769, abs=1e-5)
    assert float(cmd[cmd.index('--radius') + 1]) == pytest.approx(0.1434, abs=1e-4)


def test_a_tile_with_no_epoch_refuses_rather_than_guessing():
    tile = DRIVER.Tile(obsid='088', target='GC_88', ra=266.7, dec=-28.7,
                       radius_deg=0.14)
    with pytest.raises(ValueError, match='no epoch'):
        DRIVER.build_command(tile, '/base', epoch=None)


def test_an_existing_tile_refcat_is_found_by_token_at_any_epoch(tmp_path):
    catalogs = tmp_path / 'catalogs'
    catalogs.mkdir()
    (catalogs / 'gaia_virac2_refcat_epoch2026.71_o088.fits').write_text('x')
    (catalogs / 'gaia_virac2_refcat_epoch2026.69_o089.fits').write_text('x')
    assert DRIVER.existing_refcat(tmp_path, '088').endswith('_o088.fits')
    assert DRIVER.existing_refcat(tmp_path, '88').endswith('_o088.fits')
    assert DRIVER.existing_refcat(tmp_path, '090') is None
    # an untokened catalog in the same directory is NOT this tile's
    (catalogs / 'gaia_virac2_refcat_epoch2026.69.fits').write_text('x')
    assert DRIVER.existing_refcat(tmp_path, '090') is None


# --------------------------------------------------------------------------
# the cone covers both apertures
# --------------------------------------------------------------------------

def _rows(obsid, ra0, dec0):
    """Two rows shaped like MAST's planned 10678 output: a NIRCam prime
    footprint and a MIRI parallel footprint offset from it.

    The offsets are the measured 10678 geometry (2026-09-03, planned
    ``s_region`` polygons, identical for all 139 tiles): the MIRI aperture
    centre sits 7.79' from the NIRCam prime centre.
    """
    cosdec = np.cos(np.radians(dec0))

    def poly(dra_arcmin, ddec_arcmin, half_ra, half_dec):
        ra = ra0 + dra_arcmin / 60.0 / cosdec
        dec = dec0 + ddec_arcmin / 60.0
        corners = []
        for sra, sdec in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
            corners += [ra + sra * half_ra / 60.0 / cosdec,
                        dec + sdec * half_dec / 60.0]
        return 'POLYGON ' + ' '.join(f'{v:.8f}' for v in corners)

    return [
        {'obs_id': f'jw10678{obsid}001_xx101_00001_nircam',
         'instrument_name': 'NIRCAM/IMAGE', 'target_name': f'GC_{int(obsid)}',
         's_ra': ra0, 's_dec': dec0, 't_min': np.nan,
         's_region': poly(0.0, 0.0, 1.1, 2.55)},
        {'obs_id': f'jw10678{obsid}001_xx201_00001_miri',
         'instrument_name': 'MIRI/IMAGE', 'target_name': f'GC_{int(obsid)}',
         's_ra': ra0, 's_dec': dec0, 't_min': np.nan,
         's_region': poly(0.0, 7.79, 0.62, 0.94)},
    ]


def _released_rows(obsid, ra0, dec0, t_min=61293.0):
    """The same tile as :func:`_rows`, spelled the way MAST returns a program
    that has actually flown.

    Verified live 2026-09-03: proposal 1182 returns
    ``jw01182-o001_t001_nircam_clear-f200w`` and 2221 returns
    ``jw02221-o001_t001_nircam_clear-f187n``.  The observation number moves
    behind an ``-o``, and these rows -- and only these -- carry a finite
    ``t_min``.  10678 returns the planned spelling today because nothing of it
    has been observed; from the 2026-09-10 delivery it returns this one.
    """
    planned = _rows(obsid, ra0, dec0)
    planned[0]['obs_id'] = f'jw10678-o{obsid}_t001_nircam_f212n'
    planned[1]['obs_id'] = f'jw10678-o{obsid}_t001_miri_f770w'
    for row in planned:
        row['t_min'] = t_min
    return planned


def _table(rows):
    return Table(rows=rows, names=list(rows[0]))


def test_the_cone_reaches_the_miri_parallel_not_just_the_prime():
    """A cone fitted to the NIRCam prime alone leaves the parallel with no
    reference.  Every MIRI vertex must fall inside the cone the driver picks.
    """
    table = _table(_rows('088', 266.6926, -28.7879))
    tile, = DRIVER.tiles_from_table(table, margin_arcmin=0.0)
    assert tile.instruments == ('miri', 'nircam')
    centre = DRIVER._unit_vectors([tile.ra], [tile.dec])[0]
    for row in table:
        ra, dec = DRIVER._polygon_vertices(row['s_region'])
        angles = np.degrees(np.arccos(np.clip(
            DRIVER._unit_vectors(ra, dec) @ centre, -1, 1)))
        assert angles.max() <= tile.radius_deg + 1e-9, row['instrument_name']
    # and it is genuinely wider than the builder's 6' default, which is what
    # made the parallel unreachable
    assert tile.radius_deg * 60.0 > 6.0


def test_the_margin_is_added_on_top_of_the_footprints():
    table = _table(_rows('088', 266.6926, -28.7879))
    bare, = DRIVER.tiles_from_table(table, margin_arcmin=0.0)
    padded, = DRIVER.tiles_from_table(table, margin_arcmin=1.5)
    assert padded.radius_deg == pytest.approx(bare.radius_deg + 1.5 / 60.0)


def test_tiles_are_grouped_per_observation_and_do_not_share_a_cone():
    rows = _rows('088', 266.6926, -28.7879) + _rows('089', 266.6574, -28.7694)
    tiles = DRIVER.tiles_from_table(_table(rows), margin_arcmin=1.5)
    assert [t.obsid for t in tiles] == ['088', '089']
    assert tiles[0].ra != tiles[1].ra
    selected = DRIVER.tiles_from_table(
        _table(rows), observations=['089'], margin_arcmin=1.5)
    assert [t.obsid for t in selected] == ['089']


def test_a_tile_with_no_footprint_falls_back_to_the_target_position():
    rows = _rows('088', 266.6926, -28.7879)
    for row in rows:
        row['s_region'] = ''
    tile, = DRIVER.tiles_from_table(_table(rows), fallback_radius_deg=0.16)
    assert tile.radius_deg == pytest.approx(0.16)
    assert tile.ra == pytest.approx(266.6926, abs=1e-4)


def test_a_published_t_min_sets_the_epoch_and_beats_the_fallback():
    rows = _rows('088', 266.6926, -28.7879)
    for row in rows:
        row['t_min'] = 61293.0  # 2026-09-10
    tile, = DRIVER.tiles_from_table(_table(rows))
    assert tile.epoch == pytest.approx(2026.69, abs=0.01)


def test_the_emitted_registry_block_names_the_file_the_build_writes():
    tiles = [DRIVER.Tile(obsid='088', target='GC_88', ra=1.0, dec=2.0,
                         radius_deg=0.14)]
    block = DRIVER.registry_block(tiles, {'088': 2026.69})
    assert 'reference_catalog:' in block
    assert ("'088': catalogs/gaia_virac2_refcat_epoch2026.69_o088.fits"
            in block)


# --------------------------------------------------------------------------
# the registry
# --------------------------------------------------------------------------

def test_the_registry_no_longer_serves_one_catalog_to_every_tile():
    """Companion to the driver: with no per-tile entry the lookup raises
    rather than handing 139 tiles one CMZ-wide file."""
    for obsid in ('001', '088', '139'):
        with pytest.raises(F.FieldRegistryError):
            F.reference_catalog_path('10678', obsid)


def test_a_wildcard_obsid_field_still_takes_per_observation_keys(monkeypatch):
    """What makes registering per tile possible at all.

    docs/FIELDS.md used to say a wildcard-obsid proposal needs the default
    because "there is nothing for per-obsid keys to hang on".  A
    ``reference_catalog`` key is looked up by the obsid STRING, with no
    reference to the declared obsid list, so the wildcard and per-tile keys
    coexist -- and an unregistered tile still raises rather than borrowing a
    neighbour's file.
    """
    wildcard = F.Field('wildcard-refcat-probe', root='blue', observations=(
        F.Obs(proposal='4242', obsids={'nircam': ('*',)},
              reference_catalogs={'088': ('catalogs/o088.fits',)}),))
    monkeypatch.setattr(F, 'FIELDS', F.FIELDS + (wildcard,))
    monkeypatch.setitem(F.BY_NAME, 'wildcard-refcat-probe', wildcard)
    got, = F.reference_catalog_candidates('4242', '088')
    assert got.endswith('catalogs/o088.fits')
    with pytest.raises(F.FieldRegistryError):
        F.reference_catalog_candidates('4242', '089')


# --------------------------------------------------------------------------
# the two spellings MAST uses for an observation number
# --------------------------------------------------------------------------

def test_the_observation_number_is_read_from_both_mast_spellings():
    """A released program spells the observation behind an ``-o``; a program
    that has not flown spells it inline.  10678 changes spelling on
    2026-09-10, so the driver has to read both or select nothing at all."""
    planned = 'jw10678088001_02101_00001_nrca1'
    released = 'jw10678-o088_t001_nircam_f212n'
    assert DRIVER.observation_number(planned) == '088'
    assert DRIVER.observation_number(released) == '088'
    # and the observation-level form of another program, measured live
    assert DRIVER.observation_number('jw01182-o001_t001_nircam_clear-f200w',
                                     proposal='1182') == '001'
    assert DRIVER.observation_number('jw02221-o001_t001_nircam_clear-f187n',
                                     proposal='2221') == '001'


@pytest.mark.parametrize('obs_id', [
    'jw01182-o001_t001_nircam_clear-f200w',   # another proposal entirely
    'jw10678-c1001_t001_nircam_f212n',        # a multi-observation candidate
    'jw10678-a3001_t001_nircam_f212n',        # an association, not one tile
    'jw10678_t001_nircam_f212n',              # no observation at all
    'jw1067808_x',                            # truncated
])
def test_a_row_that_names_no_single_observation_is_dropped(obs_id):
    assert DRIVER.observation_number(obs_id) is None


def test_a_released_program_still_yields_its_tiles():
    """The regression this pins: with the observation number parsed off the
    planned spelling alone, a released 10678 row matched nothing, so
    ``tiles_from_table`` returned [] and the driver printed "no MAST rows for
    observation(s) ['088']" and exited 0 having built no catalog."""
    table = _table(_released_rows('088', 266.6926, -28.7879))
    tiles = DRIVER.tiles_from_table(table, observations=['088'])
    assert [t.obsid for t in tiles] == ['088']
    assert tiles[0].instruments == ('miri', 'nircam')
    assert tiles[0].radius_deg * 60.0 > 6.0


def test_a_released_row_is_what_carries_the_epoch():
    """``t_min`` is NaN on every planned row, so the per-tile epoch can only
    ever come from a released row -- which is exactly the row the positional
    parser dropped."""
    table = _table(_released_rows('088', 266.6926, -28.7879, t_min=61293.0))
    tile, = DRIVER.tiles_from_table(table)
    assert tile.epoch == pytest.approx(2026.69, abs=0.01)


def test_the_two_spellings_of_one_observation_merge_into_one_tile():
    """A program mid-delivery returns both.  They are the same sky and must
    not become two tiles competing for one filename."""
    rows = (_rows('088', 266.6926, -28.7879)
            + _released_rows('088', 266.6926, -28.7879))
    tiles = DRIVER.tiles_from_table(_table(rows))
    assert [t.obsid for t in tiles] == ['088']
    assert tiles[0].epoch == pytest.approx(2026.69, abs=0.01)


def test_the_catalogs_directory_is_created_before_the_first_query(tmp_path):
    """The per-tile builder writes with ``ref.write(out)`` only after both
    cone queries have finished, so a missing ``catalogs/`` costs the whole
    query and then fails.  gc-treasury is a new field: neither its basepath
    nor ``catalogs/`` existed on either root on 2026-09-03."""
    base = tmp_path / 'gc-treasury'
    assert not (base / 'catalogs').exists()
    made = DRIVER.ensure_catalog_dir(base)
    assert os.path.isdir(made)
    # idempotent: a rerun over a populated directory is fine
    assert DRIVER.ensure_catalog_dir(base) == made
