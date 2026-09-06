"""Release selection of per-filter vetted catalogs and combined tables.

`discover_catalogs` `continue`s on a name it cannot parse, so an unmatched
filename is indistinguishable from an absent file: the field simply ships
nothing and the manifest looks complete.  Every case below is a real on-disk
spelling that produced exactly that silence.
"""
import importlib.util
import os

import pytest

_SR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..',
    'scripts', 'release', 'stage_release.py'))
_spec = importlib.util.spec_from_file_location('stage_release_sel', _SR)
sr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sr)


def _m(name):
    return sr.VETTED_RE.match(name)


# --- the observation token appears in either position ------------------------

def test_token_after_the_module_is_matched():
    # brick NIRCam.  Only the after-dao_basic spelling was accepted before, so
    # 180 of brick's 190 vetted tables were invisible.
    m = _m('f115w_merged_o004_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits')
    assert m is not None
    assert m.group('filt') == 'f115w'
    assert m.group('module') == 'merged'
    assert sr.vetted_obs(m) == 'o004'
    assert m.group('iter') == 'resbgsub_m7'


def test_token_after_dao_basic_is_still_matched():
    # brick MIRI -- the spelling that used to be the only one accepted.
    m = _m('f2550w_mirimage_indivexp_merged_resbgsub_m5_dao_basic_o002_vetted.fits')
    assert m is not None
    assert sr.vetted_obs(m) == 'o002'


def test_no_token_at_all_is_matched_with_obs_none():
    # arches / sgra: single-observation fields write no token.
    m = _m('f212n_nrca_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits')
    assert m is not None
    assert sr.vetted_obs(m) is None


# --- compound observation tokens --------------------------------------------

@pytest.mark.parametrize('name,obs', [
    ('f1280w_mirimage_indivexp_merged_m2_dao_basic_o002-998_vetted.fits', 'o002-998'),
    ('f770w_mirimage_indivexp_merged_m3_dao_basic_o001-002_vetted.fits', 'o001-002'),
])
def test_jointly_registered_observations(name, obs):
    # sgrb2 MIRI and sickle register two observations as one; `o\d+` stopped at
    # the hyphen and dropped them.
    m = _m(name)
    assert m is not None
    assert sr.vetted_obs(m) == obs


# --- per-detector modules ----------------------------------------------------

@pytest.mark.parametrize('module', ['nrca1', 'nrca2', 'nrca3', 'nrca4',
                                    'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4',
                                    'nrcalong', 'nrcblong', 'merged', 'mirimage'])
def test_every_module_spelling_on_disk(module):
    # gc2211 writes per-DETECTOR tables; 80 of its 103 were invisible.
    m = _m(f'f200w_{module}_indivexp_merged_m4_dao_basic_vetted.fits')
    assert m is not None, module
    assert m.group('module') == module


# --- filters with a trailing width digit -------------------------------------

@pytest.mark.parametrize('filt', ['f150w2', 'f322w2'])
def test_wide_filters(filt):
    # ngc6397 and m4 matched ZERO vetted tables without this.
    m = _m(f'{filt}_nrcb_indivexp_merged_m3_dao_basic_vetted.fits')
    assert m is not None
    assert m.group('filt') == filt


def test_ordinary_filters_still_match():
    for filt in ('f115w', 'f187n', 'f410m', 'f1130w', 'f2550w'):
        assert _m(f'{filt}_merged_indivexp_merged_m5_dao_basic_vetted.fits')


# --- things that must NOT match ----------------------------------------------

@pytest.mark.parametrize('name', [
    # not a vetted table
    'f115w_merged_indivexp_merged_m5_dao_basic.fits',
    # a combined table, not per-filter
    'basic_merged_indivexp_photometry_tables_merged_resbgsub_m8.fits',
    # unknown module
    'f115w_nrcc_indivexp_merged_m5_dao_basic_vetted.fits',
    # hand-curated date-stamped catalog
    'f115w_merged_indivexp_merged_m5_dao_basic_20260713_vetted.fits',
])
def test_non_vetted_names_are_rejected(name):
    assert _m(name) is None


# --- the untokened combined table --------------------------------------------

def test_untokened_combined_is_dropped_when_per_obs_tables_exist(tmp_path):
    # brick shipped an Aug-29 untokened `_m7` pair beside the correct
    # `_m8_o001`/`_m8_o004`, and the quality-cut half carried the 2221 oksep
    # suffix while holding 1182's bands -- #661 one layer up.
    #
    # The suffix comes from the field's own helper rather than a literal: it
    # is built per field from that field's registered proposals, and spelling
    # one program's token in a test is what `test_no_hardcoded_qualcuts_token`
    # exists to catch (a hardcoded token once skipped every other field's
    # quality-filtered table).
    qc = sr.field_qualcuts_suffix('brick')
    assert qc, 'brick should have a quality-cut suffix'
    cat = tmp_path / 'catalogs'
    cat.mkdir()
    base = 'basic_merged_indivexp_photometry_tables_merged'
    for n in (f'{base}_resbgsub_m7.fits',
              f'{base}_resbgsub_m7{qc}.fits',
              f'{base}_resbgsub_m8_o001.fits',
              f'{base}_resbgsub_m8_o004.fits'):
        (cat / n).write_text('')
    cfg = {'data_dir': tmp_path, 'observations': ['o001', 'o004']}
    items = sr.discover_catalogs(cfg, 'brick')
    full = [i for i in items if i['kind'] == 'catalog_full']
    assert sorted(i['observation'] for i in full) == ['o001', 'o004']
    assert not any('_m7' in i['src'] for i in items)


def test_untokened_combined_is_kept_when_no_per_obs_table_exists(tmp_path):
    # arches / sgra have no per-observation tables; the untokened name is the
    # real deliverable and must still ship.
    cat = tmp_path / 'catalogs'
    cat.mkdir()
    base = 'basic_merged_indivexp_photometry_tables_merged'
    (cat / f'{base}_resbgsub_m8.fits').write_text('')
    cfg = {'data_dir': tmp_path}
    items = sr.discover_catalogs(cfg, 'sgra')
    full = [i for i in items if i['kind'] == 'catalog_full']
    assert len(full) == 1
    assert full[0]['observation'] is None


# --- same-run pairing must speak one language about observations -------------

def _img(filt, src, obs=None):
    return {'category': 'image', 'kind': 'science', 'filter': filt,
            'observation': obs, 'src': src}


def _cat(filt, obs=None, module=None):
    # `category` is present on every real manifest item; same_run_pairs reads
    # it directly, so a synthetic item without one is not a faithful stand-in.
    return {'category': 'catalog', 'kind': 'catalog_per_filter_vetted',
            'filter': filt, 'observation': obs, 'module': module,
            'src': f'/x/{filt.lower()}_cat.fits'}


def test_image_takes_its_observation_from_its_own_filename():
    # brick: images are NOT laid out per observation, so discover_images left
    # `observation: None` while the catalogs carried o001/o004.  The key never
    # matched and all ten filters reported "no catalog partner" -- a refusal
    # that only appeared once the selector fix made catalogs visible at all.
    pairs, unpaired = sr.same_run_pairs([
        _img('F115W', '/x/jw01182-o004_t001_nircam_clear-f115w-merged_i2d.fits'),
        _cat('F115W', obs='o004'),
    ])
    assert not unpaired
    assert len(pairs) == 1
    assert pairs[0][0][1] == 'o004'


def test_an_explicit_observation_tag_wins_over_the_filename():
    pairs, _ = sr.same_run_pairs([
        _img('F115W', '/x/jw01182-o004_t001_x-f115w-merged_i2d.fits', obs='o009'),
        _cat('F115W', obs='o009'),
    ])
    assert len(pairs) == 1


def test_untokened_catalogs_still_pair_with_tokened_image_names():
    # arches / sgra: single-observation fields write no token on the catalog,
    # but their mosaics are named jw02045-o001_...  Requiring equal tokens
    # would unpair every image on exactly the fields that paired correctly
    # before the filename fallback was added.
    pairs, unpaired = sr.same_run_pairs([
        _img('F212N', '/x/jw02045-o001_t001_nircam_clear-f212n-nrca_i2d.fits'),
        _cat('F212N', obs=None, module='nrca'),
    ])
    assert not unpaired
    assert len(pairs) == 1


def test_a_genuinely_missing_partner_is_still_reported():
    # The fallback must not pair an image with a different filter's catalog.
    pairs, unpaired = sr.same_run_pairs([
        _img('F115W', '/x/jw01182-o004_t001_x-f115w-merged_i2d.fits'),
        _cat('F200W', obs='o004'),
    ])
    assert len(pairs) == 0
    assert len(unpaired) == 1
    assert unpaired[0][0] == 'F115W'


@pytest.mark.parametrize('name,expect', [
    ('jw01182-o004_t001_nircam_clear-f115w-merged_i2d.fits', 'o004'),
    ('jw02221-o001_t001_nircam_clear-f182m-merged_i2d.fits', 'o001'),
    ('basic_merged_indivexp_photometry_tables_merged_m8.fits', None),
    ('', None),
])
def test_obs_token_from_name(name, expect):
    assert sr._obs_token_from_name(name) == expect
