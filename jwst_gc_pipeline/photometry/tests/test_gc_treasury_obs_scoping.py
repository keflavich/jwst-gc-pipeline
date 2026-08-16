"""Obs scoping for program 10678 (gc-treasury) -- the #416 stopgap.

139 Treasury tiles share ONE basepath, every tile images F212N+F480M, and the
(visit, vgroup, exposure) numbering restarts per observation, so without an
observation token the per-frame catalog names are byte-identical across tiles
and tile 2 silently overwrites tile 1 (the documented 2211 data-loss mode).

The stopgap has a WRITER side (``obs_token`` stamps ``_o{field}``; the merge
bakes it into the merged-catalog name) and a READER side (the merge glob, the
m7 seed reader, ``merge_daophot``'s input glob).  The failure mode these tests
pin is the reader/writer split #316 documents: writer stamps ``_oNNN`` while a
reader globs without it (finds nothing) or globs ``o*`` (pools tiles).
"""
import os
import types

import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from jwst_gc_pipeline.photometry import merge_catalogs as MC
from jwst_gc_pipeline.photometry import naming
from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
    _predict_tblfilename, obs_token)


def _options(field='001', proposal_id='10678', target='gc-treasury'):
    return types.SimpleNamespace(
        desaturated=False, bgsub=False, use_iter3_residual_bg=False,
        epsf=False, blur=False, group=False, iteration_label=None,
        proposal_id=proposal_id, field=field, target=target,
        cutout_region='')


# ---------------------------------------------------------------------------
# the merged-catalog token: per-obs for 10678 ONLY
# ---------------------------------------------------------------------------

def test_merged_catalog_token_is_per_obs_for_treasury_only():
    """gc2211 pools its five pointings into one UNTOKENED merged catalog by
    design; at 139 tiles that pooling is itself the corruption mode, so 10678
    scopes the merged names too."""
    assert naming.merged_catalog_obs_token('10678', '001') == '_o001'
    assert naming.merged_catalog_obs_token('10678', '002') == '_o002'
    assert naming.merged_catalog_obs_token('2211', '023') == ''
    assert naming.merged_catalog_obs_token('2221', '001') == ''
    assert naming.merged_catalog_obs_token('10678', None) == ''
    assert naming.merged_catalog_obs_token('10678', '') == ''


def test_treasury_is_a_multiobs_proposal():
    assert '10678' in naming.MULTIOBS_PROPOSALS
    assert '10678' in naming.PER_OBS_MERGED_PROPOSALS
    assert '2211' in naming.MULTIOBS_PROPOSALS
    assert '2211' not in naming.PER_OBS_MERGED_PROPOSALS


# ---------------------------------------------------------------------------
# per-frame writer: two tiles -> two names
# ---------------------------------------------------------------------------

def _perframe_name(basepath, field, exp=1):
    """The name the daophot per-frame WRITER uses (via its sanctioned
    predictor, kept in sync by --skip-if-done)."""
    return _predict_tblfilename(
        str(basepath), 'F480M', 'nrcblong', _options(field),
        1, '02101', exp, iteration_label='m2', method='daophot',
        basic_or_iterative='basic')


def test_two_tiles_write_distinct_per_frame_catalog_names(tmp_path):
    a = _perframe_name(tmp_path, '001')
    b = _perframe_name(tmp_path, '002')
    assert a != b
    assert '_o001_' in os.path.basename(a)
    assert '_o002_' in os.path.basename(b)


# ---------------------------------------------------------------------------
# round-trip: each tile's merge reads back ONLY its own per-frame catalogs
# ---------------------------------------------------------------------------

def _write_perframe(basepath, field, exp=1):
    (basepath / 'F480M').mkdir(exist_ok=True)
    fn = _perframe_name(basepath, field, exp)
    t = Table({'flux_fit': [1.0]})
    t.meta['FILENAME'] = (f'/x/jw10678{field}001_02101_{exp:05d}_nrcblong'
                          f'_align_o{field}_crf.fits')
    t.write(fn)
    return fn


def _stub_combine(seen):
    def _fake(tbls, offsets_table=None, filtername=None):
        seen.append(sorted(t.meta['FILENAME'] for t in tbls))
        return Table({'skycoord_avg': SkyCoord([10.0] * u.deg, [-5.0] * u.deg)})
    return _fake


def test_each_tile_merge_reads_back_only_its_own(tmp_path, monkeypatch):
    (tmp_path / 'catalogs').mkdir()
    for f in ('001', '002'):
        _write_perframe(tmp_path, f)
    seen = []
    monkeypatch.setattr(MC, 'combine_singleframe', _stub_combine(seen))

    for f in ('001', '002'):
        MC.merge_individual_frames(
            module='nrcblong', filtername='f480m', progid='10678',
            method='dao', suffix='_basic', target='gc-treasury',
            basepath=str(tmp_path), iteration_label='m2', field=f,
            do_replace_saturated=False)
        # the merge pooled THIS tile's frame and nothing else
        assert seen[-1] == [f'/x/jw10678{f}001_02101_00001_nrcblong'
                            f'_align_o{f}_crf.fits']
        # ...and wrote the per-obs tokened merged name that
        # cataloging._merged_path reads back
        out = (tmp_path / 'catalogs' /
               f'f480m_nrcblong_o{f}_indivexp_merged_m2_dao_basic.fits')
        assert out.exists(), sorted(os.listdir(tmp_path / 'catalogs'))


def test_treasury_merge_without_field_is_refused(tmp_path):
    """A field-less 10678 merge can only glob nothing (writer stamps _o{field})
    or pool tiles (_o*); both are silent, so it must refuse instead."""
    (tmp_path / 'catalogs').mkdir()
    _write_perframe(tmp_path, '001')
    with pytest.raises(ValueError, match='pass field='):
        MC.merge_individual_frames(
            module='nrcblong', filtername='f480m', progid='10678',
            method='dao', suffix='_basic', target='gc-treasury',
            basepath=str(tmp_path), iteration_label='m2',
            do_replace_saturated=False)


def test_gc2211_all_obs_pooling_is_unchanged(tmp_path, monkeypatch):
    """The field-less gc2211 merge keeps its design: pool EVERY obs (_o* glob)
    into the untokened all-obs merged catalog."""
    (tmp_path / 'catalogs').mkdir()
    (tmp_path / 'F480M').mkdir()
    for f in ('023', '046'):
        fn = _predict_tblfilename(
            str(tmp_path), 'F480M', 'nrcblong', _options(f, '2211', 'gc2211'),
            1, '02101', 1, iteration_label='m2', method='daophot',
            basic_or_iterative='basic')
        t = Table({'flux_fit': [1.0]})
        t.meta['FILENAME'] = (f'/x/jw02211{f}001_02101_00001_nrcblong'
                              f'_align_o{f}_crf.fits')
        t.write(fn)
    seen = []
    monkeypatch.setattr(MC, 'combine_singleframe', _stub_combine(seen))
    MC.merge_individual_frames(
        module='nrcblong', filtername='f480m', progid='2211',
        method='dao', suffix='_basic', target='gc2211',
        basepath=str(tmp_path), iteration_label='m2',
        do_replace_saturated=False)
    assert seen[-1] == [
        '/x/jw02211023001_02101_00001_nrcblong_align_o023_crf.fits',
        '/x/jw02211046001_02101_00001_nrcblong_align_o046_crf.fits']
    assert (tmp_path / 'catalogs' /
            'f480m_nrcblong_indivexp_merged_m2_dao_basic.fits').exists()


# ---------------------------------------------------------------------------
# m7 seed reader: reads the module-slot-tokened m6 vetted of ITS OWN tile
# ---------------------------------------------------------------------------

def _write_m6_vetted(base, filt, modtok='', endtok='', ra=266.0, dec=-28.9):
    t = Table({'skycoord': SkyCoord([ra] * u.deg, [dec] * u.deg),
               'qfit': [0.01], 'flux': [1000.0], 'flux_err': [1.0]})
    p = (base / 'catalogs' /
         f'{filt}_nrcblong{modtok}_indivexp_merged_resbgsub'
         f'_m6_dao_basic{endtok}_vetted.fits')
    t.write(p)
    return p


def test_m7_seed_reads_its_own_tiles_tokened_m6_vetted(tmp_path):
    from jwst_gc_pipeline.photometry.cataloging import _build_crossband_seed
    (tmp_path / 'catalogs').mkdir()
    for filt in ('f212n', 'f480m'):
        _write_m6_vetted(tmp_path, filt, modtok='_o001')
    out = _build_crossband_seed(str(tmp_path), ['nrcblong'],
                                ['F212N', 'F480M'], _options('001'))
    # the seed itself is per-obs too (obs_token end slot)
    assert os.path.basename(out) == 'crossband_seed_manual_o001.fits'
    seed = Table.read(out)
    assert len(seed) == 1
    assert seed['n_filt_confirmed'][0] == 2


def test_m7_seed_does_not_pool_another_tiles_m6(tmp_path):
    """Tile 002's seed must refuse rather than silently read tile 001's
    catalogs -- the pooling failure the obs scoping exists to prevent."""
    from jwst_gc_pipeline.photometry.cataloging import _build_crossband_seed
    (tmp_path / 'catalogs').mkdir()
    for filt in ('f212n', 'f480m'):
        _write_m6_vetted(tmp_path, filt, modtok='_o001')
    with pytest.raises(ValueError, match='no confirmed m6'):
        _build_crossband_seed(str(tmp_path), ['nrcblong'],
                              ['F212N', 'F480M'], _options('002'))


def test_m7_seed_still_reads_gc2211s_end_token_spelling(tmp_path):
    """gc2211's vetted names carry the token at the END
    (``..._m6_dao_basic_o023_vetted.fits``); the 10678 module-slot change must
    not regress that reader."""
    from jwst_gc_pipeline.photometry.cataloging import _build_crossband_seed
    (tmp_path / 'catalogs').mkdir()
    for filt in ('f212n', 'f480m'):
        _write_m6_vetted(tmp_path, filt, endtok='_o023')
    out = _build_crossband_seed(str(tmp_path), ['nrcblong'],
                                ['F212N', 'F480M'],
                                _options('023', '2211', 'gc2211'))
    assert len(Table.read(out)) == 1


# ---------------------------------------------------------------------------
# frozen-stage discriminator: another tile's merge is not OUR merge
# ---------------------------------------------------------------------------

def test_frozen_stage_discriminator_spells_this_tiles_merged_name(tmp_path):
    """m5 with no per-frame inputs: another tile's merged catalog must not
    stand in for our own (the stage did not run here -> skip), while OUR
    tokened merged catalog means the gate's inputs are broken (raise)."""
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        AstrometryRegressionError)
    from jwst_gc_pipeline.photometry.cataloging import (
        _run_astrometry_stage_checkpoint)
    (tmp_path / 'F480M').mkdir()
    (tmp_path / 'catalogs').mkdir()

    def _run():
        return _run_astrometry_stage_checkpoint(
            'm5', 'nrcblong', 'f480m', str(tmp_path), str(tmp_path), '10678',
            _options('001'), {}, context='test')

    # only tile 002's m5 merged exists -> the stage did not run for tile 001
    (tmp_path / 'catalogs' /
     'f480m_nrcblong_o002_indivexp_merged_resbgsub_m5_dao_basic.fits').touch()
    _run()      # must not raise

    # tile 001's own merged exists -> frozen stage with no inputs = broken gate
    (tmp_path / 'catalogs' /
     'f480m_nrcblong_o001_indivexp_merged_resbgsub_m5_dao_basic.fits').touch()
    with pytest.raises(AstrometryRegressionError, match='(?i)silently disabled'):
        _run()


# ---------------------------------------------------------------------------
# merge_daophot: per-obs input glob + the end-slot output token m8 reads
# ---------------------------------------------------------------------------

def _fake_svo(monkeypatch):
    jfilts = Table({'filterID': ['JWST/NIRCam.F212N', 'JWST/NIRCam.F480M'],
                    'ZeroPoint': [1.0, 1.0]})
    monkeypatch.setattr(
        MC, 'SvoFps', types.SimpleNamespace(get_filter_list=lambda fac: jfilts))


def _treasury_registry(monkeypatch):
    monkeypatch.setattr(MC, '_obs_filters_for',
                        lambda target: {'10678': ['f212n', 'f480m']})


def _write_m7_vetted(base, filt, field):
    t = Table({'skycoord': SkyCoord([266.0] * u.deg, [-28.9] * u.deg),
               'flux_fit': [1000.0], 'flux_err': [1.0]})
    t.meta['PIXSCALE'] = 0.063
    t.write(base / 'catalogs' /
            f'{filt}_nrcblong_o{field}_indivexp_merged_resbgsub'
            f'_m7_dao_basic_vetted.fits')


def test_merge_daophot_input_glob_is_obs_scoped(tmp_path, monkeypatch):
    """The old glob (``{filt}*{module}*indivexp_merged...``) would swallow ANY
    module-slot token, so tile 001's cross-band merge could pick up tile 002's
    vetted catalog when its own is absent."""
    (tmp_path / 'catalogs').mkdir()
    _fake_svo(monkeypatch)
    _treasury_registry(monkeypatch)
    # ONLY tile 002's m7 vetted catalogs exist
    for filt in ('f212n', 'f480m'):
        _write_m7_vetted(tmp_path, filt, '002')
    # tile 001's merge must see NO catalogs (never tile 002's)
    with pytest.raises(ValueError, match='No daophot basic catalogs found'):
        MC.merge_daophot(module='nrcblong', daophot_type='basic',
                         indivexp=True, resbgsub=True, iteration_label='m7',
                         target='gc-treasury', basepath=str(tmp_path),
                         filternames_override=['f212n', 'f480m'],
                         field='001', vetted=True)


def test_merge_daophot_reads_own_tile_and_stamps_the_end_slot_token(
        tmp_path, monkeypatch):
    """Tile 001's cross-band merge pools ONLY tile 001's per-filter vetted
    catalogs, and hands ``merge_catalogs`` the end-slot ``obs_suffix`` that
    cataloging's m7/m8 readers spell via ``obs_token``
    (``..._photometry_tables_merged_resbgsub_m7_o001.fits``)."""
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
        calls['files'] = sorted(t.meta['filename'] for t in tbls)
        calls['obs_suffix'] = kwargs['obs_suffix']

    monkeypatch.setattr(MC, 'merge_catalogs', _recorder)
    for field in ('001', '002'):
        for filt in ('f212n', 'f480m'):
            _write_m7_vetted(tmp_path, filt, field)
    MC.merge_daophot(module='nrcblong', daophot_type='basic',
                     indivexp=True, resbgsub=True, iteration_label='m7',
                     target='gc-treasury', basepath=str(tmp_path),
                     ref_filter='f480m',
                     filternames_override=['f212n', 'f480m'],
                     field='001', vetted=True)
    assert calls['obs_suffix'] == '_o001'
    assert [os.path.basename(f) for f in calls['files']] == [
        'f212n_nrcblong_o001_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits',
        'f480m_nrcblong_o001_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits']
