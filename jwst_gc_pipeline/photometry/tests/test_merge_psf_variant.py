"""The merges must not read the other PSF variant of their inputs (issue #1197).

Every per-frame catalog name carries the PSF variant (``_hybpsf`` when
``PSF_EPSF_CORE_DIR`` is set), but the per-filter merged name does not.  A
merge run without ``PSF_EPSF_CORE_DIR`` therefore globbed the older STPSF
per-frame catalogs of a tile and wrote over the merged catalog its hybrid-PSF
rerun had produced; the 10678 per-filter and cross-band m7 merges were built
from pre-roll per-frame catalogs that way.  The cross-band name does carry the
token, but the manual pipeline read it back without it, so a hybrid-PSF run
annotated, provenance-stamped and m8-filled the STPSF catalog of an earlier run.

The per-frame names below come from the WRITER's predictor
(``_predict_tblfilename``) under each ``PSF_EPSF_CORE_DIR`` setting, so the
guard's sibling spelling is tested against what is written.
"""
import importlib.util
import os
import pathlib
import types

import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

from jwst_gc_pipeline.photometry import merge_catalogs as MC
from jwst_gc_pipeline.photometry.crowdsource_catalogs_long import (
    _predict_tblfilename)

ENV = 'PSF_EPSF_CORE_DIR'
OPTOUT = 'GC_ALLOW_STALE_PSF_VARIANT'


def _options(field='001'):
    return types.SimpleNamespace(
        desaturated=False, bgsub=False, use_iter3_residual_bg=False,
        epsf=False, blur=False, group=False, iteration_label=None,
        proposal_id='10678', field=field, target='gc-treasury',
        cutout_region='')


def _perframe(basepath, monkeypatch, hybrid, mtime, exp=1):
    """Write one m2 per-frame catalog under the given PSF_EPSF_CORE_DIR setting."""
    if hybrid:
        monkeypatch.setenv(ENV, '/some/epsf/cores')
    else:
        monkeypatch.delenv(ENV, raising=False)
    fn = _predict_tblfilename(str(basepath), 'F480M', 'nrcblong', _options(),
                              1, '02101', exp, iteration_label='m2',
                              method='daophot', basic_or_iterative='basic')
    monkeypatch.delenv(ENV, raising=False)
    (basepath / 'F480M').mkdir(exist_ok=True)
    t = Table({'flux_fit': [1.0]})
    t.meta['FILENAME'] = f'/x/{"hyb" if hybrid else "stpsf"}_exp{exp:05d}_crf.fits'
    t.write(fn)
    os.utime(fn, (mtime, mtime))
    return fn


def _stub_combine(seen):
    def _fake(tbls, offsets_table=None, filtername=None):
        seen.append(sorted(t.meta['FILENAME'] for t in tbls))
        return Table({'skycoord_avg': SkyCoord([10.0] * u.deg, [-5.0] * u.deg)})
    return _fake


def _merge(tmp_path):
    MC.merge_individual_frames(
        module='nrcblong', filtername='f480m', progid='10678',
        method='dao', suffix='_basic', target='gc-treasury',
        basepath=str(tmp_path), iteration_label='m2', field='001',
        do_replace_saturated=False)
    return tmp_path / 'catalogs' / 'f480m_nrcblong_o001_indivexp_merged_m2_dao_basic.fits'


@pytest.fixture
def tree(tmp_path, monkeypatch):
    (tmp_path / 'catalogs').mkdir()
    monkeypatch.delenv(OPTOUT, raising=False)
    seen = []
    monkeypatch.setattr(MC, 'combine_singleframe', _stub_combine(seen))
    return tmp_path, seen


# ---------------------------------------------------------------------------
# per-filter merge (merge_individual_frames)
# ---------------------------------------------------------------------------

def test_stpsf_merge_is_refused_when_the_hybrid_rerun_is_newer(tree, monkeypatch):
    """The #1197 case: STPSF per-frame catalogs of 09-25, hybrid-PSF rerun of
    10-17, merge run without PSF_EPSF_CORE_DIR.  It read the 09-25 catalogs and
    overwrote the merged file; now it refuses."""
    tmp_path, seen = tree
    stp = _perframe(tmp_path, monkeypatch, hybrid=False, mtime=1_000_000)
    hyb = _perframe(tmp_path, monkeypatch, hybrid=True, mtime=2_000_000)
    assert '_hybpsf' in os.path.basename(hyb) and '_hybpsf' not in stp
    with pytest.raises(ValueError, match='newer sibling of the other PSF variant'):
        _merge(tmp_path)
    assert seen == []
    assert not list((tmp_path / 'catalogs').iterdir())


def test_hybrid_merge_reads_the_hybrid_catalogs_and_records_the_token(
        tree, monkeypatch):
    tmp_path, seen = tree
    _perframe(tmp_path, monkeypatch, hybrid=False, mtime=1_000_000)
    _perframe(tmp_path, monkeypatch, hybrid=True, mtime=2_000_000)
    monkeypatch.setenv(ENV, '/some/epsf/cores')
    out = _merge(tmp_path)
    assert seen == [['/x/hyb_exp00001_crf.fits']]
    meta = Table.read(out).meta
    assert meta['PSFTOKEN'] == '_hybpsf'
    assert '_hybpsf' in meta['FN0']


def test_stpsf_merge_runs_when_the_hybrid_catalogs_are_older(tree, monkeypatch):
    """A later STPSF per-frame run is the current one; merging it is allowed."""
    tmp_path, seen = tree
    _perframe(tmp_path, monkeypatch, hybrid=True, mtime=1_000_000)
    _perframe(tmp_path, monkeypatch, hybrid=False, mtime=2_000_000)
    out = _merge(tmp_path)
    assert seen == [['/x/stpsf_exp00001_crf.fits']]
    assert Table.read(out).meta['PSFTOKEN'] == 'none'


def test_hybrid_merge_is_refused_when_an_stpsf_rerun_is_newer(tree, monkeypatch):
    tmp_path, _ = tree
    _perframe(tmp_path, monkeypatch, hybrid=True, mtime=1_000_000)
    _perframe(tmp_path, monkeypatch, hybrid=False, mtime=2_000_000)
    monkeypatch.setenv(ENV, '/some/epsf/cores')
    with pytest.raises(MC.StalePSFVariantError):
        _merge(tmp_path)


def test_the_opt_out_merges_the_older_variant(tree, monkeypatch):
    tmp_path, seen = tree
    _perframe(tmp_path, monkeypatch, hybrid=False, mtime=1_000_000)
    _perframe(tmp_path, monkeypatch, hybrid=True, mtime=2_000_000)
    monkeypatch.setenv(OPTOUT, '1')
    _merge(tmp_path)
    assert seen == [['/x/stpsf_exp00001_crf.fits']]


@pytest.mark.parametrize('head,epsf_,tail,name,want', [
    ('_resbgsub', '', '_m7',
     'f480m_nrcblong_o001_visit001_vgroup02101_exp00001_resbgsub_m7_daophot_basic.fits',
     'f480m_nrcblong_o001_visit001_vgroup02101_exp00001_resbgsub_hybpsf_m7_daophot_basic.fits'),
    ('_resbgsub', '_hybpsf', '_m7',
     'f480m_nrcb_visit001_vgroup02101_exp00012_resbgsub_hybpsf_m7_chunk2of4_daophot_basic.fits',
     'f480m_nrcb_visit001_vgroup02101_exp00012_resbgsub_m7_chunk2of4_daophot_basic.fits'),
    ('', '_epsf', '',
     'f212n_nrca1_visit002_vgroup02101_exp00003_epsf_daophot_basic.fits',
     'f212n_nrca1_visit002_vgroup02101_exp00003_epsf_hybpsf_daophot_basic.fits'),
    ('', '', '_m2',
     'f212n_nrca1_visit002_vgroup02101_exp00003_hybpsf_m2_daophot_basic.fits',
     None),          # a hybrid file never matches the STPSF slot
])
def test_other_variant_path(head, epsf_, tail, name, want):
    got = MC._other_psf_variant_path(f'/bp/F480M/{name}', head, epsf_, tail)
    assert got == (None if want is None else f'/bp/F480M/{want}')


# ---------------------------------------------------------------------------
# cross-band merge (merge_daophot)
# ---------------------------------------------------------------------------

def _fake_svo(monkeypatch):
    jfilts = Table({'filterID': ['JWST/NIRCam.F212N', 'JWST/NIRCam.F480M'],
                    'ZeroPoint': [1.0, 1.0]})
    monkeypatch.setattr(
        MC, 'SvoFps', types.SimpleNamespace(get_filter_list=lambda fac: jfilts))


def _vetted(base, filt, meta):
    t = Table({'skycoord': SkyCoord([266.0] * u.deg, [-28.9] * u.deg),
               'flux_fit': [1000.0], 'flux_err': [1.0]})
    t.meta['PIXSCALE'] = 0.063
    t.meta.update(meta)
    t.write(base / 'catalogs' /
            f'{filt}_nrcblong_o001_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits')


def _crossband(tmp_path, monkeypatch, metas):
    (tmp_path / 'catalogs').mkdir()
    (tmp_path / 'reduction').mkdir()
    Table({'Filter': ['F212N', 'F480M'],
           'PSF FWHM (arcsec)': [0.072, 0.162],
           'PSF FWHM (pixel)': [2.3, 2.5]}).write(
        tmp_path / 'reduction' / 'fwhm_table.ecsv')
    _fake_svo(monkeypatch)
    monkeypatch.setattr(MC, '_obs_filters_for',
                        lambda target: {'10678': ['f212n', 'f480m']})
    monkeypatch.setattr(MC, 'sanity_check_individual_table', lambda tbl: None)
    calls = []
    monkeypatch.setattr(MC, 'merge_catalogs',
                        lambda tbls, **kw: calls.append(len(tbls)))
    for filt, meta in zip(('f212n', 'f480m'), metas):
        _vetted(tmp_path, filt, meta)
    MC.merge_daophot(module='nrcblong', daophot_type='basic',
                     indivexp=True, resbgsub=True, iteration_label='m7',
                     target='gc-treasury', basepath=str(tmp_path),
                     ref_filter='f480m', filternames_override=['f212n', 'f480m'],
                     field='001', progid='10678', vetted=True)
    return calls


_HYB_FN = {'FN0': 'f212n_nrcb1_o001_visit001_vgroup02101_exp00001_resbgsub_hybpsf_m7_daophot_basic.fits'}
_STP_FN = {'FN0': 'f480m_nrcblong_o001_visit001_vgroup02101_exp00001_resbgsub_m7_daophot_basic.fits'}


def test_hybrid_crossband_refuses_an_stpsf_per_filter_input(tmp_path, monkeypatch):
    """The o138/o139 case: F480M per-filter m7 still from the STPSF run, the
    cross-band merge run with PSF_EPSF_CORE_DIR set."""
    monkeypatch.delenv(OPTOUT, raising=False)
    monkeypatch.setenv(ENV, '/some/epsf/cores')
    with pytest.raises(ValueError, match='merged from another PSF variant'):
        _crossband(tmp_path, monkeypatch, [_HYB_FN, _STP_FN])


def test_the_psftoken_card_takes_precedence_over_the_fn_cards(tmp_path, monkeypatch):
    monkeypatch.delenv(OPTOUT, raising=False)
    monkeypatch.delenv(ENV, raising=False)
    with pytest.raises(MC.StalePSFVariantError):
        _crossband(tmp_path, monkeypatch,
                   [{'PSFTOKEN': 'none'}, {'PSFTOKEN': '_hybpsf', **_STP_FN}])


def test_matching_and_unrecorded_inputs_merge(tmp_path, monkeypatch):
    monkeypatch.delenv(OPTOUT, raising=False)
    monkeypatch.setenv(ENV, '/some/epsf/cores')
    # one recorded hybrid input, one with neither card (older products)
    assert _crossband(tmp_path, monkeypatch, [_HYB_FN, {}]) == [2]


# ---------------------------------------------------------------------------
# the manual pipeline's cross-band reader and the m8 names
# ---------------------------------------------------------------------------

_XB = 'basic_merged_indivexp_photometry_tables_merged_resbgsub'


def test_crossband_reader_spells_the_psf_token(monkeypatch):
    from jwst_gc_pipeline.photometry.cataloging import crossband_merged_path
    monkeypatch.setenv(ENV, '/some/epsf/cores')
    assert crossband_merged_path('/bp', 'm7', 'merged', '10678', '1') == (
        f'/bp/catalogs/{_XB}_hybpsf_m7_o001.fits')
    # m8 keeps the untokened name its readers spell
    assert crossband_merged_path('/bp', 'm8', 'merged', '10678', '1',
                                 psf_token='') == f'/bp/catalogs/{_XB}_m8_o001.fits'
    monkeypatch.delenv(ENV)
    assert crossband_merged_path('/bp', 'm7', 'merged', '10678', '1') == (
        f'/bp/catalogs/{_XB}_m7_o001.fits')


def test_the_manual_pipeline_derives_no_crossband_name_by_str_replace():
    """``_xb.replace('resbgsub_m7', ...)`` matches nothing once ``_xb`` carries
    the PSF token, so the m8 output would be the m7 catalog itself."""
    import inspect
    from jwst_gc_pipeline.photometry import cataloging
    code = '\n'.join(line for line in
                     inspect.getsource(cataloging.run_manual_pipeline).splitlines()
                     if not line.lstrip().startswith('#'))
    assert "_xb.replace(" not in code
    assert "indivexp_photometry_tables_'" not in code


_SCRIPT = (pathlib.Path(__file__).resolve().parents[3]
           / 'scripts' / 'reduction' / 'm8_merge_partials.py')


def _partials():
    spec = importlib.util.spec_from_file_location('m8_merge_partials', _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_m8_partial_merge_anchors_on_this_runs_psf_variant(tmp_path, monkeypatch):
    for stem in (f'{_XB}_m7_o001.fits', f'{_XB}_hybpsf_m7_o001.fits'):
        (tmp_path / stem).touch()
    mod = _partials()
    monkeypatch.setenv(ENV, '/some/epsf/cores')
    assert os.path.basename(mod._default_m7(str(tmp_path), 'merged', '001')) == (
        f'{_XB}_hybpsf_m7_o001.fits')
    monkeypatch.delenv(ENV)
    assert os.path.basename(mod._default_m7(str(tmp_path), 'merged', '001')) == (
        f'{_XB}_m7_o001.fits')


@pytest.mark.parametrize('anchor', [f'{_XB}_hybpsf_m7_o001.fits',
                                    f'{_XB}_m7_o001.fits'])
def test_m8_partial_names_never_name_the_anchor(anchor):
    mod = _partials()
    m7 = f'/bp/catalogs/{anchor}'
    assert mod._m8_name(m7, 'resbgsub_m8') == f'/bp/catalogs/{_XB}_m8_o001.fits'
    assert mod._m8_name(m7, 'resbgsub_m8_partial_F212N') == (
        f'/bp/catalogs/{_XB}_m8_partial_F212N_o001.fits')
