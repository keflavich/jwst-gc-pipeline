"""#983: the satstar readers that pooled every cataloging phase on disk.

#982 limited ``load_satstar_catalog`` / ``load_rejected_satstar_catalog`` to
one phase per exposure.  The pooled wing-calibration table, the partner-band
seed fallback and (tested in test_offfov_infield_dedup) the in-field co-fit
positions now follow the same rule, and the rejected channel reads the same
product as the accepted one when the NIRCam merged_i2d primary exists.
"""
import os

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table

from jwst_gc_pipeline.photometry import cataloging as C
from jwst_gc_pipeline.photometry import merge_catalogs as MC
from jwst_gc_pipeline.photometry.satstar_phase_selection import (
    satstar_file_phase_token, satstar_phase_group_key)

EXP = 'jw10678040001_02101_00005_nrcalong'


def _calib(pdir, exposure, phase, ratio, n=4, tag='destreak_o040_crf',
           mtime=None):
    fn = pdir / f'{exposure}_{tag}_{phase}_wingcal_calibrators.fits'
    Table({'rmask_px': [3], 'ratio_median': [ratio], 'n_stars': [n],
           'ratio_madstd': [0.02]}).write(fn, overwrite=True)
    if mtime is not None:
        os.utime(fn, (mtime, mtime))
    return fn


@pytest.fixture
def tree(tmp_path):
    pdir = tmp_path / 'F410M' / 'pipeline'
    pdir.mkdir(parents=True)
    (tmp_path / 'catalogs').mkdir()
    return tmp_path, pdir


def test_wingcal_calibrator_names_carry_a_phase_and_an_exposure():
    name = f'{EXP}_destreak_o040_crf_resbgsub_m5_wingcal_calibrators.fits'
    assert satstar_file_phase_token(name) == 'm5'
    assert satstar_phase_group_key(name) == EXP
    assert satstar_phase_group_key('a_m12_wingcal_calibrators.fits') == 'a'


@pytest.mark.parametrize('phase, want', [
    ('m4', 1.10),       # m12 is the latest phase at or before m4
    ('m7', 1.30),
    (None, 1.30),       # latest per exposure
    ('m2', 1.10),       # the first phase's files are named _m12
])
def test_pooled_wingcal_reads_one_phase_per_exposure(tree, phase, want):
    base, pdir = tree
    _calib(pdir, EXP, 'm12', 1.10)
    _calib(pdir, EXP, 'm7', 1.30)
    pooled = MC.build_pooled_wingcal('f410m', basepath=str(base), phase=phase)
    assert pooled['ratio'].tolist() == [pytest.approx(want)]
    assert pooled['n_frames'].tolist() == [1]


def test_pooled_wingcal_path_is_per_phase(tree):
    base, _ = tree
    assert MC.pooled_wingcal_path(str(base), 'F410M').endswith(
        'catalogs/f410m_pooled_wingcal.ecsv')
    assert MC.pooled_wingcal_path(str(base), 'F410M', 'm7').endswith(
        'catalogs/f410m_pooled_wingcal_m7.ecsv')
    for first in ('m1', 'm2', 'm12'):
        assert MC.pooled_wingcal_path(str(base), 'F410M', first).endswith(
            'catalogs/f410m_pooled_wingcal_m12.ecsv')
    assert MC.pooled_wingcal_path(str(base), 'F410M', 'resbgsub_m6').endswith(
        '_m6.ecsv')


def _legacy_table(base, ratio, mtime):
    """A pooled table as the pre-#983 code wrote it: no WCALSEL."""
    fn = base / 'catalogs' / 'f410m_pooled_wingcal_m7.ecsv'
    Table({'rmask_px': [3], 'ratio': [ratio], 'n_stars_total': [4],
           'n_frames': [1]}).write(fn, format='ascii.ecsv')
    os.utime(fn, (mtime, mtime))
    return fn


def test_pooled_wingcal_without_a_selection_signature_is_rebuilt(tree):
    # Before #983 the table was built once and served forever, whatever
    # phases and code version first built it.
    base, pdir = tree
    _calib(pdir, EXP, 'm7', 1.30, mtime=1_000)
    _legacy_table(base, 1.99, mtime=2_000)          # newer than the inputs
    pooled = MC.load_pooled_wingcal('f410m', str(base), phase='m7')
    assert pooled['ratio'].tolist() == [pytest.approx(1.30)]
    assert Table.read(MC.pooled_wingcal_path(str(base), 'f410m', 'm7'),
                      format='ascii.ecsv').meta['WCALSEL']


def test_pooled_wingcal_is_reused_while_its_inputs_are_unchanged(tree, monkeypatch):
    base, pdir = tree
    _calib(pdir, EXP, 'm7', 1.30, mtime=1_000)
    MC.load_pooled_wingcal('f410m', str(base), phase='m7')

    def _no_rebuild(*a, **k):
        raise AssertionError('rebuilt a fresh pooled wingcal table')
    monkeypatch.setattr(MC, 'build_pooled_wingcal', _no_rebuild)
    pooled = MC.load_pooled_wingcal('f410m', str(base), phase='m7')
    assert pooled['ratio'].tolist() == [pytest.approx(1.30)]


def test_pooled_wingcal_is_rebuilt_when_a_calibrator_file_is_refit(tree):
    # A refit with changed satstar code rewrites the file under the same name.
    base, pdir = tree
    _calib(pdir, EXP, 'm7', 1.30, mtime=1_000)
    MC.load_pooled_wingcal('f410m', str(base), phase='m7')
    later = os.path.getmtime(MC.pooled_wingcal_path(str(base), 'f410m', 'm7')) + 10
    _calib(pdir, EXP, 'm7', 1.20, mtime=later)
    pooled = MC.load_pooled_wingcal('f410m', str(base), phase='m7')
    assert pooled['ratio'].tolist() == [pytest.approx(1.20)]


def test_pooled_wingcal_is_rebuilt_when_the_selection_changes(tree):
    # Another exposure's calibrators appear (a sibling observation finished),
    # with an OLDER mtime than the table: the mtime check alone would miss it.
    base, pdir = tree
    _calib(pdir, EXP, 'm7', 1.30, n=4, mtime=1_000)
    MC.load_pooled_wingcal('f410m', str(base), phase='m7')
    _calib(pdir, 'jw10678040001_02101_00006_nrcalong', 'm7', 1.10, n=4,
           mtime=500)
    pooled = MC.load_pooled_wingcal('f410m', str(base), phase='m7')
    assert pooled['ratio'].tolist() == [pytest.approx(1.20)]
    assert pooled['n_frames'].tolist() == [2]


def test_apply_pooled_wingcal_uses_the_merge_phase(tree):
    base, pdir = tree
    _calib(pdir, EXP, 'm12', 1.10)
    _calib(pdir, EXP, 'm7', 1.30)
    cat = Table({'flux_fit': [1000.0], 'flux_err': [10.0],
                 'wingcal_ratio': [1.0], 'wingcal_rmask': [3.0]})
    out = MC.apply_pooled_wingcal(cat.copy(), 'f410m', basepath=str(base),
                                  phase='m4')
    assert out['wingcal_ratio'][0] == pytest.approx(1.10)
    out = MC.apply_pooled_wingcal(cat.copy(), 'f410m', basepath=str(base),
                                  phase='m7')
    assert out['wingcal_ratio'][0] == pytest.approx(1.30)


def test_pooled_wingcal_without_calibrator_files_keeps_an_existing_table(tree):
    # A tree whose per-frame products were cleaned up: the table on disk is
    # the only calibration left, so it is used as written.
    base, _ = tree
    _legacy_table(base, 1.25, mtime=2_000)
    pooled = MC.load_pooled_wingcal('f410m', str(base), phase='m7')
    assert pooled['ratio'].tolist() == [pytest.approx(1.25)]
    assert MC.load_pooled_wingcal('f410m', str(base), phase='m6') is None


# --- partner-band seed fallback --------------------------------------------

def _satcat(pdir, exposure, phase, ra):
    fn = pdir / f'{exposure}_destreak_o040_crf_{phase}_satstar_catalog.fits'
    t = Table({'flux_fit': [1e4]})
    t['skycoord_fit'] = SkyCoord([ra], [-29.0], unit='deg')
    t.write(fn, overwrite=True)
    return fn


@pytest.mark.parametrize('phase, want', [
    ('m4', ['m12']), ('m7', ['m7']), ('m12', ['m12']), (None, ['m7'])])
def test_partner_seed_fallback_reads_one_phase_per_exposure(tmp_path, phase, want):
    pdir = tmp_path / 'F480M' / 'pipeline'
    pdir.mkdir(parents=True)
    _satcat(pdir, EXP, 'm12', 266.40)
    _satcat(pdir, EXP, 'm7', 266.41)
    files = C._partner_satstar_seed_files(str(tmp_path), 'f480m',
                                          phase=phase)
    assert [satstar_file_phase_token(f) for f in files] == want


def test_partner_seed_prefers_the_consolidated_catalog(tmp_path):
    pdir = tmp_path / 'F480M' / 'pipeline'
    pdir.mkdir(parents=True)
    _satcat(pdir, EXP, 'm7', 266.41)
    cons = MC.consolidated_satstar_cache_path(str(tmp_path), 'f480m', '')
    os.makedirs(os.path.dirname(cons), exist_ok=True)
    Table({'flux_fit': [1.0]}).write(cons)
    assert C._partner_satstar_seed_files(str(tmp_path), 'f480m',
                                         phase='m7') == [cons]


# --- rejected channel follows the merged_i2d primary -------------------------

def _rejected_tree(tmp_path, monkeypatch, with_sibling):
    pdir = tmp_path / 'F480M' / 'pipeline'
    pdir.mkdir(parents=True)
    primary = pdir / ('jw10678-o040_t001_nircam_clear-f480m-merged_i2d'
                      '_satstar_catalog.fits')
    Table({'flux_fit': [1.0]}).write(primary)
    if with_sibling:
        t = Table({'flux_fit': [2.0, 3.0]})
        t['skycoord_fit'] = SkyCoord([266.4, 266.5], [-29.0, -29.0], unit='deg')
        t.write(str(primary).replace('_catalog.fits', '_rejected.fits'))
    # a per-exposure run the accepted channel does not read
    t = Table({'flux_fit': [9.0]})
    t['skycoord_fit'] = SkyCoord([266.6], [-29.0], unit='deg')
    t.write(pdir / f'{EXP}_destreak_o040_crf_m7_satstar_rejected.fits')
    monkeypatch.setattr(MC, '_primary_satstar_catalog_path',
                        lambda *a, **k: str(primary))
    return tmp_path


def test_rejected_channel_reads_the_primary_products_rejected_file(tmp_path,
                                                                   monkeypatch):
    base = _rejected_tree(tmp_path, monkeypatch, with_sibling=True)
    rej = MC.load_rejected_satstar_catalog('f480m', target='gc-treasury',
                                           basepath=str(base), phase='m7')
    assert sorted(np.asarray(rej['flux_fit']).tolist()) == [2.0, 3.0]


def test_rejected_channel_is_empty_when_the_primary_has_no_rejected_file(
        tmp_path, monkeypatch):
    base = _rejected_tree(tmp_path, monkeypatch, with_sibling=False)
    assert MC.load_rejected_satstar_catalog(
        'f480m', target='gc-treasury', basepath=str(base), phase='m7') is None


# --- call sites no unit test drives -----------------------------------------

def _keyword_calls(module, func):
    """``{keyword: source}`` of every call to ``func`` in ``module``'s source."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(module))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, 'id', getattr(n.func, 'attr', None)) == func]
    return [{k.arg: ast.unparse(k.value) for k in c.keywords} for c in calls]


@pytest.mark.parametrize('module, func, keyword, value', [
    # run_manual_pipeline: the in-field co-fit dedup reads the merge's phase
    (C, '_satstar_cofit_positions', 'phase', 'merge_label'),
    # _prepare_frame_for_photometry: the partner seeds read the phase being fit
    (C, '_partner_satstar_seed_files', 'phase', 'satstar_label'),
    # load_satstar_catalog: primary and consolidated branches
    (MC, 'apply_pooled_wingcal', 'phase', 'phase'),
])
def test_production_call_sites_pass_the_phase(module, func, keyword, value):
    calls = _keyword_calls(module, func)
    assert calls, f'no call to {func}'
    assert [c.get(keyword) for c in calls] == [value] * len(calls)
