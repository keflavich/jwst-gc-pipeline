"""A finalize must not merge fewer modules than the field has fitted (#734).

``--manual-finalize-only`` reported success for whatever ``--modules`` list it
was handed.  ``w516151-o001-m12-finalize`` ran ``modules=nrca`` against a field
carrying nrcb1-4 per-frame catalogs for twelve filters, exited 0, and the loss
surfaced only when m3 crashed on the missing input 2 h 24 m later.
"""
import os
import types

import pytest

from jwst_gc_pipeline.photometry.cataloging import (
    FinalizeModuleCoverageError, assert_finalize_covers_disk_modules,
    perframe_families_on_disk, perframe_module_family,
    requested_nircam_families)


def _write_perframe(cut_bp, filt, detector, *, obs='', label='m2',
                    visit='001', vgroup='04101', exp='00001'):
    """One per-frame catalog, named exactly as the phase loop writes it."""
    d = os.path.join(cut_bp, filt.upper())
    os.makedirs(d, exist_ok=True)
    obs_ = f'_{obs}' if obs else ''
    path = os.path.join(
        d, f'{filt.lower()}_{detector}{obs_}_visit{visit}_vgroup{vgroup}'
           f'_exp{exp}_{label}_daophot_basic.fits')
    with open(path, 'wb') as fh:
        fh.write(b'')
    return path


def _tree(tmp_path, spec, obs=''):
    """spec: {FILTER: [detector, ...]} -> a cut_bp with those per-frame m2 catalogs."""
    cut_bp = str(tmp_path)
    for filt, dets in spec.items():
        for i, det in enumerate(dets, start=1):
            _write_perframe(cut_bp, filt, det, obs=obs, exp=f'{i:05d}')
    return cut_bp


def _options(**kw):
    kw.setdefault('allow_partial_modules', False)
    return types.SimpleNamespace(**kw)


# --- the family/coverage arithmetic ----------------------------------------

@pytest.mark.parametrize('basename,filt,expect', [
    ('f200w_nrca1_visit001_vgroup04101_exp00001_m2_daophot_basic.fits',
     'F200W', ('nrca', '')),
    ('f200w_nrcb3_visit001_vgroup04101_exp00001_m2_daophot_basic.fits',
     'F200W', ('nrcb', '')),
    ('f405n_nrcalong_visit001_vgroup04101_exp00001_m2_daophot_basic.fits',
     'F405N', ('nrca', '')),
    # 10678's 139 tiles share one tree, so the name carries the observation.
    ('f212n_nrcb1_o037_visit001_vgroup04101_exp00001_m2_daophot_basic.fits',
     'F212N', ('nrcb', 'o037')),
    # single-detector instruments have no module family
    ('f770w_mirimage_visit001_vgroup04101_exp00001_m2_daophot_basic.fits',
     'F770W', ('mirimage', '')),
])
def test_perframe_module_family(basename, filt, expect):
    assert perframe_module_family(basename, filt) == expect


def test_merged_covers_everything():
    """`merged` expands to every detector token, so it can never be truncated."""
    assert requested_nircam_families(['merged']) is None
    assert requested_nircam_families(['nrca', 'nrcb', 'merged']) is None


def test_non_nircam_modules_ask_for_no_family():
    assert requested_nircam_families(['mirimage']) == set()
    assert requested_nircam_families(['nis']) == set()
    assert requested_nircam_families(['nrca']) == {'nrca'}
    assert requested_nircam_families(['nrcb1', 'nrcb4']) == {'nrcb'}


# --- what it catches --------------------------------------------------------

def test_truncated_modules_refuse(tmp_path):
    """The w51 incident: modules=nrca against a field that fitted nrcb too."""
    cut_bp = _tree(tmp_path, {f'F{n}W': ['nrca1', 'nrcb1', 'nrcb2', 'nrcb3',
                                         'nrcb4']
                              for n in (115, 150, 200)})
    with pytest.raises(FinalizeModuleCoverageError) as ei:
        assert_finalize_covers_disk_modules(
            cut_bp, ['nrca'], ['F115W', 'F150W', 'F200W'], 'm2', _options())
    msg = str(ei.value)
    assert 'nrcb' in msg
    assert 'F115W' in msg and 'F200W' in msg


def test_truncated_modules_refuse_even_for_one_filter(tmp_path):
    """One uncovered filter is enough -- a half-field catalog is half a field."""
    cut_bp = _tree(tmp_path, {'F200W': ['nrca1', 'nrcb1'],
                              'F212N': ['nrca1']})
    with pytest.raises(FinalizeModuleCoverageError):
        assert_finalize_covers_disk_modules(
            cut_bp, ['nrca'], ['F200W', 'F212N'], 'm2', _options())


def test_full_module_list_passes(tmp_path):
    cut_bp = _tree(tmp_path, {'F200W': ['nrca1', 'nrcb1', 'nrcb2']})
    assert_finalize_covers_disk_modules(
        cut_bp, ['nrca', 'nrcb', 'merged'], ['F200W'], 'm2', _options())
    assert_finalize_covers_disk_modules(
        cut_bp, ['nrca', 'nrcb'], ['F200W'], 'm2', _options())


def test_single_module_field_stays_quiet(tmp_path):
    """sickle's NIRCam is nrcb alone: nothing was fitted that is being dropped."""
    cut_bp = _tree(tmp_path, {'F187N': ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']})
    assert_finalize_covers_disk_modules(
        cut_bp, ['nrcb'], ['F187N'], 'm2', _options())


def test_miri_run_carrying_the_nircam_sbatch_default_stays_quiet(tmp_path):
    """submit_cataloging.sbatch defaults to MODULES=nrcb, MIRI runs included."""
    cut_bp = _tree(tmp_path, {'F770W': ['mirimage']})
    assert_finalize_covers_disk_modules(
        cut_bp, ['nrcb'], ['F770W'], 'm2', _options())


def test_other_phases_are_not_read_as_modules(tmp_path):
    """Only the requested merge label counts; m3's catalogs are not m2's."""
    cut_bp = str(tmp_path)
    _write_perframe(cut_bp, 'F200W', 'nrca1', label='m2')
    _write_perframe(cut_bp, 'F200W', 'nrcb1', label='m3')
    assert_finalize_covers_disk_modules(
        cut_bp, ['nrca'], ['F200W'], 'm2', _options())
    with pytest.raises(FinalizeModuleCoverageError):
        assert_finalize_covers_disk_modules(
            cut_bp, ['nrca'], ['F200W'], 'm3', _options())


# --- 10678: 139 tiles in one tree ------------------------------------------

def test_another_tiles_catalogs_do_not_refuse_this_tile(tmp_path):
    """A treasury tile is finalized against ITS OWN observation's catalogs."""
    cut_bp = str(tmp_path)
    _write_perframe(cut_bp, 'F212N', 'nrca1', obs='o037', exp='00001')
    _write_perframe(cut_bp, 'F212N', 'nrcb1', obs='o037', exp='00002')
    _write_perframe(cut_bp, 'F212N', 'nrcb1', obs='o042', exp='00003')
    # tile 042 fitted nrcb only; tile 037's nrca must not refuse it
    assert_finalize_covers_disk_modules(
        cut_bp, ['nrcb'], ['F212N'], 'm2', _options(), obs_token='_o042')
    # ...and tile 037 truly did fit both, so a truncated 037 finalize refuses
    with pytest.raises(FinalizeModuleCoverageError):
        assert_finalize_covers_disk_modules(
            cut_bp, ['nrcb'], ['F212N'], 'm2', _options(), obs_token='_o037')


def test_families_on_disk_are_obs_scoped(tmp_path):
    cut_bp = str(tmp_path)
    _write_perframe(cut_bp, 'F212N', 'nrca1', obs='o037')
    _write_perframe(cut_bp, 'F212N', 'nrcb1', obs='o042', exp='00002')
    assert set(perframe_families_on_disk(cut_bp, 'F212N', 'm2',
                                         obs_token='_o037')) == {'nrca'}
    assert set(perframe_families_on_disk(cut_bp, 'F212N', 'm2',
                                         obs_token='_o042')) == {'nrcb'}


# --- the override is two-factor --------------------------------------------

def test_flag_alone_still_refuses(tmp_path):
    cut_bp = _tree(tmp_path, {'F200W': ['nrca1', 'nrcb1']})
    with pytest.raises(FinalizeModuleCoverageError) as ei:
        assert_finalize_covers_disk_modules(
            cut_bp, ['nrca'], ['F200W'], 'm2',
            _options(allow_partial_modules=True))
    assert 'ALLOW_PARTIAL_MODULES=1' in str(ei.value)


def test_flag_plus_environment_allows(tmp_path, monkeypatch):
    cut_bp = _tree(tmp_path, {'F200W': ['nrca1', 'nrcb1']})
    monkeypatch.setenv('ALLOW_PARTIAL_MODULES', '1')
    assert_finalize_covers_disk_modules(
        cut_bp, ['nrca'], ['F200W'], 'm2',
        _options(allow_partial_modules=True))


def test_environment_alone_still_refuses(tmp_path, monkeypatch):
    cut_bp = _tree(tmp_path, {'F200W': ['nrca1', 'nrcb1']})
    monkeypatch.setenv('ALLOW_PARTIAL_MODULES', '1')
    with pytest.raises(FinalizeModuleCoverageError):
        assert_finalize_covers_disk_modules(
            cut_bp, ['nrca'], ['F200W'], 'm2', _options())


# --- wiring: the finalize itself refuses, before it merges anything ---------

def _run_manual_options(**kw):
    base = dict(cutout_region='', each_suffix='destreak_o001_crf',
                extended_emission=False, no_miri_tuning=True,
                manual_stop_after_phase='', manual_skip_finalize=False,
                manual_finalize_only=True, manual_frame_shard='',
                allow_partial_modules=False, parallel_workers=1)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _stub_frames(monkeypatch, tmp_path):
    """One candidate frame per (filter, module, visit) -- enough for preflight."""
    from jwst_gc_pipeline.photometry import cataloging as _cat

    def _fake(basepath, filtername, proposal_id, field, each_suffix, module,
              pupil='clear', visitid='001', allow_empty=False):
        return [os.path.join(str(tmp_path), filtername, 'pipeline',
                             f'jw02221{field}{visitid}_04101_00001_'
                             f'nrca1_{each_suffix}.fits')]
    monkeypatch.setattr(_cat._L, 'get_filenames', _fake)


def test_finalize_only_refuses_a_truncated_module_list(tmp_path, monkeypatch):
    from jwst_gc_pipeline.photometry.cataloging import run_manual_pipeline
    _stub_frames(monkeypatch, tmp_path)
    _tree(tmp_path, {'F200W': ['nrca1', 'nrcb1']})
    with pytest.raises(FinalizeModuleCoverageError) as ei:
        run_manual_pipeline(
            _run_manual_options(), ['nrca'], ['F200W'],
            {'2221': {'brick': 1}}, '2221', 'brick', '001', str(tmp_path),
            {}, {})
    assert 'nrcb' in str(ei.value)


def test_a_fitting_run_is_not_gated(tmp_path, monkeypatch):
    """The refusal belongs to the barrier: a fan-out worker fits its own slice.

    Without this the ordinary per-module fan-out (`--modules nrca` writing
    nrca's per-frame catalogs) could not run at all.
    """
    from jwst_gc_pipeline.photometry.cataloging import run_manual_pipeline
    _stub_frames(monkeypatch, tmp_path)
    _tree(tmp_path, {'F200W': ['nrca1', 'nrcb1']})
    opts = _run_manual_options(manual_finalize_only=False,
                               manual_skip_finalize=True)
    # It gets PAST the coverage check and dies later, on its own business.
    with pytest.raises(Exception) as ei:
        run_manual_pipeline(opts, ['nrca'], ['F200W'], {'2221': {'brick': 1}},
                            '2221', 'brick', '001', str(tmp_path), {}, {})
    assert not isinstance(ei.value, FinalizeModuleCoverageError)
