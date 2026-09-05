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
    frame_detector_tokens, module_token_covers, perframe_families_on_disk,
    perframe_module_family, requested_nircam_families, uncovered_module_tokens)


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
            cut_bp, ['nrca'], ['F115W', 'F150W', 'F200W'], 'm2', _options(), frames_by_filter={})
    msg = str(ei.value)
    assert 'nrcb' in msg
    assert 'F115W' in msg and 'F200W' in msg


def test_truncated_modules_refuse_even_for_one_filter(tmp_path):
    """One uncovered filter is enough -- a half-field catalog is half a field."""
    cut_bp = _tree(tmp_path, {'F200W': ['nrca1', 'nrcb1'],
                              'F212N': ['nrca1']})
    with pytest.raises(FinalizeModuleCoverageError):
        assert_finalize_covers_disk_modules(
            cut_bp, ['nrca'], ['F200W', 'F212N'], 'm2', _options(), frames_by_filter={})


def test_full_module_list_passes(tmp_path):
    cut_bp = _tree(tmp_path, {'F200W': ['nrca1', 'nrcb1', 'nrcb2']})
    assert_finalize_covers_disk_modules(
        cut_bp, ['nrca', 'nrcb', 'merged'], ['F200W'], 'm2', _options(), frames_by_filter={})
    assert_finalize_covers_disk_modules(
        cut_bp, ['nrca', 'nrcb'], ['F200W'], 'm2', _options(), frames_by_filter={})


def test_single_module_field_stays_quiet(tmp_path):
    """sickle's NIRCam is nrcb alone: nothing was fitted that is being dropped."""
    cut_bp = _tree(tmp_path, {'F187N': ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']})
    assert_finalize_covers_disk_modules(
        cut_bp, ['nrcb'], ['F187N'], 'm2', _options(), frames_by_filter={})


def test_miri_run_carrying_the_nircam_sbatch_default_stays_quiet(tmp_path):
    """submit_cataloging.sbatch defaults to MODULES=nrcb, MIRI runs included."""
    cut_bp = _tree(tmp_path, {'F770W': ['mirimage']})
    assert_finalize_covers_disk_modules(
        cut_bp, ['nrcb'], ['F770W'], 'm2', _options(), frames_by_filter={})


def test_other_phases_are_not_read_as_modules(tmp_path):
    """Only the requested merge label counts; m3's catalogs are not m2's."""
    cut_bp = str(tmp_path)
    _write_perframe(cut_bp, 'F200W', 'nrca1', label='m2')
    _write_perframe(cut_bp, 'F200W', 'nrcb1', label='m3')
    assert_finalize_covers_disk_modules(
        cut_bp, ['nrca'], ['F200W'], 'm2', _options(), frames_by_filter={})
    with pytest.raises(FinalizeModuleCoverageError):
        assert_finalize_covers_disk_modules(
            cut_bp, ['nrca'], ['F200W'], 'm3', _options(), frames_by_filter={})


# --- 10678: 139 tiles in one tree ------------------------------------------

def test_another_tiles_catalogs_do_not_refuse_this_tile(tmp_path):
    """A treasury tile is finalized against ITS OWN observation's catalogs."""
    cut_bp = str(tmp_path)
    _write_perframe(cut_bp, 'F212N', 'nrca1', obs='o037', exp='00001')
    _write_perframe(cut_bp, 'F212N', 'nrcb1', obs='o037', exp='00002')
    _write_perframe(cut_bp, 'F212N', 'nrcb1', obs='o042', exp='00003')
    # tile 042 fitted nrcb only; tile 037's nrca must not refuse it
    assert_finalize_covers_disk_modules(
        cut_bp, ['nrcb'], ['F212N'], 'm2', _options(), obs_token='_o042', frames_by_filter={})
    # ...and tile 037 truly did fit both, so a truncated 037 finalize refuses
    with pytest.raises(FinalizeModuleCoverageError):
        assert_finalize_covers_disk_modules(
            cut_bp, ['nrcb'], ['F212N'], 'm2', _options(), obs_token='_o037', frames_by_filter={})


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
            _options(allow_partial_modules=True), frames_by_filter={})
    assert 'ALLOW_PARTIAL_MODULES=1' in str(ei.value)


def test_flag_plus_environment_allows(tmp_path, monkeypatch):
    cut_bp = _tree(tmp_path, {'F200W': ['nrca1', 'nrcb1']})
    monkeypatch.setenv('ALLOW_PARTIAL_MODULES', '1')
    assert_finalize_covers_disk_modules(
        cut_bp, ['nrca'], ['F200W'], 'm2',
        _options(allow_partial_modules=True), frames_by_filter={})


def test_environment_alone_still_refuses(tmp_path, monkeypatch):
    cut_bp = _tree(tmp_path, {'F200W': ['nrca1', 'nrcb1']})
    monkeypatch.setenv('ALLOW_PARTIAL_MODULES', '1')
    with pytest.raises(FinalizeModuleCoverageError):
        assert_finalize_covers_disk_modules(
            cut_bp, ['nrca'], ['F200W'], 'm2', _options(), frames_by_filter={})


# --- wiring: the finalize itself refuses, before it merges anything ---------

def _run_manual_options(**kw):
    base = dict(cutout_region='', each_suffix='destreak_o001_crf',
                extended_emission=False, no_miri_tuning=True,
                manual_stop_after_phase='', manual_skip_finalize=False,
                manual_finalize_only=True, manual_frame_shard='',
                allow_partial_modules=False, parallel_workers=1)
    base.update(kw)
    return types.SimpleNamespace(**base)


def _stub_frames(monkeypatch, tmp_path, detectors=('nrca1',)):
    """Candidate frames per (filter, module, visit), named like the real ones.

    ``module='merged'`` expands to every detector, exactly as ``get_filenames``
    does; any other token selects by the same substring rule its glob uses.
    """
    from jwst_gc_pipeline.photometry import cataloging as _cat

    def _fake(basepath, filtername, proposal_id, field, each_suffix, module,
              pupil='clear', visitid='001', allow_empty=False):
        dets = (detectors if module == 'merged'
                else [d for d in detectors if module in d])
        return [os.path.join(str(tmp_path), filtername, 'pipeline',
                             f'jw02221{field}{visitid}_04101_{i:05d}_'
                             f'{d}_{each_suffix}.fits')
                for i, d in enumerate(dets, start=1)]
    monkeypatch.setattr(_cat._L, 'get_filenames', _fake)


def test_finalize_only_refuses_a_truncated_module_list(tmp_path, monkeypatch):
    from jwst_gc_pipeline.photometry.cataloging import run_manual_pipeline
    _stub_frames(monkeypatch, tmp_path, detectors=('nrca1', 'nrcb1'))
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
    _stub_frames(monkeypatch, tmp_path, detectors=('nrca1', 'nrcb1'))
    _tree(tmp_path, {'F200W': ['nrca1', 'nrcb1']})
    opts = _run_manual_options(manual_finalize_only=False,
                               manual_skip_finalize=True)
    # It gets PAST the coverage check and dies later, on its own business.
    with pytest.raises(Exception) as ei:
        run_manual_pipeline(opts, ['nrca'], ['F200W'], {'2221': {'brick': 1}},
                            '2221', 'brick', '001', str(tmp_path), {}, {})
    assert not isinstance(ei.value, FinalizeModuleCoverageError)


# --- the FRAMES are the authority, not the catalogs (review B1) -------------
#
# Fan-out and finalize take MODULES from ONE exported variable in ONE wrapper
# (submit_cataloging_perframe.sh: both sbatch calls inherit through
# COMMON="ALL,..."), so a truncation truncates both.  A catalogs-only check then
# sees only what the truncated fan-out wrote and passes.  The historical w51 job
# was caught only because w51 already carried nrcb catalogs from an earlier full
# run; every fresh tile's FIRST m12 is in the blind case instead.

def _frames(tmp_path, filt, detectors, *, suffix='destreak_o001_crf',
            visit='001'):
    """Per-exposure frame paths named exactly as the reduction writes them."""
    return [os.path.join(str(tmp_path), filt, 'pipeline',
                         f'jw02221001{visit}_04101_{i:05d}_{d}_{suffix}.fits')
            for i, d in enumerate(detectors, start=1)]


def test_frame_detector_tokens_read_real_detectors(tmp_path):
    fns = _frames(tmp_path, 'F200W', ['nrca1', 'nrcb3'])
    fns += _frames(tmp_path, 'F405N', ['nrcalong', 'nrcblong'])
    assert frame_detector_tokens(fns) == {'nrca1', 'nrcb3', 'nrcalong',
                                          'nrcblong'}
    # single-detector instruments carry no module family and none is invented
    assert frame_detector_tokens(_frames(tmp_path, 'F770W', ['mirimage'])) == set()


def test_fresh_tree_with_no_catalogs_still_refuses(tmp_path):
    """THE regression: nothing has been fitted yet, so only the frames answer.

    With a catalogs-only authority this tree is empty and the truncated run
    passes -- the state of all 139 treasury tiles on their first m12.
    """
    cut_bp = str(tmp_path)          # deliberately NO per-frame catalogs
    frames = {'F200W': _frames(tmp_path, 'F200W',
                               ['nrca1', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'])}
    assert perframe_families_on_disk(cut_bp, 'F200W', 'm2') == {}
    with pytest.raises(FinalizeModuleCoverageError) as ei:
        assert_finalize_covers_disk_modules(
            cut_bp, ['nrca'], ['F200W'], 'm2', _options(),
            frames_by_filter=frames)
    assert 'nrcb1' in str(ei.value)


def test_fresh_tree_full_module_list_passes(tmp_path):
    frames = {'F200W': _frames(tmp_path, 'F200W', ['nrca1', 'nrcb1'])}
    assert_finalize_covers_disk_modules(
        str(tmp_path), ['nrca', 'nrcb', 'merged'], ['F200W'], 'm2', _options(),
        frames_by_filter=frames)
    assert_finalize_covers_disk_modules(
        str(tmp_path), ['merged'], ['F200W'], 'm2', _options(),
        frames_by_filter=frames)


def test_fresh_sickle_tree_stays_quiet(tmp_path):
    """sickle's NIRCam frames are nrcb alone; nothing is being dropped."""
    frames = {'F187N': _frames(tmp_path, 'F187N',
                               ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'])}
    assert_finalize_covers_disk_modules(
        str(tmp_path), ['nrcb'], ['F187N'], 'm2', _options(),
        frames_by_filter=frames)


def test_fresh_miri_tree_under_the_nircam_sbatch_default_stays_quiet(tmp_path):
    frames = {'F770W': _frames(tmp_path, 'F770W', ['mirimage'] * 4)}
    assert_finalize_covers_disk_modules(
        str(tmp_path), ['nrcb'], ['F770W'], 'm2', _options(),
        frames_by_filter=frames)


def test_frames_by_filter_is_required(tmp_path):
    """Fail CLOSED: a caller that forgets the frames gets a TypeError.

    Keyword-only with no default, so the frame layer cannot be dropped by
    omission the way a defaulted argument would let it be.
    """
    with pytest.raises(TypeError):
        assert_finalize_covers_disk_modules(
            str(tmp_path), ['nrca'], ['F200W'], 'm2', _options())


def test_finalize_only_refuses_on_a_tree_that_has_never_been_cataloged(
        tmp_path, monkeypatch):
    """End to end through `run_manual_pipeline`, with zero catalogs on disk."""
    from jwst_gc_pipeline.photometry.cataloging import run_manual_pipeline
    _stub_frames(monkeypatch, tmp_path,
                 detectors=('nrca1', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'))
    with pytest.raises(FinalizeModuleCoverageError) as ei:
        run_manual_pipeline(
            _run_manual_options(), ['nrca'], ['F200W'],
            {'2221': {'brick': 1}}, '2221', 'brick', '001', str(tmp_path),
            {}, {})
    assert 'nrcb' in str(ei.value)


# --- intra-family truncation (review B2) ------------------------------------
#
# `requested_nircam_families` collapses nrcXN to its family, so a family-only
# comparison reads `--modules nrcb1` against nrcb1-4 on disk as covered and
# merges a quarter of the field.  Same loss class as #532, one module deep:
# the pipeline itself produces nrcb1..nrcb4 lists (sickle 3958/007), and
# `sbatch --export` truncates a comma-valued MODULES to its first entry.

def test_one_detector_of_four_refuses(tmp_path):
    frames = {'F187N': _frames(tmp_path, 'F187N',
                               ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'])}
    with pytest.raises(FinalizeModuleCoverageError) as ei:
        assert_finalize_covers_disk_modules(
            str(tmp_path), ['nrcb1'], ['F187N'], 'm2', _options(),
            frames_by_filter=frames)
    msg = str(ei.value)
    assert 'nrcb2' in msg and 'nrcb3' in msg and 'nrcb4' in msg
    assert 'nrcb1' not in msg.split('not covered:')[-1]


def test_the_whole_detector_list_passes(tmp_path):
    frames = {'F187N': _frames(tmp_path, 'F187N',
                               ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'])}
    assert_finalize_covers_disk_modules(
        str(tmp_path), ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'], ['F187N'], 'm2',
        _options(), frames_by_filter=frames)
    # and the family token covers all four, as get_filenames' glob does
    assert_finalize_covers_disk_modules(
        str(tmp_path), ['nrcb'], ['F187N'], 'm2', _options(),
        frames_by_filter=frames)


def test_sw_detector_token_does_not_cover_the_lw_detector(tmp_path):
    """`nrcb1` must not be read as covering `nrcblong` in a mixed run."""
    frames = {'F187N': _frames(tmp_path, 'F187N', ['nrcb1']),
              'F480M': _frames(tmp_path, 'F480M', ['nrcblong'])}
    with pytest.raises(FinalizeModuleCoverageError) as ei:
        assert_finalize_covers_disk_modules(
            str(tmp_path), ['nrcb1'], ['F187N', 'F480M'], 'm2', _options(),
            frames_by_filter=frames)
    assert 'nrcblong' in str(ei.value)


@pytest.mark.parametrize('requested,token,expect', [
    ('nrcb', 'nrcb1', True),
    ('nrcb', 'nrcblong', True),
    ('nrcb1', 'nrcb1', True),
    ('nrcb1', 'nrcb2', False),
    ('nrcb1', 'nrcblong', False),
    ('nrca', 'nrcb1', False),
    ('merged', 'nrcblong', True),
])
def test_module_token_covers(requested, token, expect):
    assert module_token_covers(requested, token) is expect


def test_uncovered_module_tokens():
    assert uncovered_module_tokens(
        ['nrcb1'], ['nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']) == ['nrcb2', 'nrcb3',
                                                             'nrcb4']
    assert uncovered_module_tokens(['nrca', 'nrcb'],
                                   ['nrca1', 'nrcblong']) == []
    assert uncovered_module_tokens([], ['nrca1']) == ['nrca1']


def test_intra_family_truncation_survives_the_override(tmp_path, monkeypatch):
    frames = {'F187N': _frames(tmp_path, 'F187N', ['nrcb1', 'nrcb2'])}
    with pytest.raises(FinalizeModuleCoverageError):
        assert_finalize_covers_disk_modules(
            str(tmp_path), ['nrcb1'], ['F187N'], 'm2',
            _options(allow_partial_modules=True), frames_by_filter=frames)
    monkeypatch.setenv('ALLOW_PARTIAL_MODULES', '1')
    assert_finalize_covers_disk_modules(
        str(tmp_path), ['nrcb1'], ['F187N'], 'm2',
        _options(allow_partial_modules=True), frames_by_filter=frames)
