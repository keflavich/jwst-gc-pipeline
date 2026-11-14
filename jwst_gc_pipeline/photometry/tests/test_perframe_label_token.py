"""The m3..m7 per-frame products carry the merge label of the pass that fit them.

``run_manual_pipeline`` fits a NIRCam frame once per merge label that has it in
scope -- its module pass (``nrca``/``nrcb``) and the ``merged`` pass.  From m3
on those fits take label-scoped inputs (seed catalog, smoothed residual
background, satstar overrides/drops), so they are two measurements.  The
per-frame names were keyed on the DETECTOR only, so both passes wrote one path
and each label's finalize read whichever pass finished last: on the
gc-treasury m7 fan-out the ``nrca`` m7 catalogs held the merged pass's fit for
2019 of 2020 frames.

The fix puts ``_lbl{label}`` after ``_daophot_{kind}`` at m3..m7
(``naming.perframe_label_token``).  These tests pin

* the token's grammar, and that no reader glob written before it can match it;
* the writer/predictor/sentinel agreement;
* the readers' variant choice -- a per-label reader reads its own label, else
  the LEGACY name, never another label's; a module-blind reader picks one per
  frame in a fixed order;
* the cross-label path guard and the resume refit of legacy-only markers.

The end-to-end check (two passes through the real writer, then the merge) is
``test_perframe_label_fanout.py``.
"""
import fnmatch
import os
import re
import types

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
import astropy.units as u

from jwst_gc_pipeline.photometry import cataloging as C
from jwst_gc_pipeline.photometry import crowdsource_catalogs_long as L
from jwst_gc_pipeline.photometry import merge_catalogs as MC
from jwst_gc_pipeline.photometry.naming import (
    LABEL_TOKEN_PHASES, MODULE_BLIND_LABEL_ORDER, choose_perframe_variants,
    glob_perframe_catalogs, label_token_phase, perframe_label_of,
    perframe_label_token, perframe_provenance_meta, perframe_variant_summary,
    strip_perframe_label)

ENV = 'PSF_EPSF_CORE_DIR'

LEGACY = 'f480m_nrcblong_o105_visit001_vgroup02101_exp00004_resbgsub_m7_daophot_basic.fits'
TOK = {lab: LEGACY.replace('_daophot_basic.fits', f'_daophot_basic_lbl{lab}.fits')
       for lab in ('nrcb', 'merged', 'nrca')}


@pytest.fixture(autouse=True)
def _stpsf(monkeypatch):
    """Every name here is the STPSF spelling (no ``_hybpsf``)."""
    monkeypatch.delenv(ENV, raising=False)


# ---------------------------------------------------------------------------
# grammar
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('lbl', ['m3', 'm4', 'm5', 'm6', 'm7', '_m7',
                                 'm7_chunk03of08', '_m5_chunk00of02'])
def test_token_phases(lbl):
    assert label_token_phase(lbl)
    assert perframe_label_token(lbl, 'nrcb') == '_lblnrcb'


@pytest.mark.parametrize('lbl', [None, '', 'm1', 'm2', '_m2', 'm12', 'iter2',
                                 'iter3', 'm8', 'm2_chunk00of04'])
def test_no_token_outside_m3_m7(lbl):
    """m12 keeps the legacy names: one fit serves every label there (#841)."""
    assert not label_token_phase(lbl)
    assert perframe_label_token(lbl, 'nrcb') == ''


def test_phase_list_is_m3_to_m7():
    assert LABEL_TOKEN_PHASES == ('m3', 'm4', 'm5', 'm6', 'm7')


@pytest.mark.parametrize('label', [None, ''])
def test_no_merge_label_writes_the_legacy_name(label):
    assert perframe_label_token('m7', label) == ''


@pytest.mark.parametrize('label', ['NRCB', 'nrc_b', 'nrc-b', 'nrcb.1', ' nrcb'])
def test_a_label_the_readers_cannot_split_off_is_refused(label):
    with pytest.raises(ValueError, match='lower-case alphanumeric'):
        perframe_label_token('m7', label)


@pytest.mark.parametrize('name, label, legacy', [
    (TOK['nrcb'], 'nrcb', LEGACY),
    ('x_m7_chunk03of08_daophot_basic_lblmerged.fits', 'merged',
     'x_m7_chunk03of08_daophot_basic.fits'),
    ('jw10678-o105_t001_nircam_clear-f480m-nrcblong_visit001_vgroup02101_'
     'exp00004_resbgsub_m7_daophot_basic_lblnrcb_residual.fits', 'nrcb',
     'jw10678-o105_t001_nircam_clear-f480m-nrcblong_visit001_vgroup02101_'
     'exp00004_resbgsub_m7_daophot_basic_residual.fits'),
    ('a_m6_daophot_basic_lblnrca_mergedcat_model.fits', 'nrca',
     'a_m6_daophot_basic_mergedcat_model.fits'),
    (LEGACY, None, LEGACY),
])
def test_label_of_and_strip(name, label, legacy):
    assert perframe_label_of('/d/' + name) == label
    assert strip_perframe_label('/d/' + name) == '/d/' + legacy


# Every reader shape in the tree before the token (globs and regexes), as the
# scripts spell them.  None may match a tokened name -- so a reader that was
# not updated sees only the legacy names it saw before, never both spellings
# of one frame.  This is why the token sits AFTER ``_daophot_<kind>``.
LEGACY_READER_GLOBS = [
    '*_daophot_basic.fits',
    '*_m[0-9]*_daophot_basic.fits',
    '*_exp?????_m7_daophot_basic.fits',
    '*_visit*_vgroup*_exp*_m7_daophot_basic.fits',
    '*_m7_chunk*of*_daophot_basic.fits',
    '*_daophot_basic_residual.fits',
    '*_m7_daophot_basic_residual.fits',
    '*_daophot_basic_model.fits',
    '*_m7_daophot_basic_mergedcat_residual.fits',
]
LEGACY_READER_RES = [
    r'_daophot_basic\.fits$',
    r'_m\d+_daophot_[a-z]+_(residual|model)\.fits$',
    r'_m\d+_daophot_[a-z]+_mergedcat_(residual|model)\.fits$',
    r'_exp\d+_m7_daophot_basic\.fits$',
]
TOKENED_NAMES = [
    TOK['nrcb'],
    'x_m7_chunk03of08_daophot_basic_lblnrcb.fits',
    'jw-x_m7_daophot_basic_lblnrcb_residual.fits',
    'jw-x_m7_daophot_basic_lblnrcb_model.fits',
    'jw-x_m7_daophot_basic_lblnrcb_mergedcat_residual.fits',
]


@pytest.mark.parametrize('name', TOKENED_NAMES)
def test_no_pre_token_reader_shape_matches_a_tokened_name(name):
    for g in LEGACY_READER_GLOBS:
        assert not fnmatch.fnmatch(name, g), (g, name)
    for r in LEGACY_READER_RES:
        assert not re.search(r, name), (r, name)


# ---------------------------------------------------------------------------
# variant choice
# ---------------------------------------------------------------------------

def _paths(*names):
    return ['/d/' + n for n in names]


def test_per_label_reader_prefers_its_own_then_legacy_never_another():
    a = 'f_nrcb1_visit001_vgroup1_exp00001_m7_daophot_basic'
    b = 'f_nrcb1_visit001_vgroup1_exp00002_m7_daophot_basic'
    c = 'f_nrcb1_visit001_vgroup1_exp00003_m7_daophot_basic'
    paths = _paths(f'{a}.fits', f'{a}_lblnrcb.fits', f'{a}_lblmerged.fits',
                   f'{b}.fits', f'{b}_lblmerged.fits',
                   f'{c}_lblmerged.fits')
    chosen, counts = choose_perframe_variants(paths, prefer=('nrcb',))
    assert chosen == _paths(f'{a}_lblnrcb.fits', f'{b}.fits')
    assert counts == {'frames': 3, 'labelled': 1, 'legacy': 1, 'skipped': 1,
                      'by_label': {'nrcb': 1}}


def test_module_blind_reader_takes_one_per_frame_in_a_fixed_order():
    a = 'f_nrcb1_visit001_vgroup1_exp00001_m7_daophot_basic'
    b = 'f_nrcb1_visit001_vgroup1_exp00002_m7_daophot_basic'
    c = 'f_nrcb1_visit001_vgroup1_exp00003_m7_daophot_basic'
    paths = _paths(f'{a}.fits', f'{a}_lblnrcb.fits', f'{a}_lblmerged.fits',
                   f'{b}.fits', f'{b}_lblnrcb.fits', f'{c}.fits')
    chosen, counts = choose_perframe_variants(
        paths, prefer=MODULE_BLIND_LABEL_ORDER, other_labels=True)
    assert chosen == _paths(f'{a}_lblmerged.fits', f'{b}_lblnrcb.fits',
                            f'{c}.fits')
    assert counts['by_label'] == {'merged': 1, 'nrcb': 1}
    assert counts['legacy'] == 1 and counts['skipped'] == 0
    # the choice does not depend on the order the glob returned them in
    assert choose_perframe_variants(
        paths[::-1], prefer=MODULE_BLIND_LABEL_ORDER, other_labels=True
    )[0] == chosen


def test_other_labels_outside_the_order_come_alphabetically_before_legacy():
    a = 'f_x_exp00001_m4_daophot_basic'
    chosen, _ = choose_perframe_variants(
        _paths(f'{a}.fits', f'{a}_lblzzz.fits', f'{a}_lblaaa.fits'),
        prefer=('nrcb',), other_labels=True)
    assert chosen == _paths(f'{a}_lblaaa.fits')


def test_all_chunks_of_the_chosen_spelling_and_none_of_the_other():
    a = 'f_nrcb1_visit001_vgroup1_exp00001_m7'
    paths = _paths(f'{a}_chunk00of02_daophot_basic.fits',
                   f'{a}_chunk01of02_daophot_basic.fits',
                   f'{a}_chunk00of02_daophot_basic_lblnrcb.fits',
                   f'{a}_chunk01of02_daophot_basic_lblnrcb.fits')
    chosen, counts = choose_perframe_variants(paths, prefer=('nrcb',))
    assert chosen == sorted(p for p in paths if '_lblnrcb' in p)
    assert counts['frames'] == 1 and counts['labelled'] == 1


def test_summary_line_counts_the_legacy_fallback():
    _, counts = choose_perframe_variants(
        _paths('f_exp00001_m7_daophot_basic.fits',
               'f_exp00002_m7_daophot_basic_lblnrcb.fits',
               'f_exp00003_m7_daophot_basic_lblmerged.fits'),
        prefer=('nrcb',))
    line = perframe_variant_summary(counts, 'merge X', 'nrcb')
    assert '3 frame(s); 1 read from merge-label-tokened names (nrcb=1)' in line
    assert '1 from the LEGACY untokened name' in line
    assert "refit this phase for label 'nrcb'" in line
    assert "1 frame(s) carry only another label's fit" in line
    clean = perframe_variant_summary(
        choose_perframe_variants(
            _paths('f_exp00002_m7_daophot_basic_lblnrcb.fits'),
            prefer=('nrcb',))[1], 'merge X', 'nrcb')
    assert 'LEGACY untokened name' in clean and 'refit' not in clean


def _touch(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, 'w').close()
    return path


def test_glob_perframe_catalogs_reads_both_spellings_and_only_catalogs(tmp_path):
    d = tmp_path / 'F480M'
    leg = _touch(str(d / LEGACY))
    tok = _touch(str(d / TOK['nrcb']))
    # residual/model products that the tokened glob ``_lbl*`` would also hit
    _touch(str(d / TOK['nrcb'].replace('.fits', '_residual.fits')))
    _touch(str(d / LEGACY.replace('exp00004', 'exp00005')
               .replace('.fits', '_lblnrcb_mergedcat_model.fits')))
    other = _touch(str(d / LEGACY.replace('exp00004', 'exp00006')))
    pat = str(d / 'f480m_*_visit*_vgroup*_exp*_resbgsub_m7_daophot_basic.fits')
    got, counts = glob_perframe_catalogs(pat)
    assert got == sorted([tok, other])
    assert counts['legacy'] == 1 and counts['by_label'] == {'nrcb': 1}
    got, _ = glob_perframe_catalogs(pat, prefer=('nrca',), other_labels=False)
    assert got == sorted([leg, other])


@pytest.mark.parametrize('pat', ['/d/*_residual.fits', '/d/*_daophot_basic',
                                 '/d/*_daophot_basic_lbl*.fits'])
def test_glob_perframe_catalogs_wants_the_legacy_catalog_shape(pat):
    with pytest.raises(ValueError, match='_daophot_<kind>.fits'):
        glob_perframe_catalogs(pat)


# ---------------------------------------------------------------------------
# provenance header keys
# ---------------------------------------------------------------------------

def test_provenance_keys_round_trip_through_fits(tmp_path):
    seed = ('/orange/x/catalogs/crossband_nrcb_o105_indivexp_merged_'
            'resbgsub_hybpsf_m6_vetted_i2dseed_with_a_long_tail_daophot_basic.fits')
    meta = perframe_provenance_meta('nrcb', seed, None)
    assert meta['MERGELBL'][0] == 'nrcb'
    assert meta['SEEDCAT'] == os.path.basename(seed)
    assert meta['RESBG'] == 'none'
    assert len(meta['SEEDCAT']) > 68, 'needs the CONTINUE convention'
    t = Table({'x': [1.0]})
    t.meta.update(meta)
    fn = str(tmp_path / 'c.fits')
    t.write(fn)
    h = fits.getheader(fn, 1)
    assert h['MERGELBL'] == 'nrcb'
    assert h['SEEDCAT'] == os.path.basename(seed)
    assert h['RESBG'] == 'none'
    assert Table.read(fn).meta['SEEDCAT'] == os.path.basename(seed)


def test_provenance_without_a_label():
    meta = perframe_provenance_meta(None, '', None)
    assert [meta['MERGELBL'][0], meta['SEEDCAT'], meta['RESBG']] == [
        'none', 'none', 'none']


@pytest.mark.parametrize('n', [20, 45, 60, 68, 69, 120])
def test_provenance_cards_write_without_a_truncation_warning(n):
    """Written once per frame: a warning here is thousands of log lines."""
    import warnings
    meta = perframe_provenance_meta('merged', 's' * n, 'r' * n)
    h = fits.Header()
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for k, v in meta.items():
            h[k] = v
        h.tostring()
    assert h['SEEDCAT'] == 's' * n and h['RESBG'] == 'r' * n


# ---------------------------------------------------------------------------
# writer prediction and the skip-if-done sentinel
# ---------------------------------------------------------------------------

def _opts(**kw):
    o = types.SimpleNamespace(
        desaturated=False, bgsub=False, use_iter3_residual_bg=True,
        epsf=False, blur=False, group=False, iteration_label='m7',
        proposal_id='10678', field='105', target='gc-treasury',
        cutout_region='')
    o.__dict__.update(kw)
    return o


@pytest.mark.parametrize('phase', ['m3', 'm4', 'm5', 'm6', 'm7'])
def test_predictor_spells_the_token_at_m3_m7(phase):
    o = _opts(use_iter3_residual_bg=False)
    p = L._predict_tblfilename('/b', 'F480M', 'nrcblong', o, 1, '02101', 4,
                               iteration_label=phase, method='daophot',
                               basic_or_iterative='basic', merge_label='nrcb')
    assert p.endswith(f'_{phase}_daophot_basic_lblnrcb.fits'), p
    legacy = L._predict_tblfilename('/b', 'F480M', 'nrcblong', o, 1, '02101',
                                    4, iteration_label=phase, method='daophot',
                                    basic_or_iterative='basic')
    assert strip_perframe_label(p) == legacy


@pytest.mark.parametrize('phase', ['m1', 'm2'])
def test_predictor_keeps_the_legacy_name_at_m12(phase):
    p = L._predict_tblfilename('/b', 'F480M', 'nrcblong', _opts(), 1, '02101',
                               4, iteration_label=phase, method='daophot',
                               basic_or_iterative='basic', merge_label='nrcb')
    assert p.endswith(f'_{phase}_daophot_basic.fits'), p


def test_predictor_reads_the_phase_off_the_options_when_not_given():
    p = L._predict_tblfilename('/b', 'F480M', 'nrcblong', _opts(), 1, '02101',
                               4, method='daophot', basic_or_iterative='basic',
                               merge_label='merged')
    assert p.endswith('_m7_daophot_basic_lblmerged.fits'), p


class _SentinelOptions:
    def __init__(self):
        self.daophot = False
        self.basic_only = False
        self.desaturated = False
        self.bgsub = False
        self.use_iter3_residual_bg = False
        self.epsf = False
        self.blur = False
        self.group = False
        self.iteration_label = ''
        self.field = '001'
        self.proposal_id = '5365'
        self.manual_start_phase = ''
        self.manual_stop_after_phase = ''


SENT_M7 = 'f150w_nrca1_visit001_vgroup13101_exp00001_resbgsub_m7_daophot_basic'
SENT_M2 = 'f150w_nrca1_visit001_vgroup13101_exp00001_m2_daophot_basic'


def _exists(basepath, phase, merge_label):
    return L._expected_output_exists(
        str(basepath), 'F150W', 'nrca1', _SentinelOptions(), '001', '13101',
        '00001', manual_phase=phase, merge_label=merge_label)


def test_sentinel_accepts_this_labels_file(tmp_path):
    _touch(str(tmp_path / 'F150W' / f'{SENT_M7}_lblnrca.fits'))
    assert _exists(tmp_path, 'm7', 'nrca') is True
    assert _exists(tmp_path, 'm7', 'merged') is False, "another label's file"


def test_sentinel_does_not_accept_a_legacy_file_for_a_label(tmp_path):
    """It predates the token and may hold the other pass's fit: refit it."""
    _touch(str(tmp_path / 'F150W' / f'{SENT_M7}.fits'))
    assert _exists(tmp_path, 'm7', 'nrca') is False
    # a caller that names no label keeps the legacy behaviour
    assert _exists(tmp_path, 'm7', None) is True


def test_sentinel_at_m12_ignores_the_label(tmp_path):
    _touch(str(tmp_path / 'F150W' / f'{SENT_M2}.fits'))
    assert _exists(tmp_path, 'm12', 'nrca') is True
    assert _exists(tmp_path, 'm12', 'merged') is True


# ---------------------------------------------------------------------------
# the cross-label path guard
# ---------------------------------------------------------------------------

def _frame_args(n=3):
    return [{'filtername': 'F480M', 'module': 'nrcblong', 'visit_id': '001',
             'vgroup_id': '02101', 'exposurenumber': e,
             'filename': f'/x/jw10678105001_02101_{e:05d}_nrcblong_crf.fits'}
            for e in range(1, n + 1)]


def _claim(owners, phase, label, merge_label_in_name):
    paths = [C._predicted_perframe_catalog('/cut', a, _opts(), phase,
                                           merge_label_in_name)
             for a in _frame_args()]
    C.claim_perframe_outputs(owners, phase, label, 'F480M', paths)
    return paths


def test_tokened_names_of_two_labels_do_not_clash():
    owners = {}
    p_m = _claim(owners, 'm7', 'merged', 'merged')
    p_b = _claim(owners, 'm7', 'nrcb', 'nrcb')
    assert not set(p_m) & set(p_b)
    assert len(owners) == 6


def test_the_guard_refuses_two_labels_on_one_path_at_m7():
    """What the token fixes, held fixed: names without the label clash."""
    owners = {}
    _claim(owners, 'm7', 'merged', None)
    with pytest.raises(C.PerframeLabelCollisionError,
                       match=r"3 per-frame output path\(s\) are also written"):
        _claim(owners, 'm7', 'nrcb', None)


def test_the_guard_lets_one_label_claim_its_paths_again():
    owners = {}
    _claim(owners, 'm4', 'nrcb', None)
    _claim(owners, 'm4', 'nrcb', None)


@pytest.mark.parametrize('phase', ['m2', 'm1'])
def test_the_guard_is_silent_at_m12(phase):
    """One fit serves every label at m12 by design (#841)."""
    owners = {}
    _claim(owners, phase, 'merged', 'merged')
    _claim(owners, phase, 'nrcb', 'nrcb')
    assert owners == {}


def test_the_fan_out_calls_the_guard_on_the_tokened_prediction():
    """``run_manual_pipeline`` cannot run in a unit test; pin the call site."""
    src = open(C.__file__).read()
    call = src.split('claim_perframe_outputs(\n')[1].split('for a in frame_args])')[0]
    assert '_phase_output_owner, merge_label, module, filt' in call
    assert '_predicted_perframe_catalog(cut_bp, a, opts_phase' in call
    assert 'merge_label, module)' in call
    # one owner map per PHASE, shared by every label of the phase
    phase_loop = src.split("merge_label = 'm2' if phase == 'm12' else phase")[1]
    assert phase_loop.split('\n')[3].strip() == '_phase_output_owner = {}'
    # and the per-frame args carry the label to the writer
    assert "'merge_label': module," in src
    assert "merge_label=args.get('merge_label'))" in src


# ---------------------------------------------------------------------------
# the resume: a marker without this label's catalog is refit
# ---------------------------------------------------------------------------

def test_markers_without_the_labelled_catalog_are_refit(tmp_path, capsys):
    args = _frame_args()
    cut = str(tmp_path)
    have = C._predicted_perframe_catalog(cut, args[0], _opts(), 'm7', 'nrcb')
    _touch(have)
    # frame 2 has only the legacy name, frame 3 nothing
    _touch(C._predicted_perframe_catalog(cut, args[1], _opts(), 'm7', None))
    ok = [a['filename'] for a in args]
    todo, ok2 = C._refit_markers_without_labelled_product(
        [], ok, args, cut, _opts(), 'm7', 'nrcb', 'F480M')
    assert ok2 == [args[0]['filename']]
    assert [a['filename'] for a in todo] == [args[1]['filename'],
                                             args[2]['filename']]
    assert 'REFITTING 2 frame(s)' in capsys.readouterr().out


def test_markers_at_m12_are_left_alone(tmp_path):
    args = _frame_args()
    ok = [a['filename'] for a in args]
    todo, ok2 = C._refit_markers_without_labelled_product(
        ['t'], ok, args, str(tmp_path), _opts(), 'm2', 'nrcb', 'F480M')
    assert todo == ['t'] and ok2 == ok


def test_the_resume_calls_the_refit():
    """``run_manual_pipeline`` cannot run in a unit test; pin the call site,
    including the condition it runs under (every resume, not never)."""
    src = open(C.__file__).read()
    block = src.split('_todo, _ok, _nov, _stale = select_resumable_frames(')[1]
    block = block.split('if _resume and not _cross_label:')[0]
    assert '_refit_markers_without_labelled_product(' in block
    assert 'merge_label, module, filt)' in block
    lines = block.splitlines()
    at = next(i for i, ln in enumerate(lines)
              if '_refit_markers_without_labelled_product(' in ln)
    assert lines[at].strip().startswith('_todo, _ok = ')
    assert lines[at - 1].strip() == 'if _resume:'


# ---------------------------------------------------------------------------
# the per-label merge: own label, else legacy, never another label's
# ---------------------------------------------------------------------------

def _perframe(basepath, exp, label, tag, phase='m4'):
    o = _opts(use_iter3_residual_bg=False, field='001', iteration_label=None)
    fn = L._predict_tblfilename(str(basepath), 'F480M', 'nrcblong', o, 1,
                                '02101', exp, iteration_label=phase,
                                method='daophot', basic_or_iterative='basic',
                                merge_label=label)
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    t = Table({'flux_fit': [1.0]})
    t.meta['FILENAME'] = f'/x/{tag}_exp{exp:05d}_crf.fits'
    t.write(fn)
    return fn


@pytest.fixture
def merge_tree(tmp_path, monkeypatch):
    (tmp_path / 'catalogs').mkdir()
    seen = []

    def _fake(tbls, offsets_table=None, filtername=None):
        seen.append(sorted(t.meta['FILENAME'] for t in tbls))
        return Table({'skycoord_avg': SkyCoord([10.0] * u.deg, [-5.0] * u.deg)})

    monkeypatch.setattr(MC, 'combine_singleframe', _fake)
    # exp1: own + other + legacy; exp2: other + legacy; exp3: other only
    _perframe(tmp_path, 1, 'nrcb', 'own')
    _perframe(tmp_path, 1, 'merged', 'other')
    _perframe(tmp_path, 1, None, 'legacy')
    _perframe(tmp_path, 2, 'merged', 'other')
    _perframe(tmp_path, 2, None, 'legacy')
    _perframe(tmp_path, 3, 'merged', 'other')
    return tmp_path, seen


def _merge(basepath, module, phase='m4'):
    MC.merge_individual_frames(
        module=module, filtername='f480m', progid='10678', method='dao',
        suffix='_basic', target='gc-treasury', basepath=str(basepath),
        iteration_label=phase, field='001', do_replace_saturated=False)


def test_per_label_merge_reads_own_then_legacy_never_another(merge_tree, capsys):
    tmp_path, seen = merge_tree
    _merge(tmp_path, 'nrcb')
    assert seen == [['/x/legacy_exp00002_crf.fits', '/x/own_exp00001_crf.fits']]
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines()
             if ln.startswith('merge_individual_frames F480M/nrcb m4:')]
    assert len(lines) == 1, out
    # exp3 (another label's fit only) is not even globbed: the merge globs
    # its own token and the legacy name
    assert '2 frame(s); 1 read from merge-label-tokened names (nrcb=1)' in lines[0]
    assert '1 from the LEGACY untokened name' in lines[0]


def test_the_merged_label_reads_its_own_fits(merge_tree):
    tmp_path, seen = merge_tree
    _merge(tmp_path, 'merged')
    assert seen == [[f'/x/other_exp{e:05d}_crf.fits' for e in (1, 2, 3)]]


def test_m2_merge_is_unchanged(tmp_path, monkeypatch):
    (tmp_path / 'catalogs').mkdir()
    seen = []
    monkeypatch.setattr(
        MC, 'combine_singleframe',
        lambda tbls, **k: seen.append(sorted(t.meta['FILENAME'] for t in tbls))
        or Table({'skycoord_avg': SkyCoord([10.0] * u.deg, [-5.0] * u.deg)}))
    _perframe(tmp_path, 1, 'nrcb', 'legacy', phase='m2')   # m2: no token
    assert '_lbl' not in os.listdir(tmp_path / 'F480M')[0]
    _merge(tmp_path, 'nrcb', phase='m2')
    assert seen == [['/x/legacy_exp00001_crf.fits']]


# ---------------------------------------------------------------------------
# the per-label residual mosaic
# ---------------------------------------------------------------------------

def test_residual_mosaic_reads_own_then_legacy(tmp_path, monkeypatch, capsys):
    pipe = tmp_path / 'F480M' / 'pipeline'
    stem = ('jw10678-o001_t001_nircam_clear-f480m-nrcblong_visit001_vgroup02101_'
            'exp{e:05d}_m4_daophot_basic{t}_residual.fits')
    want = []
    for e, tokens in ((1, ('', '_lblmerged', '_lblnrcb')), (2, ('', '_lblnrcb')),
                      (3, ('_lblnrcb',))):
        for t in tokens:
            p = _touch(str(pipe / stem.format(e=e, t=t)))
            if (e, t) in ((1, '_lblmerged'), (2, '')):
                want.append(p)

    class _Seen(Exception):
        pass

    def _asn(files, **kw):
        raise _Seen(sorted(files))

    monkeypatch.setattr(L.asn_from_list, 'asn_from_list', _asn)
    with pytest.raises(_Seen) as ex:
        L.mosaic_each_exposure_residuals(
            str(tmp_path), 'F480M', '10678', '001', 'merged',
            residual_kind='basic', iteration_label='m4')
    assert ex.value.args[0] == sorted(want)
    out = capsys.readouterr().out
    # exp3 (nrcb's fit only) is not globbed for the merged mosaic
    assert '2 frame(s); 1 read from merge-label-tokened names (merged=1)' in out
    assert '1 from the LEGACY untokened name' in out


# ---------------------------------------------------------------------------
# module-blind readers
# ---------------------------------------------------------------------------

def _cat(dirpath, name, src='/x/jw10678001001_10678001_00001_nrcblong_o001_crf.fits'):
    p = os.path.join(str(dirpath), name)
    t = Table({'x': [1.0]})
    t.meta['FILENAME'] = src
    t.write(p)
    return p


def test_astrometry_checkpoint_measures_one_fit_per_frame(tmp_path, monkeypatch):
    cut_bp = tmp_path / 'cutouts' / 'merged'
    d = cut_bp / 'F212N'
    d.mkdir(parents=True)
    base = 'f212n_nrcblong_o001_visit001_vgroup10678001_exp{e:05d}_m4_daophot_basic'
    # exp4: only a label outside MODULE_BLIND_LABEL_ORDER (a module-blind
    # reader still takes it -- it is a fit of the frame)
    for e, toks in ((1, ('', '_lblnrcb', '_lblmerged')), (2, ('', '_lblnrcb')),
                    (3, ('',)), (4, ('_lblnrcblong',))):
        for t in toks:
            _cat(d, base.format(e=e) + t + '.fits')

    class _Stop(Exception):
        pass

    seen = {}

    def _watch(fns, *a, **k):
        seen['in'] = sorted(os.path.basename(f) for f in fns)
        raise _Stop

    monkeypatch.setattr(C, '_drop_foreign_obs_duplicates', _watch)
    monkeypatch.delenv('ASTROM_CHECKPOINT', raising=False)
    o = _opts(field='001', iteration_label=None, use_iter3_residual_bg=False)
    o.modules = 'nrcblong'
    o.each_exposure = True
    with pytest.raises(_Stop):
        C._run_astrometry_stage_checkpoint(
            'm4', 'nrcblong', 'F212N', str(cut_bp), str(tmp_path), '10678',
            o, {}, context='test')
    assert seen['in'] == [base.format(e=1) + '_lblmerged.fits',
                          base.format(e=2) + '_lblnrcb.fits',
                          base.format(e=3) + '.fits',
                          base.format(e=4) + '_lblnrcblong.fits']


def test_families_on_disk_sees_a_tokened_only_tree(tmp_path):
    d = tmp_path / 'F212N'
    d.mkdir()
    _cat(d, 'f212n_nrcb1_o001_visit001_vgroup10678001_exp00001_m4_daophot_basic'
            '_lblnrcb.fits')
    got = C.perframe_families_on_disk(str(tmp_path), 'F212N', 'm4',
                                      obs_token='_o001', psf_token='')
    assert set(got) == {'nrcb'}


def test_retention_patterns_take_the_token():
    from jwst_gc_pipeline import retention as R
    raw = ('jw10678-o105_t001_nircam_clear-f480m-nrcblong_visit001_vgroup02101_'
           'exp00004_resbgsub_m7_daophot_basic_lblnrcb_residual.fits')
    mc = raw.replace('_residual.fits', '_mergedcat_model.fits')
    m = R.PERFRAME_RAW_RE.search(raw)
    assert m and m.group('label') == 'm7'
    assert not R.PERFRAME_MERGEDCAT_RE.search(raw)
    m = R.PERFRAME_MERGEDCAT_RE.search(mc)
    assert m and m.group('label') == 'm7'
    assert not R.PERFRAME_RAW_RE.search(mc)
    # the legacy spellings still match as before
    assert R.PERFRAME_RAW_RE.search(strip_perframe_label(raw))
    assert R.PERFRAME_MERGEDCAT_RE.search(strip_perframe_label(mc))


def test_virac2_perframe_matches_takes_the_token_not_the_group_variant():
    from jwst_gc_pipeline.reduction.build_virac2_offsets import perframe_matches
    assert perframe_matches('f_nrcb1_visit001_vgroup1_exp00001_m3_daophot_basic'
                            '_lblnrcb.fits', '_m3')
    assert perframe_matches('f_nrcb1_visit001_vgroup1_exp00001_m3_daophot_basic'
                            '.fits', '_m3')
    assert not perframe_matches('f_nrcb1_visit001_vgroup1_exp00001_group_m3_'
                                'daophot_basic_lblnrcb.fits', '_m3')
    assert not perframe_matches('f_nrcb1_visit001_vgroup1_exp00001_m3_daophot_'
                                'basic_lblnrcb_residual.fits', '_m3')


def test_step0_default_catalogs_one_per_frame(tmp_path):
    from jwst_gc_pipeline.reduction.bulk_offset_step0 import (
        default_catalog_glob, default_catalog_paths, duplicate_exposure_catalogs)
    d = tmp_path / 'F212N'
    names = ['f212n_nrcb1_visit001_vgroup1_exp00001_m3_daophot_basic.fits',
             'f212n_nrcb1_visit001_vgroup1_exp00001_m3_daophot_basic_lblnrcb.fits',
             'f212n_nrcb1_visit001_vgroup1_exp00001_m3_daophot_basic_lblmerged.fits',
             'f212n_nrcb1_visit001_vgroup1_exp00002_m3_daophot_basic.fits']
    for n in names:
        _touch(str(d / n))
    paths, counts = default_catalog_paths(str(tmp_path), 'F212N', '_m3')
    assert [os.path.basename(p) for p in paths] == [names[2], names[3]]
    assert counts['by_label'] == {'merged': 1} and counts['legacy'] == 1
    assert not duplicate_exposure_catalogs(paths)
    # m1 is the glob's result
    _touch(str(d / 'f212n_nrcb1_visit001_vgroup1_exp00001_m1_daophot_basic.fits'))
    import glob
    assert default_catalog_paths(str(tmp_path), 'F212N', '_m1')[0] == sorted(
        glob.glob(default_catalog_glob(str(tmp_path), 'F212N', '_m1')))
