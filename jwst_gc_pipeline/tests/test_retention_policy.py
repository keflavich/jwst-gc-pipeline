"""The retention policy is only as good as what it REFUSES to select.

Every test here is written from that side: the live-product list is the contract
(if a future rule starts matching one of those names, the pipeline loses an
input), the guard tests check each veto independently, and the apply tests check
that the safe defaults really are safe.
"""
import json
import os

import pytest

from jwst_gc_pipeline import retention


PRE = 'jw02221-o001_t001_nircam_clear-f410m'
FRAME = 'nrcalong_visit001_vgroup11101_exp00001'


def _perframe(label, kind='basic', what='residual', mergedcat=False):
    tag = '_mergedcat' if mergedcat else ''
    return f'{PRE}-{FRAME}_{label}_daophot_{kind}{tag}_{what}.fits'


def _mosaic(label, what='residual_i2d'):
    return f'{PRE}-merged_{label}_daophot_basic_mergedcat_{what}.fits'


def _write(directory, name, size=1024, age_days=400):
    p = os.path.join(directory, name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'wb') as fh:
        fh.write(b'\0' * size)
    old = os.stat(p).st_mtime - age_days * 86400
    os.utime(p, (old, old))
    return p


# --------------------------------------------------------------------------
# What the policy must never touch
# --------------------------------------------------------------------------

LIVE_PRODUCTS = [
    # mosaics -- every phase's, because the next phase reads them
    _mosaic('m4'),
    _mosaic('m4', 'residual_smoothed_bg_i2d'),
    _mosaic('m7'),
    'jw02221-o001_t001_nircam_clear-f410m-merged_i2d.fits',
    # exposure-level science products, including the ones a release symlinks to
    'jw02221001001_03101_00003_nrcalong_destreak_o001_crf.fits',
    'jw02221001001_03101_00003_nrcalong_cal.fits',
    'jw02221001001_03101_00003_nrcalong_destreak.fits',
    'jw02221001001_03101_00003_nrcalong_rate.fits',
    # catalogs at every stage: the scientific record, explicitly out of scope
    'f410m_merged_indivexp_merged_resbgsub_m7_dao_basic.fits',
    'f410m_merged_indivexp_merged_m3_dao_basic.fits',
    'basic_merged_indivexp_photometry_tables_merged_resbgsub_m7.fits',
    'basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup.fits',
    'f410m_merged_consensus.fits',
    'f410m_merged_satstar_reconciled_m12.fits',
    # astrometry state
    'jw02221-o001_t001_nircam_clear-f410m-merged_mergedcat_grid_o001_f410m.asdf',
]


@pytest.mark.parametrize('name', LIVE_PRODUCTS)
def test_live_products_are_never_selected(name):
    """A live product must not match any rule, including the opt-in ones.

    This is the test that fails loudly if someone widens a pattern: the cost of
    a false positive here is a pipeline that cannot restart.
    """
    rule, why = retention.classify(f'/data/brick/F410M/pipeline/{name}',
                                   {}, enabled=set(retention.POLICY))
    assert rule is None, f'{name} selected by {rule.name if rule else None}: {why}'


def test_final_phase_perframe_pair_is_kept(tmp_path):
    """The last phase on disk keeps its raw residual/model.

    There is no later barrier to retire them, and they are the only rendering of
    what the released catalog did and did not subtract.
    """
    d = str(tmp_path)
    for label in ('m3', 'm4', 'm5'):
        _write(d, _perframe(label))
        _write(d, _perframe(label, what='model'))
        _write(d, _mosaic(label))
    ctx = retention._directory_context(d, os.listdir(d))
    assert ctx['final_label'] == 'm5'

    rule, _ = retention.classify(os.path.join(d, _perframe('m5')), ctx)
    assert rule is None
    rule, why = retention.classify(os.path.join(d, _perframe('m3')), ctx)
    assert rule is not None and rule.name == 'superseded_perframe'
    assert 'm3' in why and 'm5' in why


def test_m12_sorts_before_m3():
    """m12 is iter1+iter2 fused, and runs FIRST -- not phase twelve.

    Read as a number it would look like the newest phase and would protect
    itself while retiring everything real.
    """
    assert retention._label_ordinal('m12') < retention._label_ordinal('m3')
    assert retention._label_ordinal('m3') < retention._label_ordinal('m7')
    assert retention._label_ordinal('iter2') < retention._label_ordinal('m12')


def test_mergedcat_frames_need_their_i2d(tmp_path):
    """An interrupted phase is not a cleanup target."""
    d = str(tmp_path)
    _write(d, _perframe('m4', mergedcat=True))
    _write(d, _perframe('m4', mergedcat=True, what='model'))

    ctx = retention._directory_context(d, os.listdir(d))
    rule, _ = retention.classify(os.path.join(d, _perframe('m4', mergedcat=True)),
                                 ctx)
    assert rule is None, 'no i2d on disk yet, so the render is still needed'

    _write(d, _mosaic('m4'))
    ctx = retention._directory_context(d, os.listdir(d))
    rule, why = retention.classify(
        os.path.join(d, _perframe('m4', mergedcat=True)), ctx)
    assert rule is not None and rule.name == 'spent_mergedcat_frame'


# --------------------------------------------------------------------------
# The in-run helpers the phase barrier calls
# --------------------------------------------------------------------------

def test_perframe_helpers_scope_to_one_observation(tmp_path):
    """A pipeline directory shared by two observations must not leak.

    cloudef's obs002 and obs005 share one directory and one per-frame naming
    scheme, so an unscoped glob deletes the other observation's frames.
    """
    d = str(tmp_path)
    other = _perframe('m3').replace('-o001_', '-o005_')
    _write(d, _perframe('m3'))
    _write(d, _perframe('m3', what='model'))
    _write(d, other)
    _write(d, _mosaic('m4'))
    _write(d, _perframe('m4', mergedcat=True))

    doomed = retention.superseded_perframe_products(
        d, proposal_id='2221', field='001', filtername='F410M', label='m3')
    assert [os.path.basename(p) for p in doomed] == sorted(
        [_perframe('m3'), _perframe('m3', what='model')])
    assert all('-o005_' not in p for p in doomed)


def test_superseded_helper_never_returns_mergedcat_or_i2d(tmp_path):
    d = str(tmp_path)
    _write(d, _perframe('m4'))
    _write(d, _perframe('m4', mergedcat=True))
    _write(d, _mosaic('m4'))
    _write(d, _mosaic('m4', 'residual_smoothed_bg_i2d'))

    doomed = retention.superseded_perframe_products(
        d, proposal_id='2221', field='001', filtername='F410M', label='m4')
    assert [os.path.basename(p) for p in doomed] == [_perframe('m4')]

    spent = retention.spent_mergedcat_frames(
        d, proposal_id='2221', field='001', filtername='F410M', label='m4')
    assert [os.path.basename(p) for p in spent] == [
        _perframe('m4', mergedcat=True)]


def test_spent_mergedcat_helper_silent_without_i2d(tmp_path):
    d = str(tmp_path)
    _write(d, _perframe('m4', mergedcat=True))
    assert retention.spent_mergedcat_frames(
        d, proposal_id='2221', field='001', filtername='F410M',
        label='m4') == []


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------

def test_release_symlink_target_is_protected(tmp_path):
    """The v1.3 pattern: a release publishes a symlink into the live tree."""
    live = tmp_path / 'brick' / 'F410M' / 'pipeline'
    live.mkdir(parents=True)
    target = _write(str(live), 'core.python-7-1684788000-x.63550')
    releases = tmp_path / 'releases' / 'v1.3' / 'brick' / 'exposures'
    releases.mkdir(parents=True)
    os.symlink(target, str(releases / 'published.fits'))

    targets = retention.release_symlink_targets(str(tmp_path / 'releases'))
    assert os.path.realpath(target) in targets

    guard = retention.Guard(release_targets=targets, min_age_days=0)
    kept = retention.plan([str(tmp_path / 'brick')], guard=guard)
    assert kept == []

    shown = retention.plan([str(tmp_path / 'brick')], guard=guard,
                           include_vetoed=True)
    assert len(shown) == 1 and 'release symlinks' in shown[0].vetoed_by


def test_busy_field_vetoes(tmp_path):
    d = tmp_path / 'brick' / 'F410M' / 'pipeline'
    d.mkdir(parents=True)
    _write(str(d), 'core.python-7-1684788000-x.63550')
    guard = retention.Guard(busy_fields=frozenset({'brick'}), min_age_days=0)
    assert retention.plan([str(tmp_path)], guard=guard) == []


def test_busy_targets_parses_job_names():
    out = '\n'.join(['brick2221-o001-m5-finalize',
                     'cloudc2221-o002-m7-fanout',
                     'gc-monitor-refresh',
                     'interactive'])
    busy = retention.busy_targets(['brick', 'cloudc', 'sgrb2'],
                                  squeue_output=out)
    assert busy == frozenset({'brick', 'cloudc'})


def test_missing_squeue_refuses_rather_than_assuming_idle(monkeypatch):
    """No scheduler answer means no plan; silence must not read as 'idle'."""
    def _boom(*a, **k):
        raise OSError('squeue: command not found')
    monkeypatch.setattr(retention.subprocess, 'run', _boom)
    with pytest.raises(retention.RetentionError, match='refusing to plan'):
        retention.busy_targets(['brick'])


def test_age_floor_protects_recent_files(tmp_path):
    d = tmp_path / 'brick' / 'F410M' / 'pipeline'
    d.mkdir(parents=True)
    _write(str(d), 'core.python-7-1684788000-x.63550', age_days=1)
    guard = retention.Guard(min_age_days=30)
    assert retention.plan([str(tmp_path)], guard=guard) == []


def test_walk_visits_a_doubly_reachable_tree_once(tmp_path):
    """brick is reachable as /orange/.../brick and /blue/.../brick.

    A naive walk offers the same bytes twice and a manifest double-counts what
    it freed.
    """
    real = tmp_path / 'blue' / 'brick' / 'F410M' / 'pipeline'
    real.mkdir(parents=True)
    _write(str(real), 'core.python-7-1684788000-x.63550')
    orange = tmp_path / 'orange'
    orange.mkdir()
    os.symlink(str(tmp_path / 'blue' / 'brick'), str(orange / 'brick'))

    guard = retention.Guard(min_age_days=0)
    found = retention.plan([str(tmp_path / 'blue'), str(orange)], guard=guard,
                           follow_symlinks=True)
    assert len(found) == 1


# --------------------------------------------------------------------------
# Applying
# --------------------------------------------------------------------------

def test_dry_run_deletes_nothing(tmp_path):
    d = tmp_path / 'brick' / 'F410M' / 'pipeline'
    d.mkdir(parents=True)
    victim = _write(str(d), 'core.python-7-1684788000-x.63550')
    manifest = tmp_path / 'plan.json'

    candidates = retention.plan([str(tmp_path)],
                                guard=retention.Guard(min_age_days=0))
    summary = retention.apply(candidates, dry_run=True,
                              manifest_path=str(manifest))

    assert os.path.exists(victim), 'a dry run must not touch the filesystem'
    assert summary['deleted'] == 0 and summary['deletable'] == 1
    written = json.loads(manifest.read_text())
    assert written['dry_run'] is True
    assert written['candidates'][0]['path'] == victim
    assert 'core dump' in written['candidates'][0]['reason']


def test_apply_without_manifest_refuses(tmp_path):
    d = tmp_path / 'brick' / 'F410M' / 'pipeline'
    d.mkdir(parents=True)
    victim = _write(str(d), 'core.python-7-1684788000-x.63550')
    candidates = retention.plan([str(tmp_path)],
                                guard=retention.Guard(min_age_days=0))
    with pytest.raises(retention.RetentionError, match='without a manifest'):
        retention.apply(candidates, dry_run=False, manifest_path=None)
    assert os.path.exists(victim)


def test_apply_writes_manifest_before_deleting(tmp_path):
    d = tmp_path / 'brick' / 'F410M' / 'pipeline'
    d.mkdir(parents=True)
    victim = _write(str(d), 'core.python-7-1684788000-x.63550')
    manifest = tmp_path / 'done.json'
    candidates = retention.plan([str(tmp_path)],
                                guard=retention.Guard(min_age_days=0))
    summary = retention.apply(candidates, dry_run=False,
                              manifest_path=str(manifest))
    assert not os.path.exists(victim)
    assert summary['deleted'] == 1
    assert json.loads(manifest.read_text())['candidates'][0]['path'] == victim


# --------------------------------------------------------------------------
# Quarantine directories
# --------------------------------------------------------------------------

def test_quarantine_directory_is_planned_as_one_unit(tmp_path):
    field = tmp_path / 'brick' / 'F410M' / 'pipeline'
    backup = field / 'pre_skycoord_fix_backup_20260602'
    backup.mkdir(parents=True)
    _write(str(backup), 'anything.fits', size=2048)
    _write(str(backup), 'anything_else.fits', size=2048)
    _write(str(field), _mosaic('m7'))

    found = retention.plan_quarantine_directories(
        [str(tmp_path)], guard=retention.Guard(min_age_days=0),
        min_age_days=0)
    assert len(found) == 1
    assert found[0].path.endswith('pre_skycoord_fix_backup_20260602')
    assert found[0].size == 4096, 'size is the whole tree, not one file'


def test_quarantine_directory_with_a_release_symlink_is_vetoed(tmp_path):
    backup = tmp_path / 'brick' / 'F410M' / 'pipeline' / 'backup_20251210'
    backup.mkdir(parents=True)
    inside = _write(str(backup), 'still_published.fits')
    releases = tmp_path / 'releases' / 'v1.1'
    releases.mkdir(parents=True)
    os.symlink(inside, str(releases / 'published.fits'))

    guard = retention.Guard(
        release_targets=retention.release_symlink_targets(
            str(tmp_path / 'releases')),
        min_age_days=0)
    assert retention.plan_quarantine_directories(
        [str(tmp_path)], guard=guard, min_age_days=0) == []


def test_recently_touched_backup_is_not_stale(tmp_path):
    backup = tmp_path / 'brick' / 'catalogs' / 'backup_20251210'
    backup.mkdir(parents=True)
    _write(str(backup), 'old.fits', age_days=400)
    _write(str(backup), 'someone_is_still_using_this.fits', age_days=1)
    found = retention.plan_quarantine_directories(
        [str(tmp_path)], guard=retention.Guard(min_age_days=0),
        min_age_days=90)
    assert found == []


# --------------------------------------------------------------------------
# In-run smoothed-bg pruning
# --------------------------------------------------------------------------

def test_superseded_smoothed_bg_selects_only_that_labels_map(tmp_path):
    bg = _write(str(tmp_path), _mosaic('m4', 'residual_smoothed_bg_i2d'))
    assert retention.superseded_smoothed_bg(bg, label='m4') == [bg]
    # the label must match: m4's map is not m5's, and m2 is not m12
    assert retention.superseded_smoothed_bg(bg, label='m5') == []
    m12 = _write(str(tmp_path), _mosaic('m12', 'residual_smoothed_bg_i2d'))
    assert retention.superseded_smoothed_bg(m12, label='m2') == []


@pytest.mark.parametrize('name', [
    _mosaic('m4', 'residual_i2d'),                    # the diagnostic residual
    _mosaic('m4', 'model_i2d'),                       # the diagnostic model
    # the m2 checkpoint's renamed quarantine copy
    f'{PRE}-merged_m4_daophot_basic_mergedcat_residual_smoothed_bg_i2d'
    '_im0_badastrom.fits',
    f'{PRE}-merged_data_i2d.fits',
    _perframe('m4', what='residual'),
])
def test_superseded_smoothed_bg_refuses_every_other_mosaic(tmp_path, name):
    p = _write(str(tmp_path), name)
    assert retention.superseded_smoothed_bg(p, label='m4') == []


def test_superseded_smoothed_bg_refuses_protected_directories(tmp_path):
    p = _write(str(tmp_path / 'releases' / 'v1'),
               _mosaic('m4', 'residual_smoothed_bg_i2d'))
    assert retention.superseded_smoothed_bg(p, label='m4') == []


def test_superseded_smoothed_bg_missing_file_is_empty(tmp_path):
    p = str(tmp_path / _mosaic('m4', 'residual_smoothed_bg_i2d'))
    assert retention.superseded_smoothed_bg(p, label='m4') == []


def test_phase_loop_prunes_two_phases_back(tmp_path):
    """m6's barrier removes m4's map and keeps m5's (a retry of m6 reads it)."""
    from types import SimpleNamespace
    from jwst_gc_pipeline.photometry import cataloging as C

    opts = SimpleNamespace(desaturated=False, bgsub=False, group=False)
    kw = dict(cut_bp=str(tmp_path), proposal_id='2221', field='001',
              module='merged', filt='F410M', options=opts, pupil='clear')
    pipe = tmp_path / 'F410M' / 'pipeline'
    paths = {}
    for label in ('m2', 'm3', 'm4', 'm5', 'm6'):
        p = C._reconstruct_smoothed_bg_path(label=label, **kw)
        _write(str(pipe), os.path.basename(p))
        resid = p.replace('_smoothed_bg_i2d.fits', '_i2d.fits')
        _write(str(pipe), os.path.basename(resid))
        paths[label] = (p, resid)

    for phase in ('m12', 'm3'):           # nothing two phases back yet
        C._gc_superseded_smoothed_bg(phase=phase, **kw)
    assert all(os.path.exists(p) for p, _ in paths.values())

    C._gc_superseded_smoothed_bg(phase='m4', **kw)  # m12 -> m2
    C._gc_superseded_smoothed_bg(phase='m5', **kw)
    C._gc_superseded_smoothed_bg(phase='m6', **kw)
    assert [lab for lab, (p, _) in paths.items() if os.path.exists(p)] == \
        ['m5', 'm6']
    # residual mosaics are diagnostics and stay
    assert all(os.path.exists(r) for _, r in paths.values())


def test_prune_does_not_depend_on_runtime_phase_list(tmp_path):
    """Per-phase finalize jobs run --manual-start-phase=X
    --manual-stop-after-phase=X, so the driver's ``phases`` list holds only X.
    Pruning must still remove the map two phases back in the chain order."""
    import inspect
    from types import SimpleNamespace
    from jwst_gc_pipeline.photometry import cataloging as C

    assert 'phases' not in inspect.signature(C._gc_superseded_smoothed_bg).parameters
    opts = SimpleNamespace(desaturated=False, bgsub=False, group=False)
    kw = dict(cut_bp=str(tmp_path), proposal_id='10678', field='067',
              module='merged', filt='F212N', options=opts, pupil='clear')
    pipe = tmp_path / 'F212N' / 'pipeline'
    for label in ('m2', 'm3', 'm4', 'm5', 'm6', 'm7'):
        _write(str(pipe), os.path.basename(
            C._reconstruct_smoothed_bg_path(label=label, **kw)))
    # one separate finalize job per phase
    for phase in ('m12', 'm3', 'm4', 'm5', 'm6', 'm7'):
        C._gc_superseded_smoothed_bg(phase=phase, **kw)
    left = sorted(f for f in os.listdir(pipe) if 'smoothed_bg' in f)
    assert len(left) == 2
    assert any('_m6_' in f for f in left) and any('_m7_' in f for f in left)


def _smoothed_bg_kw(tmp_path, **opt):
    from types import SimpleNamespace
    opts = SimpleNamespace(desaturated=False, bgsub=False, group=False, **opt)
    return dict(cut_bp=str(tmp_path), proposal_id='10678', field='067',
                module='merged', filt='F212N', options=opts, pupil='clear')


@pytest.mark.parametrize('keep', [False, True])
def test_barrier_hook_honours_keep_flag(tmp_path, keep):
    """The barrier hook prunes m4's map at m6 unless the keep flag is set."""
    from jwst_gc_pipeline.photometry import cataloging as C
    kw = _smoothed_bg_kw(tmp_path, manual_keep_intermediate_smoothed_bg=keep)
    pipe = tmp_path / 'F212N' / 'pipeline'
    m4 = C._reconstruct_smoothed_bg_path(label='m4', **kw)
    _write(str(pipe), os.path.basename(m4))
    C._prune_smoothed_bg_after_barrier(phase='m6', **kw)
    assert os.path.exists(m4) is keep


def test_phase_loop_calls_barrier_hook():
    """run_manual_pipeline's barrier calls the prune hook (a deleted or
    disabled call would otherwise pass every unit test above)."""
    import ast
    import inspect
    import textwrap
    from jwst_gc_pipeline.photometry import cataloging as C
    tree = ast.parse(textwrap.dedent(inspect.getsource(C.run_manual_pipeline)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, 'id', None) == '_prune_smoothed_bg_after_barrier']
    assert len(calls) == 1
    # not guarded by a constant-false test
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and isinstance(node.test, ast.Constant) \
                and not node.test.value:
            assert not any(c in list(ast.walk(node)) for c in calls)


@pytest.mark.parametrize('start', ['m5', 'm6', 'm7'])
def test_start_phase_requires_bg_for_phases_that_read_it(tmp_path, start):
    from jwst_gc_pipeline.photometry import cataloging as C
    with pytest.raises(FileNotFoundError):
        C._start_phase_prev_smoothed_bg(start, 'mX', str(tmp_path / 'nope.fits'),
                                        'F212N', 'merged')


@pytest.mark.parametrize('start', ['m3', 'm4'])
def test_start_phase_tolerates_absent_bg_for_raw_frame_phases(tmp_path, start):
    from jwst_gc_pipeline.photometry import cataloging as C
    assert C._start_phase_prev_smoothed_bg(
        start, 'mX', str(tmp_path / 'nope.fits'), 'F212N', 'merged') is None
    present = tmp_path / 'bg.fits'
    present.write_bytes(b'x')
    assert C._start_phase_prev_smoothed_bg(
        start, 'mX', str(present), 'F212N', 'merged') == str(present)


def test_reading_phases_match_chain_order():
    """Every phase after m4 in the chain subtracts its predecessor's map."""
    from jwst_gc_pipeline.photometry import cataloging as C
    order = C._SMOOTHED_BG_PHASE_ORDER
    assert C._PHASES_READING_PREV_SMOOTHED_BG == order[order.index('m5'):]


def _reffield_run(tmp_path, with_bg):
    cat = tmp_path / 'catalogs'
    cat.mkdir()
    (cat / 'f212n_merged_indivexp_merged_resbgsub_m4_dao_basic_vetted.fits').write_bytes(b'x')
    pipe = tmp_path / 'F212N' / 'pipeline'
    pipe.mkdir(parents=True)
    stem = 'jw1-o1_t1_nircam_clear-f212n-merged_m4_daophot_basic_mergedcat_residual'
    (pipe / f'{stem}_i2d.fits').write_bytes(b'x')
    (pipe / 'jw1-o1_t1_nircam_clear-f212n-merged_data_i2d.fits').write_bytes(b'x')
    if with_bg:
        (pipe / f'{stem}_smoothed_bg_i2d.fits').write_bytes(b'x')
    return str(tmp_path)


def test_find_products_raises_on_missing_bg(tmp_path):
    from jwst_gc_pipeline.photometry.reference_fields import evaluate as EV
    rdir = _reffield_run(tmp_path, with_bg=False)
    with pytest.raises(FileNotFoundError, match='keep-intermediate'):
        EV.find_products(rdir, 'F212N', phase='m4')
    with pytest.warns(UserWarning, match='smoothed-bg'):
        prod = EV.find_products(rdir, 'F212N', phase='m4', allow_missing_bg=True)
    assert prod['smoothed_bg'] is None


def test_find_products_returns_present_bg(tmp_path):
    from jwst_gc_pipeline.photometry.reference_fields import evaluate as EV
    rdir = _reffield_run(tmp_path, with_bg=True)
    assert EV.find_products(rdir, 'F212N', phase='m4')['smoothed_bg'].endswith(
        '_smoothed_bg_i2d.fits')
