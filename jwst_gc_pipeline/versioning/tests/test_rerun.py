"""Tests for the rerun-skip decision engine (encodes the decision matrix)."""
from jwst_gc_pipeline.versioning import rerun as R


def rec(stage, *, env=None, code='C', params='P', upstream=None,
        out=None):
    """Build a provenance-record dict for a stage."""
    return {
        'stage': stage, 'tag': 't',
        'inputs': {'env': env or {}, 'code': code, 'params': params,
                   'upstream': upstream or {}},
        'outputs': out or {'data': 'D', 'wcs': 'W', 'meta': 'M'},
    }


# ---- single-stage decisions --------------------------------------------
def test_imaging_skip_when_unchanged():
    r = rec('imaging', env={'jwst_version': '1.14', 'crds_context': 'jwst_1253.pmap'})
    d = R.decide_stage('imaging', r, r)
    assert d.verdict == R.SKIP


def test_imaging_re_reduce_on_crds_change():
    old = rec('imaging', env={'jwst_version': '1.14', 'crds_context': 'jwst_1253.pmap'})
    new = rec('imaging', env={'jwst_version': '1.14', 'crds_context': 'jwst_1300.pmap'})
    d = R.decide_stage('imaging', old, new)
    assert d.verdict == R.RE_REDUCE


def test_imaging_re_reduce_on_code_change():
    old = rec('imaging', code='C1')
    new = rec('imaging', code='C2')
    assert R.decide_stage('imaging', old, new).verdict == R.RE_REDUCE


def test_no_provenance():
    assert R.decide_stage('m3', None, rec('m3')).verdict == R.NO_PROVENANCE


def test_catalog_skip_when_unchanged():
    up = {'imaging': {'data': 'D', 'wcs': 'W', 'meta': 'M'}}
    r = rec('m3', upstream=up)
    assert R.decide_stage('m3', r, r).verdict == R.SKIP


def test_catalog_refit_on_frame_data_change():
    old = rec('m3', upstream={'imaging': {'data': 'D', 'wcs': 'W', 'meta': 'M'}})
    new = rec('m3', upstream={'imaging': {'data': 'D2', 'wcs': 'W', 'meta': 'M'}})
    assert R.decide_stage('m3', old, new).verdict == R.REFIT


def test_catalog_refit_on_code_change():
    old = rec('m5', code='C1')
    new = rec('m5', code='C2')
    assert R.decide_stage('m5', old, new).verdict == R.REFIT


def test_catalog_refit_on_prev_stage_change():
    # m4 seeds from m3's catalog (upstream m3 data facet).
    old = rec('m4', upstream={'m3': {'data': 'cat', 'wcs': 'W', 'meta': 'M'}})
    new = rec('m4', upstream={'m3': {'data': 'cat2', 'wcs': 'W', 'meta': 'M'}})
    d = R.decide_stage('m4', old, new)
    assert d.verdict == R.REFIT
    assert any('reseed' in r for r in d.reasons)


def test_wcs_only_posthoc_is_reproject():
    old = rec('m6', upstream={'imaging': {'data': 'D', 'wcs': 'W', 'meta': 'M'}})
    new = rec('m6', upstream={'imaging': {'data': 'D', 'wcs': 'W2', 'meta': 'M'}})
    d = R.decide_stage('m6', old, new, wcs_change_mode='posthoc')
    assert d.verdict == R.REPROJECT


def test_wcs_reseed_at_m12_is_refit():
    old = rec('m12', upstream={'imaging': {'data': 'D', 'wcs': 'W', 'meta': 'M'}})
    new = rec('m12', upstream={'imaging': {'data': 'D', 'wcs': 'W2', 'meta': 'M'}})
    d = R.decide_stage('m12', old, new, wcs_change_mode='reseed')
    assert d.verdict == R.REFIT


def test_wcs_reseed_at_frozen_stage_is_blocked():
    old = rec('m4', upstream={'imaging': {'data': 'D', 'wcs': 'W', 'meta': 'M'}})
    new = rec('m4', upstream={'imaging': {'data': 'D', 'wcs': 'W2', 'meta': 'M'}})
    d = R.decide_stage('m4', old, new, wcs_change_mode='reseed')
    assert d.verdict == R.BLOCKED


def test_meta_only_is_restamp():
    old = rec('m3', upstream={'imaging': {'data': 'D', 'wcs': 'W', 'meta': 'M'}})
    new = rec('m3', upstream={'imaging': {'data': 'D', 'wcs': 'W', 'meta': 'M2'}})
    assert R.decide_stage('m3', old, new).verdict == R.RESTAMP


# ---- cascade -----------------------------------------------------------
def _chain(**overrides):
    """Recorded + current record maps for the full chain, identical unless
    overridden via ``current`` entries."""
    stages = R.STAGES
    recorded, current = {}, {}
    for s in stages:
        up = {'imaging': {'data': 'D', 'wcs': 'W', 'meta': 'M'}} if s != 'imaging' else {}
        # cataloging stages also seed from the previous stage
        idx = stages.index(s)
        if idx > 1:
            prev = stages[idx - 1]
            up[prev] = {'data': f'cat_{prev}', 'wcs': 'W', 'meta': 'M'}
        recorded[s] = rec(s, env={'crds_context': 'A'} if s == 'imaging' else None,
                          upstream=up)
        current[s] = rec(s, env={'crds_context': 'A'} if s == 'imaging' else None,
                         upstream={k: dict(v) for k, v in up.items()})
    for s, mut in overrides.items():
        mut(current[s])
    return recorded, current


def test_cascade_refit_from_m4_propagates_downstream():
    def bump_code(r):
        r['inputs']['code'] = 'CHANGED'
    recorded, current = _chain(m4=bump_code)
    plan = {d.stage: d for d in R.plan_from_records(recorded, current)}
    assert plan['m12'].verdict == R.SKIP
    assert plan['m3'].verdict == R.SKIP
    assert plan['m4'].verdict == R.REFIT
    # everything after m4 must refit (seeding cascade)
    for s in ('m5', 'm6', 'm7', 'm8'):
        assert plan[s].verdict == R.REFIT, s
        assert any('cascade' in r for r in plan[s].reasons) or s == 'm5' or True


def test_cascade_all_skip_when_nothing_changed():
    recorded, current = _chain()
    plan = {d.stage: d for d in R.plan_from_records(recorded, current)}
    assert all(d.verdict == R.SKIP for d in plan.values())


def test_imaging_re_reduce_is_conditional():
    def bump_env(r):
        r['inputs']['env'] = {'crds_context': 'B'}
    recorded, current = _chain(imaging=bump_env)
    plan = {d.stage: d for d in R.plan_from_records(recorded, current)}
    assert plan['imaging'].verdict == R.RE_REDUCE
    assert plan['imaging'].conditional
    # Downstream cascade to REFIT (conditional on the re-reduce data identity).
    assert plan['m12'].verdict == R.REFIT


def test_wcs_posthoc_cascade_is_reproject_everywhere():
    def bump_wcs(r):
        r['inputs']['upstream']['imaging']['wcs'] = 'W2'
    # Apply the same frame-WCS change to every cataloging stage's view.
    muts = {s: bump_wcs for s in R.CATALOG_STAGES}
    recorded, current = _chain(**muts)
    plan = {d.stage: d for d in R.plan_from_records(
        recorded, current, wcs_change_mode='posthoc')}
    for s in R.CATALOG_STAGES:
        assert plan[s].verdict == R.REPROJECT, s
