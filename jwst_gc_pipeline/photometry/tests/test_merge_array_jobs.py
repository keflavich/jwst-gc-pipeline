"""The SLURM array index -> (program, filter) map, and the merge error policy.

`merge_catalogs.main` used to enumerate array indices with six nested loops over
(module, desat, bgsub, epsf, fitpsf, blur).  Those five flags had no CLI switch,
so every index above the first hit products the pipeline stopped writing in 2023
and failed into an except-and-print.  The map is now one flat list; these tests
pin its order (an index shift would send an array task at the wrong filter) and
the three-state error policy.
"""
import pytest
from astropy.table import Table

from jwst_gc_pipeline.photometry import merge_catalogs as MC


def test_brick_job_order_is_pinned():
    # Written out, not recomputed from obs_filters: reordering that dict must
    # fail this test, because it silently moves every array task.
    assert MC.individual_frame_merge_jobs('brick') == [
        ('2221', 'f410m'), ('2221', 'f212n'), ('2221', 'f466n'),
        ('2221', 'f405n'), ('2221', 'f187n'), ('2221', 'f182m'),
        ('2221', 'f2550w'),
        ('1182', 'f444w'), ('1182', 'f356w'), ('1182', 'f200w'),
        ('1182', 'f115w'),
    ]


def test_job_list_follows_the_instrument_override(monkeypatch):
    # main() reads _obs_filters_for(), which swaps in the NIRISS filter set.
    # A job list built from the raw obs_filters dict would ignore that, and a
    # NIRISS array job would land on the wrong filter.
    sgrc_nircam = MC.individual_frame_merge_jobs('sgrc')
    monkeypatch.setenv('GC_INSTRUMENT_OVERRIDE', 'niriss')
    assert MC.individual_frame_merge_jobs('sgrc') == [
        ('4147', 'f158m'), ('4147', 'f200w'),
        ('4147', 'f356w'), ('4147', 'f480m')]
    assert MC.individual_frame_merge_jobs('sgrc') != sgrc_nircam


def test_every_target_has_a_job_list():
    for target in MC.obs_filters:
        jobs = MC.individual_frame_merge_jobs(target)
        assert jobs, f'{target} has no merge jobs'
        assert len(set(jobs)) == len(jobs), f'{target} has duplicate jobs'


def test_method_suffixes_are_the_per_frame_filename_tokens():
    assert MC.INDIV_MERGE_SUFFIX == {'crowdsource': '_nsky0',
                                     'dao': '_basic',
                                     'daoiterative': '_iterative',
                                     'iterative': '_iterative'}


def _boom(exc):
    def merge(**kwargs):
        raise exc
    return merge


def test_raise_policy_propagates():
    with pytest.raises(ValueError):
        MC._run_merge(_boom(ValueError('No tables found')), 'x', 'raise')


def test_daophot_basic_no_input_is_fatal(tmp_path, monkeypatch):
    """A merge with zero daophot-basic inputs must fail, not exit 0.

    This is what the deleted 'missing-ok' policy got wrong: it swallowed
    merge_daophot's real no-input message, so a run with no inputs finished
    successfully having built no cross-band product.  Asserted against the
    message merge_daophot actually raises, not against the source text.
    """
    # merge_daophot queries the SVO filter service before it checks its inputs;
    # stub it so a cold cache cannot masquerade as the failure under test.
    monkeypatch.setattr(MC.SvoFps, 'get_filter_list',
                        lambda *a, **k: Table({'filterID': ['JWST/NIRCam.F410M'],
                                               'WavelengthEff': [40971.0]}))
    with pytest.raises(ValueError) as excinfo:
        MC.merge_daophot(daophot_type='basic', module='merged', target='brick',
                         basepath=str(tmp_path), indivexp=True)
    assert 'catalogs found' in str(excinfo.value) or 'no matches' in str(
        excinfo.value), f'unexpected failure mode: {excinfo.value}'

    policy = dict(MC.DAOPHOT_MERGES)['basic']
    assert policy == 'raise', 'daophot basic must not swallow a missing input'


def test_only_two_policies_exist():
    # A third state that nothing legitimately used is how 'missing-ok' became
    # both dead code and, once "fixed", a silent skip of a required product.
    policies = {p for _, p in MC.CROWDSOURCE_MERGES + MC.DAOPHOT_MERGES}
    assert policies <= {'raise', 'skip'}
    with pytest.raises(ValueError):
        MC._run_merge(_boom(ValueError('x')), 'x', 'missing-ok')


def test_skip_policy_swallows_expected_failures():
    MC._run_merge(_boom(NotImplementedError()), 'x', 'skip')
    MC._run_merge(_boom(FileNotFoundError('nope')), 'x', 'skip')


def test_skip_policy_still_reports_a_programming_error():
    # 'skip' must not hide a bug in our own code.
    with pytest.raises(AttributeError):
        MC._run_merge(_boom(AttributeError('typo')), 'x', 'skip')


def test_the_required_merge_is_the_current_science_product():
    strict = [suffix for suffix, policy in MC.CROWDSOURCE_MERGES
              if policy == 'raise']
    assert strict == ['_nsky0']


# --- registration gate scope ---------------------------------------------
# The gate scans the whole field, so it belongs to a run that merged the whole
# field.  Before the flattening it was the reverse by accident: the old loop
# returned early on a plain run (never reaching the gate) while every array task
# fell through to it, so N tasks each scanned a half-built field.

def _run_main(monkeypatch, argv, env, gate_calls):
    import sys
    monkeypatch.setattr(sys, 'argv', argv)
    for k, v in env.items():
        if v is None:
            monkeypatch.delenv(k, raising=False)
        else:
            monkeypatch.setenv(k, v)
    monkeypatch.setattr(MC, 'merge_crowdsource', lambda **kw: None)
    monkeypatch.setattr(MC, 'merge_daophot', lambda **kw: None)
    monkeypatch.setattr(MC, 'merge_individual_frames', lambda **kw: None)

    import jwst_gc_pipeline.photometry.registration_gate as RG
    monkeypatch.setattr(RG, 'run_registration_gate',
                        lambda *a, **k: gate_calls.append(a) or {'returncode': 0})
    MC.main()


def test_gate_runs_on_a_whole_field_run(monkeypatch):
    calls = []
    _run_main(monkeypatch,
              ['merge_catalogs', '--target', 'brick'],
              {'RUN_REGISTRATION_GATE': '1', 'SLURM_ARRAY_TASK_ID': None},
              calls)
    assert len(calls) == 1


def test_gate_does_not_run_per_array_task(monkeypatch, capsys):
    calls = []
    _run_main(monkeypatch,
              ['merge_catalogs', '--target', 'brick', '--merge-singlefields'],
              {'RUN_REGISTRATION_GATE': '1', 'SLURM_ARRAY_TASK_ID': '3'},
              calls)
    assert calls == []
    assert 'registration gate skipped' in capsys.readouterr().out


def test_gate_stays_off_unless_asked(monkeypatch):
    calls = []
    _run_main(monkeypatch,
              ['merge_catalogs', '--target', 'brick'],
              {'RUN_REGISTRATION_GATE': None, 'SLURM_ARRAY_TASK_ID': None},
              calls)
    assert calls == []
