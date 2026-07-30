"""The SLURM array index -> (program, filter) map, and the merge error policy.

`merge_catalogs.main` used to enumerate array indices with six nested loops over
(module, desat, bgsub, epsf, fitpsf, blur).  Those five flags had no CLI switch,
so every index above the first hit products the pipeline stopped writing in 2023
and failed into an except-and-print.  The map is now one flat list; these tests
pin its order (an index shift would send an array task at the wrong filter) and
the three-state error policy.
"""
import pytest

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


def test_missing_ok_swallows_only_missing_inputs():
    MC._run_merge(_boom(ValueError('had no matches')), 'x', 'missing-ok')
    with pytest.raises(ValueError):
        MC._run_merge(_boom(ValueError('column mismatch')), 'x', 'missing-ok')


def _raised_message_text():
    """Every string literal that appears inside a `raise` in merge_catalogs."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(MC))
    text = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise):
            for sub in ast.walk(node):
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    text.append(sub.value)
    return text


@pytest.mark.parametrize('message', MC._MISSING_INPUT_MESSAGES)
def test_each_missing_input_phrase_is_one_a_merge_actually_raises(message):
    # A phrase no merge function raises makes 'missing-ok' dead code, which is
    # how the daophot no-input case stayed fatal despite the policy saying
    # otherwise.  Match against `raise` statements only -- a phrase that appears
    # solely in a comment or a docstring does not count.
    assert any(message in raised for raised in _raised_message_text()), (
        f'no `raise` in merge_catalogs contains {message!r}')
    MC._run_merge(_boom(ValueError(f'... {message} ...')), 'x', 'missing-ok')


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
