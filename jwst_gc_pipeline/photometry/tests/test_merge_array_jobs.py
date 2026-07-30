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


def test_job_order_is_program_then_filter():
    jobs = MC.individual_frame_merge_jobs('brick')
    expected = [(progid, filt)
                for progid, filts in MC.obs_filters['brick'].items()
                for filt in filts]
    assert jobs == expected


def test_every_target_has_a_job_list():
    for target in MC.obs_filters:
        jobs = MC.individual_frame_merge_jobs(target)
        assert jobs, f'{target} has no merge jobs'
        assert len(set(jobs)) == len(jobs), f'{target} has duplicate jobs'


def test_each_method_maps_to_a_suffix():
    # main() indexes INDIV_MERGE_SUFFIX with the --indiv-merge-methods values.
    for method in 'dao', 'crowdsource', 'daoiterative':
        assert MC.INDIV_MERGE_SUFFIX[method].startswith('_')


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
