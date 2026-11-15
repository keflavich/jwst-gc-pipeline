"""gc-treasury F480M reduces without destreak; its F212N still does.

``destreak_policy.NO_DESTREAK_FILTERS`` turns destreaking off for one filter of
a field that otherwise destreaks.  The answer has to reach three places the same
way, or stage 2 photometers frames that stage 1 did not write:

* ``destreaks`` -- stage 1's own gate;
* ``crf_suffix`` -- the ``destreak_`` / ``align_`` token in the frame name;
* ``run_pipeline._suffix_overrides`` -- the ``--each-suffix-overrides`` that lets
  the two filters of one observation read different lineages (sickle's case).

The exception is keyed on the field AND the filter: nothing else changes lineage.
"""
import pytest

from jwst_gc_pipeline import run_pipeline as rp
from jwst_gc_pipeline.reduction import destreak_policy
from jwst_gc_pipeline.reduction.destreak_policy import (
    NO_DESTREAK_FILTERS, crf_suffix, destreaks, suffixes_by_filter)


def test_f480m_is_the_only_gc_treasury_filter_that_skips_destreak():
    assert destreaks('gc-treasury', 'F480M', instrument='nircam') is False
    assert destreaks('gc-treasury', 'F212N', instrument='nircam') is True
    assert NO_DESTREAK_FILTERS == {'gc-treasury': ('F480M',)}


def test_the_filter_name_is_case_insensitive():
    """The registry spells filters lower-case (``f480m``); products upper-case."""
    assert destreaks('gc-treasury', 'f480m', instrument='nircam') is False
    assert destreaks('gc-treasury', 'f212n', instrument='nircam') is True


def test_the_lineage_token_follows_the_policy():
    assert crf_suffix('gc-treasury', 'F480M', '102',
                      instrument='nircam') == 'align_o102_crf'
    assert crf_suffix('gc-treasury', 'F212N', '102',
                      instrument='nircam') == 'destreak_o102_crf'


def test_a_run_that_asked_for_no_destreak_still_gets_none():
    """The policy can only turn destreaking off, never on."""
    assert destreaks('gc-treasury', 'F212N', requested=False,
                     instrument='nircam') is False
    assert destreaks('gc-treasury', 'F480M', requested=False,
                     instrument='nircam') is False


def test_no_other_field_or_filter_changes_lineage():
    """brick keeps destreaking F480M-adjacent filters, and F480M elsewhere is
    decided by that field's own policy, not by gc-treasury's."""
    assert destreaks('brick', 'F410M', instrument='nircam') is True
    assert destreaks('brick', 'F480M', instrument='nircam') is True
    assert destreaks('cloudc', 'F480M', instrument='nircam') is True
    assert destreaks('w51', 'F480M', instrument='nircam') is False
    # sickle keeps its own per-filter rule
    assert destreaks('sickle', 'F212N', instrument='nircam') is True
    assert destreaks('sickle', 'F480M', instrument='nircam') is False


def test_the_exception_does_not_reach_the_other_instruments():
    """MIRI parallels of the same observation write the untokened name."""
    assert destreaks('gc-treasury', 'F770W') is False
    assert crf_suffix('gc-treasury', 'F770W', '102',
                      instrument='miri') == 'o102_crf'


def test_the_two_filters_of_an_observation_take_different_suffixes():
    per_filter = suffixes_by_filter('gc-treasury', ['f212n', 'f480m'], '102',
                                    instrument='nircam')
    assert per_filter == {'F212N': 'destreak_o102_crf',
                          'F480M': 'align_o102_crf'}


def test_the_catalog_stage_is_told_about_the_split():
    """``each_suffix`` is the first filter's; the other side of the split rides
    on ``EACH_SUFFIX_OVERRIDES``, or its glob matches no frame at m1."""
    plan = {'each_suffix': crf_suffix('gc-treasury', 'F212N', '102',
                                      instrument='nircam'),
            'suffix_by_filter': suffixes_by_filter(
                'gc-treasury', ['F212N', 'F480M'], '102', instrument='nircam')}
    assert plan['each_suffix'] == 'destreak_o102_crf'
    assert rp._suffix_overrides(plan) == 'F480M:align_o102_crf'


def test_the_policy_module_names_what_it_exports():
    assert 'NO_DESTREAK_FILTERS' in vars(destreak_policy)


@pytest.mark.parametrize('obs', ['102', '139'])
def test_the_registered_plan_carries_the_split(obs):
    """End to end through the registry, for observations that have a reference
    catalog registered (the plan refuses the ones that do not)."""
    plan = rp.resolve('10678', obs)
    assert plan['suffix_by_filter'] == {'F212N': f'destreak_o{obs}_crf',
                                        'F480M': f'align_o{obs}_crf'}
    assert rp._suffix_overrides(plan) == f'F480M:align_o{obs}_crf'
