"""The m2 correction floor is a per-FIELD property, not an operator's memory.

``ASTROM_M2_CORRECTION_FLOOR_MAS`` defaulted to 0, so a field whose own
per-exposure scatter reaches the 2 mas checkpoint tolerance died at m12 on every
run submitted without the env var -- taking m3-m7 with it as
DependencyNeverSatisfied.  brick paid it twice (jobs 37614271 and, on
2026-08-22, 39884095: F115W consensus scatter 2.274 mas, all 96 exposure offsets
inside 0.19-2.65 mas with no outlier, 21 flagged for exceeding 2 mas, applied
correction ~1 mas).

The floor now comes from the field, and the env var still wins when set.
"""
import os

import pytest

from jwst_gc_pipeline.photometry.m2_correction_floors import (
    FLOOR_ENV, PER_FIELD_FLOOR_MAS, m2_correction_floor)


# --------------------------------------------------------------------------
# resolution order
# --------------------------------------------------------------------------

def test_a_registered_field_gets_its_own_floor():
    assert m2_correction_floor('brick', env={}) == (4.0, 'per-field')
    assert m2_correction_floor('cloudc', env={}) == (8.0, 'per-field')


def test_an_unregistered_field_keeps_the_strict_default():
    """Absent means strict, never lenient: a field nobody has measured must not
    inherit somebody else's tolerance."""
    for target in ('sgra', 'gc2211_o023', 'w51', 'arches', 'quintuplet'):
        assert m2_correction_floor(target, env={}) == (0.0, 'default'), target


def test_no_target_is_the_strict_default_too():
    """A caller that cannot say which field it is gets the safe direction."""
    assert m2_correction_floor(None, env={}) == (0.0, 'default')


def test_an_explicit_env_var_wins_over_the_field():
    """Overriding deliberately is a different act from a default nobody
    remembered, and it stays possible in both directions."""
    assert m2_correction_floor(
        'brick', env={FLOOR_ENV: '12'}) == (12.0, 'env')
    # including DOWN, which is what makes it an override rather than a maximum
    assert m2_correction_floor(
        'cloudc', env={FLOOR_ENV: '0'}) == (0.0, 'env')
    # and on a field with no entry
    assert m2_correction_floor(
        'sgra', env={FLOOR_ENV: '5.5'}) == (5.5, 'env')


def test_an_empty_env_var_is_not_an_override():
    """`export VAR=` in a submit script sets it to '' -- that is an unset
    intent, not a request for a 0 mas floor, and reading it as 0 would silently
    restore the exact bug this replaces."""
    assert m2_correction_floor('brick', env={FLOOR_ENV: ''}) == (4.0, 'per-field')


def test_it_reads_the_real_environment_by_default(monkeypatch):
    monkeypatch.delenv(FLOOR_ENV, raising=False)
    assert m2_correction_floor('brick') == (4.0, 'per-field')
    monkeypatch.setenv(FLOOR_ENV, '9')
    assert m2_correction_floor('brick') == (9.0, 'env')


# --------------------------------------------------------------------------
# the table itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize('target,floor', sorted(PER_FIELD_FLOOR_MAS.items()))
def test_every_entry_exceeds_the_checkpoint_tolerance(target, floor):
    """A floor at or below the 2 mas tolerance changes nothing, so an entry
    that small is a mistake rather than a decision."""
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        EXPOSURE_CONSENSUS_TOL_MAS)
    assert floor > EXPOSURE_CONSENSUS_TOL_MAS, (
        f'{target} floor {floor} does not clear the {EXPOSURE_CONSENSUS_TOL_MAS} '
        'mas tolerance it exists to sit above')


@pytest.mark.parametrize('target,floor', sorted(PER_FIELD_FLOOR_MAS.items()))
def test_no_entry_is_large_enough_to_hide_a_real_shift(target, floor):
    """The floor covers SIAF/DVA-class per-detector scatter (single-digit mas).
    An entry in the tens would swallow the misregistrations the checkpoint
    exists to catch -- the 90 mas brick-1182 F200W seam, say."""
    assert floor <= 10.0, f'{target} floor {floor} is large enough to hide a seam'


def test_the_registered_fields_are_real_fields():
    """A typo here fails open: the field silently keeps the 0 mas default and
    dies at m12 exactly as before, with the table looking correct."""
    import yaml

    here = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    with open(os.path.join(here, 'fields.yaml')) as fh:
        doc = yaml.safe_load(fh)
    known = set(doc.get('fields', doc))
    unknown = sorted(set(PER_FIELD_FLOOR_MAS) - known)
    assert not unknown, f'not fields in fields.yaml: {unknown}'


# --------------------------------------------------------------------------
# wiring
# --------------------------------------------------------------------------

def test_cataloging_resolves_the_floor_per_field_not_from_the_env():
    """Pins the call site.  The unit tests above pass even if `cataloging` still
    reads os.environ directly, which is the whole defect."""
    import inspect

    from jwst_gc_pipeline.photometry import cataloging

    src = inspect.getsource(cataloging._run_astrometry_stage_checkpoint)
    assert 'm2_correction_floor(' in src, (
        'the checkpoint no longer resolves the floor per field')
    assert "os.environ.get('ASTROM_M2_CORRECTION_FLOOR_MAS'" not in src, (
        'the checkpoint still reads the env var directly, so an unset var is '
        'still a 0 mas floor')


def test_the_record_carries_the_floor_and_its_source():
    """`correction_floor_mas` was already recorded; without the SOURCE, a pass
    at the field's standing floor is indistinguishable from one where somebody
    raised it by hand for this run."""
    import inspect

    from jwst_gc_pipeline.photometry import astrometry_checkpoint as ac

    src = inspect.getsource(ac.run_visit_checkpoint)
    assert 'correction_floor_mas=' in src
    assert 'correction_floor_source=' in src
    assert 'target' in inspect.signature(ac.run_visit_checkpoint).parameters


@pytest.mark.parametrize('basepath,expect', [
    ('/orange/adamginsburg/jwst/brick', 'brick'),
    ('/orange/adamginsburg/jwst/brick/', 'brick'),
    ('/orange/adamginsburg/jwst/gc2211_o023', 'gc2211_o023'),
    ('/orange/adamginsburg/jwst/cloudef_controlfield//', 'cloudef_controlfield'),
    ('', None),
    (None, None),
])
def test_basepath_fallback_recovers_the_field(basepath, expect):
    """For callers that pass no ``target`` (the CLI), the field is the leaf of
    the basepath -- and an unusable basepath lands on the strict default."""
    from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
        _target_from_basepath)
    assert _target_from_basepath(basepath) == expect
