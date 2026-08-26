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
import warnings

import pytest

from jwst_gc_pipeline.photometry.m2_correction_floors import (
    FLOOR_ENV, PER_FIELD_FLOOR_MAS, m2_correction_floor)


# --------------------------------------------------------------------------
# resolution order
# --------------------------------------------------------------------------

def test_a_registered_field_gets_its_own_floor():
    assert m2_correction_floor('brick', env={}) == (4.0, 'per-field')
    assert m2_correction_floor('cloudc', env={}) == (8.0, 'per-field')
    assert m2_correction_floor('sgra', env={}) == (4.0, 'per-field')
    assert m2_correction_floor('w51', env={}) == (6.0, 'per-field')


def test_an_unregistered_field_keeps_the_strict_default():
    """Absent means strict, never lenient: a field nobody has measured must not
    inherit somebody else's tolerance."""
    for target in ('gc2211_o023', 'quintuplet', 'sgrb2_nonexistent'):
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


def test_sgra_has_a_floor_matching_its_own_scatter():
    """sgra was in the unregistered list until 2026-08-22, when its m12 finalize
    stopped twice (jobs 39933168, 39972201) on a single 2.08 mas correction
    against a consensus scatter of 1.84 mas over 96 measurements -- credible
    (contrast 976, rank 77/96), but the same size as the field's own noise, so
    not a displacement the module-locked table should express.

    4.0 matches brick, whose scatter is 2.27 mas.  The entry has to clear the
    2 mas checkpoint tolerance to change anything, and stay far below the ~90 mas
    seams the checkpoint exists to catch; the table-wide bounds test covers both.
    """
    floor, source = m2_correction_floor('sgra', env={})
    assert source == 'per-field'
    assert floor == 4.0


def test_w51_floor_covers_its_systematic_class_but_not_f444w():
    """Set from ALL of w51's filters, not the first one that tripped.

    4.0 came from F140M alone (max 3.64 mas) and F162M then exceeded it at 5.42,
    because m2 stops at the FIRST filter with an actionable correction.  A
    WARN_ONLY measurement pass measured all eleven: eight are clean (no
    corrections, residuals < 2 mas), the per-detector systematic class tops out
    at 5.42, and F444W sits apart at 9.58 with 15 of its 16 exposures displaced.

    6.0 covers the systematic class and deliberately leaves F444W stopping the
    run -- it is not a module split (antisymmetry detected=False, A-B = -1.6 mas)
    and is not understood.  A floor of 10 would have hidden it.
    """
    floor, source = m2_correction_floor('w51', env={})
    assert source == 'per-field'
    assert floor == 6.0
    assert floor > 5.42, 'must cover the per-detector systematic class (F162M)'
    assert floor < 8.74, 'must NOT swallow F444W (p90 8.74, max 9.58)'


def test_arches_has_a_floor_matching_its_measured_corrections():
    """arches ran at 4.0 from the env var on every record it has ever written,
    so it depended on the operator memory this table exists to replace.

    Measured from its own m2 records (2026-08-23): consensus scatter 1.18-1.53
    mas, 107 corrections across F212N and F323N spanning 2.00-4.22 mas, all but
    one below 4.  The two current records read ``passed: true`` only because
    ``correction_floor_source`` is ``env``; at the 0.0 default the same
    measurements are 51 actionable corrections and an m12 stop.
    """
    floor, source = m2_correction_floor('arches', env={})
    assert source == 'per-field'
    assert floor == 4.0
    assert floor > 3.93, 'must cover the largest current-record correction'
    # and NOT chosen to swallow the one 4.22 mas outlier of 2026-08-01
    assert floor < 4.22


def test_only_the_gc2211_pointings_that_measure_scatter_have_floors():
    """o046 and o050 measure the per-exposure scatter class and nothing else
    (58 corrections, 2.00-3.52 mas, consensus scatter 0.68-6.91), and o028 now
    joins them at a wider 6.0.

    o028 previously sat on the strict default because its record showed a
    coherent ~200 mas exposure-2 shift.  That shift was never on the sky -- m2
    wrote it into the offsets table on 2026-07-23, and F277W, simultaneous with
    SW and never corrected, put exposure 2 within 2.9 mas of exposure 3
    throughout.  After the revert and a regeneration from _cal, F150W measures
    **zero** corrections at 0.86 mas scatter (it had 192, median 28.47) and
    F277W measures 20 spanning 2.19-4.33 on 2.60 mas scatter -- the scatter
    class.

    o023 and o049 still measure something else: 118 mas of trailed-exposure
    displacement (excluded outright in #493) and three lowest-contrast cells at
    5.4-22.4 mas (#484).  They stay on the strict default rather than inheriting
    a sibling's tolerance.
    """
    for target in ('gc2211_o046', 'gc2211_o050'):
        assert m2_correction_floor(target, env={}) == (4.0, 'per-field'), target
    assert m2_correction_floor('gc2211_o028', env={}) == (6.0, 'per-field')
    for target in ('gc2211_o023', 'gc2211_o049'):
        assert m2_correction_floor(target, env={}) == (0.0, 'default'), target


def test_the_o028_floor_covers_its_measured_scatter_with_margin():
    """4.33 mas is the largest correction its post-revert record holds.  5.0
    would clear that by 0.67 mas, inside the run-to-run variation these
    distributions show; it must also stay under the seam-hiding cap."""
    floor, _ = m2_correction_floor('gc2211_o028', env={})
    assert floor > 4.33, 'must cover the largest measured correction'
    assert floor >= 5.5, 'and with more margin than 5.0 would give'
    assert floor <= 10.0


def test_quintuplet_stays_unregistered_until_it_measures_something():
    """Its records carry ``correction_floor_mas: 4.0`` from the env var like
    arches's do, but all four (2026-08-01 and 2026-08-15, F212N and F323N) hold
    ZERO corrections at a consensus scatter of 1.16-1.32 mas.  An entry there
    would record the operator's setting rather than the field's scatter, which
    is the distinction this table is built on.
    """
    assert m2_correction_floor('quintuplet', env={}) == (0.0, 'default')


# --------------------------------------------------------------------------
# an env-var floor on an unregistered field is a registration that is owed
# --------------------------------------------------------------------------

def test_env_floor_on_an_unregistered_field_warns():
    """The state every retroactively-added entry was in first.

    sgra (#494), w51 (#508), arches (#512) and gc2211_o028 (#533) were each
    added AFTER a chain stopped, and each had been running with
    ``ASTROM_M2_CORRECTION_FLOOR_MAS`` set by hand for weeks beforehand.  The
    records said so in ``correction_floor_source`` and nothing surfaced it.
    The warning fires when the override is USED, which is when the evidence
    exists and the operator is there to act on it.
    """
    from jwst_gc_pipeline.photometry.m2_correction_floors import (
        UnregisteredM2FloorWarning)
    with pytest.warns(UnregisteredM2FloorWarning, match='quintuplet'):
        floor, source = m2_correction_floor('quintuplet',
                                            env={FLOOR_ENV: '4.0'})
    # and the override still wins -- the warning changes nothing about the run
    assert (floor, source) == (4.0, 'env')


def test_env_floor_on_a_registered_field_is_silent():
    """A registered field being overridden is a deliberate act, not a gap."""
    from jwst_gc_pipeline.photometry.m2_correction_floors import (
        UnregisteredM2FloorWarning)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UnregisteredM2FloorWarning)
        assert m2_correction_floor('brick', env={FLOOR_ENV: '12'}) == (12.0, 'env')


def test_a_zero_env_floor_does_not_warn():
    """``ASTROM_M2_CORRECTION_FLOOR_MAS=0`` is somebody asking for the STRICT
    behaviour on purpose.  No entry is owed for that, and warning would train
    the reader to ignore the message."""
    from jwst_gc_pipeline.photometry.m2_correction_floors import (
        UnregisteredM2FloorWarning)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UnregisteredM2FloorWarning)
        assert m2_correction_floor('gc2211_o023',
                                   env={FLOOR_ENV: '0'}) == (0.0, 'env')


def test_the_unregistered_warning_names_the_table_to_edit():
    from jwst_gc_pipeline.photometry.m2_correction_floors import (
        UnregisteredM2FloorWarning)
    with pytest.warns(UnregisteredM2FloorWarning) as rec:
        m2_correction_floor('some_new_field', env={FLOOR_ENV: '5'})
    msg = str(rec[0].message)
    assert 'PER_FIELD_FLOOR_MAS' in msg
    assert 'm2_correction_floors.py' in msg
