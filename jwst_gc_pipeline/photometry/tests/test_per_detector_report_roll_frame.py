"""The static-term test must not be defeated by the telescope's roll.

``per_detector_offset_report.py`` is the standing justification for keeping the
offsets table module-keyed (#340).  Its verdict rests on "each detector's
apparent offset changes sign between fields", measured in ON-SKY dRA/dDec.

A detector's placement error is fixed in the INSTRUMENT frame, and this
archive's fields sit in two roll clusters ~180 deg apart (PA_V3 ~89 and ~275).
A perfectly static instrument-frame term therefore reverses sign on sky between
those clusters -- so on-sky sign reversal is what a REAL per-detector term looks
like here, and the on-sky test alone cannot reject it.

These tests pin the second reading: de-rotate by each band's own PA_V3 and run
the same estimator again.  ``test_a_term_static_in_the_instrument_frame_...``
is the one that fails if ``derotate`` stops rotating.
"""
import importlib.util
import math
import os

import numpy as np
import pytest

_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                     'scripts', 'analysis', 'per_detector_offset_report.py')


def _mod():
    spec = importlib.util.spec_from_file_location('pdor', os.path.abspath(_PATH))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# A per-detector term in the instrument frame, summing to zero over the module
# so that subtracting the group mean leaves it intact rather than removing it.
_INSTRUMENT_TERM = {'nrca1': (0.0, +1.2), 'nrca2': (0.0, +0.4),
                    'nrca3': (0.0, -0.4), 'nrca4': (0.0, -1.2)}

# Two clusters ~180 deg apart, as the archive has: brick/sgrb2/sgrc/arches near
# 89, wd1/cloudef/m4 near 275-290.
_ROLLS = {'fieldA': 89.0, 'fieldB': 275.0, 'fieldC': 91.0, 'fieldD': 284.0,
          'fieldE': 88.0, 'fieldF': 272.0}


def _sky_rows(term, rolls, nexp=60, noise=0.25, seed=3):
    """Rows as ``collect`` returns them, carrying ``term`` in the INSTRUMENT
    frame, written out on sky at each field's roll."""
    rng = np.random.default_rng(seed)
    rows = []
    for field, pa in rolls.items():
        th = math.radians(pa)
        c, s = math.cos(th), math.sin(th)
        for e in range(nexp):
            for det, (u, v) in term.items():
                # inverse of `derotate`: instrument -> on sky
                dra = c * u + s * v
                ddec = -s * u + c * v
                rows.append(dict(
                    field=field, date='2026-08-25T00:00:00', filt='F212N',
                    visit='1', exp=e, vgroup='01201', module=det[:4], det=det,
                    dra=dra + rng.normal(0, noise),
                    ddec=ddec + rng.normal(0, noise)))
    return rows


def _roll_lookup(rolls):
    return {(f, 'F212N'): pa for f, pa in rolls.items()}


def test_a_term_static_in_the_instrument_frame_is_missed_on_sky_and_found_after_derotation():
    """The regression this file exists for.

    Same measurements, two readings.  On sky the term reverses between the roll
    clusters and the report calls it "not static" -- which is the verdict #340
    rests on.  De-rotated, it is a constant and the report calls it STATIC.
    If ``derotate`` stops rotating, the second half of this test fails and the
    report is back to a test that cannot tell the two cases apart.
    """
    m = _mod()
    rows = _sky_rows(_INSTRUMENT_TERM, _ROLLS)
    roll = _roll_lookup(_ROLLS)

    sky = m.analyse(m.deviations(rows)[0], min_per_field=10)
    inst = m.analyse(m.deviations(m.derotate(rows, roll))[0], min_per_field=10)

    for det in ('nrca1', 'nrca4'):
        assert sky[det]['static'] is False, (
            f'{det} must read "not static" on sky -- that is the degeneracy')
        assert sky[det]['between_field_sd'] > abs(sky[det]['mean'])

        assert inst[det]['static'] is True, (
            f'{det} carries a constant instrument-frame term and must read '
            f'STATIC once de-rotated')
        assert inst[det]['mean'] == pytest.approx(_INSTRUMENT_TERM[det][1],
                                                  abs=0.25)


def test_the_shuffled_control_does_not_recover_the_term():
    """De-rotating by the WRONG angle must not read STATIC.

    Without this, "the de-rotated column found a term" could be axis mixing:
    a ~90 deg rotation exchanges the two axes, which shrinks the scatter of
    both whatever angle was used.
    """
    m = _mod()
    rows = _sky_rows(_INSTRUMENT_TERM, _ROLLS)
    roll = _roll_lookup(_ROLLS)
    for seed in (1, 2, 3):
        bad = m.shuffle_rolls(roll, seed)
        if bad == roll:
            continue                       # a permutation that fixed every key
        res = m.analyse(m.deviations(m.derotate(rows, bad))[0], min_per_field=10)
        assert not any(r['static'] for r in res.values()), (
            f'seed {seed}: a wrong-angle de-rotation reported a static term')


def test_pure_noise_stays_not_static_in_both_frames():
    """De-rotation must not manufacture a term where there is none."""
    m = _mod()
    flat = {d: (0.0, 0.0) for d in _INSTRUMENT_TERM}
    rows = _sky_rows(flat, _ROLLS, seed=11)
    roll = _roll_lookup(_ROLLS)
    sky = m.analyse(m.deviations(rows)[0], min_per_field=10)
    inst = m.analyse(m.deviations(m.derotate(rows, roll))[0], min_per_field=10)
    assert not any(r['static'] for r in sky.values())
    assert not any(r['static'] for r in inst.values())


def test_shuffle_rolls_keeps_the_same_angles():
    """The control must differ from the real lookup only in WHICH band gets
    which angle -- a control drawn from a different distribution would compare
    two things at once."""
    m = _mod()
    roll = _roll_lookup(_ROLLS)
    bad = m.shuffle_rolls(roll, 1)
    assert sorted(bad) == sorted(roll)
    assert sorted(bad.values()) == sorted(roll.values())


def test_a_band_with_no_measured_roll_is_dropped_not_guessed():
    """Assigning a wrong angle rotates a real vector into a wrong one, which is
    indistinguishable from noise in the output.  Dropping the band is visible in
    the counts; guessing is not."""
    m = _mod()
    rows = _sky_rows(_INSTRUMENT_TERM, _ROLLS)
    roll = _roll_lookup(_ROLLS)
    del roll[('fieldB', 'F212N')]
    out = m.derotate(rows, roll)
    assert {r['field'] for r in out} == set(_ROLLS) - {'fieldB'}
    assert len(out) == len(rows) - sum(r['field'] == 'fieldB' for r in rows)


def test_derotation_is_a_rotation_and_preserves_length():
    m = _mod()
    rows = _sky_rows(_INSTRUMENT_TERM, _ROLLS)
    out = m.derotate(rows, _roll_lookup(_ROLLS))
    a = np.array([[r['dra'], r['ddec']] for r in rows])
    b = np.array([[r['dra'], r['ddec']] for r in out])
    assert np.allclose(np.hypot(*a.T), np.hypot(*b.T))
