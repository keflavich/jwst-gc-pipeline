"""A re-tie loop that repeats itself must stop, not run to MAXITER.

`run_field_retie_loop.sh` stops when the checkpoint passes, or when the offsets
table is byte-identical after a failed finalize.  Neither fires at a fixed
point: the checkpoint keeps measuring the same residual, keeps correcting it,
and the table changes in the last decimal place, so the md5 differs.

sgrc ran four passes at ~7 h each that way, and would have run twelve.  The
records (F115W/o012, F162M/o012, 2026-08-08/09) show the two shapes this has to
tell apart:

  * F115W repeats EXACTLY -- 48 exposures, largest pass-to-pass change 0.00 mas
  * F162M OSCILLATES -- consecutive passes differ by ~5.9 mas, but every pass
    reproduces the one before last to 0.20 mas

Neither is progress, and the second was invisible to a first attempt that
compared the CORRECTIONS list: which exposures cross the 4 mas floor churns
pass to pass (F162M went 21 -> 21 corrections with five keys swapped) purely
because exposures sitting at 2-4 mas drift across the threshold.  Comparing the
per-exposure MEASUREMENT is what makes both visible.
"""
import json

import pytest

from jwst_gc_pipeline.photometry.retie_fixed_point import (
    DEFAULT_REPEATS, compare, find_fixed_point, measurements)


def _rec(offsets, corrections=1):
    """A checkpoint record carrying ``{exposure key: (dra, ddec)}``."""
    return {
        'visits': [{'visit': '1', 'filtername': 'F162M', 'exposures': [
            {'key': list(k), 'dra': v[0], 'ddec': v[1],
             'off': (v[0] ** 2 + v[1] ** 2) ** 0.5}
            for k, v in sorted(offsets.items())]}],
        'corrections': [{'visit': '1', 'filtername': 'F162M', 'exposure': 1,
                         'module': 'nrca1', 'dra_onsky_mas': 1.0,
                         'ddec_onsky_mas': 1.0}] * corrections,
    }


A = {('1', 1, 'nrca1'): (-0.21, -2.52), ('1', 2, 'nrca1'): (-0.11, -3.43),
     ('1', 1, 'nrcb4'): (-0.79, -5.14)}
#: the same field state, re-measured (sgrc's pass-to-pass wobble is ~0.1 mas)
A_AGAIN = {k: (v[0] + 0.03, v[1] + 0.06) for k, v in A.items()}
#: the OTHER state of the oscillation
B = {('1', 1, 'nrca1'): (-0.24, -2.45), ('1', 2, 'nrca1'): (-0.14, -3.36),
     ('1', 1, 'nrcb4'): (-0.78, +0.80)}


def test_a_re_measurement_of_the_same_state_is_the_same():
    same, detail = compare(_rec(A), _rec(A_AGAIN))
    assert same, detail
    assert '3 exposure(s)' in detail


def test_a_state_that_actually_moved_is_not():
    same, detail = compare(_rec(A), _rec(B))
    assert not same
    assert 'nrcb4' in detail


def test_the_comparison_is_of_MEASUREMENTS_not_of_the_correction_list():
    """The defect in the first version.  Which exposures appear in
    `corrections` depends on which cross the floor, and that membership churns
    while the field state does not: sgrc F162M read 21 -> 21 corrections with
    five keys swapped and would have been called 'still converging'."""
    a, b = _rec(A, corrections=21), _rec(A_AGAIN, corrections=21)
    b['corrections'] = b['corrections'][:16]        # 5 dropped below the floor
    same, _ = compare(a, b)
    assert same, 'a changed correction SET must not read as a changed state'


def test_no_corrections_at_all_is_the_loops_own_converged_exit():
    """Zero corrections in both passes is the checkpoint PASSING.  Reporting
    that as a fixed point would turn a success into a stop."""
    same, detail = compare(_rec(A, corrections=0), _rec(A_AGAIN, corrections=0))
    assert not same
    assert 'no corrections' in detail


def test_disjoint_exposures_are_not_silently_called_identical():
    other = {('1', 9, 'nrcz9'): (0.0, 0.0)}
    same, detail = compare(_rec(A), _rec(other))
    assert not same
    assert 'no exposure measured in both' in detail


# ---------------------------------------------------------------------------
# find_fixed_point over a history
# ---------------------------------------------------------------------------

def _history(tmp_path, states, filt='F162M', token='o012'):
    d = tmp_path / 'astrometry_checkpoints'
    d.mkdir(exist_ok=True)
    for i, st in enumerate(states):
        p = d / f'checkpoint_m2_{filt}_{token}_2026080{i}T000000Z.json'
        p.write_text(json.dumps(_rec(st)))
    # the loop also writes a _latest COPY of the newest record; counting it
    # would compare a pass against itself and invent a repeat
    (d / f'checkpoint_m2_{filt}_{token}_latest.json').write_text(
        json.dumps(_rec(states[-1])))
    return str(d)


def test_an_exact_repeat_is_caught(tmp_path):
    d = _history(tmp_path, [A, A_AGAIN, A, A_AGAIN])
    stuck, lines = find_fixed_point(d)
    assert stuck
    assert any('REPEATING' in ln for ln in lines)


def test_a_period_2_OSCILLATION_is_caught(tmp_path):
    """sgrc F162M: consecutive passes differ, but every pass reproduces the one
    before last.  Not progress, and invisible to a consecutive-only test."""
    d = _history(tmp_path, [A, B, A_AGAIN, B])
    stuck, lines = find_fixed_point(d)
    assert stuck
    assert any('OSCILLATING' in ln for ln in lines)


def test_a_loop_that_is_REALLY_converging_is_left_alone(tmp_path):
    """Each pass halves the residual.  Stopping this would abandon a run that
    was about to succeed -- the failure mode that matters most here."""
    states = [{k: (v[0] / 2 ** i, v[1] / 2 ** i) for k, v in A.items()}
              for i in range(5)]
    d = _history(tmp_path, states)
    stuck, lines = find_fixed_point(d)
    assert not stuck, lines


def test_too_little_history_does_not_guess(tmp_path):
    d = _history(tmp_path, [A, A_AGAIN])
    stuck, lines = find_fixed_point(d)
    assert not stuck
    assert any('need' in ln for ln in lines)


def test_the_latest_COPY_is_not_counted_as_a_pass(tmp_path):
    """`*_latest.json` duplicates the newest record.  Counting it would compare
    a pass against itself and report a repeat that never happened."""
    d = _history(tmp_path, [A, B, A_AGAIN])
    from jwst_gc_pipeline.photometry.retie_fixed_point import load_records
    assert len(load_records(d)) == 3


def test_filters_are_judged_SEPARATELY(tmp_path):
    """A field runs several filters, each with its own history; pooling them
    would compare F162M's pass against F115W's and find a meaningless
    difference."""
    d = _history(tmp_path, [A, A_AGAIN, A, A_AGAIN], filt='F115W')
    _history(tmp_path, [{k: (v[0] / 2 ** i, v[1] / 2 ** i)
                         for k, v in A.items()} for i in range(4)],
             filt='F162M')
    stuck, lines = find_fixed_point(d)
    assert stuck
    assert any('F115W' in ln and 'REPEATING' in ln for ln in lines)
    assert any('F162M' in ln and 'still moving' in ln for ln in lines)


def test_one_stuck_filter_stops_the_loop(tmp_path):
    """The loop re-reduces the WHOLE field every pass, so the cost is the same
    whether one filter is stuck or all of them are."""
    d = _history(tmp_path, [A, A_AGAIN, A, A_AGAIN], filt='F115W')
    _history(tmp_path, [{k: (v[0] / 2 ** i, v[1] / 2 ** i)
                         for k, v in A.items()} for i in range(4)],
             filt='F162M')
    stuck, _ = find_fixed_point(d)
    assert stuck


@pytest.mark.parametrize('n', [2, 3])
def test_period_2_needs_TWO_lag_comparisons(tmp_path, n):
    """One lag-2 match could be chance.  With too little history the
    oscillation branch must decline rather than guess."""
    d = _history(tmp_path, [A, B, A_AGAIN, B][:n])
    stuck, _ = find_fixed_point(d, repeats=n)
    assert not stuck


def test_the_default_window_can_see_an_oscillation():
    """DEFAULT_REPEATS must be large enough for two lag-2 comparisons, or the
    shape sgrc actually exhibits would never be reported."""
    assert DEFAULT_REPEATS >= 4


def test_measurements_skips_an_exposure_with_no_tie():
    rec = _rec(A)
    rec['visits'][0]['exposures'].append(
        {'key': ['1', 7, 'nrca9'], 'dra': None, 'ddec': None})
    assert ('1', 7, 'nrca9') not in measurements(rec)
    assert len(measurements(rec)) == 3
