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
    tok = f'_{token}' if token else ''
    for i, st in enumerate(states):
        p = d / f'checkpoint_m2_{filt}{tok}_2026080{i}T000000Z.json'
        p.write_text(json.dumps(_rec(st)))
    # the loop also writes a _latest COPY of the newest record; counting it
    # would compare a pass against itself and invent a repeat
    (d / f'checkpoint_m2_{filt}{tok}_latest.json').write_text(
        json.dumps(_rec(states[-1])))
    return str(d)


def test_an_exact_repeat_is_caught(tmp_path):
    d = _history(tmp_path, [A, A_AGAIN, A, A_AGAIN])
    stuck, lines = find_fixed_point(d)
    assert stuck
    assert any('REPEATING' in ln for ln in lines)


def test_a_period_2_OSCILLATION_is_caught(tmp_path):
    """sgrc F162M: consecutive passes differ, but every pass reproduces the one
    before last.  Not progress, and invisible to a consecutive-only test.

    Needs four passes (two lag-2 comparisons), hence MAXITER>=4 -- sgrc ran 12.
    """
    d = _history(tmp_path, [A, B, A_AGAIN, B])
    stuck, lines = find_fixed_point(d, repeats=4)
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


def test_the_default_catches_a_plain_repeat():
    """Three passes is enough for two lag-1 comparisons.  Period-2 needs four
    and therefore an explicit MAXITER>=4; see
    test_an_OSCILLATION_declines_at_three_and_fires_at_four."""
    assert DEFAULT_REPEATS >= 3


def test_measurements_skips_an_exposure_with_no_tie():
    rec = _rec(A)
    rec['visits'][0]['exposures'].append(
        {'key': ['1', 7, 'nrca9'], 'dra': None, 'ddec': None})
    assert ('1', 7, 'nrca9') not in measurements(rec)
    assert len(measurements(rec)) == 3


# ---------------------------------------------------------------------------
# REACHABILITY.  Only sgrc and gc2211 write _oNNN m2 records; brick, cloudc,
# cloudef, sgrb2, arches, quintuplet, sickle, sgra and ngc6334 are untokened.
# Requiring the token made the glob match nothing, and the check exited 0 in
# silence -- inert on nine of the ten fields with checkpoint history, including
# the two with the longest.
# ---------------------------------------------------------------------------

def test_an_UNTOKENED_field_is_still_judged(tmp_path):
    """The token is a preference, not a filter."""
    d = _history(tmp_path, [A, A_AGAIN, A, A_AGAIN], token=None)
    stuck, lines = find_fixed_point(d, obs_token='o012')
    assert stuck, lines


def test_a_TOKENED_field_still_prefers_its_own_token(tmp_path):
    """The fallback must not make a token meaningless: gc2211's five
    observations share a directory, and o023's records are not o050's."""
    _history(tmp_path, [A, A_AGAIN, A, A_AGAIN], token='o023')
    d = _history(tmp_path, [B, B, B, B], token='o050')
    stuck, lines = find_fixed_point(d, obs_token='o023')
    assert stuck
    assert all('o050' not in ln for ln in lines), lines


def test_an_EMPTY_scan_SAYS_SO(tmp_path):
    """Silence reads as 'nothing is wrong'.  An operator must be able to tell a
    check that did not apply from a clean one."""
    d = tmp_path / 'astrometry_checkpoints'
    d.mkdir()
    stuck, lines = find_fixed_point(str(d))
    assert not stuck
    assert any('did NOT run' in ln for ln in lines), lines


def test_one_filter_with_BOTH_histories_gives_ONE_verdict(tmp_path):
    """sgrc F115W is continuous -- untokened to 2026-08-06, _o012 from
    2026-08-07 -- and splitting it reported two contradictory verdicts for one
    filter, judging a stale tail nobody writes as if it were live."""
    _history(tmp_path, [A, A_AGAIN, A, A_AGAIN], filt='F115W', token=None)
    d = _history(tmp_path, [{k: (v[0] / 2 ** i, v[1] / 2 ** i)
                             for k, v in A.items()} for i in range(4)],
                 filt='F115W', token='o012')
    stuck, lines = find_fixed_point(d)
    f115 = [ln for ln in lines if ln.startswith('F115W')]
    assert len(f115) == 1, f115
    assert 'o012' in f115[0], 'the live tokened history is the one that counts'


# ---------------------------------------------------------------------------
# --since.  brick, cloudc and cloudef all carry REPEATING histories from July
# campaigns; without a bound the first re-run of any of them stops at iteration
# 2 citing passes from a different campaign.
# ---------------------------------------------------------------------------

def test_an_EARLIER_campaigns_passes_are_not_THIS_loops(tmp_path):
    d = _history(tmp_path, [A, A_AGAIN, A, A_AGAIN])
    assert find_fixed_point(d)[0], 'unbounded, the stale history judges'
    stuck, lines = find_fixed_point(d, since='20260901T000000Z')
    assert not stuck
    assert any('did NOT run' in ln for ln in lines), lines


def test_since_keeps_the_records_written_after_it(tmp_path):
    d = _history(tmp_path, [A, A_AGAIN, A, A_AGAIN])
    assert find_fixed_point(d, since='20260800T000000Z')[0]


def test_the_loop_passes_its_own_start_time():
    """Source guard: the bound is worthless if the shell does not send it."""
    import pathlib
    sh = (pathlib.Path(__file__).parents[3] / 'scripts' / 'reduction'
          / 'run_field_retie_loop.sh').read_text()
    assert 'RETIE_RUN_START=$(date -u +%Y%m%dT%H%M%SZ)' in sh
    assert '--since "$RETIE_RUN_START"' in sh
    assert 'retie_fixed_point' in sh


# ---------------------------------------------------------------------------
# The default must be REACHABLE.  `--since` bounds the scan to this run, and the
# loop's own default cap is MAXITER=3 -- at most one record per filter per
# iteration.  A DEFAULT_REPEATS of 4 therefore printed "3 pass(es) recorded,
# need 4 to judge" and exited 0 on every default-configured loop.
# ---------------------------------------------------------------------------

def test_the_default_is_reachable_at_the_loops_default_MAXITER():
    import pathlib
    import re
    sh = (pathlib.Path(__file__).parents[3] / 'scripts' / 'reduction'
          / 'run_field_retie_loop.sh').read_text()
    m = re.search(r'^MAXITER=\$\{MAXITER:-(\d+)\}', sh, re.M)
    assert m, 'MAXITER default moved; re-check DEFAULT_REPEATS against it'
    assert DEFAULT_REPEATS <= int(m.group(1)), (
        f'DEFAULT_REPEATS={DEFAULT_REPEATS} needs more passes than the loop '
        f'can produce at MAXITER={m.group(1)}, so the check never runs')


def test_a_REPEAT_is_caught_with_only_three_passes(tmp_path):
    d = _history(tmp_path, [A, A_AGAIN, A])
    stuck, lines = find_fixed_point(d)
    assert stuck, lines
    assert any('REPEATING' in ln for ln in lines)


def test_an_OSCILLATION_declines_at_three_and_fires_at_four(tmp_path):
    """Two lag-2 comparisons are still required, so period-2 needs MAXITER>=4.
    Declining is the conservative direction: it reads as 'still moving'."""
    three = _history(tmp_path, [A, B, A_AGAIN])
    assert not find_fixed_point(three)[0]
    fourdir = tmp_path / 'four'
    fourdir.mkdir()
    four = _history(fourdir, [A, B, A_AGAIN, B])
    stuck, lines = find_fixed_point(four, repeats=4)
    assert stuck
    assert any('OSCILLATING' in ln for ln in lines)


# ---------------------------------------------------------------------------
# --accept-below-mas: a fixed point is a repeat, not a size
# ---------------------------------------------------------------------------
#
# Stopping on every fixed point holds a field for a decision the records have
# already made.  The two cases need opposite answers and the loop could not
# tell them apart:
#
#   * cloudc F212N and cloudef F360M repeat at 7-11 mas -- a per-detector
#     SIAF/DVA term the per-exposure offsets table cannot express, so no number
#     of further passes removes it;
#   * arches F212N exposure 4 repeats at 18-26 mas on six detectors at once,
#     which is a correction not reaching the frame -- a defect, and stopping is
#     right.
#
# So the ceiling is the decision, and it is written down rather than taken per
# run.  Everything below it still gets MEASURED and RECORDED; only the
# CORRECTION is withheld, by raising the m2 floor to just above it.

#: repeats at ~5.2 mas -- the systematic shape
SMALL = {('1', 1, 'nrca1'): (-0.21, -2.52), ('1', 2, 'nrca1'): (-0.11, -3.43),
         ('1', 1, 'nrcb4'): (-0.79, -5.14)}
SMALL_AGAIN = {k: (v[0] + 0.03, v[1] + 0.06) for k, v in SMALL.items()}
#: repeats at ~26 mas -- arches exposure 4, the defect shape
BIG = {**SMALL, ('1', 4, 'nrca4'): (-15.70, 20.88)}
BIG_AGAIN = {k: (v[0] + 0.03, v[1] + 0.06) for k, v in BIG.items()}


def _cli(record_dir, *extra):
    from jwst_gc_pipeline.photometry.retie_fixed_point import main
    return main(['--record-dir', record_dir, *extra])


def test_largest_residual_is_over_the_NEWEST_pass_only(tmp_path):
    """The older passes are what the loop has already superseded.  Including
    them reports a residual that no longer exists, and the caller sizes its
    floor against it."""
    from jwst_gc_pipeline.photometry.retie_fixed_point import (
        largest_measured_residual)
    huge = {k: (v[0] * 100, v[1] * 100) for k, v in SMALL.items()}
    d = _history(tmp_path, [huge, SMALL, SMALL_AGAIN])
    worst, key, label = largest_measured_residual(d)
    assert worst == pytest.approx(5.2, abs=0.3), (worst, key)
    assert 'F162M' in label


def test_without_the_flag_every_fixed_point_still_stops(tmp_path):
    """The behaviour before this existed, kept as the default: a ceiling is a
    decision, and it has to be made deliberately rather than inherited."""
    d = _history(tmp_path, [SMALL, SMALL_AGAIN, SMALL, SMALL_AGAIN])
    assert _cli(d) == 3


def test_a_bounded_residual_is_accepted(tmp_path, capsys):
    d = _history(tmp_path, [SMALL, SMALL_AGAIN, SMALL, SMALL_AGAIN])
    assert _cli(d, '--accept-below-mas', '15') == 4
    out = capsys.readouterr().out
    assert 'BOUNDED' in out
    assert 'ASTROM_M2_CORRECTION_FLOOR_MAS=' in out


def test_the_floor_it_prints_is_ABOVE_the_residual_it_has_to_clear(tmp_path,
                                                                  capsys):
    """A floor equal to the measurement re-corrects it on the very next pass,
    and the frozen m3+ stages then raise on the shift -- so the field stops one
    stage later than before, having spent a whole reduce to get there."""
    from jwst_gc_pipeline.photometry.retie_fixed_point import (
        largest_measured_residual)
    d = _history(tmp_path, [SMALL, SMALL_AGAIN, SMALL, SMALL_AGAIN])
    _cli(d, '--accept-below-mas', '15')
    line = [ln for ln in capsys.readouterr().out.splitlines()
            if ln.startswith('ASTROM_M2_CORRECTION_FLOOR_MAS=')][-1]
    floor = float(line.split('=')[1])
    worst, _, _ = largest_measured_residual(d)
    assert floor > worst


def test_a_residual_too_big_to_be_a_systematic_still_stops(tmp_path, capsys):
    """arches exposure 4: 26 mas repeating on six detectors at once is not a
    distortion term the table cannot express, it is a correction that is not
    reaching the frame."""
    d = _history(tmp_path, [BIG, BIG_AGAIN, BIG, BIG_AGAIN])
    assert _cli(d, '--accept-below-mas', '15') == 3
    out = capsys.readouterr().out
    assert 'NOT BOUNDED' in out
    assert '26' in out                     # the number the decision needs


def test_the_ceiling_does_not_make_a_MOVING_loop_stop_early(tmp_path):
    """Acceptance applies to a fixed point only.  A loop still converging must
    keep iterating however small its residual is -- it is about to succeed."""
    states = [{k: (v[0] / 2 ** i, v[1] / 2 ** i) for k, v in SMALL.items()}
              for i in range(5)]
    d = _history(tmp_path, states)
    assert _cli(d, '--accept-below-mas', '15') == 0


# ---------------------------------------------------------------------------
# What the floor may be sized against
# ---------------------------------------------------------------------------

def _multi_history(tmp_path, per_filter):
    """One record directory holding several filters' histories."""
    d = tmp_path / 'astrometry_checkpoints'
    d.mkdir(exist_ok=True)
    for filt, states in per_filter.items():
        for i, st in enumerate(states):
            (d / f'checkpoint_m2_{filt}_o012_2026080{i}T000000Z.json').write_text(
                json.dumps(_rec(st)))
    return str(d)


#: still converging: 96 -> 48 -> 24 -> 12 mas, one pass from done
_CONVERGING = [{('1', 7, 'nrca1'): (96.0, 0.0)},
               {('1', 7, 'nrca1'): (48.0, 0.0)},
               {('1', 7, 'nrca1'): (24.0, 0.0)},
               {('1', 7, 'nrca1'): (12.0, 0.0)}]


def test_a_still_converging_filter_blocks_acceptance(tmp_path, capsys):
    """The floor is applied to EVERY filter in the field.

    Accepting while one band is still halving its residual hands that band an
    amnesty it never needed and was one pass from clearing -- and the number
    the floor is sized from would be its residual, not the stuck one's.
    """
    d = _multi_history(tmp_path, {
        'F162M': [SMALL, SMALL_AGAIN, SMALL, SMALL_AGAIN],   # stuck at ~5 mas
        'F212N': _CONVERGING,                                # still moving
    })
    assert _cli(d, '--accept-below-mas', '15') == 3
    out = capsys.readouterr().out
    assert 'NOT ACCEPTED' in out
    assert 'F212N' in out


def test_the_floor_is_sized_only_from_the_STUCK_filters(tmp_path):
    """A group that has too little history to judge is not stuck, so its
    residual must not set the ceiling the whole field then runs under."""
    from jwst_gc_pipeline.photometry.retie_fixed_point import (
        group_verdicts, largest_measured_residual)
    d = _multi_history(tmp_path, {
        'F162M': [SMALL, SMALL_AGAIN, SMALL, SMALL_AGAIN],
        'F444W': [{('1', 8, 'nrcb1'): (300.0, 0.0)}],        # 1 pass, unjudged
    })
    stuck, moving, unjudged, _ = group_verdicts(d)
    assert ('F162M', 'o012') in stuck
    assert not moving, 'one pass is not "still moving", it is unjudged'
    assert ('F444W', 'o012') in unjudged, (
        'a group with too little history must be reported as unjudged: the '
        'floor is applied to the whole field, so accepting while it is unknown '
        'waives a residual nothing has looked at')
    worst, _, label = largest_measured_residual(d, only_groups=stuck)
    assert worst == pytest.approx(5.2, abs=0.3), worst
    assert 'F162M' in label
    unrestricted, _, _ = largest_measured_residual(d)
    assert unrestricted > 200, 'the guard is what keeps the 300 mas group out'


def test_an_exposure_the_checkpoint_would_never_correct_does_not_size_the_floor(
        tmp_path):
    """A module-antisymmetric residual is filed `alias_suspect` and is never
    corrected -- it is a detector-naming collision, not a misalignment.  Letting
    it set the floor would waive real corrections up to its size."""
    from jwst_gc_pipeline.photometry.retie_fixed_point import (
        largest_measured_residual)
    rec = _rec(SMALL)
    rec['visits'][0]['exposures'].append(
        {'key': ['1', 9, 'nrcb2'], 'dra': 900.0, 'ddec': 0.0, 'off': 900.0,
         'alias_suspect': True})
    d = tmp_path / 'astrometry_checkpoints'
    d.mkdir()
    (d / 'checkpoint_m2_F162M_o012_20260801T000000Z.json').write_text(
        json.dumps(rec))
    worst, key, _ = largest_measured_residual(str(d))
    assert worst == pytest.approx(5.2, abs=0.3), (worst, key)


def test_the_margin_is_the_tolerance_that_defined_the_fixed_point(tmp_path,
                                                                  capsys):
    """Consecutive passes may differ by up to --tol-mas and still count as
    repeating, so the next pass can legitimately read that much larger.  A
    margin below the tolerance leaves the field one ordinary wobble from
    re-correcting at m2 and raising in the frozen stages."""
    d = _history(tmp_path, [SMALL, SMALL_AGAIN, SMALL, SMALL_AGAIN])
    _cli(d, '--accept-below-mas', '15', '--tol-mas', '2.0')
    floor = float([ln for ln in capsys.readouterr().out.splitlines()
                   if ln.startswith('ASTROM_M2_CORRECTION_FLOOR_MAS=')
                   ][-1].split('=')[1])
    from jwst_gc_pipeline.photometry.retie_fixed_point import (
        largest_measured_residual)
    worst, _, _ = largest_measured_residual(d)
    assert floor >= worst + 2.0, (floor, worst)


def test_an_unjudged_group_blocks_acceptance(tmp_path, capsys):
    """w51 today: three stuck groups reading zero beside an unjudged one whose
    newest residual is 29 arcseconds.  The floor is applied to every filter in
    the field, so accepting waives a residual nothing has looked at."""
    d = _multi_history(tmp_path, {
        'F162M': [SMALL, SMALL_AGAIN, SMALL, SMALL_AGAIN],   # stuck at ~5 mas
        'F444W': [{('1', 8, 'nrcb1'): (29000.0, 0.0)}],      # 1 pass, unjudged
    })
    assert _cli(d, '--accept-below-mas', '15') == 3
    out = capsys.readouterr().out
    assert 'NOT ACCEPTED' in out
    assert 'F444W' in out


def test_a_stuck_group_with_nothing_correctable_is_not_BOUNDED(tmp_path, capsys):
    """"the largest residual is 0.00 mas" is the absence of a measurement, not a
    small one -- and the floor derived from it (0.5 mas) is LOWER than the 4.0
    the campaign runs at, so the next pass corrects everything and the loop
    never converges."""
    rec = _rec(SMALL)
    for exp in rec['visits'][0]['exposures']:
        exp['misaligned'] = False              # measured, and none actionable
    d = tmp_path / 'astrometry_checkpoints'
    d.mkdir()
    for i in range(4):
        (d / f'checkpoint_m2_F162M_o012_2026080{i}T000000Z.json').write_text(
            json.dumps(rec))
    assert _cli(str(d), '--accept-below-mas', '15') == 3
    assert 'no correctable residual' in capsys.readouterr().out


@pytest.mark.parametrize('bad', [float('inf'), float('nan'), 1e9, 100000.0])
def test_a_ceiling_that_is_not_a_small_number_is_refused(tmp_path, bad):
    """`inf`, `nan` and 1e9 all reached the acceptance branch, caught only by a
    character class in the shell wrapper -- which let 100000 through."""
    d = _history(tmp_path, [SMALL, SMALL_AGAIN, SMALL, SMALL_AGAIN])
    with pytest.raises(SystemExit) as exc:
        _cli(d, '--accept-below-mas', str(bad))
    assert exc.value.code == 2


def test_the_floor_may_not_exceed_the_ceiling_it_was_accepted_under(tmp_path,
                                                                    capsys):
    """The tolerance margin can push the floor past the ceiling: a 14.6 mas
    residual under a 15 mas ceiling needs a 15.1 mas floor, which waives more
    than was agreed to."""
    big = {('1', 1, 'nrca1'): (14.6, 0.0)}
    big_again = {k: (v[0] + 0.03, v[1]) for k, v in big.items()}
    d = _history(tmp_path, [big, big_again, big, big_again])
    assert _cli(d, '--accept-below-mas', '15') == 3
    assert 'exceeds the 15.0 mas ceiling' in capsys.readouterr().out


# ---------------------------------------------------------------------------
# the scan is scoped; the floor is not  (PR #388 re-review, blocker 2)
# ---------------------------------------------------------------------------

def test_a_filter_the_scoped_scan_never_saw_refuses_acceptance(tmp_path):
    """`--obs-token` and `--since` shrink the SCAN. They do not shrink the
    correction floor, which is applied to every filter in the field.

    Live: `--obs-token o012` on sgrc hid six groups the bare scan calls
    unjudged, with residuals up to 4.99 mas; `--obs-token o002` on cloudc hid
    three, up to 3.96 mas. Those filters were not stuck, not moving, not even
    unjudged -- they were absent, and the acceptance never considered them.
    """
    d = _history(tmp_path, [SMALL, SMALL_AGAIN, SMALL, SMALL_AGAIN],
                 filt='F162M', token='o012')
    assert _cli(d, '--accept-below-mas', '15', '--obs-token', 'o012') == 4
    assert _cli(d, '--accept-below-mas', '15', '--obs-token', 'o012',
                '--expect-filters', 'F162M F480M') == 3


def test_the_refusal_names_the_filters_that_were_not_covered(tmp_path, capsys):
    d = _history(tmp_path, [SMALL, SMALL_AGAIN, SMALL, SMALL_AGAIN],
                 filt='F162M', token='o012')
    _cli(d, '--accept-below-mas', '15', '--obs-token', 'o012',
         '--expect-filters', 'F162M F480M F405N')
    out = capsys.readouterr().out
    assert 'F405N' in out and 'F480M' in out
    assert 'F162M produced no' not in out, 'the covered filter must not be named'


def test_expect_filters_is_case_insensitive(tmp_path):
    d = _history(tmp_path, [SMALL, SMALL_AGAIN, SMALL, SMALL_AGAIN],
                 filt='F162M', token='o012')
    assert _cli(d, '--accept-below-mas', '15', '--obs-token', 'o012',
                '--expect-filters', 'f162m') == 4


def test_expect_filters_does_nothing_when_acceptance_is_off(tmp_path):
    """It is a guard on ACCEPTING, not a new way to fail a scan."""
    d = _history(tmp_path, [SMALL, SMALL_AGAIN, SMALL, SMALL_AGAIN],
                 filt='F162M', token='o012')
    assert _cli(d, '--obs-token', 'o012', '--expect-filters', 'F162M F480M') == 3
