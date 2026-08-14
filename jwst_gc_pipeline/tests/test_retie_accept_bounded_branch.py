"""The bounded-fixed-point branch must be REACHED, and must raise the floor.

`--accept-below-mas` exits 4 to say "this residual repeats, and it is small
enough to be the systematic a per-exposure offsets table cannot express".  The
loop then has to do two things, and doing only the first is worse than doing
neither:

1. break out to the m3-m7 chain instead of `exit 3`;
2. raise `ASTROM_M2_CORRECTION_FLOOR_MAS` above the residual first.

Skipping (2) means m2 re-measures the same residual inside the final chain,
CORRECTS it (the floor is still 4 mas), and the frozen m3+ stages then raise on
the shift -- so the field stops one stage later than it used to, having spent a
whole reduce and an m12 to get there.

The same `set -e` trap that hid the reduce guard applies to the check itself:
its nonzero exit codes ARE its output, so a bare `fp_out=$(...)` would exit the
script one line before the branch that reads them.
"""
import pathlib
import re

LOOP = (pathlib.Path(__file__).parents[2] / 'scripts' / 'reduction'
        / 'run_field_retie_loop.sh')


def _src():
    return LOOP.read_text()


def _branch():
    """The `elif [ "$fp_rc" -eq 4 ]` arm, up to the next arm."""
    src = _src()
    start = src.index('elif [ "$fp_rc" -eq 4 ]')
    return src[start:src.index('elif [ "$fp_rc" -ne 0 ]', start)]


def test_the_check_is_asked_for_a_ceiling():
    assert '--accept-below-mas' in _src()
    assert 'RETIE_ACCEPT_RESIDUAL_MAS' in _src()


def test_the_ceiling_defaults_to_accepting_nothing():
    """Unset means every fixed point stops, which is what it did before."""
    assert re.search(r'RETIE_ACCEPT_RESIDUAL_MAS:-0', _src())


def test_the_check_invocation_cannot_kill_the_loop():
    """rc 3 and rc 4 are the two verdicts; under `set -e` a bare assignment
    exits at the first of them."""
    src = _src()
    start = src.index('fp_out=$(PYTHONPATH=')
    stmt = src[start:src.index('echo "$fp_out"', start)]
    assert '|| fp_rc=' in stmt, (
        'the fixed-point check exits nonzero to report its verdict; without '
        '`|| fp_rc=$?` set -e kills the loop before the branch reads it')


def test_a_bounded_fixed_point_proceeds_rather_than_exiting():
    """rc=4's whole point: leave the iteration loop for the m3-m7 chain at the
    bottom of the script, which is what `break` does and `exit` does not."""
    assert 'break' in _branch()


def test_it_raises_the_correction_floor_before_proceeding():
    branch = _branch()
    assert 'ASTROM_M2_CORRECTION_FLOOR_MAS="$fp_floor"' in branch
    assert 'export ASTROM_M2_CORRECTION_FLOOR_MAS' in branch
    assert branch.index('ASTROM_M2_CORRECTION_FLOOR_MAS="$fp_floor"') \
        < branch.index('break'), 'the floor must be raised BEFORE the break'


def test_a_missing_floor_stops_rather_than_running_at_the_old_one():
    """If the check said BOUNDED but printed no floor, proceeding would run the
    frozen stages at the 4 mas floor -- the failure this branch exists to
    avoid, reached by the path that was supposed to avoid it."""
    branch = _branch()
    assert 'exit 3' in branch
    assert 'no floor' in branch


def test_the_floor_is_taken_from_the_checks_own_output():
    """Not recomputed by the shell: a second run of the check would judge a
    different set of records, and the number that reaches the chain has to be
    the one whose reasoning is in the log."""
    assert 'ASTROM_M2_CORRECTION_FLOOR_MAS=' in _branch()
    assert re.search(r"sed -n 's/\^ASTROM_M2_CORRECTION_FLOOR_MAS=//p'",
                     _branch())
