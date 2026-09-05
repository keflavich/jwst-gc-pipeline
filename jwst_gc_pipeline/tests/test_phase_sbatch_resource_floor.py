"""A hand-launched phase job must not inherit the cluster's default wall time.

``submit_cataloging_perframe_phase.sbatch`` is normally driven by
``submit_cataloging_perframe.sh``, which passes ``--cpus-per-task/--mem/--time``
on the command line.  Launching one phase by hand is nonetheless a supported and
useful thing to do -- it is how you re-run a failed finalize without repeating
its multi-hour fan-out.

Without its own ``#SBATCH`` lines the bare path inherits the CLUSTER default,
about 10 minutes, and dies mid-phase having done partial work.  Measured
2026-08-22: wd2's m7 finalize was re-run by hand for exactly that reason and hit
``TIMEOUT 00:10:18`` (job 39963352), taking its 18 chained m8 jobs down with it.
The successful attempt via the driver had run 10918 s.

Command-line flags override ``#SBATCH`` directives, so these are a floor for the
bare path and change nothing for driver-submitted jobs.
"""
import os
import re

import pytest


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PHASE = os.path.join(REPO, 'scripts', 'reduction',
                     'submit_cataloging_perframe_phase.sbatch')
DRIVER = os.path.join(REPO, 'scripts', 'reduction',
                      'submit_cataloging_perframe.sh')


def _directives(path):
    """#SBATCH directives SLURM will actually read.

    sbatch stops scanning at the first line that is neither blank nor a
    comment, so a directive placed after real code is silently ignored -- the
    parse has to be modelled, not just grepped.
    """
    out = {}
    for line in open(path):
        s = line.strip()
        if s.startswith('#SBATCH'):
            m = re.match(r'#SBATCH\s+--([a-z-]+)=(\S+)', s)
            if m:
                out[m.group(1)] = m.group(2)
        elif s and not s.startswith('#'):
            break          # end of the directive block
    return out


@pytest.mark.parametrize('flag', ['time', 'mem', 'cpus-per-task'])
def test_the_phase_script_declares_a_resource_floor(flag):
    d = _directives(PHASE)
    assert flag in d, (
        f'{flag} is not declared, so a hand-launched phase job takes the '
        'cluster default and dies mid-phase')


def test_the_declared_wall_time_is_hours_not_minutes():
    """The specific failure: 10 minutes against a finalize that needs hours."""
    t = _directives(PHASE)['time']
    parts = [int(x) for x in t.split('-')[-1].split(':')]
    hours = parts[0] + (parts[1] / 60 if len(parts) > 1 else 0)
    if '-' in t:
        hours += int(t.split('-')[0]) * 24
    assert hours >= 6, f'--time={t} is too short for a finalize'


def test_the_floor_matches_the_drivers_finalize_slice():
    """Two numbers for the same job should not drift apart.

    cpu and memory the driver still names once (its smallest tier).  Wall time
    it now sizes per field tier and stage (`_stage_time`, issue #737), so the
    floor mirrors the SHORTEST limit any finalize gets -- the same relationship
    ``--mem=64gb`` already has to the smallest memory tier.  A hand-launched
    phase that needs a big field's wall passes ``--time`` on the command line,
    which overrides this; the floor exists only so the bare path does not
    inherit the cluster's ~10 minute default.
    """
    src = open(DRIVER).read()
    want = {
        'cpus-per-task': re.search(r'FINALIZE_CPUS=\$\{FINALIZE_CPUS:-(\d+)\}', src).group(1),
        'mem': re.search(r'FINALIZE_MEM=\$\{FINALIZE_MEM:-(\S+?)\}', src).group(1),
        'time': min(_driver_stage_times(src), key=_hours),
    }
    got = _directives(PHASE)
    for k, v in want.items():
        assert got[k] == v, f'{k}: script says {got[k]}, driver default is {v}'


def _driver_stage_times(src):
    """Every --time the driver's `_stage_time` table can hand a phase job."""
    body = src.split('_stage_time() {', 1)[1].split('\n}', 1)[0]
    times = re.findall(r'\)\s*echo\s+(\S+)\s*;;', body)
    assert times, 'the driver no longer declares per-stage wall times'
    return times


def _hours(t):
    parts = [int(x) for x in t.split('-')[-1].split(':')]
    hours = parts[0] + (parts[1] / 60 if len(parts) > 1 else 0)
    return hours + (int(t.split('-')[0]) * 24 if '-' in t else 0)


def test_the_floor_clears_a_small_fields_measured_finalize_maximum():
    """The floor exists so a hand-launched phase survives its own stage.

    wd2's m7 finalize was re-run by hand, inherited the cluster default and
    died at 00:10:18.  The floor has to beat that by a wide margin, and it is
    sized on the SMALL tier because that is the wall the driver itself hands a
    small field: 14 days of sacct put every small-field finalize under 3.6 h
    (m92 m12-finalize), so 12 h is ~3x the measurement.  A hand-launched
    finalize on sgrb2 or brick needs the big-field wall and has to pass
    ``--time``; baking 36 h in here instead would cost a 107-second
    finalize-only recovery (crowded_l3, 2026-09-04) a 36 h backfill window.
    """
    assert _hours(_directives(PHASE)['time']) >= 3 * 3.6, (
        'the hand-launch floor is below 3x a small field\'s measured '
        'finalize maximum'
    )


def test_the_directives_are_reachable_by_the_parser():
    """They sit after a comment block explaining them.  Comments do not end the
    directive block -- verified against a live sbatch on this cluster
    (TimeLimit=03:00:00 read from a script with the same shape) -- but a stray
    non-comment line above them would silently drop every one."""
    d = _directives(PHASE)
    for flag in ('job-name', 'account', 'qos', 'time', 'mem', 'cpus-per-task'):
        assert flag in d, (
            f'{flag} is below the first non-comment line, so sbatch never '
            'reads it')


def test_the_driver_still_passes_its_own_flags():
    """The floor must not become the only source of truth: the driver sizes
    fan-out and finalize differently, and overrides on the command line."""
    src = open(DRIVER).read()
    for flag in ('--cpus-per-task=', '--mem=', '--time='):
        assert flag in src, f'driver no longer passes {flag}'
