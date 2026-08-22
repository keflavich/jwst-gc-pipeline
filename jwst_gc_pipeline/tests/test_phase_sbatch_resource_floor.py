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
    """Two numbers for the same job should not drift apart."""
    src = open(DRIVER).read()
    want = {
        'cpus-per-task': re.search(r'FINALIZE_CPUS=\$\{FINALIZE_CPUS:-(\d+)\}', src).group(1),
        'mem': re.search(r'FINALIZE_MEM=\$\{FINALIZE_MEM:-(\S+?)\}', src).group(1),
        'time': re.search(r'FINALIZE_TIME=\$\{FINALIZE_TIME:-(\S+?)\}', src).group(1),
    }
    got = _directives(PHASE)
    for k, v in want.items():
        assert got[k] == v, f'{k}: script says {got[k]}, driver default is {v}'


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
