"""The MIRI reduce must ask for a memory size its own measurements support.

``submit_reduction_miri.sbatch`` asked ``--mem=128gb``.  Measured MIRI reduce
MaxRSS, from the 68 array tasks this script submitted on 2026-09-07..10 (each a
plain ``sbatch`` with no ``--mem``, so the directive was the operative ask):
median 0.93 GB, p90 3.37 GB, worst 4.33 GB, nothing above 4.4 GB.  Twenty-two
hand-launched ``PipelineMIRI`` jobs from 2026-06 reach 5.86 GB; nothing that
has ever run ``PipelineMIRI`` exceeded 6 GB.

An over-ask costs QOS headroom rather than wall-clock.  Node placement is not
the mechanism on this cluster -- every hpg-default node has RealMemory=1004 GB
-- but ``astronomy-dept-b`` carries ``GrpTRES=cpu=6336,mem=49500G`` with
``DenyOnLimit``, so the ask is charged against a shared pool whose ratio is
7.81 GB per cpu.  At 128gb/16cpu a reduce sits at 8.0 GB per cpu and memory
becomes co-binding with cpu once the QOS saturates; at 32gb it is 2.0 GB per
cpu.  10678 puts a MIRI F770W parallel on every one of its 139 tiles, so the
ask is multiplied by the tile count.

The bound below is two-sided on purpose, because both directions have a real
cost.  Too high and the job sits in Priority behind the QOS; too low and it
OOM-kills a reduce several hours in, which during the 10678 window is worse
than the waste it fixes.  So this pins a *band* justified by the measurement
rather than one magic number: at least twice the worst MIRI reduce ever
observed, and no more than the NIRCam script's own 64 GB (NIRCam's tail really
does reach 57 GB; MIRI's does not, so MIRI must not ask for more than NIRCam).

The directive block is parsed the way sbatch parses it rather than grepped:
sbatch stops scanning at the first line that is neither blank nor a comment, so
a ``#SBATCH --mem`` placed after real code is silently ignored and a grep would
still find it.
"""
import os
import re

import pytest


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MIRI = os.path.join(REPO, 'scripts', 'reduction', 'submit_reduction_miri.sbatch')

#: Worst MaxRSS ever recorded for a job running ``PipelineMIRI`` (GB):
#: cloudc-2526-G0-F770W-PipelineMIRI, job 35443480, 2026-06-22.
WORST_OBSERVED_GB = 5.86

#: The floor: twice that worst case.  Below this the ask stops being headroom.
MIN_MEM_GB = 2 * WORST_OBSERVED_GB

#: The ceiling: the NIRCam reduce's own ask.  NIRCam's measured tail reaches
#: 57 GB and MIRI's reaches 6, so a MIRI ask above NIRCam's is unsupported by
#: any measurement either script has.
MAX_MEM_GB = 64


def _directives(path):
    """``#SBATCH`` settings SLURM will actually read, as a dict."""
    out = {}
    with open(path) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith('#SBATCH'):
                match = re.match(r'#SBATCH\s+--([a-z-]+)=(\S+)', stripped)
                if match:
                    out[match.group(1)] = match.group(2)
            elif stripped and not stripped.startswith('#'):
                break           # end of the directive block
    return out


def _to_gb(value):
    """A SLURM ``--mem`` value in GB.  Bare digits are megabytes, per sbatch."""
    match = re.fullmatch(r'(\d+(?:\.\d+)?)\s*([KMGT]?)B?', value.strip(),
                         re.IGNORECASE)
    if not match:
        raise ValueError(f'{value!r} is not a SLURM memory size')
    scale = {'': 1 / 1024, 'K': 1 / 1024 ** 2, 'M': 1 / 1024,
             'G': 1.0, 'T': 1024.0}[match.group(2).upper()]
    return float(match.group(1)) * scale


@pytest.mark.parametrize('value,expected', [
    ('32gb', 32.0), ('64GB', 64.0), ('128gb', 128.0),
    ('131072M', 128.0), ('1T', 1024.0), ('2048', 2.0),
])
def test_memory_sizes_parse(value, expected):
    """Guard the guard: a parser that read every size as 0 would pass nothing."""
    assert _to_gb(value) == pytest.approx(expected, rel=1e-6)


def test_the_miri_submit_script_is_there():
    """Guard the guard: a moved script would make the check below vacuous."""
    assert os.path.exists(MIRI), f'{MIRI} is gone; this test checks nothing'


def test_miri_reduce_declares_a_memory_ask():
    directives = _directives(MIRI)
    assert 'mem' in directives, (
        'submit_reduction_miri.sbatch has no `#SBATCH --mem` in its directive '
        'block, so a bare `sbatch` submission inherits the cluster default '
        'rather than a sized ask.  (A directive placed after the first line of '
        'real code is not read by sbatch and does not count.)')


def test_miri_reduce_memory_is_sized_from_its_measurements():
    asked = _to_gb(_directives(MIRI)['mem'])
    assert asked >= MIN_MEM_GB, (
        f'submit_reduction_miri.sbatch asks --mem={asked:g}gb, under '
        f'{MIN_MEM_GB:g} GB -- less than twice the {WORST_OBSERVED_GB} GB '
        f'worst MIRI reduce on record.  An OOM does not fail early: it fails '
        f'after the reduce has done most of its work.')
    assert asked <= MAX_MEM_GB, (
        f'submit_reduction_miri.sbatch asks --mem={asked:g}gb, above the '
        f'{MAX_MEM_GB} GB the NIRCam reduce asks.  No MIRI reduce has ever '
        f'exceeded {WORST_OBSERVED_GB} GB, and a big ask only fits on a node '
        f'with that much free -- with a MIRI parallel on every 10678 tile that '
        f'is paid 139 times over.')
