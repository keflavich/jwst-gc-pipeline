"""A target that already ends in its proposal number must not be doubled.

CLAUDE.md's standing rule is ``<target><program>-o<obsid>-<stage>``, which for
almost every field concatenates cleanly (``brick`` + ``2221`` ->
``brick2221-o001-m12-fanout``).  ``gc2211`` on proposal ``2211`` does not: naive
concatenation gives ``gc22112211``.

The jicama runner has collapsed this since it was written::

    local slug="${target}${proposal}"
    case "$target" in *"$proposal") slug="$target";; esac

``submit_cataloging_perframe.sh`` did not, so the SAME field was named two
different ways depending on which entry point submitted it -- the runner's
``gc2211_o046...`` for a driven run, and ``gc22112211-o028-m12-fanout`` for the
o028 measurement pass submitted directly (2026-08-23, renamed by hand while it
was still pending).  A queue that names one field two ways is the problem the
standing rule exists to prevent.
"""
import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUBMITTER = os.path.join(REPO, 'scripts', 'reduction',
                         'submit_cataloging_perframe.sh')


def _stem(target, proposal):
    """Run the script's own stem logic, so the test cannot drift from it."""
    src = open(SUBMITTER).read()
    m = re.search(r'^JOB_STEM=.*?\ncase .*?esac$', src, re.M | re.S)
    assert m, 'no JOB_STEM block found in ' + os.path.basename(SUBMITTER)
    r = subprocess.run(
        ['bash', '-c', f'TARGET={target}\nPROPOSAL={proposal}\n{m.group(0)}\n'
                       'echo "$JOB_STEM"'],
        capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


def test_the_submitter_defines_a_stem():
    src = open(SUBMITTER).read()
    assert 'JOB_STEM=' in src, (
        'the per-frame submitter must build its job names from a collapsed stem, '
        'not from a bare ${TARGET}${PROPOSAL}')


@pytest.mark.parametrize('target,proposal,expected', [
    ('gc2211', '2211', 'gc2211'),        # the doubling case
    ('brick', '2221', 'brick2221'),
    ('sgrc', '4147', 'sgrc4147'),
    ('w51', '6151', 'w516151'),
    ('cloudef', '2092', 'cloudef2092'),
])
def test_stem_collapses_only_when_the_target_ends_in_the_proposal(
        target, proposal, expected):
    assert _stem(target, proposal) == expected


def test_both_job_names_use_the_stem():
    """fan-out AND finalize -- naming one and not the other splits a chain."""
    src = open(SUBMITTER).read()
    names = re.findall(r'--job-name="([^"]+)"', src)
    assert len(names) >= 2, f'expected fan-out and finalize names, got {names}'
    bad = [n for n in names if '${TARGET}${PROPOSAL}' in n]
    assert not bad, (
        f'{bad} still concatenate TARGET and PROPOSAL directly; use $JOB_STEM '
        f'so gc2211/2211 does not become gc22112211')
    assert all('${JOB_STEM}' in n for n in names), names


def test_names_still_carry_obsid_and_stage():
    """The collapse must not cost the rest of the required shape."""
    src = open(SUBMITTER).read()
    for n in re.findall(r'--job-name="([^"]+)"', src):
        assert '-o${FIELD}-' in n, f'{n} lost the obsid'
        assert n.endswith(('-fanout', '-finalize')), f'{n} lost the stage'
        assert '${ph}' in n, f'{n} lost the phase'
