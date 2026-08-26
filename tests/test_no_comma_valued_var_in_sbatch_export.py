"""No submit script may inline a comma-valued variable into ``--export``.

Issue #532.  ``sbatch --export`` takes a COMMA-separated list of
``NAME=VALUE`` pairs, so a variable whose own value contains commas cannot be
listed in it:

    --export=ALL,TARGET=w51,MODULES=nrca,nrcb,merged,PHASE=m12

arrives in the job as ``MODULES=nrca``, with ``nrcb`` and ``merged`` read as
names of variables to inherit from the submitting environment.  The job then
runs on one module, merges half the field, and reports success -- w51's and
brick2221's m12 finalizes both COMPLETED that way on 2026-08-24, and the
truncation only surfaced because m3 hard-crashes on the products m12 never
wrote.  A phase that tolerated the absence would have produced a half-field
catalog silently.

The remedy every affected script already uses for ``EXTRA_ARGS`` is to
``export`` the variable and let the leading ``ALL`` carry it.  This test keeps
that from being re-litigated one script at a time.

Only variables KNOWN to be comma-valued are listed here.  Space-separated ones
(``FILTERS``) are safe in an ``--export`` list and are deliberately not
flagged.
"""
import re
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'

#: Variables whose values are comma-separated by construction.
COMMA_VALUED = ('MODULES', 'EXTRA_ARGS', 'EACH_SUFFIX_OVERRIDES')

#: ``,NAME=$...`` -- a comma immediately before the assignment means the pair
#: is already inside a comma-separated list, which is the trap.  ``--export=``
#: (or ``--export="``) immediately before it is the same thing at the head of
#: the list.
_INLINE = re.compile(
    r'(?:,|--export=["\']?)(' + '|'.join(COMMA_VALUED) + r')=\$')


def _shell_scripts():
    found = sorted(p for p in SCRIPTS.rglob('*')
                   if p.suffix in ('.sh', '.sbatch') and p.is_file())
    assert found, f'no shell scripts found under {SCRIPTS}'
    return found


def _code_lines(path):
    """Lines with whole-line comments dropped.

    Whole-line comments are where the trap is *documented*, at length, in
    several of these scripts; flagging the documentation would make the guard
    argue with itself.
    """
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if line.lstrip().startswith('#'):
            continue
        yield lineno, line


@pytest.mark.parametrize('script', _shell_scripts(), ids=lambda p: p.name)
def test_no_comma_valued_variable_inlined_into_an_export_list(script):
    offenders = [
        f'{script.name}:{lineno}: {line.strip()}'
        for lineno, line in _code_lines(script)
        if _INLINE.search(line)
    ]
    assert offenders == [], (
        'comma-valued variable inlined into a comma-separated --export list; '
        'sbatch truncates it at the first comma of its VALUE.  `export` the '
        'variable instead and let --export=ALL carry it:\n  '
        + '\n  '.join(offenders))


def test_the_pattern_catches_the_form_that_broke_w51():
    # The exact string `submit_cataloging_chain.sh` carried before #532.
    bad = 'COMMON_EXPORT="ALL,PROPOSAL=$PROPOSAL,TARGET=$TARGET,MODULES=$MODULES"'
    assert _INLINE.search(bad)
    # ...and the head-of-list form a hand-written sbatch uses.
    assert _INLINE.search('sbatch --export=MODULES=$MODULES script.sbatch')
    # Space-separated values stay legal.
    assert not _INLINE.search('COMMON="ALL,FILTERS=$FILTERS"')
    # So does the fixed form.
    assert not _INLINE.search('export MODULES')
