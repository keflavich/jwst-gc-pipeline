"""A driver script must not DEFAULT the m2 correction floor.

``m2_correction_floor`` resolves the environment BEFORE the per-field table, so
anything a driver exports overrides every entry in
``PER_FIELD_FLOOR_MAS``.  Two drivers used to do exactly that:

  * ``scripts/reduction/run_field_retie_loop.sh`` had
    ``ASTROM_M2_CORRECTION_FLOOR_MAS=${ASTROM_M2_CORRECTION_FLOOR_MAS:-0}``
    and exported it.  ``0`` is a VALUE, not "unset" -- it resolves to
    ``(0.0, 'env')`` and disables the floor outright, so every residual above
    the 2 mas consensus tolerance becomes actionable and m2 corrects a field's
    intrinsic per-exposure scatter.
  * the jicama runner's ``common.sh`` defaulted it to ``4.0``, which silently
    drove sgrc and cloudc (per-field 8.0) and w51 (6.0) at 4.0.

The second masked the first: the 0 never reached cataloging because the 4.0 was
exported first.  Removing either one alone is a regression, which is what this
test exists to catch -- ``grep`` the driver rather than trusting that the two
stay in step.
"""
import re
import subprocess
from pathlib import Path

import pytest

from jwst_gc_pipeline.photometry.m2_correction_floors import (
    FLOOR_ENV, m2_correction_floor)

LOOP = (Path(__file__).resolve().parents[3]
        / 'scripts' / 'reduction' / 'run_field_retie_loop.sh')


def _body():
    """The script with comment lines removed -- the env name appears in prose."""
    return '\n'.join(ln for ln in LOOP.read_text().splitlines()
                     if not ln.lstrip().startswith('#'))


def test_zero_is_a_value_not_an_absence():
    """The premise.  If this ever became (per-field, ...) the guard below is moot."""
    assert m2_correction_floor('sgrc', env={FLOOR_ENV: '0'}) == (0.0, 'env')
    assert m2_correction_floor('sgrc', env={FLOOR_ENV: '0.0'}) == (0.0, 'env')
    # ... whereas absent and empty both fall through to the field.
    assert m2_correction_floor('sgrc', env={}) == (8.0, 'per-field')
    assert m2_correction_floor('sgrc', env={FLOOR_ENV: ''}) == (8.0, 'per-field')


def test_the_retie_loop_does_not_default_the_floor():
    assigns = re.findall(rf'^\s*{FLOOR_ENV}=\$\{{{FLOOR_ENV}:-([^}}]*)\}}',
                         _body(), re.M)
    assert not assigns, (
        f'{LOOP.name} defaults {FLOOR_ENV} to {assigns!r}.  Any default here '
        f'overrides the per-field table for every field; a default of 0 '
        f'disables the floor entirely.  Leave it unset instead.')


def test_the_retie_loop_forwards_the_floor_only_when_set():
    """Children get the variable via an array that is empty when nothing was set."""
    body = _body()
    assert 'floor_env=()' in body, (
        'expected an empty-by-default `floor_env` array to carry the override')
    bare = re.findall(rf'^\s*{FLOOR_ENV}=\${FLOOR_ENV}\s*\\?$', body, re.M)
    assert not bare, (
        f'{LOOP.name} forwards {FLOOR_ENV} unconditionally at {len(bare)} site(s).  '
        f'Under `set -u` that also aborts the script when it is unset.  '
        f'Use "${{floor_env[@]}}" so an unset floor forwards nothing.')


def test_the_loop_is_syntactically_valid():
    """The array rewrite is easy to get wrong; `set -u` hides it until runtime."""
    assert subprocess.run(['bash', '-n', str(LOOP)],
                          capture_output=True).returncode == 0


@pytest.mark.parametrize('target,expected', [('sgrc', 8.0), ('cloudc', 8.0),
                                             ('w51', 6.0), ('brick', 4.0)])
def test_an_undriven_field_still_gets_its_per_field_floor(target, expected):
    """What the drivers are supposed to leave in place."""
    assert m2_correction_floor(target, env={}) == (expected, 'per-field')
