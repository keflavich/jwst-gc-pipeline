"""Readers of per-frame catalog names must allow the observation token (#316).

`obs_token` inserts `_o023` / `_j6778` BETWEEN the detector and the visit:

    f200w_nrca1_o023_visit001_vgroup02201_exp00001_m2_daophot_basic.fits
                  ^^^^^

A pattern that requires `_visit` immediately after the detector does not raise
on those names -- it just doesn't match, so the reader silently solves on a
subset.  Measured on the live trees before this fix:

    gc2211  F200W   592 globbed   192 parsed   400 skipped   (68%)
    ngc6334 F200W   560 globbed   280 parsed   280 skipped   (50%)

`cataloging.py::_DETECTOR_TOKEN_RE` was fixed for exactly this in #302
(de28585); these are the same property in the readers #316 enumerates.
"""
import os
import re

import pytest

from jwst_gc_pipeline.photometry.naming import perframe_name_re

REPO = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..'))

#: Real names from the live trees, both spellings of each.
TOKENED = [
    'f200w_nrca1_o023_visit001_vgroup02201_exp00001_m2_daophot_basic.fits',
    'f405n_nrcalong_j6778_visit001_vgroup02201_exp00003_m2_daophot_basic.fits',
    'f410m_nrcblong_o002_visit002_vgroup02201_exp00008_m2_daophot_basic.fits',
]
UNTOKENED = [
    'f200w_nrca1_visit001_vgroup02201_exp00001_m2_daophot_basic.fits',
    'f405n_nrcalong_visit001_vgroup02201_exp00003_m2_daophot_basic.fits',
    'f410m_nrcblong_visit002_vgroup02201_exp00008_m2_daophot_basic.fits',
]


@pytest.mark.parametrize('name', TOKENED + UNTOKENED)
def test_the_shared_pattern_reads_both_spellings(name):
    m = perframe_name_re().search(name)
    assert m is not None, name
    band, det, visit, vgroup, exp = m.groups()
    assert name.startswith(f'{band}_{det}_')
    assert f'_visit{visit}_' in name
    assert f'_exp{exp}_' in name


def test_the_observation_token_is_not_mistaken_for_the_vgroup():
    """`_o023` sits where a careless pattern could swallow it into the visit or
    vgroup group, which would be worse than skipping -- it would mis-attribute
    the frame rather than drop it."""
    m = perframe_name_re().search(TOKENED[0])
    assert m.group(3) == '001', m.groups()
    assert m.group(4) == '02201', m.groups()


# ---------------------------------------------------------------------------
# repo sweep: nobody may reintroduce the strict form
# ---------------------------------------------------------------------------

#: `{detector}_visit` with nothing allowed in between.  Matches the source TEXT
#: of a pattern, so it fires on the regex literal rather than on a filename.
_STRICT = re.compile(r'nrc\[ab\][^\'"]{0,24}?\)_visit')

#: Files that legitimately contain the strict form.
_PARDON = {
    # this test's own docstring and the _STRICT pattern above
    'jwst_gc_pipeline/photometry/tests/test_perframe_name_readers_allow_obs_token.py',
}


def _py_files():
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs
                   if d not in {'.git', '__pycache__', 'build', 'dist'}]
        for f in files:
            if f.endswith('.py'):
                yield os.path.join(root, f)


def test_no_reader_requires_visit_immediately_after_the_detector():
    offenders = []
    for path in _py_files():
        rel = os.path.relpath(path, REPO)
        if rel in _PARDON:
            continue
        try:
            text = open(path, encoding='utf-8').read()
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith('#'):
                continue
            if _STRICT.search(line):
                offenders.append(f'{rel}:{i}: {line.strip()}')
    assert not offenders, (
        'these patterns require `_visit` immediately after the detector, so '
        'they SILENTLY SKIP every tokened frame (gc2211, ngc6334) rather than '
        'failing:\n  ' + '\n  '.join(offenders) +
        '\n\nUse naming.OBS_TOKEN_PATTERN (or naming.perframe_name_re()).')


def test_globs_leave_room_for_the_token_between_detector_and_visit():
    """`decompose_selfcal` used `{det}_visit*`, so the tokened files never even
    reached its reader: 24 of gc2211's 68 nrca1 F200W frames were globbed."""
    strict_glob = re.compile(r'\{det\}_visit\*|\{detector\}_visit\*')
    offenders = []
    for path in _py_files():
        rel = os.path.relpath(path, REPO)
        if rel in _PARDON:
            continue
        try:
            text = open(path, encoding='utf-8').read()
        except (OSError, UnicodeDecodeError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith('#'):
                continue
            if strict_glob.search(line):
                offenders.append(f'{rel}:{i}: {line.strip()}')
    assert not offenders, (
        'these globs cannot see a tokened frame at all:\n  '
        + '\n  '.join(offenders))


# ---------------------------------------------------------------------------
# the specific readers #316 names, exercised through their own patterns
# ---------------------------------------------------------------------------

def _pattern_from(relpath, marker):
    """Pull the compiled pattern text off the line containing `marker`."""
    text = open(os.path.join(REPO, relpath), encoding='utf-8').read()
    for line in text.splitlines():
        if marker in line and 're.search' in line:
            return line
    raise AssertionError(f'{relpath}: no re.search line containing {marker!r}')


@pytest.mark.parametrize('relpath,marker', [
    ('jwst_gc_pipeline/photometry/generate_offsets_table.py', '_visit'),
    ('scripts/analysis/siaf_selfcal/network_selfcal.py', '_visit'),
    ('scripts/analysis/solve_filter_frame_offsets.py', '_visit'),
])
def test_each_named_reader_admits_the_token(relpath, marker):
    line = _pattern_from(relpath, marker)
    assert 'o\\d{3}' in line or 'OBS_TOKEN_PATTERN' in line, (
        f'{relpath} still requires _visit right after the detector: {line.strip()}')
