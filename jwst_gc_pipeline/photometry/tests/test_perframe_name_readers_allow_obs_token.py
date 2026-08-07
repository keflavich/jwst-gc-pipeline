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

# ---------------------------------------------------------------------------
# BEHAVIOURAL: compile the pattern each reader actually ships and run it.
#
# The first version of this file only grepped source TEXT for `o\d{3}`.  Both
# broken readers CONTAINED that text while compiling to something else
# entirely: they were `rf''`-strings, where `{3}` is a replacement field
# evaluating to `3` and `{4,5}` to the tuple `(4, 5)`.  So the shipped regex
# was `(?:_(?:o\d3|j\d(4, 5)))?` -- which does not match a token, AND adds a
# CAPTURING group that shifts every index after the detector.  On brick that
# turned 192 usable catalogs into 16.
#
# #316's own closing paragraph makes this criticism of
# test_lw_is_named_by_detector_at_every_naming_site.  A source-grep here would
# have repeated it, so these load the module and exercise the pattern.
# ---------------------------------------------------------------------------

READER_MODULES = [
    'scripts/analysis/siaf_selfcal/network_selfcal.py',
    'scripts/analysis/solve_filter_frame_offsets.py',
]


def _compiled_patterns(relpath):
    """Every regex the module builds, compiled the way the module builds it.

    Executes the `re.search(...)` argument expression with the module's own
    names, so an f-string that mangles its braces is compiled here exactly as
    it is at runtime.
    """
    import ast
    import re as _re
    from jwst_gc_pipeline.photometry import naming
    path = os.path.join(REPO, relpath)
    tree = ast.parse(open(path, encoding='utf-8').read())
    env = {'re': _re, 'band': 'f200w',
           'OBS_TOKEN_PATTERN': naming.OBS_TOKEN_PATTERN,
           'OBS_TOKEN_CAPTURE': naming.OBS_TOKEN_CAPTURE}
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'search' and node.args):
            continue
        expr = ast.Expression(body=node.args[0])
        ast.fix_missing_locations(expr)
        try:
            pat = eval(compile(expr, path, 'eval'), env)
        except (NameError, TypeError, SyntaxError, ValueError):
            continue
        if isinstance(pat, str) and '_visit' in pat and 'nrc' in pat:
            out.append(pat)
    return out


TOKENED_ONE = 'f200w_nrca1_o023_visit001_vgroup02201_exp00001_m2_daophot_basic.fits'
UNTOKENED_ONE = 'f200w_nrca1_visit001_vgroup02201_exp00001_m2_daophot_basic.fits'


@pytest.mark.parametrize('relpath', READER_MODULES)
def test_each_readers_SHIPPED_pattern_reads_a_tokened_name(relpath):
    """Compile what the module ACTUALLY builds, not what its source looks like.

    The first version of this PR failed exactly here and the source-grep tests
    could not see it: both readers were `rf''`-strings, where `{3}` is a
    replacement field evaluating to `3` and `{4,5}` to the tuple `(4, 5)`.  The
    shipped regex was `(?:_(?:o\\d3|j\\d(4, 5)))?` -- no token match, plus a
    stray CAPTURING group that shifted every index after the detector.  On
    brick that turned 192 usable catalogs into 16.
    """
    import re as _re
    pats = _compiled_patterns(relpath)
    assert pats, f'{relpath}: found no per-frame pattern to compile'
    for pat in pats:
        rx = _re.compile(pat)
        assert rx.search(TOKENED_ONE) is not None, (
            f'{relpath}: shipped pattern does not match a tokened name.\n'
            f'  compiled to: {pat!r}\n'
            f'  (an rf-string eats `{{3}}`/`{{4,5}}` -- concatenate instead)')
        assert rx.search(UNTOKENED_ONE) is not None, (
            f'{relpath}: regressed the untokened name: {pat!r}')


@pytest.mark.parametrize('relpath', READER_MODULES)
def test_the_token_is_CAPTURED_so_it_can_reach_the_key(relpath):
    """Matching the token is only half of it.

    These readers KEY on the frame identity, and gc2211's five pointings reuse
    the same (visit, vgroup, exposure) tuples.  A pattern that MATCHES the
    token without CAPTURING it lets those frames parse and then collide -- which
    trades "silently skips 400 frames" for "silently overwrites 400 frames".
    Measured on gc2211 F200W: 592 files -> 32 keys, 560 colliding.
    """
    import re as _re
    for pat in _compiled_patterns(relpath):
        rx = _re.compile(pat)
        mt, mu = rx.search(TOKENED_ONE), rx.search(UNTOKENED_ONE)
        assert 'o023' in mt.groups(), (
            f'{relpath}: the observation token is matched but NOT captured, so '
            f'it cannot reach the frame key: {pat!r} -> {mt.groups()}')
        # and the two names must differ ONLY in that token
        assert [g for g in mt.groups() if g != 'o023'] == \
               [g for g in mu.groups() if g is not None], (
            f'{relpath}: tokened and untokened names disagree on something '
            f'other than the token.\n  tokened   {mt.groups()}\n'
            f'  untokened {mu.groups()}')


@pytest.mark.parametrize('relpath', READER_MODULES)
def test_the_pattern_captures_the_right_components(relpath):
    """A pattern can match and still be wrong: the stray `(4, 5)` group matched
    nothing but still shifted `m.group(2)` off the visit onto None."""
    import re as _re
    name = 'f200w_nrcalong_j6778_visit007_vgroup02201_exp00042_m2_daophot_basic.fits'
    for pat in _compiled_patterns(relpath):
        m = _re.compile(pat).search(name)
        assert m is not None, pat
        g = m.groups()
        assert g[0] == 'nrcalong', (pat, g)
        assert 'j6778' in g, (pat, g)
        assert '007' in g, (pat, g)
        assert g[-1] == '00042', (pat, g)
        assert None not in g, (
            f'{relpath}: a group came back None -- an optional group is '
            f'capturing where it should not: {pat!r} -> {g}')
