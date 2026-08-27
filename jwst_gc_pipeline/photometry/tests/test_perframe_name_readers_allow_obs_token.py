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
import subprocess

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


#: A tree this size has ~480 python files.  A floor turns "the enumeration
#: broke" into a failure instead of a clean sweep over nothing: an empty file
#: list makes every sweep below pass vacuously.
MIN_SCANNED_FILES = 350


def _py_files():
    """This repository's python files: tracked, plus untracked-not-ignored.

    Enumerating with ``os.walk(REPO)`` swept every sibling git worktree
    checked out UNDER the repo root -- 26 of them on the machine this runs on
    -- so the guard reported offenders in files that are not in this
    repository at this commit, including those worktrees' copies of this very
    file (``_PARDON`` is keyed on the repo-relative path, so a copy does not
    pardon itself).  A guard that is red for a reason unrelated to the change
    is one people learn to ignore.

    ``git ls-files`` refuses to descend into a directory that owns a ``.git``,
    so a nested worktree comes back as a bare directory entry and is dropped
    by the trailing-slash test.  ``-o --exclude-standard`` keeps the property
    the ``os.walk`` was reaching for: a NEW file is policed before it is
    staged, rather than passing locally and failing in CI once it is added.

    ``check=True`` on purpose.  A guard that cannot enumerate cannot do its
    job, and reporting a clean tree is the one answer it must not give.
    """
    out = subprocess.run(
        ['git', '-C', REPO, 'ls-files', '-c', '-o', '--exclude-standard', '-z'],
        capture_output=True, text=True, check=True).stdout
    found = 0
    for rel in out.split('\0'):
        # A nested repository/worktree yields `name/`; a submodule yields a
        # gitlink path with no trailing slash, which `isfile` drops as well.
        if not rel or rel.endswith('/') or not rel.endswith('.py'):
            continue
        path = os.path.join(REPO, rel)
        if os.path.isfile(path):
            found += 1
            yield path
    assert found >= MIN_SCANNED_FILES, (
        f'only {found} python files enumerated under {REPO}; the sweeps below '
        f'would pass over almost nothing.  The enumeration is broken, not the '
        f'tree.')


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


# ---------------------------------------------------------------------------
# Capturing the token is only half of it: it has to reach the KEY.
#
# Both readers key on the frame identity.  gc2211's five pointings reuse the
# same (visit, vgroup, exposure) tuples, so a key without the token collides
# them -- 56 of 80 gc2211 F200W identities, which is WORSE than the skip this
# PR fixes: it silently overwrites rather than silently omits.  Dropping
# `obstok` from either key left all 15 tests above green.
# ---------------------------------------------------------------------------

KEY_BUILDERS = {
    # module -> (line marker, the tuple/namedtuple construction to check)
    'scripts/analysis/siaf_selfcal/network_selfcal.py': 'cats[FrameKey(',
    'scripts/analysis/solve_filter_frame_offsets.py': 'key = (',
}


@pytest.mark.parametrize('relpath,marker', sorted(KEY_BUILDERS.items()))
def test_the_observation_token_reaches_the_frame_key(relpath, marker):
    """Two names differing ONLY in the observation token must not produce the
    same key."""
    import re as _re
    text = open(os.path.join(REPO, relpath), encoding='utf-8').read()
    line = [ln for ln in text.splitlines()
            if marker in ln and not ln.lstrip().startswith('#')]
    assert line, f'{relpath}: no key construction found ({marker!r})'
    assert 'obstok' in line[0], (
        f'{relpath}: the frame key does not carry the observation token, so '
        f"gc2211's five pointings collide onto one key: {line[0].strip()}")

    # and behaviourally: the captured groups must differ between the two names
    pats = _compiled_patterns(relpath)
    assert pats
    for pat in pats:
        rx = _re.compile(pat)
        a = rx.search(TOKENED_ONE).groups()
        b = rx.search(UNTOKENED_ONE).groups()
        assert a != b, (
            f'{relpath}: a tokened and an untokened name are indistinguishable '
            f'from this pattern, so no key built on it can separate them: {pat!r}')


def test_the_vgroup_reaches_the_frame_key_too():
    """The keys were also dropping the vgroup, which is what let gc2211 collapse
    592 catalogs onto 32 identities on `main`."""
    for relpath, marker in KEY_BUILDERS.items():
        text = open(os.path.join(REPO, relpath), encoding='utf-8').read()
        line = [ln for ln in text.splitlines()
                if marker in ln and not ln.lstrip().startswith('#')][0]
        assert 'vgroup' in line, (
            f'{relpath}: the frame key does not carry the vgroup: {line.strip()}')


# ---------------------------------------------------------------------------
# network_selfcal's design matrix must be indexed the SAME way its unknowns
# were enumerated.  FrameKey gained `obs` at position 1, so the positional
# lookups k[1]/k[2]/k[3] silently stopped meaning det/visit/vgroup and named
# tuples that are never inserted -- the solve died with KeyError on the first
# overlapping pair, and nothing noticed because the sweep greps for a source
# substring and no test builds a FrameKey.
# ---------------------------------------------------------------------------

def _selfcal():
    """Load network_selfcal without running its __main__ body."""
    import ast
    import os
    import types
    path = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                        'scripts', 'analysis', 'siaf_selfcal',
                        'network_selfcal.py')
    src = open(path).read()
    tree = ast.parse(src)
    # keep only the imports and defs; the module body does I/O at import time
    # imports, function defs, and the FrameKey definition only.  Every other
    # module-level assignment depends on catalogs read at import time.
    keep = [n for n in tree.body
            if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
            or (isinstance(n, ast.Assign)
                and any(getattr(t, 'id', '') == 'FrameKey' for t in n.targets))]
    mod = types.ModuleType('network_selfcal_partial')
    exec(compile(ast.Module(body=keep, type_ignores=[]), path, 'exec'),
         mod.__dict__)
    return mod


def test_two_observations_get_DISTINCT_attitude_keys():
    """gc2211's five pointings reuse (visit, exp); carrying `obs` in the
    attitude key is the whole reason FrameKey was widened."""
    m = _selfcal()
    a = m.FrameKey('f200w', 'o023', 'nrca1', '001', '02201', '00001')
    b = m.FrameKey('f200w', 'o046', 'nrca1', '001', '02201', '00001')
    assert m._attitude_of(a) != m._attitude_of(b)


def test_the_design_matrix_indexes_with_the_SAME_key_it_enumerated():
    """The 🔴: build ai/di exactly as the script does, then look up the way the
    row-assembly does.  Positional access raises KeyError here."""
    m = _selfcal()
    keys = [m.FrameKey('f200w', 'o023', 'nrca1', '001', '02201', '00001'),
            m.FrameKey('f200w', 'o023', 'nrca2', '001', '02201', '00001'),
            m.FrameKey('f200w', 'o046', 'nrca1', '001', '02201', '00001')]
    ai = {k: i for i, k in enumerate(sorted({m._attitude_of(k) for k in keys}))}
    di = {k: i for i, k in enumerate(sorted({(k.band, k.det) for k in keys}))}
    for k in keys:
        assert m._attitude_of(k) in ai
        assert (k.band, k.det) in di
    # and the shapes the OLD positional forms named are absent, which is why
    # they raised rather than silently mis-indexing
    assert (keys[0][0], keys[0][1]) not in di        # ('f200w', 'o023')
    assert (keys[0][0], keys[0][2], keys[0][3]) not in ai   # ('f200w','nrca1','001')


def test_the_row_assembly_source_uses_attribute_access():
    """Source-level, because running the solve needs real catalogs.  A return
    to `k2[0], k2[2], k2[3]` is the regression."""
    import os
    path = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                        'scripts', 'analysis', 'siaf_selfcal',
                        'network_selfcal.py')
    body = open(path).read()
    assert 'row[ai[_attitude_of(k2)]]' in body
    assert 'di[(k2.band, k2.det)]' in body
    assert 'row[ai[(k2[0], k2[2], k2[3])]]' not in body
