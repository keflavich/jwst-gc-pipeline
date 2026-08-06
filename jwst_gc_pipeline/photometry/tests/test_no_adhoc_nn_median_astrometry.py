"""Grep-guard: forbid NEW ad-hoc dense-nearest-neighbour-median astrometry.

The in-code guard (``measure_offsets.assert_sparse_reference_for_nn_median``) only
protects the pipeline call sites that import it.  An agent (or human) who writes a
standalone script that does ``match_to_catalog_sky(...)`` and then takes
``np.median`` of the separations/offsets bypasses that guard entirely -- which is
exactly how the brick-1182 / prop-2221 4" astrometry errors kept recurring.

This test is the language-level shield: it FAILS if any Python file in the repo
pairs a nearest-neighbour match with a median/mean reduction, UNLESS the file is on
the reviewed allowlist below.  A new file that trips it must either

  (a) switch to offset-histogram stacking -- use
      ``jwst_gc_pipeline.photometry.astrometry_offsets.measure_offset`` -- or

  (b) be added to ``ALLOWLIST`` with a one-line justification, after a human
      confirms its match+median usage is source-association or histogram-refinement,
      NOT a dense-NN-median astrometric correction.

See CLAUDE.md and reduction/ASTROMETRY_WCS_CORRECTION_FLOW.md.
"""
import ast
import re

import pytest
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# Key on match_to_catalog_sky specifically -- that is NEAREST-neighbour (nthneighbor
# defaults to 1), the method that collapses to ~0 in a crowded field.  We deliberately
# do NOT flag search_around_sky: it returns ALL pairs within a radius, which is the
# BASIS of the sanctioned offset-histogram stacking (which then medians only to refine
# the peak).  Flagging it would fire on the correct method.  File-level co-occurrence
# of a NN match with a median/mean is a strong tripwire for a NEW ad-hoc NN-median
# astrometry script; the allowlist carries the already-reviewed legitimate users.
#: Attribution label for a file whose match and reduce sit in DIFFERENT
#: functions.
#:
#: **An entry carrying this is a WHOLE-FILE exemption.** There is no way around
#: that: a cross-function split cannot be attributed to a function, so clearing
#: it cannot be scoped to one.  A file with such an entry can therefore hide a
#: SECOND, genuinely new split -- which is what the file-level allowlist did for
#: every file before the function scoping.
#:
#: What the scoping buys is that this is now true of EIGHT files instead of all
#: of them: a file whose entries are all function-scoped does catch a new split
#: (verified by test_a_function_scoped_file_still_catches_a_new_split).  Adding
#: a `<unattributed>` entry gives that up for that file, so
#: test_unattributed_entries_do_not_multiply fails if the count grows.
UNATTRIBUTED = "<unattributed>"

_MATCH = re.compile(r"\bmatch_to_catalog_sky\b")
_REDUCE = re.compile(r"\b(np\.n?median|np\.n?mean|\.median\(|\.mean\()")

# Reviewed files where match+median is legitimate (source association for merging /
# dedup, guarded NN, or histogram-stacking refinement -- NOT dense-NN-median
# correction). Keep this list SHORT and justified; do not add to it to silence the
# guard on a genuine violation.
#: Reviewed (file, function) pairs where a NN match beside a reduce is
#: legitimate -- source association for merging/dedup, a guarded NN, or
#: histogram-stacking refinement -- and NOT a dense-NN-median correction.
#:
#: Keyed on the FUNCTION, not the file.  A file-level key silences the whole
#: file: `merge_catalogs.py` was allowlisted for its source association, and a
#: new dense-NN-median added anywhere else in its 3000 lines would have passed
#: unseen.  Keying on the function is what makes an entry a statement about
#: reviewed code rather than about a filename.
#:
#: `<module>` means module-level code outside any function.
ALLOWLIST = {
    # This IS a nearest-neighbour median of positional offsets.  It is
    # legitimate only because of the RUNTIME guard
    # `assert_sparse_reference_for_nn_median` at measure_offsets.py:102, which
    # raises DenseNNMedianAstrometryError unless the reference is sparse.  The
    # guard, not the shape of the code, is what makes it safe.
    ("jwst_gc_pipeline/photometry/measure_offsets.py", "measure_offsets"),
    ("jwst_gc_pipeline/reduction/build_virac2_offsets.py", "coord_shift"),
    # ALSO a median of positional offsets -- merge_catalogs.py:444 medians
    # radiff/decdiff into tbl.meta['ra_offset'] -- and its sparse-reference
    # guard is under `if realign and basecrds is not None` (:398) only.  With
    # the production default realign=False the median is COMPUTED and stored
    # ungated, and only not APPLIED.  Allowlisted because nothing in the
    # production path consumes that metadata as a correction, but the
    # "median is over fluxes, not offsets" reading this entry used to carry was
    # simply wrong, and the gap is worth its own issue.
    ("jwst_gc_pipeline/photometry/merge_catalogs.py", "combine_singleframe"),
    # source ASSOCIATION for saturated replacement; the only reduce is a
    # magnitude median in an f-string (:2927), no offsets involved
    ("jwst_gc_pipeline/photometry/merge_catalogs.py", "replace_saturated"),
    # sanctioned: masking extended emission, no astrometry in it
    ("jwst_gc_pipeline/photometry/cataloging.py", "_filter_extended_emission"),
    # ---- cross-function splits: the match and the reduce are in DIFFERENT
    # functions, so they cannot be attributed to one and are allowlisted as
    # `<unattributed>`.  These entries are WEAKER than a function-scoped one --
    # they silence the whole file -- so each needs a reason that survives
    # re-reading, and a new one should be resisted.
    #
    # the sanctioned estimator itself: measure_offset's histogram peak is the
    # replacement this rule points at, and local_residual_map's medians are
    # per-cell refinements of an already-verified tie
    ("jwst_gc_pipeline/photometry/astrometry_offsets.py", "<unattributed>"),
    # source association for merging + a magnitude median; no astrometric
    # correction is derived from a match here
    ("jwst_gc_pipeline/photometry/crowdsource_catalogs_long.py", "<unattributed>"),
    # satstar correction-data collection: matches to label stars, medians their
    # photometry
    ("scripts/satstar_deblend/collect_correction_data.py", "<unattributed>"),
    # These five ALSO carry a co-occurrence outside their attributed functions.
    # Each already had a reviewed per-function entry; the whole-file entry is
    # what a cross-function split needs, and adding it is the price of the
    # tripwire actually covering these files -- previously one attributed hit
    # made the rest of the file invisible.
    ("jwst_gc_pipeline/photometry/cataloging.py", "<unattributed>"),
    ("jwst_gc_pipeline/photometry/merge_catalogs.py", "<unattributed>"),
    ("jwst_gc_pipeline/reduction/build_virac2_offsets.py", "<unattributed>"),
    ("scripts/miri_reduction/miri_f2550w_image3_rerun_v2.py", "<unattributed>"),
    ("docs/pr57_recovery_investigation/make_caveat_figs.py", "<unattributed>"),
    # one-off scripts outside the pipeline's astrometric path
    ("scripts/reduction/combine_brick_allband.py", "main"),
    # :130-137 medians NN matches against a DENSE NIRCam F405N reference and
    # prints "astrometry: median offset" -- the validation-fools-you pattern by
    # name.  PRINT ONLY: nothing reads it and no WCS is written from it.  It
    # should still be converted to measure_offset rather than left as an
    # example of the thing the rule forbids.
    ("scripts/miri_reduction/miri_f2550w_image3_rerun_v2.py", "<module>"),
    ("docs/pr57_recovery_investigation/make_caveat_figs.py", "<module>"),
}


def _iter_py_files():
    """Only GIT-TRACKED .py files -- the guard polices committed code, not local
    scratch scripts in the working tree (which would false-positive on CI runners
    that never see them and annoy locally)."""
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "ls-files", "*.py"],
                             capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return  # not a git checkout (e.g. sdist install) -> nothing to police
    for line in out.splitlines():
        rel = Path(line)
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        # tests reference the forbidden tokens in strings on purpose
        if "tests" in rel.parts or p.name.startswith("test_"):
            continue
        yield rel, p


def _offending_functions(text):
    """(function name) pairs where a NN match and a reduce occur TOGETHER.

    File-level co-occurrence is too coarse: a 3000-line module that does source
    association in one function and takes an unrelated median in another trips
    it, and the only remedy is an allowlist entry that then silences the whole
    file.  Scoping to the enclosing function keeps the tripwire on the pattern
    the rule is about -- a nearest-neighbour match reduced by a median, right
    there -- and lets the allowlist name reviewed code instead of filenames.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        # unparseable: fall back to the coarse test rather than skipping it
        return {"<unparseable>"} if (_MATCH.search(text) and _REDUCE.search(text)) else set()
    lines = text.splitlines()
    hits = set()
    spans = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = node.end_lineno or node.lineno
            spans.append((node.lineno, end))
            seg = "\n".join(lines[node.lineno - 1:end])
            if _MATCH.search(seg) and _REDUCE.search(seg):
                hits.add(node.name)
    module_only = "\n".join(
        line for i, line in enumerate(lines, 1)
        if not any(lo <= i <= hi for lo, hi in spans))
    if _MATCH.search(module_only) and _REDUCE.search(module_only):
        hits.add("<module>")
    return hits


def _has_unattributed_cooccurrence(text, attributed):
    """Do a match and a reduce co-occur OUTSIDE the attributed functions?

    Blank out every function the scoping could attribute a hit to; if the
    remainder still contains both tokens, the file carries a split the
    per-function entries do not cover.
    """
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return bool(_MATCH.search(text) and _REDUCE.search(text))
    lines = text.splitlines()
    blanked = list(lines)
    for node in ast.walk(tree):
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in attributed):
            for i in range(node.lineno - 1, (node.end_lineno or node.lineno)):
                if 0 <= i < len(blanked):
                    blanked[i] = ""
    rest = "\n".join(blanked)
    return bool(_MATCH.search(rest) and _REDUCE.search(rest))


def test_no_adhoc_nn_median_astrometry():
    """FILE-level co-occurrence is the tripwire; the function scoping only says
    WHERE.

    Scoping the tripwire itself to the enclosing function looked like a
    tightening and is a loosening: it is blind to a helper that does the match
    and a caller that takes the median, to a class with the two in different
    methods, to a match at module level reduced inside a function, and to a
    match on a decorator line.  Every one of those is the documented bypass --
    "a standalone script that does match_to_catalog_sky(...) and then takes
    np.median" -- with the two halves one `def` apart.

    So a file still trips on co-occurrence, and it is cleared only when EVERY
    function the scoping can attribute a hit to is allowlisted.  A file whose
    hit cannot be attributed to any allowlisted function (the cross-function
    split) fails with `::<unattributed>`.
    """
    offenders = []
    for rel, path in _iter_py_files():
        text = path.read_text(errors="replace")
        if not (_MATCH.search(text) and _REDUCE.search(text)):
            continue
        # `or {UNATTRIBUTED}` would fire only when NOTHING was attributed, so
        # any file with one allowlisted hit could hide a second, genuinely
        # cross-function one -- the exact bypass this design was reversed to
        # catch, in all ten allowlisted files.  A file trips on co-occurrence,
        # so it must be CLEARED by co-occurrence: every hit it can attribute
        # must be allowlisted AND, if the match and the reduce also occur
        # outside those functions, `<unattributed>` must be too.
        funcs = _offending_functions(text)
        for func in sorted(funcs):
            if (rel.as_posix(), func) not in ALLOWLIST:
                offenders.append(f"{rel.as_posix()}::{func}")
        if _has_unattributed_cooccurrence(text, funcs) and \
                (rel.as_posix(), UNATTRIBUTED) not in ALLOWLIST:
            offenders.append(f"{rel.as_posix()}::{UNATTRIBUTED}")
    assert not offenders, (
        "FORBIDDEN dense-NN-median astrometry pattern (a nearest-neighbour match "
        "co-occurring with a median/mean) found in non-allowlisted location(s):\n  "
        + "\n  ".join(sorted(offenders))
        + "\n\n`::<unattributed>` means the match and the reduce are in "
        "DIFFERENT functions of one file -- the cross-function split, which is "
        "the classic bypass and is never allowlistable per-function.\n\n"
        "Use jwst_gc_pipeline.photometry.astrometry_offsets.measure_offset "
        "(2D offset-histogram peak) instead. If this really is legitimate "
        "(source association for merging/dedup, or a histogram refinement -- "
        "NOT a dense-NN-median astrometric correction), add the (file, "
        "function) pair to ALLOWLIST with a one-line justification, after a "
        "human review. See CLAUDE.md ASTROMETRY RULE #1.")


def test_allowlist_has_no_dead_entries():
    """An allowlist entry that no longer trips is rot, and rot is how a list of
    20 grows until nobody reads it.  Delete entries whose code moved or was
    fixed -- half of this list was dead when the function scoping landed."""
    live = set()
    for rel, path in _iter_py_files():
        text = path.read_text(errors="replace")
        if not (_MATCH.search(text) and _REDUCE.search(text)):
            continue
        funcs = _offending_functions(text)
        for func in funcs:
            live.add((rel.as_posix(), func))
        if _has_unattributed_cooccurrence(text, funcs):
            live.add((rel.as_posix(), UNATTRIBUTED))
    dead = sorted(ALLOWLIST - live)
    assert not dead, (
        "ALLOWLIST entries that no longer trip the guard (remove them):\n  "
        + "\n  ".join(f"{f}::{fn}" for f, fn in dead))


def test_allowlist_entries_exist():
    """Every entry must point at a real file."""
    missing = sorted({rel for rel, _ in ALLOWLIST
                      if not (REPO_ROOT / rel).is_file()})
    assert not missing, (
        "ALLOWLIST references files that no longer exist (remove them):\n  "
        + "\n  ".join(missing))


# The five ways a cross-function split evades a function-scoped tripwire.  Each
# is the documented bypass -- "a standalone script that does
# match_to_catalog_sky(...) and then takes np.median" -- with the two halves one
# `def` apart.  Scoping the TRIPWIRE (rather than only the attribution) to the
# enclosing function made every one of these invisible.
_EVASIONS = {
    "helper_matches_caller_medians": (
        "def h(a, b):\n    return a.match_to_catalog_sky(b)\n"
        "def c(a, b):\n    i = h(a, b)\n    return np.median(i)\n"),
    "class_two_methods": (
        "class C:\n"
        "    def m1(self, a, b):\n        return a.match_to_catalog_sky(b)\n"
        "    def m2(self, d):\n        return np.median(d)\n"),
    "module_match_function_median": (
        "idx = a.match_to_catalog_sky(b)\n"
        "def f(d):\n    return np.median(d)\n"),
    "function_match_module_median": (
        "def f(a, b):\n    return a.match_to_catalog_sky(b)\n"
        "out = np.median(f(a, b))\n"),
    "decorator_line": (
        "@register(a.match_to_catalog_sky)\n"
        "def f(d):\n    return np.median(d)\n"),
}


@pytest.mark.parametrize("name", sorted(_EVASIONS))
def test_cross_function_split_still_trips(name):
    src = _EVASIONS[name]
    assert _MATCH.search(src) and _REDUCE.search(src), "file-level tripwire"
    # ...and it cannot be attributed to a function, so it reports as
    # <unattributed> and needs a whole-file entry rather than a per-function one
    assert not _offending_functions(src), name


@pytest.mark.parametrize("src,expect", [
    ("def f(a, b):\n    i = a.match_to_catalog_sky(b)\n    return np.median(i)\n", {"f"}),
    ("async def f(a, b):\n    i = a.match_to_catalog_sky(b)\n    return np.median(i)\n", {"f"}),
    ("def o(a, b):\n    def i2(c, d):\n        j = c.match_to_catalog_sky(d)\n"
     "        return np.median(j)\n    return i2(a, b)\n", {"o", "i2"}),
])
def test_same_function_is_attributed(src, expect):
    assert _offending_functions(src) == expect


def test_a_clean_file_does_not_trip():
    src = "def f(a, b):\n    return a.match_to_catalog_sky(b)\n"
    assert not (_MATCH.search(src) and _REDUCE.search(src))
    assert not _offending_functions(src)


#: Files whose ONLY protection is a whole-file exemption.  Each gave up
#: per-function coverage; do not add to this without reading the file.
_EXPECTED_UNATTRIBUTED = 8


def test_unattributed_entries_do_not_multiply():
    """A `<unattributed>` entry silences a whole file.  Eight is the number the
    tree needs today; a ninth means a file just lost per-function coverage and
    somebody should have noticed."""
    n = sum(1 for _f, fn in ALLOWLIST if fn == UNATTRIBUTED)
    assert n == _EXPECTED_UNATTRIBUTED, (
        f"{n} whole-file exemptions (expected {_EXPECTED_UNATTRIBUTED}). "
        f"Adding one trades away the cross-function tripwire for that file; "
        f"removing one is good news -- update the constant.")


def test_a_function_scoped_file_still_catches_a_new_split(tmp_path):
    """The gain the reversal actually buys, stated as a test rather than a
    claim: a file whose entries are all function-scoped fails on a NEW
    cross-function split, where the old file-level allowlist would not have.
    """
    src = ("def known(a, b):\n"
           "    i = a.match_to_catalog_sky(b)\n"
           "    return np.median(i)\n")
    assert _offending_functions(src) == {"known"}
    assert not _has_unattributed_cooccurrence(src, {"known"})
    # now add a split elsewhere in the same file
    src += ("\n\ndef _seps(a, b):\n"
            "    return a.match_to_catalog_sky(b)\n"
            "\n\ndef tie(a, b):\n"
            "    return np.median(_seps(a, b))\n")
    assert _has_unattributed_cooccurrence(src, _offending_functions(src)), (
        "a new cross-function split in a function-scoped file must trip")
