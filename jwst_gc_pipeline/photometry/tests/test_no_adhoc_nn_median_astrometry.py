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
    # sanctioned: histogram-stacking refinement, median only refines the peak
    ("jwst_gc_pipeline/photometry/measure_offsets.py", "measure_offsets"),
    ("jwst_gc_pipeline/reduction/build_virac2_offsets.py", "coord_shift"),
    # sanctioned: source ASSOCIATION for merging / saturated replacement, not a
    # correction -- the median is over fluxes/columns, not over offsets
    ("jwst_gc_pipeline/photometry/merge_catalogs.py", "combine_singleframe"),
    ("jwst_gc_pipeline/photometry/merge_catalogs.py", "replace_saturated"),
    # sanctioned: masking extended emission, no astrometry in it
    ("jwst_gc_pipeline/photometry/cataloging.py", "_filter_extended_emission"),
    # one-off scripts outside the pipeline's astrometric path
    ("scripts/reduction/combine_brick_allband.py", "main"),
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


def test_no_adhoc_nn_median_astrometry():
    offenders = []
    for rel, path in _iter_py_files():
        for func in sorted(_offending_functions(path.read_text(errors="replace"))):
            if (rel.as_posix(), func) not in ALLOWLIST:
                offenders.append(f"{rel.as_posix()}::{func}")
    assert not offenders, (
        "FORBIDDEN dense-NN-median astrometry pattern (a nearest-neighbour match "
        "reduced by median/mean, in the SAME function) found in "
        "non-allowlisted location(s):\n  "
        + "\n  ".join(sorted(offenders))
        + "\n\nUse jwst_gc_pipeline.photometry.astrometry_offsets.measure_offset "
        "(2D offset-histogram peak) instead. If this really is legitimate (source "
        "association for merging/dedup, or a histogram refinement -- NOT a dense-"
        "NN-median astrometric correction), add the (file, function) pair to "
        "ALLOWLIST in this test with a one-line justification, after a human "
        "review. See CLAUDE.md ASTROMETRY RULE #1.")


def test_allowlist_has_no_dead_entries():
    """An allowlist entry that no longer trips is rot, and rot is how a list of
    20 grows until nobody reads it.  Delete entries whose code moved or was
    fixed -- half of this list was dead when the function scoping landed."""
    live = set()
    for rel, path in _iter_py_files():
        for func in _offending_functions(path.read_text(errors="replace")):
            live.add((rel.as_posix(), func))
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
