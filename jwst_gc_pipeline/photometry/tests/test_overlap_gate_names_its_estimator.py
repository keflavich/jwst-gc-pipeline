"""CLAUDE.md's release-gate sentence must name the estimator the gate calls.

Issue #392. `CLAUDE.md` named `astrometry_offsets.measure_offset_grid(...,
max_off_mas=...)` as "the offset-magnitude gate", and:

* the release gate stopped using that estimator in 0fb1958 (2026-07-13) --
  `check_interframe_overlap.py` per-tile layer is `overlap_offset_grid`;
* `max_off_mas` is passed non-None nowhere outside the test suite, so with the
  parameter left None `off_ok` is unconditionally True;
* both `check_interframe_overlap.py` and `interframe_overlap.py` imported
  `measure_offset_grid` and never called it, which is what made the sentence
  look current.

An unused import is what let the documentation and the code drift apart while
each still mentioned the other, so the import is what this pins.
"""
import ast
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
GATE_SCRIPT = os.path.join(REPO, "scripts", "release",
                           "check_interframe_overlap.py")
OVERLAP_MODULE = os.path.join(REPO, "jwst_gc_pipeline", "photometry",
                              "interframe_overlap.py")
CLAUDE_MD = os.path.join(REPO, "CLAUDE.md")


def _imported_names(path):
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.Import):
            names.update((a.asname or a.name).split(".")[0] for a in node.names)
    return names


def _called_names(path):
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    called = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name):
                called.add(fn.id)
            elif isinstance(fn, ast.Attribute):
                called.add(fn.attr)
    return called


def _release_gate_paragraph():
    """The CLAUDE.md sentence that names the gate, and what follows it."""
    with open(CLAUDE_MD) as fh:
        text = fh.read()
    start = text.index("The real gate:")
    end = text.index("\n---", start)
    return text[start:end]


def test_the_gate_script_does_not_import_an_estimator_it_never_calls():
    imported = _imported_names(GATE_SCRIPT)
    called = _called_names(GATE_SCRIPT)
    assert "overlap_offset_grid" in called, \
        "the per-tile layer is overlap_offset_grid; if that changed, so must " \
        "CLAUDE.md's release-gate sentence"
    assert "measure_offset_grid" not in imported, (
        "check_interframe_overlap.py imports measure_offset_grid without "
        "calling it. That dead import is what made CLAUDE.md name it as the "
        "gate (issue #392) -- if the gate is going back to it, call it.")


def test_interframe_overlap_does_not_import_an_estimator_it_never_calls():
    imported = _imported_names(OVERLAP_MODULE)
    called = _called_names(OVERLAP_MODULE)
    assert "measure_offset_grid" not in imported or \
        "measure_offset_grid" in called, (
            "interframe_overlap.py imports measure_offset_grid and never "
            "calls it (issue #392)")
    assert "local_residual_map" in called, \
        "the cell layer of the overlap gate is local_residual_map"


def test_claude_md_names_the_estimator_the_gate_actually_uses():
    para = _release_gate_paragraph()
    # the sentence that states the gate, before the note explaining the change
    stated = para.split("This sentence used to name")[0]
    assert "overlap_offset_grid" in stated
    assert "tol_mas" in stated
    assert not re.search(r"measure_offset_grid\(\.\.\.,\s*max_off_mas", stated), (
        "CLAUDE.md's release-gate sentence names measure_offset_grid's "
        "max_off_mas as the gate. No production caller passes it, and the "
        "gate's magnitude limit is overlap_offset_grid's tol_mas (issue #392).")


def test_max_off_mas_is_still_documented_as_unused_in_production():
    """The parameter is kept, so the docstring must not tell a reader to pass
    it "for any release/QC sign-off" while no caller does."""
    from jwst_gc_pipeline.photometry import astrometry_offsets
    doc = astrometry_offsets.measure_offset_grid.__doc__
    assert "No production caller passes it" in doc


# ---------------------------------------------------------------------------
# The docstring must name the caller that actually calls it (issue #392)
# ---------------------------------------------------------------------------

MONITORING = os.path.join(REPO, "jwst_gc_pipeline", "monitoring")
VISIT_CONSENSUS = os.path.join(REPO, "jwst_gc_pipeline", "photometry",
                               "visit_consensus.py")


def test_monitoring_does_not_call_measure_offset_grid():
    """`monitoring/` mentions `measure_offset_grid` in comments and prose; it
    reads the per-tile map out of a checkpoint record instead.  If that ever
    changes, the docstring below has to change with it."""
    for name in sorted(os.listdir(MONITORING)):
        if not name.endswith(".py"):
            continue
        assert "measure_offset_grid" not in _called_names(
            os.path.join(MONITORING, name)), (
                f"monitoring/{name} now CALLS measure_offset_grid; "
                "measure_offset_grid's docstring says monitoring is not a "
                "caller (issue #392)")


def test_visit_consensus_is_the_checkpoint_caller():
    """`measure_reference_tie` is the m2-m6 caller, and its `clean` is
    `per_tile_ok`, a term of `apply_ok`."""
    assert "measure_offset_grid" in _called_names(VISIT_CONSENSUS)
    src = open(VISIT_CONSENSUS).read()
    assert re.search(r"per_tile_ok\s*=\s*bool\(grid\.get\(\"clean\"\)\)", src), (
        "measure_offset_grid's `clean` is no longer read as `per_tile_ok`; "
        "its docstring says it is (issue #392)")
    assert re.search(r"apply_ok\s*=.*per_tile_ok", src, re.S)


def test_docstring_names_the_real_caller_not_the_monitoring_scan():
    from jwst_gc_pipeline.photometry import astrometry_offsets
    doc = astrometry_offsets.measure_offset_grid.__doc__
    assert "visit_consensus.measure_reference_tie" in doc, (
        "the docstring must name the caller that calls it (issue #392)")
    assert "apply_ok" in doc, (
        "the consequence of leaving max_off_mas None is that `clean` feeds "
        "`apply_ok`; the docstring must say so")
    assert not re.search(r"remaining production caller[^.]*monitoring", doc), (
        "the docstring attributed the call to `monitoring/`, which contains "
        "the string only in comments and prose (issue #392)")


def test_docstring_bounds_the_census_it_quotes():
    """The census counts `clean` maps whose worst CELL is large.  Most of those
    cells were measured at a swept window, which is the per-tile noise/geometry
    regime of issue #158 -- so the count bounds what `clean` does not say and
    does not measure how many fields are misregistered.  Quoting the count
    without that qualification invites the stronger reading."""
    from jwst_gc_pipeline.photometry import astrometry_offsets
    doc = astrometry_offsets.measure_offset_grid.__doc__
    marker = "above 30 mas"
    assert marker in doc, "the #392 census is gone from the docstring"
    tail = doc.split(marker, 1)[1]
    assert re.search(r"\bSWEPT\b", tail), (
        "the docstring quotes the per-tile census without saying that most of "
        "those worst cells are SWEPT per-tile peaks (issue #158), which reads "
        "as a claim that that many fields are misregistered")
    assert "#158" in tail, (
        "name the issue that explains a swept per-tile peak (#158)")
