"""The #697 per-detector decision, pinned against the code and the report it cites.

PR #809 records the maintainer's 2026-09-06 answer -- a per-detector CALIBRATION
TERM does not go into the offsets table -- in the docstrings where the
granularity is decided.  Prose drifts from the thing it describes, and the first
draft of that prose did drift twice: it said the pooler is where the term is
disposed of "for good" (it runs on the module-LOCKED channel only, so on the
``consensus`` channel nothing is pooled and every detector keeps its own row),
and it quoted figures from the wrong run of ``reports/per_detector_offsets.md``.

These tests pin BOTH sides of each claim, in the idiom of
``test_astrometry_docs_match_code.py``: the code fact, and the sentence that
describes it.
"""
import ast
import os
import re

from jwst_gc_pipeline.photometry.astrometry_checkpoint import (
    pool_corrections_to_table_granularity)
from jwst_gc_pipeline.reduction.build_virac2_offsets import module_key

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
CATALOGING = os.path.join(REPO, "jwst_gc_pipeline", "photometry", "cataloging.py")
CHECKPOINTS_MD = os.path.join(REPO, "jwst_gc_pipeline", "photometry",
                              "ASTROMETRY_CHECKPOINTS.md")
REPORT = os.path.join(REPO, "reports", "per_detector_offsets.md")

POOLER = "pool_corrections_to_table_granularity"


def _read(path):
    with open(path) as fh:
        return fh.read()


def _calls(node, name):
    return {n.lineno for n in ast.walk(node)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == name}


def _is_locked_test(test):
    """``_channel == 'locked'``, however it is spelled around the operands."""
    if not isinstance(test, ast.Compare) or not isinstance(test.left, ast.Name):
        return False
    return (test.left.id == "_channel"
            and any(isinstance(op, ast.Eq) for op in test.ops)
            and any(isinstance(c, ast.Constant) and c.value == "locked"
                    for c in test.comparators))


def test_the_pooler_is_reached_only_on_the_locked_channel():
    """The code fact the docstring's scope sentence describes.

    A consensus table already keys every detector, so pooling there would
    collapse resolution the table has -- which is why ``cataloging.py`` guards
    the one production call.  Drop the guard and the docstring's "the module-
    LOCKED channel, the only one whose rows are coarser than the corrections"
    becomes false.
    """
    tree = ast.parse(_read(CATALOGING))
    everywhere = _calls(tree, POOLER)
    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and _is_locked_test(node.test):
            for stmt in node.body:
                guarded |= _calls(stmt, POOLER)
    assert everywhere, f"no call to {POOLER} in cataloging.py at all"
    assert everywhere == guarded, (
        f"{POOLER} is called at lines {sorted(everywhere - guarded)} of "
        f"cataloging.py outside a `_channel == 'locked'` branch; the pooler's "
        f"docstring and ASTROMETRY_CHECKPOINTS.md both scope the per-detector "
        f"discard to that channel")


def test_the_pooler_docstring_scopes_the_discard_to_that_channel():
    """The prose side: the discard is not described as unconditional."""
    doc = pool_corrections_to_table_granularity.__doc__
    assert "_channel == 'locked'" in doc, (
        "the docstring must name the branch that reaches this function")
    assert "seed_offsets_table_from_consensus" in doc, (
        "the docstring must say what the consensus channel does instead -- it "
        "writes the per-detector corrections unpooled, one row per detector")


def test_the_checkpoints_md_bullet_scopes_it_too():
    md = _read(CHECKPOINTS_MD)
    bullet = md.split("MAX_POOL_SPREAD_MAS")[0]
    assert "_channel == 'locked'" in bullet, (
        "the pooling bullet describes the same discard and must carry the same "
        "scope")


def _report_runs():
    """(date, measurement count) for each run the report states."""
    report = _read(REPORT)
    runs = re.findall(r"(20\d\d-\d\d-\d\d)[^\n]*?(\d{2},\d{3})[^\n]*?measurements",
                      report)
    assert len(runs) >= 2, f"could not read the report's runs: {runs}"
    return runs


def _decision_docs():
    return {"pool_corrections_to_table_granularity":
            pool_corrections_to_table_granularity.__doc__,
            "build_virac2_offsets.module_key": module_key.__doc__}


def test_every_report_figure_quoted_in_code_comes_from_the_report():
    report = _read(REPORT)
    for where, doc in _decision_docs().items():
        for num in re.findall(r"\b\d{2},\d{3}\b", doc):
            assert num in report, (
                f"{where} quotes {num} as a figure from "
                f"reports/per_detector_offsets.md, which does not contain it")


def test_a_quoted_run_carries_the_date_of_the_run_it_came_from():
    """The report holds two runs; their totals differ by a factor of two.

    Quoting 73,673 (the 2026-08-25 de-rotated re-run) beside the ON-SKY finding,
    which is the 2026-08-07 run's 34,672, attributes a measurement to a run that
    did not make it.  Requiring the date beside the count keeps them together.
    """
    for where, doc in _decision_docs().items():
        for date, count in _report_runs():
            if count in doc:
                assert date in doc, (
                    f"{where} quotes the {count}-measurement run without its "
                    f"date {date}; the report holds two runs of different size "
                    f"and the finding belongs to one of them")


def test_the_sigma_figure_is_the_one_the_report_states():
    report = _read(REPORT)
    m = re.search(r"consistent with zero at ≤([0-9.]+)σ", report)
    assert m, "could not find the report's pooled-significance sentence"
    stated = m.group(1)
    for where, doc in _decision_docs().items():
        for val in re.findall(r"([0-9]*\.[0-9]+)\s*sigma", doc):
            assert val == stated, (
                f"{where} says {val} sigma; the report says {stated}")
