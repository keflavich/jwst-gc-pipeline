"""A skip_outlier_detection run must never copy an OLDER run's product crf.

``outlier_detection`` is the only Image3 step that emits product-named crf
(``jw<prop:05d>-o<field>_t001_nircam_clear-<filt>-<module>_<N>_o<field>_crf.fits``).
With it skipped (#161) the current run writes none, so any on disk are leftovers
from a previous reduction -- carrying that reduction's WCS.

The CRF-naming block copied them forward whenever the glob matched, which
refreshes their mtime while leaving a previous generation's alignment inside.
Nothing downstream can see that: every staleness check in the tree is
mtime-based.

sickle (#270) is the case that exposed it. 96 product crf from 2026-06-27 -- its
last run with outlier_detection enabled -- were copied over the per-exposure
names on every iteration of the VIRAC2 re-tie, so all 96 carried a single
constant GNS ``RAOFFSET`` while their aligned ``_destreak.fits`` inputs carried
the new per-exposure VIRAC2 tie ~200 mas away. m2 re-measured the same ~110 mas
gap each iteration and the loop could not converge.

These tests read the source rather than driving Image3, which needs CRDS, real
exposures and ~30 min. What went wrong was a BRANCH ORDER, and branch order is
exactly what source inspection can pin.
"""
import ast
import pathlib


SRC = (pathlib.Path(__file__).resolve().parents[1]
       / "PipelineRerunNIRCAM-LONG.py")


def _crf_branch():
    """The ``if _prod_crf ...`` chain that chooses the per-exposure crf source."""
    tree = ast.parse(SRC.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        if "_prod_crf" in names:
            return node
    raise AssertionError("could not find the _prod_crf branch")


def test_product_crf_are_only_used_when_outlier_detection_ran():
    """The copy-forward branch must be guarded by ``not skip_outlier_detection``.

    Without the guard, a run that emitted no product crf still copies whatever
    older ones are lying in output_dir.
    """
    branch = _crf_branch()
    names = {n.id for n in ast.walk(branch.test) if isinstance(n, ast.Name)}
    assert "skip_outlier_detection" in names, (
        "the product-crf copy branch does not consult skip_outlier_detection -- "
        "it will copy an older reduction's crf forward whenever any are on disk")

    # and specifically NEGATED: `_prod_crf and not skip_outlier_detection`
    negated = any(isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not)
                  and any(isinstance(x, ast.Name)
                          and x.id == "skip_outlier_detection"
                          for x in ast.walk(n))
                  for n in ast.walk(branch.test))
    assert negated, (
        "skip_outlier_detection appears in the test but is not negated; the copy "
        "branch must fire only when outlier_detection actually ran")


def test_skip_outlier_detection_still_has_its_own_crf_source():
    """The fallback that copies the aligned member frames must survive.

    It is the ONLY correct source on a skip_outlier_detection run -- and it has
    to be reachable even when product crf are present, which is precisely the
    stale case.
    """
    branch = _crf_branch()
    tests = []
    node = branch
    while node.orelse and len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
        node = node.orelse[0]
        tests.append(node.test)
    assert any(isinstance(t, ast.Name) and t.id == "skip_outlier_detection"
               for t in tests), (
        "the elif that writes crf from the aligned member frames is gone")


def test_the_stale_case_is_reported_not_silent():
    """Declining stale product crf must SAY so.

    Silently doing the right thing here is nearly as bad as doing the wrong
    thing: the 96 files stay on disk looking like current products, and the next
    person to grep for them has no note explaining why they are ignored.
    """
    src = SRC.read_text()
    assert "EARLIER reduction's and carry its WCS" in src, (
        "no message when stale product crf are declined")


def test_the_sickle_incident_is_recorded_at_the_branch():
    """Whoever reorders this next needs the reason in front of them."""
    src = SRC.read_text()
    i = src.index("_prod_crf = sorted(")
    context = src[max(0, i - 2000):i]
    assert "#270" in context, "the branch lost its pointer to the incident"
    assert "ORDER MATTERS" in context
