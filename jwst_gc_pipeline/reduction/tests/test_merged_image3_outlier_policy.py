"""The merged Image3 pass must follow the same outlier_detection policy as nrca/nrcb.

`PipelineRerunNIRCAM-LONG.py` runs Image3 twice over the same exposures: once per
module (nrca, nrcb) and once over the union (`merged`).  The per-module pass has
skipped `outlier_detection` since #161/#189 -- the step's tolerance is built from
`ERR`, a photon+read-noise model, while an undersampled PSF sampled at different
dither phases legitimately disperses 5-9x ERR wherever the PSF is steep, so it
flags real bright-star spikes and the dark inter-spike gaps as OUTLIER (PR #180).

The merged pass hand-rolled its own ``steps={'tweakreg': tweakreg_parameters}``
and so kept running the step at pipeline defaults, on every field, ignoring
``--run-outlier-detection`` in both directions.  On /orange that is visible in
the products: 178 of 217 `-merged_i2d.fits` carry ``S_OUTLIR = COMPLETE`` while
their `-nrca_i2d` / `-nrcb_i2d` siblings from the same run carry ``SKIPPED``.

These tests read the source and exercise the extracted helper rather than driving
Image3, which needs CRDS, real exposures and ~30 min.  What went wrong was that
one call site duplicated a policy instead of sharing it, and a duplicated call
site is exactly what source inspection can pin.

Importing the module itself is avoided on purpose: the file's top-level imports
pull in `jwst` and initialise CRDS (~50 s, and network), which would make a test
about a five-line policy decision depend on an STScI server being up.  The helper
is executed from its own source instead, which is the same code the pipeline runs.
"""
import ast
import pathlib

import pytest


SRC = (pathlib.Path(__file__).resolve().parents[1]
       / "PipelineRerunNIRCAM-LONG.py")

HELPER = "image3_steps_for"


def _tree():
    return ast.parse(SRC.read_text())


@pytest.fixture(scope="module")
def image3_steps_for():
    """The real helper, compiled out of the source file.

    `print` is rebound to a no-op collector so the helper's log lines do not
    need the module's own `print` wrapper (which is what forces the jwst import).
    """
    for node in ast.walk(_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == HELPER:
            break
    else:
        raise AssertionError(
            f"{HELPER}() is gone -- the outlier_detection policy is no longer "
            f"shared between the per-module and merged Image3 passes")
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"print": lambda *a, **k: None}
    exec(compile(module, str(SRC), "exec"), ns)   # noqa: S102 - the repo's own source
    return ns[HELPER]


def _image3_call_sites(tree):
    """Every ``calwebb_image3.Image3Pipeline.call(...)`` in the file."""
    sites = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if (isinstance(f, ast.Attribute) and f.attr == "call"
                and isinstance(f.value, ast.Attribute)
                and f.value.attr == "Image3Pipeline"):
            sites.append(node)
    return sites


# --------------------------------------------------------------------------
# what the helper decides
# --------------------------------------------------------------------------

def test_skip_outlier_detection_puts_the_skip_in_the_steps(image3_steps_for):
    tweakreg = {"skip": True}
    steps = image3_steps_for(tweakreg, True, "merged")
    assert steps["outlier_detection"] == {"skip": True}
    assert steps["tweakreg"] is tweakreg


def test_run_outlier_detection_leaves_the_step_at_pipeline_defaults(image3_steps_for):
    """`--run-outlier-detection` must mean "do not configure it", not "skip=False".

    Passing ``{'skip': False}`` would be equivalent here, but an absent key is
    what the per-module pass has always produced, and the point of this PR is
    that the two passes hand Image3 the same thing.
    """
    steps = image3_steps_for({"skip": True}, False, "nrca")
    assert "outlier_detection" not in steps


def test_the_helper_reproduces_the_policy_the_module_pass_used_inline(image3_steps_for):
    """The per-module pass must be untouched by the extraction.

    Independent re-statement of the pre-PR inline code, so this fails if the
    refactor changed the module pass rather than only the merged one.
    """
    for skip in (True, False):
        tweakreg = {"skip": True, "abs_refcat": "x.ecsv"}
        expected = {"tweakreg": tweakreg}
        if skip:
            expected["outlier_detection"] = {"skip": True}
        assert image3_steps_for(tweakreg, skip, "nrca") == expected


# --------------------------------------------------------------------------
# that BOTH call sites use it
# --------------------------------------------------------------------------

def test_every_image3_call_takes_its_steps_from_the_helper():
    """The guard that would have caught this.

    A call site that builds its own ``steps=`` dict is a second, silent copy of
    the policy -- which is how the merged pass spent from #189 to 2026-09 running
    a step its siblings skipped.
    """
    tree = _tree()
    sites = _image3_call_sites(tree)
    assert len(sites) >= 2, (
        f"expected the per-module and merged Image3 calls; found {len(sites)}")

    from_helper = {
        t.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == HELPER
        for t in node.targets
        if isinstance(t, ast.Name)
    }
    assert from_helper, f"nothing in the file is assigned from {HELPER}()"

    for site in sites:
        steps = [kw.value for kw in site.keywords if kw.arg == "steps"]
        assert steps, (
            f"Image3Pipeline.call at line {site.lineno} passes no steps= at all")
        value = steps[0]
        assert isinstance(value, ast.Name), (
            f"Image3Pipeline.call at line {site.lineno} builds its steps= inline "
            f"({type(value).__name__}) instead of calling {HELPER}(); the "
            f"outlier_detection policy (#161) is being duplicated again")
        assert value.id in from_helper, (
            f"Image3Pipeline.call at line {site.lineno} passes steps={value.id}, "
            f"which is not assigned from {HELPER}()")


def test_the_merged_call_passes_the_skip_flag_through():
    """The merged pass must consult `skip_outlier_detection`, not hard-code it.

    Hard-coding ``True`` there would make the two passes agree by default and
    still leave `--run-outlier-detection` unable to reach the merged mosaic.
    """
    tree = _tree()
    calls = [n for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
             and n.func.id == HELPER]
    assert len(calls) >= 2, f"{HELPER}() is called {len(calls)} time(s), expected >= 2"
    for c in calls:
        names = {a.id for a in c.args if isinstance(a, ast.Name)}
        assert "skip_outlier_detection" in names, (
            f"{HELPER}() call at line {c.lineno} does not forward "
            f"skip_outlier_detection, so --run-outlier-detection cannot reach it")


def test_the_merged_pass_still_leaves_skymatch_alone():
    """Scope guard: this PR changes outlier_detection and nothing else.

    `--skymatch-method` has always configured the per-module pass only.  Folding
    skymatch into the shared helper would quietly start applying it to the merged
    mosaic too -- a separate, unmeasured product change riding along.
    """
    src = SRC.read_text()
    i = src.index("asn_file_merged,\n            steps=")
    j = src.index("DONE running Image3Pipeline", i)
    assert "skymatch" not in src[i - 1500:j], (
        "the merged Image3 pass has grown a skymatch configuration; that is a "
        "separate product change and does not belong in this PR")


def test_the_reason_the_two_passes_must_agree_is_recorded():
    """Whoever touches this next needs #161 in front of them at both sites."""
    src = SRC.read_text()
    helper_at = src.index(f"def {HELPER}(")
    assert "#161" in src[helper_at:helper_at + 4000]
    merged_at = src.index("asn_file_merged,\n            steps=")
    assert "#161" in src[merged_at - 1500:merged_at]
