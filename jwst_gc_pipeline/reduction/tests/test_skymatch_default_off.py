"""skymatch stays OFF by default, and off is a recorded decision.

Maintainer, 2026-09-06 (#419), asked before the first program-10678 treasury
reduce: "Leave skymatch off.  We will only turn it on given strong evidence for
the need."  The three layers that could turn it on -- the driver's
``--skymatch-method`` default, ``submit_reduction.sbatch``'s ``SKYMATCH``, and
data-qa's trigger -- all default to empty, which reads as an unset knob unless
something says otherwise.  These tests are that something: they fail if the
default moves, so a future flip has to be deliberate and has to restate the
evidence.

The measured cost of leaving it off, 2026-08-25 (#419), as the sigma-clipped
median of every ``_cal`` SCI frame in a filter directory relative to that set's
own median sky: F480M 34-53% in every field that has the band, F212N 75-159%,
and sickle F470N 378% (range ~99 on a 26.3 MJy/sr sky) -- the worked example of
what riding the default looks like assembled.

Source inspection rather than a reduction: running Image3 needs CRDS, real
exposures and ~30 min, and what is being pinned is a default value and a written
rationale, which is exactly what source inspection can pin.
"""
import ast
import pathlib

REDUCTION = pathlib.Path(__file__).resolve().parents[1]
SRC = REDUCTION / "PipelineRerunNIRCAM-LONG.py"
SBATCH = (REDUCTION.parents[1] / "scripts" / "reduction"
          / "submit_reduction.sbatch")


def _skymatch_option():
    """The ``parser.add_option("--skymatch-method", ...)`` call."""
    tree = ast.parse(SRC.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_option"):
            continue
        if any(isinstance(arg, ast.Constant) and arg.value == "--skymatch-method"
               for arg in node.args):
            return node
    raise AssertionError("could not find the --skymatch-method option")


def _keyword(call, name):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    raise AssertionError(f"--skymatch-method has no {name}= keyword")


def test_driver_default_is_off():
    """An empty default is what collapses to ``skymatch_method=None``."""
    default = _keyword(_skymatch_option(), "default")
    assert isinstance(default, ast.Constant), (
        "--skymatch-method's default is no longer a literal; #419 decided it "
        "stays empty (skymatch off) until strong evidence of need")
    assert default.value == "", (
        f"--skymatch-method now defaults to {default.value!r}. #419 decided "
        "skymatch stays OFF: 'We will only turn it on given strong evidence "
        "for the need.'  If that evidence exists, record it here and in the "
        "driver's skymatch block before changing this.")


def test_driver_option_records_the_decision():
    """The help text has to say off is a decision, or the next reader re-asks."""
    help_text = _keyword(_skymatch_option(), "help")
    text = "".join(part.value for part in ast.walk(help_text)
                   if isinstance(part, ast.Constant)
                   and isinstance(part.value, str))
    assert "#419" in text, (
        "--skymatch-method's help no longer cites #419, where leaving skymatch "
        "off was decided")
    assert "DECISION" in text, (
        "--skymatch-method's help no longer states that off is a decision; "
        "without that it reads as a knob nobody got round to setting")


def test_driver_skymatch_block_carries_the_rationale():
    """The block where the question arises states the bar and the cost."""
    text = SRC.read_text()
    block = text[text.index("# skymatch: OFF by default"):]
    block = block[:block.index("image3_steps = {")]
    assert "#419" in block and "2026-09-06" in block, (
        "the skymatch block no longer dates or cites the decision")
    assert "strong evidence" in block, (
        "the skymatch block no longer states the bar for turning skymatch on")
    for measurement in ("378%", "34- 53%", "75-159%"):
        assert measurement in block, (
            f"the skymatch block no longer quotes {measurement}, part of the "
            "measured cost of leaving skymatch off (#419, 2026-08-25)")


def test_sbatch_default_is_empty_and_says_why():
    """``SKYMATCH=${SKYMATCH:-}`` is the submit-layer half of the same default."""
    text = SBATCH.read_text()
    assert "SKYMATCH=${SKYMATCH:-}\n" in text, (
        "submit_reduction.sbatch no longer defaults SKYMATCH to empty; #419 "
        "decided skymatch stays off until strong evidence of need")
    assert "#419" in text, (
        "submit_reduction.sbatch no longer cites #419 beside SKYMATCH, so the "
        "empty default reads as an oversight")
