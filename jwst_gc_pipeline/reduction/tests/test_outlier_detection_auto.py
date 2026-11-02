"""outlier_detection must run where JumpStep did not reject cosmic rays.

#161 skips Image3 ``outlier_detection`` on these crowded fields because JumpStep
already rejects cosmic rays per ramp.  JumpStep needs >= 3 groups, so NGROUPS=2
(BRIGHT2) exposures carry ``S_JUMP = SKIPPED``, and with the image-space step
skipped as well their crf reach cataloging with no cosmic-ray rejection.  cloudc
F212N re-reduced that way on 2026-10-09: sharp 1-2 pixel spikes in a fixed set
of 400 data-mosaic stamps rose 68 -> 413, and m7 catalogued and left them.

``resolve_skip_outlier_detection`` makes the default (``auto``) decide from the
members' ``S_JUMP``.  These tests execute the helper from its source against
real FITS headers written to tmp_path, and pin that both Image3 passes resolve
before they build their steps and write their crf.

The module is not imported: its top-level imports pull in ``jwst`` and
initialise CRDS (~50 s, and network).
"""
import ast
import os
import pathlib
import subprocess

import pytest
from astropy.io import fits


SRC = (pathlib.Path(__file__).resolve().parents[1]
       / "PipelineRerunNIRCAM-LONG.py")
SBATCH = (pathlib.Path(__file__).resolve().parents[3]
          / "scripts" / "reduction" / "submit_reduction.sbatch")

HELPER = "resolve_skip_outlier_detection"


def _tree():
    return ast.parse(SRC.read_text())


def _func_def(name):
    for node in ast.walk(_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() is gone from {SRC.name}")


@pytest.fixture(scope="module")
def resolve():
    node = _func_def(HELPER)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"os": os, "fits": fits, "print": lambda *a, **k: None}
    exec(compile(module, str(SRC), "exec"), ns)   # noqa: S102 - the repo's own source
    return ns[HELPER]


def _frame(path, **cards):
    hdr = fits.Header()
    for k, v in cards.items():
        hdr[k] = v
    fits.PrimaryHDU(header=hdr).writeto(path)
    return str(path)


# --------------------------------------------------------------------------
# what auto decides
# --------------------------------------------------------------------------

def test_ngroups2_members_turn_the_step_on(tmp_path, resolve):
    """The cloudc case: S_JUMP=SKIPPED on every member -> run the step."""
    paths = [_frame(tmp_path / f"m{i}.fits", S_JUMP="SKIPPED", NGROUPS=2)
             for i in range(3)]
    assert resolve(None, paths, "merged") is False


def test_jump_complete_members_keep_the_161_skip(tmp_path, resolve):
    """sgrc / arches: JumpStep ran -> skip, exactly as before this change."""
    paths = [_frame(tmp_path / f"m{i}.fits", S_JUMP="COMPLETE", NGROUPS=6)
             for i in range(3)]
    assert resolve(None, paths, "nrca") is True


def test_one_member_without_jump_rejection_turns_it_on(tmp_path, resolve):
    """The step compares each member to the stack, so the vote is association-wide."""
    paths = [_frame(tmp_path / "a.fits", S_JUMP="COMPLETE", NGROUPS=3),
             _frame(tmp_path / "b.fits", S_JUMP="SKIPPED", NGROUPS=2)]
    assert resolve(None, paths, "merged") is False


def test_a_missing_s_jump_counts_as_no_rejection(tmp_path, resolve):
    paths = [_frame(tmp_path / "a.fits", NGROUPS=2)]
    assert resolve(None, paths, "merged") is False


def test_an_explicit_request_is_returned_unchanged(tmp_path, resolve):
    """--outlier-detection=on/off (and --run-outlier-detection) override auto."""
    skipped = [_frame(tmp_path / "a.fits", S_JUMP="SKIPPED", NGROUPS=2)]
    complete = [_frame(tmp_path / "b.fits", S_JUMP="COMPLETE", NGROUPS=6)]
    assert resolve(True, skipped, "merged") is True
    assert resolve(False, complete, "merged") is False


def test_unreadable_members_are_left_out_of_the_vote(tmp_path, resolve):
    paths = [str(tmp_path / "gone.fits"),
             _frame(tmp_path / "b.fits", S_JUMP="COMPLETE", NGROUPS=6)]
    assert resolve(None, paths, "nrca") is True


def test_no_readable_member_keeps_the_161_default(tmp_path, resolve):
    assert resolve(None, [str(tmp_path / "gone.fits")], "nrca") is True


# --------------------------------------------------------------------------
# the wiring
# --------------------------------------------------------------------------

def test_main_defaults_to_auto():
    node = _func_def("main")
    names = [a.arg for a in node.args.args]
    default = node.args.defaults[names.index("skip_outlier_detection")
                                 - (len(names) - len(node.args.defaults))]
    assert isinstance(default, ast.Constant) and default.value is None, (
        "main(skip_outlier_detection=...) must default to None (auto); a True "
        "default strips cosmic-ray rejection from NGROUPS=2 fields")


def test_both_passes_resolve_before_building_steps_and_writing_crf():
    """Each Image3 pass resolves auto once, before either consumer reads the flag.

    image3_steps_for and write_perexposure_crf must see the SAME resolved value:
    write_perexposure_crf picks its crf source on it, and a crf copied from the
    member frames after outlier_detection ran would drop the step's flags.
    """
    main = _func_def("main")
    resolves = sorted(
        n.lineno for n in ast.walk(main)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name) and n.value.func.id == HELPER
        and any(isinstance(t, ast.Name) and t.id == "skip_outlier_detection"
                for t in n.targets))
    consumers = sorted(
        n.lineno for n in ast.walk(main)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id in ("image3_steps_for", "write_perexposure_crf"))
    assert len(resolves) == 2, (
        f"expected one {HELPER}() per Image3 pass (module, merged); found "
        f"{len(resolves)}")
    assert len(consumers) == 4
    # module pass: resolve < steps < crf < merged resolve < steps < crf
    first, second = resolves
    assert first < consumers[0] and consumers[1] < second < consumers[2], (
        f"{HELPER}() calls at {resolves} do not precede their pass's "
        f"image3_steps_for / write_perexposure_crf at {consumers}")


def test_cli_maps_the_policy_to_the_tristate():
    src = SRC.read_text()
    assert "\"--outlier-detection\"" in src
    assert "{'auto': None, 'on': False, 'off': True}" in src
    assert "default='auto'" in src


def test_sbatch_forwards_the_policy_and_defaults_to_auto():
    text = SBATCH.read_text()
    assert "OUTLIER_DETECTION=${OUTLIER_DETECTION:-auto}" in text
    assert '--outlier-detection="$OUTLIER_DETECTION"' in text


@pytest.mark.parametrize("value,rc", [("bogus", 2)])
def test_sbatch_refuses_an_unknown_policy(tmp_path, value, rc):
    """Run only the validation block, so a typo cannot reach the pipeline as auto."""
    text = SBATCH.read_text()
    i = text.index("OUTLIER_DETECTION=${OUTLIER_DETECTION:-auto}")
    j = text.index("esac", i) + len("esac")
    script = tmp_path / "check.sh"
    script.write_text(text[i:j] + "\necho ok\n")
    out = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                         env={**os.environ, "OUTLIER_DETECTION": value})
    assert out.returncode == rc
    out = subprocess.run(["bash", str(script)], capture_output=True, text=True,
                         env={**os.environ, "OUTLIER_DETECTION": "auto"})
    assert out.returncode == 0 and out.stdout.strip() == "ok"
