"""The merged Image3 pass must write the per-exposure crf cataloging photometers.

Cataloging fits the per-exposure ``*_{align,destreak}_o{field}_crf.fits`` frames
(``--each-suffix``), not the mosaic.  Who writes them was, until 2026-09, an
accident of which Image3 steps ran:

``outlier_detection`` in this jwst version names its output after each INPUT
MODEL, not after the asn product, so on a pass that ran it the step wrote those
exact per-exposure names itself.  The merged pass ran it (the defect #848 fixes)
and runs LAST in ``nrca,nrcb,merged``, so it authored every crf on disk --
measured on brick 2221 F410M o001: 48/48 carry
``ASNTABLE=...merged_asn.json`` and ``S_OUTLIR=COMPLETE``.

Skipping the step on the merged pass removes that incidental author.  Without a
crf block of its own the merged branch would then write NONE, and ``-m merged``
-- the NIRCam stage-1 invocation GETTING_STARTED.md gives -- would emit zero crf
and silently leave the PREVIOUS reduction's in place: old WCS, untouched mtime,
invisible to every staleness check in the tree.  That is the #270 sickle failure
that ``test_crf_source_branch_order.py`` exists to prevent, arriving through a
different door.

So both passes route through one ``write_perexposure_crf()``.  These tests pin
the wiring (source inspection: what went wrong was a call site that did not
exist) and the helper's behaviour (executed for real against temporary files).

The module is not imported: its top-level imports pull in ``jwst`` and
initialise CRDS (~50 s, and network).  The helper is compiled out of the source
file instead, which is the same code the pipeline runs.
"""
import ast
import os
import pathlib
import shutil

import pytest


SRC = (pathlib.Path(__file__).resolve().parents[1]
       / "PipelineRerunNIRCAM-LONG.py")

HELPER = "write_perexposure_crf"


def _tree():
    return ast.parse(SRC.read_text())


def _func_def(name):
    for node in ast.walk(_tree()):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name}() is gone from {SRC.name}")


@pytest.fixture
def write_perexposure_crf():
    """The real helper, compiled out of the source file.

    Only the module-level names the helper actually touches are supplied, so the
    fixture cannot accidentally make a missing import look fine.
    """
    from glob import glob
    from astropy.io import fits

    node = _func_def(HELPER)
    module = ast.Module(body=[node], type_ignores=[])
    ast.fix_missing_locations(module)
    ns = {"os": os, "glob": glob, "shutil": shutil, "fits": fits,
          "print": lambda *a, **k: None}
    exec(compile(module, str(SRC), "exec"), ns)   # noqa: S102 - the repo's own source
    return ns[HELPER]


def _members(paths):
    return {"products": [{"name": "prod", "members": [{"expname": p} for p in paths]}]}


# --------------------------------------------------------------------------
# the wiring: BOTH Image3 passes must author their crf
# --------------------------------------------------------------------------

def _branch_containing(lineno):
    """Which top-level ``if`` of main() a line belongs to: 'module' or 'merged'."""
    tree = _tree()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        consts = {n.value for n in ast.walk(node.test) if isinstance(n, ast.Constant)}
        if "module" not in names:
            continue
        lo, hi = node.lineno, max(getattr(n, "lineno", node.lineno)
                                  for n in ast.walk(node))
        if lo <= lineno <= hi:
            if "merged" in consts and "nrca" not in consts:
                return "merged"
            if "nrca" in consts:
                return "module"
    return None


def _helper_calls():
    return [n for n in ast.walk(_tree())
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
            and n.func.id == HELPER]


def test_both_image3_passes_write_their_per_exposure_crf():
    """The guard that would have caught it.

    Every Image3 pass this file runs has to leave per-exposure crf behind.  A
    pass that runs Image3 and writes none does not fail: it leaves the previous
    reduction's crf, and cataloging photometers those (#270).
    """
    calls = _helper_calls()
    assert len(calls) >= 2, (
        f"{HELPER}() is called {len(calls)} time(s); both the per-module and the "
        f"merged Image3 pass must write their crf")
    branches = {_branch_containing(c.lineno) for c in calls}
    assert "merged" in branches, (
        f"no {HELPER}() call inside the `module == 'merged'` branch: a "
        f"`-m merged` run (GETTING_STARTED.md's NIRCam stage-1 invocation) "
        f"would write zero crf and silently keep the previous reduction's")
    assert "module" in branches, (
        f"no {HELPER}() call inside the `module in ('nrca','nrcb')` branch")


def test_the_crf_calls_forward_the_outlier_flag_by_keyword():
    """`--run-outlier-detection` decides which crf SOURCE is correct.

    With the step running, jwst writes the crf and the helper must not overwrite
    them; with it skipped, the aligned member frames are the only correct
    source.  Hard-coding either side here re-creates #270.  Checked by keyword:
    positionally, `skip_outlier_detection` and `skymatch_method` are adjacent
    and transposing them type-checks.
    """
    for c in _helper_calls():
        kw = {k.arg: k.value for k in c.keywords if k.arg}
        assert "skip_outlier_detection" in kw, (
            f"{HELPER}() call at line {c.lineno} does not pass "
            f"skip_outlier_detection by keyword")
        v = kw["skip_outlier_detection"]
        assert isinstance(v, ast.Name) and v.id == "skip_outlier_detection", (
            f"{HELPER}() call at line {c.lineno} hard-codes the flag "
            f"({ast.dump(v)}); --run-outlier-detection cannot reach it")
        assert "label" in kw


def test_the_crf_block_is_not_duplicated():
    """One authoring point, for the same reason `image3_steps_for` is one.

    The merged pass ran a step its siblings skipped because a second call site
    re-stated the policy instead of sharing it.  Copying the crf block into the
    merged branch would set that up again -- the next #270-class fix would land
    in one copy.
    """
    src = SRC.read_text()
    assert src.count("shutil.copy(_src, _target)") == 1, (
        "the per-exposure crf copy appears more than once; route both Image3 "
        f"passes through {HELPER}() instead of duplicating the block")
    assert src.count("_prod_crf = sorted(") == 1


def test_the_merged_crf_is_written_after_the_merged_image3_call():
    """Order: Image3 first, crf after.

    On a `--run-outlier-detection` merged run, jwst writes the crf during
    Image3; the helper must see the result, not pre-empt it.
    """
    src = SRC.read_text()
    i = src.index("asn_file_merged,\n            steps=")
    j = src.index(f"{HELPER}(asn_data, output_dir, field", i)
    assert j > i
    assert "'merged'" in src[j:j + 400]


# --------------------------------------------------------------------------
# the helper's behaviour, run for real
# --------------------------------------------------------------------------

def test_a_skipped_run_writes_one_crf_per_member(tmp_path, write_perexposure_crf):
    members = []
    for i in (1, 2, 3):
        f = tmp_path / f"jw02221001001_07101_0000{i}_nrcalong_destreak.fits"
        f.write_bytes(f"member-{i}".encode())
        members.append(str(f))

    write_perexposure_crf(_members(members), str(tmp_path), "001",
                          skip_outlier_detection=True, skymatch_method=None,
                          label="merged")

    for i, m in enumerate(members, start=1):
        crf = pathlib.Path(m.replace(".fits", "_o001_crf.fits"))
        assert crf.exists(), f"{crf.name} was not written"
        assert crf.read_bytes() == f"member-{i}".encode()


def test_the_crf_is_a_copy_not_an_alias(tmp_path, write_perexposure_crf):
    """A separate inode, because the member is rewritten after the crf exists.

    The full argument is at the copy itself and in
    tests/test_crf_copy_is_not_a_link.py; this pins that routing the merged pass
    through the helper did not turn its crf into an alias.
    """
    m = tmp_path / "jw02221001001_07101_00001_nrcalong_destreak.fits"
    m.write_bytes(b"aligned")
    write_perexposure_crf(_members([str(m)]), str(tmp_path), "001",
                          skip_outlier_detection=True, skymatch_method=None,
                          label="merged")
    crf = tmp_path / "jw02221001001_07101_00001_nrcalong_destreak_o001_crf.fits"
    assert os.stat(crf).st_ino != os.stat(m).st_ino
    assert not crf.is_symlink()
    m.write_bytes(b"rewritten by the next fix_alignment")
    assert crf.read_bytes() == b"aligned"


def test_a_stale_product_crf_is_declined_not_copied_forward(tmp_path,
                                                            write_perexposure_crf):
    """#270, reached through the merged branch.

    Product-named crf in output_dir on a skipped run are an EARLIER reduction's
    and carry its WCS.  The helper must write from this run's members anyway.
    """
    m = tmp_path / "jw02221001001_07101_00001_nrcalong_destreak.fits"
    m.write_bytes(b"this run, aligned")
    stale = tmp_path / "prod_00001_o001_crf.fits"
    stale.write_bytes(b"JUNE REDUCTION, OLD WCS")

    write_perexposure_crf(_members([str(m)]), str(tmp_path), "001",
                          skip_outlier_detection=True, skymatch_method=None,
                          label="merged")

    crf = tmp_path / "jw02221001001_07101_00001_nrcalong_destreak_o001_crf.fits"
    assert crf.read_bytes() == b"this run, aligned", (
        "the stale product crf was copied forward -- #270")


def test_an_outlier_detection_run_leaves_jwsts_own_crf_alone(tmp_path,
                                                             write_perexposure_crf):
    """With the step running, jwst already wrote the per-exposure crf.

    In this jwst version outlier_detection names its output after each input
    model, so no product-named crf exist to map and the helper must not
    overwrite what the step produced with an unflagged copy of the member.
    """
    m = tmp_path / "jw02221001001_07101_00001_nrcalong_destreak.fits"
    m.write_bytes(b"member, no OUTLIER flags")
    crf = tmp_path / "jw02221001001_07101_00001_nrcalong_destreak_o001_crf.fits"
    crf.write_bytes(b"written by outlier_detection, OUTLIER flags set")

    write_perexposure_crf(_members([str(m)]), str(tmp_path), "001",
                          skip_outlier_detection=False, skymatch_method=None,
                          label="merged")

    assert crf.read_bytes() == b"written by outlier_detection, OUTLIER flags set"


def test_a_missing_member_is_reported_and_does_not_abort(tmp_path,
                                                         write_perexposure_crf):
    """One missing frame must not cost the other 47 their crf."""
    present = tmp_path / "jw02221001001_07101_00001_nrcalong_destreak.fits"
    present.write_bytes(b"here")
    absent = str(tmp_path / "jw02221001001_07101_00002_nrcalong_destreak.fits")

    write_perexposure_crf(_members([absent, str(present)]), str(tmp_path), "001",
                          skip_outlier_detection=True, skymatch_method=None,
                          label="merged")

    assert (tmp_path / "jw02221001001_07101_00001_nrcalong_destreak_o001_crf.fits").exists()
    assert not os.path.exists(absent.replace(".fits", "_o001_crf.fits"))
