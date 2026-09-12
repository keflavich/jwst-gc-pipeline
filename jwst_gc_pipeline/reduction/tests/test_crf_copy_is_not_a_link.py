"""The per-exposure crf must be a COPY of its member frame, never a link.

On the ``skip_outlier_detection`` path (#161, the NIRCam default) each
per-exposure ``*_o<field>_crf.fits`` is written by copying the aligned member
frame -- 117 MB apiece, ~5.6 GB per module pass.  A hard link or a rename would
make that free, and the size is the obvious reason to reach for one.  It is not
safe, because the member frame is MUTATED after the crf exists, by three writers
with three different inode semantics:

* ``shutil.copyfile(cal, align)`` -- the merged pass re-derives ``_align.fits``
  from ``_cal.fits`` for the same exposures the nrca/nrcb passes already wrote
  crf for.  ``copyfile`` opens the destination ``'wb'``, so it TRUNCATES the
  existing inode in place.  A hard-linked crf would silently become the
  unaligned ``_cal`` content -- the astrometry failure this repo keeps paying
  for, with no mtime change on the crf to show for it.
* ``fits.open(fn, mode='update')`` -- ``dva_correction.apply_dva_correction``,
  ``versioning.stamping._mirror_keys`` and the placement/filter corrections all
  edit the member's headers in place.  A hard-linked crf inherits every one of
  those edits.
* ``HDUList.writeto(fn, overwrite=True)`` / ``DataModel.save`` -- astropy
  unlinks first, so this one BREAKS the link instead of following it.

Between them the crf would sometimes track its member and sometimes fork from
it, decided by which writer happened to touch the file last.  A symlink is worse
still: it follows the path across a whole re-reduction.

``os.rename`` is not available either -- the member frame is read after the crf
is written (``check_wcs``, the merged pass, cataloging's ``--each-suffix``), so
moving it would delete an input.

The 4 s the copy costs (~11 GB of Lustre traffic per reduce at ~1.3 GB/s) buys
a crf that is a snapshot.  These tests read the source rather than driving
Image3, which needs CRDS, real exposures and ~30 min.  What is being guarded is
a CALL CHOICE at one site, and that is what source inspection can pin.
"""
import ast
import pathlib


SRC = (pathlib.Path(__file__).resolve().parents[1]
       / "PipelineRerunNIRCAM-LONG.py")

#: ``os.link`` and ``shutil.move`` are used legitimately elsewhere in this file
#: (the ``../*cal.fits`` staging link, the MAST download moves), so the guard is
#: scoped to the crf branch instead of grepping the whole module.
_ALIASING_CALLS = {
    ("os", "link"): "a hard link",
    ("os", "symlink"): "a symlink",
    ("os", "rename"): "a rename",
    ("os", "replace"): "a rename",
    ("shutil", "move"): "a move",
    ("pathlib", "rename"): "a rename",
}

_COPY_CALLS = {("shutil", "copy"), ("shutil", "copyfile"), ("shutil", "copy2")}


def _skip_outlier_crf_branch():
    """The ``elif skip_outlier_detection:`` arm that writes crf from members."""
    tree = ast.parse(SRC.read_text())
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        names = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
        if "_prod_crf" not in names:
            continue
        cur = node
        while cur.orelse and len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
            cur = cur.orelse[0]
            if (isinstance(cur.test, ast.Name)
                    and cur.test.id == "skip_outlier_detection"):
                return cur
    raise AssertionError(
        "could not find the `elif skip_outlier_detection:` crf branch")


def _qualified_calls(node):
    """``{(module, attr)}`` for every ``module.attr(...)`` call under ``node``."""
    found = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        fn = sub.func
        if isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name):
            found.add((fn.value.id, fn.attr))
        elif isinstance(fn, ast.Attribute):
            found.add((None, fn.attr))
    return found


def test_the_crf_branch_copies():
    """A copy is what makes the crf a snapshot of the member at write time."""
    calls = _qualified_calls(_skip_outlier_crf_branch())
    assert calls & _COPY_CALLS, (
        "the skip_outlier_detection crf branch no longer copies the member "
        "frame; the per-exposure crf must be an independent snapshot")


def test_the_crf_branch_does_not_alias_the_member_frame():
    """No link/rename: the member is mutated after the crf is written.

    The merged pass truncates ``_align.fits`` in place via ``shutil.copyfile``
    and ``fix_alignment`` edits headers via ``mode='update'``; either would
    reach through a hard link into an already-written crf.
    """
    calls = _qualified_calls(_skip_outlier_crf_branch())
    aliased = {call: why for call, why in _ALIASING_CALLS.items() if call in calls}
    assert not aliased, (
        f"the skip_outlier_detection crf branch uses {sorted(aliased.values())} "
        f"({sorted('.'.join(c) for c in aliased)}). The member frame is written "
        f"again after the crf exists -- shutil.copyfile truncates its inode in "
        f"place in the merged pass, and fits.open(mode='update') edits it in "
        f"place in fix_alignment -- so an aliased crf silently changes content "
        f"with no mtime of its own to show it.")


def test_the_reason_is_recorded_at_the_call_site():
    """Whoever tries to save the 4 s next needs the reason in front of them."""
    src = SRC.read_text()
    i = src.index("wrote {_n_crf} per-exposure crf as ")
    context = src[max(0, i - 3000):i]
    assert "NOT a hard link" in context, (
        "the crf copy lost the note explaining why it cannot be a link")
    assert "copyfile" in context and "mode='update'" in context, (
        "the note no longer names the two in-place writers that make an "
        "aliased crf unsafe")
