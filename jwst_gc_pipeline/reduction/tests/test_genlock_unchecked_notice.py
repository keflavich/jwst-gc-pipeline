"""The genlock notice must state the true thing, once (issue #269).

The old mtime fallback compared the offsets table's mtime against the crf's and
warned when the table was older.  Re-reducing rewrites the crf, so the crf is
always newer than a table that was not rebuilt in the same pass -- which is the
normal state.  The warning therefore fired on every re-reduce by construction,
and a warning that cannot distinguish staleness from a re-run is noise.
"""
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table

from jwst_gc_pipeline.reduction import unified_alignment as ua


@pytest.fixture(autouse=True)
def _clear_notice_state():
    ua._GENLOCK_UNCHECKED_REPORTED.clear()
    yield
    ua._GENLOCK_UNCHECKED_REPORTED.clear()


def _frame(tmp_path, name="jw0_crf.fits"):
    path = tmp_path / name
    hdu0 = fits.PrimaryHDU()
    hdu0.header["CAL_VER"] = "1.17.1"
    hdu0.header["CRDS_CTX"] = "jwst_1253.pmap"
    hdu1 = fits.ImageHDU(np.zeros((4, 4)), name="SCI")
    hdu1.header["DVACORR"] = True
    fits.HDUList([hdu0, hdu1]).writeto(path)
    return str(path)


def _table_without_stamps():
    return Table({"Filter": ["F212N"], "Module": ["nrcb1"],
                  "dra": [0.0], "ddec": [0.0]})


def _table_with_stamps():
    t = _table_without_stamps()
    for col, _ in ua._GENERATION_COLUMNS:
        t[f"base_{col}"] = [""]
    return t


def test_no_stamps_reports_unchecked_not_staleness(tmp_path, capsys):
    fn = _frame(tmp_path)
    ua._check_generation(fn, _table_without_stamps(), str(tmp_path / "tbl.csv"))
    out = capsys.readouterr().out
    assert "WITHOUT a generation check" in out
    # it must NOT claim the table is stale or predates anything
    assert "predates" not in out
    assert "mtime" not in out


def test_notice_is_once_per_table_not_once_per_frame(tmp_path, capsys):
    tbl = _table_without_stamps()
    locked = str(tmp_path / "tbl.csv")
    for i in range(5):
        ua._check_generation(_frame(tmp_path, f"f{i}_crf.fits"), tbl, locked)
    assert capsys.readouterr().out.count("[genlock]") == 1


def test_a_second_table_is_reported_separately(tmp_path, capsys):
    tbl = _table_without_stamps()
    ua._check_generation(_frame(tmp_path, "a_crf.fits"), tbl, str(tmp_path / "a.csv"))
    ua._check_generation(_frame(tmp_path, "b_crf.fits"), tbl, str(tmp_path / "b.csv"))
    assert capsys.readouterr().out.count("[genlock]") == 2


def test_mtime_order_does_not_change_the_outcome(tmp_path, capsys):
    """The old check keyed on this and it is exactly what a re-reduce changes."""
    fn = _frame(tmp_path)
    locked = tmp_path / "tbl.csv"
    locked.write_text("x")
    os.utime(str(locked), (0, 0))                      # table far older than crf
    ua._check_generation(fn, _table_without_stamps(), str(locked))
    older = capsys.readouterr().out
    ua._GENLOCK_UNCHECKED_REPORTED.clear()
    os.utime(str(locked), None)                        # table newer than crf
    ua._check_generation(fn, _table_without_stamps(), str(locked))
    newer = capsys.readouterr().out
    assert older == newer and "[genlock]" in older


def test_strict_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("GENLOCK_STRICT", "1")
    with pytest.raises(RuntimeError, match="WITHOUT a generation check"):
        ua._check_generation(_frame(tmp_path), _table_without_stamps(),
                             str(tmp_path / "tbl.csv"))


def test_stamped_table_is_silent(tmp_path, capsys):
    ua._check_generation(_frame(tmp_path), _table_with_stamps(),
                         str(tmp_path / "tbl.csv"))
    assert "[genlock]" not in capsys.readouterr().out


def _base_column_writers(path):
    """Lines in ``path`` that ASSIGN into a ``base_*`` table column.

    AST, not text.  Two rounds of text matching got this wrong in opposite
    directions:

    * an ``A or (B and C)`` precedence slip let any double-quoted writer on a
      line also containing the substring ``COLUMNS`` -- in a trailing comment,
      even -- pass, so the guard went GREEN on a live writer;
    * exempting only ``#`` lines meant a DOCSTRING mentioning ``base_calver``
      reddened it, i.e. the guard forbade the repo from describing the column
      in prose;
    * and the most idiomatic stamper of all,
      ``tbl[f'base_{col}'] = gen[key]``, never spells the literal, so both
      ``git grep`` and the text walk missed the very shape the note calls the
      likely home for step 2 of #269.

    An assignment is a syntactic thing, so ask the parser.  Comments,
    docstrings and the mapping itself are invisible to it for free.
    """
    import ast
    try:
        tree = ast.parse(path.read_text(errors="replace"))
    except SyntaxError:
        return []
    hits = []
    # enclosing function per line, so a pardon can name ONE function instead of
    # a whole 2000-line module
    owner = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for ln in range(node.lineno, (node.end_lineno or node.lineno) + 1):
                owner.setdefault(ln, node.name)

    def _is_base_key(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value.startswith("base_")
        if isinstance(node, ast.JoinedStr):          # f"base_{col}"
            head = node.values[0] if node.values else None
            return (isinstance(head, ast.Constant)
                    and isinstance(head.value, str)
                    and head.value.startswith("base_"))
        return False

    for node in ast.walk(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for t in targets:
            if isinstance(t, ast.Subscript) and _is_base_key(t.slice):
                hits.append((node.lineno, owner.get(node.lineno, "<module>")))
    return hits


def test_generation_columns_note_matches_reality():
    """The comment claimed the tie builders write base_calver.  Nothing in
    PRODUCTION does -- that claim is why the strong layer read as implemented.

    `git grep` only sees TRACKED files, so a writer added to an in-progress
    branch before `git add` was invisible, and the whole-file self-exclusion of
    `unified_alignment.py` meant a writer added to that file -- the most likely
    home for a stamper, right beside `_GENERATION_COLUMNS` -- was invisible
    too.  It also made the test red in a container with no git on PATH, since
    `subprocess.run` raises FileNotFoundError before any returncode exists.
    Walk the tree and parse it instead.
    """
    import pathlib
    repo = pathlib.Path(ua.__file__).resolve().parents[2]
    here = pathlib.Path(__file__).name
    # The package AND scripts/ -- a stamper could as easily be a script.
    sources = [f for d in ("jwst_gc_pipeline", "scripts")
               for f in sorted((repo / d).rglob("*.py"))]
    # The ONE known writer, and finding it is why this became an AST walk.
    # `seed_offsets_table_from_consensus` does stamp `row[f"base_{k}"]` -- but
    # only when its `base_stamp_for` argument is not None, and NO caller in the
    # repo passes it.  So the strong generation check is dormant because it is
    # never ARMED, which is a different and more precise statement than
    # "nothing writes these columns", and neither `git grep` nor the text walk
    # could see it: the f-string never spells the literal.
    # (file, FUNCTION), not file.  A whole-file pardon is the exact shape this
    # test's own docstring gives as a reason the git-grep version failed --
    # excluding all of unified_alignment.py hid a writer added beside
    # _GENERATION_COLUMNS.  Excluding all of astrometry_checkpoint.py would be
    # worse, because the one known writer already lives there, which makes it
    # the most likely home for the next one: a plain
    # `tbl['base_calver'] = gen['cal_ver']` anywhere in those 2000 lines passed
    # green.
    KNOWN = {("jwst_gc_pipeline/photometry/astrometry_checkpoint.py",
              "seed_offsets_table_from_consensus")}
    writers = []
    for f in sources:
        if f.name == here or "tests" in f.parts:
            continue
        rel = f.relative_to(repo).as_posix()
        for lineno, func in _base_column_writers(f):
            if (rel, func) in KNOWN:
                continue
            writers.append(f"{rel}:{lineno} (in {func})")
    assert not writers, (
        "something now writes a base_* generation stamp -- the dormant-layer "
        f"note in _GENERATION_COLUMNS needs updating: {writers}")


def test_the_one_known_base_stamp_writer_is_still_unarmed():
    """The allowlist entry above is not a pardon.

    `seed_offsets_table_from_consensus` writes the stamps, so the moment any
    caller passes `base_stamp_for` the dormant `_assert_generation_row` layer
    goes LIVE on real tables -- which is a behaviour change nobody is
    currently expecting.  Pin the arming, not the writing.
    """
    import pathlib
    import re as _re
    repo = pathlib.Path(ua.__file__).resolve().parents[2]
    callers = []
    for d in ("jwst_gc_pipeline", "scripts"):
        for f in sorted((repo / d).rglob("*.py")):
            if "tests" in f.parts:
                continue
            for lineno, line in enumerate(
                    f.read_text(errors="replace").splitlines(), 1):
                if _re.search(r"base_stamp_for\s*=\s*(?!None)", line):
                    callers.append(f"{f.relative_to(repo)}:{lineno}: {line.strip()}")
    assert not callers, (
        "something now ARMS base_stamp_for, so _assert_generation_row is no "
        f"longer dormant -- the note and this PR's premise need updating: "
        f"{callers}")


def test_strict_refuses_every_time_not_once_per_process(tmp_path, monkeypatch):
    """Memoising before the raise made STRICT refuse at most once per table per
    process, so anything that caught and retried proceeded silently.  The memo
    is for the WARNING, not for the gate."""
    monkeypatch.setenv("GENLOCK_STRICT", "1")
    tbl = _table_without_stamps()
    locked = str(tmp_path / "tbl.csv")
    for i in range(3):
        with pytest.raises(RuntimeError, match="WITHOUT a generation check"):
            ua._check_generation(_frame(tmp_path, f"f{i}_crf.fits"), tbl, locked)


def test_strict_zero_does_not_enable_strict(tmp_path, monkeypatch, capsys):
    """`os.environ.get(name)` is truthy for the string "0"; the sibling gate
    compares against '1'."""
    for value in ("0", "false", ""):
        ua._GENLOCK_UNCHECKED_REPORTED.clear()
        monkeypatch.setenv("GENLOCK_STRICT", value)
        ua._check_generation(_frame(tmp_path, f"z{value or 'e'}_crf.fits"),
                             _table_without_stamps(), str(tmp_path / "t.csv"))
        assert "[genlock]" in capsys.readouterr().out


def test_the_docstring_does_not_describe_the_removed_check():
    doc = ua._check_generation.__doc__ or ""
    assert "mtime fallback used when" not in doc
    assert "nothing checks it" in doc or "says so" in doc


@pytest.mark.parametrize("value", ["2", "strict", "yes please", "TRUE!"])
def test_an_unparseable_gate_value_raises_rather_than_disabling(tmp_path, monkeypatch, value):
    """The bare-truthiness form this replaced would at least have ENABLED
    strict on `GENLOCK_STRICT=2`.  Reading it as 'off' turns a typo into a
    silently skipped check, which is what the gate is for."""
    monkeypatch.setenv("GENLOCK_STRICT", value)
    with pytest.raises(RuntimeError, match="not a recognised on/off value"):
        ua._check_generation(_frame(tmp_path), _table_without_stamps(),
                             str(tmp_path / "tbl.csv"))


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("0", False), ("false", False), ("no", False), ("off", False), ("", False),
    # WHITESPACE.  A trailing space out of a shell script or a SLURM --export
    # is ordinary, and now that an unrecognised value RAISES, dropping the
    # .strip() would turn `GENLOCK_STRICT="1 "` into a hard abort of every
    # field rather than a gate that is simply on.
    (" 1 ", True), ("\t0\n", False), ("\n1\n", True), (" ", False)])
def test_recognised_gate_values(value, expected, monkeypatch):
    monkeypatch.setenv("GENLOCK_STRICT", value)
    assert ua._strict_env("GENLOCK_STRICT") is expected


def test_both_genlock_gates_share_one_convention():
    """Two parsers in one file is how `GENLOCK_ALLOW_MISMATCH=true` silently
    fails to override."""
    import inspect
    src = inspect.getsource(ua)
    assert "os.environ.get('GENLOCK_ALLOW_MISMATCH') == '1'" not in src
    assert "_strict_env('GENLOCK_ALLOW_MISMATCH')" in src


def test_the_notice_names_the_TABLE_not_just_its_basename(tmp_path, monkeypatch):
    """brick and cloudc emit an identical line for two different tables when
    the notice prints only the basename, in a change whose whole subject is
    log legibility.  Reverting to `os.path.basename` survived the battery."""
    monkeypatch.delenv("GENLOCK_STRICT", raising=False)
    ua._GENLOCK_UNCHECKED_REPORTED.clear()
    sub = tmp_path / "brick" / "offsets"
    sub.mkdir(parents=True)
    locked = str(sub / "Offsets_JWST_Brick2221_VIRAC2locked.csv")
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ua._check_generation(_frame(tmp_path), _table_without_stamps(), locked)
    out = buf.getvalue()
    assert locked in out, out
    assert os.sep in out.split("offsets table ")[1].split(" carries")[0], out
