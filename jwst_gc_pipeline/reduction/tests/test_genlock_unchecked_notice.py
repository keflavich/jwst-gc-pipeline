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


def test_generation_columns_note_matches_reality():
    """The comment claimed the tie builders write base_calver.  Nothing in
    PRODUCTION does -- that claim is why the strong layer read as implemented.

    `git grep` only sees TRACKED files, so a writer added to an in-progress
    branch before `git add` was invisible, and the whole-file self-exclusion of
    `unified_alignment.py` meant a writer added to that file -- the most likely
    home for a stamper, right beside `_GENERATION_COLUMNS` -- was invisible
    too.  Walk the tree instead, and exclude only the comment LINES that
    legitimately name the column.
    """
    import pathlib
    repo = pathlib.Path(ua.__file__).resolve().parents[2]
    here = pathlib.Path(__file__).name
    # The package AND scripts/ -- a stamper could as easily be a script.
    # No `git grep`: it sees only TRACKED files, so this went green on any
    # branch where the writer had not been `git add`ed yet, and it made the
    # test red in a container with no git on PATH (subprocess.run raises
    # FileNotFoundError before any returncode is available).
    sources = [f for d in ("jwst_gc_pipeline", "scripts")
               for f in sorted((repo / d).rglob("*.py"))]
    writers = []
    for f in sources:
        if f.name == here or "tests" in f.parts:
            continue
        for lineno, line in enumerate(f.read_text().splitlines(), 1):
            if "base_calver" not in line:
                continue
            # The dormant-layer note and the mapping itself name the column on
            # purpose.  Everything else is a writer.
            if line.lstrip().startswith("#") or line.lstrip().startswith("#:"):
                continue
            if "('calver', 'base_calver')" in line or '"base_calver"' in line and "COLUMNS" in line:
                continue
            writers.append(f"{f.relative_to(repo)}:{lineno}: {line.strip()}")
    assert not writers, (
        "something now writes base_calver -- the dormant-layer note in "
        f"_GENERATION_COLUMNS needs updating: {writers}")


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
